"""Match 3D regions to reference patches with spatial fused Gromov–Wasserstein transport."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from .dino_distill import _pca_fingerprint, _resolve_dino_paths
from .paths import sidecar_path
from .spatial_transport import sample_style_patches, solve_patch_transport


def _load_pca_state(gt_path, expected_model=None, expected_basis=None):
    """Read the scene PCA used to produce the 3D feature field."""
    payload = torch.load(gt_path, map_location="cpu", weights_only=True)
    cached_model = payload.get("dino_model")
    if expected_model and cached_model and cached_model != expected_model:
        raise ValueError(f"DINO model mismatch between the style and the scene PCA at {gt_path}")
    pca_state = payload.get("pca_state")
    if not isinstance(pca_state, dict) or "components" not in pca_state or "mean" not in pca_state:
        raise KeyError(
            f"[fgw] GT cache {gt_path} has no pca_state (components/mean). "
            f"Re-run scripts/extract_dino_features.py to regenerate.")
    if expected_basis and _pca_fingerprint(pca_state) != expected_basis:
        raise ValueError(f"Reused regions were constructed with a different PCA basis: {gt_path}")
    return pca_state


def _project_style_primitives(prototypes, pca_state, feature_dim):
    """③ 的 [K,1024] 风格基元 → 16 维, 与 ②④ 同一 PCA 空间. 复用 GT 提取的共享投影核心
    _project_with_components (scripts/extract_dino_features.py), 公式 (x-mean)@components.T
    与提取端保持一致; 投影后 L2 归一化, 与 F^X 在 ⑤ 中统一归一化以平衡特征/结构代价
    (见 MetricMeasureSpace 注释)."""
    from scripts.extract_dino_features import _project_with_components
    comps = pca_state["components"].float()                # [D, 1024]
    mean = pca_state["mean"].float()                       # [1024]
    if comps.shape[1] != prototypes.shape[1]:
        raise ValueError(f"[fgw] PCA components dim {comps.shape[1]} "
                          f"!= prototypes {prototypes.shape[1]}")
    if comps.shape[0] > feature_dim:
        raise ValueError(f"[fgw] PCA effective dim {comps.shape[0]} "
                          f"!= feature_dim {feature_dim}")
    proj = _project_with_components(prototypes.float(), comps, mean, feature_dim)
    return F.normalize(proj, dim=-1)


def _cosine_contrast(features):
    comparison = np.clip((1.0 - features @ features.T) * 0.5, 0.0, 1.0)
    np.fill_diagonal(comparison, 0.0)
    return comparison


def _patch_channels(target, patch_hw, node_zone=None):
    """Group sampled Voronoi cells by appearance bank and spatial quadrant."""
    width = patch_hw[1]
    patches = np.flatnonzero(target.patch_cells >= 0)
    ys, xs = np.divmod(patches, width)
    anchor_y, anchor_x = np.divmod(target.patch_indices, width)
    right = anchor_x > (xs.min() + xs.max()) / 2
    bottom = anchor_y > (ys.min() + ys.max()) / 2
    bank_quadrants = 4 * target.bank_ids + 2 * bottom + right
    if node_zone is not None:
        bank_quadrants = 2 * bank_quadrants + node_zone
    keys, node_channels = np.unique(bank_quadrants, return_inverse=True)
    channels = np.full(target.patch_cells.shape, -1, dtype=np.int64)
    channels[patches] = node_channels[target.patch_cells[patches]]
    membership = np.eye(len(keys), dtype=np.float64)[node_channels]
    return torch.from_numpy(channels), membership


def _conditional_texture_routes(plan, membership, points):
    cell_plan = plan @ membership
    row_mass = cell_plan.sum(axis=1)
    active = row_mass > 0
    conditional = cell_plan[active] / row_mass[active, None]
    return points[active], torch.from_numpy(conditional.astype(np.float32))


def _source_anchor(space, anchor_xyz, measure):
    active = np.flatnonzero(measure > 0)
    if not active.size:
        raise ValueError("The source anchor needs a representative with positive mass")
    points = space.points.detach().cpu().numpy().astype(np.float64)
    anchor_delta = points[active] - np.asarray(anchor_xyz, dtype=np.float64)
    return int(active[np.argmin(np.linalg.norm(anchor_delta, axis=1))])


def _geodesic_source_partition(space, anchor_xyz, measure):
    """Split positive source mass around the nearest representative to the anchor."""
    active = np.flatnonzero(measure > 0)
    if active.size < 2:
        raise ValueError("Source bias needs at least two representatives with positive mass")
    anchor_idx = _source_anchor(space, anchor_xyz, measure)
    distances = space.distances[anchor_idx].detach().cpu().numpy()
    ordered = active[np.lexsort((active, active != anchor_idx, distances[active]))]
    prefix_mass = np.cumsum(measure[ordered].astype(np.float64))[:-1]
    split = int(np.argmin(np.abs(prefix_mass - measure.sum(dtype=np.float64) * 0.5))) + 1
    near = np.zeros(len(measure), dtype=bool)
    near[ordered[:split]] = True
    return near, anchor_idx


def _geodesic_source_coordinate(space, anchor_idx):
    """Anchor proximity on the region graph; disconnected nodes have score zero."""
    distance = space.distances[anchor_idx].detach().cpu().numpy().astype(np.float64)
    if not np.isfinite(space.diameter) or space.diameter <= 0:
        raise ValueError("Centroid control needs a positive finite region diameter")
    return 1. - np.clip(distance / space.diameter, 0., 1.)


def _align_spatial_regions(metric_spaces, gaussians, primitives, pca_state, style, alpha):
    from .control import load_style_control, style_box_mask

    patch_hw = tuple(getattr(gaussians, "style_patch_hw", ()))
    assignments = getattr(gaussians, "style_patch_assignments", None)
    K = primitives.shape[0]
    if len(patch_hw) != 2 or min(patch_hw) < 1 or assignments is None:
        raise ValueError("Spatial FGW needs a reference patch grid and style assignments")
    patch_count = int(np.prod(patch_hw))
    assignments = assignments.detach().cpu().long().reshape(-1)
    if assignments.numel() != patch_count:
        raise ValueError("Style assignments do not match the reference patch grid")
    eligible = (assignments >= 0) & (assignments < K)
    valid_mask = getattr(gaussians, "style_patch_valid_mask", None)
    if valid_mask is not None:
        valid_mask = valid_mask.detach().cpu().bool().reshape(-1)
        if valid_mask.shape != eligible.shape:
            raise ValueError("Reference validity mask does not match the patch grid")
        eligible &= valid_mask
    if not eligible.any():
        raise ValueError("The reference contains no valid patches assigned to style banks")

    feature_dim = int(style.dino_feature_dim)
    raw_features = getattr(gaussians, "style_patch_features", None)
    if raw_features is not None:
        if raw_features.ndim != 2 or raw_features.shape[0] != patch_count:
            raise ValueError("Raw reference features do not match the patch grid")
        features = _project_style_primitives(raw_features.detach().cpu(), pca_state, feature_dim)
        feature_source = "raw_patch_tokens"
    else:
        # Legacy checkpoints only retain unit-norm prototypes. Restore a
        # typical raw-token scale before applying the scene's centered PCA.
        token_scale = pca_state["mean"].float().norm().clamp_min(1.0)
        projected = _project_style_primitives(primitives.cpu() * token_scale, pca_state, feature_dim)
        features = projected[assignments.clamp(0, K - 1)]
        feature_source = "legacy_rescaled_prototypes"
    features = features.numpy().astype(np.float64)
    if not np.isfinite(features[eligible.numpy()]).all():
        raise ValueError("Style patch projection contains non-finite features")

    region_count = len(metric_spaces)
    if sorted(ms.region_id for ms in metric_spaces) != list(range(region_count)):
        raise ValueError("Spatial FGW requires unique contiguous region IDs")
    control = load_style_control(
        getattr(style, "style_control_file", ""), region_count, patch_hw,
        default_strength=getattr(style, "style_control_strength", 1.0))
    max_nodes = int(getattr(style, "spatial_fgw_nodes", 48))
    prior_weight = float(getattr(style, "spatial_fgw_prior_weight", 0.08))
    entropy = float(getattr(style, "spatial_fgw_entropy", 0.01))
    max_iter = int(getattr(style, "spatial_fgw_max_iter", 120))
    contrast_weight = float(getattr(style, "spatial_fgw_contrast_weight", 0.0))
    routing = getattr(style, "spatial_fgw_routing", "region")
    if control.texture_mixes and routing == "region":
        raise ValueError("texture_mix requires --spatial_fgw_routing local")
    if control.texture_source_biases and routing != "local":
        raise ValueError("texture_mix.source_bias requires --spatial_fgw_routing local")

    print(f"[correspondence] ⑤ spatial FGW: {region_count} regions, {max_nodes} target nodes, "
          f"α={alpha:g}, prior={prior_weight:g}, entropy={entropy:g}, "
          f"contrast={contrast_weight:g}, routing={routing}, {feature_source}")
    all_weights = torch.zeros((region_count, K), dtype=torch.float32)
    diagnostics = [None] * region_count
    targets = {}
    memberships = {}
    target_contrasts = {}
    target_zones = {}
    texture_routes = {}
    with threadpool_limits(limits=1, user_api="blas"):
        for ms in tqdm(metric_spaces, desc="Spatial FGW align"):
            box = control.target_boxes.get(ms.region_id)
            mix_box, fraction = control.texture_mixes.get(ms.region_id, (None, None))
            group = box if mix_box is None else (box, mix_box)
            if group not in targets:
                allowed = eligible if box is None else eligible & style_box_mask(box, patch_hw).reshape(-1)
                if not allowed.any():
                    raise ValueError(f"Region {ms.region_id} reference box contains no valid patch centers")
                zone = None
                if mix_box is not None:
                    zone = style_box_mask(mix_box, patch_hw)
                    if not (zone & allowed).any() or not (~zone & allowed).any():
                        raise ValueError(f"Region {ms.region_id} texture_mix.box must split the valid reference patches")
                    zone = zone.numpy()
                target = sample_style_patches(
                    features, assignments.numpy(), patch_hw, allowed.numpy(), max_nodes, zone=zone)
                targets[group] = target
                node_zone = None if zone is None else zone[target.patch_indices]
                target_zones[group] = node_zone
                if contrast_weight > 0:
                    target_contrasts[group] = _cosine_contrast(target.features)
                if routing != "region":
                    channels, membership = _patch_channels(target, patch_hw, node_zone)
                    memberships[group] = membership
                    texture_routes[group] = {
                        "patch_channels": channels, "region_routes": {}, "reference_box": box,
                    }
            target = targets[group]
            node_zone = target_zones[group]

            source_features = F.normalize(ms.features.detach().cpu().float(), dim=1).numpy()
            if source_features.shape[1] != feature_dim:
                raise ValueError(f"Region {ms.region_id} features do not match PCA dimension {feature_dim}")
            # Unit-normalized feature vectors share a global cosine-distance scale.
            feature_cost = np.clip((1.0 - source_features @ target.features.T) * 0.5, 0.0, 1.0)
            geometry = ms.distances.detach().cpu().numpy().astype(np.float64) / ms.diameter
            # Older region caches encode disconnected pairs as twice the
            # maximum finite distance. Keep them distinct in the matching cost.
            geometry = np.minimum(geometry, 1.25)
            measure = ms.measure.detach().cpu().numpy()
            zone_options = {} if node_zone is None else {"target_zone": node_zone, "zone_fraction": fraction}
            source_split = None
            coordinate = None
            anchor_idx = None
            try:
                if ms.region_id in control.texture_source_biases:
                    anchor_xyz, bias = control.texture_source_biases[ms.region_id]
                    if control.texture_source_profiles[ms.region_id] == "centroid":
                        anchor_idx = _source_anchor(ms, anchor_xyz, measure)
                        coordinate = _geodesic_source_coordinate(ms, anchor_idx)
                        zone_options.update(source_coordinate=coordinate, source_bias=bias)
                    else:
                        source_split, anchor_idx = _geodesic_source_partition(ms, anchor_xyz, measure)
                        zone_options.update(source_near=source_split, source_bias=bias)
                plan, info = solve_patch_transport(
                    feature_cost, geometry, target.distances, measure, target.prior,
                    alpha, prior_weight, entropy, max_iter,
                    source_contrast=_cosine_contrast(source_features) if contrast_weight > 0 else None,
                    target_contrast=target_contrasts.get(group), contrast_weight=contrast_weight,
                    **zone_options)
            except ValueError as exc:
                raise ValueError(f"Spatial FGW failed for region {ms.region_id}: {exc}") from exc
            marginal = plan.sum(axis=0)
            weights = np.bincount(target.bank_ids, weights=marginal, minlength=K)
            all_weights[ms.region_id] = torch.from_numpy(weights.astype(np.float32))
            if routing != "region":
                texture_routes[group]["region_routes"][ms.region_id] = _conditional_texture_routes(
                    plan, memberships[group], ms.points)
            diagnostics[ms.region_id] = {
                "target_patch_indices": torch.from_numpy(target.patch_indices.copy()),
                "target_patch_xy": torch.from_numpy(target.positions.astype(np.float32)),
                "target_bank_ids": torch.from_numpy(target.bank_ids.copy()),
                "target_prior": torch.from_numpy(target.prior.astype(np.float32)),
                "transport_plan": torch.from_numpy(plan.astype(np.float32)),
                **info,
            }
            if node_zone is not None:
                diagnostics[ms.region_id]["target_zone"] = torch.from_numpy(node_zone.copy())
            if source_split is not None:
                diagnostics[ms.region_id]["source_near"] = torch.from_numpy(source_split.copy())
            if anchor_idx is not None:
                diagnostics[ms.region_id].update(
                    source_anchor_index=anchor_idx,
                    source_anchor_xyz=ms.points[anchor_idx].detach().cpu())
                diagnostics[ms.region_id]["source_profile"] = control.texture_source_profiles[ms.region_id]
                if coordinate is not None:
                    diagnostics[ms.region_id]["source_coordinate"] = torch.from_numpy(coordinate.copy())

    spatial_metadata = {
        "region_primitive_weights": all_weights,
        "region_style_target_boxes": control.target_boxes,
        "region_texture_mixes": control.texture_mixes,
        "region_texture_source_biases": control.texture_source_biases,
        "region_texture_source_profiles": control.texture_source_profiles,
        "region_style_strengths": control.strengths,
        "region_style_locked": control.locked,
        "spatial_fgw_diagnostics": diagnostics,
        "spatial_fgw_nodes": max_nodes,
        "spatial_fgw_prior_weight": prior_weight,
        "spatial_fgw_entropy": entropy,
        "spatial_fgw_max_iter": max_iter,
        "spatial_fgw_contrast_weight": contrast_weight,
        "spatial_fgw_routing": routing,
        "spatial_fgw_texture_routes": texture_routes or None,
        "spatial_fgw_feature_source": feature_source,
        "feature": alpha < 1,
        "structure": alpha > 0,
    }
    return all_weights.argmax(dim=1), spatial_metadata


def align_regions_with_primitives(dataset, gaussians, style):
    """Write regional style assignments for stage ⑥ and correspondence inspection."""
    primitives = getattr(gaussians, "style_primitives", None)
    metric_spaces = getattr(gaussians, "metric_spaces", None)
    gaussians.region_primitive_weights = None
    gaussians.region_texture_mixes = {}
    gaussians.region_style_target_boxes = {}
    gaussians.region_style_strengths = None
    gaussians.region_style_locked = None
    gaussians.spatial_fgw_texture_routes = None
    if primitives is None or primitives.numel() == 0:
        print("[fgw] ⑤ skipped: no style_primitives (③ did not run / no --style_image).")
        gaussians.region_primitive_map = None
        gaussians.primitive_labels = None
        return
    if not metric_spaces:
        print("[fgw] ⑤ skipped: no metric_spaces (④ did not run).")
        gaussians.region_primitive_map = None
        gaussians.primitive_labels = None
        return

    effective_alpha = float(style.fgw_alpha)
    gt_path, _ = _resolve_dino_paths(dataset, style)
    expected_model = getattr(style, "dino_model", None)
    expected_basis = getattr(gaussians, "region_pca_fingerprint", None)
    pca_state = _load_pca_state(gt_path, expected_model, expected_basis)
    K = primitives.shape[0]
    region_primitive_map, spatial_metadata = _align_spatial_regions(
        metric_spaces, gaussians, primitives, pca_state, style, effective_alpha)

    # 逐高斯基元标签: cluster_labels[N] (-1=被滤掉) → region_primitive_map[region]
    cluster_labels = getattr(gaussians, "cluster_labels", None)
    if cluster_labels is not None:
        primitive_labels = torch.full_like(cluster_labels, -1)
        valid = cluster_labels >= 0
        primitive_labels[valid] = region_primitive_map[cluster_labels[valid].clamp(min=0)]
    else:
        primitive_labels = None

    n_matched = int((region_primitive_map >= 0).sum())
    usage = torch.bincount(region_primitive_map[region_primitive_map >= 0], minlength=K).tolist()
    print(f"[correspondence] ⑤ matched {n_matched}/{len(metric_spaces)} regions; "
          f"primitive usage (per-region count) {usage}", flush=True)

    sidecar = sidecar_path(
        dataset.model_path, style.style_iterations, "region_primitives.pt")
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    torch.save({
        "version": 2,
        "region_primitive_map": region_primitive_map,      # [R] region_id → primitive_id
        "primitive_labels": primitive_labels,              # [N_full] per-gaussian (-1=none)
        "num_primitives": K,
        "num_regions": len(metric_spaces),
        "fgw_alpha": effective_alpha,
        "configured_fgw_alpha": float(style.fgw_alpha),
        "effective_alpha": effective_alpha,
        "correspondence_method": "spatial_fgw",
        "feature_weight": 1.0 - effective_alpha,
        "structure_weight": effective_alpha,
        "pca_state_path": gt_path,
        "feature_dim": int(style.dino_feature_dim),
        **spatial_metadata,
    }, sidecar)
    gaussians.region_primitive_map = region_primitive_map
    gaussians.primitive_labels = primitive_labels
    for attribute in ("region_primitive_weights", "region_style_target_boxes", "region_texture_mixes",
                      "region_style_strengths", "region_style_locked", "spatial_fgw_texture_routes"):
        if attribute in spatial_metadata:
            setattr(gaussians, attribute, spatial_metadata[attribute])
    print(f"[correspondence] ⑤ done → {sidecar}", flush=True)
