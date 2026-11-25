"""Optimize Gaussian SH with visibility-weighted VGG texture banks.

Stage V gives distributions over texture banks or reference patch cells. Render the
weights directly and use private reference crops for regions with an explicit target.
Fit full-strength appearance, then compose the requested per-region SH strengths.
The nearest-neighbor feature matching follows StylizedGS.
"""

import json
import math
import os
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import network_gui, render, render_features
from .control import StyleControl, load_style_control, style_box_mask
from .geometry import _compute_normals, compute_knn_graph
from .paths import sidecar_path, style_output_dir
from .tone import calibrate_style_tone


_STYLE_VGG_BLOCKS = [2, 3]   # NNFM 取 VGG block (StylizedGS 默认 [2,3]); block→layer 见下表
# VGG16.features 层索引 (自 StylizedGS nnfm_loss.py): block i 取第 i+1 个卷积块的所有 Conv2d.
_VGG_BLOCK_INDEXES = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]
_MIN_STYLE_COVERAGE = 1e-6




class _VGGFeatureExtractor(torch.nn.Module):
    """VGG16 卷积特征提取器 (NNFM 用). 输入须已 ImageNet 归一化 (与 ③ DINO 预处理一致,
    故 ⑥ 复用 ③ 的风格图张量可直接喂 VGG, patch 网格与 VGG 特征精确对齐)."""
    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)

    def get_feats(self, x, layers):
        final_ix = max(layers)
        outputs = []
        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs


def _vgg_imagenet_normalize(img_01, device):
    """img_01 in [0,1], shape [1,3,H,W] → ImageNet 归一化 (与 DINO 预处理 mean/std 一致)."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (img_01 - mean) / std


def _style_block_features(extractor, image, blocks):
    layers = [layer for block in blocks for layer in _VGG_BLOCK_INDEXES[block]]
    features_by_layer = dict(zip(layers, extractor.get_feats(image, layers)))
    return [torch.cat([features_by_layer[layer] for layer in _VGG_BLOCK_INDEXES[block]], 1)[0]
            for block in blocks]


def _build_style_banks(extractor, style_tensor, patch_assignments, patch_hw, patch_size,
                       blocks, bank_routes, target_boxes, valid_patches, texture_routes=None):
    """Build shared banks and private banks with VGG evaluated inside each selected crop."""
    if style_tensor.dim() == 3:
        style_tensor = style_tensor.unsqueeze(0)
    label_grid = patch_assignments.reshape(patch_hw).cpu()
    valid_grid = valid_patches.reshape(patch_hw).cpu()
    groups = {}
    for channel, (group, label) in enumerate(bank_routes):
        groups.setdefault(group, []).append((channel, label))
    banks = [{} for _ in blocks]

    with torch.no_grad():
        for group, routes in groups.items():
            selected = valid_grid
            image = style_tensor
            if texture_routes is None:
                labels = label_grid
                box = target_boxes[group] if group is not None else None
            else:
                target = texture_routes[group]
                labels = target["patch_channels"].reshape(patch_hw).cpu()
                selected = selected & (labels >= 0)
                box = target.get("reference_box", group)
            if box is not None:
                selected = selected & style_box_mask(box, patch_hw).reshape(patch_hw)
                rows, cols = selected.nonzero(as_tuple=True)
                if rows.numel() == 0:
                    raise ValueError(f"reference box {box} contains no valid patches")
                top, bottom = int(rows.min()), int(rows.max()) + 1
                left, right = int(cols.min()), int(cols.max()) + 1
                image = image[..., top * patch_size:bottom * patch_size,
                             left * patch_size:right * patch_size]
                selected = selected[top:bottom, left:right]
                labels = labels[top:bottom, left:right]
            labels = torch.where(selected, labels, -1).float().to(image.device)[None, None]

            for bank, feature in zip(banks, _style_block_features(extractor, image, blocks)):
                resized = F.interpolate(labels, size=feature.shape[-2:], mode="nearest").reshape(-1).long()
                features = feature.flatten(1)
                for channel, label in routes:
                    bank[channel] = features[:, resized == label].contiguous()
    return banks




def _routed_style_loss_per_view(extractor, rendered, rendered_weights, banks, blocks, hard):
    """Weight per-bank NNFM by posterior probability and the edited surface's visible coverage."""
    image = _vgg_imagenet_normalize(rendered.unsqueeze(0), rendered.device)
    total_loss = rendered.new_zeros(())
    total_coverage = rendered.new_zeros(())
    for feat, bank in zip(_style_block_features(extractor, image, blocks), banks):
        channels = [channel for channel, texture in bank.items() if texture.shape[1] > 0]
        if not channels:
            continue
        mass = F.interpolate(rendered_weights[channels][None], size=feat.shape[-2:],
                             mode="area")[0].flatten(1).clamp_min(0)
        coverage = mass.sum(dim=0)
        valid = coverage > _MIN_STYLE_COVERAGE
        if hard:
            # Class channels vote by visibility. Numeric label magnitudes never mix.
            assignments = mass.argmax(dim=0)
            mass = torch.zeros_like(mass).scatter_(0, assignments[None], coverage[None])

        direction = F.normalize(feat.flatten(1), dim=0, eps=1e-8)
        with torch.no_grad():
            weighted_targets = torch.zeros_like(direction)
            for row, channel in enumerate(channels):
                weights = mass[row]
                indices = (valid & (weights > 0)).nonzero().flatten()
                if indices.numel() == 0:
                    continue
                reference = bank[channel]
                # Cosine nearest-neighbor search on unit columns. Keep each match
                # matrix small even when many pixel routes share one large bank.
                chunk = max(1, (1 << 24) // reference.shape[1])
                for subset in indices.split(chunk):
                    best = (reference.T @ direction[:, subset]).argmax(dim=0)
                    weighted_targets[:, subset] += reference[:, best] * weights[subset]

        block_coverage = coverage[valid].sum()
        # Sum of expected cosine losses, with a single source gradient per pixel.
        total_loss = total_loss + block_coverage - (direction * weighted_targets).sum()
        total_coverage = total_coverage + block_coverage

    # The render mass is the per-pixel posterior times visible edited coverage.
    # Zero coverage represents a pixel without a style target.
    return total_loss / total_coverage.clamp_min(_MIN_STYLE_COVERAGE)


def _precompute_style_graph(gaussians, k, backend, device):
    """⑥ 一致性正则用的 KNN 图 + 法线点积 (几何冻结 → 一次性预计算). 复用 ④ 的
    compute_knn_graph / _compute_normals. 返回 (knn_idx[N,k], normal_dot[N,k])."""
    xyz = gaussians.get_xyz.detach()
    knn_idx, _ = compute_knn_graph(xyz, k, backend=backend)
    scaling = gaussians.get_scaling.detach()
    rotation = gaussians.get_rotation.detach()
    normals = _compute_normals(scaling, rotation)                   # [N,3]
    nbr_n = normals[knn_idx]                                         # [N,k,3]
    normal_dot = torch.einsum("nkd,nd->nk", nbr_n, normals).abs()   # [N,k]
    return knn_idx, normal_dot


def _normal_tv_loss(features_dc, knn_idx, normal_dot, tau_normal):
    """L_normal (法线一致邻居 SH DC 一致) + L_tv (空间邻居 SH DC 全变分).
    "风格特征" 取 SH DC (视角无关基础色): 正则化它直接抑制视角切换闪烁, 且 [N,3] 内存可控.
    features_dc [N,1,3]; L_tv 覆盖所有 KNN 邻居, L_normal 是其中 |n_i·n_j|>τ 的子集."""
    c = features_dc.reshape(-1, 3)                                  # [N,3]
    nbr_c = c[knn_idx]                                              # [N,k,3]
    diff = (c.unsqueeze(1) - nbr_c).abs()                          # [N,k,3]
    l_tv = diff.mean()
    normal_mask = (normal_dot > tau_normal)                         # [N,k]
    n_normal = normal_mask.sum()
    if n_normal.item() > 0:
        l_normal = (diff * normal_mask.unsqueeze(-1).float()).sum() / \
                   (n_normal.float() * 3.0 + 1e-8)
    else:
        l_normal = torch.zeros((), device=c.device)
    return l_normal, l_tv


@torch.no_grad()
def _prepare_patch_routing(xyz, point_regions, region_map, control, texture_routes):
    """Lift representative patch probabilities to Gaussians in the same region."""
    device = xyz.device
    candidates = ~control.locked & (region_map.cpu() >= 0)
    trained_regions = torch.zeros_like(candidates)
    bank_routes = []
    region_routes = {}
    for group, target in texture_routes.items():
        box = target.get("reference_box", group)
        region_plans = target["region_routes"]
        channel_count = int(target["patch_channels"].max()) + 1
        if channel_count <= 0:
            continue
        start = len(bank_routes)
        for region_id, (points, probabilities) in region_plans.items():
            if not 0 <= region_id < len(candidates):
                raise ValueError(f"invalid patch routing region {region_id}")
            if not candidates[region_id]:
                continue
            if box != control.target_boxes.get(region_id) or region_id in region_routes:
                raise ValueError(f"invalid reference box for patch routing region {region_id}")
            if (points.ndim != 2 or points.shape[1] != 3 or len(points) == 0
                    or probabilities.ndim != 2 or probabilities.shape[1] != channel_count
                    or len(probabilities) not in (1, len(points))):
                raise ValueError(f"invalid patch routing dimensions for region {region_id}")
            if (not torch.isfinite(points).all() or not torch.isfinite(probabilities).all()
                    or (probabilities < 0).any()):
                raise ValueError(f"invalid patch routing data for region {region_id}")
            mass = probabilities.sum(dim=1, keepdim=True)
            valid = mass[:, 0] > 0
            if not valid.any():
                continue
            rows = probabilities[valid] / mass[valid]
            reps = points[valid.to(points.device)] if len(probabilities) > 1 else points
            region_routes[region_id] = (slice(start, start + channel_count), reps, rows)
            trained_regions[region_id] = True
        if any(trained_regions[region_id] for region_id in region_plans):
            bank_routes.extend((group, channel) for channel in range(channel_count))

    counts = torch.bincount(point_regions + 1, minlength=len(candidates) + 1)
    offsets = [0, *counts.cumsum(dim=0).tolist()]
    indices_by_region = point_regions.argsort(stable=True)
    render_routes = torch.zeros((len(xyz), len(bank_routes)), dtype=torch.float32, device=device)
    for region_id, (channels, reps, weights) in region_routes.items():
        indices = indices_by_region[offsets[region_id + 1]:offsets[region_id + 2]].to(device)
        weights = weights.to(device=device, dtype=render_routes.dtype)
        if len(weights) == 1:
            render_routes[indices, channels] = weights
            continue
        reps = reps.to(device=device, dtype=xyz.dtype)
        tiny = torch.finfo(xyz.dtype).tiny
        for chunk in indices.split(8192):
            distances = torch.cdist(xyz[chunk], reps, compute_mode="donot_use_mm_for_euclid_dist")
            nearest_distances, nearest_indices = distances.topk(min(3, len(reps)), largest=False)
            closest = nearest_distances[:, :1]
            # Coincident representatives take all mass. For other points, scale
            # inverse distances by the nearest distance before normalization.
            blend = torch.where(closest == 0, nearest_distances == 0,
                                closest / nearest_distances.clamp_min(tiny))
            blend = blend.to(weights.dtype)
            blend = blend / blend.sum(dim=1, keepdim=True)
            render_routes[chunk, channels] = (weights[nearest_indices] * blend[..., None]).sum(dim=1)
    return render_routes, bank_routes, trained_regions


def _prepare_style_routing(gaussians, primitives, control, texture_routes=None):
    """Map region probabilities to shared texture banks or banks private to a crop."""
    region_map = gaussians.region_primitive_map
    point_regions = gaussians.cluster_labels.cpu().long()
    num_regions = region_map.numel()
    num_primitives = int(primitives.shape[0])
    if point_regions.shape != (gaussians.get_xyz.shape[0],):
        raise ValueError("cluster_labels does not match the Gaussian point count")
    if point_regions.numel() and (point_regions.min() < -1 or point_regions.max() >= num_regions):
        raise ValueError("cluster_labels contains an invalid region ID")

    # The final lookup row is reserved for invalid Gaussians (-1).
    point_rows = torch.where(point_regions >= 0, point_regions, num_regions)
    strengths = torch.cat((control.strengths, control.strengths.new_zeros(1)))[point_rows]
    if texture_routes is not None:
        render_routes, bank_routes, trained_regions = _prepare_patch_routing(
            gaussians.get_xyz, point_regions, region_map, control, texture_routes)
        editable = torch.cat((trained_regions, trained_regions.new_zeros(1)))[point_rows]
        return render_routes, bank_routes, trained_regions, editable, strengths

    weights = getattr(gaussians, "region_primitive_weights", None)
    if weights is None:
        weights = F.one_hot(region_map.cpu().clamp_min(0), num_primitives).float()
        weights[region_map.cpu() < 0] = 0
    else:
        weights = weights.float().cpu().clone()
        if weights.shape != (num_regions, num_primitives):
            raise ValueError("region texture weights do not match the scene regions and style banks")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("region texture weights must be finite and nonnegative")
    weight_sum = weights.sum(dim=1, keepdim=True)
    trained_regions = ~control.locked & (weight_sum[:, 0] > 0)
    weights = weights / weight_sum.clamp_min(_MIN_STYLE_COVERAGE)
    weights[~trained_regions] = 0

    shared_weights = weights.clone()
    anchored_regions = sorted(control.target_boxes)
    shared_weights[anchored_regions] = 0
    used_shared = shared_weights.any(dim=0).nonzero().flatten().tolist()
    shared_routes = [(None, primitive) for primitive in used_shared]
    private_routes = [(region_id, primitive) for region_id in anchored_regions
                      for primitive in weights[region_id].nonzero().flatten().tolist()]
    bank_routes = shared_routes + private_routes
    region_routes = weights.new_zeros(num_regions + 1, len(bank_routes))
    region_routes[:num_regions, :len(shared_routes)] = shared_weights[:, used_shared]
    for channel, (region_id, primitive) in enumerate(private_routes, start=len(shared_routes)):
        region_routes[region_id, channel] = weights[region_id, primitive]

    device = gaussians.get_xyz.device
    render_routes = region_routes.to(device)[point_rows.to(device)].contiguous()
    editable = torch.cat((trained_regions, trained_regions.new_zeros(1)))[point_rows]
    return render_routes, bank_routes, trained_regions, editable, strengths


def _output_controlled_appearance(gaussians, dataset, style, control, trained_regions,
                                  editable, strengths, base_blend, training_stats):
    styled_dir = style_output_dir(dataset.model_path, style.style_iterations)
    os.makedirs(styled_dir, exist_ok=True)
    point_indices = None if editable.all() else editable.nonzero().flatten()
    payload = {
        "version": 1,
        "num_points": len(editable),
        "source_ply_path": getattr(gaussians, "style_source_ply_path", None),
        "xyz_fingerprint": gaussians._xyz_fingerprint(),
        "point_indices": point_indices,
        "strengths": control.strengths,
        "trained_regions": trained_regions,
    }
    for key, tensor in (("styled_dc", gaussians._features_dc),
                        ("styled_rest", gaussians._features_rest)):
        full_appearance = tensor.detach()
        if point_indices is not None:
            full_appearance = full_appearance[point_indices.to(tensor.device)]
        payload[key] = full_appearance.cpu()
    path = sidecar_path(dataset.model_path, style.style_iterations, "style_sh.pt")
    torch.save(payload, path)

    blend_indices, orig_dc, orig_rest = base_blend
    if blend_indices.numel() > 0:
        with torch.no_grad():
            for tensor, original in ((gaussians._features_dc, orig_dc),
                                     (gaussians._features_rest, orig_rest)):
                index = blend_indices.to(tensor.device)
                amount = strengths[blend_indices].to(tensor.device)[:, None, None]
                full_strength = tensor[index]
                tensor[index] = torch.lerp(original.to(tensor.device), full_strength, amount)

    styled_ply = os.path.join(styled_dir, "point_cloud.ply")
    gaussians.save_ply(styled_ply)
    with open(os.path.join(styled_dir, "style_training.json"), "w", encoding="utf-8") as handle:
        json.dump(training_stats, handle, indent=2)
    print(f"[style] ⑥ done: styled ply -> {styled_ply}")


def generate_style(dataset, gaussians, scene, pipe, background, style):
    """Train appearance against the Stage V texture routing and save its controlled SH."""
    if style.style_iterations <= 0:
        print("[style] ⑥ skipped: style_iterations <= 0")
        return
    primitives = getattr(gaussians, "style_primitives", None)
    primitive_labels = getattr(gaussians, "primitive_labels", None)
    region_map = getattr(gaussians, "region_primitive_map", None)
    if primitives is None or primitives.numel() == 0:
        print("[style] ⑥ skipped: no style_primitives (③ did not run / no --style_image)")
        return
    if primitive_labels is None or region_map is None:
        print("[style] ⑥ skipped: no region assignment (⑤ did not run / failed)")
        return
    if style.style_lambda_photo > 0:
        print("[style] ⑥ λ_photo>0 但 L_photo 未实现 (相机原图已释放); 忽略 L_photo")

    device = gaussians.get_xyz.device
    K = int(primitives.shape[0])
    blocks = list(_STYLE_VGG_BLOCKS)
    n_points = gaussians.get_xyz.shape[0]
    texture_routes = getattr(gaussians, "spatial_fgw_texture_routes", None)
    soft = texture_routes is not None or getattr(gaussians, "region_primitive_weights", None) is not None
    control_file = getattr(style, "style_control_file", "")
    global_strength = float(getattr(style, "style_control_strength", 1.0))

    # Preserve the exact patch layout used by Stage III to construct the texture banks.
    patch_assignments = getattr(gaussians, "style_patch_assignments", None)
    patch_hw = getattr(gaussians, "style_patch_hw", None)
    patch_size = getattr(gaussians, "style_patch_size", None)
    preprocess_size = getattr(gaussians, "style_preprocess_max_size", style.style_primitive_size)
    valid_patches = getattr(gaussians, "style_patch_valid_mask", None)
    if patch_assignments is None or patch_hw is None:
        path = sidecar_path(dataset.model_path, style.style_iterations, "style_primitives.pt")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        patch_assignments = payload["patch_assignments"]
        patch_hw = payload["patch_hw"]
        patch_size = payload.get("patch_size", 16)
        preprocess_size = payload.get("preprocess_max_size", preprocess_size)
        valid_patches = payload.get("patch_valid_mask")
    if valid_patches is None:
        valid_patches = torch.ones_like(patch_assignments, dtype=torch.bool)

    if soft:
        control = StyleControl(gaussians.region_style_strengths, gaussians.region_style_locked,
                               gaussians.region_style_target_boxes, getattr(gaussians, "region_texture_mixes", {}))
    else:
        control = load_style_control(control_file, region_map.numel(), patch_hw,
                                     default_strength=global_strength)
    if control.target_boxes and not soft:
        raise ValueError("reference patch selection requires spatial FGW texture matching")

    render_routes, bank_routes, trained_regions, editable, strengths = _prepare_style_routing(
        gaussians, primitives, control, texture_routes)
    blend_indices = (editable & (strengths != 1)).nonzero().flatten()
    blend_index_device = blend_indices.to(device)
    base_blend = (blend_indices, gaussians._features_dc.detach()[blend_index_device].cpu(),
                  gaussians._features_rest.detach()[blend_index_device].cpu())
    mode = "soft" if soft else "hard"
    training_stats = {
        "mode": mode,
        "num_gaussians": n_points,
        "editable_gaussians": int(editable.sum()),
        "trained_regions": int(trained_regions.sum()),
        "bank_count": len(bank_routes),
        "target_boxes": control.target_boxes,
        "texture_mixes": control.texture_mixes,
        "loss_history": [],
    }
    if texture_routes is not None:
        training_stats["spatial_fgw_routing"] = style.spatial_fgw_routing
    if not editable.any():
        _output_controlled_appearance(gaussians, dataset, style, control, trained_regions,
                                      editable, strengths, base_blend, training_stats)
        return
    locked_indices = (~editable).nonzero().flatten().to(device)

    # A fresh Adam state and zeroed gradients keep locked Gaussian coefficients exact.
    for p in (gaussians._xyz, gaussians._scaling, gaussians._rotation,
              gaussians._opacity, gaussians._dino_features):
        if p is not None:
            p.requires_grad_(False)
    for p in (gaussians._features_dc, gaussians._features_rest):
        p.requires_grad_(True)
    style_optimizer = torch.optim.Adam([
        {"params": [gaussians._features_dc], "lr": style.style_lr},
        {"params": [gaussians._features_rest], "lr": style.style_lr / 20.0},
    ], lr=0.0, eps=1e-15)

    from scripts.extract_dino_features import _load_and_preprocess_image
    print(f"[style] ⑥ start: {training_stats['editable_gaussians']:,}/{n_points:,} editable Gaussians, "
          f"{len(bank_routes)} texture banks, {mode} routing, "
          f"{style.style_iterations} iters, blocks={blocks}, λ(style/normal/tv)="
          f"{style.style_lambda}/{style.style_lambda_normal}/{style.style_lambda_tv}")
    extractor = _VGGFeatureExtractor(device)
    style_tensor, _, loaded_patch_hw = _load_and_preprocess_image(
        Path(style.style_image), patch_size=patch_size, device=device,
        max_size=preprocess_size)
    if tuple(loaded_patch_hw) != tuple(patch_hw):
        raise ValueError("reference image patch layout does not match Stage III texture banks")
    banks = _build_style_banks(extractor, style_tensor, patch_assignments, patch_hw, patch_size,
                               blocks, bank_routes, control.target_boxes, valid_patches, texture_routes)
    for bank in banks:
        if not any(features.shape[1] for features in bank.values()):
            raise ValueError("a style loss layer has no valid reference texture features")
    banks = [{channel: F.normalize(features, dim=0, eps=1e-8) for channel, features in bank.items()}
             for bank in banks]
    del style_tensor, patch_assignments

    knn_idx, normal_dot = _precompute_style_graph(
        gaussians, style.style_tv_knn_k, style.style_knn_backend, device)

    cameras = scene.getTrainCameras().copy()
    viewpoint_stack = []
    rs = style.style_render_scale
    rasterize_mode = "antialiased" if pipe.antialiasing else "classic"
    progress = tqdm(total=style.style_iterations, desc="Style ⑥")
    ema_loss = ema_style = ema_normal = ema_tv = 0.0
    for it in range(1, style.style_iterations + 1):
        network_gui.update(it, num_train_rays_per_step=1)
        if not network_gui.disable_pause:
            network_gui.wait_if_paused(it)

        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        with network_gui.lock:
            render_pkg = render(cam, gaussians, pipe, background,
                                use_trained_exp=False, no_densify_hook=True)
            with torch.no_grad():
                route_map = render_features(cam, gaussians, render_routes, rasterize_mode=rasterize_mode)
        rgb = render_pkg["render"]

        if rs != 1.0:
            rgb = F.interpolate(rgb[None], scale_factor=rs, mode="bilinear", align_corners=False)[0]
        L_style = _routed_style_loss_per_view(extractor, rgb, route_map, banks, blocks, hard=not soft)
        L_normal, L_tv = _normal_tv_loss(gaussians._features_dc, knn_idx,
                                        normal_dot, style.style_normal_tau)
        loss = style.style_lambda * (L_style if L_style is not None else 0.0)
        loss = loss + style.style_lambda_normal * L_normal + style.style_lambda_tv * L_tv
        loss_value = float(loss.detach())
        if not math.isfinite(loss_value):
            raise RuntimeError(f"nonfinite stylization loss at iteration {it}")

        style_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if locked_indices.numel():
            for tensor in (gaussians._features_dc, gaussians._features_rest):
                if tensor.grad is not None:
                    tensor.grad.index_fill_(0, locked_indices, 0.0)
        style_optimizer.step()

        with torch.no_grad():
            style_value = float(L_style) if L_style is not None else 0.0
            normal_value = float(L_normal)
            tv_value = float(L_tv)
            ema_loss = 0.4 * loss_value + 0.6 * ema_loss
            ema_style = 0.4 * style_value + 0.6 * ema_style
            ema_normal = 0.4 * normal_value + 0.6 * ema_normal
            ema_tv = 0.4 * tv_value + 0.6 * ema_tv
            if it == 1 or it % 10 == 0 or it == style.style_iterations:
                stats = {"iteration": it, "loss": loss_value, "style": style_value,
                         "normal": normal_value, "tv": tv_value}
                training_stats["loss_history"].append(stats)
                progress.set_postfix(L=f"{ema_loss:.4f}", style=f"{ema_style:.4f}",
                                     normal=f"{ema_normal:.4f}", tv=f"{ema_tv:.4f}")
                progress.update(it - progress.n)
    progress.close()

    del extractor, banks, knn_idx, normal_dot, render_routes
    style_optimizer.state.clear()
    if style.style_tone_strength > 0:
        reference_boxes = {control.target_boxes.get(region)
                           for region in trained_regions.nonzero().flatten().tolist()}
        if len(reference_boxes) == 1:
            reference_box = next(iter(reference_boxes))
            with network_gui.lock:
                training_stats["tone"] = calibrate_style_tone(
                    gaussians, cameras, pipe, background, editable, Path(style.style_image),
                    style.style_tone_strength, reference_box=reference_box)
        else:
            # One shared RGB offset cannot honor regions with different color targets.
            training_stats["tone"] = {"status": "mixed_reference_boxes"}
    _output_controlled_appearance(gaussians, dataset, style, control, trained_regions,
                                  editable, strengths, base_blend, training_stats)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
