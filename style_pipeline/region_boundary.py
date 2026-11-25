"""Refine selected boundaries of a saved 3D Gaussian partition on CPU."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.special import expit
import torch


_BATCH_SIZE = 16_384


def _xyz(vertices):
    return np.column_stack([vertices[axis] for axis in "xyz"]).astype(np.float32, copy=False)


def _fingerprint(xyz):
    return hashlib.md5(memoryview(xyz).cast("B")).hexdigest()


def _normals(log_scale, q):
    length = np.linalg.norm(q, axis=1)
    q = q / np.maximum(length[:, None], 1e-8)
    shortest_axis = np.eye(3, dtype=np.float32)[log_scale.argmin(axis=1)]
    cross = np.cross(q[:, 1:], shortest_axis)
    rotated = shortest_axis + 2 * (q[:, :1] * cross + np.cross(q[:, 1:], cross))
    ordered_scale = np.sort(log_scale, axis=1)
    reliable = (ordered_scale[:, 1] - ordered_scale[:, 0] >= np.log(1.5)) & (length > 1e-8)
    return rotated, reliable


def region_boundary_selection(style, *, reference):
    selection = str(getattr(style, "region_boundary", "off")).strip().lower()
    if selection == "auto":
        return "all" if not reference else None
    if selection in ("", "off"):
        return None
    if selection == "all":
        return selection
    ids = [value.strip() for value in selection.split(",")]
    if any(not value.isascii() or not value.isdecimal() for value in ids):
        raise ValueError("--region_boundary expects auto, off, all, or comma-separated region IDs")
    return sorted(set(map(int, ids)))


def load_boundary_features(gaussians, style):
    path = getattr(style, "region_dino_cache", "")
    live = gaussians.get_dino_features.detach()
    if not path and live.numel():
        return live.cpu(), "in_memory"
    if not path:
        from .refine import _resolve_refine_dir
        path = Path(_resolve_refine_dir(gaussians.style_source_ply_path)) / "dino_features_gaussian.pt"
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Boundary refinement needs cached per-Gaussian DINO features: {path}")
    cache = torch.load(path, map_location="cpu", weights_only=True)
    features = cache["dino_features"]
    if (cache.get("xyz_fingerprint") != gaussians._xyz_fingerprint()
            or features.shape != (len(gaussians.get_xyz), style.dino_feature_dim)):
        raise ValueError(f"Boundary DINO cache differs in point order or feature dimension: {path}")
    return features, str(path)


def refine_live_partition(gaussians, style, labels, valid_indices, selection, *, reference=None):
    """Refine the partition consumed by the current train or preview invocation."""
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().view(-1).cpu().numpy()
    features, dino_source = load_boundary_features(gaussians, style)
    features = features.numpy()
    normals, reliable = _normals(gaussians._scaling.detach().cpu().numpy(),
                                 gaussians._rotation.detach().cpu().numpy())
    old = labels.cpu().numpy()
    region_ids = np.unique(old[old >= 0]).tolist() if selection == "all" else selection
    correction_path = getattr(style, "region_boundary_corrections", "")
    corrections = _read_corrections(Path(correction_path)) if correction_path else {}
    updated, summary = adjust_region_boundaries(
        xyz, opacity, features, old, region_ids, corrections=corrections,
        normals=normals, reliable=reliable)
    # Every region needs a sample for its measure space. Restore an original
    # sample if automatic edits would otherwise remove the last one.
    sample = np.asarray(valid_indices, dtype=np.int64)
    old_sample = old[sample]
    populated = np.unique(old_sample[old_sample >= 0])
    missing = np.setdiff1d(populated, updated[sample])
    while missing.size:
        for lost in missing:
            members = sample[old_sample == lost]
            eligible = np.array([idx for idx in members if int(idx) not in corrections], dtype=np.int64)
            if not eligible.size:
                raise ValueError(f"Boundary corrections would empty the metric sample for region {lost}")
            updated[eligible[opacity[eligible].argmax()]] = lost
        missing = np.setdiff1d(populated, updated[sample])
    changed = updated != old
    pairs, counts = np.unique(np.column_stack((old[changed], updated[changed])), axis=0, return_counts=True)
    summary.update(changed=int(changed.sum()), regions=region_ids,
                   automatic_changes=int(changed.sum()) - summary["explicit_corrections"],
                   transitions={f"{src}->{dst}": int(count) for (src, dst), count in zip(pairs, counts)},
                   opacity_change_fraction=float(opacity[changed].sum() / opacity[old >= 0].sum()),
                   source_regions=reference, dino=dino_source, corrections=correction_path or None)
    return torch.from_numpy(updated), summary, (xyz, opacity, features)


def _neighbors(tree, xyz, ids, k, workers):
    distances, neighbors = tree.query(xyz[ids], k=k + 1, workers=workers)
    # Duplicate centers can place a point's own index anywhere in the returned row.
    distances[neighbors == ids[:, None]] = np.inf
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(distances, order, axis=1), np.take_along_axis(neighbors, order, axis=1)


def _normalized(x):
    lengths = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(lengths, 1e-8)


def _automatic_boundary(xyz, opacity, features, labels, target, locked, normals, reliable,
                        k, min_cosine, min_support, min_vote, radius_factor, max_edit_fraction, workers):
    valid_ids = np.flatnonzero(labels >= 0)
    if len(valid_ids) <= min_support:
        return labels.copy(), {"boundary_candidates": 0, "proposed": 0}
    k = min(k, len(valid_ids) - 1)
    xyz = xyz[valid_ids]
    opacity = opacity[valid_ids]
    local_labels = labels[valid_ids]
    target_member = target[local_labels]
    tree = cKDTree(xyz)
    radii = np.empty(len(xyz), np.float32)
    core = np.empty(len(xyz), bool)
    boundary_parts = []

    for start in range(0, len(xyz), _BATCH_SIZE):
        ids = np.arange(start, min(start + _BATCH_SIZE, len(xyz)))
        distance, neighbors = _neighbors(tree, xyz, ids, k, workers)
        radii[ids] = np.maximum(distance[:, min(5, k - 1)], 1e-8)
        neighbor_labels = local_labels[neighbors]
        different = neighbor_labels != local_labels[ids, None]
        core[ids] = (different.mean(axis=1) <= 0.2) & (opacity[ids] >= 0.1)
        relevant = target_member[ids] | np.any(different & target_member[neighbors], axis=1)
        boundary_parts.append(ids[relevant & different.any(axis=1) & ~core[ids]])

    locked = locked[valid_ids]
    core |= locked
    boundary = np.concatenate(boundary_parts)
    proposal_ids, proposal_labels, proposal_gaps = [], [], []
    counts = np.bincount(local_labels, minlength=len(target))
    for start in range(0, len(boundary), _BATCH_SIZE):
        ids = boundary[start:start + _BATCH_SIZE]
        ids = ids[~locked[ids]]
        if ids.size == 0:
            continue
        distance, neighbors = _neighbors(tree, xyz, ids, k, workers)
        near_scale = np.minimum(radii[ids, None], radii[neighbors])
        relative_distance = distance / near_scale
        f = _normalized(features[valid_ids[ids]].astype(np.float32, copy=False))
        fn = _normalized(features[valid_ids[neighbors]].astype(np.float32, copy=False))
        semantic = np.einsum("ijd,id->ij", fn, f)
        supported = core[neighbors] & (relative_distance <= radius_factor) & (semantic >= min_cosine)
        weights = np.exp(-0.5 * relative_distance**2) * np.maximum(semantic, 0)**2
        weights *= 0.5 + 0.5 * opacity[neighbors]

        if normals is not None:
            n = normals[valid_ids[ids]]
            nn = normals[valid_ids[neighbors]]
            both_reliable = reliable[valid_ids[ids], None] & reliable[valid_ids[neighbors]]
            cosine = np.abs(np.einsum("ijd,id->ij", nn, n)).clip(0, 1)
            delta = (xyz[neighbors] - xyz[ids, None]) / np.maximum(distance[:, :, None], 1e-8)
            offset = np.maximum(np.abs(np.einsum("ijd,id->ij", delta, n)),
                                np.abs(np.einsum("ijd,ijd->ij", delta, nn)))
            supported &= ~both_reliable | ((cosine >= 0.5) & (offset <= 0.85))
            weights *= np.where(both_reliable, cosine * (1.0 - 0.5 * offset.clip(0, 1)), 1.0)

        neighbor_labels = local_labels[neighbors]
        rows = np.arange(len(ids))
        scores = np.zeros((len(ids), len(target)))
        support = np.zeros_like(scores, dtype=np.int32)
        np.add.at(scores, (rows[:, None], neighbor_labels), np.where(supported, weights, 0.0))
        np.add.at(support, (rows[:, None], neighbor_labels), supported)
        best = scores.argmax(axis=1)
        old = local_labels[ids]
        total = scores.sum(axis=1).clip(1e-8)
        vote = scores[rows, best] / total
        changes = ((best != old) & (target[old] | target[best])
                   & (support[rows, best] >= min_support) & (vote >= min_vote))
        proposal_ids.append(ids[changes])
        proposal_labels.append(best[changes])
        proposal_gaps.append((vote - scores[rows, old] / total)[changes])

    output = labels.copy()
    n_proposed = 0
    if proposal_ids:
        ids = np.concatenate(proposal_ids)
        best = np.concatenate(proposal_labels)
        rank = np.argsort(-np.concatenate(proposal_gaps), kind="stable")
        n_proposed = len(rank)
        max_moves = np.minimum(counts - 1, np.maximum(1, (counts * max_edit_fraction).astype(int)))
        incoming, outgoing = np.zeros_like(counts), np.zeros_like(counts)
        for idx in rank:
            old, new = local_labels[ids[idx]], best[idx]
            if outgoing[old] < max_moves[old] and incoming[new] < max_moves[new]:
                output[valid_ids[ids[idx]]] = new
                outgoing[old] += 1
                incoming[new] += 1
    return output, {"boundary_candidates": len(boundary), "proposed": n_proposed}


def adjust_region_boundaries(xyz, opacity, features, labels, region_ids, *, corrections=None,
                             corrections_only=False, normals=None, reliable=None, k=12,
                             min_cosine=0.75, min_support=3, min_vote=0.8, radius_factor=1.8,
                             max_edit_fraction=0.05, workers=2):
    """Return refined labels and counts while retaining the input's point order and IDs."""
    if k < min_support or min_support < 1 or not 0.5 < min_vote <= 1:
        raise ValueError("Choose k >= min_support >= 1 and a majority threshold in (0.5, 1].")
    if not 0 < min_cosine <= 1 or radius_factor <= 0 or not 0 < max_edit_fraction < 1:
        raise ValueError("Invalid cosine limit, distance radius or edit fraction.")
    counts = np.bincount(labels[labels >= 0])
    if not region_ids or any(r < 0 or r >= len(counts) or counts[r] == 0 for r in region_ids):
        raise ValueError("Supply at least one populated region ID in the input partition.")
    target = np.zeros(len(counts), bool)
    target[list(region_ids)] = True
    working = labels.copy()
    locked = np.zeros(len(labels), bool)
    for point, new in (corrections or {}).items():
        if not 0 <= point < len(labels) or not 0 <= new < len(counts) or counts[new] == 0:
            raise ValueError(f"Invalid correction {point} -> {new}.")
        old = labels[point]
        if old < 0 or not (target[old] or target[new]):
            raise ValueError(f"Correction {point}: {old} -> {new} must touch a selected region.")
        working[point] = new
        locked[point] = True
    if np.any(np.bincount(working[working >= 0], minlength=len(counts)) == 0):
        raise ValueError("The corrected partition must retain every existing region.")

    result, diagnostics = working, {"boundary_candidates": 0, "proposed": 0}
    if not corrections_only:
        result, diagnostics = _automatic_boundary(
            xyz, opacity, features, working, target, locked, normals, reliable,
            k, min_cosine, min_support, min_vote, radius_factor, max_edit_fraction, workers)
    changed = result != labels
    transitions, num_transitions = np.unique(
        np.column_stack([labels[changed], result[changed]]), axis=0, return_counts=True)
    return result, {
        **diagnostics,
        "changed": int(changed.sum()),
        "explicit_corrections": int(((working != labels) & locked).sum()),
        "automatic_changes": int((result != working).sum()),
        "transitions": {f"{old}->{new}": int(n) for (old, new), n in zip(transitions, num_transitions)},
    }


def _rebuild_spaces(payload, labels, xyz, opacity, features, changes):
    if not changes.any():
        return payload["metric_spaces"], []
    indices = payload["valid_indices"].numpy()
    old = payload["cluster_labels"].numpy()
    changed = indices[changes[indices]]
    affected = np.union1d(old[changed], labels[changed]).tolist()
    if not affected:
        return payload["metric_spaces"], []
    if not __package__:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from style_pipeline.regions import _build_metric_measure_space

    spaces = list(payload["metric_spaces"])
    params = payload.get("params", {})
    for region_id in affected:
        region = indices[labels[indices] == region_id]
        if region.size == 0:
            raise ValueError(f"Boundary edits would empty the metric sampling for region {region_id}.")
        space = _build_metric_measure_space(
            torch.from_numpy(xyz[region]), torch.from_numpy(features[region].astype(np.float32, copy=False)),
            torch.from_numpy(opacity[region]), int(params.get("m_reps", len(spaces[region_id]["points"]))),
            int(params.get("rep_knn_k", 8)), region_id)
        metric = asdict(space)
        if (not torch.isfinite(metric["measure"]).all() or not torch.isfinite(metric["features"]).all()
                or not torch.isfinite(metric["distances"]).all()
                or not torch.isclose(metric["measure"].sum(), torch.tensor(1.0))):
            raise ValueError(f"Invalid rebuilt metric space for region {region_id}.")
        spaces[region_id] = {**spaces[region_id], **metric}
    return spaces, affected


def refine_region_sidecar(ply, dino, regions, region_ids, *, reference_ply=None,
                         with_normals=True, corrections=None, corrections_only=False, rebuild=True, **settings):
    """Validate cached geometry and return a version-compatible region sidecar and summary."""
    vertices = PlyData.read(ply, mmap="r")["vertex"].data
    xyz = _xyz(vertices)
    fingerprint = _fingerprint(xyz)
    payload = torch.load(regions, map_location="cpu", weights_only=True, mmap=True)
    feature_state = torch.load(dino, map_location="cpu", weights_only=True, mmap=True)
    labels = payload["cluster_labels"].numpy()
    features = feature_state["dino_features"].numpy()
    if labels.ndim != 1 or labels.shape != (len(xyz),) or payload["n_full"] != len(xyz):
        raise ValueError("Region labels and PLY point counts or shapes differ.")
    if features.shape != (len(xyz), payload["feature_dim"]):
        raise ValueError("DINO cache does not match the PLY point count and region feature dimension.")
    old_fingerprint = payload.get("xyz_fingerprint")
    if old_fingerprint is None:
        reference_ply = reference_ply or Path(regions).with_name("point_cloud.ply")
        sibling = PlyData.read(reference_ply, mmap="r")["vertex"].data
        old_fingerprint = _fingerprint(_xyz(sibling))
    if old_fingerprint != fingerprint or feature_state.get("xyz_fingerprint") != fingerprint:
        raise ValueError("Region or DINO geometry fingerprint does not match the requested PLY.")
    if (len(payload["metric_spaces"]) != payload["num_regions"]
            or np.unique(labels[labels >= 0]).tolist() != list(range(payload["num_regions"]))):
        raise ValueError("Region labels and metric spaces must use the same contiguous IDs.")

    opacity = expit(vertices["opacity"])
    normals = reliable = None
    if with_normals and not corrections_only:
        normals, reliable = _normals(
            np.column_stack([vertices[f"scale_{axis}"] for axis in range(3)]),
            np.column_stack([vertices[f"rot_{axis}"] for axis in range(4)]))
    refined, summary = adjust_region_boundaries(
        xyz, opacity, features, labels, region_ids, corrections=corrections, corrections_only=corrections_only,
        normals=normals, reliable=reliable, **settings)
    changed = refined != labels
    summary["opacity_change_fraction"] = float(opacity[changed].sum() / opacity[labels >= 0].sum())
    summary["regions"] = sorted(region_ids)
    summary["mode"] = "corrections_only" if corrections_only else "boundary"
    summary["uses_normals"] = normals is not None
    if rebuild:
        spaces, rebuilt = _rebuild_spaces(payload, refined, xyz, opacity, features, changed)
        summary["rebuilt_metric_regions"] = rebuilt
        payload = {**payload, "cluster_labels": torch.from_numpy(refined), "metric_spaces": spaces,
                   "xyz_fingerprint": fingerprint,
                   "boundary_refinement": {**summary, "source_regions": str(Path(regions).resolve()),
                                           "dino": str(Path(dino).resolve()), "settings": dict(settings)}}
    return payload, summary


def _read_corrections(path):
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Corrections JSON must map point indices to region IDs.")
    result = {}
    for key, value in payload.items():
        if not isinstance(value, int) or isinstance(value, bool) or str(int(key)) != key:
            raise ValueError(f"Invalid correction entry {key}: {value}.")
        result[int(key)] = value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", type=Path, required=True)
    parser.add_argument("--dino", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--reference-ply", type=Path, help="PLY to verify legacy region files without an XYZ fingerprint")
    parser.add_argument("--region", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corrections", type=Path)
    parser.add_argument("--corrections-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-normals", action="store_true")
    parser.add_argument("--k", type=int, default=12)
    parser.add_argument("--min-cosine", type=float, default=0.75)
    parser.add_argument("--min-support", type=int, default=3)
    parser.add_argument("--min-vote", type=float, default=0.8)
    parser.add_argument("--radius-factor", type=float, default=1.8)
    parser.add_argument("--max-edit-fraction", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run is used")
    if args.output is not None and args.output.resolve() in {args.regions.resolve(), args.dino.resolve(), args.ply.resolve()}:
        parser.error("--output must differ from the input paths")
    if args.corrections_only and args.corrections is None:
        parser.error("--corrections-only requires --corrections")
    corrections = _read_corrections(args.corrections) if args.corrections else None
    payload, summary = refine_region_sidecar(
        args.ply, args.dino, args.regions, args.region, reference_ply=args.reference_ply,
        with_normals=not args.no_normals, corrections=corrections, corrections_only=args.corrections_only,
        rebuild=not args.dry_run, k=args.k, min_cosine=args.min_cosine, min_support=args.min_support,
        min_vote=args.min_vote, radius_factor=args.radius_factor, max_edit_fraction=args.max_edit_fraction,
        workers=args.workers)
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=args.output.parent) as temporary:
            with open(temporary.name, "wb") as file:
                torch.save(payload, file)
            args.output.hardlink_to(temporary.name)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
