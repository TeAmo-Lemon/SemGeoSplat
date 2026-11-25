"""Inspect stylization regions and export a PLY with new regional strengths."""

import argparse
import colorsys
import hashlib
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from plyfile import PlyData, PlyElement


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def _list_regions(iteration):
    regions = _load(iteration / "regions.pt")
    labels = regions["cluster_labels"]
    counts = torch.bincount(labels[labels >= 0], minlength=regions["num_regions"])
    matches_path = iteration / "region_primitives.pt"
    matches = _load(matches_path) if matches_path.is_file() else {}
    hard = matches.get("region_primitive_map")
    weights = matches.get("region_primitive_weights")
    sh_path = iteration / "style_sh.pt"
    preset = _load(sh_path) if sh_path.is_file() else {}
    trained = preset.get("trained_regions")
    strengths = preset.get("strengths")
    diagnostics = matches.get("spatial_fgw_diagnostics", [])
    show_quota = any(row and "target_zone_fraction" in row for row in diagnostics)
    show_spatial_quota = any(row and "source_bias" in row for row in diagnostics)
    show_centroid = any(row and "target_zone_moment" in row for row in diagnostics)
    num_colors = int(labels.max()) + 1 if labels.numel() else 0
    header = "region\tcolor\tgaussians\tprimitive\tweight\ttrained\tstrength"
    if show_quota:
        header += "\trequested_fraction\ttransported_fraction"
    if show_spatial_quota:
        header += "\tsource_bias\tnear_box_fraction\tfar_box_fraction"
    if show_centroid:
        header += "\trequested_mean_proximity\tactual_mean_proximity"
    print(header)
    for i, count in enumerate(counts.tolist()):
        rgb = colorsys.hsv_to_rgb(i / max(num_colors, 1), 0.85, 1.0)
        color = "#" + "".join(f"{round(channel * 255):02x}" for channel in rgb) if count else "—"
        cols = [str(i), color, str(count), str(int(hard[i])) if hard is not None else "—"]
        cols.extend([
            f"{float(weights[i].max()):.4f}" if weights is not None else "—",
            str(bool(trained[i])) if trained is not None else "—",
            f"{float(strengths[i]):g}" if strengths is not None else "—",
        ])
        row = diagnostics[i] if i < len(diagnostics) else None
        if show_quota:
            if row and "target_zone_fraction" in row:
                cols.extend(f"{row[key]:.6f}" for key in ("target_zone_fraction", "target_zone_mass"))
            else:
                cols.extend(["—", "—"])
        if show_spatial_quota:
            keys = ("source_bias", "target_zone_near_mass_fraction", "target_zone_far_mass_fraction")
            cols.extend(f"{row[key]:.6f}" if row and key in row else "—" for key in keys)
        if show_centroid:
            fraction = row.get("target_zone_fraction", 0) if row else 0
            for key in ("target_zone_moment", "actual_zone_moment"):
                cols.append(f"{row[key] / fraction:.6f}" if fraction > 0 and key in row else "—")
        print("\t".join(cols))


def _strength(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("strength must be in [0, 1]")
    return value


def _region_override(value):
    region, separator, strength = value.partition("=")
    if not separator or not region.isascii() or not region.isdecimal():
        raise argparse.ArgumentTypeError("expected REGION_ID=STRENGTH")
    return int(region), _strength(strength)


def _choose_strengths(preset, global_strength, overrides):
    trained = preset["trained_regions"].bool()
    strengths = preset["strengths"].clone().float()
    if global_strength is not None:
        strengths.fill_(global_strength)
    strengths[~trained] = 0
    for region, value in overrides:
        if region >= trained.numel():
            raise ValueError(f"Unknown region {region}")
        if value > 0 and not trained[region]:
            raise ValueError(f"Region {region} was locked or unmatched during training")
        strengths[region] = value
    return strengths


def _apply_coefficients(vertices, indices, amount, styled, prefix):
    fields = sorted((name for name in vertices.dtype.names if name.startswith(prefix)),
                    key=lambda name: int(name.rsplit("_", 1)[1]))
    flat = styled.transpose(1, 2).flatten(1).numpy()
    if flat.shape != (indices.size, len(fields)):
        raise ValueError(f"Incompatible trained SH shape for {prefix}")
    active = amount > 0
    rows = indices[active]
    alpha = amount[active]
    if rows.size == 0:
        return
    for j, field in enumerate(fields):
        original = vertices[field][rows]
        target = flat[active, j]
        vertices[field][rows] = np.where(alpha == 1, target, original + alpha * (target - original))


def export_ply(iteration, output, source_override=None, global_strength=None, overrides=()):
    preset = _load(iteration / "style_sh.pt")
    regions = _load(iteration / "regions.pt")
    if preset.get("version") != 1:
        raise ValueError("Unsupported style_sh.pt format")
    source = source_override or Path(preset["source_ply_path"])
    source = source.resolve()
    output = output.resolve()
    if output in (source, (iteration / "point_cloud.ply").resolve()):
        raise ValueError("Choose an output path separate from the original and trained PLY")
    ply = PlyData.read(source, mmap="r")
    vertex = ply["vertex"]
    xyz = np.column_stack([vertex[axis] for axis in ("x", "y", "z")]).astype(np.float32, copy=False)
    fingerprint = hashlib.md5(xyz.tobytes()).hexdigest()
    if len(xyz) != preset["num_points"] or fingerprint != preset["xyz_fingerprint"]:
        raise ValueError("The source PLY point order or geometry differs from the trained run")

    labels = regions["cluster_labels"].long()
    strengths = _choose_strengths(preset, global_strength, overrides)
    if labels.shape != (len(xyz),) or strengths.shape != (regions["num_regions"],):
        raise ValueError("Region labels and SH preset do not match")
    indices = preset.get("point_indices")
    indices = torch.arange(len(xyz)) if indices is None else indices.long()
    if indices.ndim != 1 or (indices.numel() and (indices.min() < 0 or indices.max() >= len(xyz))):
        raise ValueError("Invalid point indices in the SH preset")
    selected_labels = labels[indices].numpy()
    if (selected_labels < -1).any() or (selected_labels >= len(strengths)).any():
        raise ValueError("Region sidecar contains invalid point labels")
    amount = np.pad(strengths.numpy(), (1, 0))[selected_labels + 1]
    indices = indices.numpy()

    data = vertex.data.copy()
    _apply_coefficients(data, indices, amount, preset["styled_dc"], "f_dc_")
    _apply_coefficients(data, indices, amount, preset["styled_rest"], "f_rest_")
    replaced = PlyElement.describe(data, "vertex", comments=vertex.comments)
    result = PlyData([replaced if element is vertex else element for element in ply.elements],
                     text=ply.text, byte_order=ply.byte_order, comments=ply.comments, obj_info=ply.obj_info)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output.parent, suffix=".ply.tmp", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        result.write(temp_path)
        os.replace(temp_path, output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Exported {output} ({int(np.count_nonzero(amount)):,} controlled Gaussians)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="Show region IDs and trained strengths")
    list_parser.add_argument("iteration", type=Path)
    export_parser = subparsers.add_parser("export", help="Write a new PLY on CPU without retraining")
    export_parser.add_argument("iteration", type=Path)
    export_parser.add_argument("--output", type=Path, required=True)
    export_parser.add_argument("--source-ply", type=Path, help="Override the recorded original PLY path")
    export_parser.add_argument("--strength", type=_strength, help="Set a strength for every trained region")
    export_parser.add_argument("--region", type=_region_override, action="append", default=[],
                               metavar="ID=STRENGTH", help="Override an individual region, repeatable")
    args = parser.parse_args()
    try:
        if args.command == "list":
            _list_regions(args.iteration)
        else:
            export_ply(args.iteration, args.output, args.source_ply, args.strength, args.region)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
