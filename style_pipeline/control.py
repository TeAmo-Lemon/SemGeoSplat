"""User controls shared by transport, appearance optimization, and export."""

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass(frozen=True)
class StyleControl:
    strengths: torch.Tensor
    locked: torch.Tensor
    target_boxes: dict[int, tuple[float, float, float, float]]
    texture_mixes: dict[int, tuple[tuple[float, float, float, float], float]] = field(default_factory=dict)
    texture_source_biases: dict[int, tuple[tuple[float, float, float], float]] = field(default_factory=dict)
    texture_source_profiles: dict[int, str] = field(default_factory=dict)


def style_box_mask(box, patch_hw):
    """Flattened row-major mask of patch centers inside a normalized XY box."""
    height, width = patch_hw
    x0, y0, x1, y1 = box
    x = (torch.arange(width, dtype=torch.float64) + 0.5) / width
    y = (torch.arange(height, dtype=torch.float64) + 0.5) / height
    return (((y >= y0) & (y < y1))[:, None]
            & ((x >= x0) & (x < x1))[None, :]).flatten()


def _unit_value(value, field):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 0 <= value <= 1):
        raise ValueError(f"{field} must be a number in [0, 1]")
    return float(value)


def _reference_box(value, field, patch_hw):
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field} must be [x0, y0, x1, y1]")
    box = tuple(_unit_value(v, field) for v in value)
    x0, y0, x1, y1 = box
    if not (x0 < x1 and y0 < y1):
        raise ValueError(f"{field} must have positive width and height")
    if not style_box_mask(box, patch_hw).any():
        raise ValueError(f"{field} contains no DINO patch center")
    return box


def _source_bias(value, region):
    field = f"Region {region} texture_mix.source_bias"
    if (not isinstance(value, dict) or not {"anchor_xyz", "bias"} <= value.keys()
            or value.keys() - {"anchor_xyz", "bias", "profile"}):
        raise ValueError(f"{field} needs anchor_xyz and bias, with optional profile")
    profile = value.get("profile", "partition")
    if profile not in ("partition", "centroid"):
        raise ValueError(f"{field}.profile must be partition or centroid")
    xyz, bias = value["anchor_xyz"], value["bias"]
    if (not isinstance(xyz, list) or len(xyz) != 3
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in xyz)):
        raise ValueError(f"{field}.anchor_xyz must be three finite world coordinates")
    if type(bias) not in (int, float) or not math.isfinite(bias) or not -1 <= bias <= 1:
        raise ValueError(f"{field}.bias must be a number in [-1, 1]")
    return (tuple(float(v) for v in xyz), float(bias)), profile


def load_style_control(path, num_regions, patch_hw, *, default_strength=1.0):
    """Read a region recipe. A zero regional strength locks SH during training."""
    multiplier = _unit_value(default_strength, "style_control_strength")
    payload = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
    if not isinstance(payload, dict) or payload.keys() - {"default_region_strength", "regions"}:
        raise ValueError("Style control must contain only default_region_strength and regions")
    default = _unit_value(payload.get("default_region_strength", 1.0), "default_region_strength")
    strengths = torch.full((num_regions,), default, dtype=torch.float32)
    boxes = {}
    texture_mixes = {}
    texture_source_biases = {}
    profiles = {}
    overrides = payload.get("regions", {})
    if not isinstance(overrides, dict):
        raise ValueError("Style control regions must be an object keyed by region ID")
    for key, entry in overrides.items():
        if not key.isascii() or not key.isdecimal() or str(int(key)) != key:
            raise ValueError(f"Invalid region ID {key!r} in style control")
        region = int(key)
        if not 0 <= region < num_regions:
            raise ValueError(f"Region {region} is outside the available range [0, {num_regions})")
        if not isinstance(entry, dict) or entry.keys() - {"strength", "reference_box", "texture_mix"}:
            raise ValueError(f"Region {region} permits strength, reference_box and texture_mix")
        if "reference_box" in entry:
            boxes[region] = _reference_box(entry["reference_box"], f"Region {region} reference_box", patch_hw)
        if "texture_mix" in entry:
            mix = entry["texture_mix"]
            required_keys = {"box", "fraction"}
            if (not isinstance(mix, dict) or not required_keys <= mix.keys()
                    or mix.keys() - (required_keys | {"source_bias"})):
                raise ValueError(f"Region {region} texture_mix needs box and fraction, with optional source_bias")
            box = _reference_box(mix["box"], f"Region {region} texture_mix.box", patch_hw)
            fraction = _unit_value(mix["fraction"], f"Region {region} texture_mix.fraction")
            texture_mixes[region] = box, fraction
            if "source_bias" in mix:
                texture_source_biases[region], profiles[region] = _source_bias(mix["source_bias"], region)
        if "strength" in entry or region in boxes or region in texture_mixes:
            strengths[region] = _unit_value(entry.get("strength", 1.0), f"Region {region} strength")
    return StyleControl(strengths * multiplier, strengths == 0, boxes, texture_mixes,
                        texture_source_biases, profiles)
