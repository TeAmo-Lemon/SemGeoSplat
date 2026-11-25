"""Fit shared tone curves to the visible stylized surface of a scene."""

import math

import torch
from PIL import Image

from gaussian_renderer import render, render_features
from semgeo_utils.sh_utils import C0, SH2RGB


def _chroma_energy(rgb, luma):
    gray = (rgb * luma).sum(dim=-1, keepdim=True)
    return (rgb - gray).square().mean(dim=-1)


def _reference_stats(reference_image, reference_box, luma):
    with Image.open(reference_image) as image:
        rgba = image.convert("RGBA")
        if reference_box is not None:
            width, height = rgba.size
            x0, y0, x1, y1 = reference_box
            rgba = rgba.crop((math.floor(x0 * width), math.floor(y0 * height),
                              math.ceil(x1 * width), math.ceil(y1 * height)))
        rgba.thumbnail((384, 384), Image.Resampling.BOX)
        pixels = torch.frombuffer(bytearray(rgba.tobytes()), dtype=torch.uint8)
    pixels = pixels.reshape(-1, 4).to(device=luma.device, dtype=torch.float32).div_(255)
    rgb, alpha = pixels[:, :3], pixels[:, 3]
    mass = alpha.sum()
    if mass == 0:
        return None
    weights = alpha / mass
    return (rgb * weights[:, None]).sum(dim=0), (_chroma_energy(rgb, luma) * weights).sum()


@torch.no_grad()
def calibrate_style_tone(gaussians, cameras, pipe, background, editable,
                         reference_image, strength, *, reference_box=None):
    """Fit Gaussian DC using sampled training camera renders.

    Reduce excess chroma variation and fit monotone, endpoint-preserving RGB
    curves. Re-render during the shared fit to account for actual visibility,
    composition and clipping. Leave higher-order SH untouched.
    """
    if not 0 <= strength <= 1:
        raise ValueError("Tone strength must be in [0, 1]")
    report = {"strength": strength}
    if strength == 0:
        return {**report, "status": "disabled"}

    dc = gaussians._features_dc
    editable = editable.to(device=dc.device, dtype=torch.bool)
    if editable.shape != (dc.shape[0],):
        raise ValueError("Tone edit mask must contain one entry per Gaussian")
    if not cameras or not editable.any():
        return {**report, "status": "no_editable_view"}

    luma = dc.new_tensor([0.2126, 0.7152, 0.0722])
    reference_stats = _reference_stats(reference_image, reference_box, luma)
    if reference_stats is None:
        return {**report, "status": "no_visible_reference"}
    reference, reference_energy = reference_stats

    cameras = sorted(cameras, key=lambda cam: cam.image_name)
    count = min(6, len(cameras))
    indices = [round(i * (len(cameras) - 1) / max(1, count - 1)) for i in range(count)]
    mask = editable[:, None].to(dc.dtype)
    mode = "antialiased" if pipe.antialiasing else "classic"
    views = []
    for i in indices:
        camera = cameras[i]
        coverage = render_features(camera, gaussians, mask, rasterize_mode=mode)[0]
        weights = coverage.clamp(0, 1).where(coverage >= 0.5, 0)
        mass = weights.sum()
        if mass > 0:
            views.append((camera, weights / mass))
    if not views:
        return {**report, "status": "no_editable_view"}

    def rendered_stats(*, include_chroma=False):
        means = []
        energies = []
        for camera, weights in views:
            rgb = render(camera, gaussians, pipe, background,
                         use_trained_exp=False, no_densify_hook=True)["render"].movedim(0, -1)
            means.append((rgb * weights[..., None]).sum(dim=(0, 1)))
            if include_chroma:
                energies.append((_chroma_energy(rgb, luma) * weights).sum())
        mean = torch.stack(means).mean(dim=0)
        return mean, torch.stack(energies).mean() if include_chroma else None

    before, before_energy = rendered_stats(include_chroma=True)
    target = before.lerp(reference, strength)
    reference_spread = (reference_energy - _chroma_energy(reference, luma)).clamp_min(0).sqrt()
    source_spread = (before_energy - _chroma_energy(before, luma)).clamp_min(0).sqrt()
    # The RGB curves fit the color cast. Chroma scaling handles variation around it,
    # so a grayscale reference also suppresses colors of individual objects.
    scale = 1 - strength * (1 - ((reference_spread + 1e-6) / (source_spread + 1e-6)).clamp(max=1))
    editable_rows = editable.nonzero().flatten()
    original = dc[editable_rows]
    original_rgb = SH2RGB(original)
    gray = (original_rgb * luma).sum(dim=-1, keepdim=True)
    base = gray.lerp(original_rgb, scale)
    bounded = base.clamp(0, 1)

    def apply_curve(log_gain):
        # Multiplying the odds of in-gamut DC colors fixes black and white and
        # compresses the correction near the extremes without creating flat tails.
        gain = log_gain.exp()
        adjusted = bounded * gain / (1 - bounded + bounded * gain)
        dc[editable_rows] = original + ((base - original_rgb) + (adjusted - bounded)) / C0

    low = torch.full_like(target, -4.0)
    high = torch.full_like(target, 4.0)
    keep = False
    try:
        for _ in range(10):
            log_gain = (low + high) / 2
            apply_curve(log_gain)
            after, _ = rendered_stats()
            too_dark = after < target
            low = torch.where(too_dark, log_gain, low)
            high = torch.where(too_dark, high, log_gain)
        log_gain = (low + high) / 2
        apply_curve(log_gain)
        after, _ = rendered_stats()
        if not torch.isfinite(after).all():
            raise RuntimeError("Nonfinite tone calibration result")
        keep = True
    finally:
        if not keep:
            dc[editable_rows] = original

    return {
        **report,
        "status": "applied",
        "cameras": [camera.image_name for camera, _ in views],
        "reference_box": reference_box,
        "reference_rgb": reference.tolist(),
        "target_rgb": target.tolist(),
        "before_rgb": before.tolist(),
        "after_rgb": after.tolist(),
        "reference_chroma_spread": float(reference_spread),
        "before_chroma_spread": float(source_spread),
        "chroma_scale": float(scale),
        "rgb_log_gain": log_gain.tolist(),
    }
