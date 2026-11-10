"""Distill cached image features into per-Gaussian semantic features."""

import os
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import render_features
from scene import GaussianModel, Scene
from semgeo_utils.loss_utils import l1_loss

from .refine import _resolve_start_ply, _resolve_refine_dir


def _resolve_dino_paths(dataset, style):

    if style.dino_features_path:
        gt_path = style.dino_features_path
    else:
        gt_path = os.path.join(
            _resolve_refine_dir(_resolve_start_ply(style)), "dino_features.pt"
        )
    image_dir = style.dino_image_dir or os.path.join(
        dataset.source_path, dataset.images or "images"
    )
    return gt_path, image_dir


def _ensure_gt_dino_features(dataset, style):
    """Return path to the GT 2D DINO feature file, extracting it if missing.

    The file caches DINOv3+PCA features per training image so repeated runs
    never re-run the (expensive) DINOv3 extraction. --force_dino ignores the
    cache: deletes the file so all images are re-extracted and PCA re-fitted
    (否则 _load_existing_payload 复用旧 pca_state + skip_existing 跳过已缓存图).
    """
    gt_path, image_dir = _resolve_dino_paths(dataset, style)
    if style.force_dino and os.path.isfile(gt_path):
        print(
            f"[dino] --force_dino set, deleting cached GT features {gt_path} for full re-extraction"
        )
        os.remove(gt_path)
    if os.path.isfile(gt_path):
        print(f"[dino] Using existing GT features: {gt_path}")
        return gt_path

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"[dino] GT features '{gt_path}' missing and image dir '{image_dir}' not found. "
            f"Set --dino_image_dir to a directory containing the scene images."
        )

    print(
        f"[dino] GT features not found; extracting DINOv3 features from {image_dir} ..."
    )
    from semgeo_utils.dino_features import (
        extract_dino_features_for_images,
        _list_images,
    )

    image_paths = _list_images(
        Path(image_dir), [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
    )
    if len(image_paths) == 0:
        raise RuntimeError(f"[dino] No images found in {image_dir}")
    extract_dino_features_for_images(
        image_paths=image_paths,
        input_dir=Path(image_dir),
        output_dir=Path(gt_path),  # .pt suffix -> written verbatim
        feature_dim=style.dino_feature_dim,
        device="cuda",
        skip_existing=True,
        max_image_size=style.dino_max_image_size
        if style.dino_max_image_size > 0
        else None,
        model_id=style.dino_model,
    )
    return gt_path


def _load_gt_dino_index(gt_path, feature_dim):
    """Load GT feature file into lookup tables for matching cameras to features.

    Returns (by_relpath, by_stem, ambiguous_stems):
      by_relpath : {'sub/dir/name': tensor(H,W,D)}  full relative path minus suffix
      by_stem    : {'name': tensor(H,W,D)}          filename stem (may collide)
      ambiguous_stems : stems that appeared under >1 subdirectory
    """
    payload = torch.load(gt_path, map_location="cpu")
    stored_dim = payload.get("feature_dim", None)
    if stored_dim is not None and stored_dim != feature_dim:
        raise ValueError(
            f"[dino] GT feature dim {stored_dim} != --dino_feature_dim {feature_dim}"
        )
    by_relpath, by_stem, ambiguous = {}, {}, set()
    for key, tensor in payload["features"].items():
        by_relpath[Path(key).with_suffix("").as_posix()] = tensor
        stem = Path(key).stem
        if stem in by_stem:
            ambiguous.add(stem)
        by_stem[stem] = tensor
    return by_relpath, by_stem, ambiguous


def _lookup_gt_feature(gt_index, image_name):
    """Match a camera's image_name to its GT feature. Prefers the full relative
    path (robust to nested multi-cam dirs); falls back to the filename stem
    (robust to a different extraction input_dir). None if unmatched; raises on an
    ambiguous stem that would silently mismatch."""
    by_relpath, by_stem, ambiguous = gt_index
    rel = Path(image_name).with_suffix("").as_posix()
    if rel in by_relpath:
        return by_relpath[rel]
    stem = Path(image_name).stem
    if stem in ambiguous:
        raise KeyError(
            f"[dino] image '{image_name}' matches ambiguous stem '{stem}' across "
            f"subdirectories; GT keys and camera names disagree on the path prefix"
        )
    return by_stem.get(stem, None)


def distill_dino(dataset, gaussians: GaussianModel, scene: Scene, style):
    """Freeze geometry and colors, then fit per-Gaussian features to cached image features. Reuse a sidecar only when its shape and geometry fingerprint match."""
    if style.dino_distill_iterations <= 0:
        print("[dino] dino_distill_iterations <= 0; skipping DINO stage")
        return

    n_points = gaussians.get_xyz.shape[0]

    sidecar = os.path.join(
        _resolve_refine_dir(_resolve_start_ply(style)), "dino_features_gaussian.pt"
    )

    if os.path.isfile(sidecar):
        if not style.force_dino:
            try:
                gaussians.load_dino_sidecar(
                    sidecar, expected_dim=style.dino_feature_dim
                )
                print(
                    f"[dino] Loaded cached per-gaussian DINO features from {sidecar} "
                    f"({n_points} pts, {style.dino_feature_dim}D); skipping distillation."
                )
                return
            except Exception as exc:
                print(f"[dino] Cached sidecar invalid ({exc}); re-distilling.")
        else:
            print(
                f"[dino] --force_dino set, deleting cached per-gaussian sidecar {sidecar}"
            )
            os.remove(sidecar)

    gt_path = _ensure_gt_dino_features(dataset, style)
    gt_index = _load_gt_dino_index(gt_path, style.dino_feature_dim)

    # Freeze everything but the new DINO features.
    for p in (
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ):
        p.requires_grad_(False)
    gaussians.dino_training_setup(
        style.dino_lr,
        style.dino_feature_dim,
        dino_lr_final=style.dino_lr_final,
        max_steps=style.dino_distill_iterations,
    )

    cameras = scene.getTrainCameras().copy()
    missing = [
        c.image_name
        for c in cameras
        if _lookup_gt_feature(gt_index, c.image_name) is None
    ]
    if missing:
        raise KeyError(
            f"[dino] {len(missing)} training views have no GT DINO feature "
            f"(e.g. {missing[:3]}). Ensure extraction covered {dataset.source_path}."
        )

    ema_loss = 0.0
    progress_bar = tqdm(range(style.dino_distill_iterations), desc="DINO distill")
    for iteration in range(1, style.dino_distill_iterations + 1):
        gaussians.update_dino_learning_rate(iteration)
        cam = cameras[randint(0, len(cameras) - 1)]
        rendered = render_features(
            cam, gaussians, gaussians.get_dino_features
        )  # [D, H, W]

        gt = (
            _lookup_gt_feature(gt_index, cam.image_name)
            .to("cuda")
            .permute(2, 0, 1)
            .unsqueeze(0)
        )  # [1, D, H, W]
        if gt.shape[-2:] != rendered.shape[-2:]:
            gt = F.interpolate(
                gt, size=rendered.shape[-2:], mode="bilinear", align_corners=False
            )
        gt = gt.squeeze(0)

        loss = style.dino_lambda * l1_loss(rendered, gt)
        loss.backward()
        gaussians.dino_optimizer.step()
        gaussians.dino_optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"DINO L1": f"{ema_loss:.6f}"})
                progress_bar.update(10)
    progress_bar.close()

    gaussians.save_dino_sidecar(sidecar)
    print(
        f"[dino] Saved DINO sidecar ({n_points} pts, {style.dino_feature_dim}D): {sidecar}"
    )

    if gaussians.dino_optimizer is not None:
        gaussians.dino_optimizer.state.clear()
        gaussians.dino_optimizer = None
    torch.cuda.empty_cache()
