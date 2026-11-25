"""Apply the region stage and show its effect on existing Gaussians without fitting appearance."""

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from render import _region_colors_from_labels, render_set
from scene import GaussianModel
from scene.dataset_readers import sceneLoadTypeCallbacks
from semgeo_utils.camera_utils import cameraList_from_camInfos
from semgeo_utils.dataset_paths import validate_scene_path

from .paths import style_output_dir
from .refine import _resolve_refine_dir, _resolve_start_ply
from .regions import build_structure_aware_regions


def _restore_changed_appearance(source, checkpoint, labels, output, fingerprint):
    old = torch.load(checkpoint / "regions.pt", map_location="cpu", weights_only=True)
    if old.get("xyz_fingerprint") not in (None, fingerprint):
        raise ValueError("The style checkpoint has a different Gaussian point order")
    before = PlyData.read(source, mmap="r")
    styled = PlyData.read(checkpoint / "point_cloud.ply", mmap="r")
    original, existing = before["vertex"].data, styled["vertex"].data
    old_labels = old["cluster_labels"]
    if (original.dtype != existing.dtype or len(original) != len(existing)
            or old_labels.shape != labels.shape or len(existing) != labels.numel()):
        raise ValueError("The existing style checkpoint does not match the source Gaussian layout")
    restored = (old_labels != labels).numpy()
    vertices = existing.copy()
    for field in original.dtype.names:
        if field.startswith(("f_dc_", "f_rest_")):
            vertices[field][restored] = original[field][restored]
        elif not np.array_equal(original[field], existing[field]):
            raise ValueError(f"The style checkpoint has different geometry in {field}")
    replacement = PlyElement.describe(vertices, "vertex", comments=styled["vertex"].comments)
    PlyData([replacement if element.name == "vertex" else element for element in styled.elements],
            text=styled.text, byte_order=styled.byte_order, comments=styled.comments,
            obj_info=styled.obj_info).write(output)
    return int(restored.sum()), old_labels


def _preview_cameras(dataset, count):
    kind = validate_scene_path(dataset.source_path, dataset.images)
    if kind == "Colmap":
        info = sceneLoadTypeCallbacks[kind](dataset.source_path, dataset.images, dataset.depths,
                                            dataset.eval, dataset.train_test_exp)
    else:
        info = sceneLoadTypeCallbacks[kind](dataset.source_path, dataset.white_background,
                                            dataset.depths, dataset.eval)
    cameras = info.test_cameras or info.train_cameras
    count = min(count, len(cameras))
    if not count:
        raise ValueError("The scene has no preview cameras")
    indices = ([len(cameras) // 2] if count == 1 else
               [round(i * (len(cameras) - 1) / (count - 1)) for i in range(count)])
    return cameraList_from_camInfos([cameras[i] for i in indices], 1., dataset,
                                   info.is_nerf_synthetic, bool(info.test_cameras))


@torch.no_grad()
def preview_regions(dataset, pipeline, style, args):
    start_ply = Path(_resolve_start_ply(style)).resolve()
    refined = Path(_resolve_refine_dir(str(start_ply))) / "point_cloud.ply"
    source = start_ply if style.skip_refine or not refined.is_file() else refined
    output = Path(dataset.model_path).resolve()
    if not dataset.model_path or (output.exists() and any(output.iterdir())):
        raise ValueError("Choose an empty --model_path for the region preview")

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(str(source), dataset.train_test_exp)
    gaussians.style_source_ply_path = str(source)
    style.style_iterations = 0
    build_structure_aware_regions(dataset, gaussians, style)
    if not gaussians.metric_spaces:
        raise ValueError("No regions were produced for preview")
    views = _preview_cameras(dataset, style.region_preview_views)
    region_labels = gaussians.cluster_labels
    background = torch.tensor([float(dataset.white_background)] * 3, device="cuda")
    directory = Path(style_output_dir(str(output), 0))
    corrected_ply = directory / "point_cloud.ply"
    before_labels = None
    reference = getattr(style, "regions_reference_path", "")
    if reference:
        before_labels = torch.load(reference, map_location="cpu", weights_only=True)["cluster_labels"]

    if style.region_preview_style:
        checkpoint = Path(style.region_preview_style).expanduser().resolve()
        restored, before_labels = _restore_changed_appearance(
            source, checkpoint, region_labels, corrected_ply, gaussians._xyz_fingerprint())
        gaussians.load_ply(str(checkpoint / "point_cloud.ply"), dataset.train_test_exp)
    else:
        restored = 0
        corrected_ply.symlink_to(source)

    if before_labels is not None:
        render_set(str(output), "before", 0, views, gaussians, pipeline, background, dataset.train_test_exp,
                   region_colors=_region_colors_from_labels(before_labels, background.device),
                   write_gt=False, force=True, save_cameras=True)
    if style.region_preview_style:
        gaussians.load_ply(str(corrected_ply), dataset.train_test_exp)
    render_set(str(output), "test", 0, views, gaussians, pipeline, background, dataset.train_test_exp,
               region_colors=_region_colors_from_labels(region_labels, background.device),
               write_gt=False, force=True, save_cameras=True)

    saved_args = dict(vars(args), model_path=str(output), style_iterations=0)
    (output / "cfg_args").write_text(str(Namespace(**saved_args)))
    (output / "region_preview.json").write_text(json.dumps({
        "source_ply": str(source), "region_reference": reference or None,
        "styled_checkpoint": style.region_preview_style or None,
        "restored_gaussians": restored, "view_count": len(views),
        "training_steps": 0,
    }, indent=2) + "\n")
    print(f"[region] Preview rendered {len(views)} views with no training steps -> {output}", flush=True)
