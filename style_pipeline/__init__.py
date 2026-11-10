"""Core stages of the SemGeoSplat stylization pipeline."""

import os


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .output import prepare_output_and_logger, training_report
from .paths import style_output_dir, sidecar_path
from .refine import (
    _resolve_start_ply,
    _resolve_refine_dir,
    _repair_loaded_ply_state,
    compute_importance_scores,
    prune_for_refine,
    _prune_schedule,
    _prune_percent_for_schedule_index,
    _release_camera_gpu_tensors,
    _release_camera_images_tensors,
    _release_training_grads,
    _release_optimizer_state,
    _release_exposure,
)
from .dino_distill import distill_dino
from .style_primitives import _decompose_style_primitives
from .regions import build_structure_aware_regions, MetricMeasureSpace
from .fgw_align import align_regions_with_primitives
from .style_generate import generate_style

__all__ = [
    "prepare_output_and_logger",
    "training_report",
    "style_output_dir",
    "sidecar_path",
    "_resolve_start_ply",
    "_resolve_refine_dir",
    "_repair_loaded_ply_state",
    "compute_importance_scores",
    "prune_for_refine",
    "_prune_schedule",
    "_prune_percent_for_schedule_index",
    "_release_camera_gpu_tensors",
    "_release_camera_images_tensors",
    "_release_training_grads",
    "_release_optimizer_state",
    "_release_exposure",
    "distill_dino",
    "_decompose_style_primitives",
    "build_structure_aware_regions",
    "MetricMeasureSpace",
    "align_regions_with_primitives",
    "generate_style",
]
