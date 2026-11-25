"""style_pipeline: train_style.py 的六阶段实现包.

阶段划分 (架构书 ①~⑥):
  ① refine              — 剪枝 + 光度精修 + refine 尾部显存释放
  ② dino_distill        — per-gaussian DINO 蒸馏
  ③ style_primitives    — 风格图纹理基元分解
  ④ regions             — 结构感知区域与度量测度空间
  ⑤ fgw_align           — FGW 跨域拓扑对齐
  ⑥ style_generate      — 多视角一致性风格生成

train_style.py 是入口: 设置 CUDA env → import 本包 → training() 编排各阶段.
本包各子模块会 import torch, 故本 __init__ 顶部也 setdefault 一下
PYTORCH_CUDA_ALLOC_CONF (train_style.py 入口已先设; 此处防御直接 import 本包的场景).
"""

import os

# 训练全程会撑大并碎片化 CUDA caching allocator 池; expandable_segments 让段可伸缩、
# 能部分归还, 基本消除碎片. 必须在 CUDA 初始化 (首次 .cuda()) 之前设置. train_style.py
# 入口已设; 此处 setdefault 防御直接 `import style_pipeline` 的场景.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from .output import prepare_output_and_logger
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
