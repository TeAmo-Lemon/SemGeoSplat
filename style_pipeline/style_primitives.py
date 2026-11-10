"""③ 风格图纹理基元分解.

将单张风格图无监督分解为 K 个 L2 归一化的 DINOv3 prototype, 供 ⑤ FGW 匹配.
DINOv3 在本阶段加载一次 (② 蒸馏用磁盘缓存, 不需模型), 用完即释放.
prototypes 常驻 gaussians.style_primitives (GPU), 同时写 sidecar style_primitives.pt.
"""

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .paths import sidecar_path


_KMEANS_SEED = 0


def _kmeans_pytorch(x, k, iters, device):
    """Plain-PyTorch K-Means on L2-normalized rows. Returns [k, D] centroids
    (not yet renormalized). Reuses a single cdist per iteration; empty clusters
    keep their previous centroid so the count stays exactly k. 初始化用固定
    seed 的局部 Generator (不动全局 RNG), 保证跨运行聚类结果确定."""
    n = x.shape[0]
    k = min(k, n)
    gen = torch.Generator(device=device).manual_seed(_KMEANS_SEED)
    centroids = x[torch.randperm(n, device=device, generator=gen)[:k]].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centroids).argmin(dim=1)
        sums = torch.zeros_like(centroids)
        counts = torch.zeros(k, device=device)
        sums.index_add_(0, assign, x)
        counts.index_add_(0, assign, torch.ones(n, device=device))
        new = sums / counts.clamp_min(1.0).unsqueeze(1)
        empty = counts == 0
        new[empty] = centroids[empty]
        centroids = new
    return centroids


def _decompose_style_primitives(dataset, gaussians, style):
    """Decompose the style image into K L2-normalized DINOv3 prototypes, kept on
    GPU as `gaussians.style_primitives` and also written to a sidecar. Reuses the
    DINOv3 preprocessor from extract_dino_features so preprocessing stays identical
    to the GT feature pipeline."""
    if not style.style_image:
        return
    if not os.path.isfile(style.style_image):
        raise FileNotFoundError(f"[style] --style_image not found: {style.style_image}")

    from semgeo_utils.dino_features import (
        _load_and_preprocess_image,
        _extract_patch_features,
        _load_dinov3,
    )

    device = torch.device("cuda")
    model, patch_size = _load_dinov3(device, style.dino_model)
    try:
        image_tensor, _, patch_hw = _load_and_preprocess_image(
            Path(style.style_image),
            patch_size=patch_size,
            device=device,
            max_size=style.style_primitive_size,
        )
        feats = _extract_patch_features(
            model, image_tensor, patch_hw
        )  # [C, ph, pw], C = hidden_size
        feat_dim = feats.shape[0]
        feats = feats.reshape(feat_dim, -1).permute(1, 0)  # [N, C]
        feats = F.normalize(feats, dim=1)
        prototypes = _kmeans_pytorch(
            feats, style.style_primitive_k, style.style_primitive_iters, device
        )
        prototypes = F.normalize(prototypes, dim=1)  # [K, C]

        patch_assignments = torch.cdist(feats, prototypes).argmin(dim=1)  # [N_patches]
    finally:
        del model
        torch.cuda.empty_cache()

    gaussians.style_primitives = prototypes.contiguous()

    gaussians.style_patch_assignments = patch_assignments.detach().cpu()
    gaussians.style_patch_hw = patch_hw
    gaussians.style_patch_size = patch_size
    sidecar = sidecar_path(
        dataset.model_path, style.style_iterations, "style_primitives.pt"
    )
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    torch.save(
        {
            "prototypes": prototypes.cpu(),
            "k": int(prototypes.shape[0]),
            "patch_assignments": patch_assignments.detach().cpu(),
            "patch_hw": tuple(patch_hw),
            "patch_size": int(patch_size),
        },
        sidecar,
    )
    print(
        f"[style] Decomposed {prototypes.shape[0]} style primitives "
        f"from {style.style_image} -> {sidecar} (kept on GPU)"
    )
