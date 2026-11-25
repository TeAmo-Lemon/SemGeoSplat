"""③ 风格图纹理基元分解.

将单张风格图无监督分解为 K 个 L2 归一化的 DINOv3 prototype, 供 ⑤ FGW 匹配.
DINOv3 在本阶段加载一次 (② 蒸馏用磁盘缓存, 不需模型), 用完即释放.
prototypes 常驻 gaussians.style_primitives (GPU), 同时写 sidecar style_primitives.pt.
"""

import hashlib
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
    image_hash = hashlib.sha256(Path(style.style_image).read_bytes()).hexdigest()

    reference_path = str(
        getattr(style, "style_primitives_reference_path", "")).strip()
    if reference_path:
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(
                f"[style] --style_primitives_reference_path not found: {reference_path}")
        try:
            payload = torch.load(
                reference_path, map_location="cpu", weights_only=False)
        except TypeError:  # 兼容较旧的 PyTorch
            payload = torch.load(reference_path, map_location="cpu")
        prototypes = payload.get("prototypes")
        patch_assignments = payload.get("patch_assignments")
        patch_hw = tuple(payload.get("patch_hw", ()))
        patch_size = int(payload.get("patch_size", 0))
        patch_features = payload.get("patch_features")
        if not torch.is_tensor(prototypes) or prototypes.ndim != 2:
            raise ValueError(f"[style] invalid reference prototypes: {reference_path}")
        if int(prototypes.shape[0]) != int(style.style_primitive_k):
            raise ValueError(
                f"[style] reference K={prototypes.shape[0]} != configured "
                f"{style.style_primitive_k}: {reference_path}")
        if (not torch.is_tensor(patch_assignments)
                or len(patch_hw) != 2 or patch_size < 1
                or patch_assignments.numel() != int(patch_hw[0]) * int(patch_hw[1])):
            raise ValueError(
                f"[style] invalid reference patch assignments: {reference_path}")
        if patch_features is not None and (
                not torch.is_tensor(patch_features)
                or patch_features.shape != (patch_assignments.numel(), prototypes.shape[1])):
            raise ValueError(f"[style] invalid reference patch features: {reference_path}")
        stored_hash = payload.get("style_image_sha256")
        if stored_hash and stored_hash != image_hash:
            raise ValueError(f"[style] cached patches belong to a different reference image: {reference_path}")
        if payload.get("dino_model", style.dino_model) != style.dino_model:
            raise ValueError(f"[style] cached patches use a different DINO model: {reference_path}")

        target_device = (gaussians.get_xyz.device
                         if torch.is_tensor(getattr(gaussians, "get_xyz", None))
                         else torch.device("cuda"))
        gaussians.style_primitives = (
            prototypes.float().to(target_device).contiguous())
        gaussians.style_patch_assignments = patch_assignments.long().cpu()
        gaussians.style_patch_hw = patch_hw
        gaussians.style_patch_size = patch_size
        gaussians.style_preprocess_max_size = payload.get("preprocess_max_size", style.style_primitive_size)
        gaussians.style_patch_features = patch_features
        gaussians.style_patch_valid_mask = payload.get("patch_valid_mask")
        output_payload = dict(payload)
        output_payload["reference_path"] = os.path.abspath(reference_path)
        output_payload["reference_mode"] = "exact_full_reuse"
        sidecar = sidecar_path(
            dataset.model_path, style.style_iterations, "style_primitives.pt")
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        torch.save(output_payload, sidecar)
        print(f"[style] Reused {prototypes.shape[0]} exact Full style primitives "
              f"from {reference_path} -> {sidecar} (kept on {target_device})")
        return

    from scripts.extract_dino_features import (
        _load_and_preprocess_image, _extract_patch_features, _load_dinov3)

    device = torch.device("cuda")
    model, patch_size = _load_dinov3(device, style.dino_model)
    try:
        image_tensor, _, patch_hw = _load_and_preprocess_image(
            Path(style.style_image), patch_size=patch_size, device=device,
            max_size=style.style_primitive_size)
        token_map = _extract_patch_features(model, image_tensor, patch_hw)  # [C, ph, pw]
        raw_tokens = token_map.flatten(1).T                           # [N, C]
        feats = F.normalize(raw_tokens, dim=1)
        prototypes = _kmeans_pytorch(
            feats, style.style_primitive_k, style.style_primitive_iters, device)
        prototypes = F.normalize(prototypes, dim=1)                    # [K, C]
        # 每个 patch 到最近 prototype 的聚类指派. ⑥ 用它把风格图按基元分段, 进而为
        # 每个基元建 VGG-NNFM 风格特征库 (与 ⑤ 的区域→基元映射对齐). feats 与 patch_hw
        # 来自同一张图同一预处理, 故 patch 网格与后续 VGG 特征精确对齐 (DINO/VGG 共用
        # ImageNet 归一化, _load_and_preprocess_image 的张量可直接喂 VGG).
        patch_assignments = torch.cdist(feats, prototypes).argmin(dim=1)   # [N_patches]
    finally:
        del model
        torch.cuda.empty_cache()

    gaussians.style_primitives = prototypes.contiguous()
    # 暂存 patch 指派供 ⑥ 直接取用 (同进程内 ⑥ 紧接 ③ 运行); 同时落盘供独立复用.
    # patch_size 一起存, ⑥ 重新预处理同一张风格图时必须用同一 patch_size 才能与 patch
    # 网格对齐 (patch_hw = resized_H/patch_size).
    gaussians.style_patch_assignments = patch_assignments.detach().cpu()
    gaussians.style_patch_hw = patch_hw
    gaussians.style_patch_size = patch_size
    gaussians.style_preprocess_max_size = style.style_primitive_size
    # Transport matches in the scene PCA space fitted on raw tokens, while
    # the appearance banks retain the normalized-token K-Means assignment.
    gaussians.style_patch_features = raw_tokens.detach().float().cpu().contiguous()
    gaussians.style_patch_valid_mask = None
    sidecar = sidecar_path(
        dataset.model_path, style.style_iterations, "style_primitives.pt")
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    torch.save({
        "prototypes": prototypes.cpu(),
        "k": int(prototypes.shape[0]),
        "patch_assignments": patch_assignments.detach().cpu(),
        "patch_hw": tuple(patch_hw),
        "patch_size": int(patch_size),
        "patch_features": gaussians.style_patch_features,
        "style_image_sha256": image_hash,
        "dino_model": style.dino_model,
        "preprocess_max_size": int(style.style_primitive_size),
    }, sidecar)
    print(f"[style] Decomposed {prototypes.shape[0]} style primitives "
          f"from {style.style_image} -> {sidecar} (kept on GPU)")
