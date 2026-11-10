"""Match 3D regions to style primitives with fused Gromov-Wasserstein transport."""

import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .dino_distill import _resolve_dino_paths
from .paths import sidecar_path


_FGW_MAX_ITER = 200


def _load_pca_state(gt_path):
    """读 ② GT 缓存里的 pca_state (components[D,1024], mean[1024]); ⑤ 据此把 ③ 的
    1024 维风格基元投到 16 维, 与 ②④ 的特征空间对齐."""
    payload = torch.load(gt_path, map_location="cpu")
    pca_state = payload.get("pca_state")
    if (
        not isinstance(pca_state, dict)
        or "components" not in pca_state
        or "mean" not in pca_state
    ):
        raise KeyError(
            f"[fgw] GT cache {gt_path} has no pca_state (components/mean). "
            f"Re-run train_style.py with --force_dino to regenerate."
        )
    return pca_state


def _project_style_primitives(prototypes, pca_state, feature_dim):
    """③ 的 [K,1024] 风格基元 → 16 维, 与 ②④ 同一 PCA 空间. 复用 GT 提取的共享投影核心
    _project_with_components (semgeo_utils/dino_features.py), 公式 (x-mean)@components.T
    与提取端保持一致; 投影后 L2 归一化, 与 F^X 在 ⑤ 中统一归一化以平衡特征/结构代价
    (见 MetricMeasureSpace 注释)."""
    from semgeo_utils.dino_features import _project_with_components

    comps = pca_state["components"].float()  # [D, 1024]
    mean = pca_state["mean"].float()  # [1024]
    if comps.shape[1] != prototypes.shape[1]:
        raise ValueError(
            f"[fgw] PCA components dim {comps.shape[1]} "
            f"!= prototypes {prototypes.shape[1]}"
        )
    if comps.shape[0] != feature_dim:
        raise ValueError(
            f"[fgw] PCA effective dim {comps.shape[0]} != feature_dim {feature_dim}"
        )
    proj = _project_with_components(prototypes.float(), comps, mean)  # [K, D]
    return F.normalize(proj, dim=-1)


def _style_primitive_metric_space(F_Y):
    """2D 度量测度空间 Y=(P, d_Y, ν, F^Y). d_Y[K,K] = 基元间 L2 距离, 按 diameter
    归一化到 [0,1] (与 ④ d_X 归一方式一致); ν 均匀 (③ 未存簇大小)."""
    K = F_Y.shape[0]
    d_Y = torch.cdist(F_Y, F_Y)  # [K, K]
    C2 = (d_Y / d_Y.max().clamp_min(1e-8)).clamp(0.0, 1.0)
    C2.fill_diagonal_(0.0)
    nu = torch.full((K,), 1.0 / K, dtype=F_Y.dtype)
    return C2, nu


def _weighted_region_vote(assignments, measure, num_primitives):
    """对逐代表点指派执行统一的 μ 加权 region vote。"""
    if torch.is_tensor(assignments):
        assignments = assignments.detach().cpu().numpy()
    if torch.is_tensor(measure):
        measure = measure.detach().cpu().numpy()
    scores = np.bincount(
        np.asarray(assignments, dtype=np.int64),
        weights=np.asarray(measure),
        minlength=num_primitives,
    )
    return int(np.argmax(scores))


def _fgw_match_region(ms, C2, nu, F_Y, alpha, feature_dim):
    """单区域 FGW: O_j (m 代表点) ↔ Y (K 基元). 返回区域匹配的基元下标 k*.
    POT ot.gromov.fused_gromov_wasserstein 求传输计划 T[m,K]; 逐代表点取主导基元
    (argmax_k T[i,k]) 后用 μ_j 加权多数票定区域基元. (T 列和被边际 ν 钉死, 不能用列和.)
    确定性: 显式传 G0=outer(μ,ν) 免依赖 RNG (POT 默认即此, 显式传入更稳)."""
    import ot

    K = F_Y.shape[0]
    if K == 1:
        return 0
    F_X = F.normalize(ms.features.float(), dim=-1)  # [m, 16]
    if F_X.shape[1] != feature_dim:
        raise ValueError(f"[fgw] region feature dim {F_X.shape[1]} != {feature_dim}")
    C1 = (ms.distances.float() / ms.diameter).clamp(0.0, 1.0)
    C1.fill_diagonal_(0.0)
    mu = ms.measure.float()  # [m], sum=1
    M = torch.cdist(F_X, F_Y) ** 2  # [m, K] ∈ [0,4]
    M = M / M.max().clamp_min(1e-8)  # → [0,1]
    G0 = torch.outer(mu, nu)
    T = ot.gromov.fused_gromov_wasserstein(
        M.numpy().astype(np.float64),
        C1.numpy().astype(np.float64),
        C2.numpy().astype(np.float64),
        mu.numpy().astype(np.float64),
        nu.numpy().astype(np.float64),
        alpha=float(alpha),
        G0=G0.numpy().astype(np.float64),
        max_iter=_FGW_MAX_ITER,
        armijo=True,
    )
    return _weighted_region_vote(T.argmax(axis=1), mu, K)


def align_regions_with_primitives(dataset, gaussians, style):
    """Project style primitives into the scene PCA space, match each region with FGW, and save the region and per-Gaussian assignments."""
    primitives = getattr(gaussians, "style_primitives", None)
    metric_spaces = getattr(gaussians, "metric_spaces", None)
    if primitives is None or primitives.numel() == 0:
        print(
            "[fgw] ⑤ skipped: no style_primitives (③ did not run / no --style_image)."
        )
        gaussians.region_primitive_map = None
        gaussians.primitive_labels = None
        return
    if not metric_spaces:
        print("[fgw] ⑤ skipped: no metric_spaces (④ did not run).")
        gaussians.region_primitive_map = None
        gaussians.primitive_labels = None
        return

    gt_path, _ = _resolve_dino_paths(dataset, style)
    pca_state = _load_pca_state(gt_path)
    F_Y = _project_style_primitives(primitives.cpu(), pca_state, style.dino_feature_dim)
    C2, nu = _style_primitive_metric_space(F_Y)
    K = F_Y.shape[0]
    alpha = float(style.fgw_alpha)
    print(
        f"[fgw] start: {len(metric_spaces)} regions x {K} primitives (alpha={alpha:g})"
    )

    region_primitive_map = torch.full((len(metric_spaces),), -1, dtype=torch.long)
    for ms in tqdm(metric_spaces, desc="FGW align"):
        try:
            k = _fgw_match_region(ms, C2, nu, F_Y, alpha, style.dino_feature_dim)
            region_primitive_map[ms.region_id] = k
        except Exception as exc:
            print(
                f"[fgw] region {ms.region_id} "
                f"({ms.n_source_points} pts) failed: {exc}; skipped."
            )

    cluster_labels = getattr(gaussians, "cluster_labels", None)
    if cluster_labels is not None:
        primitive_labels = torch.full_like(cluster_labels, -1)
        valid = cluster_labels >= 0
        primitive_labels[valid] = region_primitive_map[
            cluster_labels[valid].clamp(min=0)
        ]
    else:
        primitive_labels = None

    n_matched = int((region_primitive_map >= 0).sum())
    usage = torch.bincount(region_primitive_map.clamp(min=0), minlength=K).tolist()
    print(
        f"[fgw] ⑤ matched {n_matched}/{len(metric_spaces)} regions; "
        f"primitive usage (per-region count) {usage}",
        flush=True,
    )

    sidecar = sidecar_path(
        dataset.model_path, style.style_iterations, "region_primitives.pt"
    )
    os.makedirs(os.path.dirname(sidecar), exist_ok=True)
    torch.save(
        {
            "version": 1,
            "region_primitive_map": region_primitive_map,  # [R] region_id → primitive_id
            "primitive_labels": primitive_labels,  # [N_full] per-gaussian (-1=none)
            "num_primitives": K,
            "num_regions": len(metric_spaces),
            "fgw_alpha": alpha,
            "pca_state_path": gt_path,
            "feature_dim": int(style.dino_feature_dim),
        },
        sidecar,
    )
    gaussians.region_primitive_map = region_primitive_map
    gaussians.primitive_labels = primitive_labels
    print(f"[fgw] ⑤ done → {sidecar}", flush=True)
