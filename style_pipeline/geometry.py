"""共享几何工具 (④ 区域生长 / ⑥ 风格一致性正则 共用).

_compute_normals: 高斯法线 = 协方差最小特征向量.
compute_knn_graph: 单一可替换 KNN 接口 (cdist 默认 / faiss 可选).
"""

import numpy as np
import torch
import torch.nn.functional as F

from semgeo_utils.general_utils import build_rotation


def _compute_normals(scaling_active, rotation_norm):
    """每个高斯的法线 = 协方差矩阵最小特征值的特征向量
    = build_rotation(R) 中 argmin(scale) 对应的列 (cov = R diag(s²) R^T)."""
    R = build_rotation(rotation_norm)
    min_idx = torch.argmin(scaling_active, dim=1)  # [N]
    ar = torch.arange(R.shape[0], device=R.device)
    normals = R[ar, :, min_idx]  # [N,3]
    return F.normalize(normals, dim=-1)


def _knn_cdist(points, k_max, chunk_size):
    """Find K nearest neighbors with chunked torch.cdist; indices and distances include the query point."""
    M = points.shape[0]
    k = min(k_max, M)
    all_dist = torch.full((M, k), float("inf"), device=points.device)
    all_idx = torch.full((M, k), -1, dtype=torch.long, device=points.device)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        d = torch.cdist(points[start:end], points)

        rows = torch.arange(end - start, device=points.device)
        d[rows, start + rows] = float("inf")
        vals, idx = torch.topk(d, k, dim=1, largest=False)
        all_dist[start:end] = vals
        all_idx[start:end] = idx
        del d
    return all_idx, all_dist


def _knn_faiss(points, k_max):
    """Find K nearest neighbors with FAISS, using GPU when available and CPU otherwise."""
    import faiss

    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    n, d = pts.shape
    k = min(k_max, n)

    def _build(on_gpu):
        idx = faiss.IndexFlatL2(d)
        res = None
        if on_gpu:
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        idx.add(pts)
        return idx, res

    use_gpu = hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()
    try:
        index, _res = _build(use_gpu)
        D, I = index.search(pts, k + 1)
    except RuntimeError:
        index, _res = _build(False)
        D, I = index.search(pts, k + 1)
    I = I[:, 1:]
    D = D[:, 1:]
    D = np.sqrt(np.clip(D, 0.0, None))
    return (
        torch.from_numpy(I).long().to(points.device),
        torch.from_numpy(D.astype(np.float32)).to(points.device),
    )


def compute_knn_graph(points, k_max, chunk_size=4096, backend="cdist"):
    """Return KNN indices and Euclidean distances, including self-neighbors. Fall back to chunked cdist when FAISS is unavailable."""
    if backend == "faiss":
        return _knn_faiss(points, k_max)
    return _knn_cdist(points, k_max, chunk_size)
