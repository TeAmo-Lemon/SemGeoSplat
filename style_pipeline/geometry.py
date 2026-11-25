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
    R = build_rotation(rotation_norm)                                 # [N,3,3], 列=主轴
    min_idx = torch.argmin(scaling_active, dim=1)                     # [N]
    ar = torch.arange(R.shape[0], device=R.device)
    normals = R[ar, :, min_idx]                                       # [N,3]
    return F.normalize(normals, dim=-1)


def _knn_cdist(points, k_max, chunk_size):
    """用 PyTorch 内置 cdist 做 K 近邻搜索 (无需额外依赖).

    思路:
      cdist 是一次性计算所有点对距离的 "笨办法"——M 个点生成 M×M 距离矩阵,
      内存吃紧时按 chunk_size 分批, 减少瞬时显存。

    举例 (M=100000 个点, k=16, chunk_size=4096):
      不是一次性算 100000×100000 的矩阵 (~80 GB),
      而是每次算 4096×100000 (约 1.6 GB), 循环 ~25 次。

    返回: (indices, dists), shape 均为 [M, k], 自身到自身的距离已排除。
    """
    M = points.shape[0]
    k = min(k_max, M)
    all_dist = torch.full((M, k), float("inf"), device=points.device)
    all_idx = torch.full((M, k), -1, dtype=torch.long, device=points.device)
    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        d = torch.cdist(points[start:end], points)                  # [c, M], c = 当前 chunk 点数
        # 自己离自己的距离置为 inf, 确保不会选到自己
        rows = torch.arange(end - start, device=points.device)
        d[rows, start + rows] = float("inf")
        vals, idx = torch.topk(d, k, dim=1, largest=False)          # 每行取距离最小的 k 个
        all_dist[start:end] = vals
        all_idx[start:end] = idx
        del d
    return all_idx, all_dist


def _knn_faiss(points, k_max):
    """用 Facebook faiss 库做 K 近邻搜索 (GPU 加速, 速度远快于 cdist).

    faiss 专为向量检索优化, 底层 CUDA 实现比 cdist 暴力算快 10-100×。
    优先尝试 GPU 索引, 失败则自动回退 CPU。

    流程:
      1. 把 PyTorch tensor → numpy 数组 (faiss 只能处理 numpy)
      2. 建 IndexFlatL2 (L2 距离索引), 把所有点加进去
      3. search k+1 个最近邻 (多了 1 个是因为自己离自己最近, 要踢掉)
      4. 去掉第 1 列 (自身), 开根号 (faiss 返回平方距离)

    返回: (indices, dists), shape 均为 [M, k], 自身已排除。
    """
    import faiss
    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    n, d = pts.shape
    k = min(k_max, n)

    def _build(on_gpu):
        idx = faiss.IndexFlatL2(d)           # 暴力 L2 距离索引
        res = None
        if on_gpu:
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)  # 把索引搬到 GPU
        idx.add(pts)                         # 把所有点加入索引
        return idx, res
    use_gpu = hasattr(faiss, "StandardGpuResources") and torch.cuda.is_available()
    try:
        index, _res = _build(use_gpu)
        D, I = index.search(pts, k + 1)                             # +1 因含自身
    except RuntimeError:
        index, _res = _build(False)         # GPU 失败, 回退 CPU
        D, I = index.search(pts, k + 1)
    I = I[:, 1:]                            # 踢掉第 1 列 (自身)
    D = D[:, 1:]
    D = np.sqrt(np.clip(D, 0.0, None))      # 平方距离 → 欧氏距离
    return (torch.from_numpy(I).long().to(points.device),
            torch.from_numpy(D.astype(np.float32)).to(points.device))


def compute_knn_graph(points, k_max, chunk_size=4096, backend="cdist"):
    """为 M 个点建 K 近邻图: 每个点找离它最近的 k 个邻居。

    图是为后续的 "区域生长" (region growing) 做准备——每个点知道它的 Top-K
    邻居是谁以及距离多远, 就可以根据语义和几何相似度决定是否和邻居合并到同一区域。

    两个后端:
      cdist  (默认) : PyTorch 原生, 无依赖, 分块避免爆显存 → _knn_cdist
      faiss  (可选) : GPU 加速, 快但需装 faiss-gpu → _knn_faiss

    参数:
      points:  [M, 6] 点坐标 + 法线, 6D float tensor
      k_max:   每点保留 k 个最近邻
      backend: "cdist" 或 "faiss"

    返回: (indices[M, k], dists[M, k]), 自身已排除.
    """
    if backend == "faiss":
        return _knn_faiss(points, k_max)
    return _knn_cdist(points, k_max, chunk_size)
