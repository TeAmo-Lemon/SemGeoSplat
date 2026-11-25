"""④ 从高斯到结构感知区域与度量空间 (架构书 ④).

在 ②③ 之后, ⑤ FGW 之前; 内存中对已蒸馏的固定高斯点集继续操作.

概念主线 (架构书 4.1~4.4):
  高斯点 -> [4.1 增强特征(含 scale)] -> [4.2 加权图 + 过分割 + RAG 迭代层次合并]
         -> 结构感知区域 {R_j} -> [4.3 度量化] -> 度量测度空间 {O_j = (X_j, d_Xj, μ_j, F^Xj)}

两阶段聚类 (替代单阶段 BFS 区域生长):
  Step1 Over-segmentation: 几何+语义加权 KNN 图 -> Felzenszwalb 图割(边界切割)
        产生 super-region. 边权 W = w_d·(d/r) + w_f·(1-cos dino)
        + w_n·(1-|n·n|) + w_s·|Δlog s| (4 项归一到 [0,1] 再加权).
        DINO 默认参与过分割 (geo_only=False) 使初始 super-region 语义连贯.
        Felzenszwalb 按边权升序 + 自适应门 MInt 切割 -> 梯度大的边自动断 (boundary cutting).
  Step2 Region Merge: 区域特征聚合 -> Region Adjacency Graph (RAG) ->
        堆驱动迭代凝聚层次合并 (最多 3 轮, 每轮重建堆; 无合并时提前收敛).
        仅相邻区域对入堆 -> 不相邻的相似物体(桌子 A/B)永不合并 (根除 graph leakage).
        合并代价中法线相异度 cap 0.5 -> 垂直邻接面(椅面+椅腿)不再被硬阻断.
        cost < merge_max_cost (默认 0.60, 固定阈值) 才合并.

关键: 4.3(度量空间) 与 4.4(标签传播+sidecar) 不变 -> ⑤ FGW 契约(metric_spaces/
cluster_labels/regions.pt)完全不变; 改的只是"区域怎么形成".
"""

import os
import heapq
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import scipy.sparse as sp_sparse
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
import torch
import torch.nn.functional as F
from dataclasses import dataclass

from .geometry import _compute_normals, compute_knn_graph
from .dino_distill import _pca_fingerprint, _resolve_dino_paths
from .paths import sidecar_path
from .region_boundary import (
    _fingerprint, _rebuild_spaces, _xyz, load_boundary_features,
    refine_live_partition, region_boundary_selection,
)


# 工程内部常数 (架构书 4.4; 非 CLI 可调, 调 OOM 时改这里)
_REGION_K_MAX = 32          # KNN 候选近邻数 = max(knn_k, 32)
_REGION_REP_KNN_K = 8       # 代表点图 KNN (架构书 4.3, k=5~10)
_REGION_CDIST_CHUNK = 4096  # chunked cdist 行块大小 (~3.2GB transient @ M=200k)
_SCALE_NORM = 5.0           # log-scale 差归一上限 (|Δlog s|/此值 clamp 到 [0,1])


@dataclass
class MetricMeasureSpace:
    """O_j = (X_j, d_Xj, μ_j, F^Xj) - 架构书 4.3, 每个区域一个. 全部 CPU float32;
    ⑤ 按需上 GPU. distances 是 RAW 测地距离 (⑤ 用 diameter 归一化到 [0,1]).
    features 是 RAW (opacity 加权 DINO 均值, 未归一化; ⑤ 统一归一化以平衡特征/结构)."""
    points: torch.Tensor        # [m,3]  代表点 xyz (世界坐标)
    distances: torch.Tensor     # [m,m]  RAW 测地距离 d_X (inf 已替换为 2×diameter)
    measure: torch.Tensor       # [m]    概率测度 μ, sum=1
    features: torch.Tensor      # [m,16] F^X, 单元内不透明度加权 DINO 均值 (RAW)
    diameter: float             # max-finite d_X (⑤ 归一化 d_X/diameter->[0,1])
    feature_dim: int            # 16 (⑤ 校验与风格侧 PCA 投影一致)
    n_source_points: int        # |R_j| 区域内高斯点数
    region_id: int
    source_opacity_sum: float   # 区域总 Σσ(α) (⑤ 区域加权用)


# ── 4.1 语义特征增强 (含 scale) ────────────────────────────────────────────────

def _build_geometry_features(xyz, normals):
    """4.1 纯几何 Z-score 分量 g_z = Z-score([xyz, normals]) [N,6], 供 KNN 图构建.
    KNN 在纯几何上 -> 图拓扑反映空间结构 (d_X 结构代价不被语义污染). 语义/scale/法线
    只进入 4.2 边权 (Felzenszwalb 边界切割), 不改图拓扑."""
    g = torch.cat([xyz, normals], dim=-1)                            # [N,6]
    g_z = (g - g.mean(0, keepdim=True)) / g.std(0, keepdim=True).clamp_min(1e-6)
    return g_z


def _log_scale(scaling):
    """每点 log 尺度 = mean(log(s_x), log(s_y), log(s_z)) = log(几何均值).
    供 4.2 边权 scale 项 (天空 scale 巨大 / 树叶 scale 很小, DINO 同质时帮助区分)."""
    return torch.log(scaling.clamp_min(1e-8)).mean(dim=-1)            # [N]


def _select_valid_points(gaussians, max_points, opacity_thresh):
    """4.4 有效点筛选: σ(α) ≥ 阈值, 超过 max_points 按 opacity 取 top-k.
    返回子空间张量 (cuda) + idx_valid (cpu long) + log_scale."""
    opacity = gaussians.get_opacity.detach().reshape(-1)             # [N]
    valid = opacity >= opacity_thresh
    idx_valid = torch.where(valid)[0]
    if idx_valid.numel() > max_points:
        topk = torch.topk(opacity[idx_valid], max_points, largest=True).indices
        idx_valid = idx_valid[topk]
    idx_valid_cpu = idx_valid.cpu()
    xyz = gaussians.get_xyz.detach()[idx_valid]
    scaling = gaussians.get_scaling.detach()[idx_valid]
    rotation = gaussians.get_rotation.detach()[idx_valid]
    dino = gaussians.get_dino_features.detach()[idx_valid]
    op = opacity[idx_valid]
    normals = _compute_normals(scaling, rotation)
    g_z = _build_geometry_features(xyz, normals)
    log_scale = _log_scale(scaling)
    return idx_valid_cpu, xyz, normals, dino, op, g_z, log_scale


# ── 4.2 加权图 + 过分割(Felzenszwalb) + RAG 层次合并 ─────────────────────────────

def _density_radius(knn_dist, k):
    """4.4 局部尺度 r_i = median(前 k 近邻距离). 比"第 k 近邻×α"更稳 (用户指定
    median(k=10)); 稀疏区半径随密度自适应放大, 边权用 d/max(r_i,r_j) 归一."""
    k = min(int(k), knn_dist.shape[1])
    if k < 1:
        k = 1
    r = torch.median(knn_dist[:, :k], dim=1).values                  # [M]
    return r.clamp_min(1e-6)


def _compute_edge_weights(knn_idx, knn_dist, r_i, dino, normals, log_scale,
                          w_dist, w_feat, w_normal, w_scale, eps=1e-8,
                          geo_only=False):
    """向量化(GPU)计算每条 KNN 边的合并代价 = Felzenszwalb 边权 W (越大越该切).
    4 项 dissimilarity, 各按自身 95 分位归一到 [0,1] (放大边界对比; 不归一会让 W
    退化成均匀值 -> Felzenszwalb 贪心吃成 1 个巨簇, 复现"外围巨簇"问题):
      d_rel   = dist_ij / max(r_i, r_j)        密度归一距离 (稀疏远边 -> 大 -> 切)
      feat_d  = 0.5*(1 - cos(dino_i, dino_j))  DINO 角距离
      norm_d  = 1 - |n_i·n_j|                   法线相异 (表面棱边处大)
      scale_d = |Δlog s| / _SCALE_NORM          scale 相异 (天空 vs 物体)
    W = w_d·d_rel_n + w_f·feat_d_n + w_n·norm_d_n + w_s·scale_d_n  (≥0)
    geo_only=True: 强制 w_feat=0, 使过分割仅由几何项(距离+法线+scale)决定,
      最终合并代价仍包含 DINO (两阶段解耦, v5 优先级 1).
    返回 (src, dst, W) numpy int64/int64/float32, 去序去重的无向边."""
    if geo_only:
        w_feat = 0.0
    M, k = knn_idx.shape
    idx_safe = knn_idx.clamp(min=0)
    valid = knn_idx >= 0
    r_j = r_i[idx_safe]                                              # [M,k]
    r_max = torch.maximum(r_i.unsqueeze(1), r_j)                    # [M,k] (max 对称)
    d_rel = knn_dist / r_max.clamp_min(eps)
    dino_n = F.normalize(dino, dim=-1)                               # [M,D]
    nbr_dino = dino_n[idx_safe]                                      # [M,k,D]
    cos_d = torch.einsum("nkd,nd->nk", nbr_dino, dino_n)             # [M,k]
    feat_d = 0.5 * (1.0 - cos_d)
    del nbr_dino
    nbr_n = normals[idx_safe]                                        # [M,k,3]
    abs_nn = torch.einsum("nkd,nd->nk", nbr_n, normals).abs()        # [M,k]
    norm_d = 1.0 - abs_nn
    del nbr_n
    nbr_s = log_scale[idx_safe]                                      # [M,k]
    scale_d = (nbr_s - log_scale.unsqueeze(1)).abs() / _SCALE_NORM

    def _norm01(t):
        v = t[valid]
        if v.numel() == 0:
            return torch.zeros_like(t)
        p = torch.quantile(v.float(), 0.95).clamp_min(1e-6)
        return (t / p).clamp(0.0, 1.0)

    d_rel = _norm01(d_rel)
    feat_d = _norm01(feat_d)
    norm_d = _norm01(norm_d)
    scale_d = _norm01(scale_d)
    W = w_dist * d_rel + w_feat * feat_d + w_normal * norm_d + w_scale * scale_d
    ar = torch.arange(M, device=knn_idx.device).unsqueeze(-1).expand(-1, k)
    src = ar[valid].cpu().numpy().astype(np.int64)
    dst = idx_safe[valid].cpu().numpy().astype(np.int64)
    wflat = W[valid].cpu().numpy().astype(np.float32)
    if src.size == 0:
        return src, dst, wflat
    # 去序去重无向边: W 对称 (d_rel 用 max(r_i,r_j), 其余天然对称), 取首现即可
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    keys = lo.astype(np.int64) * np.int64(M) + hi.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    lo, hi, wflat = lo[order], hi[order], wflat[order]
    kk = keys[order]
    uniq = np.ones(lo.shape[0], dtype=bool)
    uniq[1:] = kk[1:] != kk[:-1]
    keep = uniq & (lo != hi)
    return lo[keep], hi[keep], wflat[keep]

def _uf_find(parent, x):
    """union-find find (路径压缩). parent 为 numpy int64 数组."""
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        parent[x], x = r, parent[x]
    return r


def _felzenszwalb_oversegment(src, dst, W, n_nodes, scale_k, max_size):
    """Felzenszwalb-Huttenlocher 图分割 (Boundary Cutting, 架构书 4.2 过分割).

    等价性: 按边权升序处理时, 两端在不同分量的边恰为最小生成森林的边 (Kruskal 性质),
    非 MST 边两端必属同一分量 -> 永不触发合并. 故只需在最小生成森林(MSF)上应用自适应门,
    结果与全图等价但只走 n-1 条边 (200k 点秒级).

    门: merge(a,b) iff w <= min(Int(Ca)+k/|Ca|, Int(Cb)+k/|Cb|),
        Int(C) = C 的最重 MST 边 (升序处理时 = 当前合并边权 w).
    k 大 -> 阈值高 -> 更多合并 -> 更大 super-region.
    max_size: 组件达到此规模后停止吸收邻居 (强制过分割, 打破巨簇).

    返回 labels[n_nodes] (0..R-1) + rag_pairs[E,2] (相邻区域对, 全图边去序去重)."""
    if src.size == 0:
        return np.zeros(n_nodes, dtype=np.int64), np.empty((0, 2), dtype=np.int64)
    # 最小生成森林 (scipy C): 无向图作对称邻接
    sym_src = np.concatenate([src, dst])
    sym_dst = np.concatenate([dst, src])
    sym_w = np.concatenate([W, W]).astype(np.float64)
    adj = sp_sparse.coo_matrix(
        (sym_w, (sym_src, sym_dst)), shape=(n_nodes, n_nodes)).tocsr()
    mst = minimum_spanning_tree(adj).tocoo()
    parent = np.arange(n_nodes, dtype=np.int64)
    size = np.ones(n_nodes, dtype=np.int64)
    int_diff = np.zeros(n_nodes, dtype=np.float64)
    if mst.data.size > 0:
        ew = mst.data.astype(np.float64)
        order = np.argsort(ew, kind="stable")
        es = mst.row[order].tolist()
        ed = mst.col[order].tolist()
        ewl = ew[order].tolist()
        for a, b, w in zip(es, ed, ewl):
            ra = _uf_find(parent, a)
            rb = _uf_find(parent, b)
            if ra == rb:
                continue
            # size cap: 任一组件已达上限则不再吸收 (强制过分割, 打破巨簇)
            if size[ra] >= max_size or size[rb] >= max_size:
                continue
            m = min(int_diff[ra] + scale_k / size[ra],
                    int_diff[rb] + scale_k / size[rb])
            if w <= m:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                int_diff[ra] = w            # w 升序 -> 当前分量最重边
    labels = np.array([_uf_find(parent, i) for i in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    # RAG 相邻对: 全图边中两端标签不同的 (边界), 去序去重
    ls = labels[src]
    ld = labels[dst]
    cross = ls != ld
    if cross.any():
        lo = np.minimum(ls[cross], ld[cross]).astype(np.int64)
        hi = np.maximum(ls[cross], ld[cross]).astype(np.int64)
        rag_pairs = np.unique(np.stack([lo, hi], 1), axis=0)
    else:
        rag_pairs = np.empty((0, 2), dtype=np.int64)
    return labels, rag_pairs


def _aggregate_region_features(labels, xyz, dino, op, normals, log_scale):
    """每区域聚合特征 (供 RAG 合并): feat_mean(σ 加权 DINO)、centroid(σ 加权 xyz)、
    normal_mean(σ 加权法线再归一)、scale_mean(σ 加权 log_scale)、op_sum、size.
    全 GPU scatter_add; 返回 numpy float64 数组 (CPU) 供 _hierarchical_merge 用."""
    R = int(labels.max()) + 1
    dev = xyz.device
    lab = torch.from_numpy(labels).long().to(dev)
    op_w = op.clamp_min(1e-8)
    op_sum = torch.zeros(R, device=dev).scatter_add_(0, lab, op)       # [R]
    den = op_sum.clamp_min(1e-8)
    D = dino.shape[1]
    feat_num = torch.zeros(R, D, device=dev)
    feat_num.scatter_add_(0, lab.unsqueeze(1).expand(-1, D), op_w.unsqueeze(1) * dino)
    feat_mean = feat_num / den.unsqueeze(1)
    cen_num = torch.zeros(R, 3, device=dev)
    cen_num.scatter_add_(0, lab.unsqueeze(1).expand(-1, 3), op_w.unsqueeze(1) * xyz)
    centroid = cen_num / den.unsqueeze(1)
    nn_num = torch.zeros(R, 3, device=dev)
    nn_num.scatter_add_(0, lab.unsqueeze(1).expand(-1, 3), op_w.unsqueeze(1) * normals)
    normal_mean = F.normalize(nn_num / den.unsqueeze(1), dim=-1)
    sc_num = torch.zeros(R, device=dev).scatter_add_(0, lab, op_w * log_scale)
    scale_mean = sc_num / den
    size = torch.bincount(lab, minlength=R).to(torch.float64)
    return {
        "feat_mean": feat_mean.detach().cpu().numpy().astype(np.float64),
        "centroid": centroid.detach().cpu().numpy().astype(np.float64),
        "normal_mean": normal_mean.detach().cpu().numpy().astype(np.float64),
        "scale_mean": scale_mean.detach().cpu().numpy().astype(np.float64),
        "op_sum": op_sum.detach().cpu().numpy().astype(np.float64),
        "size": size.detach().cpu().numpy().astype(np.float64),
        "n_regions": R,
    }


def _hierarchical_merge(rag_pairs, agg, r_global, weights, merge_max_cost,
                         min_cluster_size, adaptive_beta=0.0):
    """RAG 相邻约束凝聚层次合并 (堆驱动, 平均链接, 架构书 4.2 Step2).

    仅相邻区域对 (RAG 边) 入堆 -> 不相邻的相似物体(桌子 A/B)永不合并 (根除 graph
    leakage). cost < merge_max_cost 才合并 (cost 复用边权 3 项: DINO 语义 + 法线 + scale,
    不含质心距离, 同组权重); 之后 < min_cluster_size 的残片强制并入其最廉相邻区域
    (保持空间邻接, 非质心 -> 不致外围稀疏区聚成巨簇).

    adaptive_beta > 0 (v5): 自适应合并阈值 effective_mm = mm * (size_ratio)^(-beta),
      size_ratio = (sz_i+sz_j)/(2*initial_mean_size). 大区域更难合并 (防雪崩),
      小区域更易合并 (减碎片), 使 mm 从粗粒度开关变为平滑旋钮.
      beta=0 回退到 v4 全局固定阈值.

    性能 (v5.1): 大规模 super-region (M=200k -> ~12k) 时原来每个合并都把存活 region
    的全部邻居重新 push, 让 heap 累积上千万条过期记录 (每对会被 cost() 重复计算多次,
    pop 后才发现失效). 现在 ① 每轮内每对邻居只入堆一次 (pushed 集合追踪, pop 时仍重算
    cost, 因合并后特征更新, 语义不变); ② cost() 改为预归一化的 fmean_n (只在 do_merge
    时更新一次归一化, 不再每次调用 np.linalg.norm). 实测 merge 阶段 ~10min -> ~5s.
    语义与旧版本逐 pass 完全一致 (合并次序 = 每轮按当前 cost 升序贪心).

    返回 old_region -> new_root 映射 [R] (未压缩; 调用方再
    做 _merge_residual_small 兜底 + 重编号)."""
    R = agg["n_regions"]
    if R == 0:
        return np.zeros(0, dtype=np.int64)
    w_d, w_feat, w_normal, w_scale = weights
    Rg = max(float(r_global), 1e-8)
    parent = np.arange(R, dtype=np.int64)
    op_ = agg["op_sum"].copy()
    sz = agg["size"].copy()
    fmean = agg["feat_mean"].copy()
    cen = agg["centroid"].copy()
    nmean = agg["normal_mean"].copy()
    smean = agg["scale_mean"].copy()
    # 预归一化 fmean (cost 里反复用的分母; 只在 do_merge 更新一次, 不再是每次调用 norm)
    fmean_n = fmean / (np.linalg.norm(fmean, axis=1, keepdims=True) + 1e-12)
    # v5: 以初始 super-region 平均大小为基准, 用于自适应阈值
    initial_mean_size = float(sz.mean()) if adaptive_beta > 0 else 0.0
    adj = [set() for _ in range(R)]
    for a, b in rag_pairs.tolist():
        adj[int(a)].add(int(b))
        adj[int(b)].add(int(a))

    def find(x):
        r = x
        while parent[r] != r:
            r = parent[r]
        while parent[x] != r:
            parent[x], x = r, parent[x]
        return r

    # 归一化后的代表向量 (归一化对 argsort 次序无影响, fmean_n 已含归一)
    def cost(i, j):
        # 合并代价 = 特征相似 + 几何相似 (法线 + scale); 邻接由 RAG 保证, 不含质心距离
        # (质心距离会把同表面相邻碎片误判为该切, 与"先过分割再合并同物体"相悖).
        fd = 0.5 * (1.0 - float(np.dot(fmean_n[i], fmean_n[j])))
        nd = min(1.0 - abs(float(np.dot(nmean[i], nmean[j]))), 0.5)
        sd = min(abs(float(smean[i] - smean[j])) / _SCALE_NORM, 1.0)
        return w_feat * fd + w_normal * nd + w_scale * sd

    def effective_mm(ri, rj):
        """v5 自适应合并阈值: 大区域对 -> 阈值收紧 (更难合并), 小区域对 -> 阈值放宽."""
        if adaptive_beta <= 0 or initial_mean_size <= 0:
            return merge_max_cost
        size_ratio = (sz[ri] + sz[rj]) / (2.0 * initial_mean_size)
        factor = size_ratio ** (-adaptive_beta)
        return merge_max_cost * factor

    def do_merge(ri, rj):
        # rj 并入 ri (ri 存活); op 加权平均链接
        no = op_[ri] + op_[rj]
        fmean[ri] = (op_[ri] * fmean[ri] + op_[rj] * fmean[rj]) / max(no, 1e-12)
        fmean_n[ri] = fmean[ri] / (np.linalg.norm(fmean[ri]) + 1e-12)
        cen[ri] = (op_[ri] * cen[ri] + op_[rj] * cen[rj]) / max(no, 1e-12)
        nv = op_[ri] * nmean[ri] + op_[rj] * nmean[rj]
        nmean[ri] = nv / (np.linalg.norm(nv) + 1e-12)
        smean[ri] = (op_[ri] * smean[ri] + op_[rj] * smean[rj]) / max(no, 1e-12)
        sz[ri] += sz[rj]
        op_[ri] = no
        parent[rj] = ri
        for nb in list(adj[rj]):
            if nb == ri:
                continue
            adj[nb].discard(rj)
            adj[nb].add(ri)
            adj[ri].add(nb)
        adj[rj].clear()
        adj[ri].discard(ri)
        adj[ri] = {find(x) for x in adj[ri] if find(x) != ri}   # 归一到根

    # Phase 1: cost-gated 合并 (iterative: 多轮重建堆, 给被跳过的 pair 第二次机会.
    # 单轮中 cost>=threshold 的 pair 被丢弃后永不重试; 但相邻 region 合并后特征更新,
    # 下一轮重建堆时 cost 可能已降低 -> 避免碎片永久滞留.)
    # 性能: 每轮用 pushed 集合保证每个 (r, nb) 邻居对只在堆里出现一次 —— pop 时
    # cost 重新计算 (合并后特征已更新), 语义不变; heap 不再累积千万条过期记录.
    for _pass in range(3):
        heap = []
        pushed = set()
        for r in range(R):
            if parent[r] != r:
                continue
            for nb in adj[r]:
                if r < nb:                     # 每条无向边只入堆一次
                    heapq.heappush(heap, (cost(r, nb), r, nb))
                    pushed.add((r, nb))
        if not heap:
            break
        merged = False
        while heap:
            _, i, j = heapq.heappop(heap)
            pushed.discard((i, j))
            ri, rj = find(i), find(j)
            if ri == rj or rj not in adj[ri]:
                continue
            tc = cost(ri, rj)
            if tc >= effective_mm(ri, rj):
                continue
            if op_[ri] < op_[rj]:          # op 大的存活 (更稳聚合)
                ri, rj = rj, ri
            do_merge(ri, rj)
            merged = True
            for nb in adj[ri]:
                key = (nb, ri) if nb < ri else (ri, nb)
                if key in pushed:
                    continue
                heapq.heappush(heap, (cost(ri, nb), key[0], key[1]))
                pushed.add(key)
        if not merged:                         # 收敛: 本轮无任何合并, 提前退出
            break
    # Phase 2: < min_cluster_size 残片强制并入最廉相邻区域 (保持邻接, 非质心)
    while True:
        small = [r for r in range(R) if parent[r] == r and sz[r] < min_cluster_size]
        if not small:
            break
        progressed = False
        for r in small:
            if parent[r] != r:
                continue
            nbrs = sorted({find(nb) for nb in adj[r] if find(nb) != r})
            if not nbrs:
                continue                       # 无邻接, 留给 _merge_residual_small
            best = min(nbrs, key=lambda nb: cost(r, nb))
            do_merge(best, r)
            progressed = True
        if not progressed:
            break

    return np.array([find(r) for r in range(R)], dtype=np.int64)


def _merge_residual_small(labels, points, min_cluster_size):
    """残留小簇 (per-point 传播 + resplit 后仍 < min_cluster_size) -> 按质心归并到
    最近的大簇 (≥ min_cluster_size). 接受非连通 (⑤ 用 geodesic inf->2×diameter 处理).
    这是 spec 4.2 "CC 拆分" 之外的鲁棒补丁: 严格 CC 拆分会把孤立小簇保留为单点区域,
    对 ⑤ FGW 无意义且数量爆炸; 此步把它们并入最近大簇. 无大簇时保留原状."""
    if labels.size == 0 or (labels < 0).all():
        return labels
    # 压缩到连续 0..K-1 (root_map 可能有空洞 -> bincount 出 0 尺寸伪区域)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    sizes = np.bincount(labels[labels >= 0])
    large_ids = np.where(sizes >= min_cluster_size)[0]
    small_ids = np.where((sizes > 0) & (sizes < min_cluster_size))[0]
    if small_ids.size == 0 or large_ids.size == 0:
        return labels
    pts = points.cpu().numpy()
    large_centroids = np.stack([pts[labels == lid].mean(0) for lid in large_ids])  # [L,3]
    for sid in small_ids:
        members = np.where(labels == sid)[0]
        c = pts[members].mean(0)
        d = np.linalg.norm(large_centroids - c, axis=1)
        labels[members] = large_ids[int(np.argmin(d))]
    # 按大小降序重编号 0..R-1
    _, inv = np.unique(labels, return_inverse=True)
    sizes2 = np.bincount(inv)
    remap = np.empty(sizes2.size, dtype=np.int64)
    for new_id, old in enumerate(np.argsort(-sizes2)):
        remap[old] = new_id
    return remap[inv]


def _farthest_point_sampling(points, m, seed_idx=None):
    """贪心 max-min FPS, 纯 torch. seed_idx=None -> points[0] (调用方应传确定性种子).
    返回 rep_idx [min(m,n)] long on cuda."""
    n = points.shape[0]
    k = min(m, n)
    if k == n:
        return torch.arange(n, device=points.device)
    selected = torch.empty(k, dtype=torch.long, device=points.device)
    selected[0] = seed_idx if seed_idx is not None else 0
    min_dist = torch.norm(points - points[selected[0]], dim=1)     # [n]
    min_dist[selected[0]] = -1.0                                   # 已选点不再选
    for i in range(1, k):
        nxt = torch.argmax(min_dist)
        selected[i] = nxt
        min_dist = torch.minimum(min_dist, torch.norm(points - points[nxt], dim=1))
        min_dist[nxt] = -1.0
    return selected


def _build_metric_measure_space(region_xyz, region_dino, region_opacity,
                                m_reps, rep_knn_k, region_id):
    """4.3 单区域度量测度空间. FPS 代表点 -> rep KNN 图 -> scipy 最短路 d_X ->
    Voronoi 单元 μ (opacity-sum 归一化) -> F^X (opacity 加权 DINO 均值)."""
    n = region_xyz.shape[0]
    device = region_xyz.device
    seed = int(torch.argmax(region_opacity).item())                # 确定性种子: 最高 opacity 点
    rep_idx = _farthest_point_sampling(region_xyz, m_reps, seed_idx=seed)
    reps = region_xyz[rep_idx]                                     # [m,3]
    m = reps.shape[0]

    # d_Xj: 代表点 KNN 图上的最短路
    if m > 1:
        rep_d = torch.cdist(reps, reps)                            # [m,m]
        rep_d.fill_diagonal_(float("inf"))
        kk = min(rep_knn_k, m - 1)
        _, rep_nbr = torch.topk(rep_d, kk, dim=1, largest=False)   # [m,kk]
        rrow = torch.arange(m, device=device).unsqueeze(-1).expand(-1, kk).reshape(-1).cpu().numpy()
        ccol = rep_nbr.reshape(-1).cpu().numpy()
        w = torch.gather(rep_d, 1, rep_nbr).reshape(-1).cpu().numpy()
        adj = sp_sparse.coo_matrix((w, (rrow, ccol)), shape=(m, m)).tocsr()
        adj = adj.maximum(adj.T)
        dist = shortest_path(csgraph=adj, directed=False, method="auto")  # [m,m]
        finite = dist[np.isfinite(dist)]
        diameter = float(finite.max()) if finite.size > 0 else 0.0
        if not np.isfinite(dist).all():
            repl = 2.0 * diameter if diameter > 0 else 0.0
            dist = np.where(np.isfinite(dist), dist, repl)
    else:
        dist = np.zeros((1, 1), dtype=np.float32)
        diameter = 0.0

    # Voronoi: 区域点归到最近代表点 -> μ / F^X
    op = region_opacity
    D = region_dino.shape[1]
    if n > 0 and m > 0:
        assign = torch.cdist(region_xyz, reps).argmin(dim=1)       # [n]
        mu = torch.zeros(m, device=device)
        mu.scatter_add_(0, assign, op)
        feat_num = torch.zeros(m, D, device=device)
        op_w = op.unsqueeze(1) * region_dino                        # [n,D]
        feat_num.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), op_w)
        feat_den = mu.clone()
        mu = mu / mu.sum().clamp_min(1e-8)
        feat = feat_num / feat_den.clamp_min(1e-8).unsqueeze(1)
    else:
        mu = torch.zeros(m, device=device)
        feat = torch.zeros(m, D, device=device)

    return MetricMeasureSpace(
        points=reps.detach().cpu(),
        distances=torch.from_numpy(np.ascontiguousarray(dist, dtype=np.float32)),
        measure=mu.detach().cpu(),
        features=feat.detach().cpu(),
        diameter=max(float(diameter), 1e-8),  # 防 m==1/重合代表点: ⑤ 用 d_X/diameter 归一化, 0 会 NaN
        feature_dim=int(D),
        n_source_points=int(n),
        region_id=int(region_id),
        source_opacity_sum=float(op.sum().item()),
    )


# ── 标签传播到全部有效点 + sidecar ────────────────────────────────────────────────

def _propagate_labels_to_full(gaussians, idx_valid_np, labels_sub, opacity_thresh):
    """把 200k 子空间的区域标签传播到全部有效点 (σ(α)≥阈值): 未被子采样选中的有效点
    -> 最近子空间点的标签; σ(α)<阈值 -> -1. 用 faiss-gpu (回退 chunked cdist)."""
    n_full = gaussians.get_xyz.shape[0]
    full_labels = np.full(n_full, -1, dtype=np.int64)
    full_labels[idx_valid_np] = labels_sub
    opacity = gaussians.get_opacity.detach().reshape(-1).cpu().numpy()
    valid_mask = opacity >= opacity_thresh
    sub_set = np.zeros(n_full, dtype=bool)
    sub_set[idx_valid_np] = True
    need = np.where(valid_mask & ~sub_set)[0]
    if need.size == 0 or idx_valid_np.size == 0:
        return torch.from_numpy(full_labels)
    sub_xyz = gaussians.get_xyz.detach()[idx_valid_np].cpu().numpy().astype(np.float32)
    need_xyz = gaussians.get_xyz.detach()[need].cpu().numpy().astype(np.float32)
    try:
        import faiss
        index = faiss.IndexFlatL2(3)
        if torch.cuda.is_available() and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            index = faiss.index_cpu_to_gpu(res, 0, index)
        index.add(sub_xyz)
        _, nn = index.search(need_xyz, 1)
        full_labels[need] = labels_sub[nn[:, 0]]
    except Exception as exc:
        print(f"[region] faiss label-propagation failed ({exc}); falling back to cdist")
        sub_t = torch.from_numpy(sub_xyz).cuda()
        need_t = torch.from_numpy(need_xyz).cuda()
        chunk = 2048  # cdist [chunk, 200k] f32 ≈ 1.6GB/chunk (faiss 失败时的回退路径)
        for s in range(0, need_t.shape[0], chunk):
            e = min(s + chunk, need_t.shape[0])
            nn = torch.cdist(need_t[s:e], sub_t).argmin(dim=1).cpu().numpy()
            full_labels[need[s:e]] = labels_sub[nn]
    return torch.from_numpy(full_labels)


def _save_regions_sidecar(path, pca_state_path, metric_spaces, cluster_labels,
                          idx_valid_np, n_full, cfg, xyz_fingerprint, pca_fingerprint,
                          boundary_refinement=None):
    """落盘 regions.pt (与 ②③ sidecar 同目录: <model_path>/point_cloud/iteration_<style_iterations>/).
    ⑤ 读本文件 + pca_state_path 指向的 GT 缓存里的 pca_state, 把风格基元投到 16 维."""
    payload = {
        "version": 1,
        "feature_dim": int(metric_spaces[0].feature_dim) if metric_spaces else 0,
        "pca_state_path": pca_state_path,
        "pca_fingerprint": pca_fingerprint,
        "num_regions": len(metric_spaces),
        "metric_spaces": [
            {
                "points": ms.points,
                "distances": ms.distances,
                "measure": ms.measure,
                "features": ms.features,
                "diameter": ms.diameter,
                "feature_dim": ms.feature_dim,
                "n_source_points": ms.n_source_points,
                "region_id": ms.region_id,
                "source_opacity_sum": ms.source_opacity_sum,
            }
            for ms in metric_spaces
        ],
        "cluster_labels": cluster_labels,                          # LongTensor[N_full], -1=被滤掉
        "valid_indices": torch.from_numpy(idx_valid_np).long(),     # 200k 子空间在 N_full 中的下标
        "n_full": int(n_full),
        "xyz_fingerprint": xyz_fingerprint,
        "params": {
            "construction": "full",
            "opacity_threshold": cfg.region_opacity_threshold,
            "max_sample_points": cfg.region_max_sample_points,
            "density_k": cfg.region_k_base,
            "overseg_scale": cfg.region_overseg_scale,
            "w_dist": cfg.region_w_dist,
            "w_feat": cfg.region_w_feat,
            "w_normal": cfg.region_w_normal,
            "w_scale": cfg.region_w_scale,
            "merge_max_cost": cfg.region_merge_max_cost,
            "min_cluster_size": cfg.region_min_cluster_size,
            "m_reps": cfg.region_m_reps,
            "rep_knn_k": _REGION_REP_KNN_K,
            "overseg_geo_only": bool(int(getattr(cfg, "region_overseg_geo_only", 1))),
            "adaptive_merge_beta": float(getattr(cfg, "region_adaptive_merge_beta", 0.5)),
        },
    }
    if boundary_refinement is not None:
        payload["boundary_refinement"] = boundary_refinement
    _write_regions_sidecar(path, payload)


def _write_regions_sidecar(path, payload):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with NamedTemporaryFile(dir=directory, prefix="regions-", suffix=".tmp", delete=False) as stream:
        temporary = stream.name
        try:
            torch.save(payload, stream)
            stream.close()
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _load_regions_reference(dataset, gaussians, style, reference_path, n_full, selection=None):
    """从 Full sidecar 恢复④的完整输出，并在当前实验目录保存带来源的副本。"""
    try:
        payload = torch.load(reference_path, map_location="cpu", weights_only=False)
    except TypeError:  # 兼容较旧的 PyTorch
        payload = torch.load(reference_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"[region] invalid reference payload: {reference_path}")

    reference_n_full = int(payload.get("n_full", -1))
    if reference_n_full != int(n_full):
        raise ValueError(
            f"[region] reference n_full={reference_n_full} != current {n_full}: "
            f"{reference_path}")
    output_path = sidecar_path(dataset.model_path, style.style_iterations, "regions.pt")
    fingerprint = payload.get("xyz_fingerprint")
    if selection is not None:
        if Path(output_path).resolve() == Path(reference_path).resolve():
            raise ValueError("Use a separate output directory when refining a reference partition")
        if not fingerprint:
            from plyfile import PlyData
            sibling = Path(reference_path).with_name("point_cloud.ply")
            fingerprint = _fingerprint(_xyz(PlyData.read(sibling, mmap="r")["vertex"].data))
    if fingerprint and fingerprint != gaussians._xyz_fingerprint():
        raise ValueError(f"[region] reference point order or geometry has changed: {reference_path}")
    reference_dim = int(payload.get("feature_dim", -1))
    if reference_dim != int(style.dino_feature_dim):
        raise ValueError(
            f"[region] reference feature_dim={reference_dim} != current "
            f"{style.dino_feature_dim}: {reference_path}")

    raw_spaces = payload.get("metric_spaces")
    cluster_labels = payload.get("cluster_labels")
    if not isinstance(raw_spaces, list) or not raw_spaces:
        raise ValueError(f"[region] reference has no metric spaces: {reference_path}")
    if not torch.is_tensor(cluster_labels) or cluster_labels.numel() != n_full:
        raise ValueError(
            f"[region] reference cluster_labels shape mismatch: {reference_path}")

    cluster_labels = cluster_labels.long().cpu()
    valid_labels = cluster_labels[cluster_labels >= 0]
    if valid_labels.numel() and int(valid_labels.max()) >= len(raw_spaces):
        raise ValueError(f"[region] reference contains out-of-range labels: {reference_path}")

    output_payload = dict(payload)
    output_payload["reference_path"] = os.path.abspath(reference_path)
    output_payload["reference_mode"] = "exact_full_reuse"
    if selection is not None:
        refined, summary, data = refine_live_partition(
            gaussians, style, cluster_labels, payload["valid_indices"], selection,
            reference=os.path.abspath(reference_path))
        raw_spaces, affected = _rebuild_spaces(
            payload, refined.numpy(), *data, (refined != cluster_labels).numpy())
        cluster_labels = refined
        summary["rebuilt_metric_regions"] = affected
        output_payload.update(cluster_labels=refined, metric_spaces=raw_spaces,
                              xyz_fingerprint=fingerprint, reference_mode="boundary_refined",
                              boundary_refinement=summary)
        print(f"[region] ④ refined {summary['changed']:,} boundary labels in the reference partition", flush=True)

    metric_spaces = []
    for expected_id, item in enumerate(raw_spaces):
        if int(item.get("region_id", -1)) != expected_id:
            raise ValueError(
                f"[region] non-contiguous region_id at {expected_id}: {reference_path}")
        metric_spaces.append(MetricMeasureSpace(
            points=item["points"].float().cpu(),
            distances=item["distances"].float().cpu(),
            measure=item["measure"].float().cpu(),
            features=item["features"].float().cpu(),
            diameter=float(item["diameter"]),
            feature_dim=int(item["feature_dim"]),
            n_source_points=int(item["n_source_points"]),
            region_id=expected_id,
            source_opacity_sum=float(item["source_opacity_sum"]),
        ))
    _write_regions_sidecar(output_path, output_payload)
    gaussians.metric_spaces = metric_spaces
    gaussians.cluster_labels = cluster_labels
    gaussians.region_pca_fingerprint = payload.get("pca_fingerprint")
    print(f"[region] ④ {output_payload['reference_mode']}: {len(metric_spaces)} regions "
          f"from {reference_path} -> {output_path}", flush=True)


def build_structure_aware_regions(dataset, gaussians, style):
    """④ 主入口 (架构书 ④). 读 gaussians 几何 + _dino_features, 写:
      - gaussians.metric_spaces : list[MetricMeasureSpace] (CPU)  -> ⑤ FGW
      - gaussians.cluster_labels : LongTensor[N_full] (CPU, -1=被滤掉) -> ⑥ 风格生成
      - sidecar <model_path>/point_cloud/iteration_<style_iterations>/regions.pt
    失败时把两个属性置 None (⑤ 用 getattr 检测后跳过). 依赖 ② (不依赖 ③).
    两阶段: 4.2 Felzenszwalb 过分割 + RAG 层次合并; 4.3/4.4 不变 (⑤ 契约保留)."""
    n_full = gaussians.get_xyz.shape[0]
    reference_path = str(getattr(style, "regions_reference_path", "")).strip()
    selection = region_boundary_selection(style, reference=bool(reference_path))
    if getattr(style, "region_boundary_corrections", "") and (not reference_path or selection is None):
        raise ValueError("--region_boundary_corrections requires a reference partition and an explicit boundary selection")
    if reference_path:
        if not os.path.isfile(reference_path):
            raise FileNotFoundError(
                f"[region] --regions_reference_path not found: {reference_path}")
        _load_regions_reference(
            dataset, gaussians, style, reference_path, n_full, selection)
        return
    if gaussians.get_dino_features.numel() == 0 and (selection is not None or getattr(style, "region_preview", False)):
        features, source = load_boundary_features(gaussians, style)
        gaussians._dino_features = features.float().to(gaussians.get_xyz.device)
        print(f"[region] ④ loaded cached per-Gaussian DINO from {source}", flush=True)
    if gaussians.get_dino_features.numel() == 0:
        print("[region] ④ skipped: no _dino_features (② did not run).")
        gaussians.metric_spaces = None
        gaussians.cluster_labels = None
        return
    if n_full < 2:
        print("[region] ④ skipped: < 2 gaussians.")
        gaussians.metric_spaces = None
        gaussians.cluster_labels = None
        return

    print(f"[region] ④ start: {n_full:,} gaussians")

    # 4.1 有效点筛选 + 特征增强 (含 scale) -> g_z[N,6] + log_scale[N]
    idx_valid, xyz, normals, dino, op, g_z, log_scale = _select_valid_points(
        gaussians, style.region_max_sample_points, style.region_opacity_threshold)
    M = xyz.shape[0]
    print(f"[region] 4.1 valid points: {M:,} (opacity ≥ {style.region_opacity_threshold})")
    if M < 2:
        print("[region] ④ skipped: < 2 valid points after opacity filter.")
        gaussians.metric_spaces = None
        gaussians.cluster_labels = torch.full((n_full,), -1, dtype=torch.long)
        return

    geo_only = bool(int(getattr(style, "region_overseg_geo_only", 0)))
    k_max = min(_REGION_K_MAX, M)
    knn_idx, knn_dist = compute_knn_graph(
        g_z, k_max, _REGION_CDIST_CHUNK, backend="cdist")
    del g_z
    torch.cuda.empty_cache()
    r_i = _density_radius(knn_dist, style.region_k_base)
    r_global = float(r_i.median().item())
    print(f"[region] 4.2 KNN graph: k_max={k_max}, median density radius={r_global:.4f}")

    src, dst, W = _compute_edge_weights(
        knn_idx, knn_dist, r_i, dino, normals, log_scale,
        style.region_w_dist, style.region_w_feat,
        style.region_w_normal, style.region_w_scale,
        geo_only=geo_only)
    del knn_idx, knn_dist, r_i
    torch.cuda.empty_cache()
    print(f"[region] 4.2 weighted edges: {len(src):,} (undirected, deduped)")

    super_labels, rag_pairs = _felzenszwalb_oversegment(
        src, dst, W, M, float(style.region_overseg_scale),
        int(style.region_overseg_max_size))
    del src, dst, W
    n_super = int(super_labels.max()) + 1 if M > 0 else 0
    print(f"[region] 4.2 over-segmentation (Felzenszwalb k={style.region_overseg_scale}): "
          f"{n_super} super-regions, {len(rag_pairs)} RAG edges")
    agg = _aggregate_region_features(super_labels, xyz, dino, op, normals, log_scale)
    merge_weights = (
        style.region_w_dist, style.region_w_feat,
        style.region_w_normal, style.region_w_scale)
    root_map = _hierarchical_merge(
        rag_pairs, agg, r_global, merge_weights, float(style.region_merge_max_cost),
        int(style.region_min_cluster_size),
        adaptive_beta=float(getattr(style, "region_adaptive_merge_beta", 0.0)))
    del agg, rag_pairs, normals, log_scale
    labels = root_map[super_labels]
    del super_labels, root_map
    labels = _merge_residual_small(
        labels, xyz, style.region_min_cluster_size)

    n_final = int(labels.max()) + 1 if (labels >= 0).any() else 0
    sizes = np.bincount(labels[labels >= 0]) if n_final > 0 else np.zeros(0, dtype=np.int64)
    smax = int(sizes.max()) if sizes.size else 0
    smin = int(sizes.min()) if sizes.size else 0
    smean = float(sizes.mean()) if sizes.size else 0.0
    print(f"[region] 4.2 post-merge: {n_final} final regions "
          f"(sizes min={smin} max={smax} mean={smean:.0f})")

    idx_valid_np = idx_valid.numpy() if isinstance(idx_valid, torch.Tensor) else idx_valid
    cluster_labels, boundary_refinement = None, None
    if selection is not None and n_final > 0:
        initial = _propagate_labels_to_full(gaussians, idx_valid_np, labels, style.region_opacity_threshold)
        cluster_labels, boundary_refinement, _ = refine_live_partition(
            gaussians, style, initial, idx_valid_np, selection)
        labels = cluster_labels[idx_valid_np].numpy()
        print(f"[region] ④ refined {boundary_refinement['changed']:,} boundary labels before metric construction",
              flush=True)

    # 4.3 每区域度量测度空间 (单区域 try/except: 一个退化区域不拖垮整列)
    metric_spaces = []
    succ_old_ids = []                                  # 成功构建 metric space 的原 region_id
    for r in range(n_final):
        member = np.where(labels == r)[0]
        if member.size == 0:
            continue
        try:
            metric_spaces.append(_build_metric_measure_space(
                xyz[member], dino[member], op[member],
                style.region_m_reps, _REGION_REP_KNN_K, len(succ_old_ids)))
            succ_old_ids.append(r)
        except Exception as exc:
            if boundary_refinement is not None:
                raise RuntimeError(f"Could not build the refined metric space for region {r}") from exc
            print(f"[region] 4.3 region {r} ({member.size} pts) failed: {exc}; skipped.")
            continue
    print(f"[region] 4.3 built {len(metric_spaces)} metric measure spaces")
    if len(succ_old_ids) < n_final:
        remap = np.full(n_final, -1, dtype=np.int64)
        for new_id, old in enumerate(succ_old_ids):
            remap[old] = new_id
        labels = remap[labels]

    # 标签传播到全部有效点 (σ(α)≥阈值 但未进 200k 子采样的点 -> 最近子空间点)
    if cluster_labels is None:
        cluster_labels = _propagate_labels_to_full(
            gaussians, idx_valid_np, labels, style.region_opacity_threshold)

    # pca_state_path: 用 ② 解析的 GT 缓存路径 (尊重 --dino_features_path 覆盖), ⑤ 据此投影风格基元
    gt_path, _ = _resolve_dino_paths(dataset, style)
    pca_fingerprint = None
    if os.path.isfile(gt_path) or not getattr(style, "region_preview", False):
        cache = torch.load(gt_path, map_location="cpu", weights_only=True, mmap=True)
        pca_fingerprint = _pca_fingerprint(cache["pca_state"])
    sidecar = sidecar_path(
        dataset.model_path, style.style_iterations, "regions.pt")
    _save_regions_sidecar(sidecar, gt_path, metric_spaces, cluster_labels,
                          idx_valid_np, n_full, style, gaussians._xyz_fingerprint(), pca_fingerprint,
                          boundary_refinement)
    gaussians.metric_spaces = metric_spaces
    gaussians.cluster_labels = cluster_labels
    gaussians.region_pca_fingerprint = pca_fingerprint
    print(f"[region] ④ done: {len(metric_spaces)} regions -> {sidecar}")
    # 释放 ④ 中间张量 (metric_spaces/cluster_labels 已是 CPU)
    del xyz, dino, op, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
