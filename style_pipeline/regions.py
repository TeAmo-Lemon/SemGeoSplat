"""Construct semantic-geometric regions using graph segmentation and adjacency-constrained merging."""

import os
import heapq
import numpy as np
import scipy.sparse as sp_sparse
from scipy.sparse.csgraph import shortest_path, minimum_spanning_tree
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from .geometry import _compute_normals, compute_knn_graph
from .dino_distill import _resolve_dino_paths
from .paths import sidecar_path

_REGION_K_MAX = 32
_REGION_REP_KNN_K = 8
_REGION_CDIST_CHUNK = 4096
_SCALE_NORM = 5.0


@dataclass
class MetricMeasureSpace:
    """A region represented by CPU tensors: points, raw geodesic distances, opacity-weighted features and measure. FGW normalizes distances by the diameter."""

    points: torch.Tensor
    distances: torch.Tensor
    measure: torch.Tensor
    features: torch.Tensor
    diameter: float
    feature_dim: int
    n_source_points: int
    region_id: int
    source_opacity_sum: float


def _build_geometry_features(xyz, normals):
    """4.1 纯几何 Z-score 分量 g_z = Z-score([xyz, normals]) [N,6], 供 KNN 图构建.
    KNN 在纯几何上 -> 图拓扑反映空间结构 (d_X 结构代价不被语义污染). 语义/scale/法线
    只进入 4.2 边权 (Felzenszwalb 边界切割), 不改图拓扑."""
    g = torch.cat([xyz, normals], dim=-1)
    g_z = (g - g.mean(0, keepdim=True)) / g.std(0, keepdim=True).clamp_min(1e-06)
    return g_z


def _log_scale(scaling):
    """每点 log 尺度 = mean(log(s_x), log(s_y), log(s_z)) = log(几何均值).
    供 4.2 边权 scale 项 (天空 scale 巨大 / 树叶 scale 很小, DINO 同质时帮助区分)."""
    return torch.log(scaling.clamp_min(1e-08)).mean(dim=-1)


def _select_valid_points(gaussians, max_points, opacity_thresh):
    """4.4 有效点筛选: σ(α) ≥ 阈值, 超过 max_points 按 opacity 取 top-k.
    返回子空间张量 (cuda) + idx_valid (cpu long) + log_scale."""
    opacity = gaussians.get_opacity.detach().reshape(-1)
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
    return (idx_valid_cpu, xyz, normals, dino, op, g_z, log_scale)


def _density_radius(knn_dist, k):
    """Estimate the local density radius as the median distance to the first k neighbors."""
    k = min(int(k), knn_dist.shape[1])
    if k < 1:
        k = 1
    r = torch.median(knn_dist[:, :k], dim=1).values
    return r.clamp_min(1e-06)


def _compute_edge_weights(
    knn_idx,
    knn_dist,
    r_i,
    dino,
    normals,
    log_scale,
    w_dist,
    w_feat,
    w_normal,
    w_scale,
    eps=1e-08,
    geo_only=False,
):
    """Compute normalized semantic, geometric and scale costs for undirected KNN edges."""
    if geo_only:
        w_feat = 0.0
    M, k = knn_idx.shape
    idx_safe = knn_idx.clamp(min=0)
    valid = knn_idx >= 0
    r_j = r_i[idx_safe]
    r_max = torch.maximum(r_i.unsqueeze(1), r_j)
    d_rel = knn_dist / r_max.clamp_min(eps)
    dino_n = F.normalize(dino, dim=-1)
    nbr_dino = dino_n[idx_safe]
    cos_d = torch.einsum("nkd,nd->nk", nbr_dino, dino_n)
    feat_d = 0.5 * (1.0 - cos_d)
    del nbr_dino
    nbr_n = normals[idx_safe]
    abs_nn = torch.einsum("nkd,nd->nk", nbr_n, normals).abs()
    norm_d = 1.0 - abs_nn
    del nbr_n
    nbr_s = log_scale[idx_safe]
    scale_d = (nbr_s - log_scale.unsqueeze(1)).abs() / _SCALE_NORM

    def _norm01(t):
        v = t[valid]
        if v.numel() == 0:
            return torch.zeros_like(t)
        p = torch.quantile(v.float(), 0.95).clamp_min(1e-06)
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
        return (src, dst, wflat)
    lo = np.minimum(src, dst)
    hi = np.maximum(src, dst)
    keys = lo.astype(np.int64) * np.int64(M) + hi.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    lo, hi, wflat = (lo[order], hi[order], wflat[order])
    kk = keys[order]
    uniq = np.ones(lo.shape[0], dtype=bool)
    uniq[1:] = kk[1:] != kk[:-1]
    keep = uniq & (lo != hi)
    return (lo[keep], hi[keep], wflat[keep])


def _uf_find(parent, x):
    """union-find find (路径压缩). parent 为 numpy int64 数组."""
    r = x
    while parent[r] != r:
        r = parent[r]
    while parent[x] != r:
        parent[x], x = (r, parent[x])
    return r


def _felzenszwalb_oversegment(src, dst, W, n_nodes, scale_k, max_size):
    """Segment the weighted graph with Felzenszwalb-Huttenlocher merging and a component-size cap."""
    if src.size == 0:
        return (np.zeros(n_nodes, dtype=np.int64), np.empty((0, 2), dtype=np.int64))
    sym_src = np.concatenate([src, dst])
    sym_dst = np.concatenate([dst, src])
    sym_w = np.concatenate([W, W]).astype(np.float64)
    adj = sp_sparse.coo_matrix(
        (sym_w, (sym_src, sym_dst)), shape=(n_nodes, n_nodes)
    ).tocsr()
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
            if size[ra] >= max_size or size[rb] >= max_size:
                continue
            m = min(
                int_diff[ra] + scale_k / size[ra], int_diff[rb] + scale_k / size[rb]
            )
            if w <= m:
                if size[ra] < size[rb]:
                    ra, rb = (rb, ra)
                parent[rb] = ra
                size[ra] += size[rb]
                int_diff[ra] = w
    labels = np.array([_uf_find(parent, i) for i in range(n_nodes)], dtype=np.int64)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    ls = labels[src]
    ld = labels[dst]
    cross = ls != ld
    if cross.any():
        lo = np.minimum(ls[cross], ld[cross]).astype(np.int64)
        hi = np.maximum(ls[cross], ld[cross]).astype(np.int64)
        rag_pairs = np.unique(np.stack([lo, hi], 1), axis=0)
    else:
        rag_pairs = np.empty((0, 2), dtype=np.int64)
    return (labels, rag_pairs)


def _aggregate_region_features(labels, xyz, dino, op, normals, log_scale):
    """每区域聚合特征 (供 RAG 合并): feat_mean(σ 加权 DINO)、centroid(σ 加权 xyz)、
    normal_mean(σ 加权法线再归一)、scale_mean(σ 加权 log_scale)、op_sum、size.
    全 GPU scatter_add; 返回 numpy float64 数组 (CPU) 供 _hierarchical_merge 用."""
    R = int(labels.max()) + 1
    dev = xyz.device
    lab = torch.from_numpy(labels).long().to(dev)
    op_w = op.clamp_min(1e-08)
    op_sum = torch.zeros(R, device=dev).scatter_add_(0, lab, op)
    den = op_sum.clamp_min(1e-08)
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


def _hierarchical_merge(
    rag_pairs,
    agg,
    r_global,
    weights,
    merge_max_cost,
    min_cluster_size,
    adaptive_beta=0.0,
):
    """Merge adjacent regions using opacity-weighted statistics. Rebuild the heap after each pass to reconsider updated costs."""
    R = agg["n_regions"]
    if R == 0:
        return np.zeros(0, dtype=np.int64)
    w_d, w_feat, w_normal, w_scale = weights
    parent = np.arange(R, dtype=np.int64)
    op_ = agg["op_sum"].copy()
    sz = agg["size"].copy()
    fmean = agg["feat_mean"].copy()
    cen = agg["centroid"].copy()
    nmean = agg["normal_mean"].copy()
    smean = agg["scale_mean"].copy()
    fmean_n = fmean / (np.linalg.norm(fmean, axis=1, keepdims=True) + 1e-12)
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
            parent[x], x = (r, parent[x])
        return r

    def cost(i, j):
        fd = 0.5 * (1.0 - float(np.dot(fmean_n[i], fmean_n[j])))
        nd = min(1.0 - abs(float(np.dot(nmean[i], nmean[j]))), 0.5)
        sd = min(abs(float(smean[i] - smean[j])) / _SCALE_NORM, 1.0)
        return w_feat * fd + w_normal * nd + w_scale * sd

    def effective_mm(ri, rj):
        """Tighten the merge threshold for large region pairs when adaptive merging is enabled."""
        if adaptive_beta <= 0 or initial_mean_size <= 0:
            return merge_max_cost
        size_ratio = (sz[ri] + sz[rj]) / (2.0 * initial_mean_size)
        factor = size_ratio ** (-adaptive_beta)
        return merge_max_cost * factor

    def do_merge(ri, rj):
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
        adj[ri] = {find(x) for x in adj[ri] if find(x) != ri}

    for _pass in range(3):
        heap = []
        pushed = set()
        for r in range(R):
            if parent[r] != r:
                continue
            for nb in adj[r]:
                if r < nb:
                    heapq.heappush(heap, (cost(r, nb), r, nb))
                    pushed.add((r, nb))
        if not heap:
            break
        merged = False
        while heap:
            _, i, j = heapq.heappop(heap)
            pushed.discard((i, j))
            ri, rj = (find(i), find(j))
            if ri == rj or rj not in adj[ri]:
                continue
            tc = cost(ri, rj)
            if tc >= effective_mm(ri, rj):
                continue
            if op_[ri] < op_[rj]:
                ri, rj = (rj, ri)
            do_merge(ri, rj)
            merged = True
            for nb in adj[ri]:
                key = (nb, ri) if nb < ri else (ri, nb)
                if key in pushed:
                    continue
                heapq.heappush(heap, (cost(ri, nb), key[0], key[1]))
                pushed.add(key)
        if not merged:
            break
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
                continue
            best = min(nbrs, key=lambda nb: cost(r, nb))
            do_merge(best, r)
            progressed = True
        if not progressed:
            break
    return np.array([find(r) for r in range(R)], dtype=np.int64)


def _merge_residual_small(labels, points, min_cluster_size):
    """Merge residual small regions into the nearest sufficiently large region, then relabel by descending size."""
    if labels.size == 0 or (labels < 0).all():
        return labels
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.astype(np.int64)
    sizes = np.bincount(labels[labels >= 0])
    large_ids = np.where(sizes >= min_cluster_size)[0]
    small_ids = np.where((sizes > 0) & (sizes < min_cluster_size))[0]
    if small_ids.size == 0 or large_ids.size == 0:
        return labels
    pts = points.cpu().numpy()
    large_centroids = np.stack([pts[labels == lid].mean(0) for lid in large_ids])
    for sid in small_ids:
        members = np.where(labels == sid)[0]
        c = pts[members].mean(0)
        d = np.linalg.norm(large_centroids - c, axis=1)
        labels[members] = large_ids[int(np.argmin(d))]
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
    min_dist = torch.norm(points - points[selected[0]], dim=1)
    min_dist[selected[0]] = -1.0
    for i in range(1, k):
        nxt = torch.argmax(min_dist)
        selected[i] = nxt
        min_dist = torch.minimum(min_dist, torch.norm(points - points[nxt], dim=1))
        min_dist[nxt] = -1.0
    return selected


def _build_metric_measure_space(
    region_xyz, region_dino, region_opacity, m_reps, rep_knn_k, region_id
):
    """4.3 单区域度量测度空间. FPS 代表点 -> rep KNN 图 -> scipy 最短路 d_X ->
    Voronoi 单元 μ (opacity-sum 归一化) -> F^X (opacity 加权 DINO 均值)."""
    n = region_xyz.shape[0]
    device = region_xyz.device
    seed = int(torch.argmax(region_opacity).item())
    rep_idx = _farthest_point_sampling(region_xyz, m_reps, seed_idx=seed)
    reps = region_xyz[rep_idx]
    m = reps.shape[0]
    if m > 1:
        rep_d = torch.cdist(reps, reps)
        rep_d.fill_diagonal_(float("inf"))
        kk = min(rep_knn_k, m - 1)
        _, rep_nbr = torch.topk(rep_d, kk, dim=1, largest=False)
        rrow = (
            torch.arange(m, device=device)
            .unsqueeze(-1)
            .expand(-1, kk)
            .reshape(-1)
            .cpu()
            .numpy()
        )
        ccol = rep_nbr.reshape(-1).cpu().numpy()
        w = torch.gather(rep_d, 1, rep_nbr).reshape(-1).cpu().numpy()
        adj = sp_sparse.coo_matrix((w, (rrow, ccol)), shape=(m, m)).tocsr()
        adj = adj.maximum(adj.T)
        dist = shortest_path(csgraph=adj, directed=False, method="auto")
        finite = dist[np.isfinite(dist)]
        diameter = float(finite.max()) if finite.size > 0 else 0.0
        if not np.isfinite(dist).all():
            repl = 2.0 * diameter if diameter > 0 else 0.0
            dist = np.where(np.isfinite(dist), dist, repl)
    else:
        dist = np.zeros((1, 1), dtype=np.float32)
        diameter = 0.0
    op = region_opacity
    D = region_dino.shape[1]
    if n > 0 and m > 0:
        assign = torch.cdist(region_xyz, reps).argmin(dim=1)
        mu = torch.zeros(m, device=device)
        mu.scatter_add_(0, assign, op)
        feat_num = torch.zeros(m, D, device=device)
        op_w = op.unsqueeze(1) * region_dino
        feat_num.scatter_add_(0, assign.unsqueeze(1).expand(-1, D), op_w)
        feat_den = mu.clone()
        mu = mu / mu.sum().clamp_min(1e-08)
        feat = feat_num / feat_den.clamp_min(1e-08).unsqueeze(1)
    else:
        mu = torch.zeros(m, device=device)
        feat = torch.zeros(m, D, device=device)
    return MetricMeasureSpace(
        points=reps.detach().cpu(),
        distances=torch.from_numpy(np.ascontiguousarray(dist, dtype=np.float32)),
        measure=mu.detach().cpu(),
        features=feat.detach().cpu(),
        diameter=max(float(diameter), 1e-08),
        feature_dim=int(D),
        n_source_points=int(n),
        region_id=int(region_id),
        source_opacity_sum=float(op.sum().item()),
    )


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
        chunk = 2048
        for s in range(0, need_t.shape[0], chunk):
            e = min(s + chunk, need_t.shape[0])
            nn = torch.cdist(need_t[s:e], sub_t).argmin(dim=1).cpu().numpy()
            full_labels[need[s:e]] = labels_sub[nn]
    return torch.from_numpy(full_labels)


def _save_regions_sidecar(
    path, pca_state_path, metric_spaces, cluster_labels, idx_valid_np, n_full, cfg
):
    """落盘 regions.pt (与 ②③ sidecar 同目录: <model_path>/point_cloud/iteration_<style_iterations>/).
    ⑤ 读本文件 + pca_state_path 指向的 GT 缓存里的 pca_state, 把风格基元投到 16 维."""
    payload = {
        "version": 1,
        "feature_dim": int(metric_spaces[0].feature_dim) if metric_spaces else 0,
        "pca_state_path": pca_state_path,
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
        "cluster_labels": cluster_labels,
        "valid_indices": torch.from_numpy(idx_valid_np).long(),
        "n_full": int(n_full),
        "params": {
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
            "adaptive_merge_beta": float(
                getattr(cfg, "region_adaptive_merge_beta", 0.5)
            ),
        },
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)


def build_structure_aware_regions(dataset, gaussians, style):
    """Build semantic-geometric regions and metric spaces, propagate labels to all valid Gaussians, and save regions.pt."""
    n_full = gaussians.get_xyz.shape[0]
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
    print(f"[region] start: {n_full:,} gaussians")
    idx_valid, xyz, normals, dino, op, g_z, log_scale = _select_valid_points(
        gaussians, style.region_max_sample_points, style.region_opacity_threshold
    )
    M = xyz.shape[0]
    print(
        f"[region] 4.1 valid points: {M:,} (opacity ≥ {style.region_opacity_threshold})"
    )
    if M < 2:
        print("[region] ④ skipped: < 2 valid points after opacity filter.")
        gaussians.metric_spaces = None
        gaussians.cluster_labels = torch.full((n_full,), -1, dtype=torch.long)
        return
    effective_w_feat = style.region_w_feat
    geo_only = bool(int(getattr(style, "region_overseg_geo_only", 0)))
    k_max = min(_REGION_K_MAX, M)
    knn_idx, knn_dist = compute_knn_graph(
        g_z, k_max, _REGION_CDIST_CHUNK, backend="cdist"
    )
    del g_z
    torch.cuda.empty_cache()
    r_i = _density_radius(knn_dist, style.region_k_base)
    r_global = float(r_i.median().item())
    print(
        f"[region] 4.2 KNN graph: k_max={k_max}, median density radius={r_global:.4f}"
    )
    src, dst, W = _compute_edge_weights(
        knn_idx,
        knn_dist,
        r_i,
        dino,
        normals,
        log_scale,
        style.region_w_dist,
        effective_w_feat,
        style.region_w_normal,
        style.region_w_scale,
        geo_only=geo_only,
    )
    del knn_idx, knn_dist, r_i
    torch.cuda.empty_cache()
    print(f"[region] 4.2 weighted edges: {len(src):,} (undirected, deduped)")
    super_labels, rag_pairs = _felzenszwalb_oversegment(
        src,
        dst,
        W,
        M,
        float(style.region_overseg_scale),
        int(style.region_overseg_max_size),
    )
    del src, dst, W
    n_super = int(super_labels.max()) + 1 if M > 0 else 0
    print(
        f"[region] 4.2 over-segmentation (Felzenszwalb k={style.region_overseg_scale}): {n_super} super-regions, {len(rag_pairs)} RAG edges"
    )
    agg = _aggregate_region_features(super_labels, xyz, dino, op, normals, log_scale)
    merge_weights = (
        style.region_w_dist,
        effective_w_feat,
        style.region_w_normal,
        style.region_w_scale,
    )
    merge_kwargs = {
        "adaptive_beta": float(getattr(style, "region_adaptive_merge_beta", 0.0))
    }
    root_map = _hierarchical_merge(
        rag_pairs,
        agg,
        r_global,
        merge_weights,
        float(style.region_merge_max_cost),
        int(style.region_min_cluster_size),
        **merge_kwargs,
    )
    del agg, rag_pairs, normals, log_scale
    labels = root_map[super_labels]
    del super_labels, root_map
    labels = _merge_residual_small(labels, xyz, style.region_min_cluster_size)
    n_final = int(labels.max()) + 1 if (labels >= 0).any() else 0
    sizes = (
        np.bincount(labels[labels >= 0]) if n_final > 0 else np.zeros(0, dtype=np.int64)
    )
    smax = int(sizes.max()) if sizes.size else 0
    smin = int(sizes.min()) if sizes.size else 0
    smean = float(sizes.mean()) if sizes.size else 0.0
    print(
        f"[region] 4.2 post-merge: {n_final} final regions (sizes min={smin} max={smax} mean={smean:.0f})"
    )
    metric_spaces = []
    succ_old_ids = []
    for r in range(n_final):
        member = np.where(labels == r)[0]
        if member.size == 0:
            continue
        try:
            metric_spaces.append(
                _build_metric_measure_space(
                    xyz[member],
                    dino[member],
                    op[member],
                    style.region_m_reps,
                    _REGION_REP_KNN_K,
                    len(succ_old_ids),
                )
            )
            succ_old_ids.append(r)
        except Exception as exc:
            print(
                f"[region] 4.3 region {r} ({member.size} pts) failed: {exc}; skipped."
            )
            continue
    print(f"[region] 4.3 built {len(metric_spaces)} metric measure spaces")
    if len(succ_old_ids) < n_final:
        remap = np.full(n_final, -1, dtype=np.int64)
        for new_id, old in enumerate(succ_old_ids):
            remap[old] = new_id
        labels = remap[labels]
    idx_valid_np = (
        idx_valid.numpy() if isinstance(idx_valid, torch.Tensor) else idx_valid
    )
    cluster_labels = _propagate_labels_to_full(
        gaussians, idx_valid_np, labels, style.region_opacity_threshold
    )
    gt_path, _ = _resolve_dino_paths(dataset, style)
    sidecar = sidecar_path(dataset.model_path, style.style_iterations, "regions.pt")
    _save_regions_sidecar(
        sidecar, gt_path, metric_spaces, cluster_labels, idx_valid_np, n_full, style
    )
    gaussians.metric_spaces = metric_spaces
    gaussians.cluster_labels = cluster_labels
    print(f"[region] ④ done: {len(metric_spaces)} regions -> {sidecar}")
    del xyz, dino, op, labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
