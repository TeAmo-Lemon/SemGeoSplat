#
# FAISS-backed replacement for simple_knn._C.distCUDA2.
#

import numpy as np
import torch

import faiss


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """每个点到其 3 个最近邻的平均平方 L2 距离 (GPU 加速).

    用作 ``simple_knn._C.distCUDA2`` 的替代品: ``IndexFlatL2`` 返回的就是
    平方 L2 距离, 与原 CUDA kernel 的语义一致 (平均 3 个最近非自身邻居的
    平方距离).

    依赖 ``faiss-gpu-cu12`` (>= 1.9). 旧的 ``faiss-gpu`` 1.7.x (pypi) 因为
    链接的是 CUDA 11 cuBLAS, 在 CUDA 12 运行时上会以 C++ assert 形式崩溃
    (Aborted, core dumped), 必须升级到 cuVS/RAFT 路径的新版.
    """
    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    n, d = pts.shape
    k = min(4, n)  # 自身 + 最多 3 个最近邻

    def _build(on_gpu: bool):
        idx = faiss.IndexFlatL2(d)
        res = None
        if on_gpu:
            res = faiss.StandardGpuResources()
            # 关闭 ~1.5GB 的临时显存预分配池, 避免和训练显存撞车;
            # search 内部仍会按需 cudaMalloc 小块工作区.
            res.noTempMemory()
            idx = faiss.index_cpu_to_gpu(res, 0, idx)
        idx.add(pts)
        # 必须把 res 一起返回, 让调用方保住引用直到 search 结束;
        # GpuIndex 用 SWIG 包装, 不允许把 res 作为属性挂到 idx 上.
        return idx, res

    use_gpu = (
        hasattr(faiss, "StandardGpuResources")
        and torch.cuda.is_available()
    )
    try:
        index, _res = _build(on_gpu=use_gpu)
        D, _ = index.search(pts, k)
    except RuntimeError as e:
        # 新版 faiss 把 GPU 端错误抛为 Python RuntimeError (而不是 abort),
        # 这里就能稳稳兜住. 典型场景: 训练时显存被占满, faiss 拿不到工作区.
        print(f"[distCUDA2] faiss GPU 路径失败, 回退到 CPU: {str(e)[:120]}")
        index, _res = _build(on_gpu=False)
        D, _ = index.search(pts, k)

    # IndexFlatL2 返回升序排列的平方 L2 距离, 第 0 列是查询点自身 (距离 0).
    if k > 1:
        mean_d2 = D[:, 1:k].mean(axis=1)
    else:
        mean_d2 = np.zeros(n, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(mean_d2)).float().to(points.device)
