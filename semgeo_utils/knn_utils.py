#
# FAISS-backed replacement for simple_knn._C.distCUDA2.
#

import numpy as np
import torch

import faiss


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """Return each point's mean squared distance to its three nearest neighbors, using GPU FAISS when available."""
    pts = np.ascontiguousarray(points.detach().cpu().numpy(), dtype=np.float32)
    n, d = pts.shape
    k = min(4, n)

    def _build(on_gpu: bool):
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
        index, _res = _build(on_gpu=use_gpu)
        D, _ = index.search(pts, k)
    except RuntimeError as e:
        print(f"[distCUDA2] faiss GPU 路径失败, 回退到 CPU: {str(e)[:120]}")
        index, _res = _build(on_gpu=False)
        D, _ = index.search(pts, k)

    if k > 1:
        mean_d2 = D[:, 1:k].mean(axis=1)
    else:
        mean_d2 = np.zeros(n, dtype=np.float32)
    return torch.from_numpy(np.ascontiguousarray(mean_d2)).float().to(points.device)
