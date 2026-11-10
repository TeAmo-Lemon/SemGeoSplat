#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
"""Live training viewer. Training and rendering must share the module lock when accessing Gaussian parameters."""

import threading
import numpy as np
import torch

try:
    import viser
    import nerfview

    _IMPORT_ERR = None
except Exception as _e:  # pragma: no cover
    viser = None  # type: ignore
    nerfview = None  # type: ignore
    _IMPORT_ERR = _e

# Module state. Single source of truth.
_state = {
    "gaussians": None,
    "pipe": None,
    "background": None,
    "scaling_modifier": 1.0,
    "render_mode": "RGB",  # "RGB" or "DINO"; toggled once DINO features exist
}
lock = threading.Lock()  # shared with train.py
_viewer = None
_server = None
_host = None
_port = None
_dino_cmap = (
    None  # (mean[D], proj[D,3], lo[3], hi[3]); built lazily on first DINO render
)

# Kept so any stray external caller (or `if network_gui.conn is None:` checks
# elsewhere) does not crash. The new viewer runs on its own thread; this flag
# is no longer used by train.py.
conn = None


def init(host, port):
    """Start the viser+nerfview viewer. Returns the shared lock."""
    global _viewer, _server, _host, _port
    if _IMPORT_ERR is not None:
        print(
            f"[viewer] disabled (viser/nerfview unavailable: "
            f"{type(_IMPORT_ERR).__name__}: {_IMPORT_ERR})"
        )
        return lock

    def render_fn(camera_state, render_tab_state):
        from gaussian_renderer import render as _gs_render  # lazy: circular-import safe
        from scene.cameras import MiniCam

        W = int(render_tab_state.viewer_width)
        H = int(render_tab_state.viewer_height)
        fov_y = float(camera_state.fov)
        aspect = float(camera_state.aspect)
        fov_x = 2.0 * float(np.arctan(np.tan(fov_y * 0.5) * aspect))

        c2w = torch.from_numpy(camera_state.c2w).float().cuda()
        w2c = torch.inverse(c2w)
        # MiniCam expects world_view_transform stored as w2c.T (see scene/cameras.py).
        world_view = w2c.transpose(0, 1).contiguous()
        full_proj = torch.eye(4, device=c2w.device)
        cam = MiniCam(W, H, fov_y, fov_x, 0.01, 100.0, world_view, full_proj)

        with lock:
            gaussians = _state["gaussians"]
            pipe = _state["pipe"]
            background = _state["background"]
            if gaussians is None or pipe is None or background is None:
                return np.zeros((H, W, 3), dtype=np.uint8)
            # scaling_modifier is a single float; read lock-free (CPython GIL
            # guarantees atomic single-attribute write).
            sm = _state["scaling_modifier"]
            with torch.no_grad():
                if (
                    _state["render_mode"] == "DINO"
                    and gaussians.get_dino_features.shape[0] > 0
                ):
                    global _dino_cmap
                    if _dino_cmap is None:
                        _dino_cmap = _build_dino_colormap(gaussians)
                    from gaussian_renderer import render_features as _gs_render_feat

                    feat = _gs_render_feat(
                        cam, gaussians, gaussians.get_dino_features
                    )  # [D,H,W]
                    mean, proj, lo, hi = _dino_cmap
                    proj_map = (feat.permute(1, 2, 0) - mean) @ proj  # [H,W,3]
                    rgb = (
                        ((proj_map - lo) / (hi - lo).clamp_min(1e-8))
                        .clamp(0, 1)
                        .permute(2, 0, 1)
                    )
                elif (
                    _state["render_mode"] == "Region"
                    and getattr(gaussians, "cluster_labels", None) is not None
                ):
                    labels = gaussians.cluster_labels
                    device = gaussians.get_xyz.device
                    labels = labels.to(device)
                    cmap = _build_region_colormap(labels).to(device)
                    K = cmap.shape[0]
                    colors = torch.full(
                        (labels.shape[0], 3), 0.12, device=device, dtype=torch.float32
                    )
                    valid = labels >= 0
                    if valid.any() and K > 0:
                        colors[valid] = cmap[labels[valid].clamp(min=0) % K]
                    rgb = _gs_render(
                        cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=sm,
                        override_color=colors,
                    )["render"]
                else:
                    rgb = _gs_render(
                        cam,
                        gaussians,
                        pipe,
                        background,
                        scaling_modifier=sm,
                    )["render"]
        return (
            (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        )

    _server = viser.ViserServer(host=host, port=port, verbose=False)
    _viewer = nerfview.Viewer(server=_server, render_fn=render_fn, mode="training")
    _host, _port = host, port

    # nerfview's training-mode update() asserts num_train_rays_per_sec is
    # tracked AND requires the train_util slider to be 1.0 to skip the
    # FPS-based throttle branch. We don't measure rays/sec, and we want a
    # re-render every iteration (gsplat is fast enough), so pin both values.
    _viewer.render_tab_state.num_train_rays_per_sec = 1.0
    if "train_util_slider" in _viewer._training_tab_handles:
        _viewer._training_tab_handles["train_util_slider"].value = 1.0

    sm_slider = _server.gui.add_slider(
        "Scaling modifier",
        min=0.0,
        max=2.0,
        step=0.01,
        initial_value=1.0,
        order=-1.0,
        hint="缩放所有高斯椭球的尺度 (0=完全收缩, 1=训练值, >1=放大). "
        "想看每个高斯的『骨架』可以拖到 ~0.3.",
    )

    @sm_slider.on_update
    def _(_evt):
        # Float assignment is atomic in CPython; keep this on_update
        # lock-free so the viser message handler never blocks on the
        # training thread.
        _state["scaling_modifier"] = float(sm_slider.value)

    print(f"[viewer] viser+nerfview at http://{host}:{port}")
    return lock


def bind_scene(gaussians, pipe, background):
    """Bind the live scene objects the viewer will render.
    Also sets up the RGB/DINO render-mode dropdown; DINO colours become available
    once _dino_cmap is built lazily (first DINO render after distillation)."""
    global _state
    with lock:
        _state["gaussians"] = gaussians
        _state["pipe"] = pipe
        _state["background"] = background

    if _server is not None:
        mode_dropdown = _server.gui.add_dropdown(
            "Render mode",
            options=["RGB", "DINO", "Region"],
            initial_value="RGB",
            order=-0.9,
            hint="RGB = SH 颜色; DINO = 蒸馏语义特征 (PCA→RGB, 同色=同语义); Region = 结构区域聚类.",
        )

        @mode_dropdown.on_update
        def _(_evt):
            _state["render_mode"] = mode_dropdown.value


def _build_dino_colormap(gaussians):
    """Precompute a stable 16D->RGB projection (PCA over per-Gaussian features).

    Returns (mean[D], proj[D,3], lo[3], hi[3]); normalization range is mean±2std
    over the centred projection so DINO colours stay consistent across viewpoints."""
    feats = gaussians.get_dino_features.detach().float()  # [N, D]
    mean = feats.mean(dim=0)
    centred = feats - mean
    _, _, V = torch.pca_lowrank(centred, q=min(3, feats.shape[1]))
    proj = V[:, :3]  # [D, 3]
    projected = centred @ proj  # [N, 3]
    std = projected.std(dim=0)
    return mean, proj, -2.0 * std, 2.0 * std


def _build_region_colormap(labels: torch.Tensor) -> torch.Tensor:
    """Build (K, 3) HSV colormap for *labels* (unique non-negative values)."""
    import colorsys

    unique = torch.unique(labels[labels >= 0])
    K = unique.numel()
    if K == 0:
        return torch.empty(0, 3, dtype=torch.float32)
    colors = []
    for i in range(K):
        hue = i / max(K, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
        colors.append([r, g, b])
    return torch.tensor(colors, dtype=torch.float32)


def update(iteration, num_train_rays_per_step=1):
    """Call once per training step so the viewer can re-render."""
    if _viewer is None:
        return
    _viewer.update(iteration, num_train_rays_per_step)


# Pause flag set by train.py -- when True, training ignores the viewer's
# pause button (useful for headless / scripted runs).
disable_pause = False


disable_hang_on_complete = False


def wait_if_paused(iteration):
    """Block the training loop while the viewer is in the 'paused' state.

    Polls cheaply and only blocks between iterations; uses the same event
    loop pattern as gsplat's simple_trainer example.
    """
    if _viewer is None:
        return
    import time

    while _viewer.state == "paused":
        time.sleep(0.05)
    # One quick re-render right after un-pause so the user sees fresh state.
    _viewer.update(iteration, 1)


def complete():
    if _viewer is None:
        return
    try:
        _viewer.complete()
    except Exception:
        pass

    if disable_hang_on_complete:
        return

    import time

    url = f"http://{_host}:{_port}" if _host else "viser"
    print(f"[viewer] 训练完成. viser 已切换到浏览模式, 可继续访问: {url}")
    print("[viewer] 按 Ctrl+C 退出.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viewer] 收到 Ctrl+C, 退出.")


# --- legacy stubs --------------------------------------------------------
# Kept as no-ops so any stale call site fails soft, not loud.


def try_connect():
    pass


def read():
    return None


def send(_a, _b):
    pass


def receive():
    return None, None, None, None, None, None
