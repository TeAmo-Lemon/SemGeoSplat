#!/usr/bin/env python3
"""Interactive RGB and semantic-feature viewer with camera-path export."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

DEFAULT_MODEL_ROOT = "./output"

# Add project root to path so we can import local modules
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _list_scenes(base_path: Path) -> list[str]:
    """List subdirectories in *base_path* that contain a point_cloud/ folder."""
    scenes = []
    if base_path.is_dir():
        for d in sorted(base_path.iterdir()):
            if d.is_dir() and (d / "point_cloud").is_dir():
                scenes.append(d.name)
    return scenes


def _list_iterations(model_path: Path) -> list[str]:
    """List iteration_* directories under *model_path*/point_cloud/ that have a .ply."""
    iters = []
    pcd_dir = model_path / "point_cloud"
    if pcd_dir.is_dir():
        for d in sorted(pcd_dir.iterdir()):
            if d.is_dir() and d.name.startswith("iteration_"):
                if (d / "point_cloud.ply").is_file():
                    iters.append(d.name)
    return iters


def _find_latest_ply(model_path: Path) -> Path:
    """Return the ply with the highest iteration number."""
    pcd_dir = model_path / "point_cloud"
    if not pcd_dir.is_dir():
        raise FileNotFoundError(f"point_cloud/ not found in {model_path}")

    best: Optional[Path] = None
    best_iter = -1
    for d in pcd_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("iteration_"):
            continue
        try:
            it = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        ply = d / "point_cloud.ply"
        if ply.is_file() and it > best_iter:
            best_iter = it
            best = ply
    if best is None:
        raise FileNotFoundError(f"No point_cloud.ply found under {pcd_dir}")
    return best


def _find_dino_sidecar(model_path: Path, ply_path: Path) -> Optional[Path]:
    """Locate the per-gaussian DINO sidecar:
    1) 新位置: <refine_dir>/dino_features_gaussian.pt — per-gaussian 特征与几何绑定、
       跨 style/style_iterations 复用, 落盘在 refine 同树 (refine_dir 由 cfg_args 的
       --start_ply 反推, 见 style_pipeline/dino_distill.py);
    2) 旧位置兜底: ply 同目录 dino_features.pt (旧版产物)."""
    start_ply = _read_cfg_args_attr(model_path, "start_ply")
    if start_ply:
        refine_dir = Path(start_ply).parent.parent / "iteration_refine"
        new_loc = refine_dir / "dino_features_gaussian.pt"
        if new_loc.is_file():
            return new_loc
    legacy = ply_path.parent / "dino_features.pt"
    return legacy if legacy.is_file() else None


def _load_gaussians(model_path: Path, iteration: int) -> tuple:
    """Load GaussianModel from a ply file. Returns (gaussians, sh_degree)."""
    from scene.gaussian_model import GaussianModel

    if iteration < 0:
        ply_path = _find_latest_ply(model_path)
    else:
        ply_path = (
            model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
        )
    if not ply_path.is_file():
        raise FileNotFoundError(f"ply not found: {ply_path}")

    # Read the first point to determine SH degree from the feature count
    from plyfile import PlyData

    ply = PlyData.read(str(ply_path))
    extra_f_names = sorted(
        [p.name for p in ply.elements[0].properties if p.name.startswith("f_rest_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    # f_rest has 3 * ((sh_degree+1)^2 - 1) coefficients → solve for sh_degree
    n_extra = len(extra_f_names)
    sh_degree = int(math.sqrt(n_extra / 3 + 1)) - 1

    gs = GaussianModel(sh_degree)
    gs.load_ply(str(ply_path))
    print(f"Loaded {gs.get_xyz.shape[0]:,} gaussians  (SH degree {sh_degree})")
    print(f"  from: {ply_path}")

    has_dino = False
    dino_path = _find_dino_sidecar(model_path, ply_path)
    if dino_path is not None:
        try:
            gs.load_dino_sidecar(str(dino_path))
            has_dino = True
            print(
                f"  + DINO features [{gs.get_dino_features.shape[1]}D] from: {dino_path}"
            )
        except Exception as exc:
            print(f"  ! DINO sidecar found but unusable ({exc}); RGB only")

    has_regions = False
    regions_path = ply_path.parent / "regions.pt"
    if regions_path.is_file():
        try:
            regions_data = torch.load(
                str(regions_path), map_location="cpu", weights_only=True
            )
            cluster_labels = regions_data.get("cluster_labels")
            if cluster_labels is not None:
                n_gs = gs.get_xyz.shape[0]
                if cluster_labels.shape[0] != n_gs:
                    print(
                        f"  ! regions.pt label count ({cluster_labels.shape[0]}) "
                        f"!= gaussians ({n_gs}); skipping"
                    )
                else:
                    gs.cluster_labels = cluster_labels
                    n_regions = int(cluster_labels.max()) + 1
                    has_regions = True
                    print(
                        f"  + region labels [{n_regions} regions] from: {regions_path}"
                    )
        except Exception as exc:
            print(f"  ! regions.pt found but unusable ({exc})")

    has_primitives = False
    rp_path = ply_path.parent / "region_primitives.pt"
    if rp_path.is_file():
        try:
            rp_data = torch.load(str(rp_path), map_location="cpu", weights_only=True)
            primitive_labels = rp_data.get("primitive_labels")
            if primitive_labels is not None:
                n_gs = gs.get_xyz.shape[0]
                if primitive_labels.shape[0] != n_gs:
                    print(
                        f"  ! region_primitives.pt label count ({primitive_labels.shape[0]}) "
                        f"!= gaussians ({n_gs}); skipping"
                    )
                else:
                    gs.primitive_labels = primitive_labels
                    n_primitives = int(primitive_labels.max()) + 1
                    has_primitives = True
                    print(
                        f"  + primitive labels [{n_primitives} primitives] from: {rp_path}"
                    )
        except Exception as exc:
            print(f"  ! region_primitives.pt found but unusable ({exc})")

    return gs, sh_degree, has_dino, has_regions, has_primitives


#  Camera ring display  (migrated from nerfstudio viewer: frustums + GT images)


_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


class CameraDisplayConfig:
    """Configuration for the camera frustum overlay.

    Attributes:
        image_height: Frustum texture height in pixels (nerfstudio uses 100).
        scale_factor: Scale factor relative to the camera-ring extent.
        scale_min: Minimum frustum scale.
        scale_max: Maximum frustum scale.
    """

    image_height: int = 100
    scale_factor: float = 0.015
    scale_min: float = 0.01
    scale_max: float = 0.6


def _read_cfg_args_attr(model_path: Path, attr: str) -> Optional[str]:
    """Recover a string attr from the cfg_args Namespace written at train time."""
    cfg_file = model_path / "cfg_args"
    if not cfg_file.is_file():
        return None
    text = cfg_file.read_text(errors="replace")
    m = re.search(rf"\b{re.escape(attr)}=['\"]([^'\"]*)['\"]", text)
    return m.group(1) if m else None


def _read_cfg_args_source(model_path: Path) -> Optional[str]:
    """Recover the dataset source_path from the cfg_args written at train time."""
    return _read_cfg_args_attr(model_path, "source_path")


def _find_gt_image_map(source_path: Optional[str]) -> dict:
    """Map image stems -> file paths under the dataset (one rglob pass).

    COLMAP image names are unique, so the stem map is 1:1. For Blender scenes
    (train/ and test/ share stems) the first match wins, which is fine here.
    """
    if not source_path:
        return {}
    root = Path(source_path)
    if not root.is_dir():
        return {}
    stem_map: dict = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS:
            stem_map.setdefault(p.stem, p)
    return stem_map


def _load_gt_image(path: Path, config: CameraDisplayConfig) -> Optional[np.ndarray]:
    """Load a GT image as a small uint8 HWC array for the frustum texture."""
    try:
        from PIL import Image

        im = Image.open(path)
        im.draft(
            "RGB", (2 * config.image_height, config.image_height)
        )  # decode JPEG at reduced scale
        im = im.convert("RGB")
        if im.height > config.image_height:
            w = max(1, round(im.width * config.image_height / im.height))
            im = im.resize((w, config.image_height), Image.Resampling.LANCZOS)
        return np.asarray(im, dtype=np.uint8)
    except Exception as exc:
        print(f"  ! failed to load GT image {path} ({exc})")
        return None


def _load_cameras(model_path: Path, config: CameraDisplayConfig) -> list:
    """Load camera frustum data (pose + GT image) from a trained scene.

    Reads cameras.json (written by Scene at train time) and, via cfg_args,
    the dataset folder for the GT images. Returns one dict per camera with
    keys: name, position, wxyz, fov, aspect, scale, image, idx; empty if unavailable.

    Camera poses are COLMAP/OpenCV convention (+Z forward), which is exactly
    viser's convention, so no extra rotation is needed.
    """
    cam_file = model_path / "cameras.json"
    if not cam_file.is_file():
        print("  ! cameras.json not found; skipping camera display")
        return []
    try:
        with open(cam_file) as f:
            cam_list = json.load(f)
    except Exception as exc:
        print(f"  ! cameras.json unreadable ({exc}); skipping camera display")
        return []

    source_path = _read_cfg_args_source(model_path)
    gt_map = _find_gt_image_map(source_path)
    if source_path and not gt_map:
        print(f"  ! no GT images found under dataset {source_path}")

    positions = np.array([c["position"] for c in cam_list], dtype=np.float64)
    if positions.size == 0:
        return []
    center = positions.mean(axis=0)
    extent = float(np.max(np.linalg.norm(positions - center, axis=1)))
    scale = float(
        np.clip(config.scale_factor * extent, config.scale_min, config.scale_max)
    )

    from viser.transforms import SO3

    cams = []
    for i, c in enumerate(cam_list):
        rot = np.asarray(c["rotation"], dtype=np.float64)  # c2w rotation
        pos = np.asarray(c["position"], dtype=np.float64)  # c2w translation
        fov = 2.0 * math.atan((c["height"] / 2.0) / c["fy"])  # vertical FOV
        img_path = gt_map.get(
            Path(c["img_name"]).stem
        )  # img_name includes the extension
        cams.append(
            {
                "name": f"/cameras/camera_{i:05d}",
                "position": pos,
                "wxyz": SO3.from_matrix(rot).wxyz,
                "fov": float(fov),
                "aspect": float(c["width"] / c["height"]),
                "scale": scale,
                "image": _load_gt_image(img_path, config)
                if img_path is not None
                else None,
                "idx": i,
            }
        )
    return cams


def _build_dino_colormap(gs):
    """Precompute a stable 16D->RGB projection (PCA over per-Gaussian features).

    Returns (mean[D], proj[D,3], lo[3], hi[3]) used to map rendered feature maps to
    colour. Normalization range is mean±2std over the centred projection so colours
    stay consistent across viewpoints."""
    feats = gs.get_dino_features.detach().float()  # [N, D]
    mean = feats.mean(dim=0)
    centred = feats - mean
    # randomized PCA: V[:, :3] are the top-3 principal directions in D-space.
    _, _, V = torch.pca_lowrank(centred, q=min(3, feats.shape[1]))
    proj = V[:, :3]  # [D, 3]
    projected = centred @ proj  # [N, 3]
    std = projected.std(dim=0)
    lo = -2.0 * std
    hi = 2.0 * std
    return mean, proj, lo, hi


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


#  Viewer


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D Gaussian Splatting viewer"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=DEFAULT_MODEL_ROOT,
        help="Base directory containing 3DGS output folders (for UI browsing).",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=Path,
        default=None,
        help="Output directory of a trained scene (optional; can select via UI).",
    )
    parser.add_argument(
        "-i",
        "--iteration",
        type=int,
        default=-1,
        help="Iteration to load (-1 = latest). Only used with --model-path.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--bg",
        choices=["white", "black", "random"],
        default="white",
        help="Background colour (default: white).",
    )
    parser.add_argument(
        "--width", type=int, default=1920, help="Viewport width (default: 1920)."
    )
    parser.add_argument(
        "--height", type=int, default=1080, help="Viewport height (default: 1080)."
    )
    parser.add_argument(
        "--antialiasing",
        action="store_true",
        help="Use anti-aliased rasterization (slower but smoother).",
    )
    args = parser.parse_args()

    import viser
    import nerfview
    from semgeo_utils.nerfview_compat import install_nerfview_compat
    from scene.cameras import MiniCam
    from gaussian_renderer import render, render_features
    from scene.gaussian_model import OrientedBox

    install_nerfview_compat()
    server = viser.ViserServer(host=args.host, port=args.port, verbose=False)

    _bg_ptr = [
        torch.tensor(
            {"white": [1, 1, 1], "black": [0, 0, 0], "random": [1, 1, 1]}[args.bg],
            dtype=torch.float32,
            device="cuda",
        )
    ]
    _scale_mod = [1.0]
    _mode = ["RGB"]  # "RGB" or "DINO"
    # None or (gaussians, sh_degree, has_dino, dino_cmap, has_regions, has_primitives)
    _gs_state: list = [None]
    # Camera ring display: frustum handles for the loaded scene + visibility toggle.
    _cam_handles: list = []
    _cams_visible = [True]
    _cam_config = CameraDisplayConfig()

    # Crop viewport state
    _crop_enabled = [False]
    _crop_center = [0.0, 0.0, 0.0]
    _crop_scale = [
        1.0,
        1.0,
        1.0,
    ]  # full extent along each box-local axis (OrientedBox.S semantics)
    _crop_follow_orbit = [
        True
    ]  # keep crop center synced to the Orbit Origin Tool (camera look-at) position
    _crop_box_handle = None  # wireframe crop preview; recreated when the size changes (scale is one-shot)
    _crop_box_cache = [None]  # cached OrientedBox handed to the renderer
    _last_crop_scale = [
        None
    ]  # last pushed box size, to detect changes (see _crop_commit)
    _crop_initialized = [
        False
    ]  # auto-fit the crop box to the first loaded scene (once)
    _last_look_at = [None]  # last orbit-origin (axes) position, to detect axes movement
    _crop_rpy = [
        math.pi / 2,
        0.0,
        0.0,
    ]  # box rotation (radians); starts with local Y -> world +Z (viser's default up)
    _crop_axes_up = [
        None
    ]  # last axes-up direction (== camera up); change -> rotate box to match
    _crop_syncing = [False]  # guard against slider feedback loops
    _scene_xyz = [None]  # current scene gaussian positions (for adaptive slider ranges)

    # Dummy pipeline with just the antialiasing flag (render() needs a `pipe` arg).
    class _Pipe:
        convert_SHs_python = False
        debug = False
        antialiasing = args.antialiasing

    pipe = _Pipe()

    def _set_camera_display(model_path: Path):
        """(Re)create camera frustums with GT images for the loaded scene."""
        for h in _cam_handles:
            try:
                h.remove()
            except Exception:
                pass
        _cam_handles.clear()
        for c in _load_cameras(model_path, _cam_config):
            h = server.scene.add_camera_frustum(
                name=c["name"],
                fov=c["fov"],
                aspect=c["aspect"],
                scale=c["scale"],
                image=c["image"],
                wxyz=c["wxyz"],
                position=c["position"],
                visible=_cams_visible[0],
            )

            # Click-to-jump: clicking a frustum moves the viewer camera to that
            # camera's pose (matching nerfstudio's create_on_click_callback).
            def _make_on_click(cam_idx):
                def _on_click(event: viser.SceneNodePointerEvent) -> None:
                    with event.client.atomic():
                        event.client.camera.position = event.target.position
                        event.client.camera.wxyz = event.target.wxyz

                return _on_click

            h.on_click(_make_on_click(c["idx"]))

            _cam_handles.append(h)
        if _cam_handles:
            print(f"  + {len(_cam_handles)} cameras with GT on frustums (/cameras)")

    if args.model_path is not None:
        print(f"Loading scene from {args.model_path} …")
        gaussians, sh_degree, has_dino, has_regions, has_primitives = _load_gaussians(
            args.model_path, args.iteration
        )
        dino_cmap = _build_dino_colormap(gaussians) if has_dino else None
        _gs_state[0] = (
            gaussians,
            sh_degree,
            has_dino,
            dino_cmap,
            has_regions,
            has_primitives,
        )
        _scene_xyz[0] = gaussians.get_xyz
        print(f"  {gaussians.get_xyz.shape[0]:,} gaussians  |  SH degree {sh_degree}")
        _set_camera_display(args.model_path)

    def render_fn(camera_state, render_tab_state):
        W = int(render_tab_state.viewer_width)
        H = int(render_tab_state.viewer_height)
        fov_y = float(camera_state.fov)
        aspect = float(camera_state.aspect)
        fov_x = 2.0 * float(np.arctan(np.tan(fov_y * 0.5) * aspect))

        c2w = torch.from_numpy(camera_state.c2w).float().cuda()
        w2c = torch.inverse(c2w)
        world_view = w2c.transpose(0, 1).contiguous()
        full_proj = torch.eye(4, device=c2w.device)
        cam = MiniCam(W, H, fov_y, fov_x, 0.01, 100.0, world_view, full_proj)

        state = _gs_state[0]
        if state is None:
            return np.full((H, W, 3), 255 if args.bg == "white" else 0, dtype=np.uint8)

        gaussians, _sh_deg, has_dino, dino_cmap, has_regions, has_primitives = state

        # Cached crop box (rebuilt by the Crop Viewport UI on state changes).
        _crop_box = _crop_box_cache[0]

        with torch.no_grad():
            if _mode[0] == "DINO" and dino_cmap is not None:
                feat = render_features(
                    cam, gaussians, gaussians.get_dino_features, crop_box=_crop_box
                )  # [D, H, W]
                mean, proj, lo, hi = dino_cmap
                proj_map = (feat.permute(1, 2, 0) - mean) @ proj  # [H, W, 3]
                rgb = (
                    ((proj_map - lo) / (hi - lo).clamp_min(1e-8))
                    .clamp(0, 1)
                    .permute(2, 0, 1)
                )
            elif _mode[0] == "Region" and (has_regions or has_primitives):
                labels = getattr(gaussians, "primitive_labels", None)
                if labels is None:
                    labels = getattr(gaussians, "cluster_labels", None)
                if labels is not None:
                    device = gaussians.get_xyz.device
                    labels = labels.to(device)
                    cmap = _build_region_colormap(labels).to(device)
                    K = cmap.shape[0]
                    colors = torch.full((labels.shape[0], 3), 0.12, device=device)
                    valid = labels >= 0
                    if valid.any() and K > 0:
                        colors[valid] = cmap[labels[valid].clamp(min=0) % K]
                    rgb = render(
                        cam,
                        gaussians,
                        pipe,
                        _bg_ptr[0],
                        scaling_modifier=_scale_mod[0],
                        override_color=colors,
                        crop_box=_crop_box,
                    )["render"]
                else:
                    rgb = render(
                        cam,
                        gaussians,
                        pipe,
                        _bg_ptr[0],
                        scaling_modifier=_scale_mod[0],
                        crop_box=_crop_box,
                    )["render"]
            else:
                rgb = render(
                    cam,
                    gaussians,
                    pipe,
                    _bg_ptr[0],
                    scaling_modifier=_scale_mod[0],
                    crop_box=_crop_box,
                )["render"]
        return (
            (rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        )

    viewer = nerfview.Viewer(
        server=server,
        render_fn=render_fn,
        mode="rendering",
    )

    #  Scene Browser UI

    _BASE_PLACEHOLDER = "(enter base path)"
    _SCENE_PLACEHOLDER = "(select scene)"
    _ITER_PLACEHOLDER = "(select iteration)"

    with server.gui.add_folder("\U0001f4c1 Scene Browser", order=-2.0):
        base_text = server.gui.add_text(
            "Base path",
            initial_value=args.base_path or "",
            hint="Directory containing 3DGS output folders. Press Enter to refresh.",
        )

        refresh_btn = server.gui.add_button("Refresh scenes")

        scene_dropdown = server.gui.add_dropdown(
            "Scene",
            options=[_SCENE_PLACEHOLDER],
            initial_value=_SCENE_PLACEHOLDER,
        )

        iter_dropdown = server.gui.add_dropdown(
            "Iteration",
            options=[_ITER_PLACEHOLDER],
            initial_value=_ITER_PLACEHOLDER,
        )

        load_btn = server.gui.add_button("Load Scene")

        status_text = server.gui.add_text(
            "Status",
            initial_value=(
                f"Loaded: {_gs_state[0][0].get_xyz.shape[0]:,} gaussians"
                if _gs_state[0] is not None
                else "No scene loaded"
            ),
            disabled=True,
        )

    def _refresh_scenes():
        bp = Path(base_text.value)
        scenes = _list_scenes(bp)
        if scenes:
            scene_dropdown.options = scenes
            if scene_dropdown.value not in scenes:
                scene_dropdown.value = scenes[0]
            _refresh_iterations()
        else:
            scene_dropdown.options = [_SCENE_PLACEHOLDER]
            scene_dropdown.value = _SCENE_PLACEHOLDER
            iter_dropdown.options = [_ITER_PLACEHOLDER]
            iter_dropdown.value = _ITER_PLACEHOLDER

    def _refresh_iterations():
        bp = Path(base_text.value)
        sc = scene_dropdown.value
        if sc and sc not in (_SCENE_PLACEHOLDER,):
            model_path = bp / sc
            iters = _list_iterations(model_path)
            if iters:
                iter_dropdown.options = iters
                if iter_dropdown.value not in iters:
                    iter_dropdown.value = iters[-1]  # default to latest
            else:
                iter_dropdown.options = [_ITER_PLACEHOLDER]
                iter_dropdown.value = _ITER_PLACEHOLDER
        else:
            iter_dropdown.options = [_ITER_PLACEHOLDER]
            iter_dropdown.value = _ITER_PLACEHOLDER

    def _load_selected():
        bp = Path(base_text.value)
        sc = scene_dropdown.value
        it = iter_dropdown.value
        if not sc or sc in (_SCENE_PLACEHOLDER,):
            return
        if not it or it in (_ITER_PLACEHOLDER,):
            return

        model_path = bp / sc
        # Parse iteration number from name like "iteration_1500"
        try:
            iter_num = int(it.split("_")[1])
        except (IndexError, ValueError):
            iter_num = -1

        try:
            print(f"Loading scene from {model_path} (iteration={it}) …")
            gaussians, sh_degree, has_dino, has_regions, has_primitives = (
                _load_gaussians(model_path, iter_num)
            )
            dino_cmap = _build_dino_colormap(gaussians) if has_dino else None
            _gs_state[0] = (
                gaussians,
                sh_degree,
                has_dino,
                dino_cmap,
                has_regions,
                has_primitives,
            )
            _scene_xyz[0] = gaussians.get_xyz
            _update_crop_ranges(gaussians.get_xyz)
            n_gs = gaussians.get_xyz.shape[0]
            print(f"  {n_gs:,} gaussians  |  SH degree {sh_degree}")
            status_text.value = f"Loaded: {model_path.name}/{it} ({n_gs:,} gaussians)"
            _set_camera_display(model_path)
        except Exception as exc:
            print(f"Failed to load scene: {exc}")
            status_text.value = f"Error: {exc}"

    @base_text.on_update
    def _(evt):
        _refresh_scenes()

    @refresh_btn.on_click
    def _(evt):
        _refresh_scenes()

    @scene_dropdown.on_update
    def _(evt):
        _refresh_iterations()

    @load_btn.on_click
    def _(evt):
        _load_selected()

    # Initial scene list population (if base path was given)
    if args.model_path is None and args.base_path:
        _refresh_scenes()

    #  Controls UI

    with server.gui.add_folder("Controls", order=-1.0):
        # Camera ring visibility toggle (Hide/Show pair, like nerfstudio's header buttons).
        hide_cams_btn = server.gui.add_button("Hide Cameras", icon=viser.Icon.EYE_OFF)
        show_cams_btn = server.gui.add_button("Show Cameras", icon=viser.Icon.EYE)
        show_cams_btn.visible = False

        def _set_camera_visibility(visible: bool):
            _cams_visible[0] = visible
            for h in _cam_handles:
                h.visible = visible
            hide_cams_btn.visible = visible
            show_cams_btn.visible = not visible

        hide_cams_btn.on_click(lambda _: _set_camera_visibility(False))
        show_cams_btn.on_click(lambda _: _set_camera_visibility(True))

        mode_dropdown = server.gui.add_dropdown(
            "Render mode",
            options=["RGB", "DINO", "Region"],
            initial_value="RGB",
            hint="RGB = SH colours; DINO = distilled semantic features (PCA→RGB); Region = FGW primitive assignment (fallback: structural clusters).",
        )

        @mode_dropdown.on_update
        def _(evt):
            _mode[0] = mode_dropdown.value

        scale_slider = server.gui.add_slider(
            "Scaling modifier",
            min=0.0,
            max=2.0,
            step=0.01,
            initial_value=1.0,
            hint="Scale all gaussians (0=collapsed, 1=trained, >1=enlarged).",
        )

        @scale_slider.on_update
        def _(evt):
            _scale_mod[0] = float(scale_slider.value)

        bg_dropdown = server.gui.add_dropdown(
            "Background",
            options=["white", "black", "random"],
            initial_value=args.bg,
            hint="Rendering background colour.",
        )

        @bg_dropdown.on_update
        def _(evt):
            choice = bg_dropdown.value
            if choice == "random":
                import random as _rnd

                c = [_rnd.random(), _rnd.random(), _rnd.random()]
            else:
                c = [1, 1, 1] if choice == "white" else [0, 0, 0]
            _bg_ptr[0] = torch.tensor(c, dtype=torch.float32, device="cuda")

        viewport_res = server.gui.add_dropdown(
            "Resolution",
            options=["1920×1080", "1280×720", "960×540", "640×360"],
            initial_value=f"{args.width}×{args.height}",
            hint="Viewport resolution (higher = sharper but slower).",
        )

        @viewport_res.on_update
        def _(evt):
            val = viewport_res.value
            if "×" in val:
                w_str, h_str = val.split("×")
                w, h = int(w_str.strip()), int(h_str.strip())
                viewer.render_tab_state.viewer_width = w
                viewer.render_tab_state.viewer_height = h
                viewer.rerender(None)

    #  Crop Viewport (bounding box oriented to the Orbit Origin Tool's axes)

    from viser.transforms import SO3 as _SO3

    def _crop_rerender(*_):
        viewer.rerender(None)

    def _crop_box_wxyz():
        """Quaternion for the box from the accumulated rotation (matches the
        Orbit Origin Tool's orientation, whose Y is kept aligned with the
        camera up via minimal rotations)."""
        return tuple(_SO3.from_rpy_radians(*_crop_rpy).wxyz)

    def _crop_push_visuals():
        """Mirror the internal crop state onto the wireframe box."""
        if _crop_box_handle is None:
            return
        wxyz = _crop_box_wxyz()
        _crop_box_handle.position = tuple(_crop_center)
        _crop_box_handle.wxyz = wxyz
        _crop_box_handle.visible = _crop_enabled[0]

    def _crop_commit():
        """Push visuals, refresh the cached OrientedBox for the renderer, re-render."""
        nonlocal _crop_box_handle
        _crop_push_visuals()
        if _crop_enabled[0] and tuple(_crop_scale) != _last_crop_scale[0]:
            # `scale` is a one-shot prop in viser: recreate the wireframe box on size
            # changes (position/rotation still update live via _crop_push_visuals).
            try:
                _crop_box_handle.remove()
            except Exception:
                pass
            _crop_box_handle = _crop_box_from_state()
            _last_crop_scale[0] = tuple(_crop_scale)
        _crop_box_cache[0] = (
            OrientedBox.from_params(_crop_center, _crop_rpy, _crop_scale)
            if _crop_enabled[0]
            else None
        )
        _crop_rerender()

    def _crop_box_from_state():
        """Build the wireframe preview box at the current center/size/orientation."""
        return server.scene.add_box(
            "Crop Box",
            color=(255, 165, 0),
            dimensions=(1.0, 1.0, 1.0),  # unit box; the "Size" sliders drive `scale`
            scale=tuple(max(s, 1e-4) for s in _crop_scale),
            wireframe=True,
            opacity=1.0,
            position=tuple(_crop_center),
            wxyz=_crop_box_wxyz(),
            visible=_crop_enabled[0],
        )

    def _sync_crop_sliders():
        """Push the internal crop state back into the slider widgets (guarded)."""
        _crop_syncing[0] = True
        crop_center_x.value = _crop_center[0]
        crop_center_y.value = _crop_center[1]
        crop_center_z.value = _crop_center[2]
        crop_sx.value = _crop_scale[0]
        crop_sy.value = _crop_scale[1]
        crop_sz.value = _crop_scale[2]
        _crop_syncing[0] = False

    def _sync_crop_sliders():
        """Push the internal crop state back into the slider widgets (guarded)."""
        _crop_syncing[0] = True
        crop_center_x.value = _crop_center[0]
        crop_center_y.value = _crop_center[1]
        crop_center_z.value = _crop_center[2]
        crop_sx.value = _crop_scale[0]
        crop_sy.value = _crop_scale[1]
        crop_sz.value = _crop_scale[2]
        _crop_syncing[0] = False

    def _sync_crop_to_orbit(look_at):
        """Move the crop center to the Orbit Origin Tool's position, which viser
        pins to the camera's look-at point."""
        _crop_center[:] = (float(look_at[0]), float(look_at[1]), float(look_at[2]))
        _sync_crop_sliders()

    def _register_orbit_sync(client):
        """Watch each client's camera. The crop box follows the Orbit Origin Tool:
        whenever the tool's Y axis (the camera up, shown by the arrow) rotates,
        the box rotates by the same minimal rotation so its up always matches
        the arrow's up. Center follows the axes' position (== look-at)."""

        @client.camera.on_update
        def _(camera):
            if _crop_syncing[0]:
                return
            changed = _sync_crop_to_up(camera)
            if _crop_follow_orbit[0]:
                look_at = np.asarray(camera.look_at, dtype=np.float64)
                if _last_look_at[0] is None:
                    _last_look_at[0] = look_at.copy()
                elif not np.allclose(_last_look_at[0], look_at):
                    _last_look_at[0] = look_at.copy()
                    _sync_crop_to_orbit(look_at)
                    changed = True
            if changed:
                _crop_commit()

        # Baseline: start from viser's default axes up (+Z, matching the initial
        # box rotation), then align the box to the connecting client's axes up.
        _crop_axes_up[0] = np.array([0.0, 0.0, 1.0])
        _last_look_at[0] = np.asarray(client.camera.look_at, dtype=np.float64).copy()
        _sync_crop_to_up(client.camera)

    def _rot_align(a, b):
        """Minimal rotation R (3x3) taking unit vector `a` to unit vector `b`."""
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        a = a / (np.linalg.norm(a) + 1e-12)
        b = b / (np.linalg.norm(b) + 1e-12)
        axis = np.cross(a, b)
        n = np.linalg.norm(axis)
        cos_a = float(np.clip(np.dot(a, b), -1.0, 1.0))
        if n < 1e-8:
            if cos_a > 0.999:
                return np.eye(3)
            t = np.zeros(3)
            t[int(np.argmin(np.abs(a)))] = 1.0
            axis = np.cross(a, t)
            n = np.linalg.norm(axis)
            if n < 1e-8:
                return np.eye(3)
            angle = np.pi
        else:
            axis = axis / n
            angle = math.acos(cos_a)
        return _SO3.exp(axis * angle).as_matrix()

    def _sync_crop_to_up(camera):
        """Rotate the box so its up equals the Orbit Origin Tool's Y (the camera
        up). Uses the same minimal-rotation accumulation as viser's pivot, so a
        plain view orbit (up unchanged) never rotates the box."""
        new_up = np.asarray(camera.up_direction, dtype=np.float64)
        if _crop_axes_up[0] is None:
            _crop_axes_up[0] = new_up.copy()
            return False
        R_delta = _rot_align(_crop_axes_up[0], new_up)
        _crop_axes_up[0] = new_up.copy()
        if np.allclose(R_delta, np.eye(3), atol=1e-6):
            return False
        R_new = R_delta @ _SO3.from_rpy_radians(*_crop_rpy).as_matrix()
        r, p, ya = _SO3.from_matrix(R_new).as_rpy_radians()
        _crop_rpy[:] = (float(r), float(p), float(ya))
        return True

    server.on_client_connect(_register_orbit_sync)

    with server.gui.add_folder("Crop Viewport", order=-0.5):
        crop_enable_cb = server.gui.add_checkbox(
            "Enable",
            initial_value=False,
            hint="Crop the scene to a bounding box aligned with the Orbit Origin Tool's axes (orange wireframe).",
        )

        @crop_enable_cb.on_update
        def _(evt):
            _crop_enabled[0] = crop_enable_cb.value
            _last_crop_scale[0] = None  # (re)create the wireframe box on toggle
            _crop_commit()

        crop_follow_cb = server.gui.add_checkbox(
            "Follow orbit origin",
            initial_value=_crop_follow_orbit[0],
            hint="Move the crop center together with the Orbit Origin Tool (its axes sit at the orbit origin).",
        )

        @crop_follow_cb.on_update
        def _(evt):
            _crop_follow_orbit[0] = crop_follow_cb.value
            # When enabled, snap the crop center to the current orbit origin immediately.
            if _crop_follow_orbit[0]:
                clients = server.get_clients()
                if clients:
                    look_at = np.asarray(
                        next(iter(clients.values())).camera.look_at, dtype=np.float64
                    )
                    _last_look_at[0] = look_at.copy()
                    _sync_crop_to_orbit(look_at)
                    _crop_commit()

        crop_center_x = server.gui.add_slider(
            "Center X", min=-5.0, max=5.0, step=0.01, initial_value=0.0
        )
        crop_center_y = server.gui.add_slider(
            "Center Y", min=-5.0, max=5.0, step=0.01, initial_value=0.0
        )
        crop_center_z = server.gui.add_slider(
            "Center Z", min=-5.0, max=5.0, step=0.01, initial_value=0.0
        )

        def _on_center_update(*_):
            if _crop_syncing[0]:
                return
            _crop_center[:] = (
                crop_center_x.value,
                crop_center_y.value,
                crop_center_z.value,
            )
            _crop_commit()

        crop_center_x.on_update(_on_center_update)
        crop_center_y.on_update(_on_center_update)
        crop_center_z.on_update(_on_center_update)

        # Size sliders: FULL extent along each box-local axis (matches the renderer's
        # hard crop: points with |local| < size/2 survive).
        crop_sx = server.gui.add_slider(
            "Size X",
            min=0.0,
            max=10.0,
            step=0.01,
            initial_value=1.0,
            hint="Full extent along the box's local X axis.",
        )
        crop_sy = server.gui.add_slider(
            "Size Y",
            min=0.0,
            max=10.0,
            step=0.01,
            initial_value=1.0,
            hint="Full extent along the box's local Y axis.",
        )
        crop_sz = server.gui.add_slider(
            "Size Z",
            min=0.0,
            max=10.0,
            step=0.01,
            initial_value=1.0,
            hint="Full extent along the box's local Z axis.",
        )

        def _on_scale_update(*_):
            if _crop_syncing[0]:
                return
            _crop_scale[:] = (crop_sx.value, crop_sy.value, crop_sz.value)
            _crop_commit()

        crop_sx.on_update(_on_scale_update)
        crop_sy.on_update(_on_scale_update)
        crop_sz.on_update(_on_scale_update)

        crop_fit_btn = server.gui.add_button("Fit to scene")

        @crop_fit_btn.on_click
        def _(evt):
            xyz = _scene_xyz[0]
            if xyz is None or xyz.numel() == 0:
                return
            lo = xyz.detach().float().min(dim=0).values.cpu().numpy()
            hi = xyz.detach().float().max(dim=0).values.cpu().numpy()
            _crop_center[:] = (
                float(0.5 * (lo[0] + hi[0])),
                float(0.5 * (lo[1] + hi[1])),
                float(0.5 * (lo[2] + hi[2])),
            )
            _crop_scale[:] = tuple(max(float(hi[i] - lo[i]), 1e-3) for i in range(3))
            _sync_crop_sliders()
            _crop_commit()

    def _update_crop_ranges(xyz):
        """Adapt the crop sliders to the loaded scene's spatial extent and (once)
        auto-fit the crop box to the whole scene so the wireframe/state starts
        somewhere useful."""
        if xyz is None or xyz.numel() == 0:
            return
        lo = xyz.detach().float().min(dim=0).values.cpu().numpy()
        hi = xyz.detach().float().max(dim=0).values.cpu().numpy()
        extent = float((hi - lo).max())
        margin = 0.05 * extent + 1e-3
        for sl, l, h in zip(
            (crop_center_x, crop_center_y, crop_center_z),
            lo - margin,
            hi + margin,
        ):
            sl.min = float(l)
            sl.max = float(h)
            sl.value = float(np.clip(sl.value, l, h))
        scale_max = 2.0 * extent + 1e-3
        for sl in (crop_sx, crop_sy, crop_sz):
            sl.min = 0.0
            sl.max = scale_max
            sl.value = float(np.clip(sl.value, 0.0, scale_max))
        if not _crop_initialized[0]:
            # First scene: wrap the crop around the whole point cloud.
            _crop_initialized[0] = True
            _crop_center[:] = (
                float(0.5 * (lo[0] + hi[0])),
                float(0.5 * (lo[1] + hi[1])),
                float(0.5 * (lo[2] + hi[2])),
            )
            _crop_scale[:] = tuple(max(float(hi[i] - lo[i]), 1e-3) for i in range(3))
        else:
            # Keep internal state consistent with any clamped slider values.
            _crop_center[:] = (
                crop_center_x.value,
                crop_center_y.value,
                crop_center_z.value,
            )
            _crop_scale[:] = (crop_sx.value, crop_sy.value, crop_sz.value)
        _sync_crop_sliders()
        _crop_commit()
        print(f"  + crop sliders adapted to scene extent {extent:.3f}")

    # Apply adaptive ranges to any scene already loaded at startup.
    if _scene_xyz[0] is not None:
        _update_crop_ranges(_scene_xyz[0])

    # Show URL
    gs_info = ""
    if _gs_state[0] is not None:
        gs_info = f"  {_gs_state[0][0].get_xyz.shape[0]:,} gaussians  |  SH degree {_gs_state[0][1]}"
    print(f"\n{'=' * 60}")
    print(f"  Viewer ready: http://{args.host}:{args.port}")
    if gs_info:
        print(gs_info)
    else:
        print("  Use the Scene Browser panel to select and load a scene")
    print("  Press Ctrl+C to exit")
    print(f"{'=' * 60}\n")

    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
    import time

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nExiting viewer.")


if __name__ == "__main__":
    main()
