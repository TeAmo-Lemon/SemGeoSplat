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

from __future__ import annotations

import glob
import math
import os
from os import makedirs
from argparse import ArgumentParser, ArgumentTypeError
from arguments import ModelParams, PipelineParams, get_combined_args
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def _gif_fps(value):
    fps = float(value)
    if not math.isfinite(fps) or not 0 < fps <= 100:
        raise ArgumentTypeError("GIF FPS must be finite and in (0, 100].")
    return fps


def _save_gif(image_dir, fps, output_path, max_size=960, colors=128, frame_count=None):
    """Combine saved views in filename order into a looping GIF at output_path."""
    from PIL import Image

    paths = (
        sorted(glob.glob(os.path.join(image_dir, "*.png")))
        if frame_count is None
        else [
            os.path.join(image_dir, f"{index:05d}.png") for index in range(frame_count)
        ]
    )
    if not paths:
        print(f"[gif] no frames in {image_dir}; skipping")
        return

    frames = []
    try:
        for path in paths:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                try:
                    if max_size > 0:
                        rgb.thumbnail(
                            (max_size, max_size),
                            getattr(Image, "Resampling", Image).LANCZOS,
                        )
                    frames.append(rgb.quantize(colors=colors))
                finally:
                    rgb.close()
        # GIF frame delays have 10 ms precision.
        duration = max(10, round(100 / fps) * 10)
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
            optimize=True,
        )
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(
            f"[gif] saved {output_path} ({len(frames)} views, {duration} ms/frame, "
            f"{frames[0].width}x{frames[0].height}, {size_mb:.2f} MiB)"
        )
    finally:
        for frame in frames:
            frame.close()


def _renders_exist(render_dir: str) -> bool:
    """Return True if *render_dir* exists and contains PNG files."""
    if not os.path.isdir(render_dir):
        return False
    return len(glob.glob(os.path.join(render_dir, "*.png"))) > 0


def _export_cached_gifs(args):
    """Return True after exporting cached PNGs, or False when rendering is needed."""
    from pathlib import Path

    if (
        not args.save_gif
        or args.force
        or args.write_gt
        or args.save_depth
        or args.save_cameras
    ):
        return False

    model_path = Path(args.model_path).resolve()
    splits = []
    if not args.skip_train:
        splits.append("train")
    if args.gif_test and not args.skip_test:
        splits.append("test")
    if not splits:
        return False

    iteration = args.iteration
    if iteration == -1:
        iterations = [
            int(path.name[5:])
            for split in splits
            for path in (model_path / split).glob("ours_*")
            if path.is_dir()
            and path.name[5:].isdigit()
            and _renders_exist(str(path / "renders"))
        ]
        if not iterations:
            return False
        iteration = max(iterations)

    exports = []
    for split in splits:
        output_dir = model_path / split / f"ours_{iteration}"
        exports.append((output_dir / "renders", output_dir / f"{model_path.name}.gif"))
        if args.gif_regions:
            exports.append(
                (output_dir / "regions", output_dir / f"{model_path.name}_regions.gif")
            )
    for image_dir, _ in exports:
        if not _renders_exist(str(image_dir)):
            return False
    for image_dir, output_path in exports:
        _save_gif(
            str(image_dir),
            args.gif_fps,
            str(output_path),
            max_size=args.gif_max_size,
            colors=args.gif_colors,
        )
    return True


def _files_exist(directory: str, suffix: str, expected_count: int) -> bool:
    """Return whether an auxiliary render directory has one file per view."""
    return (
        os.path.isdir(directory)
        and len(glob.glob(os.path.join(directory, f"*{suffix}"))) >= expected_count
    )


def _build_region_colormap(labels: torch.Tensor) -> torch.Tensor:
    """Build (K, 3) HSV colormap for *labels* (unique non-negative values)."""
    import colorsys
    import torch

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


def _load_region_colors(model_path, iteration, device) -> torch.Tensor | None:
    """Load region labels and return per-gaussian (N, 3) RGB colors on *device*.

    优先使用 primitive_labels (⑤ FGW 分配结果, 来自 region_primitives.pt),
    回退到 cluster_labels (④ 聚类结果, 来自 regions.pt).
    Returns None if both sidecars are missing/unusable."""
    from pathlib import Path
    import torch

    regions_path = (
        Path(model_path) / "point_cloud" / f"iteration_{iteration}" / "regions.pt"
    )
    rp_path = (
        Path(model_path)
        / "point_cloud"
        / f"iteration_{iteration}"
        / "region_primitives.pt"
    )

    labels = None
    label_source = None

    if rp_path.is_file():
        rp_data = torch.load(str(rp_path), map_location="cpu", weights_only=True)
        primitive_labels = rp_data.get("primitive_labels")
        if primitive_labels is not None:
            labels = primitive_labels
            label_source = f"FGW primitive assignment [{int(primitive_labels.max()) + 1} primitives] from {rp_path}"

    if labels is None and regions_path.is_file():
        regions_data = torch.load(
            str(regions_path), map_location="cpu", weights_only=True
        )
        cluster_labels = regions_data.get("cluster_labels")
        if cluster_labels is not None:
            labels = cluster_labels
            label_source = f"structural clusters [{int(cluster_labels.max()) + 1} regions] from {regions_path}"

    if labels is None:
        print(
            "[region] neither region_primitives.pt nor regions.pt found; skipping region renders"
        )
        return None

    labels = labels.to(device)
    cmap = _build_region_colormap(labels).to(device)
    K = cmap.shape[0]
    colors = torch.full((labels.shape[0], 3), 0.12, device=device)
    valid = labels >= 0
    if valid.any() and K > 0:
        colors[valid] = cmap[labels[valid].clamp(min=0) % K]
    print(
        f"[region] loaded {label_source}, {int(valid.sum()):,}/{labels.shape[0]:,} gaussians"
    )
    return colors


def render_set(
    model_path,
    name,
    iteration,
    views,
    gaussians,
    pipeline,
    background,
    train_test_exp,
    region_colors=None,
    write_gt=True,
    force=False,
    save_depth=False,
    save_cameras=False,
    save_gif=False,
    gif_fps=10.0,
    gif_regions=False,
    gif_max_size=960,
    gif_colors=128,
    override_colors=None,
    gt_only=False,
):
    from tqdm import tqdm
    import numpy as np
    import torchvision

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    region_path = os.path.join(model_path, name, "ours_{}".format(iteration), "regions")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    camera_path = os.path.join(model_path, name, "ours_{}".format(iteration), "cameras")

    # Repair camera-aligned GT without touching cached RGB renders.  This keeps
    # previously computed temporal metrics valid when only an old GT cache is stale.
    if gt_only:
        makedirs(gts_path, exist_ok=True)
        for idx, view in enumerate(tqdm(views, desc=f"Writing {name} GT")):
            gt = view.original_image[0:3, :, :]
            if train_test_exp:
                gt = gt[..., gt.shape[-1] // 2 :]
            torchvision.utils.save_image(
                gt, os.path.join(gts_path, "{0:05d}.png".format(idx))
            )
        return

    from gaussian_renderer import render

    def export_gifs():
        if save_gif and len(views) > 0:
            model_name = os.path.basename(os.path.abspath(model_path))
            output_dir = os.path.dirname(render_path)
            _save_gif(
                render_path,
                gif_fps,
                os.path.join(output_dir, f"{model_name}.gif"),
                max_size=gif_max_size,
                colors=gif_colors,
            )
            if gif_regions and region_colors is not None:
                _save_gif(
                    region_path,
                    gif_fps,
                    os.path.join(output_dir, f"{model_name}_regions.gif"),
                    max_size=gif_max_size,
                    colors=gif_colors,
                )

    normal_exist = False if force else _renders_exist(render_path)
    regions_exist = (
        (False if force else _renders_exist(region_path))
        if region_colors is not None
        else True
    )
    depths_exist = (
        (False if force else _files_exist(depth_path, ".npy", len(views)))
        if save_depth
        else True
    )
    cameras_exist = (
        (False if force else _files_exist(camera_path, ".npz", len(views)))
        if save_cameras
        else True
    )

    # Everything already done — nothing to do
    if normal_exist and regions_exist and depths_exist and cameras_exist:
        print(f"[region] {name} renders and regions already exist, skipping")
        export_gifs()
        return

    if not normal_exist:
        makedirs(render_path, exist_ok=True)
        if write_gt:
            makedirs(gts_path, exist_ok=True)

    if region_colors is not None and not regions_exist:
        makedirs(region_path, exist_ok=True)
    if save_depth and not depths_exist:
        makedirs(depth_path, exist_ok=True)
    if save_cameras and not cameras_exist:
        makedirs(camera_path, exist_ok=True)

    desc_parts = []
    if not normal_exist and region_colors is not None and not regions_exist:
        desc_parts.append(f"Rendering {name} + regions")
    elif not normal_exist:
        desc_parts.append(f"Rendering {name}")
    elif not regions_exist:
        desc_parts.append(f"Rendering {name} regions only")
    desc = " ".join(desc_parts)

    for idx, view in enumerate(tqdm(views, desc=desc)):
        if not normal_exist or (save_depth and not depths_exist):
            render_output = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=train_test_exp,
                override_color=override_colors,
            )
            rendering = render_output["render"]
            gt = view.original_image[0:3, :, :] if write_gt else None

            if train_test_exp:
                rendering = rendering[..., rendering.shape[-1] // 2 :]
                if gt is not None:
                    gt = gt[..., gt.shape[-1] // 2 :]

            if not normal_exist:
                torchvision.utils.save_image(
                    rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png")
                )
                if write_gt:
                    torchvision.utils.save_image(
                        gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png")
                    )

            if save_depth and not depths_exist:
                depth = render_output["depth_z"]
                if train_test_exp:
                    depth = depth[..., depth.shape[-1] // 2 :]
                np.save(
                    os.path.join(depth_path, "{0:05d}.npy".format(idx)),
                    depth.squeeze(0).cpu().numpy(),
                )

        if save_cameras and not cameras_exist:
            # Camera.world_view_transform is stored transposed for PyTorch3D-style
            # row-vector transforms.  The visualizer consumes conventional W2C.
            W, H = int(view.image_width), int(view.image_height)
            fx = W / (2.0 * np.tan(view.FoVx * 0.5))
            fy = H / (2.0 * np.tan(view.FoVy * 0.5))
            K = np.array(
                [[fx, 0.0, W * 0.5], [0.0, fy, H * 0.5], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            if train_test_exp:
                # render.py retains the right half in this mode; express K in the
                # cropped image coordinate system.
                K[0, 2] -= W * 0.5
                W //= 2
            w2c = view.world_view_transform.transpose(0, 1).cpu().numpy()
            np.savez_compressed(
                os.path.join(camera_path, "{0:05d}.npz".format(idx)),
                K=K,
                w2c=w2c,
                width=np.int32(W),
                height=np.int32(H),
                image_name=view.image_name,
            )

        if region_colors is not None and not regions_exist:
            region_render = render(
                view,
                gaussians,
                pipeline,
                background,
                use_trained_exp=train_test_exp,
                override_color=region_colors,
            )["render"]
            if train_test_exp:
                region_render = region_render[..., region_render.shape[-1] // 2 :]
            torchvision.utils.save_image(
                region_render, os.path.join(region_path, "{0:05d}".format(idx) + ".png")
            )

    export_gifs()


def render_trajectory(args, pipeline):
    """Render a saved viewer trajectory without constructing a dataset Scene."""
    import hashlib
    import json
    from pathlib import Path
    from semgeo_utils.trajectory_utils import (
        load_trajectory,
        restore_viewer_c2w,
        trajectory_cache_complete,
    )

    trajectory_file = Path(args.camera_path).expanduser().resolve()
    trajectory = load_trajectory(trajectory_file)
    model_path = Path(args.model_path).expanduser().resolve()
    iteration = args.iteration
    if iteration == -1:
        candidates = [
            int(path.name[10:])
            for path in (model_path / "point_cloud").glob("iteration_*")
            if path.name[10:].isdigit() and (path / "point_cloud.ply").is_file()
        ]
        if not candidates:
            raise ValueError(f"No trained point_cloud.ply found in {model_path}.")
        iteration = max(candidates)
    ply_path = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    ply_stat = ply_path.stat()
    name = os.path.join("trajectory", trajectory_file.stem)
    output_dir = model_path / name / f"ours_{iteration}"
    frame_count = len(trajectory["camera_path"])
    fps = (
        args.gif_fps
        if args.gif_fps is not None
        else min(100.0, float(trajectory.get("fps", 30)))
    )
    signature = {
        "version": 1,
        "trajectory": hashlib.sha256(
            json.dumps(trajectory, sort_keys=True).encode()
        ).hexdigest(),
        "ply_size": ply_stat.st_size,
        "ply_mtime_ns": ply_stat.st_mtime_ns,
        "white_background": bool(args.white_background),
        "pipeline": vars(pipeline),
    }
    colors_path = (
        Path(args.override_colors).expanduser().resolve()
        if args.override_colors
        else None
    )
    if colors_path is not None:
        colors_stat = colors_path.stat()
        signature["override_colors"] = [
            str(colors_path),
            colors_stat.st_size,
            colors_stat.st_mtime_ns,
        ]
    outputs = [("renders", ".png")]
    if args.render_regions or args.gif_regions:
        outputs.append(("regions", ".png"))
        signature["region_sidecars"] = {
            filename: [path.stat().st_size, path.stat().st_mtime_ns]
            if path.is_file()
            else None
            for filename in ("regions.pt", "region_primitives.pt")
            for path in [ply_path.parent / filename]
        }
    if args.save_depth:
        outputs.append(("depths", ".npy"))
    if args.save_cameras:
        outputs.append(("cameras", ".npz"))
    signature["outputs"] = [list(output) for output in outputs]

    cached = not args.force and trajectory_cache_complete(
        output_dir, signature, frame_count, outputs
    )
    if cached:
        print(f"[trajectory] reusing {frame_count} rendered frames in {output_dir}")
    else:
        import torch
        from plyfile import PlyData
        from scene.cameras import MiniCam
        from gaussian_renderer import GaussianModel
        from semgeo_utils.general_utils import safe_state

        safe_state(args.quiet)
        extra_features = sum(
            prop.name.startswith("f_rest_")
            for prop in PlyData.read(str(ply_path)).elements[0].properties
        )
        gaussians = GaussianModel(int(math.sqrt(extra_features / 3 + 1)) - 1)
        gaussians.load_ply(str(ply_path))
        override_colors = None
        if colors_path is not None:
            import numpy as np

            colors = np.load(colors_path, allow_pickle=False)
            if (
                colors.shape != (gaussians.get_xyz.shape[0], 3)
                or not np.isfinite(colors).all()
            ):
                raise ValueError(
                    "--override-colors must contain finite RGB values with shape (N, 3)."
                )
            override_colors = torch.as_tensor(
                colors, dtype=torch.float32, device="cuda"
            )
        background = torch.tensor(
            [1, 1, 1] if args.white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        region_colors = None
        if args.render_regions or args.gif_regions:
            region_colors = _load_region_colors(
                str(model_path), iteration, background.device
            )
            if region_colors is None:
                raise ValueError(
                    "Requested trajectory regions, but no usable region labels were found."
                )
        width, height = trajectory["render_width"], trajectory["render_height"]
        views = []
        for index, frame in enumerate(trajectory["camera_path"]):
            c2w = torch.tensor(
                restore_viewer_c2w(frame["camera_to_world"]),
                dtype=torch.float32,
                device="cuda",
            )
            world_view = torch.linalg.inv(c2w).transpose(0, 1).contiguous()
            fovy = math.radians(frame["fov"])
            fovx = 2 * math.atan(math.tan(fovy / 2) * width / height)
            view = MiniCam(
                width,
                height,
                fovy,
                fovx,
                0.01,
                100.0,
                world_view,
                torch.eye(4, device="cuda"),
            )
            view.image_name = f"{index:05d}"
            views.append(view)

        # Invalidate before writing, so interrupted renders cannot reuse stale frames.
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "trajectory_render.json").unlink(missing_ok=True)
        with torch.no_grad():
            render_set(
                str(model_path),
                name,
                iteration,
                views,
                gaussians,
                pipeline,
                background,
                False,
                region_colors=region_colors,
                write_gt=False,
                force=True,
                save_depth=args.save_depth,
                save_cameras=args.save_cameras,
                override_colors=override_colors,
            )
        # A shortened trajectory must not retain extra frames from the previous run.
        for folder, suffix in outputs:
            for path in (output_dir / folder).glob(f"*{suffix}"):
                if path.stem.isdigit() and int(path.stem) >= frame_count:
                    path.unlink()
        (output_dir / "trajectory_render.json").write_text(
            json.dumps(signature, indent=2)
        )
        print(f"[trajectory] saved {frame_count} frames to {output_dir / 'renders'}")

    if args.save_gif:
        _save_gif(
            str(output_dir / "renders"),
            fps,
            str(output_dir / f"{model_path.name}.gif"),
            max_size=args.gif_max_size,
            colors=args.gif_colors,
            frame_count=frame_count,
        )
        if args.gif_regions:
            _save_gif(
                str(output_dir / "regions"),
                fps,
                str(output_dir / f"{model_path.name}_regions.gif"),
                max_size=args.gif_max_size,
                colors=args.gif_colors,
                frame_count=frame_count,
            )


def render_sets(
    dataset: ModelParams,
    iteration: int,
    pipeline: PipelineParams,
    skip_train: bool,
    skip_test: bool,
    render_regions: bool = False,
    write_gt: bool = True,
    force: bool = False,
    save_depth: bool = False,
    save_cameras: bool = False,
    save_gif: bool = False,
    gif_fps: float = 10.0,
    gif_test: bool = False,
    gif_regions: bool = False,
    gif_max_size: int = 960,
    gif_colors: int = 128,
    gt_only: bool = False,
):
    import torch
    from scene import Scene
    from gaussian_renderer import GaussianModel

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        region_colors = None
        if render_regions:
            region_colors = _load_region_colors(
                dataset.model_path, scene.loaded_iter, device=background.device
            )

        if not skip_train:
            render_set(
                dataset.model_path,
                "train",
                scene.loaded_iter,
                scene.getTrainCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                region_colors=region_colors,
                write_gt=write_gt,
                force=force,
                save_depth=save_depth,
                save_cameras=save_cameras,
                save_gif=save_gif,
                gif_fps=gif_fps,
                gif_regions=gif_regions,
                gif_max_size=gif_max_size,
                gif_colors=gif_colors,
                gt_only=gt_only,
            )

        if not skip_test:
            render_set(
                dataset.model_path,
                "test",
                scene.loaded_iter,
                scene.getTestCameras(),
                gaussians,
                pipeline,
                background,
                dataset.train_test_exp,
                region_colors=region_colors,
                write_gt=write_gt,
                force=force,
                save_depth=save_depth,
                save_cameras=save_cameras,
                save_gif=save_gif and gif_test,
                gif_fps=gif_fps,
                gif_regions=gif_regions,
                gif_max_size=gif_max_size,
                gif_colors=gif_colors,
                gt_only=gt_only,
            )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument(
        "--camera-path",
        type=str,
        help="Render a JSON path from viewer Save Trajectory instead of train/test cameras.",
    )
    parser.add_argument(
        "--override-colors",
        type=str,
        help="Trajectory only: .npy array of (N, 3) RGB colors, e.g. StyleGaussian decoder output.",
    )
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--render-regions",
        action="store_true",
        help="Additionally render each gaussian coloured by its FGW primitive "
        "assignment (region_primitives.pt) or structural region "
        "(regions.pt, fallback).",
    )
    parser.add_argument(
        "--write-gt",
        action="store_true",
        help="Write ground-truth images into the gt/ folders "
        "(train and test). Off by default to save disk space.",
    )
    parser.add_argument(
        "--gt-only",
        action="store_true",
        help="Regenerate camera-aligned gt/ images without modifying renders/.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-render and overwrite PNGs even when render folders exist.",
    )
    parser.add_argument(
        "--save-depth",
        action="store_true",
        help="Save metric z-depth maps as depths/00000.npy for geometric evaluation.",
    )
    parser.add_argument(
        "--save-cameras",
        action="store_true",
        help="Save per-view K and W2C matrices as cameras/00000.npz for geometric evaluation.",
    )
    parser.add_argument(
        "--save-gif",
        action="store_true",
        help="Export train/ours_<iteration>/<model_name>.gif. When PNGs exist, "
        "export directly without loading the scene; otherwise render first.",
    )
    parser.add_argument(
        "--gif-test",
        action="store_true",
        help="With --save-gif, also export GIFs for test views (off by default).",
    )
    parser.add_argument(
        "--gif-regions",
        action="store_true",
        help="Also export <model_name>_regions.gif (off by default); requires "
        "cached region PNGs or --render-regions when rendering.",
    )
    parser.add_argument(
        "--gif-fps",
        default=None,
        type=_gif_fps,
        help="GIF frames per second in (0, 100] (default: trajectory FPS, otherwise 10; "
        "rounded to 10 ms delays).",
    )
    parser.add_argument(
        "--gif-max-size",
        default=960,
        type=int,
        help="Maximum GIF frame width/height, preserving aspect ratio "
        "(default: 960; 0 keeps original size).",
    )
    parser.add_argument(
        "--gif-colors",
        default=128,
        type=int,
        help="Maximum colors per GIF frame, from 2 to 256 (default: 128).",
    )
    args = get_combined_args(parser)
    if args.gif_max_size < 0:
        parser.error("--gif-max-size must be >= 0")
    if not 2 <= args.gif_colors <= 256:
        parser.error("--gif-colors must be between 2 and 256")
    if args.override_colors and not args.camera_path:
        parser.error("--override-colors requires --camera-path")
    if args.gt_only:
        args.write_gt = True
        if (
            args.camera_path
            or args.render_regions
            or args.save_depth
            or args.save_cameras
        ):
            parser.error(
                "--gt-only cannot be combined with trajectory, region, depth, or camera output"
            )
    if args.camera_path:
        if args.write_gt or args.skip_train or args.skip_test or args.gif_test:
            parser.error(
                "--camera-path uses custom views and cannot be combined with "
                "--write-gt, --skip_train, --skip_test or --gif-test"
            )
        try:
            render_trajectory(args, pipeline.extract(args))
        except (OSError, ValueError) as error:
            parser.error(str(error))
        raise SystemExit(0)
    if args.gif_fps is None:
        args.gif_fps = 10.0
    if _export_cached_gifs(args):
        raise SystemExit(0)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    from semgeo_utils.general_utils import safe_state

    safe_state(args.quiet)

    render_sets(
        model.extract(args),
        args.iteration,
        pipeline.extract(args),
        args.skip_train,
        args.skip_test,
        render_regions=args.render_regions,
        write_gt=args.write_gt,
        force=args.force,
        save_depth=args.save_depth,
        save_cameras=args.save_cameras,
        save_gif=args.save_gif,
        gif_fps=args.gif_fps,
        gif_test=args.gif_test,
        gif_regions=args.gif_regions,
        gif_max_size=args.gif_max_size,
        gif_colors=args.gif_colors,
        gt_only=args.gt_only,
    )
