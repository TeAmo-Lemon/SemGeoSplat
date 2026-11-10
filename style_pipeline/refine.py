"""① Refine: 剪枝 + 光度精修.

从 --start_ply 加载已训练高斯, 在 refine_iterations 步内交替进行光度微调与周期性剪枝.
本模块含剪枝调度/重要度打分/剪枝执行, 以及 refine 结束后的显存释放辅助函数
(相机 mask/image 张量、参数 .grad、Adam 动量、_exposure).
"""

import os

import torch
from tqdm import tqdm

from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene


def _resolve_start_ply(style):
    if style.start_ply and style.start_pointcloud:
        raise ValueError("Use only one of --start_ply or --start_pointcloud, not both")
    start_ply = style.start_ply or style.start_pointcloud
    if not start_ply:
        raise ValueError(
            "The refine stage requires --start_ply PATH/TO/point_cloud.ply"
        )
    if not os.path.isfile(start_ply):
        raise FileNotFoundError(f"Could not find --start_ply file: {start_ply}")
    return start_ply


def _resolve_refine_dir(start_ply):
    """refine 产物缓存目录: --start_ply 同树的 point_cloud/iteration_refine/.

    start_ply 形如 <root>/point_cloud/iteration_30000/point_cloud.ply, 取上两级
    (<root>/point_cloud/) 再拼 iteration_refine, 与 iteration_30000 同级. refine 后
    的 ply 与 DINO GT 缓存都写在此处, 跨风格化运行复用.
    """
    return os.path.join(os.path.dirname(os.path.dirname(start_ply)), "iteration_refine")


def _repair_loaded_ply_state(gaussians: GaussianModel, scene: Scene):
    n_points = gaussians.get_xyz.shape[0]
    gaussians.spatial_lr_scale = scene.cameras_extent
    gaussians.max_radii2D = torch.zeros(n_points, device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n_points, 1), device="cuda")
    gaussians.denom = torch.zeros((n_points, 1), device="cuda")
    gaussians.tmp_radii = None
    if not hasattr(gaussians, "pretrained_exposures"):
        gaussians.pretrained_exposures = None


def _select_importance_cameras(scene: Scene, importance_views):
    cameras = scene.getTrainCameras().copy()
    if (
        importance_views is None
        or importance_views < 0
        or importance_views >= len(cameras)
    ):
        return cameras
    return cameras[:importance_views]


def compute_importance_scores(
    dataset, gaussians: GaussianModel, scene: Scene, pipe, background, style
):
    """Accumulate visibility and opacity-weighted importance over training views, with optional volume weighting."""
    cameras = _select_importance_cameras(scene, style.importance_views)
    n_points = gaussians.get_xyz.shape[0]
    visibility_score = torch.zeros(n_points, device="cuda")
    area_score = torch.zeros(n_points, device="cuda")

    print(f"Computing importance over {len(cameras)} training views...")
    for viewpoint_cam in tqdm(cameras, desc="Importance", leave=False):
        with network_gui.lock:
            with torch.no_grad():
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    background,
                    use_trained_exp=dataset.train_test_exp,
                    no_densify_hook=True,
                    return_info=True,
                )
        radii = render_pkg["radii"].detach().reshape(-1)
        visible = radii > 0
        visibility_score += visible.float()
        image_area = float(viewpoint_cam.image_width * viewpoint_cam.image_height)
        area_score += (radii.clamp_min(0.0) ** 2) / max(image_area, 1.0)

    opacity = gaussians.get_opacity.detach().reshape(-1)
    important_score = area_score * opacity
    if torch.count_nonzero(important_score).item() == 0:
        important_score = visibility_score * opacity
    if torch.count_nonzero(important_score).item() == 0:
        important_score = opacity

    volume = torch.prod(gaussians.get_scaling.detach(), dim=1)
    sorted_volume = torch.sort(volume, descending=True).values
    volume_idx = min(int(len(sorted_volume) * 0.9), len(sorted_volume) - 1)
    volume_90 = sorted_volume[volume_idx].clamp_min(1e-12)
    volume_weight = torch.pow((volume / volume_90).clamp_min(1e-12), style.v_pow)
    volume_important_score = important_score * volume_weight

    if style.prune_metric == "important_score":
        return important_score
    if style.prune_metric == "visibility":
        return visibility_score * opacity
    if style.prune_metric == "opacity":
        return opacity
    if style.prune_metric == "v_important_score":
        return volume_important_score
    raise ValueError(f"Unsupported prune metric: {style.prune_metric}")


def prune_for_refine(
    dataset,
    gaussians: GaussianModel,
    scene: Scene,
    pipe,
    background,
    style,
    iteration,
    prune_percent,
):
    """Prune the lowest-importance Gaussians while retaining at least min_gaussians."""
    before = gaussians.get_xyz.shape[0]
    if prune_percent <= 0:
        print(
            f"Skipping pruning at iteration {iteration} because prune_percent={prune_percent}"
        )
        return

    score = compute_importance_scores(
        dataset, gaussians, scene, pipe, background, style
    )
    with network_gui.lock:
        pruned = gaussians.prune_gaussians(
            prune_percent, score, min_keep=style.min_gaussians
        )
    after = gaussians.get_xyz.shape[0]
    print(
        f"[ITER {iteration}] Refine pruning: metric={style.prune_metric}, "
        f"percent={prune_percent}, pruned={pruned}, "
        f"gaussians={before}->{after}"
    )


def _release_camera_gpu_tensors(scene: Scene):
    """训练结束后把每张相机常驻在 GPU 上的图像/mask/depth 释放掉,
    只保留高斯球和相机位姿等小张量 (viewer/后续 stage 仍可用)."""
    freed_bytes = 0
    for cam_dict in (scene.train_cameras, scene.test_cameras):
        for scale_cams in cam_dict.values():
            for cam in scale_cams:
                for attr in ("alpha_mask", "depth_mask", "invdepthmap"):
                    t = getattr(cam, attr, None)
                    if isinstance(t, torch.Tensor) and t.is_cuda:
                        freed_bytes += t.numel() * t.element_size()
                        setattr(cam, attr, None)
    torch.cuda.empty_cache()
    return freed_bytes


def _release_camera_images_tensors(scene: Scene):
    """训练结束后把每张相机常驻在 GPU 上的图像释放掉"""
    freed_bytes = 0
    for cam_dict in (scene.train_cameras, scene.test_cameras):
        for scale_cams in cam_dict.values():
            for cam in scale_cams:
                t = getattr(cam, "original_image", None)
                if isinstance(t, torch.Tensor) and t.is_cuda:
                    freed_bytes += t.numel() * t.element_size()
                    setattr(cam, "original_image", None)
    torch.cuda.empty_cache()
    return freed_bytes


def _release_training_grads(gaussians: GaussianModel):
    """清掉最后一 iter 遗留的 .grad. 只处理参数本身的 .grad 张量,
    optimizer 内部的动量 state 由 _release_optimizer_state 单独处理."""
    freed_bytes = 0
    params = [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
        getattr(gaussians, "_exposure", None),
    ]
    for p in params:
        if p is not None and p.grad is not None:
            freed_bytes += p.grad.numel() * p.grad.element_size()
            p.grad = None
    torch.cuda.empty_cache()
    return freed_bytes


def _release_optimizer_state(gaussians: GaussianModel):
    """释放 optimizer / exposure_optimizer 的 Adam 动量 state
    (每个可训练参数 2 份和参数同尺寸的 exp_avg + exp_avg_sq).
    对 N ~ 1M 高斯量级约 1 GB+, 是训练结束后残留显存里最大的一块.
    后续 stage 会重新调 training_setup 建 optimizer, 旧 state 无需保留."""
    freed_bytes = 0
    for opt in (
        getattr(gaussians, "optimizer", None),
        getattr(gaussians, "exposure_optimizer", None),
    ):
        if opt is None:
            continue
        for state in opt.state.values():
            for v in state.values():
                if isinstance(v, torch.Tensor) and v.is_cuda:
                    freed_bytes += v.numel() * v.element_size()
    gaussians.optimizer = None
    gaussians.exposure_optimizer = None
    torch.cuda.empty_cache()
    return freed_bytes


def _release_exposure(gaussians: GaussianModel):
    """释放训练用的 _exposure 参数 (每张训练图一个 3x4 仿射颜色矩阵).
    仅在 render(use_trained_exp=True) 路径下被读取; refine 结束后的
    viewer / 后续 stage 走 use_trained_exp=False 默认路径, 不再需要它.
    应在 _release_optimizer_state 之后调用, 确保 exposure_optimizer
    对该参数的引用已被解除."""
    freed_bytes = 0
    t = getattr(gaussians, "_exposure", None)
    if isinstance(t, torch.Tensor) and t.is_cuda:
        freed_bytes += t.numel() * t.element_size()
    gaussians._exposure = None
    torch.cuda.empty_cache()
    return freed_bytes


def _prune_schedule(style, prune_iterations):
    """Return sorted pruning iterations from an explicit schedule or the configured interval."""
    if prune_iterations:
        return sorted(set(prune_iterations))
    if style.prune_interval <= 0:
        return []
    return list(
        range(style.prune_from_iter, style.refine_iterations + 1, style.prune_interval)
    )


def _prune_percent_for_schedule_index(style, schedule_index):
    """Apply prune_decay to the initial pruning fraction for the given zero-based round."""
    return style.prune_percent * (style.prune_decay**schedule_index)
