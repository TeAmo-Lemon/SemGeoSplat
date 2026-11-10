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

import math
import torch
from gsplat import rasterization
from scene.gaussian_model import GaussianModel
from semgeo_utils.sh_utils import eval_sh


def render(
    viewpoint_camera,
    pc: GaussianModel,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    override_color=None,
    use_trained_exp=False,
    no_densify_hook=False,
    return_info=False,
    crop_box=None,
    **_unused,
):
    """
    Render the scene with gsplat 1.4.0.

    Background tensor (bg_color) must be on GPU.

    Returned dict keys mirror the original diff-gaussian-rasterization wrapper so
    the rest of the codebase (train.py / render.py / densification) is unchanged.
    """

    device = pc.get_xyz.device

    # ---- Camera intrinsics from FoV ----------------------------------------
    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    K = torch.tensor(
        [[fx, 0.0, W * 0.5], [0.0, fy, H * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)  # [1, 3, 3]

    # Camera.world_view_transform is stored already-transposed (see scene/cameras.py:
    # `torch.tensor(getWorld2View2(...)).transpose(0, 1)`). gsplat wants a standard
    # row-major world-to-camera, so transpose it back.
    viewmat = (
        viewpoint_camera.world_view_transform.transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(torch.float32)
    )

    # ---- Crop-box filter (eval-only) ---------------------------------------
    if crop_box is not None:
        crop_ids = crop_box.within(pc.get_xyz).squeeze()
        if crop_ids.sum() == 0:
            bg = bg_color.to(torch.float32).unsqueeze(0)
            blank = bg.unsqueeze(-1).unsqueeze(-1).expand(-1, 3, H, W).clone()
            depth_z = torch.zeros(1, H, W, device=device)
            inv_depth = torch.where(
                depth_z > 0, 1.0 / depth_z.clamp(min=1e-7), torch.zeros_like(depth_z)
            )
            return {
                "render": blank[0],
                "viewspace_points": torch.zeros(1, 0, 2, device=device),
                "visibility_filter": torch.zeros(0, device=device, dtype=torch.bool),
                "radii": torch.zeros(0, device=device),
                "depth": inv_depth,
                "depth_z": depth_z,
            }
    else:
        crop_ids = None

    means3D = pc.get_xyz if crop_ids is None else pc.get_xyz[crop_ids]
    opacity = (
        pc.get_opacity.squeeze(-1)
        if crop_ids is None
        else pc.get_opacity.squeeze(-1)[crop_ids]
    )
    scales = (
        pc.get_scaling * scaling_modifier
        if crop_ids is None
        else pc.get_scaling[crop_ids] * scaling_modifier
    )
    quats = pc.get_rotation if crop_ids is None else pc.get_rotation[crop_ids]

    # ---- Colors: SH coeffs (default) / python-evaluated SH / override ------
    sh_degree_arg = None
    if override_color is not None:
        colors = (
            override_color if crop_ids is None else override_color[crop_ids]
        )  # [N, 3], post-activation
    elif pipe.convert_SHs_python:
        if crop_ids is not None:
            xyz = pc.get_xyz[crop_ids]
            features = pc.get_features[crop_ids]
        else:
            xyz = pc.get_xyz
            features = pc.get_features
        shs_view = features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
        dir_pp = xyz - viewpoint_camera.camera_center.repeat(features.shape[0], 1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
        colors = torch.clamp_min(sh2rgb + 0.5, 0.0)
    else:
        colors = (
            pc.get_features if crop_ids is None else pc.get_features[crop_ids]
        )  # [N, K, 3]
        sh_degree_arg = pc.active_sh_degree

    bg = bg_color.to(torch.float32).unsqueeze(0)  # [1, 3]

    render_colors, _render_alphas, meta = rasterization(
        means=means3D,
        quats=quats,
        scales=scales,
        opacities=opacity,
        colors=colors,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=sh_degree_arg,
        backgrounds=bg,
        render_mode="RGB+ED",
        packed=False,
        rasterize_mode="antialiased" if pipe.antialiasing else "classic",
    )
    # render_colors: [1, H, W, 4] (RGB + expected z-depth)
    rgb = render_colors[0, ..., :3].permute(2, 0, 1)  # [3, H, W]
    depth_z = render_colors[0, ..., 3:4].permute(2, 0, 1)  # [1, H, W]

    # Original 3DGS rasterizer returned inverse depth; preserve that so the
    # depth-regularization loss in train.py keeps the same scale and zero-fill
    # for empty pixels.
    inv_depth = torch.where(
        depth_z > 0,
        1.0 / depth_z.clamp(min=1e-7),
        torch.zeros_like(depth_z),
    )

    # ---- Screen-space gradient sink for densification ----------------------
    # gsplat's `means2d` are in pixel units; the upstream 3DGS densification
    # threshold (default 0.0002) was tuned against gradients in NDC units. Hook
    # the gradient and rescale by (W/2, H/2) so the threshold keeps its meaning.
    #
    # When `no_densify_hook=True` (used by train.py), the
    # DefaultStrategy handles retain_grad + NDC scaling internally, so we skip
    # both steps to avoid double-scaling.
    means2d = meta["means2d"]  # [1, N, 2], requires_grad=True
    # When rendering for the viewer we run under `torch.no_grad()`, so means2d
    # has `requires_grad=False`; calling retain_grad / register_hook would
    # raise. Skip the densification gradient sink in that case.
    if means2d.requires_grad and not no_densify_hook:
        means2d.retain_grad()

        def _ndc_grad_scale_hook(grad):
            scaled = grad.clone()
            scaled[..., 0] = scaled[..., 0] * (W * 0.5)
            scaled[..., 1] = scaled[..., 1] * (H * 0.5)
            return scaled

        means2d.register_hook(_ndc_grad_scale_hook)

    radii = meta["radii"][0]  # [N]

    # Per-image exposure (training-time only).
    if use_trained_exp:
        exposure = pc.get_exposure_from_name(viewpoint_camera.image_name)
        rgb = (
            torch.matmul(rgb.permute(1, 2, 0), exposure[:3, :3]).permute(2, 0, 1)
            + exposure[:3, 3, None, None]
        )

    rgb = rgb.clamp(0, 1)

    result = {
        "render": rgb,
        "viewspace_points": means2d,
        "visibility_filter": (radii > 0).nonzero(),
        "radii": radii,
        "depth": inv_depth,
        # Metric camera-space z depth.  Keep the legacy inverse-depth key above
        # for training, but expose z depth for geometry-aware evaluation tools.
        "depth_z": depth_z,
    }
    if return_info:
        result["gsplat_info"] = meta
    return result


def render_features(
    viewpoint_camera, pc: GaussianModel, features, scaling_modifier=1.0, crop_box=None
):
    """Rasterize an arbitrary per-Gaussian feature vector (e.g. DINO) as flat colors.

    Mirrors `render`'s camera setup but: treats `features` [N, D] as post-activation
    colors (sh_degree=None, no SH transform), renders D channels with a zero
    background, and skips depth/exposure/densification. gsplat's alpha blending
    is channel-agnostic, so this is equivalent to colour rasterization over D
    channels. Returns the rendered feature map [D, H, W].
    """
    device = pc.get_xyz.device

    W = int(viewpoint_camera.image_width)
    H = int(viewpoint_camera.image_height)
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    fx = W / (2.0 * tanfovx)
    fy = H / (2.0 * tanfovy)
    K = torch.tensor(
        [[fx, 0.0, W * 0.5], [0.0, fy, H * 0.5], [0.0, 0.0, 1.0]],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)

    viewmat = (
        viewpoint_camera.world_view_transform.transpose(0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(torch.float32)
    )

    # ---- Crop-box filter (eval-only) ---------------------------------------
    if crop_box is not None:
        crop_ids = crop_box.within(pc.get_xyz).squeeze()
        if crop_ids.sum() == 0:
            return torch.zeros(features.shape[1], H, W, device=device)
    else:
        crop_ids = None

    feat_dim = features.shape[1]
    render_features, _render_alphas, _meta = rasterization(
        means=pc.get_xyz if crop_ids is None else pc.get_xyz[crop_ids],
        quats=pc.get_rotation if crop_ids is None else pc.get_rotation[crop_ids],
        scales=(pc.get_scaling * scaling_modifier)
        if crop_ids is None
        else pc.get_scaling[crop_ids] * scaling_modifier,
        opacities=pc.get_opacity.squeeze(-1)
        if crop_ids is None
        else pc.get_opacity.squeeze(-1)[crop_ids],
        colors=features if crop_ids is None else features[crop_ids],
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        sh_degree=None,
        backgrounds=torch.zeros(1, feat_dim, device=device, dtype=torch.float32),
        render_mode="RGB",
        packed=False,
        rasterize_mode="classic",
    )
    # render_features: [1, H, W, D] -> [D, H, W]
    return render_features[0].permute(2, 0, 1)
