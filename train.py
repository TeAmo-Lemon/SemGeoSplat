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
"""Train a 3D Gaussian scene with gsplat densification."""

import os


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
from random import randint
from semgeo_utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from semgeo_utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from semgeo_utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

from gsplat.strategy import DefaultStrategy


#  Parameter / optimizer bridge for DefaultStrategy


def _build_gauss_params(gaussians: GaussianModel):
    """Return a dict of nn.Parameter references keyed as DefaultStrategy expects."""
    return {
        "means": gaussians._xyz,
        "scales": gaussians._scaling,
        "quats": gaussians._rotation,
        "opacities": gaussians._opacity,
        "features_dc": gaussians._features_dc,
        "features_rest": gaussians._features_rest,
    }


def _sync_params_to_gaussians(gaussians: GaussianModel, gauss_params: dict):
    """Copy Parameter references back to GaussianModel after strategy ops."""
    gaussians._xyz = gauss_params["means"]
    gaussians._scaling = gauss_params["scales"]
    gaussians._rotation = gauss_params["quats"]
    gaussians._opacity = gauss_params["opacities"]
    gaussians._features_dc = gauss_params["features_dc"]
    gaussians._features_rest = gauss_params["features_rest"]
    # Keep bookkeeping tensors in sync (for capture / checkpoint)
    n = gauss_params["means"].shape[0]
    if gaussians.max_radii2D.shape[0] != n:
        gaussians.max_radii2D = torch.zeros(n, device="cuda")
    if gaussians.xyz_gradient_accum.shape[0] != n:
        gaussians.xyz_gradient_accum = torch.zeros((n, 1), device="cuda")
    if gaussians.denom.shape[0] != n:
        gaussians.denom = torch.zeros((n, 1), device="cuda")


def _build_per_param_optimizers(gauss_params: dict, opt, spatial_lr_scale: float):
    """Create one Adam optimizer per gaussian parameter group.

    Learning rates match OptimizationParams defaults.
    """
    return {
        "means": torch.optim.Adam(
            [gauss_params["means"]],
            lr=opt.position_lr_init * spatial_lr_scale,
            eps=1e-15,
        ),
        "features_dc": torch.optim.Adam(
            [gauss_params["features_dc"]], lr=opt.feature_lr, eps=1e-15
        ),
        "features_rest": torch.optim.Adam(
            [gauss_params["features_rest"]], lr=opt.feature_lr / 20.0, eps=1e-15
        ),
        "opacities": torch.optim.Adam(
            [gauss_params["opacities"]], lr=opt.opacity_lr, eps=1e-15
        ),
        "scales": torch.optim.Adam(
            [gauss_params["scales"]], lr=opt.scaling_lr, eps=1e-15
        ),
        "quats": torch.optim.Adam(
            [gauss_params["quats"]], lr=opt.rotation_lr, eps=1e-15
        ),
    }


#  Training


def training(
    dataset,
    opt,
    pipe,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    debug_from,
):

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    # ---- Build params dict & per-parameter optimizers -------------------
    gauss_params = _build_gauss_params(gaussians)
    spatial_lr_scale = scene.cameras_extent
    per_param_optimizers = _build_per_param_optimizers(
        gauss_params, opt, spatial_lr_scale
    )

    # Exposure optimizer (tensors created by gaussians.create_from_pcd())
    exposure_optimizer = torch.optim.Adam([gaussians._exposure])

    # ---- DefaultStrategy (nerfstudio splatfacto exact config) -----------
    num_train_images = len(scene.getTrainCameras())
    strategy = DefaultStrategy(
        prune_opa=0.005,  # cull_alpha_thresh
        grow_grad2d=opt.densify_grad_threshold,  # densify_grad_thresh 0.0002
        grow_scale3d=opt.percent_dense,  # densify_size_thresh 0.01
        grow_scale2d=0.05,  # split_screen_size
        prune_scale3d=0.1,  # cull_scale_thresh
        prune_scale2d=0.15,  # cull_screen_size
        refine_scale2d_stop_iter=0,  # stop_screen_size_at (disabled)
        refine_start_iter=opt.densify_from_iter,  # warmup_length 500
        refine_stop_iter=opt.densify_until_iter,  # stop_split_at 15000
        reset_every=opt.opacity_reset_interval,  # reset_alpha_every * refine_every = 3000
        refine_every=opt.densification_interval,  # 100
        pause_refine_after_reset=num_train_images + opt.densification_interval,
        absgrad=False,
        revised_opacity=False,
        verbose=True,
    )
    strategy_state = strategy.initialize_state(scene_scale=spatial_lr_scale)

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        gauss_params = _build_gauss_params(gaussians)
        per_param_optimizers = _build_per_param_optimizers(
            gauss_params, opt, spatial_lr_scale
        )
        strategy_state = strategy.initialize_state(scene_scale=spatial_lr_scale)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    network_gui.bind_scene(gaussians, pipe, background)

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    depth_l1_weight = get_expon_lr_func(
        opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations
    )

    # Means LR scheduler (exponential decay, same as train.py)
    means_lr_scheduler = get_expon_lr_func(
        lr_init=opt.position_lr_init * spatial_lr_scale,
        lr_final=opt.position_lr_final * spatial_lr_scale,
        lr_delay_mult=opt.position_lr_delay_mult,
        max_steps=opt.position_lr_max_steps,
    )
    exposure_lr_scheduler = get_expon_lr_func(
        opt.exposure_lr_init,
        opt.exposure_lr_final,
        lr_delay_steps=opt.exposure_lr_delay_steps,
        lr_delay_mult=opt.exposure_lr_delay_mult,
        max_steps=opt.iterations,
    )

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    def format_cuda_memory():
        # allocated = tensors actually live. reserved ≈ what nvidia-smi shows
        # for this process (allocated + cached-but-unused fragments). If
        # allocated stays flat while reserved climbs, the growth is
        # fragmentation, not a real leak.
        allocated = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        return f"{allocated:.0f}M/{reserved:.0f}M"

    progress_bar = tqdm(
        range(first_iter, opt.iterations), desc="Training progress (nerfstudio)"
    )
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        network_gui.update(iteration, num_train_rays_per_step=1)
        if not network_gui.disable_pause:
            network_gui.wait_if_paused(iteration)

        iter_start.record()

        # ---- LR scheduling ------------------------------------------------
        per_param_optimizers["means"].param_groups[0]["lr"] = means_lr_scheduler(
            iteration
        )
        for param_group in exposure_optimizer.param_groups:
            param_group["lr"] = exposure_lr_scheduler(iteration)

        # ---- SH degree ----------------------------------------------------
        if iteration % 1000 == 0:
            with network_gui.lock:
                gaussians.oneupSHdegree()

        # ---- Pick a random camera -----------------------------------------
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        viewpoint_indices.pop(rand_idx)

        # ---- Render -------------------------------------------------------
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        with network_gui.lock:
            # no_densify_hook=True  → DefaultStrategy handles NDC scaling
            # return_info=True      → need raw gsplat meta for strategy
            render_pkg = render(
                viewpoint_cam,
                gaussians,
                pipe,
                bg,
                use_trained_exp=dataset.train_test_exp,
                no_densify_hook=True,
                return_info=True,
            )
        image = render_pkg["render"]
        depth_inv = render_pkg.get("depth", None)
        gsplat_info = render_pkg["gsplat_info"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # ---- Loss ---------------------------------------------------------
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()
            Ll1depth_pure = torch.abs((depth_inv - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0.0

        # Scale regularization (prevents elongated/spiky gaussians).
        # Hinge loss: only gaussians whose aspect ratio max(s)/min(s)
        # exceeds max_gauss_ratio are penalised.
        if opt.use_scale_regularization and iteration % 10 == 0:
            scale_exp = torch.exp(gauss_params["scales"])  # [N, 3]
            ratio = scale_exp.amax(dim=-1) / (scale_exp.amin(dim=-1) + 1e-7)
            scale_reg = 0.1 * torch.clamp(ratio - opt.max_gauss_ratio, min=0.0).mean()
            loss += scale_reg

        # ---- Strategy pre-backward (retains means2d gradient) ------------
        with network_gui.lock:
            strategy.step_pre_backward(
                gauss_params,
                per_param_optimizers,
                strategy_state,
                iteration,
                gsplat_info,
            )

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # ---- Progress bar ---------------------------------------------
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * float(Ll1depth) + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "Loss": f"{ema_loss_for_log:.{7}f}",
                        "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                        "#GS": f"{gaussians.get_xyz.shape[0]}",
                        "Mem(A/R)": format_cuda_memory(),
                    }
                )
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # ---- Log and save ---------------------------------------------
            training_report(
                tb_writer,
                iteration,
                Ll1,
                loss,
                l1_loss,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                render,
                (pipe, background, 1.0, None, dataset.train_test_exp),
                dataset.train_test_exp,
            )
            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # ---- Strategy post-backward (accumulate grad → grow → prune → reset)
            # DefaultStrategy internally handles:
            #   - gradient accumulation in _update_state()
            #   - clone/split in _grow_gs()      (every refine_every steps)
            #   - opacity+scale prune in _prune_gs()
            #   - opacity reset via reset_opa()   (every reset_every steps)
            #   - pause_refine_after_reset: skips grow/prune for N steps
            #     after each reset so gaussians can recover opacity
            with network_gui.lock:
                strategy.step_post_backward(
                    gauss_params,
                    per_param_optimizers,
                    strategy_state,
                    iteration,
                    gsplat_info,
                )
                _sync_params_to_gaussians(gaussians, gauss_params)

            # After a refine step the gaussian count (and thus every
            # per-gaussian buffer + Adam moment) just changed size, leaving
            # freed-but-cached fragments in the allocator. Release them back so
            # reserved memory tracks the real footprint instead of ratcheting.
            if (
                opt.densify_from_iter < iteration < opt.densify_until_iter
                and iteration % opt.densification_interval == 0
            ):
                torch.cuda.empty_cache()

            # ---- Optimizer step -------------------------------------------
            if iteration < opt.iterations:
                with network_gui.lock:
                    exposure_optimizer.step()
                    exposure_optimizer.zero_grad(set_to_none=True)
                    for _name, opt_ in per_param_optimizers.items():
                        opt_.step()
                        opt_.zero_grad(set_to_none=True)

            if iteration in checkpoint_iterations:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save(
                    (gaussians.capture(), iteration),
                    scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                )

    network_gui.complete()


#  Output helpers  (same as train.py)


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(
    tb_writer,
    iteration,
    Ll1,
    loss,
    l1_loss,
    elapsed,
    testing_iterations,
    scene,
    renderFunc,
    renderArgs,
    train_test_exp,
):
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": scene.getTestCameras()},
            {
                "name": "train",
                "cameras": [
                    scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                    for idx in range(5, 30, 5)
                ],
            },
        )

        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"],
                        0.0,
                        1.0,
                    )
                    gt_image = torch.clamp(
                        viewpoint.original_image.to("cuda"), 0.0, 1.0
                    )
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2 :]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2 :]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(
                            config["name"]
                            + "_view_{}/render".format(viewpoint.image_name),
                            image[None],
                            global_step=iteration,
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(
                                config["name"]
                                + "_view_{}/ground_truth".format(viewpoint.image_name),
                                gt_image[None],
                                global_step=iteration,
                            )
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config["cameras"])
                l1_test /= len(config["cameras"])
                print(
                    "\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(
                        iteration, config["name"], l1_test, psnr_test
                    )
                )
                if tb_writer:
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - l1_loss", l1_test, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/loss_viewpoint - psnr", psnr_test, iteration
                    )

        if tb_writer:
            tb_writer.add_histogram(
                "scene/opacity_histogram", scene.gaussians.get_opacity, iteration
            )
            tb_writer.add_scalar(
                "total_points", scene.gaussians.get_xyz.shape[0], iteration
            )
        torch.cuda.empty_cache()


#  Entry point


if __name__ == "__main__":
    parser = ArgumentParser(
        description="3DGS training — nerfstudio splatfacto pipeline"
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument(
        "--test_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument(
        "--save_iterations", nargs="+", type=int, default=[7_000, 30_000]
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--disable_viewer_pause", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    safe_state(args.quiet)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    network_gui.disable_pause = args.disable_viewer_pause
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\nTraining complete.")
