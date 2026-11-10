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
"""Run refinement, semantic distillation, region matching and stylization."""

import os
from argparse import ArgumentParser
from random import randint


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONHASHSEED", "0")

import torch
from tqdm import tqdm

from arguments import (
    ModelParams,
    OptimizationParams,
    PipelineParams,
    StyleParams,
    get_combined_args,
)
from gaussian_renderer import network_gui, render
from scene import GaussianModel, Scene
from semgeo_utils.general_utils import get_expon_lr_func, safe_state
from semgeo_utils.loss_utils import l1_loss, ssim
from semgeo_utils.dataset_paths import validate_scene_path

try:
    from fused_ssim import fused_ssim

    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

from style_pipeline import (
    prepare_output_and_logger,
    training_report,
    _resolve_start_ply,
    _resolve_refine_dir,
    _repair_loaded_ply_state,
    prune_for_refine,
    _prune_schedule,
    _prune_percent_for_schedule_index,
    _release_camera_gpu_tensors,
    _release_camera_images_tensors,
    _release_training_grads,
    _release_optimizer_state,
    _release_exposure,
    distill_dino,
    _decompose_style_primitives,
    build_structure_aware_regions,
    align_regions_with_primitives,
    generate_style,
)


def training(
    dataset,
    opt,
    pipe,
    style,
    prune_iterations,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    debug_from,
    args,
):
    tb_writer = prepare_output_and_logger(args)
    dataset.model_path = args.model_path
    start_ply = _resolve_start_ply(style)

    refine_dir = _resolve_refine_dir(start_ply)
    refine_ply = os.path.join(refine_dir, "point_cloud.ply")
    refine_cached = os.path.isfile(refine_ply)

    if style.skip_refine:
        load_target = start_ply
        run_refine = False
        refine_msg = (
            "--skip_refine set, skipping refine stage "
            f"(ply loaded from {start_ply}, proceeding to next stage)"
        )
        if style.force_refine:
            print(
                "[refine] --skip_refine 与 --force_refine 同时设置, 以 --skip_refine 为准"
            )
    elif style.force_refine:
        load_target = start_ply
        run_refine = True
        refine_msg = (
            "--force_refine set, re-running refine "
            f"(ignoring cached ply at {refine_ply})"
            if refine_cached
            else "--force_refine set, running refine"
        )
    elif refine_cached:
        load_target = refine_ply
        run_refine = False
        refine_msg = (
            f"cached refine ply found at {refine_ply}, skipping refine "
            f"(remove {refine_dir} to re-run)"
        )
    else:
        load_target = start_ply
        run_refine = True
        refine_msg = f"no cached refine ply, running refine from {start_ply}"

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)

    print(f"Loading trained Gaussians from {load_target}")
    gaussians.load_ply(load_target, dataset.train_test_exp)
    _repair_loaded_ply_state(gaussians, scene)
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    network_gui.bind_scene(gaussians, pipe, background)

    freed_cam = freed_grad = freed_optim = freed_exposure = 0
    before_alloc = before_reserved = before_device = 0

    def _cuda_mem_snapshot():
        """Return allocated, reserved and device-wide CUDA memory in bytes."""
        free, total = torch.cuda.mem_get_info()
        return (
            torch.cuda.memory_allocated(),
            torch.cuda.memory_reserved(),
            total - free,
        )

    print(f"[refine] {refine_msg}")
    if run_refine:
        prune_schedule = _prune_schedule(style, prune_iterations)
        prune_schedule_set = set(prune_schedule)
        print(
            f"Refine prune schedule: {prune_schedule if prune_schedule else 'disabled'}"
        )

        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)

        depth_l1_weight = get_expon_lr_func(
            opt.depth_l1_weight_init,
            opt.depth_l1_weight_final,
            max_steps=style.refine_iterations,
        )

        viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_indices = list(range(len(viewpoint_stack)))
        ema_loss_for_log = 0.0
        ema_Ll1depth_for_log = 0.0

        def format_cuda_memory():
            allocated = torch.cuda.memory_allocated() / (1024**2)
            peak_allocated = torch.cuda.max_memory_allocated() / (1024**2)
            return f"{allocated:.0f}M/{peak_allocated:.0f}M"

        torch.cuda.reset_peak_memory_stats()
        progress_bar = tqdm(range(0, style.refine_iterations), desc="Refine progress")
        for iteration in range(1, style.refine_iterations + 1):
            network_gui.update(iteration, num_train_rays_per_step=1)
            if not network_gui.disable_pause:
                network_gui.wait_if_paused(iteration)

            iter_start.record()
            gaussians.update_learning_rate(iteration)

            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
                viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            viewpoint_indices.pop(rand_idx)

            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background
            with network_gui.lock:
                render_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    use_trained_exp=dataset.train_test_exp,
                    no_densify_hook=True,
                )
            image = render_pkg["render"]

            if viewpoint_cam.alpha_mask is not None:
                image *= viewpoint_cam.alpha_mask.cuda()

            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            if FUSED_SSIM_AVAILABLE:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            else:
                ssim_value = ssim(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (
                1.0 - ssim_value
            )

            Ll1depth_pure = 0.0
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                inv_depth = render_pkg["depth"]
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_mask = viewpoint_cam.depth_mask.cuda()
                Ll1depth_pure = torch.abs(
                    (inv_depth - mono_invdepth) * depth_mask
                ).mean()
                Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure
                loss += Ll1depth
                Ll1depth = Ll1depth.item()
            else:
                Ll1depth = 0.0

            loss.backward()
            iter_end.record()

            with torch.no_grad():
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                ema_Ll1depth_for_log = (
                    0.4 * float(Ll1depth) + 0.6 * ema_Ll1depth_for_log
                )

                # Apply the gradients before pruning.  Pruning rebuilds the
                # surviving Gaussian parameters as new nn.Parameter objects, so
                # pruning first would discard the gradients attached to the old
                # tensors.  Step on every Refine iteration, including the final
                # one, so refine_iterations is also the number of updates.
                with network_gui.lock:
                    gaussians.exposure_optimizer.step()
                    gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none=True)

                if iteration in prune_schedule_set:
                    schedule_index = prune_schedule.index(iteration)
                    prune_percent = _prune_percent_for_schedule_index(
                        style, schedule_index
                    )
                    prune_for_refine(
                        dataset,
                        gaussians,
                        scene,
                        pipe,
                        background,
                        style,
                        iteration,
                        prune_percent,
                    )

                if iteration % 10 == 0:
                    progress_bar.set_postfix(
                        {
                            "Loss": f"{ema_loss_for_log:.{7}f}",
                            "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}",
                            "#GS": f"{gaussians.get_xyz.shape[0]}",
                            "Mem": format_cuda_memory(),
                        }
                    )
                    progress_bar.update(10)
                if iteration == style.refine_iterations:
                    progress_bar.close()

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
                    print(
                        "\n[ITER {}] Saving refined Gaussians -> {}".format(
                            iteration, refine_ply
                        )
                    )

                    tmp_ply = refine_ply + ".tmp"
                    gaussians.save_ply(tmp_ply)
                    os.replace(tmp_ply, refine_ply)

                if iteration in checkpoint_iterations:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save(
                        (gaussians.capture(), iteration),
                        scene.model_path + "/chkpnt" + str(iteration) + ".pth",
                    )

    before_alloc, before_reserved, before_device = _cuda_mem_snapshot()
    freed_cam = _release_camera_gpu_tensors(scene)
    freed_grad = _release_training_grads(gaussians)
    freed_optim = _release_optimizer_state(gaussians)
    freed_exposure = _release_exposure(gaussians)

    try:
        distill_dino(dataset, gaussians, scene, style)
    except Exception as exc:
        import traceback

        print(
            f"[dino] DINO distillation failed ({exc}); refine output intact, continuing cleanup."
        )
        traceback.print_exc()

    _decompose_style_primitives(dataset, gaussians, style)

    freed_img = _release_camera_images_tensors(scene)
    after_alloc, after_reserved, after_device = _cuda_mem_snapshot()
    mb = lambda b: b / (1024**2)
    print(
        "[post-train] Released:\n"
        f"  camera masks  : {mb(freed_cam):.0f} MB\n"
        f"  camera images : {mb(freed_img):.0f} MB\n"
        f"  param .grad   : {mb(freed_grad):.0f} MB\n"
        f"  Adam state    : {mb(freed_optim):.0f} MB\n"
        f"  _exposure     : {mb(freed_exposure):.0f} MB\n"
        "[post-train] CUDA memory (before -> after):\n"
        f"  allocated (存活张量)      : {mb(before_alloc):.0f}M -> {mb(after_alloc):.0f}M\n"
        f"  reserved  (allocator 池)  : {mb(before_reserved):.0f}M -> {mb(after_reserved):.0f}M\n"
        f"  device    (整卡, ~nvidia-smi): {mb(before_device):.0f}M -> {mb(after_device):.0f}M\n"
        f"  其中不可释放的 CUDA context 等 ≈ device - reserved = {mb(after_device - after_reserved):.0f}M"
    )

    try:
        build_structure_aware_regions(dataset, gaussians, style)
    except Exception as exc:
        import traceback

        print(
            f"[region] ④ region/metric-space construction failed ({exc}); "
            f"refine+DINO+primitives intact, continuing."
        )
        traceback.print_exc()
        gaussians.metric_spaces = None
        gaussians.cluster_labels = None

    try:
        align_regions_with_primitives(dataset, gaussians, style)
    except Exception as exc:
        import traceback

        print(f"[fgw] ⑤ FGW alignment failed ({exc}); prior stages intact, continuing.")
        traceback.print_exc()
        gaussians.region_primitive_map = None
        gaussians.primitive_labels = None

    try:
        generate_style(dataset, gaussians, scene, pipe, background, style)
    except Exception as exc:
        import traceback

        print(
            f"[style] ⑥ style generation failed ({exc}); prior stages intact, continuing."
        )
        traceback.print_exc()

    network_gui.complete()


#  Entry point


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Style pipeline — refine + DINO distill + style primitives"
    )
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    sp = StyleParams(parser)

    parser.set_defaults(position_lr_max_steps=1000)

    parser.add_argument("--ip", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_viewer", action="store_true", default=False)
    parser.add_argument("--disable_viewer_pause", action="store_true", default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])

    parser.add_argument("--prune_iterations", nargs="+", type=int, default=None)

    args = get_combined_args(parser)
    try:
        validate_scene_path(args.source_path, args.images)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    allowed_metrics = ("v_important_score", "important_score", "visibility", "opacity")
    if args.prune_metric not in allowed_metrics:
        raise ValueError(
            f"--prune_metric must be one of {allowed_metrics}, got {args.prune_metric!r}"
        )
    if not args.skip_refine and args.refine_iterations < 1:
        raise ValueError(
            "--refine_iterations must be at least 1 (or use --skip_refine)"
        )
    if args.prune_percent < 0 or args.prune_percent >= 1:
        raise ValueError("--prune_percent must be in [0, 1)")
    if args.prune_from_iter < 1:
        raise ValueError("--prune_from_iter must be at least 1")
    if args.prune_decay < 0:
        raise ValueError("--prune_decay must be non-negative")
    if args.prune_iterations is not None:
        invalid_prune_iterations = [
            i for i in args.prune_iterations if i < 1 or i > args.refine_iterations
        ]
        if invalid_prune_iterations:
            raise ValueError(
                f"--prune_iterations must be within [1, {args.refine_iterations}], "
                f"got {invalid_prune_iterations}"
            )
    elif args.prune_interval < 1:
        raise ValueError(
            "--prune_interval must be positive unless --prune_iterations is set"
        )
    if args.importance_views == 0 or args.importance_views < -1:
        raise ValueError("--importance_views must be -1 or a positive integer")
    if args.min_gaussians < 1:
        raise ValueError("--min_gaussians must be at least 1")
    if args.style_iterations < 0:
        raise ValueError("--style_iterations must be non-negative (0 skips ⑥)")
    if args.style_render_scale <= 0 or args.style_render_scale > 1.0:
        raise ValueError("--style_render_scale must be in (0, 1]")
    if args.style_tv_knn_k < 1:
        raise ValueError("--style_tv_knn_k must be at least 1")
    if args.test_iterations is None:
        args.test_iterations = [args.refine_iterations]
    if args.save_iterations is None:
        args.save_iterations = [args.refine_iterations]
    elif args.refine_iterations not in args.save_iterations:
        args.save_iterations.append(args.refine_iterations)

    safe_state(args.quiet)
    print("Optimizing " + args.model_path)

    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    network_gui.disable_pause = args.disable_viewer_pause
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        sp.extract(args),
        args.prune_iterations,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.debug_from,
        args,
    )

    print("\nStyle refine complete.")
