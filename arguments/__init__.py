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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true"
                    )
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t
                    )
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

    @classmethod
    def defaults(cls) -> dict:
        """Return parser defaults keyed by their command-line destination names."""
        parser = ArgumentParser(add_help=False)
        cls(parser)
        return {a.dest: a.default for a in parser._actions if a.dest}


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._depths = ""
        self._resolution = -1
        self._white_background = False
        self.train_test_exp = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        self.antialiasing = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.025
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.exposure_lr_init = 0.01
        self.exposure_lr_final = 0.001
        self.exposure_lr_delay_steps = 0
        self.exposure_lr_delay_mult = 0.0
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.depth_l1_weight_init = 1.0
        self.depth_l1_weight_final = 0.01
        self.random_background = False
        self.optimizer_type = "default"
        self.use_scale_regularization = True
        self.max_gauss_ratio = 10.0
        super().__init__(parser, "Optimization Parameters")


class StyleParams(ParamGroup):
    """Style pipeline 训练参数 (refine + DINO 蒸馏 + 风格基元分解). 不复用 OptimizationParams.iterations 等."""

    def __init__(self, parser):

        # Input
        self.start_ply = ""
        self.start_pointcloud = ""

        # Refinement
        self.refine_iterations = 500
        self.prune_percent = 0.2
        self.prune_from_iter = 1
        self.prune_interval = 100
        self.prune_decay = 1.0
        self.skip_refine = False
        self.force_refine = False
        self.prune_metric = "v_important_score"
        self.v_pow = 0.1
        self.importance_views = -1
        self.min_gaussians = 1

        # Semantic distillation
        self.dino_feature_dim = 32
        self.dino_model = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        self.dino_lr = 0.01
        self.dino_lr_final = 0.0005
        self.dino_distill_iterations = 3000
        self.dino_lambda = 1.0
        self.dino_features_path = ""
        self.dino_image_dir = ""
        self.dino_max_image_size = 1600
        self.force_dino = False

        # Style primitives
        self.style_image = ""
        self.style_primitive_k = 8
        self.style_primitive_iters = 50
        self.style_primitive_size = 560

        # Region construction
        self.region_opacity_threshold = 0.05
        self.region_max_sample_points = 200000
        self.region_k_base = 10
        self.region_overseg_scale = 120.0
        self.region_overseg_max_size = 2000
        self.region_w_dist = 1.0
        self.region_w_feat = 1.2
        self.region_w_normal = 0.4
        self.region_w_scale = 0.5
        self.region_merge_max_cost = 0.60
        self.region_min_cluster_size = 200
        self.region_m_reps = 128
        self.region_overseg_geo_only = 0
        self.region_adaptive_merge_beta = 0.0

        # Region-to-style matching
        self.fgw_alpha = 0.5

        # Appearance optimization
        self.style_iterations = 1500
        self.style_lr = 0.01
        self.style_lambda = 5.0
        self.style_lambda_normal = 0.05
        self.style_lambda_tv = 0.05
        self.style_lambda_photo = 0.0
        self.style_render_scale = 0.5
        self.style_tv_knn_k = 8
        self.style_normal_tau = 0.6
        self.style_knn_backend = "faiss"
        super().__init__(parser, "Style Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)
    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        if not getattr(args_cmdline, "quiet", False):
            print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            if not getattr(args_cmdline, "quiet", False):
                print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        if not getattr(args_cmdline, "quiet", False):
            print("Config file not found, using command-line args only")
        return args_cmdline
    args_cfgfile = eval(cfgfile_string)
    explicitly_passed = set()
    for action in parser._actions:
        for opt in action.option_strings:
            if opt in sys.argv[1:]:
                explicitly_passed.add(action.dest)
                break
    bool_action_dests = {a.dest for a in parser._actions if a.nargs == 0}
    merged_dict = vars(args_cmdline).copy()
    for k, v in vars(args_cfgfile).items():
        if k in explicitly_passed or k in bool_action_dests:
            continue
        merged_dict[k] = v
    return Namespace(**merged_dict)
