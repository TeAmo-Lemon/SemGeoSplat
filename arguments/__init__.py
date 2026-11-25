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

import ast
from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
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
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
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
        # ---- Scale regularization (train.py) ----
        self.use_scale_regularization = True
        self.max_gauss_ratio = 10.0
        super().__init__(parser, "Optimization Parameters")

class StyleParams(ParamGroup):
    """Style pipeline 训练参数 (refine + DINO 蒸馏 + 风格基元分解). 不复用 OptimizationParams.iterations 等."""
    def __init__(self, parser):
        # 输入的已训练 point_cloud.ply. start_pointcloud 是兼容旧名字.
        self.start_ply = ""
        self.start_pointcloud = ""
        # Refine stage: 独立的迭代预算 (不共用 OptimizationParams.iterations).
        self.refine_iterations = 500
        # Refine stage: 剪枝调度.
        self.prune_percent = 0.2
        self.prune_from_iter = 1
        self.prune_interval = 100
        self.prune_decay = 1.0
        # Refine stage: 重要度分数.
        # prune_metric ∈ {"v_important_score", "important_score", "visibility", "opacity"}
        self.skip_refine = False             # 跳过 refine 阶段, 直接用原始 start_ply 进入后续阶段
        self.force_refine = False            # 强制重跑 refine, 忽略 iteration_refine/ 缓存
        self.prune_metric = "v_important_score"
        self.v_pow = 0.1
        self.importance_views = -1
        self.min_gaussians = 1
        # ---- DINO 蒸馏 (refine 结束后在 train() 末尾自动执行, 无独立 stage) ----
        # 每个高斯球蒸馏一个 dino_feature_dim 维语义特征. 几何/SH 全部冻结, 只训 dino.
        self.dino_feature_dim = 32
        # DINOv3 backbone (ModelScope hub id). 换模型改这里; patch_size/hidden_size
        # 自动从模型 config 读, 下游 reshape/投影全程用动态维度, 无需改源码.
        self.dino_model = "facebook/dinov3-vith16plus-pretrain-lvd1689m"
        # LR 指数衰减 dino_lr -> dino_lr_final: 前期高 LR 快速收缩, 后期低 LR 微调.
        # Adam 对梯度幅度近似不变 (一/二阶矩同比缩放), 调 dino_lambda 几乎不改步长,
        # 故用 LR 而非 lambda 控 "梯度变化" 的大小.
        self.dino_lr = 0.01                  # 初始 (峰值) LR; 旧恒定值 0.005, 前期调高以快速收敛
        self.dino_lr_final = 0.0005          # 末期 LR; 后期减小以精修 (1/10 旧值)
        self.dino_distill_iterations = 3000   # <= 0 时跳过整个 DINO 阶段
        self.dino_lambda = 1.0                # 无 RGB 主损失竞争, 权重取 1.0 (非架构书的 0.05)
        # GT 2D 特征 .pt. 空 -> <model_path>/dino_features.pt.
        self.dino_features_path = ""
        # 待提取的图像目录. 空 -> <source_path>/<images>.
        self.dino_image_dir = ""
        # 提取 DINOv3 前把图像最长边压到该值 (>1600 的大图会 OOM/卡死). 0 = 不压缩.
        self.dino_max_image_size = 1600
        # 强制重提 DINO GT 特征, 忽略 <refine_dir>/dino_features.pt 缓存
        # (删后全量重提所有图 + 重拟合 PCA; 否则旧 PCA 被复用、已缓存图被跳过).
        self.force_dino = False
        # ---- 阶段二: 2D 风格图纹理基元分解 (training() 内, DINOv3 只加载一次) ----
        # 风格图路径; 空则跳过基元分解. 给定后在 training() 里提取 patch 特征
        # 并 K-Means 聚成 K 个 prototype, 保留在显存 (gaussians.style_primitives).
        self.style_image = ""
        self.style_primitive_k = 8          # 聚类数 (建议 5~8)
        self.style_primitive_iters = 50     # K-Means 迭代轮数
        self.style_primitive_size = 560    # 风格图最长边 (会 pad 到 16 的倍数)
        # Reuse an existing decomposition of the same reference image.
        self.style_primitives_reference_path = ""
        # ---- 阶段四: 从高斯到结构感知区域与度量空间 (②③ 之后, ⑤ FGW 之前) ----
        # 架构书 4.1~4.4. ④ 在内存中对已蒸馏的固定高斯点集 (_dino_features) 继续操作.
        # 两阶段聚类: 加权图 -> Felzenszwalb 过分割(产生大量 super-region) ->
        # 区域特征聚合 -> RAG -> 相邻约束凝聚层次合并 -> 最终区域 -> 度量测度空间, 供 ⑤ FGW.
        # 只暴露架构书附录 A 列出的可调超参; 工程内部常数 (k_max/min_size 等) 见 regions.py.
        self.region_opacity_threshold = 0.05    # 4.4 有效点筛选 σ(α) ≥ 此值
        self.region_max_sample_points = 200000  # 4.4 图构建最大采样点数 (按不透明度保留 top-k)
        # density-k: 局部尺度估计 r_i = median(前 k 近邻距离). 比"第 k 近邻×α"更稳,
        # 稀疏区半径随密度自适应放大, 边权用 d/r_i 归一 (用户指定 median(k=10)).
        self.region_k_base = 10
        # Felzenszwalb 过分割 scale 参数 k: MInt(C)=Int(C)+k/|C|, 越大 super-region 越大越少.
        # 平滑控制粒度, 避开 BFS 阈值的离散相变 (tau_feat 的 3->9->13->27 跳变).
        self.region_overseg_scale = 120.0
        # Felzenszwalb 组件大小上限: 组件达到此规模后停止吸收邻居 -> 强制过分割,
        # 打破"一个巨簇" (garden 上 DINO 平滑, W 边界不够锐, Felzenszwalb 会贪心吃成巨簇).
        self.region_overseg_max_size = 2000
        # 边权 4 项权重 (各 dissimilarity 项已归一到 [0,1] 量级, 再加权求和 = Felzenszwalb 边权 W).
        # 合并代价复用同一组权重. cost = w_f·0.5(1-cos) + w_n·(1-|n·n|) + w_s·|Δlog s|/5.
        # v5 权重策略: DINO 语义主导, 法线/scale 为辅 (避免同纹理不同朝向的碎片).
        self.region_w_dist = 1.0                # 4.2 密度归一距离项 w_d (仅过分割用)
        self.region_w_feat = 1.2                # 4.2 DINO 语义相异项 w_f (1-cos), 主导合并
        self.region_w_normal = 0.4              # 4.2 法线相异项 w_n (1-|n·n|, merge cap 0.5), 降权防碎片
        self.region_w_scale = 0.5               # 4.2 scale 相异项 w_s (|Δlog s|)
        # 凝聚层次合并代价上限: 相邻区域对 cost < 此值才合并 (cost 复用边权 4 项 + centroid 项).
        # 是"Feature Similar AND Geometry Similar AND Boundary Connected"中的相似度门槛;
        # "Boundary Connected" 由 RAG 邻接天然保证 (不相邻永不入堆).
        # 默认 0.35: garden (200k 点) 上 ~65 区域, 最大 ~12% (无主导巨簇).
        self.region_merge_max_cost = 0.60
        self.region_min_cluster_size = 200       # 4.2 最终残片兜底合并阈值 (100~300)
        self.region_m_reps = 128                 # 4.3 度量空间代表点数 m (50~200, FPS 采样)
        # ---- v5: 两阶段解耦 + 自适应阈值 (架构书 4.2 改进) ----
        # Stage1 纯几何过分割: 仅用空间距离+法线+scale (去掉 DINO), 使 os 控制几何粒度.
        # Stage2 语义合并: 保留 DINO+法线+scale 的合并代价, 使 mm 控制语义粒度.
        # 0=回退到 v4 行为 (两阶段用相同特征空间, os 几乎无效); 1=v5 几何解耦.
        self.region_overseg_geo_only = 0
        # 自适应合并阈值强度: effective_mm = mm * (size_ratio)^(-beta), size_ratio =
        # (size_i+size_j)/(2*mean_initial_size). beta=0 禁用; 推荐 0.5.
        # 效果: 大区域更难合并(防雪崩), 小区域更易合并(减碎片), 使 mm 平滑可控.
        self.region_adaptive_merge_beta = 0.0
        # Optionally reuse the labels and sampled metric spaces of a frozen scene.
        self.regions_reference_path = ""
        # 自动微调新建 full 分区。引用分区需要显式指定 all 或区域 ID。
        self.region_boundary = "auto"
        self.region_boundary_corrections = ""
        self.region_dino_cache = ""
        self.region_preview = False
        self.region_preview_views = 3
        self.region_preview_style = ""
        # 架构书 5.2: 联合特征代价 M 与结构代价 (C1=d_X, C2=d_Y), α 平衡语义/结构.
        # 0.5 = 5.2 推荐; 大 α 偏结构 (拓扑相近优先), 小 α 偏特征 (语义相近优先).
        self.fgw_alpha = 0.5                   # 5.2 FGW 特征/结构平衡系数 (附录 A: 0.5)
        self.spatial_fgw_nodes = 48            # 每个区域使用的参考图 patch 代表点预算
        self.spatial_fgw_prior_weight = 0.08   # 风格侧 KL 先验权重
        self.spatial_fgw_entropy = 0.01        # 软传输的熵正则权重
        self.spatial_fgw_max_iter = 120
        self.spatial_fgw_contrast_weight = 0.0  # 测地邻域语义对比项，0 为原始目标
        self.spatial_fgw_routing = "region"    # region / local
        self.style_control_file = ""           # JSON，区域级参考图选区和强度
        self.style_control_strength = 1.0     # 输出阶段的全局 SH 强度乘数
        # ---- 阶段六: 多视角一致性风格生成 (⑤ 之后) ----
        # 架构书 6.1/6.2: 冻结几何 (xyz/协方差/opacity), 仅优化 SH (基础颜色 f_dc + 视角相关
        # f_rest), 通过逐基元 VGG-NNFM 把每个区域 R_j 驱向 ⑤ 匹配的风格基元 P_k 的纹理.
        # L = λ_style·L_style + λ_normal·L_normal + λ_tv·L_tv (+ 可选 λ_photo·L_photo).
        # NNFM 机制参考 StylizedGS (VGG16 + 最近邻特征替换 + cosine loss),
        # 但风格目标按 ⑤ 的逐区域→基元映射路由 (每像素渲染基元标签 → 对应基元的风格图 patch 库).
        self.style_iterations = 1500           # 6 ⑥ 迭代步数 (<=0 跳过整个 ⑥)
        self.style_lr = 0.01                   # 6 SH DC 学习率 (f_rest = style_lr/20, 同 3DGS 习惯)
        self.style_tone_strength = 0.75        # Spatial FGW 输出阶段的全局色调强度，0 关闭
        # 损失权重. L_style 为主导; L_normal/L_tv 为一致性正则 (抑制视角闪烁).
        self.style_lambda = 5.0               # 6 λ_style (NNFM, 主导项)
        self.style_lambda_normal = 0.05        # 6 λ_normal (法线一致邻居的 SH DC 一致性)
        self.style_lambda_tv = 0.05           # 6 λ_tv (空间邻居 SH DC 全变分)
        self.style_lambda_photo = 0.0         # 6 λ_photo (内容 L1, 默认关; >0 需相机原图, 首版未实现重载)
        self.style_render_scale = 0.5         # 6 渲染图下采样比例 (省显存, 同 StylizedGS scale_factor=0.5)
        self.style_tv_knn_k = 8               # 6 L_normal/L_tv 的 KNN 近邻数
        self.style_normal_tau = 0.6           # 6 L_normal 法线一致阈值 |n_i·n_j|>τ
        # KNN 后端: "faiss" (faiss-gpu, 百万级点秒级) 或 "cdist" (chunked, 回退).
        self.style_knn_backend = "faiss"
        super().__init__(parser, "Style Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        if not getattr(args_cmdline, 'quiet', False):
            print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            if not getattr(args_cmdline, 'quiet', False):
                print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except (TypeError, FileNotFoundError):
        if not getattr(args_cmdline, 'quiet', False):
            print("Config file not found, using command-line args only")
        return args_cmdline

    call = ast.parse(cfgfile_string.strip(), mode="eval").body
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != "Namespace" or call.args or any(k.arg is None for k in call.keywords)):
        raise ValueError(f"Invalid model configuration in {cfgfilepath}")
    configured = {item.arg: ast.literal_eval(item.value) for item in call.keywords}
    explicit_options = {token.partition("=")[0] for token in cmdlne_string}
    merged = vars(args_cmdline).copy()
    for action in parser._actions:
        if action.dest not in configured or explicit_options.intersection(action.option_strings):
            continue
        # Runtime switches have a concrete boolean default. Model booleans use
        # None in the rendering parser and should use the saved scene setting.
        if action.nargs == 0 and action.default is not None:
            continue
        merged[action.dest] = configured[action.dest]
    return Namespace(**merged)
