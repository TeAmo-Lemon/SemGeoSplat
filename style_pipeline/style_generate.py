"""Optimize spherical harmonics with region-conditioned style and consistency losses."""

import os
from pathlib import Path
from random import randint

import torch
import torch.nn.functional as F
from tqdm import tqdm

from gaussian_renderer import network_gui, render, render_features
from .geometry import _compute_normals, compute_knn_graph
from .paths import sidecar_path, style_output_dir


_STYLE_VGG_BLOCKS = [2, 3]

_VGG_BLOCK_INDEXES = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]


def _argmin_cos_distance(a, b):
    """a: [B,C,Na], b: [B,C,Nb] → z_best [B, Na] (a 每列在 b 中最近邻下标, cosine 距离)."""
    b_norm = ((b * b).sum(1, keepdim=True) + 1e-8).sqrt()
    b = b / (b_norm + 1e-8)
    z_best = []
    loop_batch_size = max(1, int(1e8 / b.shape[-1]))
    for i in range(0, a.shape[-1], loop_batch_size):
        a_batch = a[..., i : i + loop_batch_size]
        a_batch_norm = ((a_batch * a_batch).sum(1, keepdim=True) + 1e-8).sqrt()
        a_batch = a_batch / (a_batch_norm + 1e-8)
        d_mat = 1.0 - torch.matmul(a_batch.transpose(2, 1), b)
        z_best.append(torch.argmin(d_mat, 2))
    return torch.cat(z_best, dim=-1)


def _nn_feat_replace(a, b):
    """a: [1,C,h,w], b: [1,C,h2,w2] → [1,C,h,w]. a 每个空间位置替换为 b 中最近邻特征.
    argmin 在 no_grad 下算 (非可导), 输出 gather 自 b (风格特征, 常量) → target 无梯度,
    梯度只经 a (渲染特征) 回流到 SH."""
    n, c, h, w = a.size()
    n2, c2, h2, w2 = b.size()
    assert (n == 1) and (n2 == 1)
    a_flat = a.view(n, c, -1)
    b_flat = b.view(n2, c, -1)
    b_ref = b_flat.clone()
    with torch.no_grad():
        z_best = _argmin_cos_distance(a_flat.detach(), b_flat)
    z_best = z_best.unsqueeze(1).repeat(1, c, 1)
    feat = torch.gather(b_ref, 2, z_best)
    return feat.view(n, c, h, w)


def _cos_loss(a, b):
    """a, b: [B,C,...] → mean(1 - cosine sim) over channel dim. 与 StylizedGS cos_loss 一致."""
    a_norm = (a * a).sum(1, keepdim=True).sqrt()
    b_norm = (b * b).sum(1, keepdim=True).sqrt()
    a_tmp = a / (a_norm + 1e-8)
    b_tmp = b / (b_norm + 1e-8)
    cossim = (a_tmp * b_tmp).sum(1)
    return (1.0 - cossim).mean()


class _VGGFeatureExtractor(torch.nn.Module):
    """VGG16 卷积特征提取器 (NNFM 用). 输入须已 ImageNet 归一化 (与 ③ DINO 预处理一致,
    故 ⑥ 复用 ③ 的风格图张量可直接喂 VGG, patch 网格与 VGG 特征精确对齐)."""

    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval().to(device)
        for p in self.vgg.parameters():
            p.requires_grad_(False)

    def get_feats(self, x, layers):
        final_ix = max(layers)
        outputs = []
        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)
            if ix == final_ix:
                break
        return outputs


def _vgg_imagenet_normalize(img_01, device):
    """img_01 in [0,1], shape [1,3,H,W] → ImageNet 归一化 (与 DINO 预处理 mean/std 一致)."""
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(
        1, 3, 1, 1
    )
    return (img_01 - mean) / std


def _build_style_banks(
    extractor, style_tensor, patch_assignments, patch_hw, blocks, K, device
):
    """风格图 → 逐基元 VGG 特征库. 复用 ③ 的预处理张量 (ImageNet 归一化, 与 VGG 对齐).
    patch 指派网格 [ph,pw] 最近邻上采样到各 VGG block 分辨率 → 每个风格 VGG 位置一个基元标签
    → 按基元收集成 bank. 返回 banks[block_idx][k] = [C, N_k] (常量, 无梯度)."""
    with torch.no_grad():
        x = (
            style_tensor if style_tensor.dim() == 4 else style_tensor.unsqueeze(0)
        )  # [1,3,Hs,Ws]
        all_layers = []
        for b in blocks:
            all_layers += _VGG_BLOCK_INDEXES[b]
        feats_all = extractor.get_feats(x, all_layers)
        ix_map = {layer: i for i, layer in enumerate(all_layers)}
        ph, pw = patch_hw
        label_map = patch_assignments.view(ph, pw).float().unsqueeze(0).unsqueeze(0)
        banks = []
        for b in blocks:
            layers = _VGG_BLOCK_INDEXES[b]
            feat = torch.cat([feats_all[ix_map[ix]] for ix in layers], 1)[
                0
            ]  # [C, Hf, Wf]
            C, Hf, Wf = feat.shape
            lab = F.interpolate(label_map, size=(Hf, Wf), mode="nearest")[0, 0].long()
            feat_flat = feat.reshape(C, -1)
            lab_flat = lab.reshape(-1)
            bank_k = {}
            for k in range(K):
                m = lab_flat == k
                bank_k[k] = (
                    feat_flat[:, m].contiguous()
                    if m.any()
                    else torch.empty(C, 0, device=device)
                )
            banks.append(bank_k)
    return banks


def _style_loss_per_view(
    extractor, rendered_01, prim_label, depth, banks, blocks, K, device
):
    """单视图逐基元 NNFM loss. rendered_01 [3,H,W] in [0,1]; prim_label [H,W] (alpha-blended
    基元标签); depth [1,H,W] (inv_depth, >0=前景). 渲染特征按逐像素基元路由到对应风格库做
    最近邻替换 + cosine loss; 前景掩码由 depth>0 给出 (无需相机原图/alpha_mask)."""
    x = _vgg_imagenet_normalize(rendered_01.unsqueeze(0), device)  # [1,3,H,W]
    all_layers = []
    for b in blocks:
        all_layers += _VGG_BLOCK_INDEXES[b]
    feats_all = extractor.get_feats(x, all_layers)
    ix_map = {layer: i for i, layer in enumerate(all_layers)}
    fg_full = depth[0] > 0  # [H, W]
    total_loss = torch.zeros((), device=device)
    total_n = 0
    for bi, b in enumerate(blocks):
        layers = _VGG_BLOCK_INDEXES[b]
        feat = torch.cat([feats_all[ix_map[ix]] for ix in layers], 1)[0]  # [C, Hf, Wf]
        C, Hf, Wf = feat.shape
        pl = (
            F.interpolate(
                prim_label.float().unsqueeze(0).unsqueeze(0),
                size=(Hf, Wf),
                mode="nearest",
            )[0, 0]
            .round()
            .long()
        )
        fg = F.interpolate(
            fg_full.float().unsqueeze(0).unsqueeze(0), size=(Hf, Wf), mode="nearest"
        )[0, 0].bool()
        feat_flat = feat.reshape(C, -1)
        pl_flat = pl.reshape(-1)
        fg_flat = fg.reshape(-1)
        bank_k = banks[bi]
        for k in range(K):
            mask = fg_flat & (pl_flat == k)
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue
            S_k = bank_k.get(k)
            if S_k is None or S_k.shape[1] == 0:
                continue
            R_k = feat_flat[:, mask].contiguous()  # [C, n_k]
            target_k = _nn_feat_replace(
                R_k.unsqueeze(0).unsqueeze(2), S_k.unsqueeze(0).unsqueeze(2)
            )  # [1,C,1,n_k]
            loss_k = _cos_loss(R_k.unsqueeze(0).unsqueeze(2), target_k)  # scalar
            total_loss = total_loss + loss_k * n_k
            total_n += n_k
    if total_n == 0:
        return None
    return total_loss / total_n


def _precompute_style_graph(gaussians, k, backend, device):
    """⑥ 一致性正则用的 KNN 图 + 法线点积 (几何冻结 → 一次性预计算). 复用 ④ 的
    compute_knn_graph / _compute_normals. 返回 (knn_idx[N,k], normal_dot[N,k])."""
    xyz = gaussians.get_xyz.detach()
    knn_idx, _ = compute_knn_graph(xyz, k, backend=backend)
    scaling = gaussians.get_scaling.detach()
    rotation = gaussians.get_rotation.detach()
    normals = _compute_normals(scaling, rotation)  # [N,3]
    nbr_n = normals[knn_idx]  # [N,k,3]
    normal_dot = torch.einsum("nkd,nd->nk", nbr_n, normals).abs()  # [N,k]
    return knn_idx, normal_dot


def _normal_tv_loss(features_dc, knn_idx, normal_dot, tau_normal):
    """L_normal (法线一致邻居 SH DC 一致) + L_tv (空间邻居 SH DC 全变分).
    "风格特征" 取 SH DC (视角无关基础色): 正则化它直接抑制视角切换闪烁, 且 [N,3] 内存可控.
    features_dc [N,1,3]; L_tv 覆盖所有 KNN 邻居, L_normal 是其中 |n_i·n_j|>τ 的子集."""
    c = features_dc.reshape(-1, 3)  # [N,3]
    nbr_c = c[knn_idx]  # [N,k,3]
    diff = (c.unsqueeze(1) - nbr_c).abs()  # [N,k,3]
    l_tv = diff.mean()
    normal_mask = normal_dot > tau_normal  # [N,k]
    n_normal = normal_mask.sum()
    if n_normal.item() > 0:
        l_normal = (diff * normal_mask.unsqueeze(-1).float()).sum() / (
            n_normal.float() * 3.0 + 1e-8
        )
    else:
        l_normal = torch.zeros((), device=c.device)
    return l_normal, l_tv


def generate_style(dataset, gaussians, scene, pipe, background, style):
    """Freeze geometry and optimize spherical harmonics using region-conditioned VGG style losses with normal and TV regularization. Save the styled scene and sidecars."""
    if style.style_iterations <= 0:
        print("[style] ⑥ skipped: style_iterations <= 0")
        return
    primitives = getattr(gaussians, "style_primitives", None)
    primitive_labels = getattr(gaussians, "primitive_labels", None)
    if primitives is None or primitives.numel() == 0:
        print(
            "[style] ⑥ skipped: no style_primitives (③ did not run / no --style_image)"
        )
        return
    if primitive_labels is None:
        print("[style] ⑥ skipped: no primitive_labels (⑤ did not run / failed)")
        return
    if style.style_lambda_photo > 0:
        print("[style] ⑥ λ_photo>0 但 L_photo 未实现 (相机原图已释放); 忽略 L_photo")

    device = torch.device("cuda")
    K = int(primitives.shape[0])
    blocks = list(_STYLE_VGG_BLOCKS)
    final_styled_iter = style.style_iterations
    n_points = gaussians.get_xyz.shape[0]

    for p in (
        gaussians._xyz,
        gaussians._scaling,
        gaussians._rotation,
        gaussians._opacity,
        gaussians._dino_features,
    ):
        if p is not None:
            p.requires_grad_(False)
    for p in (gaussians._features_dc, gaussians._features_rest):
        p.requires_grad_(True)
    style_optimizer = torch.optim.Adam(
        [
            {"params": [gaussians._features_dc], "lr": style.style_lr},
            {"params": [gaussians._features_rest], "lr": style.style_lr / 20.0},
        ],
        lr=0.0,
        eps=1e-15,
    )

    from semgeo_utils.dino_features import _load_and_preprocess_image

    print(
        f"[style] ⑥ start: {n_points:,} gaussians, {K} primitives, "
        f"{style.style_iterations} iters, blocks={blocks}, λ(style/normal/tv)="
        f"{style.style_lambda}/{style.style_lambda_normal}/{style.style_lambda_tv}"
    )
    extractor = _VGGFeatureExtractor(device)

    patch_assignments = getattr(gaussians, "style_patch_assignments", None)
    patch_hw = getattr(gaussians, "style_patch_hw", None)
    patch_size = getattr(gaussians, "style_patch_size", None)
    if patch_assignments is None or patch_hw is None:
        sidecar = sidecar_path(
            dataset.model_path, style.style_iterations, "style_primitives.pt"
        )
        payload = torch.load(sidecar, map_location="cpu")
        patch_assignments = payload["patch_assignments"]
        patch_hw = payload["patch_hw"]
        patch_size = payload.get("patch_size", 16)
    style_tensor, _, _ = _load_and_preprocess_image(
        Path(style.style_image),
        patch_size=patch_size,
        device=device,
        max_size=style.style_primitive_size,
    )
    banks = _build_style_banks(
        extractor,
        style_tensor,
        patch_assignments.to(device),
        patch_hw,
        blocks,
        K,
        device,
    )
    del style_tensor, patch_assignments

    knn_idx, normal_dot = _precompute_style_graph(
        gaussians, style.style_tv_knn_k, style.style_knn_backend, device
    )

    pl = primitive_labels.to(device)
    pl = torch.where(pl >= 0, pl.float(), torch.zeros_like(pl, dtype=torch.float32))
    pl_render = pl.unsqueeze(1).contiguous()  # [N, 1]

    cameras = scene.getTrainCameras().copy()
    viewpoint_stack = []
    rs = style.style_render_scale
    progress = tqdm(range(style.style_iterations), desc="Style ⑥")
    ema_loss = ema_style = ema_normal = ema_tv = 0.0
    for it in range(1, style.style_iterations + 1):
        network_gui.update(it, num_train_rays_per_step=1)
        if not network_gui.disable_pause:
            network_gui.wait_if_paused(it)

        if not viewpoint_stack:
            viewpoint_stack = cameras.copy()
        cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        with network_gui.lock:
            render_pkg = render(
                cam,
                gaussians,
                pipe,
                background,
                use_trained_exp=False,
                no_densify_hook=True,
            )
            with torch.no_grad():
                prim_map = render_features(cam, gaussians, pl_render)  # [1,H,W]
        rgb = render_pkg["render"]  # [3,H,W] in [0,1]
        depth = render_pkg["depth"]  # [1,H,W] inv_depth

        if rs != 1.0:
            rgb_v = F.interpolate(
                rgb.unsqueeze(0), scale_factor=rs, mode="bilinear", align_corners=False
            )[0]
            depth_v = F.interpolate(
                depth.unsqueeze(0),
                scale_factor=rs,
                mode="bilinear",
                align_corners=False,
            )[0]
            prim_map = F.interpolate(
                prim_map.unsqueeze(0),
                scale_factor=rs,
                mode="bilinear",
                align_corners=False,
            )[0]
        else:
            rgb_v, depth_v = rgb, depth
        prim_label = prim_map[0]  # [H,W]

        L_style = _style_loss_per_view(
            extractor, rgb_v, prim_label, depth_v, banks, blocks, K, device
        )
        L_normal, L_tv = _normal_tv_loss(
            gaussians._features_dc, knn_idx, normal_dot, style.style_normal_tau
        )

        loss = style.style_lambda * (L_style if L_style is not None else 0.0)
        loss = (
            loss + style.style_lambda_normal * L_normal + style.style_lambda_tv * L_tv
        )

        style_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        style_optimizer.step()

        with torch.no_grad():
            ema_loss = 0.4 * float(loss) + 0.6 * ema_loss
            ema_style = (
                0.4 * (float(L_style) if L_style is not None else 0.0) + 0.6 * ema_style
            )
            ema_normal = 0.4 * float(L_normal) + 0.6 * ema_normal
            ema_tv = 0.4 * float(L_tv) + 0.6 * ema_tv
            if it % 10 == 0:
                progress.set_postfix(
                    {
                        "L": f"{ema_loss:.4f}",
                        "style": f"{ema_style:.4f}",
                        "normal": f"{ema_normal:.4f}",
                        "tv": f"{ema_tv:.4f}",
                        "Mem": f"{torch.cuda.memory_allocated() / (1024**2):.0f}M",
                    }
                )
                progress.update(10)
    progress.close()

    styled_dir = style_output_dir(dataset.model_path, final_styled_iter)
    styled_ply = os.path.join(styled_dir, "point_cloud.ply")
    gaussians.save_ply(styled_ply)
    print(f"[style] ⑥ done: styled ply -> {styled_ply}")

    del extractor, banks, knn_idx, normal_dot, pl_render
    style_optimizer.state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
