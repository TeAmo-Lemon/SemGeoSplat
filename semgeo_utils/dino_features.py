from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rich.progress import track
from sklearn.decomposition import PCA


MERGED_FEATURE_FILENAME = "dino_features.pt"

# Default DINOv3 backbone (ModelScope hub id, not HuggingFace). Overridable at runtime
# via StyleParams.dino_model / --dino_model. patch_size & hidden_size are read from
# the model config, so swapping variants (vitl16=1024/16, vith16plus=1280/16, ...) needs
# no source changes.
DINOV3_MODEL_ID = "facebook/dinov3-vith16plus-pretrain-lvd1689m"

# Bumped on breaking changes to the cached feature payload (e.g. backbone swap).
# _load_existing_payload treats a mismatch as a cache miss -> full re-extraction
# + PCA refit, so a stale DINOv2 cache is never silently reused after the swap.
FEATURES_PAYLOAD_VERSION = 3


@dataclass
class ImageFeatureRecord:
    image_path: Path
    feature_key: str
    original_hw: Tuple[int, int]
    patch_hw: Tuple[int, int]
    features_1024: torch.Tensor


def _list_images(input_dir: Path, extensions: List[str]) -> List[Path]:
    ext_set = {e.lower() for e in extensions}
    image_paths = [
        p
        for p in sorted(input_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in ext_set
    ]
    return image_paths


def _resolve_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    if device.startswith("mps") and torch.backends.mps.is_available():
        return torch.device(device)
    return torch.device("cpu")


def _ceil_to_patch_multiple(value: int, patch_size: int) -> int:
    return int(math.ceil(value / patch_size) * patch_size)


def _load_and_preprocess_image(
    path: Path, patch_size: int, device: torch.device, max_size: int = None
) -> Tuple[torch.Tensor, Tuple[int, int], Tuple[int, int]]:
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, dtype=np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    h, w = image_tensor.shape[1], image_tensor.shape[2]

    target_h, target_w = h, w
    if max_size is not None and max(h, w) > max_size:
        scale = max_size / float(max(h, w))
        target_h = max(patch_size, int(round(h * scale)))
        target_w = max(patch_size, int(round(w * scale)))

    resized_h = _ceil_to_patch_multiple(target_h, patch_size)
    resized_w = _ceil_to_patch_multiple(target_w, patch_size)
    if resized_h != h or resized_w != w:
        image_tensor = F.interpolate(
            image_tensor.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)

    return image_tensor, (h, w), (resized_h // patch_size, resized_w // patch_size)


def _load_dinov3(device: torch.device, model_id: str = DINOV3_MODEL_ID):
    """Load the cached ModelScope backbone with transformers, downloading it only when absent. The caller owns the model and its GPU memory."""
    from modelscope.hub.snapshot_download import snapshot_download
    from transformers import AutoModel

    try:
        model_dir = snapshot_download(model_id, local_files_only=True)
        print(
            f"[INFO] Loading DINOv3 model ({model_id}) from local ModelScope cache ..."
        )
    except Exception:
        print(
            f"[INFO] Loading DINOv3 model ({model_id}) via ModelScope (no local cache) ..."
        )
        model_dir = snapshot_download(model_id)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True)
    patch_size = int(getattr(model.config, "patch_size", 16))
    return model.to(device).eval(), patch_size


def _extract_patch_features(
    model: torch.nn.Module, image_tensor: torch.Tensor, patch_hw: Tuple[int, int]
) -> torch.Tensor:
    # DINOv3 (transformers DINOv3ViTModel) returns last_hidden_state with token
    # layout [CLS, register_tokens..., patch_tokens...]. Slice off the leading
    # non-patch tokens (1 CLS + num_register_tokens, read from config) to recover
    # per-patch features, then reshape to a 2D feature map.
    with torch.no_grad():
        outputs = model(image_tensor)
    patch_tokens = outputs.last_hidden_state[
        :, 1 + int(getattr(model.config, "num_register_tokens", 0)) :, :
    ]
    ph, pw = patch_hw
    feature_map = (
        patch_tokens.reshape(1, ph, pw, -1).permute(0, 3, 1, 2).squeeze(0).contiguous()
    )
    return feature_map


def _fit_pca(records: List[ImageFeatureRecord], feature_dim: int) -> Tuple[PCA, int]:
    flattened = [
        r.features_1024.permute(1, 2, 0).reshape(-1, r.features_1024.shape[0]).numpy()
        for r in records
    ]
    all_features = np.concatenate(flattened, axis=0)
    effective_dim = min(feature_dim, all_features.shape[0], all_features.shape[1])
    pca = PCA(n_components=effective_dim, svd_solver="randomized", random_state=0)
    pca.fit(all_features)
    return pca, effective_dim


def _resolve_output_file(output_dir: Path) -> Path:
    if output_dir.suffix.lower() == ".pt":
        return output_dir
    return output_dir / MERGED_FEATURE_FILENAME


def _load_existing_payload(
    output_file: Path, target_dim: int, model_id: str = DINOV3_MODEL_ID
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    if not output_file.exists():
        return {}, {}

    payload = torch.load(output_file, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in {output_file}, got {type(payload)}")

    if int(payload.get("version", 0)) != FEATURES_PAYLOAD_VERSION:
        print(
            f"[INFO] Cache version {payload.get('version')} != {FEATURES_PAYLOAD_VERSION} "
            f"in {output_file}; re-extracting from scratch."
        )
        return {}, {}

    cached_model = payload.get("dino_model")
    if cached_model is not None and cached_model != model_id:
        print(
            f"[INFO] Cached model '{cached_model}' != current '{model_id}' "
            f"in {output_file}; re-extracting from scratch."
        )
        return {}, {}

    if "features" not in payload:
        raise KeyError(f"Missing 'features' key in {output_file}")
    if not isinstance(payload["features"], dict):
        raise TypeError(
            f"Expected dict under 'features' in {output_file}, got {type(payload['features'])}"
        )
    raw_features: Dict[Any, Any] = payload["features"]

    features: Dict[str, torch.Tensor] = {}
    for raw_key, raw_value in raw_features.items():
        key = str(raw_key).replace("\\", "/")
        if not isinstance(raw_value, torch.Tensor):
            raise TypeError(f"Feature entry '{key}' is not a tensor: {type(raw_value)}")
        value = raw_value.float()
        if value.shape[-1] != target_dim:
            raise ValueError(
                f"Feature dim mismatch for key '{key}' in {output_file}: "
                f"expected last dim {target_dim}, got {tuple(value.shape)}"
            )
        features[key] = value

    if "pca_state" not in payload:
        raise KeyError(f"Missing 'pca_state' key in {output_file}")
    pca_state = payload["pca_state"]
    if not isinstance(pca_state, dict):
        raise TypeError(
            f"Expected dict pca_state in {output_file}, got {type(pca_state)}"
        )
    if "components" not in pca_state or "mean" not in pca_state:
        raise KeyError(
            f"Invalid pca_state in {output_file}: expected 'components' and 'mean'"
        )

    return features, cast(Dict[str, torch.Tensor], pca_state)


def _serialize_pca_state(pca: PCA) -> Dict[str, torch.Tensor]:
    return {
        "components": torch.from_numpy(np.asarray(pca.components_, dtype=np.float32)),
        "mean": torch.from_numpy(np.asarray(pca.mean_, dtype=np.float32)),
    }


def _project_with_pca(
    record: ImageFeatureRecord, pca: PCA, target_dim: int
) -> torch.Tensor:

    components = np.asarray(pca.components_, dtype=np.float32)
    mean = np.asarray(pca.mean_, dtype=np.float32)
    flat = (
        record.features_1024.permute(1, 2, 0)
        .reshape(-1, record.features_1024.shape[0])
        .numpy()
    )
    reduced = _project_with_components(flat, components, mean, target_dim)

    ph, pw = record.patch_hw
    return (
        torch.from_numpy(reduced.astype(np.float32))
        .view(ph, pw, target_dim)
        .contiguous()
    )


def _project_with_components(x, components, mean, target_dim: int = None):
    """(x - mean) @ components.T 核心投影. x: [N, C] (numpy 或 torch, 返回同类型);
    components: [D, C]; mean: [C]. effective_dim != target_dim 时 pad 0 / 截断
    (target_dim=None 不处理). ⑤ fgw_align 复用同一公式, 保证两侧投影一致不漂移."""
    proj = (x - mean) @ components.T
    eff = components.shape[0]
    if target_dim is None or eff == target_dim:
        return proj
    if isinstance(x, torch.Tensor):
        if eff < target_dim:
            return F.pad(proj, (0, target_dim - eff))
        return proj[..., :target_dim]
    if eff < target_dim:
        return np.pad(proj, ((0, 0), (0, target_dim - eff)), mode="constant")
    return proj[..., :target_dim]


def _project_with_pca_state(
    record: ImageFeatureRecord, pca_state: Dict[str, torch.Tensor], target_dim: int
) -> torch.Tensor:
    components = pca_state["components"].cpu().numpy().astype(np.float32)
    mean = pca_state["mean"].cpu().numpy().astype(np.float32)

    flat = (
        record.features_1024.permute(1, 2, 0)
        .reshape(-1, record.features_1024.shape[0])
        .numpy()
    )
    reduced = _project_with_components(flat, components, mean, target_dim)

    ph, pw = record.patch_hw
    return (
        torch.from_numpy(reduced.astype(np.float32))
        .view(ph, pw, target_dim)
        .contiguous()
    )


def _save_payload(
    output_file: Path,
    features: Dict[str, torch.Tensor],
    feature_dim: int,
    input_dir: Path,
    pca_state: Dict[str, torch.Tensor],
    model_id: str,
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "version": FEATURES_PAYLOAD_VERSION,
        "dino_model": model_id,
        "feature_dim": feature_dim,
        "input_dir": str(input_dir),
        "features": features,
        "pca_state": pca_state,
    }
    torch.save(payload, output_file)


def extract_dino_features_for_images(
    image_paths: List[Path],
    input_dir: Path,
    output_dir: Path,
    feature_dim: int = 16,
    device: str = "cuda",
    skip_existing: bool = True,
    max_image_size: int = 1600,
    model_id: str = DINOV3_MODEL_ID,
) -> Tuple[int, int]:
    """Extract DINOv3 features for provided image paths and save PCA-reduced feature maps.

    Returns:
        Tuple[int, int]: (num_saved, num_skipped)
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    if len(image_paths) == 0:
        return 0, 0

    requested = sorted({p.resolve() for p in image_paths})
    output_file = _resolve_output_file(output_dir.resolve())
    features_payload, pca_state = _load_existing_payload(
        output_file, feature_dim, model_id
    )

    pending_keys: List[str] = []
    requested_by_key: Dict[str, Path] = {}
    skipped = 0

    for image_path in requested:
        try:
            relative = image_path.relative_to(input_dir).with_suffix(".pt")
        except ValueError as exc:
            raise ValueError(
                f"Image path {image_path} is not under input directory {input_dir}. "
                "Set input_dir to the common image root."
            ) from exc

        key = relative.as_posix()
        requested_by_key[key] = image_path

        if skip_existing and key in features_payload:
            skipped += 1
            continue

        pending_keys.append(key)

    if len(pending_keys) == 0:
        return 0, skipped

    resolved_device = _resolve_device(device)
    print(f"[INFO] Using device: {resolved_device}")
    model, patch_size = _load_dinov3(resolved_device, model_id)
    print(
        f"[INFO] DINOv3 patch_size={patch_size}, hidden_size={model.config.hidden_size}"
    )

    records: List[ImageFeatureRecord] = []
    for feature_key in track(
        pending_keys, description="Extracting DINO patch features"
    ):
        image_path = requested_by_key[feature_key]
        image_tensor, original_hw, patch_hw = _load_and_preprocess_image(
            image_path, patch_size, resolved_device, max_image_size
        )
        feature_map = _extract_patch_features(model, image_tensor, patch_hw).cpu()
        records.append(
            ImageFeatureRecord(
                image_path=image_path,
                feature_key=feature_key,
                original_hw=original_hw,
                patch_hw=patch_hw,
                features_1024=feature_map,
            )
        )

    # Free the DINOv3 model from GPU memory — it is no longer needed after patch extraction.
    del model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    if len(pca_state) == 0:
        print("[INFO] Fitting PCA...")
        pca, effective_dim = _fit_pca(records, feature_dim)
        if effective_dim < feature_dim:
            print(
                f"[WARN] Effective PCA dimension is {effective_dim} < requested {feature_dim}; "
                "remaining channels are padded with zeros."
            )

        for record in track(records, description="Projecting and upsampling features"):
            features_payload[record.feature_key] = _project_with_pca(
                record, pca, feature_dim
            )

        pca_state = _serialize_pca_state(pca)
    else:
        for record in track(records, description="Projecting and upsampling features"):
            features_payload[record.feature_key] = _project_with_pca_state(
                record, pca_state, feature_dim
            )

    _save_payload(
        output_file=output_file,
        features=features_payload,
        feature_dim=feature_dim,
        input_dir=input_dir,
        pca_state=pca_state,
        model_id=model_id,
    )

    return len(records), skipped
