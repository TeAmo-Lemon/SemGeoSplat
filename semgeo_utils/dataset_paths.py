"""Lightweight dataset discovery and layout checks, without GPU imports."""

from pathlib import Path


def resolve_dataset(raw: str, root: str) -> str:
    """Resolve a scene shorthand uniquely among root/scene and root/subset/scene."""
    if not raw.strip():
        raise ValueError("数据集名称不能为空")
    path = Path(raw).expanduser()
    if path.is_absolute() or "/" in raw:
        return str(path)

    base = Path(root)
    candidates = []
    if (base / raw).is_dir():
        candidates.append(base / raw)
    if base.is_dir():
        candidates.extend(
            subset / raw
            for subset in sorted(base.iterdir())
            if subset.is_dir() and (subset / raw).is_dir()
        )
    if len(candidates) == 1:
        return str(candidates[0].relative_to(base))
    if candidates:
        choices = ", ".join(str(p.relative_to(base)) for p in candidates)
        raise ValueError(
            f"场景名 {raw!r} 对应多个目录: {choices}。请用 --dataset 指定完整路径。"
        )
    raise ValueError(
        f"找不到数据集 {raw!r}：已检查 {base / raw} 和 {base}/<subset>/{raw}。"
        "请检查名称，或用 --dataset 指定完整路径。"
    )


def validate_scene_path(source_path: str, images: str = "images") -> str:
    """Return the reader type after checking the files its loader requires.

    This checks layout only; parsing cameras and image contents stays in the readers.
    """
    root = Path(source_path).expanduser().absolute()
    if not root.is_dir():
        raise ValueError(
            f"数据集目录不存在或不是目录: {root}。请检查 --source_path / -s。"
        )
    if (root / "sparse").is_dir():
        sparse = root / "sparse" / "0"
        if not any(
            all((sparse / f"{name}.{ext}").is_file() for name in ("cameras", "images"))
            for ext in ("bin", "txt")
        ):
            raise ValueError(
                f"COLMAP 相机文件不完整: {sparse}。需要 cameras.bin + images.bin，"
                "或 cameras.txt + images.txt。"
            )
        image_dir = root / (images if images is not None else "images")
        if not image_dir.is_dir():
            raise ValueError(f"图像目录不存在: {image_dir}。请检查 --images / -i。")
        if not any(
            (sparse / f"points3D.{ext}").is_file() for ext in ("ply", "bin", "txt")
        ):
            raise ValueError(f"缺少 COLMAP 点云: {sparse}/points3D.{{ply,bin,txt}}。")
        return "Colmap"
    if (root / "transforms_train.json").is_file():
        if not (root / "transforms_test.json").is_file():
            raise ValueError(
                f"Blender 数据集缺少文件: {root / 'transforms_test.json'}。"
            )
        return "Blender"
    raise ValueError(
        f"无法识别数据集格式: {root}。需要 sparse/0/ 下的 COLMAP 文件，"
        "或 transforms_train.json + transforms_test.json；请将 -s 指向场景目录。"
    )
