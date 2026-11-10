"""①~⑥ 阶段产物路径统一解析 (唯一出处, 防各阶段重复拼路径漂移).

②~⑥ 产物都在 <model_path>/point_cloud/iteration_<style_iterations>/ (① refine 产物在
--start_ply 同树的 iteration_refine/, 见 refine._resolve_refine_dir, 与本模块无关).
"""

import os


def style_output_dir(model_path: str, style_iterations: int) -> str:
    """<model_path>/point_cloud/iteration_<style_iterations>/ — ②~⑥ 产物目录."""
    return os.path.join(model_path, "point_cloud", f"iteration_{style_iterations}")


def sidecar_path(model_path: str, style_iterations: int, name: str) -> str:
    """<style_output_dir>/<name> — ②③④⑤ sidecar 文件路径."""
    return os.path.join(style_output_dir(model_path, style_iterations), name)
