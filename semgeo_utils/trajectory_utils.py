"""Read camera paths exported by nerfview's Save Trajectory button."""

import json
import math
from pathlib import Path


def load_trajectory(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(
            "Camera path must be a JSON object exported with Save Trajectory."
        )
    for key in ("render_width", "render_height"):
        value = data.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            or int(value) != value
        ):
            raise ValueError(f"Camera path {key} must be a positive integer.")
        data[key] = int(value)
    fps = data.get("fps", 30)
    if (
        isinstance(fps, bool)
        or not isinstance(fps, (int, float))
        or not math.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("Camera path fps must be positive and finite.")
    frames = data.get("camera_path")
    if not isinstance(frames, list) or not frames:
        raise ValueError(
            "Camera path has no frames; add at least two keyframes and Save Trajectory again."
        )
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Camera path frame {index} must be an object.")
        fov = frame.get("fov")
        if (
            isinstance(fov, bool)
            or not isinstance(fov, (int, float))
            or not math.isfinite(fov)
            or not 0 < fov < 180
        ):
            raise ValueError(
                f"Camera path frame {index}: fov must be in (0, 180) degrees."
            )
        # Validate before loading the model or allocating GPU memory.
        restore_viewer_c2w(frame.get("camera_to_world"))
    return data


def restore_viewer_c2w(matrix):
    """Undo nerfview's default scale_ratio=10 and OpenCV-to-OpenGL export.

    viewer.py passes model coordinates directly to nerfview. Save Trajectory
    divides translations by 10 and right-multiplies the rotation by Rx(pi).
    Restore both to match the camera used by viewer.py's render_fn.
    """
    if (
        isinstance(matrix, list)
        and len(matrix) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in matrix)
    ):
        matrix = [value for row in matrix for value in row]
    if (
        not isinstance(matrix, list)
        or len(matrix) != 16
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in matrix
        )
    ):
        raise ValueError(
            "camera_to_world must contain 16 finite numbers (flat or 4x4)."
        )
    if any(
        abs(matrix[12 + i] - expected) > 1e-5 for i, expected in enumerate((0, 0, 0, 1))
    ):
        raise ValueError("camera_to_world must have homogeneous last row [0, 0, 0, 1].")
    rows = [list(map(float, matrix[i : i + 4])) for i in range(0, 16, 4)]
    for row in rows[:3]:
        row[1] *= -1
        row[2] *= -1
        row[3] *= 10
    for i in range(3):
        for j in range(3):
            dot = sum(rows[k][i] * rows[k][j] for k in range(3))
            if abs(dot - int(i == j)) > 1e-3:
                raise ValueError("camera_to_world rotation must be orthonormal.")
    return rows


def trajectory_cache_complete(output_dir, signature, frame_count, outputs):
    """Require the same render inputs and every numbered output before reuse."""
    output_dir = Path(output_dir)
    try:
        if json.loads((output_dir / "trajectory_render.json").read_text()) != signature:
            return False
    except (OSError, ValueError):
        return False
    return all(
        (output_dir / folder / f"{index:05d}{suffix}").is_file()
        for folder, suffix in outputs
        for index in range(frame_count)
    )
