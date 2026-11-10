"""Project-local workarounds for nerfview 0.1.3's keyframe and MP4 export bugs."""

from __future__ import annotations

import ast
from fractions import Fraction
import functools
import inspect
from pathlib import Path
import textwrap


class MP4Writer:
    """The append_data/close interface used by nerfview, with explicit H.264."""

    def __init__(self, path, fps=30):
        import av

        self._av = av
        rate = Fraction(str(fps)).limit_denominator(100000)
        if rate <= 0:
            raise ValueError("Video FPS must be positive.")
        self._container = av.open(str(path), mode="w", format="mp4")
        self._closed = False
        self._size = None
        try:
            self._stream = self._container.add_stream("libx264", rate=rate)
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "18", "preset": "medium"}
        except Exception:
            self._container.close()
            self._closed = True
            raise

    def append_data(self, image):
        import numpy as np

        if self._closed:
            raise ValueError("Video writer is closed.")
        try:
            image = np.asarray(image)
            if (
                image.ndim != 3
                or image.shape[2] not in (3, 4)
                or image.dtype != np.uint8
            ):
                raise ValueError("Video frames must be uint8 RGB or RGBA images.")
            height, width = image.shape[:2]
            if self._size is None:
                self._size = (width, height)
                self._stream.width = width + width % 2
                self._stream.height = height + height % 2
            elif self._size != (width, height):
                raise ValueError("All video frames must have the same resolution.")
            # H.264 yuv420p needs even dimensions; pad instead of resizing the scene.
            image = np.pad(image[:, :, :3], ((0, height % 2), (0, width % 2), (0, 0)))
            frame = self._av.VideoFrame.from_ndarray(
                np.ascontiguousarray(image), format="rgb24"
            )
            for packet in self._stream.encode(frame):
                self._container.mux(packet)
        except Exception:
            self.close()
            raise

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            if self._size is not None:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        finally:
            self._container.close()


class _PanelImageIO:
    """Replace only nerfview's MP4 writer, without changing global imageio APIs."""

    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def get_writer(self, path, *args, **kwargs):
        if Path(path).suffix.lower() == ".mp4" and not args and set(kwargs) <= {"fps"}:
            return MP4Writer(path, **kwargs)
        return self._original.get_writer(path, *args, **kwargs)


def _patch_keyframe_delete(camera_path_class):
    """Replace the buggy nested callback while retaining upstream UI controls.

    The callback is inside add_camera, so it cannot be overridden as a method.
    Match that specific callback in its AST rather than vendoring the full UI
    or editing site-packages. Already fixed versions are left alone.
    """
    original = camera_path_class.add_camera
    if getattr(original, "_semgeo_stable_delete", False):
        return
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(statement, ast.Assert)
            and isinstance(statement.test, ast.Compare)
            and isinstance(statement.test.left, ast.Name)
            and statement.test.left.id == "keyframe_id"
            for statement in node.body
        ):
            matches.append(node)
    if not matches:
        return
    if len(matches) != 1:
        raise RuntimeError(
            "Unexpected nerfview keyframe callback layout; cannot apply compatibility fix."
        )
    matches[0].body = ast.parse(
        textwrap.dedent("""
        current = self._keyframes.pop(keyframe_index, None)
        modal.close()
        if current is None:
            return
        current[1].remove()
        camera_edit_panel.remove()
        if self._camera_edit_panel is camera_edit_panel:
            self._camera_edit_panel = None
        self.update_spline()
    """)
    ).body
    ast.fix_missing_locations(tree)
    namespace = original.__globals__.copy()
    exec(compile(tree, "<semgeo nerfview keyframe fix>", "exec"), namespace)
    patched = functools.wraps(original)(namespace[original.__name__])
    patched._semgeo_stable_delete = True
    camera_path_class.add_camera = patched


def install_nerfview_compat():
    """Call before creating the nerfview Viewer; changes last for this process."""
    from nerfview import render_panel

    _patch_keyframe_delete(render_panel.CameraPath)
    if not isinstance(render_panel.imageio, _PanelImageIO):
        render_panel.imageio = _PanelImageIO(render_panel.imageio)
