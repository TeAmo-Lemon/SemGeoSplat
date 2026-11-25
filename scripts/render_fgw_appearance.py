"""Render controlled RGB from fitted SH endpoints on the same Gaussian scene."""

import argparse
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from semgeo_utils.appearance_mix import (mix_coefficients, normalize_weights, read_endpoints,
                                        weights_from_transport, write_mixed_ply, xyz_fingerprint)


def _variant(value):
    name, separator, path = value.partition("=")
    if (not separator or not path or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", name)
            or name in {"original", "endpoint_a", "endpoint_b"}):
        raise argparse.ArgumentTypeError("Expected a unique NAME=TRANSPORT_PATH")
    return name, Path(path)


def _view_indices(specification, count, default):
    indices = [int(item) for item in specification.split(",")] if specification else default
    if not indices or len(set(indices)) != len(indices) or any(not 0 <= idx < count for idx in indices):
        raise ValueError(f"Each selected camera index must be unique and in [0, {count})")
    return indices


def _camera_data(args):
    """Return label, dimensions, focal lengths and conventional world-to-camera matrices."""
    if args.camera_npz:
        cameras = []
        for path in args.camera_npz:
            with np.load(path, allow_pickle=False) as payload:
                width, height = int(payload["width"]), int(payload["height"])
                k = payload["K"]
                if (k.shape != (3, 3) or not np.isfinite(k).all()
                        or not np.allclose(k[[0, 1], [2, 2]], (width / 2, height / 2),
                                           atol=1e-3, rtol=0)
                        or not np.allclose(k[[0, 1], [1, 0]], 0, atol=1e-7)
                        or not np.allclose(k[2], (0, 0, 1), atol=1e-7)):
                    raise ValueError(f"{path} uses intrinsics outside the project's centered pinhole renderer")
                cameras.append((str(path.resolve()), width, height, k[0, 0], k[1, 1], payload["w2c"]))
        return cameras

    if args.camera_path:
        from semgeo_utils.trajectory_utils import load_trajectory, restore_viewer_c2w

        trajectory = load_trajectory(args.camera_path)
        width, height = trajectory["render_width"], trajectory["render_height"]
        frames = trajectory["camera_path"]
        cameras = []
        for index in _view_indices(args.views, len(frames), list(range(len(frames)))):
            frame = frames[index]
            c2w = np.asarray(restore_viewer_c2w(frame["camera_to_world"]))
            focal = height / (2 * math.tan(math.radians(frame["fov"]) / 2))
            cameras.append((f"{args.camera_path}:{index}", width, height, focal, focal, np.linalg.inv(c2w)))
        return cameras

    if args.cameras_json:
        frames = json.loads(args.cameras_json.read_text(encoding="utf-8"))
        cameras = []
        for index in _view_indices(args.views, len(frames), [0]):
            frame = frames[index]
            c2w = np.eye(4)
            c2w[:3, :3] = frame["rotation"]
            c2w[:3, 3] = frame["position"]
            cameras.append((frame["img_name"], int(frame["width"]), int(frame["height"]),
                            float(frame["fx"]), float(frame["fy"]), np.linalg.inv(c2w)))
        return cameras

    return []


def _make_camera(data, max_size):
    import torch

    from scene.cameras import MiniCam

    name, width, height, fx, fy, w2c = data
    if (min(width, height, fx, fy) <= 0 or not np.isfinite((fx, fy)).all()
            or w2c.shape != (4, 4) or not np.isfinite(w2c).all()
            or not np.allclose(w2c[3], (0, 0, 0, 1), atol=1e-5)
            or not np.allclose(w2c[:3, :3].T @ w2c[:3, :3], np.eye(3), atol=1e-3)):
        raise ValueError(f"Invalid camera data: {name}")
    limit = min(1, max_size / max(width, height)) if max_size else 1
    w2c = torch.tensor(w2c.T.copy(), dtype=torch.float32, device="cuda")
    camera = MiniCam(max(1, round(width * limit)), max(1, round(height * limit)),
                     2 * math.atan(height / (2 * fy)), 2 * math.atan(width / (2 * fx)),
                     0.01, 100., w2c, torch.eye(4, device="cuda"))
    camera.image_name = name
    return camera


def _load_weights(path, vertices):
    data = np.load(path, allow_pickle=False)
    if not isinstance(data, np.ndarray):
        with data:
            if str(data["xyz_fingerprint"].item()) != xyz_fingerprint(vertices):
                raise ValueError("The weight file does not match the source Gaussian order")
            return data["weights"]
    return data


class AppearanceRenderer:
    def __init__(self, args, camera_data, degree):
        import torch

        from gaussian_renderer import GaussianModel

        self.gaussians = GaussianModel(degree)
        self.gaussians.load_ply(str(args.source))
        self.cameras = [_make_camera(item, args.max_size) for item in camera_data]
        self.pipeline = SimpleNamespace(convert_SHs_python=False, antialiasing=args.antialiasing)
        self.background = torch.full((3,), float(args.white_background), device="cuda")
        self.dark_background = torch.zeros_like(self.background)

    def prepare_endpoints(self, vertices, fields):
        import torch
        from numpy.lib.recfunctions import structured_to_unstructured

        # Packing once avoids rescanning a strided/mapped PLY for each SH
        # coefficient and for each edit. Use host tensors on smaller GPUs.
        size = len(vertices[0]) * len(fields) * 4
        device = "cuda" if 6 * size < torch.cuda.mem_get_info()[0] else "cpu"
        self.endpoints = []
        for vertex in vertices:
            matrix = structured_to_unstructured(vertex[list(fields)], dtype=np.float32)
            packed = np.array(matrix, dtype=np.float32, order="C", copy=True)
            if not np.isfinite(packed).all():
                raise ValueError("Non-finite SH coefficients in appearance endpoint")
            self.endpoints.append(torch.from_numpy(packed).to(device))

    def mix(self, weights):
        import torch

        original, a, b = self.endpoints
        w = torch.as_tensor(weights, device=original.device)
        mixed = original * w[:, :1]
        mixed.add_(a * w[:, 1:2])
        mixed.add_(b * w[:, 2:3])
        return mixed

    def render(self, coefficients, output, weights=None):
        import torch
        from torchvision.utils import save_image

        from gaussian_renderer import render

        with torch.no_grad():
            sh = torch.as_tensor(coefficients, dtype=torch.float32, device="cuda")
            self.gaussians._features_dc.copy_(sh[:, :3].unsqueeze(1))
            self.gaussians._features_rest.copy_(sh[:, 3:].reshape(len(sh), 3, -1).transpose(1, 2))
            del sh
            visibility = (torch.as_tensor(weights, dtype=torch.float32, device="cuda")
                          if weights is not None else None)
            frames = output / "renders"
            frames.mkdir(parents=True, exist_ok=True)
            if visibility is not None:
                (output / "weights").mkdir(parents=True, exist_ok=True)
            for index, camera in enumerate(self.cameras):
                image = render(camera, self.gaussians, self.pipeline, self.background,
                               no_densify_hook=True)["render"]
                save_image(image, frames / f"{index:05d}.png")
                if visibility is not None:
                    shares = render(camera, self.gaussians, self.pipeline, self.dark_background,
                                    override_color=visibility, no_densify_hook=True)["render"]
                    np.save(output / "weights" / f"{index:05d}.npy",
                            shares.permute(1, 2, 0).cpu().numpy())


def _run(args):
    sidecars = dict(args.variant) if args.variant else {"mixed": args.transport}
    if args.variant and len(sidecars) != len(args.variant):
        raise ValueError("Variant names must be unique")
    transport_mode = args.transport is not None or bool(args.variant)
    if transport_mode != bool(args.regions and args.region) or (not transport_mode and (args.regions or args.region)):
        raise ValueError("Transport input requires --regions and one or more --region IDs")
    if args.views and not (args.cameras_json or args.camera_path):
        raise ValueError("--views selects indices in --cameras-json or --camera-path")
    if args.max_size < 0:
        raise ValueError("--max-size must be non-negative")
    if args.export_ply is not None:
        if len(sidecars) != 1:
            raise ValueError("PLY export supports one variant at a time")
        if args.export_ply.resolve() in {p.resolve() for p in (args.source, args.endpoint_a, args.endpoint_b)}:
            raise ValueError("Choose a PLY output separate from all input checkpoints")
    camera_data = _camera_data(args)
    if not camera_data and args.export_ply is None:
        raise ValueError("Provide cameras for RGB rendering or --export-ply for CPU-only export")
    if camera_data and args.output_dir is None:
        raise ValueError("--output-dir is required for RGB rendering")
    if args.show_endpoints and not camera_data:
        raise ValueError("--show-endpoints needs render cameras")
    source, vertices, fields, degree = read_endpoints(args.source, args.endpoint_a, args.endpoint_b)
    requested_weights = _load_weights(args.weights, vertices[0]) if args.weights else None
    renderer = AppearanceRenderer(args, camera_data, degree) if camera_data else None
    if renderer is not None:
        (args.output_dir / "appearance_mix.json").unlink(missing_ok=True)
        renderer.prepare_endpoints(vertices, fields)
    variant_stats = {}
    for name, transport in sidecars.items():
        raw_weights = (weights_from_transport(vertices[0], args.regions, transport, args.region)
                       if transport is not None else requested_weights)
        weights = normalize_weights(raw_weights, len(vertices[0]), args.strength)
        coefficients = (renderer.mix(weights) if renderer is not None
                        else mix_coefficients(vertices, fields, weights))
        if args.export_ply is not None:
            export = coefficients.cpu().numpy() if renderer is not None else coefficients
            write_mixed_ply(source, args.export_ply, fields, export)
        if renderer is not None:
            renderer.render(coefficients, args.output_dir / name, None if args.no_save_weights else weights)
        del coefficients
        active = weights[:, 1:].sum(axis=1) > 0
        stats = {"controlled_gaussians": int(active.sum())}
        if active.any():
            stats["endpoint_b_share_quantiles"] = np.quantile(
                weights[active, 2] / weights[active, 1:].sum(axis=1), (0, .1, .5, .9, 1)).tolist()
        if transport is not None:
            stats["transport"] = str(transport.resolve())
        variant_stats[name] = stats
        print(f"[{name}] {stats['controlled_gaussians']:,} controlled Gaussians", flush=True)
    if args.show_endpoints:
        for name, sh in zip(("original", "endpoint_a", "endpoint_b"), renderer.endpoints):
            renderer.render(sh, args.output_dir / name)
    if renderer is not None:
        metadata = {
            "source": str(args.source.resolve()), "endpoint_a": str(args.endpoint_a.resolve()),
            "endpoint_b": str(args.endpoint_b.resolve()), "num_gaussians": len(vertices[0]),
            "xyz_fingerprint": xyz_fingerprint(vertices[0]), "gaussian_geometry_verified": True,
            "strength": args.strength, "regions": args.region, "variants": variant_stats,
            "cameras": [item[0] for item in camera_data],
            "weight_channels": ["original", "endpoint_a", "endpoint_b"],
            "weight_units": "visible compositing mass before background",
        }
        if args.weights:
            metadata["external_weights"] = str(args.weights.resolve())
        (args.output_dir / "appearance_mix.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--endpoint-a", required=True, type=Path)
    parser.add_argument("--endpoint-b", required=True, type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--weights", type=Path, help="N-vector of B shares or an N×2 array of A/B masses")
    modes.add_argument("--transport", type=Path, help="Spatial FGW correspondence sidecar")
    modes.add_argument("--variant", type=_variant, action="append", default=[], metavar="NAME=PATH",
                       help="Named correspondence sidecar, repeatable for paired renders")
    parser.add_argument("--regions", type=Path, help="Region sidecar paired with the FGW plans")
    parser.add_argument("--region", type=int, action="append", default=[])
    parser.add_argument("--strength", type=float, default=1.)
    cameras = parser.add_mutually_exclusive_group()
    cameras.add_argument("--camera-npz", type=Path, action="append", help="Saved render.py camera, repeatable")
    cameras.add_argument("--cameras-json", type=Path, help="Project scene cameras.json")
    cameras.add_argument("--camera-path", type=Path, help="Saved nerfview trajectory")
    parser.add_argument("--views", help="Comma-separated zero-based indices into a JSON camera list")
    parser.add_argument("--max-size", type=int, default=960, help="Limit the longest side (0 for native size)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-ply", type=Path, help="Optional mixed PLY, also supported without cameras")
    parser.add_argument("--show-endpoints", action="store_true", help="Also render original appearance and endpoints")
    parser.add_argument("--no-save-weights", action="store_true")
    parser.add_argument("--antialiasing", action="store_true")
    parser.add_argument("--white-background", action="store_true")
    try:
        _run(parser.parse_args())
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
