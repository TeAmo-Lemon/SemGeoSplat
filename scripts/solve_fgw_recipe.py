"""Solve a local spatial FGW recipe on CPU using saved Gaussian and style features."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import SimpleNamespace

import torch
from plyfile import PlyData

from semgeo_utils.appearance_mix import (_sh_fields, metric_space_fingerprint, partition_fingerprint,
                                        pca_paths_match, read_endpoints, regions_xyz_fingerprint,
                                        xyz_fingerprint)
from style_pipeline.control import load_style_control
from style_pipeline.dino_distill import _pca_fingerprint
from style_pipeline.fgw_align import _align_spatial_regions
from style_pipeline.regions import MetricMeasureSpace


def _cached_reference(path, image):
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    image_sha = hashlib.sha256(image.read_bytes()).hexdigest()
    if payload.get("style_image_sha256") != image_sha:
        raise ValueError("The reference file SHA-256 does not match the saved style patches")
    if not payload.get("dino_model"):
        raise ValueError("The style patches do not record their DINO model")
    prototypes = payload["prototypes"]
    patch_hw = tuple(payload["patch_hw"])
    if (prototypes.ndim != 2 or min(prototypes.shape) <= 0 or not torch.isfinite(prototypes).all()
            or len(patch_hw) != 2 or min(patch_hw) < 1
            or payload["patch_assignments"].numel() != math.prod(patch_hw)):
        raise ValueError("The style prototype and patch grid have incompatible dimensions")
    return payload, image_sha


def _scene_basis(path, regions, expected_model, feature_dim, token_dim):
    # Memory mapping keeps the cached per-pixel feature tensors on disk.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if payload.get("dino_model") != expected_model:
        raise ValueError("The scene PCA and style patches use different DINO models")
    basis = payload["pca_state"]
    if (basis["components"].shape != (feature_dim, token_dim) or basis["mean"].shape != (token_dim,)
            or any(not torch.isfinite(basis[key]).all() for key in ("components", "mean"))):
        raise ValueError("The scene PCA does not match the region and DINO token dimensions")
    fingerprint = _pca_fingerprint(basis)
    recorded = regions.get("pca_fingerprint")
    if recorded:
        if recorded != fingerprint:
            raise ValueError("The scene PCA basis differs from the basis recorded in regions.pt")
        verification = "region_basis_fingerprint"
    else:
        if not pca_paths_match(regions.get("pca_state_path"), path):
            raise ValueError("Legacy regions require their recorded scene PCA cache path")
        # Before basis fingerprints, the region cache only recorded its PCA path.
        verification = "legacy_recorded_cache_path"
    return basis, fingerprint, verification


def _run(args):
    if args.output.resolve() in {path.resolve() for path in (args.regions, args.style_primitives, args.source,
                                                              args.style_image, args.control, args.pca)}:
        raise ValueError("Choose a plan output separate from the inputs")
    if not 0 <= args.alpha <= 1 or args.nodes < 1 or args.max_iter < 1:
        raise ValueError("Alpha must be in [0,1] and nodes and max iterations must be positive")
    for value in (args.prior_weight, args.entropy, args.contrast_weight):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Prior, entropy and contrast weights must be finite and non-negative")
    if (args.endpoint_a is None) != (args.endpoint_b is None):
        raise ValueError("Supply both fitted endpoints for optional checkpoint pre-validation")
    if args.endpoint_a is not None:
        if args.output.resolve() in {args.endpoint_a.resolve(), args.endpoint_b.resolve()}:
            raise ValueError("Choose a plan output separate from the fitted endpoints")
        _, vertices, _, sh_degree = read_endpoints(args.source, args.endpoint_a, args.endpoint_b)
        original = vertices[0]
    else:
        original = PlyData.read(args.source, mmap="r")["vertex"].data
        _, sh_degree = _sh_fields(original)
    xyz_sha = xyz_fingerprint(original)
    regions = torch.load(args.regions, map_location="cpu", weights_only=True, mmap=True)
    if regions_xyz_fingerprint(regions, args.regions) != xyz_sha:
        raise ValueError("The original Gaussian order does not match the region cache")
    spaces = regions["metric_spaces"]
    labels = regions["cluster_labels"]
    if (labels.shape != (len(original),) or regions.get("n_full") != len(original)
            or (labels < -1).any() or (labels >= len(spaces)).any()
            or regions.get("num_regions") != len(spaces)):
        raise ValueError("The Gaussian labels and cached region count do not match the original PLY")
    if len(set(args.region)) != len(args.region) or any(not 0 <= value < len(spaces) for value in args.region):
        raise ValueError(f"Region IDs must be unique and in [0,{len(spaces)})")

    patch, image_sha = _cached_reference(args.style_primitives, args.style_image)
    prototypes = patch["prototypes"]
    feature_dim = int(regions["feature_dim"])
    basis, pca_sha, pca_verification = _scene_basis(args.pca, regions, patch["dino_model"],
                                                    feature_dim, prototypes.shape[1])
    control = load_style_control(args.control, len(spaces), tuple(patch["patch_hw"]))
    data = json.loads(args.control.read_text(encoding="utf-8"))
    selected_spaces = []
    selected_recipe = {}
    for local_id, region in enumerate(args.region):
        if region not in control.texture_mixes:
            raise ValueError(f"Region {region} must specify texture_mix so its box can be routed to endpoint B")
        original_space = spaces[region]
        if original_space.get("region_id") != region or not (labels == region).any():
            raise ValueError(f"Region {region} has incompatible identity or no labelled Gaussians")
        selected_spaces.append(MetricMeasureSpace(**{**original_space, "region_id": local_id}))
        selected_recipe[str(local_id)] = data["regions"][str(region)]

    cached = SimpleNamespace(style_patch_hw=tuple(patch["patch_hw"]),
                             style_patch_assignments=patch["patch_assignments"],
                             style_patch_features=patch.get("patch_features"),
                             style_patch_valid_mask=patch.get("patch_valid_mask"))
    # The production stage expects compact region IDs. Remap only this small
    # in-memory subset and restore the saved scene IDs on the resulting plan.
    with NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8") as recipe:
        json.dump({"default_region_strength": data.get("default_region_strength", 1.0),
                   "regions": selected_recipe}, recipe)
        recipe.flush()
        settings = SimpleNamespace(dino_feature_dim=feature_dim, style_control_file=recipe.name,
                                   spatial_fgw_routing="local", spatial_fgw_nodes=args.nodes,
                                   spatial_fgw_prior_weight=args.prior_weight, spatial_fgw_entropy=args.entropy,
                                   spatial_fgw_max_iter=args.max_iter,
                                   spatial_fgw_contrast_weight=args.contrast_weight)
        _, result = _align_spatial_regions(selected_spaces, cached, prototypes, basis, settings, args.alpha)

    diagnostics = [None] * len(spaces)
    for region, entry in zip(args.region, result["spatial_fgw_diagnostics"]):
        diagnostics[region] = {**entry, "source_region_id": region,
                               "source_metric_sha256": metric_space_fingerprint(spaces[region])}
    payload = {
        "version": 3, "correspondence_method": "spatial_fgw", "fgw_alpha": args.alpha,
        "num_regions": len(spaces), "num_primitives": len(prototypes), "feature_dim": feature_dim,
        "xyz_fingerprint": xyz_sha, "sh_degree": sh_degree,
        "regions_partition_sha256": partition_fingerprint(labels),
        "pca_state_path": str(args.pca.resolve()), "pca_fingerprint": pca_sha,
        "pca_verification": pca_verification, "dino_model": patch["dino_model"],
        "style_image_path": str(args.style_image.resolve()), "style_image_sha256": image_sha,
        "regions_path": str(args.regions.resolve()), "control_path": str(args.control.resolve()),
        "appearance_region_strengths": {region: float(control.strengths[region]) for region in args.region},
        "spatial_fgw_diagnostics": diagnostics,
    }
    settings = {key: value for key, value in vars(settings).items() if key.startswith("spatial_fgw_")}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=args.output.parent, suffix=".pt.tmp", delete=False) as stream:
        temp_path = Path(stream.name)
    try:
        torch.save({**payload, **settings}, temp_path)
        os.replace(temp_path, args.output)
    finally:
        temp_path.unlink(missing_ok=True)
    print(f"Saved plan for regions {args.region} to {args.output} ({pca_verification})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Original Gaussian PLY")
    parser.add_argument("--regions", type=Path, required=True, help="Saved regions.pt")
    parser.add_argument("--pca", type=Path, required=True, help="Scene DINO cache with pca_state")
    parser.add_argument("--style-primitives", type=Path, required=True)
    parser.add_argument("--style-image", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="Style control JSON")
    parser.add_argument("--region", type=int, action="append", required=True, help="Scene region ID, repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint-a", type=Path)
    parser.add_argument("--endpoint-b", type=Path)
    parser.add_argument("--alpha", type=float, default=.5)
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--prior-weight", type=float, default=.08)
    parser.add_argument("--entropy", type=float, default=.01)
    parser.add_argument("--max-iter", type=int, default=120)
    parser.add_argument("--contrast-weight", type=float, default=0.)
    try:
        _run(parser.parse_args())
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
