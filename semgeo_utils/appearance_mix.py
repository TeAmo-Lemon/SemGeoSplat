"""Mix fitted appearance endpoints on one verified, frozen Gaussian geometry."""

import hashlib
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from plyfile import PlyData, PlyElement


_XYZ = ("x", "y", "z")
_GEOMETRY = (*_XYZ, "opacity", *(f"scale_{i}" for i in range(3)),
             *(f"rot_{i}" for i in range(4)))
_DC = tuple(f"f_dc_{i}" for i in range(3))


def xyz_fingerprint(vertices):
    xyz = np.column_stack([vertices[field] for field in _XYZ]).astype(np.float32, copy=False)
    return hashlib.md5(xyz.tobytes()).hexdigest()


def regions_xyz_fingerprint(regions, regions_path):
    fingerprint = regions.get("xyz_fingerprint")
    if fingerprint:
        return fingerprint
    # Older frozen-region caches omit the fingerprint. The PLY saved
    # beside the cache supplies the ordered coordinates for that run.
    companion = Path(regions_path).with_name("point_cloud.ply")
    if companion.is_file():
        return xyz_fingerprint(PlyData.read(companion, mmap="r")["vertex"].data)
    return None


def pca_paths_match(a, b):
    if a is None or b is None:
        return False
    first, second = Path(a), Path(b)
    if first == second:
        return True
    first_tail, second_tail = first.parts[-4:], second.parts[-4:]
    return len(first_tail) == 4 and first_tail == second_tail


def _fingerprint_array(digest, array, dtype):
    array = np.ascontiguousarray(array, dtype=dtype)
    digest.update(str(array.shape).encode("ascii"))
    digest.update(memoryview(array).cast("B"))


def partition_fingerprint(labels):
    digest = hashlib.sha256()
    _fingerprint_array(digest, labels, "<i8")
    return digest.hexdigest()


def metric_space_fingerprint(space):
    """Bind a plan to its numbered region and the complete source metric space."""
    digest = hashlib.sha256()
    _fingerprint_array(digest, [space["region_id"], space["n_source_points"]], "<i8")
    _fingerprint_array(digest, [space["diameter"]], "<f8")
    for name in ("points", "distances", "measure", "features"):
        _fingerprint_array(digest, space[name], "<f4")
    return digest.hexdigest()


def _sh_fields(vertices):
    names = set(vertices.dtype.names or ())
    count = sum(name.startswith("f_rest_") for name in names)
    degree = math.isqrt(count // 3 + 1) - 1
    rest = tuple(f"f_rest_{i}" for i in range(count))
    fields = _DC + rest
    if (count != 3 * ((degree + 1) ** 2 - 1) or not set(fields) <= names
            or {name for name in names if name.startswith("f_dc_")} != set(_DC)
            or any(vertices.dtype[field].kind != "f" or vertices.dtype[field].itemsize != 4
                   for field in fields)):
        raise ValueError("PLY does not have a complete, numbered float32 RGB SH coefficient layout")
    return fields, degree


def read_endpoints(source_path, a_path, b_path):
    """Verify row alignment and all position, covariance and opacity inputs."""
    paths = map(Path, (source_path, a_path, b_path))
    source, a, b = (PlyData.read(path, mmap="r") for path in paths)
    vertices = [ply["vertex"].data for ply in (source, a, b)]
    original = vertices[0]
    fields, degree = _sh_fields(original)
    if len(original) == 0:
        raise ValueError("The source PLY has no Gaussians")

    for label, data in zip(("source", "endpoint A", "endpoint B"), vertices):
        geometry_fields = {name for name in data.dtype.names if name.startswith(("scale_", "rot_"))}
        if geometry_fields != set(_GEOMETRY[4:]):
            raise ValueError(f"The {label} PLY has an incompatible covariance layout")
    for name in _GEOMETRY:
        if name not in original.dtype.names or not np.isfinite(original[name]).all():
            raise ValueError(f"The source PLY has invalid or missing geometry field {name}")
    for label, endpoint in zip(("A", "B"), vertices[1:]):
        if len(endpoint) != len(original):
            raise ValueError(f"Endpoint {label} has a different Gaussian count")
        for name in _GEOMETRY:
            if name not in endpoint.dtype.names or not np.array_equal(endpoint[name], original[name]):
                raise ValueError(f"Endpoint {label} has different ordered geometry in field {name}")
        if _sh_fields(endpoint) != (fields, degree):
            raise ValueError(f"Endpoint {label} has an incompatible SH layout")
    return source, vertices, fields, degree


def normalize_weights(weights, point_count, strength=1.0):
    """Return [source, A, B] masses, where an N-vector supplies B's share."""
    weights = np.asarray(weights)
    if not 0 <= strength <= 1 or not math.isfinite(strength):
        raise ValueError("Strength must be finite and between zero and one")
    if weights.dtype.kind not in "bifu" or not np.isfinite(weights).all():
        raise ValueError("Appearance weights must be finite real numbers")
    if weights.shape == (point_count,):
        weights = np.column_stack((1 - weights, weights))
    if weights.shape != (point_count, 2) or (weights < 0).any() or (weights > 1).any():
        raise ValueError("Appearance weights need shape (N,) or (N,2) and entries in [0,1]")
    weights = weights.astype(np.float32, copy=False)
    totals = weights.sum(axis=1, keepdims=True)
    if (totals > 1 + 1e-6).any():
        raise ValueError("The two endpoint masses cannot add to more than one")
    weights = weights / np.maximum(totals, 1) * strength
    return np.column_stack((1 - weights.sum(axis=1), weights))


def _lift_local_shares(xyz, reps, shares):
    """Interpolate within a region using the three nearest representatives."""
    from scipy.spatial import cKDTree

    if len(reps) == 1:
        return np.full(len(xyz), shares[0], dtype=np.float32)
    distances, indices = cKDTree(reps).query(xyz, k=min(3, len(reps)))
    coincident = distances[:, :1] == 0
    inverse = distances[:, :1] / np.maximum(distances, np.finfo(distances.dtype).tiny)
    weights = np.where(coincident, distances == 0, inverse)
    return ((weights * shares[indices]).sum(axis=1) / weights.sum(axis=1)).astype(np.float32)


def weights_from_transport(vertices, regions_path, transport_path, region_ids):
    """Route the target zone to B by its conditional mass in each FGW row."""
    import torch

    regions = torch.load(regions_path, map_location="cpu", weights_only=True)
    transport = torch.load(transport_path, map_location="cpu", weights_only=True)
    fingerprint = xyz_fingerprint(vertices)
    if (regions_xyz_fingerprint(regions, regions_path) != fingerprint
            or transport.get("xyz_fingerprint", fingerprint) != fingerprint):
        raise ValueError("The region sidecar does not match the source Gaussian order")
    spaces = regions["metric_spaces"]
    diagnostics = transport["spatial_fgw_diagnostics"]
    recipe_plan = transport.get("version") == 3
    region_basis = regions.get("pca_fingerprint")
    transport_basis = transport.get("pca_fingerprint")
    pca_compatible = (region_basis == transport_basis if region_basis and transport_basis else
                      pca_paths_match(regions.get("pca_state_path"), transport.get("pca_state_path")))
    if (transport.get("correspondence_method") != "spatial_fgw"
            or len(spaces) != len(diagnostics)
            or len(spaces) != transport.get("num_regions")
            or not pca_compatible):
        raise ValueError("Use paired spatial FGW and region sidecars from the same run")
    labels = regions["cluster_labels"].numpy()
    if (labels.shape != (len(vertices),) or labels.dtype.kind not in "iu"
            or (labels < -1).any() or (labels >= len(spaces)).any()):
        raise ValueError("The region sidecar has invalid Gaussian labels")
    partition = transport.get("regions_partition_sha256")
    if recipe_plan and (not partition or not transport_basis or transport.get("xyz_fingerprint") != fingerprint
                        or transport.get("sh_degree") != _sh_fields(vertices)[1]):
        raise ValueError("The recipe plan is missing source partition, basis or SH verification")
    if partition and partition != partition_fingerprint(labels):
        raise ValueError("The transport plan was computed for a different Gaussian partition")

    weights = np.zeros((len(vertices), 2), dtype=np.float32)
    strengths = transport.get("appearance_region_strengths", {})
    for region in region_ids:
        if not 0 <= region < len(spaces):
            raise ValueError(f"Invalid region ID {region}")
        space = spaces[region]
        entry = diagnostics[region]
        if not entry or "target_zone" not in entry:
            raise ValueError(f"Region {region} has no texture_mix target zone")
        space_fingerprint = entry.get("source_metric_sha256")
        if recipe_plan and (not space_fingerprint or entry.get("source_region_id") != region):
            raise ValueError(f"Region {region} recipe plan is missing source metric verification")
        if space_fingerprint and space_fingerprint != metric_space_fingerprint(space):
            raise ValueError(f"Region {region} transport was computed for a different source metric space")
        reps = space["points"].numpy().astype(np.float64)
        measure = space["measure"].numpy().astype(np.float64)
        plan = entry["transport_plan"].numpy().astype(np.float64)
        zone = entry["target_zone"].numpy().astype(bool)
        if (space.get("region_id") != region or measure.ndim != 1 or reps.shape != (len(measure), 3)
                or zone.ndim != 1 or plan.shape != (len(reps), len(zone))
                or not np.isfinite(reps).all() or not np.isfinite(measure).all() or not np.isfinite(plan).all()
                or (plan < 0).any() or not zone.any() or zone.all()):
            raise ValueError(f"Region {region} has incompatible FGW arrays")
        mass = plan.sum(axis=1)
        if not np.allclose(mass, measure, rtol=2e-4, atol=1e-7) or not (mass > 0).any():
            raise ValueError(f"Region {region} transport rows do not match its source measure")
        positive = mass > 0
        shares = plan[positive][:, zone].sum(axis=1) / mass[positive]
        points = labels == region
        if not points.any():
            raise ValueError(f"Region {region} has no Gaussians")
        xyz = np.column_stack([vertices[field][points] for field in _XYZ])
        b_share = np.clip(_lift_local_shares(xyz, reps[positive], shares), 0, 1)
        strength = float(strengths.get(region, 1))
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise ValueError(f"Region {region} has an invalid saved appearance strength")
        weights[points, 0] = strength * (1 - b_share)
        weights[points, 1] = strength * b_share
    return weights


def mix_coefficients(vertices, fields, weights):
    """Blend every RGB SH coefficient with exact one-hot endpoints."""
    source = vertices[0]
    mixed = np.empty((len(source), len(fields)), dtype=np.float32)
    source_weight, a_weight, b_weight = weights.T
    for i, field in enumerate(fields):
        original, a_values, b_values = (data[field] for data in vertices)
        if not all(np.isfinite(values).all() for values in (original, a_values, b_values)):
            raise ValueError(f"Non-finite SH coefficient: {field}")
        mixed[:, i] = source_weight * original + a_weight * a_values + b_weight * b_values
    return mixed


def write_mixed_ply(source, output_path, fields, coefficients):
    """Export the original PLY with just its RGB SH fields replaced."""
    output_path = Path(output_path)
    vertex = source["vertex"]
    data = vertex.data.copy()
    for i, field in enumerate(fields):
        data[field] = coefficients[:, i]
    replaced = PlyElement.describe(data, vertex.name, comments=vertex.comments)
    result = PlyData([replaced if element is vertex else element for element in source.elements],
                     text=source.text, byte_order=source.byte_order, comments=source.comments,
                     obj_info=source.obj_info)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=output_path.parent, suffix=".ply.tmp", delete=False) as temp:
        temp_path = Path(temp.name)
    try:
        result.write(temp_path)
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
