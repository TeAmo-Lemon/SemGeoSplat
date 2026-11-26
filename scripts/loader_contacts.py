"""Select observed loader contacts and trim splats that bleed through the tabletop."""

from typing import NamedTuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


class LoaderContactPlan(NamedTuple):
    # indices address the full source PLY; rows address the isolated object PLY.
    indices: np.ndarray
    adjusted_rows: np.ndarray
    log_scales: np.ndarray
    rotations_wxyz: np.ndarray


def _track_frame(vertices, recipe):
    basis = np.asarray(recipe["world_to_local"])
    heading = np.radians(recipe["heading_degrees"])
    across = np.cos(heading) * basis[0] + np.sin(heading) * basis[1]
    along = -np.sin(heading) * basis[0] + np.cos(heading) * basis[1]
    p, q, intercept = recipe["support_plane"]
    height_axis = basis[2] - p * basis[0] - q * basis[1]
    xyz = np.column_stack([vertices[name] for name in "xyz"])
    displacement = xyz - basis.T @ np.asarray(recipe["offset"])
    height = displacement @ height_axis - intercept
    return xyz, displacement @ across, displacement @ along, height, across, height_axis


def _wheel_profile(along, contacts):
    # The straight span joins the bottom of the measured front and rear wheel arcs.
    front, rear = contacts["axle_positions"]
    radius = contacts["wheel_radius"]
    outside_axles = np.maximum(np.maximum(front - along, along - rear), 0.0)
    rise = radius - np.sqrt(np.maximum(radius**2 - outside_axles**2, 0.0))
    return outside_axles, rise


def plan_loader_contacts(original, region_indices, recipe):
    """Compute one geometry plan from the native-color source and its cleaned region.

    The approach is scoped to the Kitchen loader and its registered orbit recipe.
    The broad region already protects the bucket, cabin, upper track and wheels.
    Only the contact zone is reopened to recover adjacent, dark measured splats.
    """
    contacts = recipe.get("track_contacts")
    if contacts is None:
        raise ValueError("The loader contact repair requires a calibrated track_contacts recipe")
    xyz, across, along, height, across_axis, height_axis = _track_frame(original, recipe)
    gap, rise = _wheel_profile(along, contacts)
    (left_min, left_max), (right_min, right_max) = contacts["across_ranges"]
    left = (across > left_min) & (across < left_max)
    right = (across > right_min) & (across < right_max)
    inside_track = (left | right) & (gap < contacts["max_arc_offset"]) & (height >= 0.001 + rise)

    max_rgb = 0.5 + 0.2820947918 * np.maximum.reduce([original[f"f_dc_{i}"] for i in range(3)])
    max_scale = np.exp(np.maximum.reduce([original[f"scale_{i}"] for i in range(3)]))
    below_chassis = (along > contacts["chassis_front"]) & (height < 0.075)
    region = np.zeros(len(original), dtype=bool)
    region[region_indices] = True
    keep = region & (~below_chassis | (inside_track & ((height > 0.053) | (max_rgb < 0.48))))

    # Some measured contacts belong to adjacent region labels. Geometry, source
    # color and proximity jointly restrict this extension to the existing track.
    seeds = keep & below_chassis
    if not seeds.any():
        raise ValueError("The region does not intersect the loader's calibrated contact zone")
    nearby = below_chassis & inside_track & (height < 0.055) & (max_rgb < 0.35) & (max_scale < 0.045)
    candidates = np.flatnonzero(nearby & ~keep)
    distances = cKDTree(xyz[seeds]).query(xyz[candidates])[0]
    keep[candidates[distances < 0.025]] = True
    indices = np.flatnonzero(keep)

    rows = np.flatnonzero((along[indices] > contacts["chassis_front"]) & (height[indices] < 0.10))
    points = indices[rows]
    scales = np.exp(np.column_stack([original[f"scale_{i}"][points] for i in range(3)]))
    quaternions = np.column_stack([original[f"rot_{i}"][points] for i in (1, 2, 3, 0)])
    rotation = Rotation.from_quat(quaternions).as_matrix()
    covariance = (rotation * scales[:, None, :]**2) @ rotation.transpose(0, 2, 1)

    # Restrict only the covariance in the contact-normal and lateral directions.
    # The central tread span and its tangential shape remain free along the belt.
    (left_min, left_max), (right_min, right_max) = contacts["covariance_across_ranges"]
    x = across[points]
    edge_clearance = np.where(
        x < 0.5 * (left_max + right_min),
        np.minimum(x - left_min, left_max - x),
        np.minimum(x - right_min, right_max - x),
    )
    height_limit = np.clip((height[points] - (0.002 + rise[points]) + 0.0025) / 2.8, 0.0015, 0.020)
    width_limit = np.clip((edge_clearance + 0.001) / 2.8, 0.002, 0.024)
    changed = np.zeros(len(points), dtype=bool)
    for normal, limit in ((height_axis / np.linalg.norm(height_axis), height_limit), (across_axis, width_limit)):
        sigma = np.sqrt(np.maximum(np.einsum("i,nij,j->n", normal, covariance, normal), 1e-18))
        factor = np.minimum(limit / sigma, 1.0)
        changed |= factor < 0.99999
        transform = np.eye(3) + (factor - 1.0)[:, None, None] * np.outer(normal, normal)
        covariance = transform @ covariance @ transform

    vals, axes = np.linalg.eigh(covariance[changed])
    # Each covariance has a right-handed eigenbasis for the PLY quaternion.
    axes[:, :, 0] *= np.where(np.linalg.det(axes) < 0, -1.0, 1.0)[:, None]
    quaternions = Rotation.from_matrix(axes).as_quat() if changed.any() else np.empty((0, 4))
    return LoaderContactPlan(
        indices, rows[changed], np.log(np.sqrt(np.maximum(vals, 1e-14))), quaternions[:, [3, 0, 1, 2]]
    )


def apply_loader_contacts(appearance, plan):
    """Apply the same index set and covariance edits to any registered appearance."""
    result = appearance[plan.indices].copy()
    for i in range(3):
        result[f"scale_{i}"][plan.adjusted_rows] = plan.log_scales[:, i]
    for i in range(4):
        result[f"rot_{i}"][plan.adjusted_rows] = plan.rotations_wxyz[:, i]
    return result


def limit_contact_edges(cloud, recipe):
    """Clip the isolated cloud to the measured tread faces and ground-contact arc.

    The signed radial distance is positive outside the wheel arcs. Between the
    axles, its zero contour follows the straight bottom edge of the track.
    """
    envelope = recipe["track_contacts"]["edge_envelope"]
    _, across, along, height, across_axis, height_axis = _track_frame(cloud, recipe)

    left, right = envelope["center_offsets"]
    center = np.where(across < 0, left, right)
    center = center - envelope["across_drift"] * (along - envelope["along_pivot"])
    lateral_clearance = envelope["half_width"] - np.abs(across - center)
    front, rear = envelope["axle_positions"]
    beyond_axle = along - np.clip(along, front, rear)
    radius = envelope["wheel_radius"]
    axle_height = envelope["contact_height"] + (across > 0) * envelope["right_height_offset"] + radius
    radial_distance = np.hypot(beyond_axle, height - axle_height)
    radial_excess = radial_distance - radius

    start, end = envelope["lower_run_extent"]
    lower_track = (
        (along > start) & (along < end)
        & (np.abs(across - center) < 0.135) & (height < 0.14)
    )
    keep = ~(lower_track & ((radial_excess > 0.017) | (lateral_clearance < -0.006)))
    rows = np.flatnonzero(lower_track & keep)
    result = cloud[keep]
    if not len(rows):
        return result

    scales = np.exp(np.column_stack([cloud[f"scale_{i}"][rows] for i in range(3)]))
    quaternions = np.column_stack([cloud[f"rot_{i}"][rows] for i in (1, 2, 3, 0)])
    axes = Rotation.from_quat(quaternions).as_matrix()
    # Allow larger ellipsoids on the wheel faces above the contact surface.
    at_contact = (radial_excess[rows] > -0.011) & (height[rows] < 0.117)
    max_scale = np.where(at_contact, 0.0105, 0.021)
    scales = np.minimum(scales, max_scale[:, None])
    covariance = (axes * scales[:, None, :]**2) @ axes.transpose(0, 2, 1)

    basis = np.asarray(recipe["world_to_local"])
    heading = np.deg2rad(recipe["heading_degrees"])
    along_axis = -np.sin(heading) * basis[0] + np.cos(heading) * basis[1]
    normal = (
        beyond_axle[rows, None] * along_axis
        + (height[rows, None] - axle_height[rows, None]) * height_axis
    ) / np.maximum(radial_distance[rows, None], 1e-7)
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    lateral = np.broadcast_to(across_axis, (len(rows), 3))
    radial_limit = np.clip((0.014 - radial_excess[rows]) / 3, 0.001, 0.015)
    lateral_limit = np.clip((lateral_clearance[rows] + 0.001) / 3, 0.001, 0.016)
    for direction, limit in ((normal, radial_limit), (lateral, lateral_limit)):
        sigma = np.sqrt(np.einsum("ni,nij,nj->n", direction, covariance, direction))
        factor = np.minimum(limit / sigma, 1) - 1
        transform = np.eye(3) + factor[:, None, None] * direction[:, :, None] * direction[:, None, :]
        covariance = transform @ covariance @ transform.transpose(0, 2, 1)

    values, axes = np.linalg.eigh(covariance)
    axes[:, :, 0] *= np.where(np.linalg.det(axes) < 0, -1, 1)[:, None]
    rotations = Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]]
    target = np.flatnonzero(lower_track[keep])
    for i in range(3):
        result[f"scale_{i}"][target] = 0.5 * np.log(values[:, i])
    for i in range(4):
        result[f"rot_{i}"][target] = rotations[:, i]
    return result
