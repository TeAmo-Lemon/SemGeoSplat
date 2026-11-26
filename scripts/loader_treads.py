"""Complete the Kitchen loader's ground-contact runs with measured upper tread links."""

from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation


class TreadPlan(NamedTuple):
    retained: np.ndarray
    donors: np.ndarray
    centers: np.ndarray
    log_scales: np.ndarray
    rotations_wxyz: np.ndarray


def make_tread_plan(original, recipe):
    """Map a four-link upper section onto the straight and curved parts of both lower runs.

    Coordinates and distances below are calibrated for the registered Kitchen loader.
    Plan indices address the isolated loader after its contact splats have been trimmed.
    """
    xyz = np.column_stack([original[key] for key in 'xyz'])
    basis = np.asarray(recipe['world_to_local'])
    heading = np.deg2rad(recipe['heading_degrees'])
    p, q, plane_offset = recipe['support_plane']
    frame = np.stack([
        np.cos(heading) * basis[0] + np.sin(heading) * basis[1],
        -np.sin(heading) * basis[0] + np.cos(heading) * basis[1],
        basis[2] - p * basis[0] - q * basis[1],
    ])
    inverse = np.linalg.inv(frame)
    origin = basis.T @ recipe['offset']
    across, along, height = ((xyz - origin) @ frame.T - [0, 0, plane_offset]).T

    # Sample a single measured repeat from the visible upper tread. Subtract its sag
    # so that links on the straight lower run share the same supporting plane.
    donor_start, repeat_length = -0.044, 0.200
    upper_height = 0.271 + 0.72 * (along - 0.082)**2
    distance_to_upper = height - upper_height
    rgb = np.column_stack([original[f'f_dc_{i}'] for i in range(3)]) * 0.2820947918 + 0.5
    scales = np.exp(np.column_stack([original[f'scale_{i}'] for i in range(3)]))
    donor_mask = (
        (along >= donor_start) & (along < donor_start + repeat_length)
        & (across > 0.177) & (across < 0.348)
        & (distance_to_upper > -0.018) & (distance_to_upper < 0.026)
        & (rgb.max(axis=1) < 0.62) & (np.ptp(rgb, axis=1) < 0.19) & (scales.max(axis=1) < 0.031)
    )
    sample = np.flatnonzero(donor_mask)
    if len(sample) < 100:
        raise ValueError("The loader's upper tread is missing from this selection")

    envelope = recipe["track_contacts"]["edge_envelope"]
    front, rear = envelope["axle_positions"]
    radius = envelope["wheel_radius"]
    gap = np.maximum.reduce([front - along, along - rear, np.zeros(len(along))])
    path_height = 0.022 + radius - np.sqrt(np.maximum(radius * radius - gap * gap, 0))
    wheel_zone = (along > -0.465) & (along < 0.634) & (height < 0.098)
    either_side = (across < -0.285) | (across > 0.15)
    old_contact = (height < 0.060) | ((gap < 0.16) & (height < path_height + 0.024))
    retained = np.flatnonzero(~(wheel_zone & either_side & old_contact))

    donor_rows, centers, transforms = [], [], []
    for side in (-1, 1):
        for repeat in range(-3, 4):
            distance = along[sample] + repeat * repeat_length
            angle = (np.minimum(distance - front, 0) + np.maximum(distance - rear, 0)) / radius
            in_arc = np.abs(angle) < 0.966
            if not in_arc.any():
                continue
            ids = sample[in_arc]
            angle = angle[in_arc]
            distance = distance[in_arc]
            sin, cos = np.sin(angle), np.cos(angle)
            along_run = np.clip(distance, front, rear) + radius * sin
            floor = envelope["contact_height"] + (envelope["right_height_offset"] if side == 1 else 0)
            above_plane = floor + radius * (1 - cos)

            # Correct the wheel pair's small yaw relative to the orbit recipe.
            # The reflected copy has the same tooth pitch and width as the donor.
            drift, pivot = envelope["across_drift"], envelope["along_pivot"]
            lateral = (side * (across[ids] + drift * (along[ids] - pivot))
                       + (0 if side == 1 else -0.144) - drift * (along_run - pivot))
            normal_offset = distance_to_upper[ids]
            local = np.stack([
                lateral, along_run - sin * normal_offset, above_plane + cos * normal_offset + plane_offset,
            ], axis=1)
            centers.append(local @ inverse.T + origin)
            donor_rows.append(ids)

            # Orient and shear each measured covariance into the local track frame.
            jacobian = np.zeros((len(ids), 3, 3))
            upper_slope = 1.44 * (along[ids] - 0.082)
            jacobian[:, 0, 0] = side
            jacobian[:, 0, 1] = drift * (side - cos)
            jacobian[:, 1, 1] = cos + sin * upper_slope
            jacobian[:, 1, 2] = -sin
            jacobian[:, 2, 1] = sin - cos * upper_slope
            jacobian[:, 2, 2] = cos
            transforms.append(inverse @ jacobian @ frame)

    donors = np.concatenate(donor_rows)
    quaternions = np.column_stack([original[f'rot_{i}'][donors] for i in (1, 2, 3, 0)])
    axes = np.concatenate(transforms) @ Rotation.from_quat(quaternions).as_matrix()
    covariance = (axes * scales[donors, None, :]**2) @ axes.transpose(0, 2, 1)
    eigenvalues, axes = np.linalg.eigh(covariance)
    axes[:, :, 0] *= np.where(np.linalg.det(axes) < 0, -1, 1)[:, None]
    rotations = Rotation.from_matrix(axes).as_quat()[:, [3, 0, 1, 2]]
    return TreadPlan(retained, donors, np.concatenate(centers), 0.5 * np.log(eigenvalues), rotations)


def apply_tread_plan(appearance, plan):
    """Copy the chosen appearance's matching donor coefficients onto the completed runs."""
    sections = appearance[plan.donors].copy()
    for i, key in enumerate('xyz'):
        sections[key] = plan.centers[:, i]
    for i in range(3):
        sections[f'scale_{i}'] = plan.log_scales[:, i]
    for i in range(4):
        sections[f'rot_{i}'] = plan.rotations_wxyz[:, i]
    return np.concatenate((appearance[plan.retained], sections))
