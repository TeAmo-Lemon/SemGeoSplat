"""Source-preserving FGW for 3D regions and spatial samples of a style image."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class StylePatchSpace:
    patch_indices: np.ndarray
    bank_ids: np.ndarray
    positions: np.ndarray
    distances: np.ndarray
    features: np.ndarray
    prior: np.ndarray
    patch_cells: np.ndarray


def sample_style_patches(features, bank_ids, patch_hw, allowed, max_nodes, *, zone=None):
    """Choose real patches with spatial FPS and approximate area by same-bank cells."""
    ph, pw = patch_hw
    patch_indices = np.flatnonzero(allowed)
    if patch_indices.size == 0:
        raise ValueError("Reference selection contains no valid DINO patch centers")

    bank_ids = bank_ids[patch_indices]
    strata = bank_ids if zone is None else 2 * bank_ids + zone[patch_indices]
    unique_strata = np.unique(strata)
    if max_nodes < unique_strata.size:
        raise ValueError(f"Spatial FGW needs at least {unique_strata.size} nodes for the selected bank strata")
    positions = np.stack((patch_indices % pw + 0.5, patch_indices // pw + 0.5), axis=1)

    # Seed each appearance bank with an actual patch. Additional nodes cover
    # distant areas, including disconnected parts of the same bank.
    selected = []
    for key in unique_strata:
        nodes = np.flatnonzero(strata == key)
        bank_positions = positions[nodes]
        distances = ((bank_positions - bank_positions.mean(axis=0)) ** 2).sum(axis=1)
        selected.append(nodes[distances.argmin()])

    nearest = ((positions[:, None] - positions[selected]) ** 2).sum(axis=2).min(axis=1)
    nearest[selected] = -1
    for _ in range(len(selected), min(max_nodes, len(patch_indices))):
        next_node = int(nearest.argmax())
        selected.append(next_node)
        nearest = np.minimum(nearest, ((positions - positions[next_node]) ** 2).sum(axis=1))
        nearest[next_node] = -1

    cells = ((positions[:, None] - positions[selected]) ** 2).sum(axis=2)
    cells[strata[:, None] != strata[selected]] = np.inf
    assignments = cells.argmin(axis=1)
    prior = np.bincount(assignments, minlength=len(selected)) / len(patch_indices)
    patch_cells = np.full(ph * pw, -1, dtype=np.int64)
    patch_cells[patch_indices] = assignments

    sample_positions = positions[selected]
    distances = np.linalg.norm(sample_positions[:, None] - sample_positions[None, :], axis=2)
    distances /= max(distances.max(), 1.0)

    selected_indices = patch_indices[selected]
    return StylePatchSpace(
        patch_indices=selected_indices,
        bank_ids=bank_ids[selected],
        positions=sample_positions / np.array([pw, ph]),
        distances=distances,
        features=features[selected_indices],
        prior=prior,
        patch_cells=patch_cells,
    )


def _make_local_contrast(source_distances, source_contrast, target_contrast, source_mass):
    """Return a distortion evaluator on connected local source edges."""
    geodesics = np.where(source_distances <= 1.0, source_distances, np.inf)
    np.fill_diagonal(geodesics, np.inf)
    neighbors = np.argsort(geodesics, axis=1, kind="stable")[:, :6]
    rows = np.broadcast_to(np.arange(len(source_mass))[:, None], neighbors.shape)
    connected = np.isfinite(geodesics[rows, neighbors])
    edges = np.zeros(geodesics.shape, dtype=np.float64)
    edges[rows[connected], neighbors[connected]] = 1.0
    edges = np.maximum(edges, edges.T)
    mass = source_mass @ edges @ source_mass
    if mass == 0:
        return None

    edges /= mass
    weighted_source = edges * source_contrast
    source_term = source_mass @ (weighted_source * source_contrast) @ source_mass
    target_squared = target_contrast ** 2

    def evaluate(plan):
        interaction = (edges @ plan @ target_squared
                       - 2 * weighted_source @ plan @ target_contrast)
        # The source-only term is constant on the fixed-row-marginal simplex.
        return float(source_term + np.sum(plan * interaction)), 2 * interaction

    return evaluate


def _conditional_rows(logits):
    peak = logits.max(axis=1, keepdims=True)
    weights = np.exp(np.maximum(logits - peak, -690.0))
    totals = weights.sum(axis=1, keepdims=True)
    return weights / totals, (peak + np.log(totals))[:, 0]


def _make_zone_projection(zone, fraction, source_mass):
    """KL project mirror steps onto fixed source rows and one target zone mass."""
    # Work with the smaller target mass to keep occupancy checks stable near one.
    if fraction > 0.5:
        zone = ~zone
        fraction = 1 - fraction
    outside = ~zone
    logit_fraction = np.log(fraction) - np.log1p(-fraction) if fraction > 0 else None

    def project(logits):
        logits = np.asarray(logits, dtype=np.float64)
        proposal = np.zeros_like(logits)
        conditional_outside, log_outside = _conditional_rows(logits[:, outside])
        if fraction == 0:
            proposal[:, outside] = conditional_outside * source_mass[:, None]
            return proposal

        conditional_inside, log_inside = _conditional_rows(logits[:, zone])
        odds = log_inside - log_outside
        lower = odds.min() - logit_fraction
        upper = odds.max() - logit_fraction
        # A shared logit shift preserves the relative weights inside each group.
        # These bounds make every row's inside probability bracket the fraction.
        for _ in range(64):
            if upper - lower <= 1e-12:
                break
            shift = lower + (upper - lower) * 0.5
            inside = np.exp(-np.logaddexp(0, shift - odds))
            if source_mass @ inside > fraction:
                lower = shift
            else:
                upper = shift
        shifted = odds - (lower + (upper - lower) * 0.5)
        inside = np.exp(-np.logaddexp(0, -shifted))
        outside_mass = np.exp(-np.logaddexp(0, shifted))
        proposal[:, zone] = conditional_inside * (source_mass * inside)[:, None]
        proposal[:, outside] = conditional_outside * (source_mass * outside_mass)[:, None]
        return proposal

    return project


def _partition_zone_fractions(fraction, bias, near_mass, far_mass):
    """Allocate the same target-zone mass preferentially to one source group."""
    if bias == 0.0 or fraction in (0.0, 1.0):
        return fraction, fraction

    preferred, other = (near_mass, far_mass) if bias > 0 else (far_mass, near_mass)
    capacity_preferred = preferred * (1 - fraction)
    capacity_other = other * fraction
    # At maximal bias, at least one group's quota reaches a true endpoint.
    # Compute that group's normalized change directly even for very small masses.
    if capacity_preferred <= capacity_other:
        max_increase = 1 - fraction
    else:
        max_increase = capacity_other / preferred
    if capacity_other <= capacity_preferred:
        max_decrease = fraction
    else:
        max_decrease = capacity_preferred / other
    preferred_fraction = float(np.clip(fraction + abs(bias) * max_increase, 0.0, 1.0))
    other_fraction = float(np.clip(fraction - abs(bias) * max_decrease, 0.0, 1.0))
    return ((preferred_fraction, other_fraction) if bias > 0
            else (other_fraction, preferred_fraction))


def _make_partition_zone_projection(zone, source_near, source_mass, near_fraction, far_fraction):
    source_far = ~source_near
    near_mass = source_mass[source_near].sum()
    far_mass = source_mass[source_far].sum()
    near_project = _make_zone_projection(zone, near_fraction, source_mass[source_near] / near_mass)
    far_project = _make_zone_projection(zone, far_fraction, source_mass[source_far] / far_mass)

    def project(logits):
        proposal = np.empty(logits.shape, dtype=np.float64)
        proposal[source_near] = near_mass * near_project(logits[source_near])
        proposal[source_far] = far_mass * far_project(logits[source_far])
        return proposal

    return project


def _make_moment_projection(zone, fraction, mass, coordinate, bias):
    """KL projection at fixed ROI mass and first moment on the source region."""
    from scipy.optimize import brentq
    from scipy.special import expit, logit

    neutral = fraction * (mass @ coordinate)
    if fraction in (0., 1.) or np.ptp(coordinate) == 0:
        if np.ptp(coordinate) == 0 and bias and 0 < fraction < 1:
            raise ValueError("Centroid control needs distinct positive-mass source coordinates")
        return _make_zone_projection(zone, fraction, mass), float(neutral)

    # The extremum fills shells in order, sharing the remaining budget among
    # tied coordinates. A partially filled shell still has free patch choices.
    order = np.argsort(-coordinate if bias >= 0 else coordinate, kind="stable")
    cut = coordinate[order[min(np.searchsorted(np.cumsum(mass[order]), fraction), len(order) - 1)]]
    full = coordinate > cut if bias >= 0 else coordinate < cut
    edge = coordinate == cut
    remaining = float(np.clip(fraction - mass[full].sum(), 0., mass[edge].sum()))
    extreme = mass[full] @ coordinate[full] + remaining * cut
    target = float(neutral + abs(bias) * (extreme - neutral))
    if abs(bias) == 1:
        empty = ~(full | edge)
        edge_mass = mass[edge].sum()
        edge_project = _make_zone_projection(zone, remaining / edge_mass, mass[edge] / edge_mass)

        def project_endpoint(logits):
            proposal = np.zeros_like(logits)
            for rows, cols in ((full, zone), (empty, ~zone)):
                if rows.any():
                    indices = np.ix_(rows, cols)
                    proposal[indices] = mass[rows, None] * _conditional_rows(logits[indices])[0]
            proposal[edge] = edge_mass * edge_project(logits[edge])
            return proposal

        return project_endpoint, target

    # For the interior, the two dual variables tilt the inside/outside log
    # odds by intercept + slope * coordinate. Eliminating the intercept gives
    # a monotone scalar moment equation, even for highly concentrated plans.
    centered = coordinate - mass @ coordinate
    log_odds = logit(fraction)

    def project(logits):
        inside, log_inside = _conditional_rows(logits[:, zone])
        outside, log_outside = _conditional_rows(logits[:, ~zone])
        odds = log_inside - log_outside

        def shares(slope):
            base = odds + slope * centered
            def residual(intercept):
                return mass @ expit(base + intercept) - fraction
            low, high = log_odds - base.max() - 1., log_odds - base.min() + 1.
            intercept = brentq(residual, low, high, xtol=2e-13)
            return expit(base + intercept)

        def moment_error(slope):
            return (mass * shares(slope)) @ centered - (target - neutral)

        error = moment_error(0.)
        slope = 0.
        if abs(error) > 2e-13:
            bound = -1. if error > 0 else 1.
            for _ in range(64):
                if error * moment_error(bound) <= 0:
                    break
                bound *= 2.
            slope = brentq(moment_error, min(bound, 0.), max(bound, 0.), xtol=1e-12)
        share = shares(slope)
        proposal = np.empty_like(logits)
        proposal[:, zone] = (mass * share)[:, None] * inside
        proposal[:, ~zone] = (mass * (1. - share))[:, None] * outside
        return proposal

    return project, target


def solve_patch_transport(cost, source_distances, target_distances, source_mass, target_prior,
                          alpha, prior_weight, entropy, max_iter, *, source_contrast=None,
                          target_contrast=None, contrast_weight=0.0, target_zone=None,
                          zone_fraction=None, source_near=None, source_bias=0.0, source_coordinate=None):
    """Minimize the FGW objective on source row simplices by mirror descent.

    The KL penalty is on the free target marginal. Conditional entropy
    applies within each source row. Backtracking only accepts a decrease in
    the complete objective, including global geometry and optional local
    semantic contrast distortion. An optional binary target zone fixes its
    aggregate mass while preserving the individual source row marginals. A
    source partition redistributes this fixed aggregate according to the bias,
    while the objective still couples all source rows. With a partition and
    zero bias, both source groups receive the same target-zone fraction.
    """
    if not (0 <= alpha <= 1 and np.isfinite(prior_weight) and prior_weight >= 0
            and np.isfinite(entropy) and entropy >= 0 and max_iter >= 1):
        raise ValueError("Invalid spatial FGW alpha, prior weight, entropy or iteration count")
    if not all(np.isfinite(value).all() for value in
               (cost, source_distances, target_distances, source_mass, target_prior)):
        raise ValueError("Spatial FGW requires finite costs, distances and measures")
    if (source_mass < 0).any() or source_mass.sum() <= 0 or (target_prior <= 0).any():
        raise ValueError("Spatial FGW requires positive total source mass and target prior")
    if not np.isfinite(contrast_weight) or contrast_weight < 0:
        raise ValueError("Spatial FGW contrast weight must be finite and non-negative")
    if (target_zone is None) != (zone_fraction is None):
        raise ValueError("Spatial FGW target zone and zone fraction must be provided together")
    if target_zone is not None:
        target_zone = np.asarray(target_zone)
        if (target_zone.shape != target_prior.shape or target_zone.dtype != np.bool_
                or not target_zone.any() or target_zone.all()):
            raise ValueError("Spatial FGW target zone must be a binary mask with patches on both sides")
        if (isinstance(zone_fraction, (bool, np.bool_)) or not isinstance(zone_fraction, Real)
                or not np.isfinite(zone_fraction) or not 0 <= zone_fraction <= 1):
            raise ValueError("Spatial FGW target zone fraction must be a number in [0, 1]")
        zone_fraction = float(zone_fraction)
    if (isinstance(source_bias, (bool, np.bool_)) or not isinstance(source_bias, Real)
            or not np.isfinite(source_bias) or not -1 <= source_bias <= 1):
        raise ValueError("Spatial FGW source bias must be a number in [-1, 1]")
    if source_bias != 0.0 and source_near is None and source_coordinate is None:
        raise ValueError("Spatial FGW nonzero bias requires a source partition or coordinate")
    if source_coordinate is not None:
        source_coordinate = np.asarray(source_coordinate, dtype=np.float64)
        if (target_zone is None or source_near is not None or source_coordinate.shape != source_mass.shape
                or not np.isfinite(source_coordinate).all() or (source_coordinate < 0).any()
                or (source_coordinate > 1).any()):
            raise ValueError("Centroid control needs a target zone and one [0, 1] coordinate per row")
    if source_near is not None:
        source_near = np.asarray(source_near)
        if target_zone is None:
            raise ValueError("Spatial FGW near-source mask requires a target zone and fraction")
        if source_near.shape != source_mass.shape or source_near.dtype != np.bool_:
            raise ValueError("Spatial FGW near-source mask must be binary and match source rows")

    active = source_mass > 0
    mu = source_mass[active].astype(np.float64)
    mu /= mu.sum()
    prior = target_prior / target_prior.sum()
    log_prior = np.log(prior)
    near_fraction, far_fraction = None, None
    if source_near is not None:
        source_near = source_near[active]
        near_mass = float(mu[source_near].sum())
        far_mass = float(mu[~source_near].sum())
        if near_mass <= 0 or far_mass <= 0:
            raise ValueError("Spatial FGW near-source mask must split positive source mass")
        near_fraction, far_fraction = _partition_zone_fractions(
            zone_fraction, source_bias, near_mass, far_mass)
    if source_coordinate is not None:
        coordinate = source_coordinate[active]
        project_zone, moment = _make_moment_projection(target_zone, zone_fraction, mu, coordinate, source_bias)
    elif near_fraction is not None:
        project_zone = _make_partition_zone_projection(
            target_zone, source_near, mu, near_fraction, far_fraction)
    elif target_zone is not None:
        project_zone = _make_zone_projection(target_zone, zone_fraction, mu)
    else:
        project_zone = None
    C1 = source_distances[np.ix_(active, active)]
    C2 = target_distances
    cost = cost[active]
    C2_sq = C2 ** 2
    source_structure = mu @ (C1 ** 2) @ mu
    log_mu = np.log(mu)[:, None]
    minimum_mass = 1e-300
    local_contrast = None
    if contrast_weight > 0:
        source_contrast = np.asarray(source_contrast, dtype=np.float64)
        target_contrast = np.asarray(target_contrast, dtype=np.float64)
        for matrix, distances in ((source_contrast, source_distances),
                                  (target_contrast, target_distances)):
            if (matrix.shape != distances.shape or not np.isfinite(matrix).all()
                    or not np.allclose(matrix, matrix.T)):
                raise ValueError("Spatial FGW contrast matrices must be finite, symmetric and match distances")
        source_contrast = (source_contrast + source_contrast.T) * 0.5
        target_contrast = (target_contrast + target_contrast.T) * 0.5
        local_contrast = _make_local_contrast(
            C1, source_contrast[np.ix_(active, active)], target_contrast, mu)

    def evaluate(plan):
        q = plan.sum(axis=0)
        log_q = np.log(np.maximum(q, minimum_mass))
        log_conditional = np.log(np.maximum(plan, minimum_mass)) - log_mu
        feature = np.sum(cost * plan)
        mixed = C1 @ plan @ C2
        target_structure = C2_sq @ q
        structure = source_structure + q @ target_structure - 2 * np.sum(mixed * plan)
        prior_kl = q @ (log_q - log_prior)
        conditional_entropy = -np.sum(plan * log_conditional)
        objective = ((1 - alpha) * feature + alpha * structure + prior_weight * prior_kl
                     - entropy * conditional_entropy)
        # The remaining GW gradient is constant within each source row and
        # cancels when a mirror step normalizes that row back to mu.
        gradient = ((1 - alpha) * cost + alpha * (2 * target_structure - 4 * mixed)
                    + prior_weight * (log_q - log_prior) + entropy * log_conditional)
        terms = dict(feature_cost=float(feature), structure_cost=float(structure),
                     target_kl=float(prior_kl), conditional_entropy=float(conditional_entropy))
        if local_contrast is not None:
            contrast, contrast_gradient = local_contrast(plan)
            objective += contrast_weight * contrast
            gradient += contrast_weight * contrast_gradient
            terms["contrast_cost"] = contrast
        elif contrast_weight > 0:
            terms["contrast_cost"] = 0.0
        return float(objective), gradient, terms

    plan = mu[:, None] * prior[None, :]
    if project_zone is not None:
        plan = project_zone(np.broadcast_to(log_prior, plan.shape))
    objective, gradient, terms = evaluate(plan)
    initial_objective = objective
    converged = C2.shape[0] == 1
    iterations = 0
    step = 4.0
    if not converged:
        for iterations in range(1, max_iter + 1):
            step = min(1.35 * step, 128.0)
            log_conditional = np.log(np.maximum(plan, minimum_mass)) - log_mu
            update = None
            for _ in range(20):
                logits = log_conditional - step * gradient
                logits -= logits.max(axis=1, keepdims=True)
                if project_zone is None:
                    proposal = np.exp(np.maximum(logits, -690.0))
                    proposal *= mu[:, None] / proposal.sum(axis=1, keepdims=True)
                else:
                    proposal = project_zone(logits)
                delta = proposal - plan
                next_objective, next_gradient, next_terms = evaluate(proposal)
                if next_objective <= objective + min(0.0, 1e-4 * np.sum(gradient * delta)) + 1e-12:
                    update = proposal, next_objective, next_gradient, next_terms
                    break
                step *= 0.5
            if update is None:
                break
            plan, objective, gradient, terms = update
            if np.abs(delta).sum() < 1e-7:
                converged = True
                break

    full_plan = np.zeros((len(source_mass), len(prior)), dtype=np.float64)
    full_plan[active] = plan
    if project_zone is not None:
        actual_fraction = float(plan[:, target_zone].sum())
        mass_error = abs(actual_fraction - zone_fraction)
        row_error = float(np.max(np.abs(plan.sum(axis=1) - mu)))
        if max(mass_error, row_error) > 1e-9:
            raise ValueError("Spatial FGW failed to preserve the constrained target or source mass")
        terms.update(target_zone_fraction=zone_fraction, target_zone_mass=actual_fraction,
                     target_zone_mass_error=mass_error, max_row_mass_residual=row_error)
    if source_coordinate is not None:
        actual = float(coordinate @ plan[:, target_zone].sum(axis=1))
        if abs(actual - moment) > 1e-9:
            raise ValueError("Spatial FGW failed to preserve the requested source moment")
        terms.update(source_bias=float(source_bias), target_zone_moment=moment,
                     actual_zone_moment=actual, target_zone_moment_error=abs(actual - moment))
    if source_near is not None:
        zone_mass_by_row = plan[:, target_zone].sum(axis=1)
        near_actual = float(zone_mass_by_row[source_near].sum() / near_mass)
        far_actual = float(zone_mass_by_row[~source_near].sum() / far_mass)
        partition_error = max(abs(near_actual - near_fraction), abs(far_actual - far_fraction))
        if partition_error > 1e-9:
            raise ValueError("Spatial FGW failed to preserve the source partition's target shares")
        terms.update(source_bias=float(source_bias), source_near_mass=near_mass, source_far_mass=far_mass,
                     target_zone_near_fraction=near_fraction, target_zone_far_fraction=far_fraction,
                     target_zone_near_mass_fraction=near_actual, target_zone_far_mass_fraction=far_actual,
                     target_zone_partition_error=partition_error)
    return full_plan, dict(terms, objective_initial=initial_objective, objective=objective,
                           iterations=iterations, converged=converged)
