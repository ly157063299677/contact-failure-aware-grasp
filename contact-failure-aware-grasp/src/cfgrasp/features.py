"""Numerical state features; simulator measurements must be supplied by an adapter.

The manuscript does not specify the reduction/channel conventions. This module
defines one explicit, versioned implementation contract; see docs/features.md.
No candidate-selection mask enters these state-only features.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")
FEATURE_SCHEMA_VERSION = "cfgrasp-state-v1"
DIAGNOSTIC_CHANNELS = (
    "required_contact_any", "missing_required_contact", "contact_valid_fraction",
    "mean_separation_over_scale", "separation_valid", "required_contact_persistence",
    "mean_normal_force", "normal_force_valid", "mean_tangential_force",
    "tangential_force_valid", "mean_tangential_speed", "tangential_speed_valid",
)
GLOBAL_CHANNELS = (
    "max_translation_over_scale", "translation_valid", "max_rotation_radians",
    "rotation_valid", "mean_resultant_force_norm", "resultant_force_valid",
    "mean_resultant_torque_norm", "resultant_torque_valid",
    "mean_contact_qp_residual_norm", "contact_qp_valid",
)


def _scale(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("object_scale must be a finite positive length in metres")
    return value


def _finite_array(value, shape=None, name="array") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_rigid_transform(value, name="transform") -> np.ndarray:
    pose = _finite_array(value, (4, 4), name)
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name} must be a homogeneous rigid transform")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-5
    ):
        raise ValueError(f"{name} rotation must belong to SO(3)")
    return pose


def points_to_object_frame(points_world, object_to_world) -> np.ndarray:
    """Transform row-vector positions from world to object coordinates."""
    pose = validate_rigid_transform(object_to_world, "object_to_world")
    points = _finite_array(points_world, name="points_world")
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError("points must end in an xyz dimension")
    return (points - pose[:3, 3]) @ pose[:3, :3]


def vectors_to_object_frame(vectors_world, object_to_world) -> np.ndarray:
    """Transform directions/normals; translation must not affect vectors."""
    pose = validate_rigid_transform(object_to_world, "object_to_world")
    vectors = _finite_array(vectors_world, name="vectors_world")
    if vectors.ndim < 2 or vectors.shape[-1] != 3:
        raise ValueError("vectors must end in an xyz dimension")
    return vectors @ pose[:3, :3]


def _unit_normals(value, shape, name) -> np.ndarray:
    normals = _finite_array(value, shape, name)
    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    if (length <= 1e-12).any():
        raise ValueError(f"{name} must include nonzero normals for every point")
    return normals / length


def resample_point_cloud(points, normals, n_points=1024, seed=0):
    """Seeded uniform sampling; replacement only when the source is too small.

    Returns positions, corresponding unit normals, and source indices. The seed
    and sampling rule are implementation choices, not recovered experiment facts.
    """
    points = _finite_array(points, name="points")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points must have nonempty shape [M,3]")
    if not isinstance(n_points, (int, np.integer)) or n_points < 1:
        raise ValueError("n_points must be a positive integer")
    normals = _unit_normals(normals, points.shape, "normals")
    indices = np.random.default_rng(seed).choice(
        len(points), size=n_points, replace=len(points) < n_points
    )
    return points[indices], normals[indices], indices


def _validate_groups(groups: Sequence[Sequence[int]], count: int, width: int, name: str):
    if len(groups) != 5:
        raise ValueError(f"{name} must contain five anatomical groups")
    checked = []
    for group in groups:
        values = list(group)
        if any(not isinstance(i, (int, np.integer)) for i in values):
            raise ValueError(f"{name} indices must be integers")
        if len(values) > width:
            raise ValueError(f"{name} exceeds padding width {width}; update model contract explicitly")
        checked.append(np.asarray(values, dtype=int))
    all_indices = np.concatenate(checked)
    if len(all_indices) != count or sorted(all_indices.tolist()) != list(range(count)):
        raise ValueError(f"{name} must partition all {count} input entries exactly once")
    return checked


def normalize_and_pad(values, limits, groups, width=5):
    """Normalize finite measurements to (value-low)/(high-low), then zero-pad.

    Missing measurements (NaN) and padding receive value 0 and validity 0.
    Values outside the range are retained, not silently clipped. Joint and
    actuator vectors contain finger entries only, excluding wrist/palm entries.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("values must be a vector, with NaN allowed only for missing measurements")
    limits = _finite_array(limits, (len(values), 2), "limits")
    span = limits[:, 1] - limits[:, 0]
    if (span <= 0).any():
        raise ValueError("each limit interval must have positive width")
    groups = _validate_groups(groups, len(values), width, "groups")
    valid = np.isfinite(values)
    normalized = np.where(valid, (values - limits[:, 0]) / span, 0.0)
    output = np.zeros((5, width), dtype=np.float32)
    validity = np.zeros_like(output)
    for finger, indices in enumerate(groups):
        output[finger, :len(indices)] = normalized[indices]
        validity[finger, :len(indices)] = valid[indices]
    return output, validity


def _measurement(raw, shape, name):
    """Unknown samples are NaN; infinity is always an error."""
    if raw is None:
        return np.full(shape, np.nan)
    values = np.asarray(raw, dtype=np.float64)
    if values.shape != shape or np.isinf(values).any():
        raise ValueError(f"{name} must have shape {shape} with finite/NaN samples")
    return values


def _observed_mean(values):
    finite = np.isfinite(values)
    return (float(values[finite].mean()), 1.0) if finite.any() else (0.0, 0.0)


def aggregate_finger_diagnostics(
    *, link_finger, required_contact, contact_present, object_scale,
    separation=None, normal_force=None, tangential_force=None, tangential_speed=None,
):
    """Reduce diagnostic arrays [steps,links] to [5,12].

    contact_present uses 1=contact, 0=known absence, NaN=unknown. Supplied link
    requirements are mandatory booleans; they are independent of subset masks.
    Force/speed measurements must be NaN when no contact measurement is available;
    only samples with contact_present==1 contribute to their averages.
    """
    scale = _scale(object_scale)
    owners_raw = np.asarray(link_finger)
    if owners_raw.ndim != 1 or owners_raw.dtype.kind not in "iu":
        raise ValueError("link_finger must be an integer vector")
    owners = owners_raw.astype(int)
    if len(owners) == 0 or ((owners < 0) | (owners >= 5)).any():
        raise ValueError("link_finger values must be in [0,4]")
    required = np.asarray(required_contact)
    if required.shape != owners.shape or required.dtype.kind != "b":
        raise ValueError("required_contact must provide one explicit boolean per link")
    contacts = np.asarray(contact_present, dtype=np.float64)
    if contacts.ndim != 2 or contacts.shape[0] == 0 or contacts.shape[1] != len(owners):
        raise ValueError("contact_present must have shape [positive_steps,links]")
    if not np.isin(contacts[np.isfinite(contacts)], [0.0, 1.0]).all() or np.isinf(contacts).any():
        raise ValueError("contact_present supports only 0, 1, and NaN")
    separation = _measurement(separation, contacts.shape, "separation")
    normal_force = _measurement(normal_force, contacts.shape, "normal_force")
    tangential_force = _measurement(tangential_force, contacts.shape, "tangential_force")
    tangential_speed = _measurement(tangential_speed, contacts.shape, "tangential_speed")
    for name, measurement in (("normal_force", normal_force), ("tangential_force", tangential_force),
                              ("tangential_speed", tangential_speed)):
        if (measurement[np.isfinite(measurement)] < 0).any():
            raise ValueError(f"{name} must contain nonnegative magnitudes")
    result = np.zeros((5, 12), dtype=np.float32)
    for finger in range(5):
        owned = owners == finger
        required_owned = owned & required
        if required_owned.any():
            selected = contacts[:, required_owned]
            observed = np.isfinite(selected)
            link_observed = observed.any(axis=0)
            # A requirement is known missing only if a measured link never
            # makes contact in its available diagnostic observations.
            never_contact = ~(selected == 1).any(axis=0)
            result[finger, 0] = 1
            result[finger, 1] = np.any(link_observed & never_contact)
            result[finger, 2] = observed.mean()
            result[finger, 5] = _observed_mean(selected)[0]
        if owned.any():
            result[finger, 3:5] = _observed_mean(separation[:, owned] / scale)
            active = contacts[:, owned] == 1
            for column, measurement in ((6, normal_force), (8, tangential_force), (10, tangential_speed)):
                result[finger, column:column + 2] = _observed_mean(
                    np.where(active, measurement[:, owned], np.nan)
                )
    return result


def aggregate_global_diagnostics(
    *, object_scale, translation=None, rotation_angle=None, force_residual=None,
    torque_residual=None, qp_residual=None,
):
    """Return ten channels, with a validity flag after each global quantity.

    translation/force/torque are [steps,3]; translation is displacement from
    diagnosis start in metres. rotation_angle is [steps] in radians relative to
    diagnosis start. qp_residual is [steps] or [steps,K]. Vector norms require all
    components valid; absent samples are not interpreted as zero physical error.
    """
    scale = _scale(object_scale)
    result = np.zeros(10, dtype=np.float32)
    for index, raw, vector, maximum, divisor in (
        (0, translation, True, True, scale),
        (2, rotation_angle, False, True, 1.0),
        (4, force_residual, True, False, 1.0),
        (6, torque_residual, True, False, 1.0),
        (8, qp_residual, None, False, 1.0),
    ):
        if raw is None:
            continue
        values = np.asarray(raw, dtype=np.float64)
        if np.isinf(values).any():
            raise ValueError("global measurements may contain NaN, but not infinity")
        if vector is True:
            if values.ndim != 2 or values.shape[1] != 3:
                raise ValueError("translation, force, and torque must have shape [steps,3]")
            values = np.linalg.norm(values, axis=-1)
        elif vector is None and values.ndim == 2 and values.shape[1] > 0:
            values = np.linalg.norm(values, axis=-1)
        elif values.ndim != 1:
            raise ValueError("rotation must be [steps], QP residual [steps] or [steps,K]")
        if (values[np.isfinite(values)] < 0).any():
            raise ValueError("angles and scalar residual magnitudes must be nonnegative")
        valid = np.isfinite(values)
        if valid.any():
            result[index] = (values[valid].max() if maximum else values[valid].mean()) / divisor
            result[index + 1] = 1.0
    return result


def nearest_sampled_finger_surfaces(points_obj, surface_points_obj, surface_normals_obj, chunk_size=128):
    """Approximate nearest collision surface points using supplied surface samples.

    This is exact only for the provided discrete samples, not the continuous mesh.
    For exact geometry use backend closest-point queries and relative_geometry().
    """
    points = _finite_array(points_obj, name="points_obj")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_obj must have nonempty shape [N,3]")
    if len(surface_points_obj) != 5 or len(surface_normals_obj) != 5:
        raise ValueError("supply current collision surfaces for all five fingers")
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    closest = np.empty((5, len(points), 3), dtype=np.float64)
    normals = np.empty_like(closest)
    for finger, (samples, sample_normals) in enumerate(zip(surface_points_obj, surface_normals_obj)):
        samples = _finite_array(samples, name="surface points")
        if samples.ndim != 2 or samples.shape[1] != 3 or len(samples) == 0:
            raise ValueError("each finger needs nonempty collision surface samples [M,3]")
        sample_normals = _unit_normals(sample_normals, samples.shape, "surface normals")
        for start in range(0, len(points), chunk_size):
            query = points[start:start + chunk_size]
            distance2 = ((query[:, None, :] - samples[None, :, :]) ** 2).sum(axis=-1)
            index = distance2.argmin(axis=-1)
            closest[finger, start:start + len(query)] = samples[index]
            normals[finger, start:start + len(query)] = sample_normals[index]
    return closest, normals


def relative_geometry(points_obj, object_normals_obj, nearest_points_obj, nearest_normals_obj, object_scale):
    """rho=[object_point-nearest_finger_point, separation, normal_dot], [5,N,5]."""
    scale = _scale(object_scale)
    points = _finite_array(points_obj, name="points_obj")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("points_obj must have nonempty shape [N,3]")
    object_normals = _unit_normals(object_normals_obj, points.shape, "object_normals_obj")
    shape = (5, len(points), 3)
    nearest = _finite_array(nearest_points_obj, shape, "nearest_points_obj")
    normals = _unit_normals(nearest_normals_obj, shape, "nearest_normals_obj")
    displacement = (points[None] - nearest) / scale
    separation = np.linalg.norm(displacement, axis=-1, keepdims=True)
    alignment = np.clip((normals * object_normals[None]).sum(axis=-1, keepdims=True), -1, 1)
    return np.concatenate([displacement, separation, alignment], axis=-1).astype(np.float32)


def build_features(
    *, points_obj, object_normals_obj, object_scale, q, q_limits, q_groups,
    actuator_targets, actuator_limits, actuator_groups, finger_reference_obj,
    palm_to_object, diagnostics, global_diagnostics, surface_points_obj=None,
    surface_normals_obj=None, nearest_points_obj=None, nearest_normals_obj=None,
    n_points=1024, seed=0,
):
    """Assemble finite arrays consumed by the predictor.

    Exact nearest arrays, when supplied, correspond to the FULL unsampled object
    point cloud and are indexed with the same resampling indices. Otherwise the
    sampled collision-surface approximation is used. All positions are in metres
    in the object frame, and the scale is retained as the final context channel.
    """
    scale = _scale(object_scale)
    full_points = _finite_array(points_obj, name="points_obj")
    points, normals, indices = resample_point_cloud(full_points, object_normals_obj, n_points, seed)
    q_features, q_valid = normalize_and_pad(q, q_limits, q_groups)
    u_features, u_valid = normalize_and_pad(actuator_targets, actuator_limits, actuator_groups)
    reference = _finite_array(finger_reference_obj, (5, 3), "finger_reference_obj") / scale
    diagnostics = _finite_array(diagnostics, (5, 12), "diagnostics")
    global_diagnostics = _finite_array(global_diagnostics, (10,), "global_diagnostics")
    pose = validate_rigid_transform(palm_to_object, "palm_to_object")
    if (nearest_points_obj is None) != (nearest_normals_obj is None):
        raise ValueError("exact nearest points and normals must be provided together")
    if nearest_points_obj is not None:
        full_shape = (5, len(full_points), 3)
        nearest = _finite_array(nearest_points_obj, full_shape, "nearest_points_obj")[:, indices]
        nearest_normals = _unit_normals(nearest_normals_obj, full_shape, "nearest_normals_obj")[:, indices]
    else:
        if surface_points_obj is None or surface_normals_obj is None:
            raise ValueError("provide exact nearest queries or current finger collision surface samples")
        nearest, nearest_normals = nearest_sampled_finger_surfaces(points, surface_points_obj, surface_normals_obj)
    result = {
        "points": (points / scale).astype(np.float32),
        "finger_features": np.concatenate([q_features, q_valid, u_features, u_valid, reference, diagnostics], axis=1).astype(np.float32),
        "diagnostics": diagnostics.astype(np.float32),
        "geometry": relative_geometry(points, normals, nearest, nearest_normals, scale),
        "context": np.concatenate([pose[:3, :3].reshape(-1), pose[:3, 3] / scale, global_diagnostics, [scale]]).astype(np.float32),
    }
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError("feature assembly produced nonfinite values")
    return result
