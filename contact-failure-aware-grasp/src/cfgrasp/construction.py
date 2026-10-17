"""Object-disjoint partitioning and grasp perturbation proposals.

These functions do not create MuJoCo scenes or claim to reconstruct the authors'
original assets, random seeds, snapshots, or published outcome counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .features import validate_rigid_transform

SPLITS = ("train", "val", "test")
VARIANTS = ("original", "actuator", "palm", "combined")


def partition_objects(object_ids, seed=0, ratios=(0.7, 0.1, 0.2)):
    """Split base object identities BEFORE selecting grasps/creating variants.

    Default ratios give exactly 280/40/80 for 400 objects. For other counts,
    start each split with one object and repeatedly assign the next object to
    the greatest deficit against its target count (ties train, val, test).
    """
    objects = list(object_ids)
    if len(objects) < 3 or any(not isinstance(obj, str) or not obj.strip() for obj in objects):
        raise ValueError("provide at least three nonempty string object identities")
    if len(set(objects)) != len(objects):
        raise ValueError("duplicate identities: collapse all scales to one base object_id first")
    ratios = np.asarray(ratios, dtype=np.float64)
    if ratios.shape != (3,) or not np.isfinite(ratios).all() or (ratios <= 0).any() or not np.isclose(ratios.sum(), 1):
        raise ValueError("three positive split ratios must sum to one")
    counts = np.ones(3, dtype=int)
    target = len(objects) * ratios
    for _ in range(len(objects) - 3):
        counts[np.argmax(target - counts)] += 1
    # Sorting makes assignment independent of filesystem/input enumeration order.
    shuffled = np.asarray(sorted(objects), dtype=object)[np.random.default_rng(seed).permutation(len(objects))]
    result = {}
    offset = 0
    for split, count in zip(SPLITS, counts):
        result[split] = shuffled[offset:offset + count].tolist()
        offset += count
    audit_object_partitions(result)
    return result


def audit_object_partitions(partitions):
    """Raise on empty partitions, duplicates, or cross-partition identity leakage."""
    if set(partitions) != set(SPLITS):
        raise ValueError("partitions must have exactly train, val, and test keys")
    owner = {}
    for split in SPLITS:
        ids = partitions[split]
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError(f"{split} must be a nonempty object-id list")
        for object_id in ids:
            if not isinstance(object_id, str) or not object_id.strip():
                raise ValueError("object identities must be nonempty strings")
            if object_id in owner:
                raise ValueError(f"object identity {object_id!r} repeated in {owner[object_id]} and {split}")
            owner[object_id] = split
    return owner


def _rotation_xyz(angles):
    """Extrinsic object-axis x/y/z rotations, composed Rz @ Ry @ Rx."""
    x, y, z = angles
    cx, sx = np.cos(x), np.sin(x)
    cy, sy = np.cos(y), np.sin(y)
    cz, sz = np.cos(z), np.sin(z)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def propose_grasp_variants(
    actuator_targets, actuator_limits, palm_to_object, *, seed=0,
    actuator_bound=0.05, translation_bound=0.005, rotation_bound_degrees=2.5,
):
    """Return four control/pose proposals; no hidden joint-to-actuator mapping.

    Assumes supplied position-actuator targets have angular units in radians.
    The scene adapter must map proposals into feasible complete saved states,
    retaining velocities, controls, object state, and relevant simulator state.
    """
    targets = np.asarray(actuator_targets, dtype=np.float64)
    limits = np.asarray(actuator_limits, dtype=np.float64)
    pose = validate_rigid_transform(palm_to_object, "palm_to_object")
    if targets.ndim != 1 or targets.size == 0 or not np.isfinite(targets).all():
        raise ValueError("actuator targets must be a finite, nonempty vector")
    if limits.shape != (len(targets), 2) or not np.isfinite(limits).all() or (limits[:, 1] <= limits[:, 0]).any():
        raise ValueError("supply finite positive-width actuator limits [nu,2]")
    if ((targets < limits[:, 0]) | (targets > limits[:, 1])).any():
        raise ValueError("original targets exceed limits; resolve source data before constructing variants")
    bounds = np.asarray([actuator_bound, translation_bound, rotation_bound_degrees], dtype=float)
    if not np.isfinite(bounds).all() or (bounds < 0).any():
        raise ValueError("perturbation bounds must be finite and nonnegative")
    rng = np.random.default_rng(seed)
    result = []
    for variant in VARIANTS:
        proposed_targets = targets.copy()
        proposed_pose = pose.copy()
        delta_u = np.zeros_like(targets)
        delta_t = np.zeros(3)
        delta_rotation = np.zeros(3)
        if variant in ("actuator", "combined"):
            delta_u = rng.uniform(-actuator_bound, actuator_bound, size=len(targets))
            proposed_targets = np.clip(targets + delta_u, limits[:, 0], limits[:, 1])
        if variant in ("palm", "combined"):
            delta_t = rng.uniform(-translation_bound, translation_bound, size=3)
            rotation_bound = np.deg2rad(rotation_bound_degrees)
            delta_rotation = rng.uniform(-rotation_bound, rotation_bound, size=3)
            proposed_pose[:3, :3] = _rotation_xyz(delta_rotation) @ pose[:3, :3]
            proposed_pose[:3, 3] = pose[:3, 3] + delta_t
        result.append({
            "variant": variant,
            "actuator_targets": proposed_targets,
            "palm_to_object": proposed_pose,
            "sampled_actuator_delta": delta_u,
            "applied_actuator_delta": proposed_targets - targets,
            "translation_delta": delta_t,
            "rotation_xyz_radians": delta_rotation,
        })
    return result


def audit_manifest(records, partitions, *, require_paths=False, base_dir=None):
    """Check identities/splits before training; IDs never become model features.

    Canonical row keys: state_id, object_id, split, features, labels,
    initially_failed. Extra provenance keys are encouraged. Paths are relative
    to the JSONL manifest directory unless absolute. This cannot detect that two
    different names secretly denote the same object; canonicalize upstream.
    """
    object_split = audit_object_partitions(partitions)
    seen = set()
    source_owner = {}
    file_owner = {}
    counts = {split: 0 for split in SPLITS}
    root = Path(base_dir or ".")
    for row in records:
        required = {"state_id", "object_id", "split", "features", "labels", "initially_failed"}
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"manifest row requires keys {sorted(required)}")
        state_id = row["state_id"]
        if not isinstance(state_id, str) or not state_id or state_id in seen:
            raise ValueError("state IDs must be nonempty and unique")
        seen.add(state_id)
        if not isinstance(row["object_id"], str) or row["split"] not in SPLITS:
            raise ValueError("object_id must be a string and split must be train, val, or test")
        if object_split.get(row["object_id"]) != row["split"]:
            raise ValueError(f"state {state_id}: unknown object or split leakage")
        if not isinstance(row["initially_failed"], bool):
            raise ValueError("initially_failed must come from screening and be a JSON boolean")
        for field in ("features", "labels"):
            if not isinstance(row[field], str) or not row[field]:
                raise ValueError(f"{field} must be a nonempty file path")
            canonical_path = str((root / row[field]).resolve())
            if canonical_path in file_owner and file_owner[canonical_path] != row["split"]:
                raise ValueError(f"state {state_id}: shared feature/label file crosses partitions")
            file_owner[canonical_path] = row["split"]
            if require_paths and not (root / row[field]).is_file():
                raise FileNotFoundError(root / row[field])
        if "source_grasp_id" in row:
            if not isinstance(row["source_grasp_id"], str) or not row["source_grasp_id"]:
                raise ValueError("source_grasp_id must be a nonempty string when supplied")
            key = (row["object_id"], row["source_grasp_id"])
            if key in source_owner and source_owner[key] != row["split"]:
                raise ValueError("source-grasp variants cross split boundaries")
            source_owner[key] = row["split"]
        counts[row["split"]] += 1
    return counts


def load_manifest(path):
    """Read strict JSONL; no data files are created or inferred."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
