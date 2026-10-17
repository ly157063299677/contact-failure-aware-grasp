"""Object-disjoint, state-grouped NPZ dataset. Identifiers never enter the model."""
import json
from pathlib import Path
import numpy as np
from .subsets import MASKS

FEATURE_KEYS = ("points", "finger_features", "diagnostics", "geometry", "context")


def read_manifest(root):
    root = Path(root)
    records = [json.loads(line) for line in (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, object_splits = set(), {}
    for record in records:
        for key in ("state_id", "object_id", "split", "features", "labels", "initially_failed"):
            if key not in record:
                raise ValueError(f"Missing manifest field {key}")
        sid, oid, split = record["state_id"], record["object_id"], record["split"]
        if sid in seen:
            raise ValueError(f"Duplicate state identifier: {sid}")
        seen.add(sid)
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split}")
        if oid in object_splits and object_splits[oid] != split:
            raise ValueError(f"Object leakage across partitions: {oid}")
        object_splits[oid] = split
        if not isinstance(record["initially_failed"], bool):
            raise ValueError("initially_failed must be an actual JSON boolean from physical screening")
    if not records:
        raise ValueError("Empty manifest")
    return records


def resolve_data_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Dataset paths must stay inside the dataset directory")
    return path


def load_features(path, network=None):
    network = network or {}
    with np.load(path, allow_pickle=False) as stored:
        features = {key: np.asarray(stored[key], dtype=np.float32) for key in FEATURE_KEYS}
    validate_features(features, network)
    return features


def validate_features(features, network=None):
    network = network or {}
    points = features["points"]
    expected = {"points": (len(points), network.get("point_dim", 3)),
                "finger_features": (5, network.get("finger_dim", 35)),
                "diagnostics": (5, network.get("diagnostic_dim", 12)),
                "geometry": (5, len(points), 5), "context": (network.get("context_dim", 23),)}
    for key, shape in expected.items():
        if features[key].shape != shape or not np.isfinite(features[key]).all():
            raise ValueError(f"Invalid {key}: expected finite values with shape {shape}")
    if len(points) == 0:
        raise ValueError("Object point cloud is empty")


def load_labels(path):
    with np.load(path, allow_pickle=False) as stored:
        labels = {key: stored[key].copy() for key in stored.files}
    if "masks" not in labels or not np.array_equal(labels["masks"], MASKS):
        raise ValueError("Labels must use integer-mask order 1..31, thumb as bit zero")
    success, deviation = labels["success"], labels["deviation"]
    if success.shape != (31,) or deviation.shape != (31,):
        raise ValueError("Each training state must contain all 31 subset records")
    if not np.isfinite(success).all() or np.any((success < 0) | (success > 1)):
        raise ValueError("Success targets must be complete finite frequencies in [0,1]")
    if np.isinf(deviation).any() or np.any(deviation[np.isfinite(deviation)] < 0):
        raise ValueError("Unavailable deviation must be NaN; finite deviation must be nonnegative")
    if "directional" in labels:
        directional = labels["directional"]
        if directional.shape != (31, 5, 6) or not np.isin(directional, [-1, 0, 1]).all():
            raise ValueError("Directional outcomes must be [31,5,6], with -1 for unexecuted")
        complete = (directional >= 0).all(axis=(1, 2))
        observed = (directional == 1).all(axis=-1).mean(axis=-1)
        if not np.allclose(success[complete], observed[complete]):
            raise ValueError("Empirical success conflicts with directional outcomes")
    if "repetition" in labels:
        repetitions = labels["repetition"]
        if repetitions.shape != (31, 5) or not np.isin(repetitions, [-1, 0, 1]).all():
            raise ValueError("Repetition outcomes must be [31,5] using -1,0,1")
        completed = (repetitions >= 0).all(-1)
        if not np.allclose(success[completed], repetitions[completed].mean(-1)):
            raise ValueError("Success frequency conflicts with repetition outcomes")
        if "directional" in labels:
            complete_repeats = (labels["directional"] >= 0).all(-1)
            observed_repeats = (labels["directional"] == 1).all(-1).astype(np.int8)
            if not np.array_equal(repetitions[complete_repeats], observed_repeats[complete_repeats]):
                raise ValueError("Repetition conflicts with six-direction validation")
        incomplete = ~completed
        if incomplete.any():
            if "solver_status" not in labels or labels["solver_status"].shape != (31,):
                raise ValueError("Unexecuted repetitions require explicit terminal solver/settling status")
            if np.any(labels["solver_status"][incomplete] == "ok") or np.any(success[incomplete] != 0):
                raise ValueError("Incomplete repetition labels are only allowed for terminal failed attempts")
    return labels


class StateDataset:
    """PyTorch-compatible duck-typed Dataset; torch is imported only on retrieval."""

    def __init__(self, root, split, network=None):
        self.root = Path(root)
        self.records = [record for record in read_manifest(root) if record["split"] == split]
        self.network = network
        if not self.records:
            raise ValueError(f"No states in {split} partition")

    def __len__(self):
        return len(self.records)

    def load_numpy(self, index):
        record = self.records[index]
        data = load_features(resolve_data_path(self.root, record["features"]), self.network)
        labels = load_labels(resolve_data_path(self.root, record["labels"]))
        data.update({key: np.asarray(labels[key], dtype=np.float32) for key in ("success", "deviation")})
        return data

    def __getitem__(self, index):
        import torch
        data = {key: torch.from_numpy(value) for key, value in self.load_numpy(index).items()}
        data.update({key: self.records[index][key] for key in ("state_id", "object_id")})
        return data


def audit_dataset(root, expected_paper_counts=False):
    records = read_manifest(root)
    report = {}
    for split in ("train", "val", "test"):
        rows = [record for record in records if record["split"] == split]
        report[split] = {"states": len(rows), "objects": len({row["object_id"] for row in rows}),
                         "initially_failed": sum(row["initially_failed"] for row in rows), "finite_deviations": 0}
        for row in rows:
            load_features(resolve_data_path(root, row["features"]))
            labels = load_labels(resolve_data_path(root, row["labels"]))
            report[split]["finite_deviations"] += int(np.isfinite(labels["deviation"]).sum())
    report["synthetic"] = any(row.get("synthetic", False) for row in records)
    if expected_paper_counts:
        if report["synthetic"]:
            raise ValueError("Synthetic fixtures cannot pass the paper dataset audit")
        for split, objects, states in (("train", 280, 5600), ("val", 40, 800), ("test", 80, 1600)):
            if report[split]["objects"] != objects or report[split]["states"] != states:
                raise ValueError(f"{split} counts do not match Table S1")
        if report["test"]["initially_failed"] != 480:
            raise ValueError("Test evaluation set must contain the reported 480 initially failed states")
        if sum(report[s]["initially_failed"] for s in ("train", "val", "test")) != 2400:
            raise ValueError("Initially failed total does not match Table S1")
        if sum(report[s]["finite_deviations"] for s in ("train", "val", "test")) != 238080:
            raise ValueError("Finite deviation count does not match Table S1")
    return report
