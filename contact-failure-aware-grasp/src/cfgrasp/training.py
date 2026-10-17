"""State-grouped training, validation early stopping and provenance recording.

One dataset item contains an initial state and all 31 subset labels. The held-out
test partition is never read by this module. Early stopping uses the complete
validation objective (Eq. 12), recomputed from family-specific retained counts.
The manuscript specifies patience but not the stopping criterion; that choice
and all otherwise unspecified implementation choices are recorded explicitly.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time
import tomllib
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import StateDataset
from .losses import OutcomeComparisonLoss, estimate_deviation_scale, loss_for_batch
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Warn instead of silently selecting a different experiment for unsupported
    # deterministic CUDA kernels. Exact cross-platform determinism is not claimed.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch environment has no available CUDA device")
    return device


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
            for name, value in batch.items()}


def _run_epoch(model: torch.nn.Module, loader: DataLoader,
               criterion: OutcomeComparisonLoss, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None,
               gradient_clip: float = 0.0) -> dict[str, float | int]:
    training = optimizer is not None
    model.train(training)
    fields = {
        "success_bce": "n_success", "deviation_huber": "n_deviation",
        "equal_pair": "n_equal_pair", "expansion_pair": "n_expansion_pair",
    }
    numerators = {key: 0.0 for key in fields}
    counts = {key: 0 for key in fields}
    states = 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = _move_batch(batch, device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            losses = loss_for_batch(prediction, batch, criterion)
            if not bool(torch.isfinite(losses["loss"])):
                raise FloatingPointError("Nonfinite objective; inspect feature and label validity")
            if optimizer is not None:
                losses["loss"].backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip,
                                                   error_if_nonfinite=True)
                optimizer.step()
            states += int(batch["success"].shape[0])
            for metric, count_key in fields.items():
                count = int(losses[count_key].detach())
                numerators[metric] += float(losses[metric].detach()) * count
                counts[metric] += count
    if states == 0:
        raise ValueError("An empty dataset split cannot be used for training/validation")
    metrics: dict[str, float | int] = {
        name: numerators[name] / counts[name] if counts[name] else 0.0
        for name in fields
    }
    metrics["outcome"] = metrics["success_bce"] + criterion.lambda_d * metrics["deviation_huber"]
    metrics["comparison"] = 0.5 * metrics["equal_pair"] + 0.5 * metrics["expansion_pair"]
    metrics["loss"] = metrics["outcome"] + criterion.lambda_pair * metrics["comparison"]
    metrics["states"] = states
    metrics.update({fields[name]: count for name, count in counts.items()})
    return metrics


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_checkpoint(path: Path, checkpoint: dict[str, Any]) -> None:
    # Replace only our own checkpoint in this run directory after a complete save.
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load the recorded architecture and weights for inference.

    ``weights_only=True`` avoids loading arbitrary serialized Python objects.
    Only load checkpoints from trusted sources even with this reduced loader.
    The returned model is in evaluation mode; predictions remain in original
    normalized-deviation units (``sd`` belongs exclusively to the loss).
    """
    target = resolve_device(device)
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported or missing checkpoint format_version")
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(target).eval(), checkpoint


def train(config: dict[str, Any], data_root: str | Path, output: str | Path,
          seed: int, device_override: str | None = None) -> dict[str, Any]:
    """Train one seed into a new/empty run directory and return its summary."""
    config = deepcopy(config)
    settings = config.get("training", {})
    utility = config.get("utility", {})
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory is not empty: {output}. Choose a new --output root.")
    epochs, patience = int(settings.get("epochs", 100)), int(settings.get("patience", 15))
    batch_size, workers = int(settings.get("batch_size", 4)), int(settings.get("num_workers", 0))
    if epochs < 1 or patience < 1 or batch_size < 1 or workers < 0:
        raise ValueError("epochs, patience and batch_size must be positive; num_workers nonnegative")
    seed_everything(seed)
    device = resolve_device(device_override or settings.get("device", "auto"))
    training_data = StateDataset(data_root, "train", config.get("network", {}))
    validation_data = StateDataset(data_root, "val", config.get("network", {}))
    if len(training_data) == 0 or len(validation_data) == 0:
        raise ValueError("Both train and val partitions must contain state records")
    sd = estimate_deviation_scale(training_data)
    generator = torch.Generator().manual_seed(seed)
    loader_options = dict(batch_size=batch_size, num_workers=workers,
                          pin_memory=device.type == "cuda", worker_init_fn=_seed_worker)
    training_loader = DataLoader(training_data, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_data, shuffle=False, **loader_options)
    model = build_model(config).to(device)
    criterion = OutcomeComparisonLoss(
        sd=sd, lambda_d=float(settings.get("lambda_d", 1.0)),
        lambda_pair=float(settings.get("lambda_pair", 0.5)),
        temperature=float(settings.get("temperature", 0.1)),
        alpha=float(utility.get("alpha", 1.0)), beta=float(utility.get("beta", 0.5)),
        gamma=float(utility.get("gamma", 0.1)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings.get("lr", 3e-4)),
                                  weight_decay=float(settings.get("weight_decay", 1e-4)))
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config)
    source = Path(__file__).parent
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "started", "seed": seed, "data_root": str(data_root),
        "train_states": len(training_data), "val_states": len(validation_data),
        "synthetic_dataset": any(record.get("synthetic", False)
                                 for record in training_data.records + validation_data.records),
        "device": str(device), "python": platform.python_version(),
        "platform": platform.platform(), "torch": str(torch.__version__),
        "numpy": str(np.__version__), "cuda_runtime": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "sd": sd, "sd_estimator": "finite training deviations; population std, ddof=0; floor=1e-3",
        "early_stopping": "minimum complete validation objective; strict improvement; patience in epochs",
        "test_partition_used": False,
        "source_sha256": {path.name: _sha256(path) for path in sorted(source.glob("*.py"))},
        "dataset_metadata_sha256": {
            str(path.relative_to(data_root)): _sha256(path)
            for path in sorted(data_root.glob("*.json*")) if path.is_file()
        },
        "scope": "Reference implementation run on supplied data; no manuscript reproduction claim.",
    }
    _write_json(output / "provenance.json", provenance)
    best_loss, best_epoch, stale = float("inf"), 0, 0
    started = time.perf_counter()
    try:
        for epoch in range(1, epochs + 1):
            train_metrics = _run_epoch(model, training_loader, criterion, device, optimizer,
                                       float(settings.get("gradient_clip", 0.0)))
            val_metrics = _run_epoch(model, validation_loader, criterion, device)
            improved = val_metrics["loss"] < best_loss
            if improved:
                best_loss, best_epoch, stale = float(val_metrics["loss"]), epoch, 0
            else:
                stale += 1
            checkpoint = {
                "format_version": 1, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "config": config,
                "sd": sd, "seed": seed, "epoch": epoch,
                "best_epoch": best_epoch, "best_val_loss": best_loss,
                "validation": val_metrics,
            }
            if improved:
                _save_checkpoint(output / "best.pt", checkpoint)
            _save_checkpoint(output / "last.pt", checkpoint)
            record = {"epoch": epoch, "train": train_metrics, "validation": val_metrics,
                      "best_epoch": best_epoch, "seconds_elapsed": time.perf_counter() - started}
            with (output / "history.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(f"seed={seed} epoch={epoch} train={train_metrics['loss']:.6f} "
                  f"val={val_metrics['loss']:.6f} best_epoch={best_epoch}", flush=True)
            if stale >= patience:
                break
        summary = {
            "seed": seed, "epochs_completed": epoch, "best_epoch": best_epoch,
            "best_validation_loss": best_loss, "sd": sd,
            "early_stopped": stale >= patience,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoint": str(output / "best.pt"),
            "scope": "Training objective only; physical repair metrics require separate evaluation.",
        }
        _write_json(output / "summary.json", summary)
        provenance["status"] = "completed"
        provenance["finished_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(output / "provenance.json", provenance)
        return summary
    except BaseException as error:
        provenance["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        provenance["error"] = f"{type(error).__name__}: {error}"
        _write_json(output / "provenance.json", provenance)
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train state-wise finger-subset outcome prediction")
    parser.add_argument("--config", type=Path, required=True, help="Experiment TOML configuration")
    parser.add_argument("--data", type=Path, required=True, help="Exported state-dataset root")
    parser.add_argument("--output", type=Path, required=True, help="Run root; each seed gets seed_<n>/")
    parser.add_argument("--seed", type=int, help="One initialization seed; default: all configured seeds")
    parser.add_argument("--device", help="auto, cpu, cuda or cuda:<index>")
    parser.add_argument("--no-interaction", action="store_true", help="PointNet--Finger MLP architecture")
    parser.add_argument("--outcome-only", action="store_true", help="Set pairwise-loss weight to zero")
    args = parser.parse_args(argv)
    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    if args.no_interaction:
        config.setdefault("network", {})["interaction"] = False
    if args.outcome_only:
        config.setdefault("training", {})["lambda_pair"] = 0.0
    config["source_config"] = {"path": str(args.config.resolve()), "sha256": _sha256(args.config)}
    seeds = [args.seed] if args.seed is not None else config.get("training", {}).get("seeds", [0, 1, 2])
    if not seeds or len(set(seeds)) != len(seeds):
        parser.error("training.seeds must contain at least one unique seed")
    for seed in seeds:
        train(config, args.data, args.output / f"seed_{int(seed)}", int(seed), args.device)


if __name__ == "__main__":
    main()
