"""Measure online repair and five independent final suites using a native backend."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from cfgrasp.config import load_config
from cfgrasp.data import read_manifest
from cfgrasp.labeling import state_rng
from cfgrasp.physics import OnlineConfig, SimulationConfig
from cfgrasp.plugins import load_factory
from cfgrasp.policies import LearnedPolicy, RandomPolicy, contact_policy, singleton_policy
from cfgrasp.runtime import run_subset_search, run_full_hand, independent_final_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.toml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--backend", required=True, help="module:factory(config,state_record)")
    parser.add_argument("--method", choices=["proposed", "mlp", "topk", "random", "contact", "full_hand"], required=True)
    parser.add_argument("--run-name", help="Unique result-group label, e.g. interaction_outcome for an ablation")
    parser.add_argument("--checkpoint", help="Required for proposed, mlp and topk")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--output", required=True, help="New JSONL log path")
    args = parser.parse_args()
    config = load_config(args.config)
    simulation = SimulationConfig(**config["simulation"])
    if simulation.repeats != 5:
        parser.error("The published evaluation protocol requires five final repetitions")
    online = OnlineConfig(budget_seconds=args.budget)
    factory = load_factory(args.backend)
    predictor = None
    if args.method in ("proposed", "mlp", "topk"):
        if not args.checkpoint:
            parser.error("A trained checkpoint is required for a learned method")
        from cfgrasp.inference import Predictor
        predictor = Predictor(args.checkpoint, args.device)
        if predictor.seed != args.seed:
            parser.error("--seed must match the trained checkpoint seed")
        if predictor.config.get("utility", {}) != config.get("utility", {}):
            parser.error("Training and online utility weights differ")
        interaction = predictor.config.get("network", {}).get("interaction", True)
        if (args.method == "proposed" and not interaction) or (args.method == "mlp" and interaction):
            parser.error("Checkpoint architecture does not match --method")
        if args.method == "topk" and (interaction or predictor.config.get("training", {}).get("lambda_pair", 0.5) != 0):
            parser.error("Table S3 requires Top-k to use the outcome-only PointNet-Finger MLP checkpoint")
        predictor.synchronize()
    rows = [r for r in read_manifest(args.data) if r["split"] == "test" and r["initially_failed"]]
    if not rows:
        parser.error("No initially failed held-out test states")
    if any(row.get("synthetic", False) for row in rows):
        parser.error("Synthetic fixtures cannot be used for physical paper evaluation")
    output = Path(args.output)
    if output.exists():
        parser.error("Choose a new output path; previous experiment logs are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
                "config": config, "python": platform.python_version(), "platform": platform.platform(),
                "evaluation_states": len(rows), "status": "started"}
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        with output.open("x", encoding="utf-8") as stream:
            for index, row in enumerate(rows):
                # Explicitly excluded from timing: loading assets and the original snapshot.
                backend = factory(config, row)
                online_rng = state_rng(args.seed, row["state_id"], "online_validation")
                policy_rng = state_rng(args.seed, row["state_id"], "random_subset_policy")

                def policy_factory(features, diagnostics, deadline):
                    del diagnostics  # Features contain the standardized diagnostic matrix.
                    if args.method == "random":
                        return RandomPolicy(policy_rng)
                    if args.method == "contact":
                        return contact_policy(features["diagnostics"])
                    singleton_indices = [0, 1, 3, 7, 15] if args.method == "topk" else None
                    probability, deviation = predictor(features, deadline, subset_indices=singleton_indices)
                    if args.method == "topk":
                        return singleton_policy(probability, deviation)
                    return LearnedPolicy(probability, deviation, **config["utility"])

                if args.method == "full_hand":
                    result = run_full_hand(backend, simulation, online, rng=online_rng)
                else:
                    result = run_subset_search(backend, policy_factory, simulation, online,
                                               rng=online_rng,
                                               synchronize=None if predictor is None else predictor.synchronize)
                # Final evaluation is independent and happens AFTER the frozen online result.
                final = independent_final_validation(result, simulation,
                                                      state_rng(args.seed, row["state_id"], "independent_final"))
                record = {"state_id": row["state_id"], "object_id": row["object_id"], "method": args.run_name or args.method,
                          "strategy": args.method,
                          "seed": args.seed, "budget_seconds": args.budget, **result.to_record(),
                          "final_outcomes": [int(suite.passed) for suite in final],
                          "final_directional": [suite.directional.tolist() for suite in final],
                          "final_status": [suite.status for suite in final]}
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print(f"{index + 1}/{len(rows)} {row['state_id']}: {record['status']}")
        metadata["status"] = "completed"
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
