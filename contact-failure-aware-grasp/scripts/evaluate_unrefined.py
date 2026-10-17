"""Separate no-refinement baseline over the COMPLETE test partition (not RSR)."""
import argparse
import json
from pathlib import Path
from cfgrasp.config import load_config
from cfgrasp.data import read_manifest
from cfgrasp.labeling import state_rng
from cfgrasp.physics import Deadline, SimulationConfig, perturbation_bank, validation_suite
from cfgrasp.plugins import load_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper.toml")
    parser.add_argument("--data", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    simulation = SimulationConfig(**config["simulation"])
    factory = load_factory(args.backend)
    rows = [r for r in read_manifest(args.data) if r["split"] == "test"]
    if not rows or any(r.get("synthetic", False) for r in rows):
        parser.error("Requires actual complete test data")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    scores = []
    with output.open("x", encoding="utf-8") as stream:
        for row in rows:
            backend = factory(config, row)
            env = backend.validation_copy(backend.snapshot(), Deadline())
            saved = env.snapshot()
            bank = perturbation_bank(state_rng(args.seed, row["state_id"], "unrefined_final"), 5, simulation)
            suites = [validation_suite(env, saved, perturbation, simulation) for perturbation in bank]
            score = sum(s.passed for s in suites) / 5
            scores.append(score)
            stream.write(json.dumps({"state_id": row["state_id"], "scope": "complete_test_unrefined",
                                     "success_frequency": score, "directional": [s.directional.tolist() for s in suites]}) + "\n")
    print(json.dumps({"complete_test_states": len(rows), "unrefined_success_percent": 100 * sum(scores) / len(scores)}))
