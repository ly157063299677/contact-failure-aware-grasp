"""Generate physical labels through an explicitly supplied native backend adapter."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfgrasp.config import load_config
from cfgrasp.labeling import generate_state_labels, state_rng
from cfgrasp.physics import SimulationConfig
from cfgrasp.plugins import load_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", required=True, help="JSONL asset/snapshot records with state_id,object_id,split")
    parser.add_argument("--backend", required=True, help="module:factory taking config and state record")
    parser.add_argument("--config", default="configs/paper.toml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(args.config)
    simulation = SimulationConfig(**config["simulation"])
    factory = load_factory(args.backend)
    rows = [json.loads(line) for line in Path(args.states).read_text(encoding="utf-8").splitlines() if line.strip()]
    seen, object_splits = set(), {}
    for row in rows:
        if row["state_id"] in seen:
            raise ValueError("Duplicate state identifier")
        seen.add(row["state_id"])
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Invalid partition")
        previous = object_splits.setdefault(row["object_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("Object leakage; split objects BEFORE constructing variants")
        if row.get("synthetic", False):
            raise ValueError("This command requires real physical source states")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Choose an empty destination to avoid mixing label-generation runs")
    for folder in ("features", "labels"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    (output / "generation_config.json").write_text(json.dumps({"config": config, "backend": args.backend, "seed": args.seed}, indent=2), encoding="utf-8")
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, row in enumerate(rows):
            backend = factory(config, row)
            features, labels, initially_failed = generate_state_labels(backend, simulation, state_rng(args.seed, row["state_id"], "offline_labels"))
            stem = f"state_{index:06d}"
            np.savez_compressed(output / "features" / f"{stem}.npz", **features)
            np.savez_compressed(output / "labels" / f"{stem}.npz", **labels)
            record = dict(row, features=f"features/{stem}.npz", labels=f"labels/{stem}.npz", initially_failed=initially_failed)
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"{index + 1}/{len(rows)}: {row['state_id']}")
