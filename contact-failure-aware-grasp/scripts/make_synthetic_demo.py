"""Create small, explicitly synthetic schema fixtures. NOT physics or paper data."""
import argparse
import json
from pathlib import Path
import numpy as np
from cfgrasp.subsets import MASKS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    root = Path(args.output)
    if root.exists() and any(root.iterdir()):
        raise SystemExit("Choose an empty output directory")
    (root / "features").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    rng, rows = np.random.default_rng(args.seed), []
    for split in ("train", "val", "test"):
        for index in range(4):
            sid = f"SYNTHETIC_{split}_{index}"
            values = {"points": rng.normal(size=(1024, 3)), "finger_features": rng.normal(size=(5, 35)),
                      "diagnostics": rng.normal(size=(5, 12)), "geometry": rng.normal(size=(5, 1024, 5)),
                      "context": rng.normal(size=(23,))}
            values["finger_features"][:, 23:35] = values["diagnostics"]
            np.savez_compressed(root / "features" / f"{sid}.npz", **{k: v.astype(np.float32) for k, v in values.items()})
            directional = rng.integers(0, 2, size=(31, 5, 6), dtype=np.int8)
            deviation = rng.uniform(0, 0.2, size=31).astype(np.float32)
            deviation[0] = np.nan
            np.savez_compressed(root / "labels" / f"{sid}.npz", masks=MASKS,
                                success=(directional == 1).all(-1).mean(-1).astype(np.float32), deviation=deviation,
                                directional=directional, solver_status=np.full(31, "synthetic"))
            rows.append({"state_id": sid, "object_id": f"SYNTHETIC_{split}", "split": split,
                         "features": f"features/{sid}.npz", "labels": f"labels/{sid}.npz", "initially_failed": True, "synthetic": True})
    (root / "manifest.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    (root / "SYNTHETIC_ONLY.txt").write_text("Artificial tensors for interface checks. No physical interpretation or paper reproduction.\n", encoding="utf-8")
    print(f"Wrote {len(rows)} explicitly synthetic states to {root}")
