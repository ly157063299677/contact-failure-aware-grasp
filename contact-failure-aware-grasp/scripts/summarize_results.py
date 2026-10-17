import argparse
import csv
import json
from pathlib import Path
from cfgrasp.data import read_manifest
from cfgrasp.metrics import aggregate_runs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate measured evaluation logs, never manuscript constants")
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data", required=True, help="Check every run against the complete initially-failed test set")
    args = parser.parse_args()
    rows = [json.loads(line) for path in args.logs for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = {r["state_id"] for r in read_manifest(args.data) if r["split"] == "test" and r["initially_failed"]}
    result = aggregate_runs(rows, expected_ids=ids)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["seed_averages"][0]))
        writer.writeheader()
        writer.writerows(result["seed_averages"])
