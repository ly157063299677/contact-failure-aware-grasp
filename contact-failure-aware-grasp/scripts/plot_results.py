"""Plot measured budget sensitivity; no paper results are hard-coded."""
import argparse
import json
from pathlib import Path
from collections import defaultdict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for row in json.loads(Path(args.summary).read_text(encoding="utf-8"))["seed_averages"]:
        groups[row["method"]].append(row)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), constrained_layout=True)
    for method, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["budget_seconds"])
        ax.plot([r["budget_seconds"] for r in rows], [r["repair_success_rate_percent"] for r in rows], marker="o", label=method)
    ax.set(xlabel="Online wall-clock budget (s)", ylabel="Repair success rate (%)", ylim=(0, 100))
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
