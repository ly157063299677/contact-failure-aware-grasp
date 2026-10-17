"""Paper Eq. (15): seed-wise metrics over a fixed failed-state evaluation set."""
from collections import defaultdict
import numpy as np


def summarize_run(records):
    records = list(records)
    if not records:
        raise ValueError("Cannot evaluate an empty run")
    ids = [row["state_id"] for row in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate state records within one method/seed/budget run")
    accepted, credits = [], []
    for row in records:
        if row.get("synthetic", False):
            raise ValueError("Synthetic records must not be reported as physical evaluation")
        elapsed = float(row["online_seconds"])
        if elapsed < 0 or not np.isfinite(elapsed):
            raise ValueError("Invalid runtime")
        budget = float(row["budget_seconds"])
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError("Budget must be finite and positive")
        if not isinstance(row["accepted"], bool):
            raise ValueError("accepted must be a JSON boolean")
        if not row["accepted"]:
            credits.append(0.0)
            continue
        outcomes = np.asarray(row["final_outcomes"])
        if outcomes.shape != (5,) or not np.isin(outcomes, [0, 1]).all():
            raise ValueError("Every accepted result requires five complete independent final suites")
        if elapsed >= budget:
            raise ValueError("Output accepted after the online deadline")
        if not np.isfinite(row["deviation"]) or row["deviation"] < 0:
            raise ValueError("Accepted output needs a finite pre-validation deviation")
        groups = row["selected_groups"]
        if isinstance(groups, bool) or not isinstance(groups, (int, np.integer)) or not 1 <= groups <= 5:
            raise ValueError("Invalid selected group count")
        accepted.append(row)
        credits.append(float(outcomes.mean()))
    return {"n_states": len(records), "n_accepted": len(accepted),
            "repair_success_rate_percent": 100.0 * float(np.mean(credits)),
            "configuration_deviation": float(np.mean([r["deviation"] for r in accepted])) if accepted else None,
            "selected_groups": float(np.mean([r["selected_groups"] for r in accepted])) if accepted else None,
            "online_seconds": float(np.mean([r["online_seconds"] for r in records])),
            "optimization_seconds": float(np.mean([r.get("optimization_seconds", 0.0) for r in records])),
            "validation_seconds": float(np.mean([r.get("validation_seconds", 0.0) for r in records])),
            "attempts": float(np.mean([r.get("attempts", 0) for r in records])),
            "budget_exhaustion_fraction": float(np.mean([r.get("budget_exhausted", False) for r in records]))}


def aggregate_runs(records, expected_ids=None):
    groups = defaultdict(list)
    for row in records:
        groups[(row["method"], int(row["seed"]), float(row["budget_seconds"]))].append(row)
    if not groups:
        raise ValueError("No evaluation records")
    if expected_ids is None:
        raise ValueError("Supply the full initially-failed evaluation set; partial logs cannot define the denominator")
    reference = set(expected_ids)
    if not reference:
        raise ValueError("Empty evaluation set")
    run_metrics, by_method = [], defaultdict(list)
    for (method, seed, budget), rows in sorted(groups.items()):
        if {row["state_id"] for row in rows} != reference:
            raise ValueError(f"Evaluation state sets differ for {method}, seed={seed}, budget={budget}")
        result = {"method": method, "seed": seed, "budget_seconds": budget, **summarize_run(rows)}
        run_metrics.append(result)
        by_method[(method, budget)].append(result)
    averages = []
    metric_keys = [key for key in run_metrics[0] if key not in ("method", "seed", "budget_seconds")]
    for (method, budget), values in sorted(by_method.items()):
        average = {"method": method, "budget_seconds": budget, "seed_count": len(values),
                   "seeds": [r["seed"] for r in values]}
        for key in metric_keys:
            # Do not silently omit a seed with no accepted grasps from a conditional mean.
            average[key] = None if any(r[key] is None for r in values) else float(np.mean([r[key] for r in values]))
        averages.append(average)
    return {"seed_wise": run_metrics, "seed_averages": averages,
            "note": "Metrics are computed per seed, then averaged. Null means undefined, not zero."}
