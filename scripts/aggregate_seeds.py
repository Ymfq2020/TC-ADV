"""Aggregate seed-sweep outputs across two runs (e.g. baseline vs +TC-ADV).

Reads the `seed_summary.json` files produced by `tc-adv seed-sweep` for two
runs and reports:
  - mean / std for each metric
  - paired-sample t test on the matched seeds (Chapter 4.3.2 reports
    df=4, two-sided t-tests on MRR / TVR)

Example::
    python scripts/aggregate_seeds.py \
        --baseline outputs/seed_sweep_lmca_baseline/seed_summary.json \
        --treated  outputs/seed_sweep_lmca_tcadv/seed_summary.json \
        --metrics MRR Hits@10 TVR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_adv.experiments.sweeps import aggregate_rows, paired_t_test
from tc_adv.utils.io import read_json, write_json


def _flatten(rows: list[dict], metric: str) -> list[float]:
    out = []
    for row in rows:
        value = row.get(metric)
        if isinstance(value, dict):
            value = value.get("mean")
        if value is None:
            continue
        out.append(float(value))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired-seed comparison")
    parser.add_argument("--baseline", required=True, help="seed_summary.json from baseline run")
    parser.add_argument("--treated", required=True, help="seed_summary.json from +TC-ADV run")
    parser.add_argument("--metrics", nargs="+", default=["MRR", "Hits@10", "TVR"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = read_json(args.baseline)
    treated = read_json(args.treated)
    base_rows = baseline["rows"]
    treat_rows = treated["rows"]
    if len(base_rows) != len(treat_rows):
        raise SystemExit(f"seed counts differ: {len(base_rows)} vs {len(treat_rows)}")

    base_summary = aggregate_rows(base_rows, group_by=None)["all"]
    treat_summary = aggregate_rows(treat_rows, group_by=None)["all"]

    results = {}
    for metric in args.metrics:
        base_values = _flatten(base_rows, metric)
        treat_values = _flatten(treat_rows, metric)
        if not base_values or not treat_values:
            continue
        # Multiply by 100 for percentage-point reporting (matches Chapter 4 tables)
        base_pct = [v * 100.0 for v in base_values]
        treat_pct = [v * 100.0 for v in treat_values]
        ttest = paired_t_test(treat_pct, base_pct)
        results[metric] = {
            "baseline_mean_pct": sum(base_pct) / len(base_pct),
            "treated_mean_pct": sum(treat_pct) / len(treat_pct),
            "delta_mean_pct": ttest["mean_diff"],
            "t": ttest["t"],
            "df": ttest["df"],
            "p_two_sided": ttest["p_two_sided"],
            "ci95_low_pct": ttest["ci95_low"],
            "ci95_high_pct": ttest["ci95_high"],
        }
        print(f"\n[{metric}]")
        print(f"  baseline = {results[metric]['baseline_mean_pct']:.2f}%")
        print(f"  +TC-ADV  = {results[metric]['treated_mean_pct']:.2f}%")
        print(f"  delta    = {results[metric]['delta_mean_pct']:+.2f}pct (95% CI [{results[metric]['ci95_low_pct']:+.2f}, {results[metric]['ci95_high_pct']:+.2f}])")
        print(f"  paired t = {results[metric]['t']:.2f}, df={results[metric]['df']}, p={results[metric]['p_two_sided']:.4f}")

    payload = {
        "baseline_path": args.baseline,
        "treated_path": args.treated,
        "baseline_summary": base_summary,
        "treated_summary": treat_summary,
        "paired_tests": results,
    }
    if args.output:
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
