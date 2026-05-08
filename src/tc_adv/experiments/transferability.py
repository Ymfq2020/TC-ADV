"""Transferability scatter analysis (Figure 4-5).

Reads multiple per-generator outcomes (each with a base TVR and a TC-ADV TVR)
and produces the scatter coordinates (base_tvr, delta_tvr) plus a least-squares
linear fit slope. Chapter 4.3.6 reports the empirical slope ≈ -0.64.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tc_adv.utils.io import read_json, write_json


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = num / den if den else 0.0
    intercept = mean_y - slope * mean_x
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


def collect_pairs(pairs_jsonl: str | Path) -> list[dict[str, float]]:
    """Each line of `pairs_jsonl` should specify:
        {"generator": "RE-GCN", "dataset": "GDELT",
         "base_tvr": 0.138, "tcadv_tvr": 0.049}
    base_tvr/tcadv_tvr are in [0,1] (raw rate).
    """
    rows: list[dict[str, float]] = []
    path = Path(pairs_jsonl)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            base_tvr = float(row["base_tvr"])
            tcadv_tvr = float(row["tcadv_tvr"])
            rows.append({
                "generator": row.get("generator", "?"),
                "dataset": row.get("dataset", "?"),
                "base_tvr": base_tvr,
                "tcadv_tvr": tcadv_tvr,
                "delta_tvr": tcadv_tvr - base_tvr,
            })
    return rows


def analyze(rows: list[dict[str, float]]) -> dict[str, Any]:
    xs = [row["base_tvr"] * 100.0 for row in rows]
    ys = [row["delta_tvr"] * 100.0 for row in rows]
    slope, intercept, r2 = _least_squares(xs, ys)
    return {
        "rows": rows,
        "fit": {
            "slope_per_pct": slope,
            "intercept_pct": intercept,
            "r_squared": r2,
            "note": "x = base TVR (%), y = delta TVR (pct points)",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Transferability scatter for Figure 4-5")
    parser.add_argument("--pairs", required=True, help="JSONL with generator/dataset/base_tvr/tcadv_tvr fields")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    rows = collect_pairs(args.pairs)
    payload = analyze(rows)
    print(f"Slope: {payload['fit']['slope_per_pct']:.3f}  (chapter target ≈ -0.64)")
    print(f"R^2:   {payload['fit']['r_squared']:.3f}")
    for row in payload["rows"]:
        print(f"  {row['generator']:<10s} {row['dataset']:<8s} base={row['base_tvr']*100:5.2f}% "
              f"delta={row['delta_tvr']*100:+5.2f}pct")
    if args.output:
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
