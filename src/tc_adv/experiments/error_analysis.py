"""Validation-set error type classification (Figure 4-2).

Reads `test_diagnostics.json` (or `valid_diagnostics.json`) produced by the
trainer and classifies high-confidence violations into three categories
matching Chapter 4.3.4:

  - lifecycle_out_of_bounds   生命周期越界    (52% of high-confidence
                                              errors per Figure 4-2)
  - evolution_mutation        演化轨迹反常突变 (34%)
  - static_semantic_conflict  静态语义冲突    (14%)

Mapping rule (Chapter 4.3.4 stipulates the discriminator's TRM/ECM channels
align with these error types):

    if p_trm > p_ecm and p_trm >= violation_threshold and prediction != gold:
        lifecycle_out_of_bounds
    elif p_ecm > p_trm and p_ecm >= violation_threshold and prediction != gold:
        evolution_mutation
    elif p_fake >= violation_threshold and prediction != gold:
        static_semantic_conflict
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from tc_adv.utils.io import read_json, write_json


CATEGORIES = ("lifecycle_out_of_bounds", "evolution_mutation", "static_semantic_conflict")


def classify_error_types(
    diagnostics_path: str | Path,
    violation_threshold: float = 0.7,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = read_json(Path(diagnostics_path))
    counts: Counter[str] = Counter()
    samples: dict[str, list[dict[str, Any]]] = {key: [] for key in CATEGORIES}

    for row in rows:
        gold = row.get("gold")
        prediction = row.get("top1_prediction")
        p_trm = float(row.get("p_trm", 0.0))
        p_ecm = float(row.get("p_ecm", 0.0))
        p_fake = float(row.get("p_fake", 0.0))
        if prediction is None or prediction == gold:
            continue
        if p_fake < violation_threshold and max(p_trm, p_ecm) < violation_threshold:
            continue
        if p_trm >= violation_threshold and p_trm >= p_ecm:
            category = "lifecycle_out_of_bounds"
        elif p_ecm >= violation_threshold and p_ecm > p_trm:
            category = "evolution_mutation"
        else:
            category = "static_semantic_conflict"
        counts[category] += 1
        if len(samples[category]) < 5:
            samples[category].append(row)

    total = sum(counts.values())
    distribution = {key: counts.get(key, 0) / max(total, 1) for key in CATEGORIES}
    payload = {
        "diagnostics_path": str(diagnostics_path),
        "violation_threshold": violation_threshold,
        "total_high_confidence_errors": total,
        "counts": {key: counts.get(key, 0) for key in CATEGORIES},
        "distribution": distribution,
        "examples": samples,
    }
    if output_path:
        write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify high-confidence errors into Figure 4-2 categories")
    parser.add_argument("--diagnostics", required=True, help="Path to test_diagnostics.json or valid_diagnostics.json")
    parser.add_argument("--threshold", type=float, default=0.7, help="violation probability threshold (default 0.7)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = classify_error_types(args.diagnostics, args.threshold, args.output)
    distribution = payload["distribution"]
    print("Error type distribution (n=%d, threshold=%.2f):" % (
        payload["total_high_confidence_errors"], args.threshold))
    for category in CATEGORIES:
        print(f"  {category}: {distribution[category]:.1%} ({payload['counts'][category]} samples)")


if __name__ == "__main__":
    main()
