"""Long-tail entity analysis (Figure 4-9).

Splits test entities into low/mid/high frequency bins based on their
training-set interaction count, then reports MRR per bin for a base-vs-TC-ADV
comparison.

Default bin boundaries follow Chapter 4.3.7:
  - low frequency:   training count < 10   (~42% of ICEWS18 entities)
  - mid frequency:   10 <= count <= 50     (~35%)
  - high frequency:  count > 50            (~23%)

Inputs:
  --train-quadruples : path to processed train.tsv / train.json with
                       {subject, object} fields
  --predictions      : path to {test|valid}_predictions.jsonl produced by the
                       trainer (one record per query)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from tc_adv.utils.io import read_jsonl, write_json


def _load_train_counts(train_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    suffix = train_path.suffix.lower()
    if suffix == ".jsonl":
        for row in read_jsonl(train_path):
            for key in ("subject", "object", "head", "tail"):
                value = row.get(key)
                if value is not None:
                    counts[str(value)] += 1
    elif suffix == ".json":
        payload = json.loads(train_path.read_text(encoding="utf-8"))
        for row in payload:
            for key in ("subject", "object", "head", "tail"):
                value = row.get(key)
                if value is not None:
                    counts[str(value)] += 1
    else:
        with train_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    counts[parts[0]] += 1
                    counts[parts[2]] += 1
    return counts


def _bin_for(count: int, low_max: int, mid_max: int) -> str:
    if count < low_max:
        return "low"
    if count <= mid_max:
        return "mid"
    return "high"


def _reciprocal_rank(scores: dict[str, float], gold: str) -> float:
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    for rank, (entity_id, _) in enumerate(ordered, start=1):
        if entity_id == gold:
            return 1.0 / rank
    return 0.0


def analyze_predictions(
    predictions_path: str | Path,
    train_counts: Counter[str],
    low_max: int = 10,
    mid_max: int = 50,
) -> dict[str, dict[str, float]]:
    rows = read_jsonl(predictions_path)
    bin_metrics: dict[str, dict[str, list[float]]] = {
        "low": {"mrr": [], "hits10": []},
        "mid": {"mrr": [], "hits10": []},
        "high": {"mrr": [], "hits10": []},
    }
    for row in rows:
        gold = row.get("gold")
        scores = row.get("scores", {})
        if not gold or not scores:
            continue
        rr = _reciprocal_rank(scores, gold)
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits10 = 1.0 if any(entity_id == gold for entity_id, _ in ordered[:10]) else 0.0
        # Bin by gold entity's training-set frequency (corresponds to the
        # entity that the model is asked to predict).
        count = int(train_counts.get(str(gold), 0))
        bucket = _bin_for(count, low_max, mid_max)
        bin_metrics[bucket]["mrr"].append(rr)
        bin_metrics[bucket]["hits10"].append(hits10)

    summary = {}
    for bucket, data in bin_metrics.items():
        summary[bucket] = {
            "n": len(data["mrr"]),
            "MRR": fmean(data["mrr"]) if data["mrr"] else 0.0,
            "Hits@10": fmean(data["hits10"]) if data["hits10"] else 0.0,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Long-tail entity MRR analysis (Figure 4-9)")
    parser.add_argument("--train-quadruples", required=True,
                        help="Path to a TSV/JSON/JSONL of training quadruples (subject, relation, object, timestamp)")
    parser.add_argument("--predictions", required=True, action="append",
                        help="Path to test_predictions.jsonl. Repeat for multiple runs (e.g. base vs +TC-ADV).")
    parser.add_argument("--label", action="append", default=None,
                        help="Display label aligned with --predictions (defaults to file stem)")
    parser.add_argument("--low-max", type=int, default=10)
    parser.add_argument("--mid-max", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    train_counts = _load_train_counts(Path(args.train_quadruples))
    payload: dict[str, Any] = {
        "train_quadruples": str(args.train_quadruples),
        "low_max": args.low_max,
        "mid_max": args.mid_max,
        "runs": [],
    }
    labels = args.label or [Path(p).stem for p in args.predictions]
    for predictions, label in zip(args.predictions, labels):
        summary = analyze_predictions(predictions, train_counts, args.low_max, args.mid_max)
        payload["runs"].append({"label": label, "predictions": predictions, "summary": summary})
        print(f"\n[{label}]")
        for bucket in ("low", "mid", "high"):
            row = summary[bucket]
            print(f"  {bucket:>4}: n={row['n']:>6}  MRR={row['MRR']:.4f}  Hits@10={row['Hits@10']:.4f}")

    if args.output:
        write_json(args.output, payload)


if __name__ == "__main__":
    main()
