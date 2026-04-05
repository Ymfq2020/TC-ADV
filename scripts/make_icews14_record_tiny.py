"""Build an ultra-light ICEWS14 recording subset for TC-ADV demos.

Default target size:
- train: 80
- valid: 10
- test: 10

The script reads an existing LMCA-TIC local dataset and writes a tiny subset
under the TC-ADV workspace so that recording runs stay fast and reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_split(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def collect_entities(lines: list[str]) -> set[str]:
    entities: set[str] = set()
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 4:
            parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Expected at least 4 columns in line: {line}")
        entities.add(parts[0])
        entities.add(parts[2])
    return entities


def load_bie(source_path: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    if not source_path.exists():
        return rows
    for line in source_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        rows[str(payload["entity_id"])] = payload
    return rows


def build_subset(
    source_root: Path,
    target_root: Path,
    train_size: int,
    valid_size: int,
    test_size: int,
    bie_source: Path | None = None,
) -> dict[str, object]:
    raw_source = source_root / "raw"
    if not raw_source.exists():
        raise FileNotFoundError(f"Missing source raw dir: {raw_source}")

    split_limits = {"train": train_size, "valid": valid_size, "test": test_size}
    selected: dict[str, list[str]] = {}
    entities: set[str] = set()

    for split, limit in split_limits.items():
        rows = read_split(raw_source / f"{split}.txt")
        if len(rows) < limit:
            raise ValueError(f"Source split {split} has only {len(rows)} rows, requested {limit}.")
        subset = rows[:limit]
        selected[split] = subset
        entities.update(collect_entities(subset))

    raw_target = target_root / "raw"
    bie_target = target_root / "bie"
    raw_target.mkdir(parents=True, exist_ok=True)
    bie_target.mkdir(parents=True, exist_ok=True)

    for split, rows in selected.items():
        (raw_target / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    bie_lookup = load_bie(
        bie_source
        if bie_source is not None
        else source_root / "bie" / "entity_metadata.jsonl"
    )
    bie_rows = []
    for entity_id in sorted(entities):
        row = bie_lookup.get(entity_id, {
            "entity_id": entity_id,
            "entity_name": entity_id,
            "entity_type": "UNKNOWN",
        })
        bie_rows.append(json.dumps(row, ensure_ascii=False))
    (bie_target / "entity_metadata.jsonl").write_text("\n".join(bie_rows) + "\n", encoding="utf-8")

    return {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "split_sizes": {split: len(rows) for split, rows in selected.items()},
        "num_entities": len(entities),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build tiny ICEWS14 subset for recording.")
    parser.add_argument("--source-root", default="/mnt/workspace/LMCA-TIC/data/local/icews14_record_small")
    parser.add_argument("--target-root", default="/mnt/workspace/TC-ADV/data/local/icews14_record_tiny")
    parser.add_argument("--train-size", type=int, default=80)
    parser.add_argument("--valid-size", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=10)
    parser.add_argument("--bie-source", default="")
    args = parser.parse_args()

    payload = build_subset(
        source_root=Path(args.source_root),
        target_root=Path(args.target_root),
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        bie_source=Path(args.bie_source) if args.bie_source else None,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
