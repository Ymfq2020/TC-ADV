"""Dataset preparation for TC-ADV and LMCA-TIC.

This utility converts raw CSV/TSV/JSONL event files into the canonical local
file-database layout required by the training pipeline:

- `raw/train.txt`
- `raw/valid.txt`
- `raw/test.txt`
- `bie/entity_metadata.jsonl`

The split is chronological on unique time buckets to prevent future leakage
into training, which is the required evaluation protocol for temporal KG
completion and matches the thesis workflow.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from tc_adv.utils.io import ensure_dir, write_json, write_text


@dataclass(frozen=True)
class RawEvent:
    subject: str
    relation: str
    object: str
    bucket_label: str
    bucket_index: int


def prepare_dataset(
    events_path: str | Path,
    output_root: str | Path,
    *,
    entities_path: str | Path | None = None,
    head_col: str = "subject",
    relation_col: str = "relation",
    tail_col: str = "object",
    time_col: str = "timestamp",
    entity_id_col: str = "entity_id",
    entity_name_col: str = "entity_name",
    entity_extra_cols: list[str] | None = None,
    delimiter: str = ",",
    entity_delimiter: str = ",",
    time_granularity: str = "auto",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    default_entity_type: str = "UNKNOWN",
    keep_duplicates: bool = False,
) -> dict[str, Any]:
    _validate_ratios(train_ratio=train_ratio, valid_ratio=valid_ratio)
    event_rows = _load_rows(events_path, delimiter=delimiter)
    if not event_rows:
        raise ValueError(f"No events found in {events_path}")

    canonical_rows = []
    for row in event_rows:
        subject = _require_string(row, head_col)
        relation = _require_string(row, relation_col)
        obj = _require_string(row, tail_col)
        raw_time = _require_string(row, time_col)
        sort_key, bucket_label = normalize_timestamp(raw_time, time_granularity)
        canonical_rows.append(
            {
                "subject": subject,
                "relation": relation,
                "object": obj,
                "raw_time": raw_time,
                "sort_key": sort_key,
                "bucket_label": bucket_label,
            }
        )

    bucket_index_map = build_time_index(canonical_rows)
    events = [
        RawEvent(
            subject=row["subject"],
            relation=row["relation"],
            object=row["object"],
            bucket_label=row["bucket_label"],
            bucket_index=bucket_index_map[row["bucket_label"]],
        )
        for row in canonical_rows
    ]
    if not keep_duplicates:
        events = _deduplicate_events(events)

    split_payload = split_events_by_time(
        events=events,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )

    entities = build_entity_payload(
        events=events,
        entities_path=entities_path,
        entity_id_col=entity_id_col,
        entity_name_col=entity_name_col,
        entity_extra_cols=entity_extra_cols or [],
        delimiter=entity_delimiter,
        default_entity_type=default_entity_type,
    )

    output_root = Path(output_root)
    raw_dir = ensure_dir(output_root / "raw")
    bie_dir = ensure_dir(output_root / "bie")
    metadata_dir = ensure_dir(output_root / "metadata")

    for split_name, split_events in split_payload["events"].items():
        write_text(raw_dir / f"{split_name}.txt", _render_split(split_events))
    _write_entity_metadata(bie_dir / "entity_metadata.jsonl", entities)
    write_json(metadata_dir / "time_index.json", split_payload["time_index"])
    write_json(metadata_dir / "split_manifest.json", split_payload["manifest"])
    write_json(metadata_dir / "entity_manifest.json", {"num_entities": len(entities), "entity_ids": sorted(entities.keys())})

    return {
        "raw_dir": str(raw_dir),
        "bie_path": str(bie_dir / "entity_metadata.jsonl"),
        "time_index_path": str(metadata_dir / "time_index.json"),
        "split_manifest_path": str(metadata_dir / "split_manifest.json"),
        "num_entities": len(entities),
        "num_events": len(events),
        "split_sizes": split_payload["manifest"]["split_sizes"],
    }


def prepare_dataset_cli(args) -> None:
    extra_cols = [value.strip() for value in args.entity_extra_cols.split(",") if value.strip()]
    payload = prepare_dataset(
        events_path=args.events,
        entities_path=args.entities,
        output_root=args.output_root,
        head_col=args.head_col,
        relation_col=args.relation_col,
        tail_col=args.tail_col,
        time_col=args.time_col,
        entity_id_col=args.entity_id_col,
        entity_name_col=args.entity_name_col,
        entity_extra_cols=extra_cols,
        delimiter=_decode_delimiter(args.delimiter),
        entity_delimiter=_decode_delimiter(args.entity_delimiter),
        time_granularity=args.time_granularity,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        default_entity_type=args.default_entity_type,
        keep_duplicates=args.keep_duplicates,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def normalize_timestamp(raw_value: str, granularity: str) -> tuple[Any, str]:
    value = raw_value.strip()
    numeric = _try_parse_numeric(value)
    if numeric is not None:
        return ("numeric", numeric), str(numeric)

    dt = _parse_datetime(value)
    if dt is None:
        return ("string", value), value

    if granularity in {"auto", "day"}:
        bucket = date(dt.year, dt.month, dt.day)
        normalized = datetime(dt.year, dt.month, dt.day).isoformat()
        return ("datetime", normalized), bucket.isoformat()
    if granularity == "hour":
        bucket = datetime(dt.year, dt.month, dt.day, dt.hour)
        return ("datetime", bucket.isoformat()), bucket.strftime("%Y-%m-%dT%H")
    return ("datetime", dt.isoformat()), dt.isoformat()


def build_time_index(rows: list[dict[str, Any]]) -> dict[str, int]:
    unique = {}
    for row in rows:
        unique[row["bucket_label"]] = row["sort_key"]
    ordered = sorted(unique.items(), key=lambda item: item[1])
    return {label: index for index, (label, _) in enumerate(ordered, start=1)}


def split_events_by_time(events: list[RawEvent], train_ratio: float, valid_ratio: float) -> dict[str, Any]:
    time_buckets = sorted({event.bucket_index for event in events})
    train_buckets, valid_buckets, test_buckets = _bucket_split_boundaries(
        num_buckets=len(time_buckets),
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    bucket_lookup = {}
    for bucket in time_buckets[:train_buckets]:
        bucket_lookup[bucket] = "train"
    for bucket in time_buckets[train_buckets: train_buckets + valid_buckets]:
        bucket_lookup[bucket] = "valid"
    for bucket in time_buckets[train_buckets + valid_buckets:]:
        bucket_lookup[bucket] = "test"

    splits = {"train": [], "valid": [], "test": []}
    for event in sorted(events, key=lambda item: (item.bucket_index, item.subject, item.relation, item.object)):
        splits[bucket_lookup[event.bucket_index]].append(event)

    manifest = {
        "num_time_buckets": len(time_buckets),
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "split_time_ranges": {
            name: _range_for_split(values)
            for name, values in splits.items()
        },
    }
    time_index = {
        event.bucket_label: event.bucket_index
        for event in sorted(events, key=lambda item: item.bucket_index)
    }
    return {"events": splits, "manifest": manifest, "time_index": time_index}


def build_entity_payload(
    *,
    events: list[RawEvent],
    entities_path: str | Path | None,
    entity_id_col: str,
    entity_name_col: str,
    entity_extra_cols: list[str],
    delimiter: str,
    default_entity_type: str,
) -> dict[str, dict[str, str]]:
    event_entity_ids = sorted({event.subject for event in events} | {event.object for event in events})
    payload: dict[str, dict[str, str]] = {}
    if entities_path:
        for row in _load_rows(entities_path, delimiter=delimiter):
            entity_id = _require_string(row, entity_id_col)
            if entity_extra_cols:
                attributes = {
                    key: str(row[key])
                    for key in entity_extra_cols
                    if key in row and row[key] not in ("", None)
                }
            else:
                attributes = {
                    str(key): str(value)
                    for key, value in row.items()
                    if key not in {entity_id_col, entity_name_col} and value not in ("", None)
                }
            if "entity_type" not in attributes:
                attributes["entity_type"] = default_entity_type
            payload[entity_id] = {
                "entity_id": entity_id,
                "entity_name": str(row.get(entity_name_col, entity_id)),
                **attributes,
            }
    for entity_id in event_entity_ids:
        payload.setdefault(
            entity_id,
            {
                "entity_id": entity_id,
                "entity_name": entity_id,
                "entity_type": default_entity_type,
            },
        )
    return {entity_id: payload[entity_id] for entity_id in event_entity_ids}


def _load_rows(path: str | Path, delimiter: str) -> list[dict[str, Any]]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in source.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        raise ValueError(f"JSON payload must be a list: {source}")
    if suffix in {".csv", ".tsv", ".txt"}:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            return list(reader)
    raise ValueError(f"Unsupported file format: {source}")


def _render_split(events: list[RawEvent]) -> str:
    lines = [
        f"{event.subject}\t{event.relation}\t{event.object}\t{event.bucket_index}"
        for event in events
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def _write_entity_metadata(path: str | Path, payload: dict[str, dict[str, str]]) -> None:
    rows = []
    for entity_id in sorted(payload):
        rows.append(json.dumps(payload[entity_id], ensure_ascii=False))
    write_text(path, "\n".join(rows) + ("\n" if rows else ""))


def _deduplicate_events(events: list[RawEvent]) -> list[RawEvent]:
    seen = set()
    output = []
    for event in events:
        key = (event.subject, event.relation, event.object, event.bucket_index)
        if key in seen:
            continue
        seen.add(key)
        output.append(event)
    return output


def _bucket_split_boundaries(num_buckets: int, train_ratio: float, valid_ratio: float) -> tuple[int, int, int]:
    if num_buckets <= 0:
        raise ValueError("No time buckets available for splitting.")
    if num_buckets == 1:
        return 1, 0, 0
    if num_buckets == 2:
        return 1, 0, 1
    train_count = max(1, int(round(num_buckets * train_ratio)))
    valid_count = max(1, int(round(num_buckets * valid_ratio)))
    if train_count >= num_buckets:
        train_count = num_buckets - 2
    if train_count + valid_count >= num_buckets:
        valid_count = 1
    test_count = num_buckets - train_count - valid_count
    if test_count <= 0:
        test_count = 1
        if valid_count > 1:
            valid_count -= 1
        else:
            train_count = max(1, train_count - 1)
    return train_count, valid_count, test_count


def _range_for_split(events: list[RawEvent]) -> dict[str, int] | None:
    if not events:
        return None
    indices = [event.bucket_index for event in events]
    return {"start": min(indices), "end": max(indices)}


def _validate_ratios(train_ratio: float, valid_ratio: float) -> None:
    if train_ratio <= 0.0 or train_ratio >= 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    if valid_ratio < 0.0 or valid_ratio >= 1.0:
        raise ValueError("valid_ratio must be in [0, 1).")
    if train_ratio + valid_ratio >= 1.0:
        raise ValueError("train_ratio + valid_ratio must be smaller than 1.0.")


def _require_string(row: dict[str, Any], key: str) -> str:
    if key not in row or row[key] in ("", None):
        raise KeyError(f"Missing required column '{key}' in row: {row}")
    return str(row[key]).strip()


def _try_parse_numeric(value: str) -> int | float | None:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return None


def _parse_datetime(value: str) -> datetime | None:
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(candidate)
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
    ):
        try:
            return datetime.strptime(candidate, fmt)
        except ValueError:
            continue
    return None


def _decode_delimiter(value: str) -> str:
    if value == r"\t":
        return "\t"
    return value
