import csv
import json
from pathlib import Path

from tc_adv.data.prepare import prepare_dataset


def test_prepare_dataset_from_csv(tmp_path: Path):
    events_path = tmp_path / "events.csv"
    entities_path = tmp_path / "entities.csv"

    with events_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["head", "rel", "tail", "time"])
        writer.writeheader()
        writer.writerows(
            [
                {"head": "A", "rel": "ally", "tail": "B", "time": "2014-01-01"},
                {"head": "A", "rel": "ally", "tail": "C", "time": "2014-01-02"},
                {"head": "B", "rel": "trade", "tail": "C", "time": "2014-01-03"},
                {"head": "C", "rel": "ally", "tail": "D", "time": "2014-01-04"},
                {"head": "D", "rel": "trade", "tail": "A", "time": "2014-01-05"},
            ]
        )

    with entities_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "name", "entity_type", "country"])
        writer.writeheader()
        writer.writerows(
            [
                {"id": "A", "name": "OrgA", "entity_type": "company", "country": "US"},
                {"id": "B", "name": "OrgB", "entity_type": "company", "country": "US"},
            ]
        )

    payload = prepare_dataset(
        events_path=events_path,
        entities_path=entities_path,
        output_root=tmp_path / "dataset",
        head_col="head",
        relation_col="rel",
        tail_col="tail",
        time_col="time",
        entity_id_col="id",
        entity_name_col="name",
        train_ratio=0.6,
        valid_ratio=0.2,
    )

    assert payload["num_events"] == 5
    assert (tmp_path / "dataset" / "raw" / "train.txt").exists()
    assert (tmp_path / "dataset" / "raw" / "valid.txt").exists()
    assert (tmp_path / "dataset" / "raw" / "test.txt").exists()
    mapping = json.loads((tmp_path / "dataset" / "metadata" / "time_index.json").read_text(encoding="utf-8"))
    assert list(mapping.values()) == [1, 2, 3, 4, 5]
    entity_lines = (tmp_path / "dataset" / "bie" / "entity_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert any('"entity_id": "A"' in line for line in entity_lines)
    assert any('"entity_id": "D"' in line for line in entity_lines)


def test_prepare_dataset_without_entity_table_generates_minimal_bie(tmp_path: Path):
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        "\n".join(
            [
                json.dumps({"subject": "X", "relation": "r", "object": "Y", "timestamp": 10}),
                json.dumps({"subject": "Y", "relation": "r", "object": "Z", "timestamp": 20}),
                json.dumps({"subject": "Z", "relation": "r", "object": "X", "timestamp": 30}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    prepare_dataset(events_path=events_path, output_root=tmp_path / "dataset", train_ratio=0.67, valid_ratio=0.0)
    entity_lines = (tmp_path / "dataset" / "bie" / "entity_metadata.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(entity_lines) == 3
    assert all('"entity_type": "UNKNOWN"' in line for line in entity_lines)
