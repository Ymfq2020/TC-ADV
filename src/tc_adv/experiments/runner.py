"""Experiment orchestration and result aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

from tc_adv.config.loader import load_tcadv_config
from tc_adv.training.trainer import TCADVTrainer
from tc_adv.utils.io import write_json


def train_config(config_path: str) -> dict[str, object]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics = trainer.train()
    return {"name": config.name, **metrics, "output_dir": config.output_dir}


def evaluate_config(config_path: str) -> dict[str, object]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics = trainer.evaluate(split="test", checkpoint_name="best")
    return {"name": config.name, **metrics, "output_dir": config.output_dir}


def evaluate_noise_config(config_path: str, sigma: float = 1.0) -> dict[str, object]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics = trainer.evaluate_with_noise(split="test", checkpoint_name="best", sigma=sigma)
    return {"name": config.name, **metrics, "output_dir": config.output_dir, "sigma": sigma}


def evaluate_multi_step_config(config_path: str, max_steps: int = 5) -> dict[int, dict[str, float]]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics_by_step = trainer.evaluate_multi_step(split="test", checkpoint_name="best", max_steps=max_steps)
    return metrics_by_step


def evaluate_tvr_offline(config_path: str, predictions_path: str) -> dict[str, float]:
    from tc_adv.experiments.tvr_evaluator import TVREvaluator
    evaluator = TVREvaluator(config_path=config_path)
    metrics = evaluator.evaluate_predictions(predictions_path)
    return metrics


def run_experiment_suite(config_paths: list[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for config_path in config_paths:
        config = load_tcadv_config(config_path)
        try:
            metrics = train_config(config_path)
            rows.append(metrics)
        except Exception as exc:  # pragma: no cover - suite should continue on missing full datasets
            failures.append({"config": config.name, "error": str(exc)})
            rows.append({"name": config.name, "status": "failed", "error": str(exc)})
    if rows:
        output_dir = Path(load_tcadv_config(config_paths[0]).output_dir)
        _write_csv(output_dir / "suite_metrics.csv", rows)
        write_json(output_dir / "suite_metrics.json", rows)
        write_json(output_dir / "violation_breakdown.json", {"failures": failures, "runs": rows})
    return {"runs": rows, "failures": failures}


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
