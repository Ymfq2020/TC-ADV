"""Complexity / efficiency benchmark (Table 4-10).

Reports for a given config:
    - parameter count of generator (LMCA-TIC) and discriminators (TRM + ECM)
    - average training-step wall time (ms / batch) over a warm-up loop
    - average inference latency (ms / query) over the validation set
    - final TVR (read from `valid_metrics.json` if available)

This script is meant to be run after training has produced a `best`
checkpoint; if no checkpoint exists it still measures forward latency on
freshly-initialized weights.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from tc_adv.config.loader import load_tcadv_config
from tc_adv.training.trainer import TCADVTrainer
from tc_adv.utils.io import read_json, write_json


def _count_parameters(module) -> int:
    if torch is None or module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def benchmark(
    config_path: str,
    train_steps: int = 20,
    eval_queries: int = 200,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)

    generator_params = _count_parameters(getattr(trainer.generator, "model", None))
    trm_params = _count_parameters(trainer.trm)
    ecm_params = _count_parameters(trainer.ecm)

    samples = list(trainer.generator.train_dataset.samples)[:max(train_steps, 1)]
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_start = time.perf_counter()
    for sample in samples:
        try:
            trainer._discriminator_step(sample)
            trainer._generator_step(sample)
        except Exception as exc:  # pragma: no cover - defensive bookkeeping
            return {
                "config": config_path,
                "error": f"training-step benchmark failed: {exc}",
                "generator_params": generator_params,
                "trm_params": trm_params,
                "ecm_params": ecm_params,
            }
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    train_elapsed = time.perf_counter() - train_start
    avg_train_step_ms = 1000.0 * train_elapsed / max(len(samples), 1)
    estimated_epoch_seconds = avg_train_step_ms / 1000.0 * len(trainer.generator.train_dataset.samples)

    eval_samples = list(trainer.generator.test_dataset.samples)[:eval_queries]
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_start = time.perf_counter()
    for sample in eval_samples:
        trainer.generator.score_candidates(sample)
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
    eval_elapsed = time.perf_counter() - eval_start
    avg_inference_ms = 1000.0 * eval_elapsed / max(len(eval_samples), 1)

    valid_metrics_path = Path(config.output_dir) / "valid_metrics.json"
    valid_tvr = None
    if valid_metrics_path.exists():
        try:
            valid_tvr = float(read_json(valid_metrics_path).get("TVR", 0.0))
        except Exception:
            valid_tvr = None

    payload = {
        "config": config_path,
        "model_name": config.name,
        "generator_params": generator_params,
        "trm_params": trm_params,
        "ecm_params": ecm_params,
        "total_params": generator_params + trm_params + ecm_params,
        "avg_train_step_ms": avg_train_step_ms,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "avg_inference_ms_per_query": avg_inference_ms,
        "valid_tvr": valid_tvr,
        "n_train_samples_seen": len(samples),
        "n_eval_samples_seen": len(eval_samples),
    }
    if output_path:
        write_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Complexity / efficiency benchmark (Table 4-10)")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-steps", type=int, default=20)
    parser.add_argument("--eval-queries", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    payload = benchmark(args.config, args.train_steps, args.eval_queries, args.output)
    for key, value in payload.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
