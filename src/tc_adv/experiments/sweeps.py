"""Sweep helpers for Chapter 4 experiments.

Each sweep:
1) clones a base TCADVExperimentConfig
2) injects a per-cell override (gamma / temperature / seed / noise sigma / ...)
3) calls the trainer/evaluator
4) writes a per-cell metrics JSON and a sweep-level summary

The implementation is deliberately I/O-driven so that the user can resume / inspect
intermediate cells, and so that downstream plotting scripts read from disk only.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from tc_adv.config.loader import dump_tcadv_config, load_tcadv_config
from tc_adv.config.schemas import TCADVExperimentConfig
from tc_adv.training.trainer import TCADVTrainer
from tc_adv.utils.io import ensure_dir, read_json, write_json


def _clone_config(base: TCADVExperimentConfig) -> TCADVExperimentConfig:
    return copy.deepcopy(base)


def _suffix(value: float | int | str) -> str:
    if isinstance(value, float):
        return ("%g" % value).replace(".", "p").replace("-", "m")
    return str(value)


def _attach_paths(cfg: TCADVExperimentConfig, base: TCADVExperimentConfig, suffix: str) -> None:
    cfg.name = f"{base.name}_{suffix}"
    cfg.output_dir = str(Path(base.output_dir).with_name(Path(base.output_dir).name + f"_{suffix}"))
    cfg.log_dir = str(Path(base.log_dir).with_name(Path(base.log_dir).name + f"_{suffix}"))
    cfg.checkpoint_dir = str(Path(base.checkpoint_dir).with_name(Path(base.checkpoint_dir).name + f"_{suffix}"))


def run_seed_sweep(
    base_config_path: str,
    seeds: Sequence[int],
    summary_dir: str | None = None,
) -> dict[str, Any]:
    """Run identical training under different seeds and aggregate mean/std.

    Used for Table 4-4 / Table 4-5 / Table 4-7 statistics where the chapter
    reports values as `mean (std)` over 5 independent runs.
    """
    base = load_tcadv_config(base_config_path)
    summary_dir = ensure_dir(summary_dir or (Path(base.output_dir).parent / f"seed_sweep_{base.name}"))
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        cfg = _clone_config(base)
        _attach_paths(cfg, base, suffix=f"seed{seed}")
        cfg.metadata = {**base.metadata, "seed": int(seed)}
        cfg_path = Path(summary_dir) / f"config_seed{seed}.json"
        cfg_path.write_text(json.dumps(dump_tcadv_config(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        trainer = TCADVTrainer(config=cfg)
        metrics = trainer.train()
        record = {"seed": int(seed), "config": str(cfg_path), **_metric_payload(metrics)}
        write_json(Path(cfg.output_dir) / "seed_metrics.json", record)
        rows.append(record)
    summary = aggregate_rows(rows, group_by=None)
    summary_payload = {"base_config": base_config_path, "seeds": list(map(int, seeds)), "rows": rows, "summary": summary}
    write_json(Path(summary_dir) / "seed_summary.json", summary_payload)
    return summary_payload


def run_gamma_sweep(
    base_config_path: str,
    gammas: Sequence[float],
    summary_dir: str | None = None,
) -> dict[str, Any]:
    """Sweep fusion gamma for Table 4-7 (default cells 0.1/0.3/0.5/0.7/0.9)."""
    base = load_tcadv_config(base_config_path)
    summary_dir = ensure_dir(summary_dir or (Path(base.output_dir).parent / f"gamma_sweep_{base.name}"))
    rows: list[dict[str, Any]] = []
    for gamma in gammas:
        cfg = _clone_config(base)
        _attach_paths(cfg, base, suffix=f"gamma{_suffix(gamma)}")
        cfg.tc_adv.fusion.gamma = float(gamma)
        trainer = TCADVTrainer(config=cfg)
        metrics = trainer.train()
        record = {"gamma": float(gamma), **_metric_payload(metrics)}
        rows.append(record)
    summary_payload = {"base_config": base_config_path, "gammas": list(map(float, gammas)), "rows": rows}
    write_json(Path(summary_dir) / "gamma_summary.json", summary_payload)
    return summary_payload


def run_temperature_sweep(
    base_config_path: str,
    temperatures: Sequence[float],
    summary_dir: str | None = None,
) -> dict[str, Any]:
    """Sweep Gumbel-Softmax fixed temperature for Figure 4-8.

    The chapter explicitly notes this analysis trains at *fixed* T (no
    annealing) so each cell can be attributed to its temperature alone.
    """
    base = load_tcadv_config(base_config_path)
    summary_dir = ensure_dir(summary_dir or (Path(base.output_dir).parent / f"temperature_sweep_{base.name}"))
    rows: list[dict[str, Any]] = []
    for temperature in temperatures:
        cfg = _clone_config(base)
        _attach_paths(cfg, base, suffix=f"T{_suffix(temperature)}")
        cfg.tc_adv.gumbel.fixed_temp = float(temperature)
        cfg.tc_adv.gumbel.start_temp = float(temperature)
        cfg.tc_adv.gumbel.min_temp = float(temperature)
        trainer = TCADVTrainer(config=cfg)
        metrics = trainer.train()
        record = {"temperature": float(temperature), **_metric_payload(metrics)}
        rows.append(record)
    summary_payload = {"base_config": base_config_path, "temperatures": list(map(float, temperatures)), "rows": rows}
    write_json(Path(summary_dir) / "temperature_summary.json", summary_payload)
    return summary_payload


def run_noise_sweep(
    base_config_path: str,
    sigmas: Sequence[float],
    checkpoint_name: str = "best",
    summary_dir: str | None = None,
) -> dict[str, Any]:
    """Sweep timestamp Gaussian perturbation sigma for Table 4-9.

    Assumes the base config has already been trained — this only runs the
    `evaluate_with_noise` side, which mirrors the chapter setup that injects
    noise at evaluation time.
    """
    base = load_tcadv_config(base_config_path)
    summary_dir = ensure_dir(summary_dir or (Path(base.output_dir).parent / f"noise_sweep_{base.name}"))
    trainer = TCADVTrainer(config=base)
    rows: list[dict[str, Any]] = []
    for sigma in sigmas:
        if float(sigma) == 0.0:
            metrics = trainer.evaluate(split="test", checkpoint_name=checkpoint_name)
        else:
            metrics = trainer.evaluate_with_noise(split="test", checkpoint_name=checkpoint_name, sigma=float(sigma))
        record = {"sigma": float(sigma), **_metric_payload(metrics)}
        rows.append(record)
    summary_payload = {"base_config": base_config_path, "sigmas": list(map(float, sigmas)), "rows": rows}
    write_json(Path(summary_dir) / "noise_summary.json", summary_payload)
    return summary_payload


def run_multistep_eval(
    base_config_path: str,
    max_steps: int = 5,
    checkpoint_name: str = "best",
    summary_dir: str | None = None,
) -> dict[str, Any]:
    """Multi-step rollout evaluation for Table 4-8 / Figure 4-3."""
    base = load_tcadv_config(base_config_path)
    summary_dir = ensure_dir(summary_dir or (Path(base.output_dir).parent / f"multistep_{base.name}"))
    trainer = TCADVTrainer(config=base)
    metrics_by_step = trainer.evaluate_multi_step(split="test", checkpoint_name=checkpoint_name, max_steps=max_steps)
    rows = [{"step": int(step), **_metric_payload(metrics)} for step, metrics in metrics_by_step.items()]
    summary_payload = {"base_config": base_config_path, "max_steps": int(max_steps), "rows": rows}
    write_json(Path(summary_dir) / "multistep_summary.json", summary_payload)
    return summary_payload


def aggregate_rows(rows: list[dict[str, Any]], group_by: str | None) -> dict[str, Any]:
    """Compute mean/std/CI per metric for a list of metric rows."""
    metric_keys = sorted({k for row in rows for k, v in row.items() if isinstance(v, (int, float)) and k != group_by})
    if group_by is None:
        groups = {"all": rows}
    else:
        groups: dict[Any, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row.get(group_by), []).append(row)
    summary: dict[str, Any] = {}
    for key, items in groups.items():
        cell: dict[str, Any] = {"n": len(items)}
        for metric in metric_keys:
            values = [float(item[metric]) for item in items if metric in item]
            if not values:
                continue
            mean = statistics.fmean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
            cell[metric] = {
                "mean": mean,
                "std": std,
                "sample_std": sample_std,
                "ci95_half_width": _t_ci95_half_width(values),
                "n": len(values),
            }
        summary[str(key)] = cell
    return summary


def paired_t_test(
    treated_values: Sequence[float],
    baseline_values: Sequence[float],
) -> dict[str, float]:
    """Two-sided paired t-test (matched samples).

    Returns t statistic, df, p-value (Student's t), mean diff, and 95% CI of
    the mean difference. We implement this manually to avoid a SciPy
    dependency and so the user can run it in the slim ModelScope env.
    """
    if len(treated_values) != len(baseline_values):
        raise ValueError("paired t-test requires equal-length sequences")
    diffs = [float(t) - float(b) for t, b in zip(treated_values, baseline_values)]
    n = len(diffs)
    if n < 2:
        return {"n": n, "t": 0.0, "p_two_sided": 1.0, "mean_diff": diffs[0] if diffs else 0.0, "df": 0,
                "ci95_low": 0.0, "ci95_high": 0.0}
    mean_diff = statistics.fmean(diffs)
    sample_std = statistics.stdev(diffs)
    se = sample_std / math.sqrt(n)
    t_stat = mean_diff / se if se > 0 else 0.0
    df = n - 1
    p_two_sided = _student_t_p_two_sided(t_stat, df)
    half = _student_t_critical_two_sided(df, 0.05) * se
    return {
        "n": n,
        "t": t_stat,
        "df": df,
        "p_two_sided": p_two_sided,
        "mean_diff": mean_diff,
        "ci95_low": mean_diff - half,
        "ci95_high": mean_diff + half,
    }


def _t_ci95_half_width(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    sample_std = statistics.stdev(values)
    se = sample_std / math.sqrt(n)
    return _student_t_critical_two_sided(n - 1, 0.05) * se


def _student_t_p_two_sided(t: float, df: int) -> float:
    if df <= 0:
        return 1.0
    x = df / (df + t * t)
    p_one_tail = 0.5 * _regularized_incomplete_beta(0.5 * df, 0.5, x)
    return min(1.0, max(0.0, 2.0 * p_one_tail))


def _student_t_critical_two_sided(df: int, alpha: float) -> float:
    if df <= 0:
        return 0.0
    target = 1.0 - alpha
    low, high = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        p_one_tail = 1.0 - 0.5 * _regularized_incomplete_beta(0.5 * df, 0.5, df / (df + mid * mid))
        cdf = p_one_tail
        coverage = 2.0 * cdf - 1.0
        if coverage < target:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log1p(-x) * b - log_b) / a
    cf = _beta_continued_fraction(a, b, x)
    return front * cf


def _beta_continued_fraction(a: float, b: float, x: float, max_iter: int = 200, eps: float = 1e-12) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _metric_payload(metrics: dict[str, Any]) -> dict[str, Any]:
    """Strip non-scalar diagnostic fields so summary JSON stays compact."""
    payload: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            payload[key] = float(value)
        elif isinstance(value, dict):
            payload[key] = {sub_k: float(sub_v) if isinstance(sub_v, (int, float)) else sub_v
                            for sub_k, sub_v in value.items()}
    return payload
