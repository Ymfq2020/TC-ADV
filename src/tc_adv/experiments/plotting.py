"""Generate the figures for Chapter 4.

All figures are produced from JSON / JSONL summaries written by the sweep
helpers in `tc_adv.experiments.sweeps` and the analysis modules. matplotlib is
imported lazily so the module is still importable in headless environments
without matplotlib.

Generated figures (one PNG + one PDF each):
    fig_4_2_error_types.png       error type pie / bar    (Figure 4-2)
    fig_4_3_multistep_decay.png   MRR decay vs t+k        (Figure 4-3)
    fig_4_4_noise_tvr.png         TVR under sigma=2.0     (Figure 4-4)
    fig_4_5_transfer_scatter.png  base TVR vs delta TVR   (Figure 4-5)
    fig_4_7_train_dynamics.png    L_adv + valid MRR       (Figure 4-7)
    fig_4_8_temperature.png       TVR vs Gumbel T         (Figure 4-8)
    fig_4_9_longtail_mrr.png      MRR by frequency bin    (Figure 4-9)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from tc_adv.utils.io import ensure_dir, read_json, read_jsonl


def _import_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install via `pip install matplotlib`"
        ) from exc
    return plt


def _save_pair(plt, fig, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=200)
    fig.savefig(output_dir / f"{name}.pdf")
    plt.close(fig)


def plot_error_types(error_summary_path: Path, output_dir: Path) -> None:
    plt = _import_matplotlib()
    payload = read_json(error_summary_path)
    distribution = payload["distribution"]
    labels = ["生命周期越界", "演化轨迹反常突变", "静态语义冲突"]
    keys = ["lifecycle_out_of_bounds", "evolution_mutation", "static_semantic_conflict"]
    sizes = [distribution.get(k, 0.0) for k in keys]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    wedges, _, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct="%1.0f%%",
        startangle=120,
        colors=["#5B8DEF", "#F2A65A", "#7FB069"],
    )
    for txt in autotexts:
        txt.set_color("white")
        txt.set_fontweight("bold")
    ax.set_title("Figure 4-2  验证集高置信度错误样本类型分布")
    _save_pair(plt, fig, output_dir, "fig_4_2_error_types")


def plot_multistep_decay(multistep_paths: list[Path], output_dir: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for path in multistep_paths:
        payload = read_json(path)
        rows = sorted(payload["rows"], key=lambda r: r["step"])
        steps = [row["step"] for row in rows]
        mrrs = [row.get("MRR", 0.0) * 100.0 for row in rows]
        label = payload.get("base_config", str(path)).split("/")[-1].replace(".yaml", "")
        ax.plot(steps, mrrs, marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("Prediction step (t+k)")
    ax.set_ylabel("MRR (%)")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=8)
    ax.set_title("Figure 4-3  多步连续预测 MRR 衰减")
    _save_pair(plt, fig, output_dir, "fig_4_3_multistep_decay")


def plot_noise_tvr(noise_summary_paths: list[Path], output_dir: Path, target_sigma: float = 2.0) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bar_width = 0.18
    runs = []
    for path in noise_summary_paths:
        payload = read_json(path)
        target_row = next((row for row in payload["rows"] if abs(row["sigma"] - target_sigma) < 1e-6), None)
        if target_row is None:
            continue
        runs.append({
            "label": payload.get("base_config", str(path)).split("/")[-1].replace(".yaml", ""),
            "tvr": target_row.get("TVR", 0.0) * 100.0,
        })
    xs = list(range(len(runs)))
    bars = ax.bar(xs, [run["tvr"] for run in runs], width=0.6, color="#5B8DEF")
    for bar, run in zip(bars, runs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.4, f"{run['tvr']:.1f}%",
                ha="center", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([run["label"] for run in runs], rotation=20, ha="right")
    ax.set_ylabel("TVR (%)")
    ax.set_title(f"Figure 4-4  极端时间戳扰动 (σ={target_sigma}) 下的 TVR")
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_pair(plt, fig, output_dir, "fig_4_4_noise_tvr")


def plot_transfer_scatter(transfer_summary_path: Path, output_dir: Path) -> None:
    plt = _import_matplotlib()
    payload = read_json(transfer_summary_path)
    rows = payload["rows"]
    fit = payload["fit"]
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    color_by_gen = {}
    palette = ["#5B8DEF", "#F2A65A", "#7FB069", "#E07A5F", "#9D8DF1", "#3D5A80"]
    for row in rows:
        generator = row["generator"]
        if generator not in color_by_gen:
            color_by_gen[generator] = palette[len(color_by_gen) % len(palette)]
        ax.scatter(
            row["base_tvr"] * 100.0,
            row["delta_tvr"] * 100.0,
            color=color_by_gen[generator],
            label=generator,
            s=80,
            zorder=3,
        )
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        seen.setdefault(label, handle)
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best", fontsize=8)

    if rows:
        xs = [row["base_tvr"] * 100.0 for row in rows]
        slope = fit["slope_per_pct"]
        intercept = fit["intercept_pct"]
        x_min, x_max = min(xs), max(xs)
        line_xs = [x_min, x_max]
        line_ys = [slope * x + intercept for x in line_xs]
        ax.plot(line_xs, line_ys, color="black", linestyle="--", linewidth=1.0, label=f"slope={slope:.2f}")

    ax.set_xlabel("Base TVR (%)")
    ax.set_ylabel("ΔTVR (pct points)")
    ax.set_title("Figure 4-5  基础 TVR 与 ΔTVR 散点关系")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save_pair(plt, fig, output_dir, "fig_4_5_transfer_scatter")


def plot_train_dynamics(train_history_path: Path, output_dir: Path) -> None:
    plt = _import_matplotlib()
    rows = read_jsonl(train_history_path)
    epochs = [row["epoch"] for row in rows]
    losses = [row.get("generator_loss", row.get("discriminator_loss", 0.0)) for row in rows]
    if rows and "discriminator_loss" in rows[0]:
        losses = [row.get("discriminator_loss", 0.0) for row in rows]
    mrrs = [row.get("valid_mrr", 0.0) * 100.0 for row in rows]
    fig, ax_loss = plt.subplots(figsize=(7.0, 4.2))
    ax_mrr = ax_loss.twinx()
    line_loss, = ax_loss.plot(epochs, losses, color="#5B8DEF", linewidth=1.6, label="判别器对抗损失 L_adv")
    line_mrr, = ax_mrr.plot(epochs, mrrs, color="#E07A5F", linewidth=1.6, linestyle="--", label="验证集 MRR")
    ax_loss.set_xlabel("对抗迭代轮数 (Epochs)")
    ax_loss.set_ylabel("L_adv")
    ax_mrr.set_ylabel("验证集 MRR (%)")
    lines = [line_loss, line_mrr]
    ax_loss.legend(lines, [line.get_label() for line in lines], loc="best", fontsize=9)
    ax_loss.set_title("Figure 4-7  生成器对抗损失与验证集 MRR 联动")
    ax_loss.grid(True, linestyle="--", alpha=0.5)
    _save_pair(plt, fig, output_dir, "fig_4_7_train_dynamics")


def plot_temperature(temperature_summary_paths: list[Path], output_dir: Path) -> None:
    plt = _import_matplotlib()
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for path in temperature_summary_paths:
        payload = read_json(path)
        rows = sorted(payload["rows"], key=lambda r: r["temperature"])
        ts = [row["temperature"] for row in rows]
        tvrs = [row.get("TVR", 0.0) * 100.0 for row in rows]
        label = payload.get("base_config", str(path)).split("/")[-1].replace(".yaml", "")
        ax.plot(ts, tvrs, marker="o", linewidth=2.0, label=label)
    ax.set_xlabel("Gumbel-Softmax 温度参数 T")
    ax.set_ylabel("TVR (%)")
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Figure 4-8  连续松弛温度 T 对 TVR 的影响")
    ax.grid(True, linestyle="--", alpha=0.5)
    _save_pair(plt, fig, output_dir, "fig_4_8_temperature")


def plot_longtail(longtail_summary_path: Path, output_dir: Path) -> None:
    plt = _import_matplotlib()
    payload = read_json(longtail_summary_path)
    runs = payload["runs"]
    if not runs:
        return
    buckets = ["low", "mid", "high"]
    labels = ["低频(<10)", "中频(10-50)", "高频(>50)"]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    palette = ["#5B8DEF", "#F2A65A"]
    width = 0.35
    xs = list(range(len(buckets)))
    for index, run in enumerate(runs):
        offset = (index - 0.5 * (len(runs) - 1)) * width
        values = [run["summary"][bucket]["MRR"] * 100.0 for bucket in buckets]
        bars = ax.bar([x + offset for x in xs], values, width=width,
                      color=palette[index % len(palette)], label=run["label"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{value:.1f}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MRR (%)")
    ax.set_title("Figure 4-9  不同交互频次实体上的 MRR 对比")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_pair(plt, fig, output_dir, "fig_4_9_longtail_mrr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Chapter 4 figures from sweep summaries")
    parser.add_argument("--output-dir", default="outputs/figures",
                        help="Where PNG/PDF figures are written")
    parser.add_argument("--error-types", default=None,
                        help="JSON produced by error_analysis (Figure 4-2)")
    parser.add_argument("--multistep", action="append", default=None,
                        help="multistep_summary.json paths; repeat for ICEWS14 + GDELT (Figure 4-3)")
    parser.add_argument("--noise", action="append", default=None,
                        help="noise_summary.json paths (Figure 4-4)")
    parser.add_argument("--transfer", default=None,
                        help="transferability summary JSON (Figure 4-5)")
    parser.add_argument("--train-history", default=None,
                        help="train_history.jsonl (Figure 4-7)")
    parser.add_argument("--temperature", action="append", default=None,
                        help="temperature_summary.json (Figure 4-8)")
    parser.add_argument("--longtail", default=None,
                        help="long-tail summary JSON (Figure 4-9)")
    args = parser.parse_args()
    output_dir = ensure_dir(args.output_dir)
    if args.error_types:
        plot_error_types(Path(args.error_types), output_dir)
    if args.multistep:
        plot_multistep_decay([Path(p) for p in args.multistep], output_dir)
    if args.noise:
        plot_noise_tvr([Path(p) for p in args.noise], output_dir)
    if args.transfer:
        plot_transfer_scatter(Path(args.transfer), output_dir)
    if args.train_history:
        plot_train_dynamics(Path(args.train_history), output_dir)
    if args.temperature:
        plot_temperature([Path(p) for p in args.temperature], output_dir)
    if args.longtail:
        plot_longtail(Path(args.longtail), output_dir)
    print(f"figures written to {output_dir}")


if __name__ == "__main__":
    main()
