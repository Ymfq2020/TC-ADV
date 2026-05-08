"""Generate all Chapter 4 figures from existing sweep outputs.

Wrapper that the user can call once after running every sweep. By default it
looks for sweep summaries under `outputs/` and writes figures into
`outputs/figures/`. Override the per-figure paths if your layout differs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_adv.experiments.plotting import (
    plot_error_types,
    plot_longtail,
    plot_multistep_decay,
    plot_noise_tvr,
    plot_temperature,
    plot_train_dynamics,
    plot_transfer_scatter,
)
from tc_adv.utils.io import ensure_dir


def _existing_paths(paths: list[str]) -> list[Path]:
    return [Path(p) for p in paths if Path(p).exists()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 figures (4-2 to 4-9)")
    parser.add_argument("--output-dir", default="outputs/figures")
    parser.add_argument("--error-types", default="outputs/error_types.json")
    parser.add_argument("--multistep", nargs="+", default=[
        "outputs/multistep_tcadv_a10_icews14/multistep_summary.json",
        "outputs/multistep_tcadv_a10_gdelt/multistep_summary.json",
    ])
    parser.add_argument("--noise", nargs="+", default=[
        "outputs/noise_sweep_tcadv_a10_icews18/noise_summary.json",
        "outputs/noise_sweep_tcadv_a10_gdelt/noise_summary.json",
    ])
    parser.add_argument("--transfer", default="outputs/transfer_summary.json")
    parser.add_argument("--train-history", default="outputs/tcadv_a10_icews14/train_history.jsonl")
    parser.add_argument("--temperature", nargs="+", default=[
        "outputs/temperature_sweep_tcadv_a10_icews14/temperature_summary.json",
    ])
    parser.add_argument("--longtail", default="outputs/longtail_summary.json")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    if Path(args.error_types).exists():
        print(f"[Figure 4-2] {args.error_types}")
        plot_error_types(Path(args.error_types), output_dir)

    multistep_paths = _existing_paths(args.multistep)
    if multistep_paths:
        print(f"[Figure 4-3] {len(multistep_paths)} runs")
        plot_multistep_decay(multistep_paths, output_dir)

    noise_paths = _existing_paths(args.noise)
    if noise_paths:
        print(f"[Figure 4-4] {len(noise_paths)} runs")
        plot_noise_tvr(noise_paths, output_dir)

    if Path(args.transfer).exists():
        print(f"[Figure 4-5] {args.transfer}")
        plot_transfer_scatter(Path(args.transfer), output_dir)

    if Path(args.train_history).exists():
        print(f"[Figure 4-7] {args.train_history}")
        plot_train_dynamics(Path(args.train_history), output_dir)

    temperature_paths = _existing_paths(args.temperature)
    if temperature_paths:
        print(f"[Figure 4-8] {len(temperature_paths)} runs")
        plot_temperature(temperature_paths, output_dir)

    if Path(args.longtail).exists():
        print(f"[Figure 4-9] {args.longtail}")
        plot_longtail(Path(args.longtail), output_dir)

    print(f"\nFigures written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
