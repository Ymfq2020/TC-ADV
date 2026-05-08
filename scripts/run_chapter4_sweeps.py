"""Run all chapter-4 sweeps for a given config family in a single shot.

Used by the user when reproducing experiments on the ModelScope server.
Default order:
  1. baseline + TC-ADV training across 5 seeds
  2. gamma sweep on ICEWS14
  3. temperature sweep on ICEWS14
  4. noise sweep on ICEWS18
  5. multi-step evaluation on ICEWS14 + GDELT
  6. error-type classification + long-tail analysis from the trained model

Edit the `JOBS` list to match what your server should actually run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tc_adv.experiments.sweeps import (
    run_gamma_sweep,
    run_multistep_eval,
    run_noise_sweep,
    run_seed_sweep,
    run_temperature_sweep,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Chapter 4 sweeps end-to-end")
    parser.add_argument("--icews14-config", default="configs/experiments/a10_icews14.yaml")
    parser.add_argument("--icews18-config", default="configs/experiments/a10_icews18.yaml")
    parser.add_argument("--gdelt-config", default="configs/experiments/a10_gdelt.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024, 7, 9])
    parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--multistep", type=int, default=5)
    parser.add_argument("--skip", choices=["seeds", "gamma", "temperature", "noise", "multistep"],
                        nargs="*", default=[])
    args = parser.parse_args()

    skip = set(args.skip)

    if "seeds" not in skip:
        print("==> Seed sweep on ICEWS18 (Table 4-4)")
        run_seed_sweep(args.icews18_config, args.seeds)
        print("==> Seed sweep on GDELT (Table 4-4)")
        run_seed_sweep(args.gdelt_config, args.seeds)

    if "gamma" not in skip:
        print("==> Gamma sweep on ICEWS14 (Table 4-7)")
        run_gamma_sweep(args.icews14_config, args.gammas)

    if "temperature" not in skip:
        print("==> Temperature sweep on ICEWS14 (Figure 4-8)")
        run_temperature_sweep(args.icews14_config, args.temperatures)

    if "noise" not in skip:
        print("==> Noise sweep on ICEWS18 (Table 4-9)")
        run_noise_sweep(args.icews18_config, args.sigmas)
        print("==> Noise sweep on GDELT (Table 4-9)")
        run_noise_sweep(args.gdelt_config, args.sigmas)

    if "multistep" not in skip:
        print("==> Multi-step evaluation on ICEWS14 (Table 4-8 / Figure 4-3)")
        run_multistep_eval(args.icews14_config, args.multistep)
        print("==> Multi-step evaluation on GDELT (Table 4-8 / Figure 4-3)")
        run_multistep_eval(args.gdelt_config, args.multistep)


if __name__ == "__main__":
    main()
