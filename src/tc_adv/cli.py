"""Command-line entry points."""

from __future__ import annotations

import argparse

from tc_adv.experiments.runner import evaluate_config, run_experiment_suite, train_config, evaluate_noise_config, evaluate_multi_step_config, evaluate_tvr_offline
from tc_adv.experiments.sweeps import (
    aggregate_rows,
    paired_t_test,
    run_gamma_sweep,
    run_multistep_eval,
    run_noise_sweep,
    run_seed_sweep,
    run_temperature_sweep,
)
from tc_adv.experiments.error_analysis import classify_error_types
from tc_adv.experiments.complexity import benchmark as complexity_benchmark
from tc_adv.data.prepare import prepare_dataset_cli
from tc_adv.report.exporter import export_repository_code


def main() -> None:
    parser = argparse.ArgumentParser(description="TC-ADV")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--config", required=True)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument("--config", required=True)

    suite_parser = subparsers.add_parser("run-suite")
    suite_parser.add_argument("--config", nargs="+", required=True)

    eval_noise_parser = subparsers.add_parser("evaluate-noise")
    eval_noise_parser.add_argument("--config", required=True)
    eval_noise_parser.add_argument("--sigma", type=float, default=1.0)

    eval_ms_parser = subparsers.add_parser("evaluate-multi-step")
    eval_ms_parser.add_argument("--config", required=True)
    eval_ms_parser.add_argument("--max-steps", type=int, default=5)

    eval_tvr_parser = subparsers.add_parser("evaluate-tvr")
    eval_tvr_parser.add_argument("--config", required=True)
    eval_tvr_parser.add_argument("--predictions", required=True)

    seed_parser = subparsers.add_parser("seed-sweep", help="Train under multiple seeds and report mean/std (Table 4-4)")
    seed_parser.add_argument("--config", required=True)
    seed_parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2024, 7, 9])
    seed_parser.add_argument("--summary-dir", default=None)

    gamma_parser = subparsers.add_parser("gamma-sweep", help="Sweep fusion gamma (Table 4-7)")
    gamma_parser.add_argument("--config", required=True)
    gamma_parser.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    gamma_parser.add_argument("--summary-dir", default=None)

    temp_parser = subparsers.add_parser("temperature-sweep", help="Sweep fixed Gumbel-Softmax temperature (Figure 4-8)")
    temp_parser.add_argument("--config", required=True)
    temp_parser.add_argument("--temperatures", type=float, nargs="+",
                             default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    temp_parser.add_argument("--summary-dir", default=None)

    noise_parser = subparsers.add_parser("noise-sweep", help="Sweep timestamp Gaussian sigma (Table 4-9)")
    noise_parser.add_argument("--config", required=True)
    noise_parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    noise_parser.add_argument("--checkpoint", default="best")
    noise_parser.add_argument("--summary-dir", default=None)

    ms_sweep_parser = subparsers.add_parser("multistep-eval", help="Multi-step rollout (Table 4-8 / Figure 4-3)")
    ms_sweep_parser.add_argument("--config", required=True)
    ms_sweep_parser.add_argument("--max-steps", type=int, default=5)
    ms_sweep_parser.add_argument("--checkpoint", default="best")
    ms_sweep_parser.add_argument("--summary-dir", default=None)

    err_parser = subparsers.add_parser("error-types", help="Classify high-confidence violations (Figure 4-2)")
    err_parser.add_argument("--diagnostics", required=True)
    err_parser.add_argument("--threshold", type=float, default=0.7)
    err_parser.add_argument("--output", default=None)

    complexity_parser = subparsers.add_parser("complexity", help="Parameter / time / latency benchmark (Table 4-10)")
    complexity_parser.add_argument("--config", required=True)
    complexity_parser.add_argument("--train-steps", type=int, default=20)
    complexity_parser.add_argument("--eval-queries", type=int, default=200)
    complexity_parser.add_argument("--output", default=None)

    export_parser = subparsers.add_parser("export-code")
    export_parser.add_argument("--output", required=True)

    prepare_parser = subparsers.add_parser("prepare-data")
    prepare_parser.add_argument("--events", required=True)
    prepare_parser.add_argument("--entities")
    prepare_parser.add_argument("--output-root", required=True)
    prepare_parser.add_argument("--head-col", default="subject")
    prepare_parser.add_argument("--relation-col", default="relation")
    prepare_parser.add_argument("--tail-col", default="object")
    prepare_parser.add_argument("--time-col", default="timestamp")
    prepare_parser.add_argument("--entity-id-col", default="entity_id")
    prepare_parser.add_argument("--entity-name-col", default="entity_name")
    prepare_parser.add_argument("--entity-extra-cols", default="")
    prepare_parser.add_argument("--delimiter", default=",")
    prepare_parser.add_argument("--entity-delimiter", default=",")
    prepare_parser.add_argument("--time-granularity", default="auto", choices=["auto", "raw", "day", "hour"])
    prepare_parser.add_argument("--train-ratio", type=float, default=0.8)
    prepare_parser.add_argument("--valid-ratio", type=float, default=0.1)
    prepare_parser.add_argument("--default-entity-type", default="UNKNOWN")
    prepare_parser.add_argument("--keep-duplicates", action="store_true")

    args = parser.parse_args()
    if args.command == "train":
        train_config(args.config)
        return
    if args.command == "evaluate":
        evaluate_config(args.config)
        return
    if args.command == "run-suite":
        run_experiment_suite(args.config)
        return
    if args.command == "evaluate-noise":
        evaluate_noise_config(args.config, args.sigma)
        return
    if args.command == "evaluate-multi-step":
        evaluate_multi_step_config(args.config, args.max_steps)
        return
    if args.command == "evaluate-tvr":
        import json
        metrics = evaluate_tvr_offline(args.config, args.predictions)
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "seed-sweep":
        run_seed_sweep(args.config, args.seeds, args.summary_dir)
        return
    if args.command == "gamma-sweep":
        run_gamma_sweep(args.config, args.gammas, args.summary_dir)
        return
    if args.command == "temperature-sweep":
        run_temperature_sweep(args.config, args.temperatures, args.summary_dir)
        return
    if args.command == "noise-sweep":
        run_noise_sweep(args.config, args.sigmas, args.checkpoint, args.summary_dir)
        return
    if args.command == "multistep-eval":
        run_multistep_eval(args.config, args.max_steps, args.checkpoint, args.summary_dir)
        return
    if args.command == "error-types":
        import json
        payload = classify_error_types(args.diagnostics, args.threshold, args.output)
        print(json.dumps({"counts": payload["counts"], "distribution": payload["distribution"]}, indent=2))
        return
    if args.command == "complexity":
        import json
        payload = complexity_benchmark(args.config, args.train_steps, args.eval_queries, args.output)
        print(json.dumps(payload, indent=2))
        return
    if args.command == "export-code":
        export_repository_code(output_path=args.output)
        return
    if args.command == "prepare-data":
        prepare_dataset_cli(args)
        return
    raise ValueError(f"Unknown command: {args.command}")
