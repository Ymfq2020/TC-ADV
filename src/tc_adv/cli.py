"""Command-line entry points."""

from __future__ import annotations

import argparse

from tc_adv.experiments.runner import evaluate_config, run_experiment_suite, train_config
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
    if args.command == "export-code":
        export_repository_code(output_path=args.output)
        return
    if args.command == "prepare-data":
        prepare_dataset_cli(args)
        return
    raise ValueError(f"Unknown command: {args.command}")
