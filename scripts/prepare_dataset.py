from tc_adv.data.prepare import prepare_dataset_cli


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare raw CSV/JSONL files into TC-ADV dataset layout")
    parser.add_argument("--events", required=True)
    parser.add_argument("--entities")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--head-col", default="subject")
    parser.add_argument("--relation-col", default="relation")
    parser.add_argument("--tail-col", default="object")
    parser.add_argument("--time-col", default="timestamp")
    parser.add_argument("--entity-id-col", default="entity_id")
    parser.add_argument("--entity-name-col", default="entity_name")
    parser.add_argument("--entity-extra-cols", default="")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--entity-delimiter", default=",")
    parser.add_argument("--time-granularity", default="auto", choices=["auto", "raw", "day", "hour"])
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--default-entity-type", default="UNKNOWN")
    parser.add_argument("--keep-duplicates", action="store_true")
    prepare_dataset_cli(parser.parse_args())
