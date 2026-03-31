import json
from pathlib import Path

from tc_adv.config.loader import dump_tcadv_config, load_tcadv_config
from tc_adv.experiments.runner import run_experiment_suite
from tc_adv.training.trainer import TCADVTrainer


def _write_temp_config(tmp_path: Path, name: str, source_path: str) -> Path:
    config = load_tcadv_config(source_path)
    config.name = name
    config.output_dir = str(tmp_path / "outputs" / name)
    config.log_dir = str(tmp_path / "logs" / name)
    config.checkpoint_dir = str(tmp_path / "checkpoints" / name)
    destination = tmp_path / f"{name}.json"
    destination.write_text(json.dumps(dump_tcadv_config(config), ensure_ascii=False), encoding="utf-8")
    return destination


def test_local_fixture_pipeline_runs_end_to_end(tmp_path: Path):
    config = load_tcadv_config("configs/experiments/smoke.yaml")
    config.name = "tcadv_smoke_test"
    config.output_dir = str(tmp_path / "outputs")
    config.log_dir = str(tmp_path / "logs")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    trainer = TCADVTrainer(config)
    metrics = trainer.train()
    assert "MRR" in metrics
    assert "TVR" in metrics
    assert (Path(config.output_dir) / "train_history.jsonl").exists()
    assert (Path(config.output_dir) / "test_metrics.json").exists()


def test_suite_runner_writes_aggregate_outputs(tmp_path: Path):
    config_path = _write_temp_config(tmp_path, "suite_smoke", "configs/experiments/smoke.yaml")
    results = run_experiment_suite([str(config_path)])
    assert results["runs"]
    output_dir = tmp_path / "outputs" / "suite_smoke"
    assert (output_dir / "suite_metrics.csv").exists()
    assert (output_dir / "suite_metrics.json").exists()
