from pathlib import Path

from tc_adv.config.loader import load_tcadv_config
from tc_adv.training.trainer import TCADVTrainer


def test_sibling_bridge_smoke_pipeline_runs(tmp_path: Path):
    config = load_tcadv_config("configs/experiments/bridge_smoke.yaml")
    config.name = "tcadv_bridge_smoke_test"
    config.output_dir = str(tmp_path / "outputs")
    config.log_dir = str(tmp_path / "logs")
    config.checkpoint_dir = str(tmp_path / "checkpoints")
    trainer = TCADVTrainer(config)
    metrics = trainer.train()
    assert "MRR" in metrics
    assert "TVR" in metrics
    assert (Path(config.output_dir) / "test_diagnostics.json").exists()
