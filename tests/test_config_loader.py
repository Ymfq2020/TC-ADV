import json
from pathlib import Path

from tc_adv.config.loader import dump_tcadv_config, load_tcadv_config


def test_load_smoke_config():
    config = load_tcadv_config("configs/experiments/smoke.yaml")
    assert config.name == "tcadv_smoke"
    assert config.tc_adv.fusion.gamma == 0.6
    assert config.tc_adv.trainer.g_steps == 3
    assert config.tc_adv.gumbel.min_temp == 0.05


def test_dump_round_trip(tmp_path: Path):
    config = load_tcadv_config("configs/experiments/smoke.yaml")
    dumped = dump_tcadv_config(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
    loaded = load_tcadv_config(path)
    assert loaded.tc_adv.loss.alpha == config.tc_adv.loss.alpha
    assert loaded.lmca_experiment_config == config.lmca_experiment_config
