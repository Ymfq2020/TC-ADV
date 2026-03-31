"""Typed configuration objects for the TC-ADV repository."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TRMConfig:
    bandwidth: str | float = "auto"
    epsilon: float = 1e-5


@dataclass
class ECMConfig:
    num_heads: int = 4
    hidden_dim: int | None = None
    time_encoding_dim: int = 32
    history_window: int | None = None


@dataclass
class FusionConfig:
    gamma: float = 0.6


@dataclass
class LossConfig:
    alpha: float = 1.0
    beta: float = 2.5
    use_static_margin: bool = False
    static_margin: float = 1.0


@dataclass
class GumbelConfig:
    start_temp: float = 1.0
    anneal_rate: float = 0.95
    min_temp: float = 0.05


@dataclass
class TrainerConfig:
    generator_lr: float = 1e-3
    discriminator_lr: float = 5e-4
    max_epochs: int = 100
    early_stopping_patience: int = 10
    g_steps: int = 3
    d_steps: int = 1
    topk_fake_candidates: int = 32
    generator_loss_weight: float = 1.0
    adversarial_loss_weight: float = 1.0


@dataclass
class ReportConfig:
    include_configs: bool = True
    include_package: bool = True
    include_scripts: bool = True
    include_tests: bool = True
    output_format: str = "markdown"


@dataclass
class TCADVConfig:
    trm: TRMConfig = field(default_factory=TRMConfig)
    ecm: ECMConfig = field(default_factory=ECMConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gumbel: GumbelConfig = field(default_factory=GumbelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    report: ReportConfig = field(default_factory=ReportConfig)


@dataclass
class TCADVExperimentConfig:
    name: str
    lmca_experiment_config: str
    tc_adv: TCADVConfig = field(default_factory=TCADVConfig)
    output_dir: str = "outputs/default"
    log_dir: str = "logs/default"
    checkpoint_dir: str = "checkpoints/default"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
