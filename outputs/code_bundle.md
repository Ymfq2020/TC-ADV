```text
# FILE: configs/bridge/full_gdelt_lmca.yaml
```
```yaml
name: full_gdelt
dataset_name: GDELT
raw_dir: data/local/gdelt/raw
processed_dir: data/processed/gdelt
bie_path: data/local/gdelt/bie/entity_metadata.jsonl
bie_ordered_keys: [entity_type, country, sector, city, region, role, alias, description]
ontology_keys: [entity_type, country]
delimiter: "\t"
seed: 42
num_epochs: 100
micro_batch_size: 64
gradient_accumulation_steps: 16
learning_rate: 0.001
warmup_ratio: 0.1
early_stopping_patience: 10
output_dir: outputs/full_gdelt
log_dir: logs/full_gdelt
checkpoint_dir: checkpoints/full_gdelt
model:
  llm_name: Qwen/Qwen3-8B
  smoke_llm_name: distilbert-base-uncased
  embedding_dim: 200
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  tcn_kernel_size: 2
  tcn_dilations: [1, 2, 4, 8]
  tgn_neighbor_size: 10
  tgn_time_window_days: 30
  tgn_memory_dim: 200
  tgn_time_decay_init: 0.1
  fusion_hidden_dim: 128
  use_llm: true
  use_tcn: true
  use_tgn: true
  use_gate: true
  use_gs: false
  use_ni: true
  use_sl: true
  use_4bit: true
negative_sampling:
  mode: ontology_weighted
  k_recall: 256
  n_neg: 64
  tau: 0.7
  alpha: 0.5
  faiss_enabled: false

```
```text
# FILE: configs/bridge/full_icews18_lmca.yaml
```
```yaml
name: full_icews18
dataset_name: ICEWS18
raw_dir: data/local/icews18/raw
processed_dir: data/processed/icews18
bie_path: data/local/icews18/bie/entity_metadata.jsonl
bie_ordered_keys: [entity_type, country, sector, city, region, role, alias, description]
ontology_keys: [entity_type, country]
delimiter: "\t"
seed: 42
num_epochs: 100
micro_batch_size: 64
gradient_accumulation_steps: 16
learning_rate: 0.001
warmup_ratio: 0.1
early_stopping_patience: 10
output_dir: outputs/full_icews18
log_dir: logs/full_icews18
checkpoint_dir: checkpoints/full_icews18
model:
  llm_name: Qwen/Qwen3-8B
  smoke_llm_name: distilbert-base-uncased
  embedding_dim: 200
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  tcn_kernel_size: 2
  tcn_dilations: [1, 2, 4, 8]
  tgn_neighbor_size: 10
  tgn_time_window_days: 30
  tgn_memory_dim: 200
  tgn_time_decay_init: 0.1
  fusion_hidden_dim: 128
  use_llm: true
  use_tcn: true
  use_tgn: true
  use_gate: true
  use_gs: false
  use_ni: true
  use_sl: true
  use_4bit: true
negative_sampling:
  mode: ontology_weighted
  k_recall: 256
  n_neg: 64
  tau: 0.7
  alpha: 0.5
  faiss_enabled: false

```
```text
# FILE: configs/bridge/local_smoke_lmca.yaml
```
```yaml
name: local_smoke_lmca
dataset_name: LOCAL-SMOKE
raw_dir: data/fixtures/smoke/icews14
processed_dir: data/processed/local_smoke_lmca
bie_path: data/fixtures/smoke/bie/entity_metadata.jsonl
bie_ordered_keys: [entity_type, country, sector]
ontology_keys: [entity_type, country]
delimiter: "\t"
seed: 7
num_epochs: 2
micro_batch_size: 2
gradient_accumulation_steps: 1
learning_rate: 0.001
warmup_ratio: 0.0
early_stopping_patience: 2
output_dir: outputs/local_smoke_lmca
log_dir: logs/local_smoke_lmca
checkpoint_dir: checkpoints/local_smoke_lmca
baseline_reference_path: ../LMCA-TIC/references/table3_2_reference.csv
model:
  llm_name: Qwen/Qwen3-8B
  smoke_llm_name: distilbert-base-uncased
  embedding_dim: 64
  lora_r: 4
  lora_alpha: 8
  lora_dropout: 0.05
  tcn_kernel_size: 2
  tcn_dilations: [1, 2]
  tgn_neighbor_size: 4
  tgn_time_window_days: 7
  tgn_memory_dim: 64
  tgn_time_decay_init: 0.1
  fusion_hidden_dim: 32
  use_llm: true
  use_tcn: true
  use_tgn: true
  use_gate: true
  use_gs: false
  use_ni: true
  use_sl: true
  use_4bit: false
negative_sampling:
  mode: ontology_weighted
  k_recall: 16
  n_neg: 4
  tau: 0.7
  alpha: 0.5
  faiss_enabled: false
metadata:
  smoke_mode: true

```
```text
# FILE: configs/experiments/ablation_gamma_sweep.yaml
```
```yaml
name: tcadv_ablation_gamma_sweep
lmca_experiment_config: ../LMCA-TIC/configs/experiments/full_icews14.yaml
output_dir: outputs/tcadv_ablation_gamma_sweep
log_dir: logs/tcadv_ablation_gamma_sweep
checkpoint_dir: checkpoints/tcadv_ablation_gamma_sweep
tc_adv:
  fusion:
    gamma: 0.8
  trainer:
    g_steps: 3
    d_steps: 1

```
```text
# FILE: configs/experiments/ablation_no_ecm.yaml
```
```yaml
name: tcadv_ablation_no_ecm
lmca_experiment_config: ../LMCA-TIC/configs/experiments/full_icews14.yaml
output_dir: outputs/tcadv_ablation_no_ecm
log_dir: logs/tcadv_ablation_no_ecm
checkpoint_dir: checkpoints/tcadv_ablation_no_ecm
metadata:
  disable_ecm: true
tc_adv:
  fusion:
    gamma: 0.6
  trainer:
    g_steps: 3
    d_steps: 1

```
```text
# FILE: configs/experiments/ablation_no_trm.yaml
```
```yaml
name: tcadv_ablation_no_trm
lmca_experiment_config: ../LMCA-TIC/configs/experiments/full_icews14.yaml
output_dir: outputs/tcadv_ablation_no_trm
log_dir: logs/tcadv_ablation_no_trm
checkpoint_dir: checkpoints/tcadv_ablation_no_trm
metadata:
  disable_trm: true
tc_adv:
  fusion:
    gamma: 0.6
  trainer:
    g_steps: 3
    d_steps: 1

```
```text
# FILE: configs/experiments/ablation_static_margin.yaml
```
```yaml
name: tcadv_ablation_static_margin
lmca_experiment_config: ../LMCA-TIC/configs/experiments/full_icews14.yaml
output_dir: outputs/tcadv_ablation_static_margin
log_dir: logs/tcadv_ablation_static_margin
checkpoint_dir: checkpoints/tcadv_ablation_static_margin
tc_adv:
  loss:
    use_static_margin: true
    static_margin: 1.0
  trainer:
    g_steps: 3
    d_steps: 1

```
```text
# FILE: configs/experiments/bridge_smoke.yaml
```
```yaml
name: tcadv_bridge_smoke
lmca_experiment_config: ../LMCA-TIC/configs/experiments/smoke_icews14.yaml
output_dir: outputs/tcadv_bridge_smoke
log_dir: logs/tcadv_bridge_smoke
checkpoint_dir: checkpoints/tcadv_bridge_smoke
tc_adv:
  fusion:
    gamma: 0.6
  loss:
    alpha: 1.0
    beta: 2.5
  gumbel:
    start_temp: 1.0
    anneal_rate: 0.95
    min_temp: 0.05
  trainer:
    generator_lr: 0.001
    discriminator_lr: 0.0005
    max_epochs: 2
    early_stopping_patience: 2
    g_steps: 3
    d_steps: 1
    topk_fake_candidates: 4

```
```text
# FILE: configs/experiments/full_gdelt.yaml
```
```yaml
name: tcadv_full_gdelt
lmca_experiment_config: configs/bridge/full_gdelt_lmca.yaml
output_dir: outputs/tcadv_full_gdelt
log_dir: logs/tcadv_full_gdelt
checkpoint_dir: checkpoints/tcadv_full_gdelt
tc_adv:
  fusion:
    gamma: 0.6
  loss:
    alpha: 1.0
    beta: 2.5
  gumbel:
    start_temp: 1.0
    anneal_rate: 0.95
    min_temp: 0.05
  trainer:
    generator_lr: 0.001
    discriminator_lr: 0.0005
    max_epochs: 100
    early_stopping_patience: 10
    g_steps: 3
    d_steps: 1
    topk_fake_candidates: 32

```
```text
# FILE: configs/experiments/full_icews14.yaml
```
```yaml
name: tcadv_full_icews14
lmca_experiment_config: ../LMCA-TIC/configs/experiments/full_icews14.yaml
output_dir: outputs/tcadv_full_icews14
log_dir: logs/tcadv_full_icews14
checkpoint_dir: checkpoints/tcadv_full_icews14
tc_adv:
  fusion:
    gamma: 0.6
  loss:
    alpha: 1.0
    beta: 2.5
  gumbel:
    start_temp: 1.0
    anneal_rate: 0.95
    min_temp: 0.05
  trainer:
    generator_lr: 0.001
    discriminator_lr: 0.0005
    max_epochs: 100
    early_stopping_patience: 10
    g_steps: 3
    d_steps: 1
    topk_fake_candidates: 32

```
```text
# FILE: configs/experiments/full_icews18.yaml
```
```yaml
name: tcadv_full_icews18
lmca_experiment_config: configs/bridge/full_icews18_lmca.yaml
output_dir: outputs/tcadv_full_icews18
log_dir: logs/tcadv_full_icews18
checkpoint_dir: checkpoints/tcadv_full_icews18
tc_adv:
  fusion:
    gamma: 0.6
  loss:
    alpha: 1.0
    beta: 2.5
  gumbel:
    start_temp: 1.0
    anneal_rate: 0.95
    min_temp: 0.05
  trainer:
    generator_lr: 0.001
    discriminator_lr: 0.0005
    max_epochs: 100
    early_stopping_patience: 10
    g_steps: 3
    d_steps: 1
    topk_fake_candidates: 32

```
```text
# FILE: configs/experiments/smoke.yaml
```
```yaml
name: tcadv_smoke
lmca_experiment_config: configs/bridge/local_smoke_lmca.yaml
output_dir: outputs/tcadv_smoke
log_dir: logs/tcadv_smoke
checkpoint_dir: checkpoints/tcadv_smoke
tc_adv:
  fusion:
    gamma: 0.6
  loss:
    alpha: 1.0
    beta: 2.5
  gumbel:
    start_temp: 1.0
    anneal_rate: 0.95
    min_temp: 0.05
  trainer:
    generator_lr: 0.001
    discriminator_lr: 0.0005
    max_epochs: 2
    early_stopping_patience: 2
    g_steps: 3
    d_steps: 1
    topk_fake_candidates: 3
  trm:
    bandwidth: auto
    epsilon: 1e-5
  ecm:
    num_heads: 4
    hidden_dim: 64
    time_encoding_dim: 32
    history_window: 4

```
```text
# FILE: src/tc_adv/__init__.py
```
```python
"""TC-ADV package."""

__all__ = ["__version__"]

__version__ = "0.1.0"

```
```text
# FILE: src/tc_adv/bridge/__init__.py
```
```python
"""Bridge utilities for sibling repositories."""

```
```text
# FILE: src/tc_adv/bridge/lmca_tic.py
```
```python
"""Bridge layer for importing the sibling LMCA-TIC repository.

This module is the only place where sibling-path injection is allowed. The
rest of the codebase consumes typed wrappers exposed here.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BridgeSymbols:
    load_experiment_config: Any
    dump_experiment_config: Any
    LocalTKGPreprocessor: Any
    LMCATICTrainer: Any
    LocalProcessedDataset: Any
    FilteredEvaluator: Any


class LMCATICBridge:
    """Resolve and expose the sibling LMCA-TIC package.

    The repository contract assumed by this bridge is the current public
    interface of `../LMCA-TIC/src/lmca_tic`.
    """

    def __init__(self, cwd: str | Path | None = None, sibling_name: str = "LMCA-TIC") -> None:
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.sibling_name = sibling_name
        self.sibling_root = self._find_sibling_root()
        self._symbols: BridgeSymbols | None = None

    def validate(self) -> BridgeSymbols:
        if self._symbols is None:
            self._symbols = self._load_symbols()
        return self._symbols

    def resolve_lmca_path(self, path: str | Path) -> str:
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate)
        cwd_candidate = (self.cwd / candidate).resolve()
        if cwd_candidate.exists():
            return str(cwd_candidate)
        sibling_candidate = (self.sibling_root / candidate).resolve()
        return str(sibling_candidate)

    def load_experiment_config(self, path: str | Path):
        resolved_path = self.resolve_lmca_path(path)
        config = self.validate().load_experiment_config(resolved_path)
        return self._materialize_config_paths(config=config, config_path=Path(resolved_path))

    def dump_experiment_config(self, config) -> dict[str, Any]:
        return self.validate().dump_experiment_config(config)

    def create_preprocessor(self, config):
        return self.validate().LocalTKGPreprocessor(config)

    def create_trainer(self, config, smoke_mode: bool = False):
        return self.validate().LMCATICTrainer(config=config, smoke_mode=smoke_mode)

    def create_dataset(self, processed_dir: str | Path, split: str):
        return self.validate().LocalProcessedDataset(processed_dir, split)

    def create_filtered_evaluator(self, filtered_targets: dict[str, list[str]]):
        return self.validate().FilteredEvaluator(filtered_targets)

    def _find_sibling_root(self) -> Path:
        candidates = [
            self.cwd / self.sibling_name,
            self.cwd.parent / self.sibling_name,
            Path(__file__).resolve().parents[4] / self.sibling_name,
        ]
        for candidate in candidates:
            if (candidate / "src" / "lmca_tic").exists():
                return candidate.resolve()
        raise FileNotFoundError(
            f"Unable to locate sibling repository '{self.sibling_name}' from {self.cwd}."
        )

    def _load_symbols(self) -> BridgeSymbols:
        try:
            import lmca_tic  # type: ignore  # noqa: F401
        except Exception:
            sibling_src = str((self.sibling_root / "src").resolve())
            if sibling_src not in sys.path:
                sys.path.insert(0, sibling_src)
        load_mod = importlib.import_module("lmca_tic.config.loader")
        preprocess_mod = importlib.import_module("lmca_tic.data.preprocess")
        dataset_mod = importlib.import_module("lmca_tic.data.dataset")
        trainer_mod = importlib.import_module("lmca_tic.training.trainer")
        eval_mod = importlib.import_module("lmca_tic.evaluation.filtered")
        return BridgeSymbols(
            load_experiment_config=getattr(load_mod, "load_experiment_config"),
            dump_experiment_config=getattr(load_mod, "dump_experiment_config"),
            LocalTKGPreprocessor=getattr(preprocess_mod, "LocalTKGPreprocessor"),
            LMCATICTrainer=getattr(trainer_mod, "LMCATICTrainer"),
            LocalProcessedDataset=getattr(dataset_mod, "LocalProcessedDataset"),
            FilteredEvaluator=getattr(eval_mod, "FilteredEvaluator"),
        )

    @staticmethod
    def _materialize_config_paths(config, config_path: Path):
        repo_root = LMCATICBridge._repo_root_for_config(config_path)
        for field_name in (
            "raw_dir",
            "processed_dir",
            "bie_path",
            "output_dir",
            "log_dir",
            "checkpoint_dir",
            "baseline_reference_path",
        ):
            value = getattr(config, field_name, None)
            if not value:
                continue
            path_value = Path(value)
            if path_value.is_absolute():
                continue
            setattr(config, field_name, str((repo_root / path_value).resolve()))
        return config

    @staticmethod
    def _repo_root_for_config(config_path: Path) -> Path:
        for parent in config_path.resolve().parents:
            if (parent / ".git").exists() or (parent / "src").exists():
                return parent
        return config_path.parent.resolve()

```
```text
# FILE: src/tc_adv/cli.py
```
```python
"""Command-line entry points."""

from __future__ import annotations

import argparse

from tc_adv.experiments.runner import evaluate_config, run_experiment_suite, train_config
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
    raise ValueError(f"Unknown command: {args.command}")

```
```text
# FILE: src/tc_adv/config/__init__.py
```
```python
"""Configuration objects and loaders."""

```
```text
# FILE: src/tc_adv/config/loader.py
```
```python
"""Configuration loading from YAML/JSON files."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .schemas import ECMConfig, FusionConfig, GumbelConfig, LossConfig, ReportConfig, TCADVConfig, TCADVExperimentConfig, TRMConfig, TrainerConfig


def load_tcadv_config(path: str | Path) -> TCADVExperimentConfig:
    data = _load_mapping(path)
    tc_adv_payload = data.pop("tc_adv", {})
    tc_adv = TCADVConfig(
        trm=TRMConfig(**tc_adv_payload.get("trm", {})),
        ecm=ECMConfig(**tc_adv_payload.get("ecm", {})),
        fusion=FusionConfig(**tc_adv_payload.get("fusion", {})),
        loss=LossConfig(**tc_adv_payload.get("loss", {})),
        gumbel=GumbelConfig(**tc_adv_payload.get("gumbel", {})),
        trainer=TrainerConfig(**tc_adv_payload.get("trainer", {})),
        report=ReportConfig(**tc_adv_payload.get("report", {})),
    )
    return TCADVExperimentConfig(tc_adv=tc_adv, **data)


def dump_tcadv_config(config: TCADVExperimentConfig) -> dict[str, Any]:
    return {
        "name": config.name,
        "lmca_experiment_config": config.lmca_experiment_config,
        "output_dir": config.output_dir,
        "log_dir": config.log_dir,
        "checkpoint_dir": config.checkpoint_dir,
        "metadata": config.metadata,
        "tc_adv": asdict(config.tc_adv),
    }


def _load_mapping(path: str | Path) -> dict[str, Any]:
    payload = Path(path).read_text(encoding="utf-8")
    suffix = Path(path).suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if yaml is None:
            return _mini_yaml_load(payload)
        return yaml.safe_load(payload)
    return json.loads(payload)


def _mini_yaml_load(payload: str) -> dict[str, Any]:
    lines = []
    for raw_line in payload.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(raw_line.rstrip("\n"))

    def parse_mapping(index: int, indent: int) -> tuple[dict[str, Any], int]:
        mapping: dict[str, Any] = {}
        while index < len(lines):
            line = lines[index]
            curr_indent = len(line) - len(line.lstrip(" "))
            if curr_indent < indent:
                break
            if curr_indent != indent:
                raise ValueError(f"Invalid indentation in config line: {line}")
            stripped = line.strip()
            key, _, rest = stripped.partition(":")
            if not _:
                raise ValueError(f"Invalid YAML mapping line: {line}")
            if rest.strip():
                mapping[key] = _parse_scalar(rest.strip())
                index += 1
                continue
            if index + 1 >= len(lines):
                mapping[key] = {}
                index += 1
                continue
            next_line = lines[index + 1]
            next_indent = len(next_line) - len(next_line.lstrip(" "))
            if next_indent <= curr_indent:
                mapping[key] = {}
                index += 1
                continue
            if next_line.strip().startswith("- "):
                value, index = parse_list(index + 1, next_indent)
            else:
                value, index = parse_mapping(index + 1, next_indent)
            mapping[key] = value
        return mapping, index

    def parse_list(index: int, indent: int) -> tuple[list[Any], int]:
        items: list[Any] = []
        while index < len(lines):
            line = lines[index]
            curr_indent = len(line) - len(line.lstrip(" "))
            if curr_indent < indent:
                break
            if curr_indent != indent:
                raise ValueError(f"Invalid list indentation in config line: {line}")
            stripped = line.strip()
            if not stripped.startswith("- "):
                break
            content = stripped[2:].strip()
            if content:
                items.append(_parse_scalar(content))
                index += 1
                continue
            value, index = parse_mapping(index + 1, indent + 2)
            items.append(value)
        return items, index

    parsed, final_index = parse_mapping(0, 0)
    if final_index != len(lines):
        raise ValueError("Mini YAML parser did not consume the entire file.")
    return parsed


def _parse_scalar(value: str) -> Any:
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"null", "none"}:
        return None
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    if value.startswith(("'", '"')) and value.endswith(("'", '"')):
        return ast.literal_eval(value)
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value

```
```text
# FILE: src/tc_adv/config/schemas.py
```
```python
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

```
```text
# FILE: src/tc_adv/discriminators/__init__.py
```
```python
"""Discriminator modules for TC-ADV."""

```
```text
# FILE: src/tc_adv/discriminators/ecm.py
```
```python
"""Evolutionary Consistency Module.

The implementation follows Chapter 4.2.2 and Eq. (4-5) to Eq. (4-7):
relative-time encoding + temporal attention over local history + MLP scoring.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from tc_adv.config.schemas import ECMConfig


_BaseModule = nn.Module if hasattr(nn, "Module") else object


def merge_neighbor_histories(
    subject_neighbors: Sequence[str],
    object_neighbors: Sequence[str],
    subject_deltas: Sequence[float],
    object_deltas: Sequence[float],
    history_window: int,
) -> tuple[list[str], list[float]]:
    merged: list[tuple[float, str]] = []
    for entity_id, delta in zip(subject_neighbors, subject_deltas):
        if float(delta) >= 0.0:
            merged.append((float(delta), entity_id))
    for entity_id, delta in zip(object_neighbors, object_deltas):
        if float(delta) >= 0.0:
            merged.append((float(delta), entity_id))
    merged.sort(key=lambda item: item[0])
    trimmed = merged[:history_window]
    return [entity_id for _, entity_id in trimmed], [delta for delta, _ in trimmed]


def sinusoidal_time_encoding(deltas, dim: int):
    if torch is not None and hasattr(deltas, "shape"):
        device = deltas.device
        half = max(dim // 2, 1)
        positions = deltas.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        phase = positions * div_term
        encoding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
        if encoding.size(-1) < dim:
            padding = torch.zeros(*encoding.shape[:-1], dim - encoding.size(-1), device=device)
            encoding = torch.cat([encoding, padding], dim=-1)
        return encoding[..., :dim]
    output = []
    half = max(dim // 2, 1)
    for delta in deltas:
        row: list[float] = []
        for index in range(half):
            scale = math.exp(index * (-math.log(10000.0) / max(half - 1, 1)))
            row.append(math.sin(float(delta) * scale))
            row.append(math.cos(float(delta) * scale))
        if len(row) < dim:
            row.extend([0.0] * (dim - len(row)))
        output.append(row[:dim])
    return output


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _dot(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(lhs, rhs))


class EvolutionaryConsistencyModule(_BaseModule):
    """Temporal attention discriminator for evolution consistency."""

    def __init__(self, embedding_dim: int, config: ECMConfig) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.config = config
        self.hidden_dim = int(config.hidden_dim or embedding_dim)
        if torch is not None:
            self.state_proj = nn.Linear(embedding_dim + config.time_encoding_dim, self.hidden_dim)
            self.query_proj = nn.Linear(embedding_dim, self.hidden_dim)
            self.attn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=config.num_heads,
                batch_first=True,
            )
            self.mlp = nn.Sequential(
                nn.Linear(self.hidden_dim + embedding_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, 1),
            )

    def probability(
        self,
        subject_embed,
        relation_embed,
        object_embed,
        history_entity_embed,
        history_deltas,
        history_mask=None,
    ):
        if torch is not None and hasattr(subject_embed, "shape"):
            return self._probability_torch(
                subject_embed=subject_embed,
                relation_embed=relation_embed,
                object_embed=object_embed,
                history_entity_embed=history_entity_embed,
                history_deltas=history_deltas,
                history_mask=history_mask,
            )
        return self._probability_python(
            subject_embed=subject_embed,
            relation_embed=relation_embed,
            object_embed=object_embed,
            history_entity_embed=history_entity_embed,
            history_deltas=history_deltas,
            history_mask=history_mask,
        )

    def _probability_torch(
        self,
        subject_embed,
        relation_embed,
        object_embed,
        history_entity_embed,
        history_deltas,
        history_mask=None,
    ):
        batch_size = subject_embed.size(0)
        if history_entity_embed.size(1) == 0:
            context = torch.zeros((batch_size, self.hidden_dim), dtype=subject_embed.dtype, device=subject_embed.device)
        else:
            time_enc = sinusoidal_time_encoding(history_deltas, self.config.time_encoding_dim)
            states = self.state_proj(torch.cat([history_entity_embed, time_enc], dim=-1))
            query = self.query_proj(relation_embed).unsqueeze(1)
            key_padding_mask = None
            if history_mask is not None:
                key_padding_mask = ~history_mask.bool()
            context, _ = self.attn(query, states, states, key_padding_mask=key_padding_mask)
            context = context.squeeze(1)
        e_fake = subject_embed + relation_embed - object_embed
        logits = self.mlp(torch.cat([context, e_fake], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)

    def _probability_python(
        self,
        subject_embed,
        relation_embed,
        object_embed,
        history_entity_embed,
        history_deltas,
        history_mask=None,
    ):
        batch: list[float] = []
        for index, subj in enumerate(subject_embed):
            rel = relation_embed[index]
            obj = object_embed[index]
            hist_emb = history_entity_embed[index] if history_entity_embed else []
            hist_delta = history_deltas[index] if history_deltas else []
            mask = history_mask[index] if history_mask else [True] * len(hist_emb)
            supports: list[float] = []
            for row_embed, delta, keep in zip(hist_emb, hist_delta, mask):
                if not keep:
                    continue
                supports.append(_dot(row_embed, rel) * math.exp(-float(delta)))
            context = sum(supports) / len(supports) if supports else 0.0
            fake_vec = [float(a) + float(b) - float(c) for a, b, c in zip(subj, rel, obj)]
            fake_signal = sum(abs(value) for value in fake_vec) / max(len(fake_vec), 1)
            batch.append(_sigmoid(context - fake_signal))
        return batch

```
```text
# FILE: src/tc_adv/discriminators/fusion.py
```
```python
"""Violation fusion helpers for Eq. (4-8)."""

from __future__ import annotations


def fuse_violation_probabilities(p_trm, p_ecm, gamma: float):
    return gamma * p_trm + (1.0 - gamma) * p_ecm


def classify_violation(p_trm: float, p_ecm: float, threshold: float = 0.5) -> str:
    trm_bad = p_trm >= threshold
    ecm_bad = p_ecm >= threshold
    if trm_bad and ecm_bad:
        return "both"
    if trm_bad:
        return "TRM-only"
    if ecm_bad:
        return "ECM-only"
    return "none"

```
```text
# FILE: src/tc_adv/discriminators/trm.py
```
```python
"""Temporal Rationality Module.

The scoring flow follows Chapter 4.2.1 and Eq. (4-1) to Eq. (4-4) from
`核心工作二.pdf`: KDE-based activity estimation -> normalization -> linear map ->
sigmoid -> complement as violation probability.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from types import SimpleNamespace
from typing import Iterable, Sequence

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(Module=object)

from tc_adv.config.schemas import TRMConfig


_BaseModule = nn.Module if hasattr(nn, "Module") else object


def infer_bandwidth(timestamps: Sequence[int], configured: str | float) -> float:
    if configured != "auto":
        return max(float(configured), 1e-6)
    if len(timestamps) < 2:
        return 1.0
    ordered = sorted(timestamps)
    diffs = [max(float(ordered[i + 1] - ordered[i]), 1.0) for i in range(len(ordered) - 1)]
    avg_gap = sum(diffs) / len(diffs)
    spread = statistics.pstdev(ordered) if len(ordered) > 1 else 0.0
    return max(avg_gap, spread / max(math.sqrt(len(ordered)), 1.0), 1.0)


def gaussian_kde_score(
    timestamps: Sequence[int],
    query_timestamp: int,
    bandwidth: float,
    epsilon: float,
) -> float:
    if not timestamps:
        return epsilon
    h = max(float(bandwidth), epsilon)
    coeff = 1.0 / (len(timestamps) * math.sqrt(2.0 * math.pi) * h)
    accum = sum(
        math.exp(-((float(query_timestamp) - float(ts)) ** 2) / (2.0 * h * h))
        for ts in timestamps
    )
    return coeff * accum + epsilon


def normalize_activity_score(score: float, max_score: float, epsilon: float) -> float:
    return float(score) / max(float(max_score), epsilon)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


class TemporalRationalityModule(_BaseModule):
    """KDE-driven lifecycle violation detector."""

    def __init__(self, config: TRMConfig) -> None:
        super().__init__()
        self.config = config
        self.entity_timestamps: dict[str, list[int]] = defaultdict(list)
        self.entity_bandwidths: dict[str, float] = {}
        self.entity_score_max: dict[str, float] = {}
        if torch is not None:
            self.linear = nn.Linear(2, 1)
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.zero_()
        else:
            self.weights = (1.0, 1.0)
            self.bias = 0.0

    def build_index(self, samples: Iterable[object]) -> None:
        timestamps_by_entity: dict[str, list[int]] = defaultdict(list)
        for sample in samples:
            quadruple = sample.quadruple
            timestamps_by_entity[quadruple.subject].append(int(quadruple.timestamp))
            timestamps_by_entity[quadruple.object].append(int(quadruple.timestamp))
        self.entity_timestamps = {
            entity_id: sorted(values)
            for entity_id, values in timestamps_by_entity.items()
        }
        self.entity_bandwidths = {
            entity_id: infer_bandwidth(values, self.config.bandwidth)
            for entity_id, values in self.entity_timestamps.items()
        }
        self.entity_score_max = {}
        for entity_id, values in self.entity_timestamps.items():
            bandwidth = self.entity_bandwidths[entity_id]
            max_score = max(
                gaussian_kde_score(values, ts, bandwidth, self.config.epsilon)
                for ts in values
            )
            self.entity_score_max[entity_id] = max_score

    def raw_activity_score(self, entity_id: str, timestamp: int) -> float:
        timestamps = self.entity_timestamps.get(entity_id, [])
        bandwidth = self.entity_bandwidths.get(entity_id, 1.0)
        return gaussian_kde_score(timestamps, timestamp, bandwidth, self.config.epsilon)

    def normalized_activity_score(self, entity_id: str, timestamp: int) -> float:
        raw = self.raw_activity_score(entity_id, timestamp)
        max_score = self.entity_score_max.get(entity_id, raw or self.config.epsilon)
        return normalize_activity_score(raw, max_score + self.config.epsilon, self.config.epsilon)

    def probability_from_scores(self, subject_scores, object_scores):
        if torch is not None and hasattr(subject_scores, "shape"):
            features = torch.stack([subject_scores, object_scores], dim=-1).float()
            validity = torch.sigmoid(self.linear(features)).squeeze(-1)
            return 1.0 - validity
        if isinstance(subject_scores, Sequence) and not isinstance(subject_scores, (str, bytes)):
            return [
                self.probability_from_scores(sub_score, obj_score)
                for sub_score, obj_score in zip(subject_scores, object_scores)
            ]
        validity = _sigmoid(
            float(subject_scores) * self.weights[0] + float(object_scores) * self.weights[1] + self.bias
        )
        return 1.0 - validity

    def predict(self, subjects: Sequence[str], objects: Sequence[str], timestamps: Sequence[int]):
        subject_scores = [self.normalized_activity_score(subject, timestamp) for subject, timestamp in zip(subjects, timestamps)]
        object_scores = [self.normalized_activity_score(obj, timestamp) for obj, timestamp in zip(objects, timestamps)]
        return {
            "subject_scores": subject_scores,
            "object_scores": object_scores,
            "probabilities": self.probability_from_scores(subject_scores, object_scores),
        }

```
```text
# FILE: src/tc_adv/experiments/__init__.py
```
```python
"""Experiment orchestration."""

```
```text
# FILE: src/tc_adv/experiments/runner.py
```
```python
"""Experiment orchestration and result aggregation."""

from __future__ import annotations

import csv
from pathlib import Path

from tc_adv.config.loader import load_tcadv_config
from tc_adv.training.trainer import TCADVTrainer
from tc_adv.utils.io import write_json


def train_config(config_path: str) -> dict[str, object]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics = trainer.train()
    return {"name": config.name, **metrics, "output_dir": config.output_dir}


def evaluate_config(config_path: str) -> dict[str, object]:
    config = load_tcadv_config(config_path)
    trainer = TCADVTrainer(config=config)
    metrics = trainer.evaluate(split="test", checkpoint_name="best")
    return {"name": config.name, **metrics, "output_dir": config.output_dir}


def run_experiment_suite(config_paths: list[str]) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for config_path in config_paths:
        config = load_tcadv_config(config_path)
        try:
            metrics = train_config(config_path)
            rows.append(metrics)
        except Exception as exc:  # pragma: no cover - suite should continue on missing full datasets
            failures.append({"config": config.name, "error": str(exc)})
            rows.append({"name": config.name, "status": "failed", "error": str(exc)})
    if rows:
        output_dir = Path(load_tcadv_config(config_paths[0]).output_dir)
        _write_csv(output_dir / "suite_metrics.csv", rows)
        write_json(output_dir / "suite_metrics.json", rows)
        write_json(output_dir / "violation_breakdown.json", {"failures": failures, "runs": rows})
    return {"runs": rows, "failures": failures}


def _write_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

```
```text
# FILE: src/tc_adv/report/__init__.py
```
```python
"""Reporting and export helpers."""

```
```text
# FILE: src/tc_adv/report/exporter.py
```
```python
"""Repository code exporter for thesis appendices."""

from __future__ import annotations

from pathlib import Path

from tc_adv.utils.io import write_text


def export_repository_code(output_path: str | Path, repo_root: str | Path | None = None) -> Path:
    root = Path(repo_root or Path.cwd()).resolve()
    ordered_groups = [
        root / "configs",
        root / "src",
        root / "scripts",
        root / "tests",
    ]
    parts: list[str] = []
    for group in ordered_groups:
        if not group.exists():
            continue
        for file_path in sorted(path for path in group.rglob("*") if path.is_file()):
            rel = file_path.relative_to(root)
            parts.append(f"```text\n# FILE: {rel}\n```")
            parts.append(f"```{_language_for(file_path)}\n{file_path.read_text(encoding='utf-8')}\n```")
    destination = Path(output_path)
    write_text(destination, "\n".join(parts) + "\n")
    return destination


def _language_for(path: Path) -> str:
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
        ".txt": "text",
        ".json": "json",
        ".jsonl": "json",
    }.get(path.suffix.lower(), "text")

```
```text
# FILE: src/tc_adv/training/__init__.py
```
```python
"""Training utilities."""

```
```text
# FILE: src/tc_adv/training/backend.py
```
```python
"""Generator backend adapters for TC-ADV."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency
    torch = None

from tc_adv.bridge.lmca_tic import LMCATICBridge
from tc_adv.discriminators.ecm import merge_neighbor_histories
from tc_adv.utils.io import read_json


@dataclass
class GeneratorContext:
    subject_embedding: Any
    relation_embedding: Any
    object_embeddings: Any
    candidate_scores: Any
    subject_trm_score: float
    object_trm_scores: list[float]
    history_embeddings: Any
    history_deltas: Any
    history_mask: Any


class LMCAAdapter:
    """Real bridge-backed adapter on top of the sibling LMCA-TIC trainer."""

    supports_gradient = True

    def __init__(self, bridge: LMCATICBridge, base_config, smoke_mode: bool = False) -> None:
        self.bridge = bridge
        self.base_config = base_config
        self.smoke_mode = smoke_mode
        self.bridge.create_preprocessor(base_config).run()
        self.base_trainer = bridge.create_trainer(base_config, smoke_mode=smoke_mode)
        self.train_dataset = self.base_trainer.train_dataset
        self.valid_dataset = self.base_trainer.valid_dataset
        self.test_dataset = self.base_trainer.test_dataset
        self.filtered_targets = self.base_trainer.filtered_targets
        self.entities_payload = self.base_trainer.entities
        self.entities = list(self.entities_payload.keys())
        self.entity_to_idx = self.base_trainer.entity_to_idx
        self.relation_to_idx = self.base_trainer.relation_to_idx
        self.entity_prompts = self.base_trainer.entity_prompts
        self.entity_neighbor_cache = self.base_trainer.entity_neighbor_cache
        self.entity_delta_cache = self.base_trainer.entity_delta_cache
        self.model = self.base_trainer.model
        self.optimizer = self.base_trainer.optimizer
        self.loss_fn = self.base_trainer.loss_fn
        self.embedding_dim = int(base_config.model.embedding_dim)
        self.entity_types = {
            entity_id: self.base_trainer._entity_types(entity_id)
            for entity_id in self.entities
        }

    def score_candidates(self, sample) -> dict[str, float]:
        return self.base_trainer._current_candidate_scores(sample)

    def topk_candidates(self, sample, k: int, exclude_gold: bool = True) -> list[str]:
        scores = self.score_candidates(sample)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        gold = sample.quadruple.object
        output = []
        for entity_id, _ in ranked:
            if exclude_gold and entity_id == gold:
                continue
            output.append(entity_id)
            if len(output) >= k:
                break
        return output

    def semantic_loss(self, sample, fake_candidates: list[str]):
        batch = self.base_trainer._build_batch([sample], forced_negative_candidates=fake_candidates)
        outputs = self.model(batch)
        positive_scores = outputs["positive_scores"]
        positive_labels = torch.ones_like(positive_scores)
        loss = self.loss_fn(positive_scores, positive_labels)
        negative_scores = outputs["negative_scores"]
        if negative_scores is not None and batch["negative_mask"].any():
            masked = negative_scores[batch["negative_mask"]]
            negative_labels = torch.zeros_like(masked)
            loss = loss + self.loss_fn(masked, negative_labels)
        return loss

    def real_score(self, sample):
        context = self.context_for_candidates(sample, [sample.quadruple.object], requires_grad=True)
        return context.candidate_scores[0]

    def context_for_candidates(self, sample, candidate_ids: Sequence[str], requires_grad: bool = True) -> GeneratorContext:
        context_manager = _nullcontext() if requires_grad else torch.no_grad()
        with context_manager:
            relation_history = torch.tensor([sample.relation_history], dtype=torch.float32)
            subject_ids = torch.tensor([self.entity_to_idx[sample.quadruple.subject]], dtype=torch.long)
            relation_ids = torch.tensor([self.relation_to_idx[sample.quadruple.relation]], dtype=torch.long)
            subject_neighbor_ids, subject_neighbor_deltas = self.base_trainer._pad_neighbors(
                [sample.subject_neighbors],
                [sample.extra.get("subject_neighbor_deltas", [])],
            )
            subject_embed, _ = self.model.encode_entities(
                prompts=[sample.subject_prompt],
                relation_histories=relation_history,
                entity_ids=subject_ids,
                neighbor_ids=subject_neighbor_ids,
                neighbor_deltas=subject_neighbor_deltas,
            )
            subject_embed = subject_embed.repeat(len(candidate_ids), 1)
            object_ids = torch.tensor([self.entity_to_idx[candidate] for candidate in candidate_ids], dtype=torch.long)
            repeated_histories = relation_history.repeat(len(candidate_ids), 1)
            object_neighbor_ids, object_neighbor_deltas = self.base_trainer._pad_neighbors(
                [self.entity_neighbor_cache.get(candidate, []) for candidate in candidate_ids],
                [self.entity_delta_cache.get(candidate, []) for candidate in candidate_ids],
            )
            object_embed, _ = self.model.encode_entities(
                prompts=[self.entity_prompts[candidate] for candidate in candidate_ids],
                relation_histories=repeated_histories,
                entity_ids=object_ids,
                neighbor_ids=object_neighbor_ids,
                neighbor_deltas=object_neighbor_deltas,
            )
            repeated_relation_ids = relation_ids.repeat(len(candidate_ids))
            relation_embed = self.model.scorer.relation_embedding(repeated_relation_ids)
            candidate_scores = self.model.scorer(subject_embed, repeated_relation_ids, object_embed)
            history_ids, history_deltas, history_mask = self._history_tensors(sample, candidate_ids)
            history_embeddings = self.model.graph_encoder.embedding(history_ids)
        subject_trm_score = 1.0
        return GeneratorContext(
            subject_embedding=subject_embed,
            relation_embedding=relation_embed,
            object_embeddings=object_embed,
            candidate_scores=candidate_scores,
            subject_trm_score=subject_trm_score,
            object_trm_scores=[],
            history_embeddings=history_embeddings,
            history_deltas=history_deltas,
            history_mask=history_mask,
        )

    def _history_tensors(self, sample, candidate_ids: Sequence[str]):
        history_lists: list[list[str]] = []
        history_deltas: list[list[float]] = []
        for candidate in candidate_ids:
            neighbors, deltas = merge_neighbor_histories(
                sample.subject_neighbors,
                self.entity_neighbor_cache.get(candidate, []),
                sample.extra.get("subject_neighbor_deltas", []),
                self.entity_delta_cache.get(candidate, []),
                self.base_config.model.tgn_neighbor_size,
            )
            history_lists.append(neighbors)
            history_deltas.append(deltas)
        max_neighbors = max((len(items) for items in history_lists), default=0)
        if max_neighbors == 0:
            return (
                torch.zeros((len(candidate_ids), 0), dtype=torch.long),
                torch.zeros((len(candidate_ids), 0), dtype=torch.float32),
                torch.zeros((len(candidate_ids), 0), dtype=torch.bool),
            )
        padded_ids = []
        padded_deltas = []
        padded_mask = []
        for neighbors, deltas in zip(history_lists, history_deltas):
            ids = [self.entity_to_idx.get(entity_id, 0) for entity_id in neighbors]
            ids = ids + [0] * (max_neighbors - len(ids))
            deltas = list(deltas) + [0.0] * (max_neighbors - len(deltas))
            mask = [True] * len(neighbors) + [False] * (max_neighbors - len(neighbors))
            padded_ids.append(ids)
            padded_deltas.append(deltas)
            padded_mask.append(mask)
        return (
            torch.tensor(padded_ids, dtype=torch.long),
            torch.tensor(padded_deltas, dtype=torch.float32),
            torch.tensor(padded_mask, dtype=torch.bool),
        )

    def save_generator_checkpoint(self, name: str) -> None:
        self.base_trainer._save_checkpoint(name)

    def load_generator_checkpoint(self, name: str) -> None:
        self.base_trainer._load_checkpoint(name)


class ToyGeneratorAdapter:
    """Pure-Python fallback backend used when torch-based LMCA training is unavailable."""

    supports_gradient = False

    def __init__(self, bridge: LMCATICBridge, base_config) -> None:
        self.bridge = bridge
        self.base_config = base_config
        self.bridge.create_preprocessor(base_config).run()
        self.train_dataset = bridge.create_dataset(base_config.processed_dir, "train")
        self.valid_dataset = bridge.create_dataset(base_config.processed_dir, "valid")
        self.test_dataset = bridge.create_dataset(base_config.processed_dir, "test")
        self.entities_payload = read_json(Path(base_config.processed_dir) / "entities.json")
        self.filtered_targets = read_json(Path(base_config.processed_dir) / "filtered_targets.json")
        relation_manifest = read_json(Path(base_config.processed_dir) / "relations.json")
        self.entities = list(self.entities_payload.keys())
        self.embedding_dim = 16
        self.entity_to_idx = {entity_id: idx for idx, entity_id in enumerate(self.entities)}
        self.relation_to_idx = {
            relation: idx
            for idx, relation in enumerate(relation_manifest["relations"] + relation_manifest["inverse_relations"])
        }
        self.entity_neighbor_cache, self.entity_delta_cache = self._build_entity_context_cache(self.train_dataset.samples)
        self.bias = {entity_id: 0.0 for entity_id in self.entities}

    def _build_entity_context_cache(self, samples):
        latest_neighbors: dict[str, tuple[int, list[str], list[float]]] = {}
        for sample in samples:
            timestamp = sample.quadruple.timestamp
            for entity_id, neighbors, deltas in (
                (sample.quadruple.subject, sample.subject_neighbors, sample.extra.get("subject_neighbor_deltas", [])),
                (sample.quadruple.object, sample.object_neighbors, sample.extra.get("object_neighbor_deltas", [])),
            ):
                previous = latest_neighbors.get(entity_id)
                if previous is None or timestamp >= previous[0]:
                    latest_neighbors[entity_id] = (timestamp, list(neighbors), list(deltas))
        neighbor_cache = {}
        delta_cache = {}
        for entity_id, (_, neighbors, deltas) in latest_neighbors.items():
            neighbor_cache[entity_id] = neighbors
            delta_cache[entity_id] = deltas
        return neighbor_cache, delta_cache

    def score_candidates(self, sample) -> dict[str, float]:
        subject_neighbors = set(sample.subject_neighbors)
        score_base = sum(sample.relation_history) / max(len(sample.relation_history), 1)
        output: dict[str, float] = {}
        for entity_id in self.entities:
            overlap = len(subject_neighbors & set(self.entity_neighbor_cache.get(entity_id, [])))
            type_bonus = 0.3 if entity_id == sample.quadruple.object else 0.0
            recency_bonus = 0.0
            deltas = self.entity_delta_cache.get(entity_id, [])
            if deltas:
                recency_bonus = 1.0 / (1.0 + min(deltas))
            output[entity_id] = score_base + 0.1 * overlap + 0.2 * recency_bonus + type_bonus + self.bias[entity_id]
        return output

    def topk_candidates(self, sample, k: int, exclude_gold: bool = True) -> list[str]:
        ranked = sorted(self.score_candidates(sample).items(), key=lambda item: item[1], reverse=True)
        gold = sample.quadruple.object
        result = []
        for entity_id, _ in ranked:
            if exclude_gold and entity_id == gold:
                continue
            result.append(entity_id)
            if len(result) >= k:
                break
        return result

    def semantic_loss(self, sample, fake_candidates: list[str]) -> float:
        scores = self.score_candidates(sample)
        real = scores[sample.quadruple.object]
        fake = max(scores[candidate] for candidate in fake_candidates) if fake_candidates else real
        return max(0.0, 1.0 + fake - real)

    def real_score(self, sample) -> float:
        return self.score_candidates(sample)[sample.quadruple.object]

    def context_for_candidates(self, sample, candidate_ids: Sequence[str], requires_grad: bool = True) -> GeneratorContext:
        subject_vector = _stable_vector(sample.quadruple.subject, self.embedding_dim)
        relation_vector = _stable_vector(sample.quadruple.relation, self.embedding_dim)
        score_map = self.score_candidates(sample)
        object_vectors = [_stable_vector(candidate, self.embedding_dim) for candidate in candidate_ids]
        history_embeddings = []
        history_deltas = []
        history_mask = []
        object_scores = []
        for candidate in candidate_ids:
            neighbors, deltas = merge_neighbor_histories(
                sample.subject_neighbors,
                self.entity_neighbor_cache.get(candidate, []),
                sample.extra.get("subject_neighbor_deltas", []),
                self.entity_delta_cache.get(candidate, []),
                self.base_config.model.tgn_neighbor_size,
            )
            history_embeddings.append([_stable_vector(entity_id, self.embedding_dim) for entity_id in neighbors])
            history_deltas.append(list(deltas))
            history_mask.append([True] * len(neighbors))
            object_scores.append(score_map[candidate])
        return GeneratorContext(
            subject_embedding=[subject_vector for _ in candidate_ids],
            relation_embedding=[relation_vector for _ in candidate_ids],
            object_embeddings=object_vectors,
            candidate_scores=[score_map[candidate] for candidate in candidate_ids],
            subject_trm_score=1.0,
            object_trm_scores=object_scores,
            history_embeddings=history_embeddings,
            history_deltas=history_deltas,
            history_mask=history_mask,
        )

    def save_generator_checkpoint(self, name: str) -> None:
        target = Path(self.base_config.checkpoint_dir) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(self.bias), encoding="utf-8")

    def load_generator_checkpoint(self, name: str) -> None:
        return None

    def apply_adversarial_feedback(self, sample, fake_candidate: str, penalty: float) -> None:
        self.bias[fake_candidate] -= 0.05 * float(penalty)
        self.bias[sample.quadruple.object] += 0.02 * float(penalty)


def build_generator_backend(bridge: LMCATICBridge, base_config, smoke_mode: bool = False):
    if torch is None:
        return ToyGeneratorAdapter(bridge=bridge, base_config=base_config)
    try:
        return LMCAAdapter(bridge=bridge, base_config=base_config, smoke_mode=smoke_mode)
    except Exception:
        return ToyGeneratorAdapter(bridge=bridge, base_config=base_config)


def _stable_vector(key: str, dim: int) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    vector = []
    for index in range(dim):
        byte = digest[index % len(digest)]
        vector.append((byte / 255.0) * 2.0 - 1.0)
    return vector


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False

```
```text
# FILE: src/tc_adv/training/objectives.py
```
```python
"""Objective functions and schedules for TC-ADV."""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    import torch
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    F = None


def dynamic_margin(
    probability,
    alpha: float,
    beta: float,
    use_static_margin: bool = False,
    static_margin: float = 1.0,
):
    if use_static_margin:
        if torch is not None and hasattr(probability, "shape"):
            return torch.full_like(probability, float(static_margin))
        return float(static_margin)
    if torch is not None and hasattr(probability, "shape"):
        return alpha * torch.exp(beta * probability) + alpha
    return alpha * math.exp(beta * float(probability)) + alpha


def relu_margin_loss(real_score, fake_score, violation_probability, alpha: float, beta: float, use_static_margin: bool = False, static_margin: float = 1.0):
    margin = dynamic_margin(
        probability=violation_probability,
        alpha=alpha,
        beta=beta,
        use_static_margin=use_static_margin,
        static_margin=static_margin,
    )
    if torch is not None and hasattr(real_score, "shape"):
        return torch.relu(margin + fake_score - real_score)
    return max(0.0, float(margin) + float(fake_score) - float(real_score))


def anneal_temperature(current: float, anneal_rate: float, min_temp: float) -> float:
    return max(float(current) * float(anneal_rate), float(min_temp))


def gumbel_softmax(logits, temperature: float):
    if torch is not None and hasattr(logits, "shape"):
        return F.gumbel_softmax(logits, tau=float(temperature), hard=False, dim=-1)
    exps = [math.exp(float(value) / max(float(temperature), 1e-6)) for value in logits]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


@dataclass
class StepRatioScheduler:
    g_steps: int
    d_steps: int

    def cycle(self) -> list[str]:
        return ["G"] * self.g_steps + ["D"] * self.d_steps

    def phase_at(self, step_index: int) -> str:
        phases = self.cycle()
        return phases[step_index % len(phases)]

```
```text
# FILE: src/tc_adv/training/trainer.py
```
```python
"""Training orchestration for TC-ADV."""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
except ImportError:  # pragma: no cover - optional dependency
    torch = None
    nn = SimpleNamespace(BCELoss=object)
    AdamW = None

from tc_adv.bridge.lmca_tic import LMCATICBridge
from tc_adv.config.loader import dump_tcadv_config
from tc_adv.config.schemas import TCADVExperimentConfig
from tc_adv.discriminators.ecm import EvolutionaryConsistencyModule
from tc_adv.discriminators.fusion import classify_violation, fuse_violation_probabilities
from tc_adv.discriminators.trm import TemporalRationalityModule
from tc_adv.training.backend import GeneratorContext, build_generator_backend
from tc_adv.training.objectives import StepRatioScheduler, anneal_temperature, gumbel_softmax, relu_margin_loss
from tc_adv.utils.io import ensure_dir, write_json, write_jsonl
from tc_adv.utils.logging import build_logger, capture_manifest, write_manifest


class TCADVTrainer:
    """Adversarial trainer for the second core contribution."""

    def __init__(self, config: TCADVExperimentConfig, bridge: LMCATICBridge | None = None) -> None:
        self.config = config
        self.bridge = bridge or LMCATICBridge()
        self.base_config = self.bridge.load_experiment_config(config.lmca_experiment_config)
        self._override_base_config()
        self.output_dir = ensure_dir(config.output_dir)
        self.log_dir = ensure_dir(config.log_dir)
        self.checkpoint_dir = ensure_dir(config.checkpoint_dir)
        self.logger = build_logger(self.log_dir)
        self.generator = build_generator_backend(
            bridge=self.bridge,
            base_config=self.base_config,
            smoke_mode=bool(self.base_config.metadata.get("smoke_mode", False)),
        )
        history_window = self.config.tc_adv.ecm.history_window or self.base_config.model.tgn_neighbor_size
        self.config.tc_adv.ecm.history_window = history_window
        hidden_dim = self.config.tc_adv.ecm.hidden_dim or self.base_config.model.embedding_dim
        self.config.tc_adv.ecm.hidden_dim = hidden_dim
        self.trm = TemporalRationalityModule(self.config.tc_adv.trm)
        self.trm.build_index(self.generator.train_dataset.samples)
        self.ecm = EvolutionaryConsistencyModule(
            embedding_dim=int(self.generator.embedding_dim),
            config=self.config.tc_adv.ecm,
        )
        self.scheduler = StepRatioScheduler(
            g_steps=self.config.tc_adv.trainer.g_steps,
            d_steps=self.config.tc_adv.trainer.d_steps,
        )
        self.temperature = float(self.config.tc_adv.gumbel.start_temp)
        self.discriminator_loss_fn = (
            nn.BCELoss() if torch is not None and isinstance(self.generator.supports_gradient, bool) else None
        )
        self.discriminator_optimizer = None
        if torch is not None and self.generator.supports_gradient and AdamW is not None:
            params = list(self.trm.parameters()) + list(self.ecm.parameters())
            self.discriminator_optimizer = AdamW(params, lr=self.config.tc_adv.trainer.discriminator_lr)

    def train(self) -> dict[str, float]:
        manifest = capture_manifest(
            extra={
                "config": dump_tcadv_config(self.config),
                "lmca_config": self.bridge.dump_experiment_config(self.base_config),
                "backend": type(self.generator).__name__,
            }
        )
        write_manifest(self.output_dir / "run_manifest.json", manifest)
        best_mrr = -1.0
        patience = 0
        train_history: list[dict[str, float | int | str]] = []
        violation_history: list[dict[str, float]] = []

        for epoch in range(1, self.config.tc_adv.trainer.max_epochs + 1):
            start = time.perf_counter()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            steps = 0
            phase_trace: list[str] = []
            for sample in self.generator.train_dataset.samples:
                for phase in self.scheduler.cycle():
                    phase_trace.append(phase)
                    if phase == "G":
                        epoch_g_loss += float(self._generator_step(sample))
                    else:
                        epoch_d_loss += float(self._discriminator_step(sample))
                    steps += 1
            valid_metrics = self.evaluate(split="valid")
            step_time = time.perf_counter() - start
            record = {
                "epoch": epoch,
                "generator_loss": epoch_g_loss / max(steps, 1),
                "discriminator_loss": epoch_d_loss / max(steps, 1),
                "valid_mrr": float(valid_metrics["MRR"]),
                "valid_tvr": float(valid_metrics["TVR"]),
                "temperature": float(self.temperature),
                "step_time_sec": float(step_time),
                "phase_trace_length": len(phase_trace),
            }
            train_history.append(record)
            violation_history.append(
                {
                    "epoch": float(epoch),
                    "tvr": float(valid_metrics["TVR"]),
                    "trm_violation_rate": float(valid_metrics["trm_violation_rate"]),
                    "ecm_violation_rate": float(valid_metrics["ecm_violation_rate"]),
                    "fused_violation_rate": float(valid_metrics["fused_violation_rate"]),
                }
            )
            self.logger.info(
                "epoch=%s generator_loss=%.6f discriminator_loss=%.6f valid_mrr=%.4f valid_tvr=%.4f temp=%.4f",
                epoch,
                record["generator_loss"],
                record["discriminator_loss"],
                record["valid_mrr"],
                record["valid_tvr"],
                record["temperature"],
            )
            if record["valid_mrr"] > best_mrr:
                best_mrr = float(record["valid_mrr"])
                patience = 0
                self._save_checkpoint_pair("best")
            else:
                patience += 1
                if patience >= self.config.tc_adv.trainer.early_stopping_patience:
                    self.logger.info("early stopping triggered at epoch %s", epoch)
                    break
            self.temperature = anneal_temperature(
                self.temperature,
                self.config.tc_adv.gumbel.anneal_rate,
                self.config.tc_adv.gumbel.min_temp,
            )

        write_jsonl(self.output_dir / "train_history.jsonl", train_history)
        write_jsonl(self.output_dir / "violation_history.jsonl", violation_history)
        return self.evaluate(split="test", checkpoint_name="best")

    def evaluate(self, split: str = "test", checkpoint_name: str | None = None) -> dict[str, float | dict[str, int]]:
        if checkpoint_name:
            self._load_checkpoint_pair(checkpoint_name)
        dataset = {
            "train": self.generator.train_dataset,
            "valid": self.generator.valid_dataset,
            "test": self.generator.test_dataset,
        }[split]
        evaluator = self.bridge.create_filtered_evaluator(self.generator.filtered_targets)
        predictions: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        breakdown = {"TRM-only": 0, "ECM-only": 0, "both": 0, "none": 0}

        for sample in dataset.samples:
            scores = self.generator.score_candidates(sample)
            predictions.append(
                {
                    "subject": sample.quadruple.subject,
                    "relation": sample.quadruple.relation,
                    "timestamp": sample.quadruple.timestamp,
                    "gold": sample.quadruple.object,
                    "scores": scores,
                }
            )
            top1 = max(scores.items(), key=lambda item: item[1])[0]
            fact_prob = self._violation_probabilities(sample=sample, candidate_ids=[top1], requires_grad=False)
            label = classify_violation(float(fact_prob["p_trm"][0]), float(fact_prob["p_ecm"][0]))
            breakdown[label] += 1
            diagnostics.append(
                {
                    "subject": sample.quadruple.subject,
                    "relation": sample.quadruple.relation,
                    "timestamp": sample.quadruple.timestamp,
                    "gold": sample.quadruple.object,
                    "top1_prediction": top1,
                    "p_trm": float(fact_prob["p_trm"][0]),
                    "p_ecm": float(fact_prob["p_ecm"][0]),
                    "p_fake": float(fact_prob["p_fake"][0]),
                    "classification": label,
                }
            )

        metrics = evaluator.evaluate(predictions).to_dict()
        total = max(len(diagnostics), 1)
        metrics.update(
            {
                "TVR": sum(1.0 for row in diagnostics if row["p_fake"] >= 0.5) / total,
                "trm_violation_rate": sum(1.0 for row in diagnostics if row["p_trm"] >= 0.5) / total,
                "ecm_violation_rate": sum(1.0 for row in diagnostics if row["p_ecm"] >= 0.5) / total,
                "fused_violation_rate": sum(1.0 for row in diagnostics if row["p_fake"] >= 0.5) / total,
                "violation_breakdown": breakdown,
            }
        )
        write_json(self.output_dir / f"{split}_metrics.json", metrics)
        write_jsonl(self.output_dir / f"{split}_predictions.jsonl", predictions)
        write_json(self.output_dir / f"{split}_diagnostics.json", diagnostics)
        return metrics

    def _override_base_config(self) -> None:
        self.base_config.learning_rate = self.config.tc_adv.trainer.generator_lr
        self.base_config.num_epochs = self.config.tc_adv.trainer.max_epochs
        self.base_config.early_stopping_patience = self.config.tc_adv.trainer.early_stopping_patience
        self.base_config.processed_dir = str((Path.cwd() / "data" / "processed" / self.config.name).resolve())
        self.base_config.output_dir = str((Path(self.config.output_dir) / "generator").resolve())
        self.base_config.log_dir = str((Path(self.config.log_dir) / "generator").resolve())
        self.base_config.checkpoint_dir = str((Path(self.config.checkpoint_dir) / "generator").resolve())

    def _generator_step(self, sample) -> float:
        fake_candidates = self.generator.topk_candidates(
            sample,
            k=self.config.tc_adv.trainer.topk_fake_candidates,
            exclude_gold=True,
        )
        if not fake_candidates:
            return 0.0
        if torch is None or not self.generator.supports_gradient:
            return self._generator_step_fallback(sample, fake_candidates)
        self._freeze_discriminator()
        self._unfreeze_generator()
        context = self.generator.context_for_candidates(sample, fake_candidates, requires_grad=True)
        real_score = self.generator.real_score(sample)
        weights = gumbel_softmax(context.candidate_scores.unsqueeze(0), self.temperature).squeeze(0)
        fake_score = (weights * context.candidate_scores).sum()
        weighted_object = torch.matmul(weights.unsqueeze(0), context.object_embeddings).squeeze(0).unsqueeze(0)
        weighted_relation = context.relation_embedding[:1]
        weighted_subject = context.subject_embedding[:1]
        weighted_history = torch.einsum("k,knd->nd", weights, context.history_embeddings).unsqueeze(0)
        weighted_deltas = torch.einsum("k,kn->n", weights, context.history_deltas).unsqueeze(0)
        history_mask = context.history_mask.any(dim=0).unsqueeze(0)
        subject_score = torch.tensor(
            [self.trm.normalized_activity_score(sample.quadruple.subject, sample.quadruple.timestamp)],
            dtype=torch.float32,
        )
        candidate_scores = torch.tensor(
            [self.trm.normalized_activity_score(candidate, sample.quadruple.timestamp) for candidate in fake_candidates],
            dtype=torch.float32,
        )
        object_score = (weights * candidate_scores).sum().unsqueeze(0)
        p_trm = self.trm.probability_from_scores(subject_score, object_score)
        p_ecm = self.ecm.probability(
            subject_embed=weighted_subject,
            relation_embed=weighted_relation,
            object_embed=weighted_object,
            history_entity_embed=weighted_history,
            history_deltas=weighted_deltas,
            history_mask=history_mask,
        )
        p_fake = fuse_violation_probabilities(p_trm, p_ecm, self.config.tc_adv.fusion.gamma)
        adversarial_loss = relu_margin_loss(
            real_score=real_score.unsqueeze(0),
            fake_score=fake_score.unsqueeze(0),
            violation_probability=p_fake,
            alpha=self.config.tc_adv.loss.alpha,
            beta=self.config.tc_adv.loss.beta,
            use_static_margin=self.config.tc_adv.loss.use_static_margin,
            static_margin=self.config.tc_adv.loss.static_margin,
        ).mean()
        semantic_loss = self.generator.semantic_loss(sample, fake_candidates)
        total_loss = (
            self.config.tc_adv.trainer.generator_loss_weight * semantic_loss
            + self.config.tc_adv.trainer.adversarial_loss_weight * adversarial_loss
        )
        self.generator.optimizer.zero_grad()
        total_loss.backward()
        self.generator.optimizer.step()
        return float(total_loss.item())

    def _generator_step_fallback(self, sample, fake_candidates: list[str]) -> float:
        context = self.generator.context_for_candidates(sample, fake_candidates, requires_grad=False)
        weights = gumbel_softmax(context.candidate_scores, self.temperature)
        score_fake = sum(weight * score for weight, score in zip(weights, context.candidate_scores))
        score_real = float(self.generator.real_score(sample))
        subject_score = self.trm.normalized_activity_score(sample.quadruple.subject, sample.quadruple.timestamp)
        object_score = sum(
            weight * self.trm.normalized_activity_score(candidate, sample.quadruple.timestamp)
            for weight, candidate in zip(weights, fake_candidates)
        )
        p_trm = float(self.trm.probability_from_scores(subject_score, object_score))
        object_index = max(range(len(fake_candidates)), key=lambda index: weights[index])
        discrete_context = self.generator.context_for_candidates(sample, [fake_candidates[object_index]], requires_grad=False)
        p_ecm = float(
            self.ecm.probability(
                subject_embed=discrete_context.subject_embedding,
                relation_embed=discrete_context.relation_embedding,
                object_embed=discrete_context.object_embeddings,
                history_entity_embed=discrete_context.history_embeddings,
                history_deltas=discrete_context.history_deltas,
                history_mask=discrete_context.history_mask,
            )[0]
        )
        p_fake = float(fuse_violation_probabilities(p_trm, p_ecm, self.config.tc_adv.fusion.gamma))
        adversarial_loss = float(
            relu_margin_loss(
                real_score=score_real,
                fake_score=score_fake,
                violation_probability=p_fake,
                alpha=self.config.tc_adv.loss.alpha,
                beta=self.config.tc_adv.loss.beta,
                use_static_margin=self.config.tc_adv.loss.use_static_margin,
                static_margin=self.config.tc_adv.loss.static_margin,
            )
        )
        semantic_loss = float(self.generator.semantic_loss(sample, fake_candidates))
        self.generator.apply_adversarial_feedback(sample, fake_candidates[object_index], adversarial_loss)
        return semantic_loss + adversarial_loss

    def _discriminator_step(self, sample) -> float:
        fake_candidates = self.generator.topk_candidates(
            sample,
            k=self.config.tc_adv.trainer.topk_fake_candidates,
            exclude_gold=True,
        )
        if torch is None or not self.generator.supports_gradient or self.discriminator_optimizer is None:
            return float(self._discriminator_step_fallback(sample, fake_candidates))
        self._freeze_generator()
        self._unfreeze_discriminator()
        self.discriminator_optimizer.zero_grad()
        positive = self._violation_probabilities(sample=sample, candidate_ids=[sample.quadruple.object], requires_grad=False)
        negative = self._violation_probabilities(sample=sample, candidate_ids=fake_candidates, requires_grad=False)
        zero_target = torch.zeros_like(positive["p_fake"])
        one_target = torch.ones_like(negative["p_fake"])
        loss = (
            self.discriminator_loss_fn(positive["p_trm"], zero_target)
            + self.discriminator_loss_fn(positive["p_ecm"], zero_target)
            + self.discriminator_loss_fn(positive["p_fake"], zero_target)
        )
        if fake_candidates:
            loss = loss + (
                self.discriminator_loss_fn(negative["p_trm"], one_target)
                + self.discriminator_loss_fn(negative["p_ecm"], one_target)
                + self.discriminator_loss_fn(negative["p_fake"], one_target)
            )
        loss.backward()
        self.discriminator_optimizer.step()
        return float(loss.item())

    def _discriminator_step_fallback(self, sample, fake_candidates: list[str]) -> float:
        positive = self._violation_probabilities(sample=sample, candidate_ids=[sample.quadruple.object], requires_grad=False)
        loss = (
            float(positive["p_trm"][0]) ** 2
            + float(positive["p_ecm"][0]) ** 2
            + float(positive["p_fake"][0]) ** 2
        )
        if fake_candidates:
            negative = self._violation_probabilities(sample=sample, candidate_ids=fake_candidates, requires_grad=False)
            loss += sum((1.0 - float(value)) ** 2 for value in negative["p_trm"])
            loss += sum((1.0 - float(value)) ** 2 for value in negative["p_ecm"])
            loss += sum((1.0 - float(value)) ** 2 for value in negative["p_fake"])
        return loss

    def _violation_probabilities(self, sample, candidate_ids: list[str], requires_grad: bool):
        if not candidate_ids:
            if torch is not None and self.generator.supports_gradient:
                empty = torch.zeros((0,), dtype=torch.float32)
                return {"p_trm": empty, "p_ecm": empty, "p_fake": empty}
            return {"p_trm": [], "p_ecm": [], "p_fake": []}
        context = self.generator.context_for_candidates(sample, candidate_ids, requires_grad=requires_grad)
        subject_scores = [
            self.trm.normalized_activity_score(sample.quadruple.subject, sample.quadruple.timestamp)
            for _ in candidate_ids
        ]
        object_scores = [
            self.trm.normalized_activity_score(candidate, sample.quadruple.timestamp)
            for candidate in candidate_ids
        ]
        if torch is not None and self.generator.supports_gradient:
            subject_tensor = torch.tensor(subject_scores, dtype=torch.float32)
            object_tensor = torch.tensor(object_scores, dtype=torch.float32)
            if self._module_enabled("trm"):
                p_trm = self.trm.probability_from_scores(subject_tensor, object_tensor)
            else:
                p_trm = torch.zeros_like(subject_tensor)
            if self._module_enabled("ecm"):
                p_ecm = self.ecm.probability(
                    subject_embed=context.subject_embedding,
                    relation_embed=context.relation_embedding,
                    object_embed=context.object_embeddings,
                    history_entity_embed=context.history_embeddings,
                    history_deltas=context.history_deltas,
                    history_mask=context.history_mask,
                )
            else:
                p_ecm = torch.zeros_like(subject_tensor)
            p_fake = fuse_violation_probabilities(p_trm, p_ecm, self.config.tc_adv.fusion.gamma)
            return {"p_trm": p_trm, "p_ecm": p_ecm, "p_fake": p_fake}
        if self._module_enabled("trm"):
            p_trm = self.trm.probability_from_scores(subject_scores, object_scores)
        else:
            p_trm = [0.0 for _ in candidate_ids]
        if self._module_enabled("ecm"):
            p_ecm = self.ecm.probability(
                subject_embed=context.subject_embedding,
                relation_embed=context.relation_embedding,
                object_embed=context.object_embeddings,
                history_entity_embed=context.history_embeddings,
                history_deltas=context.history_deltas,
                history_mask=context.history_mask,
            )
        else:
            p_ecm = [0.0 for _ in candidate_ids]
        p_fake = [
            float(fuse_violation_probabilities(trm_value, ecm_value, self.config.tc_adv.fusion.gamma))
            for trm_value, ecm_value in zip(p_trm, p_ecm)
        ]
        return {"p_trm": p_trm, "p_ecm": p_ecm, "p_fake": p_fake}

    def _module_enabled(self, name: str) -> bool:
        disable_key = f"disable_{name}"
        return not bool(self.config.metadata.get(disable_key, False))

    def _save_checkpoint_pair(self, stem: str) -> None:
        self.generator.save_generator_checkpoint(f"{stem}_generator.pt")
        discriminator_path = self.checkpoint_dir / f"{stem}_discriminator.pt"
        if torch is not None and self.generator.supports_gradient:
            torch.save(
                {
                    "trm": self.trm.state_dict(),
                    "ecm": self.ecm.state_dict(),
                    "temperature": self.temperature,
                },
                discriminator_path,
            )
        else:
            write_json(
                discriminator_path.with_suffix(".json"),
                {"backend": type(self.generator).__name__, "temperature": self.temperature},
            )

    def _load_checkpoint_pair(self, stem: str) -> None:
        self.generator.load_generator_checkpoint(f"{stem}_generator.pt")
        if torch is None or not self.generator.supports_gradient:
            return
        discriminator_path = self.checkpoint_dir / f"{stem}_discriminator.pt"
        if discriminator_path.exists():
            state = torch.load(discriminator_path, map_location="cpu")
            self.trm.load_state_dict(state["trm"])
            self.ecm.load_state_dict(state["ecm"])
            self.temperature = float(state.get("temperature", self.temperature))

    def _freeze_generator(self) -> None:
        if torch is None or not self.generator.supports_gradient:
            return
        for parameter in self.generator.model.parameters():
            parameter.requires_grad = False

    def _unfreeze_generator(self) -> None:
        if torch is None or not self.generator.supports_gradient:
            return
        for parameter in self.generator.model.parameters():
            parameter.requires_grad = True

    def _freeze_discriminator(self) -> None:
        if torch is None or not self.generator.supports_gradient:
            return
        for parameter in self.trm.parameters():
            parameter.requires_grad = False
        for parameter in self.ecm.parameters():
            parameter.requires_grad = False

    def _unfreeze_discriminator(self) -> None:
        if torch is None or not self.generator.supports_gradient:
            return
        for parameter in self.trm.parameters():
            parameter.requires_grad = True
        for parameter in self.ecm.parameters():
            parameter.requires_grad = True

```
```text
# FILE: src/tc_adv/utils/__init__.py
```
```python
"""Utility helpers."""

```
```text
# FILE: src/tc_adv/utils/deps.py
```
```python
"""Optional dependency guards."""

from __future__ import annotations


class MissingDependencyError(RuntimeError):
    """Raised when an optional runtime dependency is unavailable."""


def require_dependency(module: object | None, package_name: str) -> None:
    if module is None:
        raise MissingDependencyError(
            f"Missing optional dependency '{package_name}'. "
            f"Install the package or switch to the smoke fallback path."
        )

```
```text
# FILE: src/tc_adv/utils/io.py
```
```python
"""Filesystem and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any, indent: int = 2) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=indent, ensure_ascii=False),
        encoding="utf-8",
    )


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, payload: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")

```
```text
# FILE: src/tc_adv/utils/logging.py
```
```python
"""Minimal logging helpers for reproducible experiments."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir


def build_logger(log_dir: str | Path, name: str = "tc_adv") -> logging.Logger:
    log_path = ensure_dir(log_dir) / f"{name}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def capture_manifest(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _safe_git_hash(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_git_hash() -> str:
    try:
        output = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return output.decode("utf-8").strip()
    except Exception:
        return "not-a-git-repository"

```
```text
# FILE: scripts/evaluate.py
```
```python
from tc_adv.experiments.runner import evaluate_config


if __name__ == "__main__":
    evaluate_config("configs/experiments/smoke.yaml")

```
```text
# FILE: scripts/export_code.py
```
```python
from tc_adv.report.exporter import export_repository_code


if __name__ == "__main__":
    export_repository_code("outputs/code_bundle.md")

```
```text
# FILE: scripts/run_suite.py
```
```python
from tc_adv.experiments.runner import run_experiment_suite


if __name__ == "__main__":
    run_experiment_suite(["configs/experiments/smoke.yaml"])

```
```text
# FILE: scripts/train.py
```
```python
from tc_adv.experiments.runner import train_config


if __name__ == "__main__":
    train_config("configs/experiments/smoke.yaml")

```
```text
# FILE: tests/test_bridge_smoke_pipeline.py
```
```python
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

```
```text
# FILE: tests/test_config_loader.py
```
```python
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

```
```text
# FILE: tests/test_ecm.py
```
```python
from tc_adv.config.schemas import ECMConfig
from tc_adv.discriminators.ecm import EvolutionaryConsistencyModule, merge_neighbor_histories, sinusoidal_time_encoding


def test_merge_neighbor_histories_filters_future_neighbors():
    neighbors, deltas = merge_neighbor_histories(
        subject_neighbors=["B", "C"],
        object_neighbors=["D"],
        subject_deltas=[0.0, -1.0],
        object_deltas=[2.0],
        history_window=4,
    )
    assert "C" not in neighbors
    assert all(delta >= 0.0 for delta in deltas)


def test_time_encoding_dimension_matches_config():
    encoding = sinusoidal_time_encoding([0.0, 1.0, 2.0], dim=8)
    assert len(encoding) == 3
    assert len(encoding[0]) == 8


def test_ecm_python_path_returns_probability():
    module = EvolutionaryConsistencyModule(embedding_dim=4, config=ECMConfig(hidden_dim=4, time_encoding_dim=4, num_heads=1))
    prob = module.probability(
        subject_embed=[[0.1, 0.2, 0.3, 0.4]],
        relation_embed=[[0.0, 0.1, 0.0, 0.1]],
        object_embed=[[0.2, 0.1, 0.0, -0.1]],
        history_entity_embed=[[[0.1, 0.1, 0.1, 0.1]]],
        history_deltas=[[1.0]],
        history_mask=[[True]],
    )
    assert 0.0 <= prob[0] <= 1.0

```
```text
# FILE: tests/test_exporter.py
```
```python
from pathlib import Path

from tc_adv.report.exporter import export_repository_code


def test_exporter_covers_source_tree(tmp_path: Path):
    output = tmp_path / "code_bundle.md"
    export_repository_code(output_path=output, repo_root=Path.cwd())
    payload = output.read_text(encoding="utf-8")
    assert "# FILE: src/tc_adv/cli.py" in payload
    assert "```python" in payload
    assert "Summary" not in payload

```
```text
# FILE: tests/test_objectives.py
```
```python
from tc_adv.training.objectives import StepRatioScheduler, anneal_temperature, dynamic_margin


def test_dynamic_margin_monotonic():
    low = dynamic_margin(0.1, alpha=1.0, beta=2.5)
    high = dynamic_margin(0.9, alpha=1.0, beta=2.5)
    assert high > low > 0.0


def test_dynamic_margin_keeps_floor_when_probability_zero():
    margin = dynamic_margin(0.0, alpha=1.0, beta=2.5)
    assert margin >= 1.0


def test_temperature_annealing_respects_floor():
    assert anneal_temperature(0.1, anneal_rate=0.5, min_temp=0.05) == 0.05


def test_scheduler_maintains_3_to_1_ratio():
    scheduler = StepRatioScheduler(g_steps=3, d_steps=1)
    assert scheduler.cycle() == ["G", "G", "G", "D"]

```
```text
# FILE: tests/test_pipeline_fixture.py
```
```python
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

```
```text
# FILE: tests/test_trm.py
```
```python
from types import SimpleNamespace

from tc_adv.discriminators.trm import TemporalRationalityModule
from tc_adv.config.schemas import TRMConfig


def _sample(subject: str, obj: str, timestamp: int):
    return SimpleNamespace(quadruple=SimpleNamespace(subject=subject, object=obj, timestamp=timestamp))


def test_trm_scores_decay_outside_active_window():
    module = TemporalRationalityModule(TRMConfig())
    module.build_index(
        [
            _sample("A", "B", 1),
            _sample("A", "C", 2),
            _sample("A", "D", 3),
        ]
    )
    near = module.normalized_activity_score("A", 2)
    far = module.normalized_activity_score("A", 20)
    assert near > far


def test_trm_normalization_is_bounded():
    module = TemporalRationalityModule(TRMConfig())
    module.build_index([_sample("A", "B", 5), _sample("A", "C", 6)])
    score = module.normalized_activity_score("A", 5)
    assert 0.0 <= score <= 1.0 + 1e-6

```
