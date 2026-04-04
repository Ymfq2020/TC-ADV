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
        model = getattr(config, "model", None)
        if model is not None:
            for field_name in ("llm_name", "smoke_llm_name"):
                value = getattr(model, field_name, None)
                if not value:
                    continue
                path_value = Path(value)
                if path_value.is_absolute():
                    continue
                resolved = (repo_root / path_value).resolve()
                if resolved.exists():
                    setattr(model, field_name, str(resolved))
        return config

    @staticmethod
    def _repo_root_for_config(config_path: Path) -> Path:
        for parent in config_path.resolve().parents:
            if (parent / ".git").exists() or (parent / "src").exists():
                return parent
        return config_path.parent.resolve()
