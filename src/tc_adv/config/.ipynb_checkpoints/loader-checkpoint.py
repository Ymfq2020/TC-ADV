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
