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
