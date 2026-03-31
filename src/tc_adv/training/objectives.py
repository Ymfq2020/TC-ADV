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
