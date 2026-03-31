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
