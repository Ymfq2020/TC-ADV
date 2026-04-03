"""Generator backend adapters for TC-ADV."""

from __future__ import annotations

import hashlib
import inspect
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
        self.device = getattr(
            self.base_trainer,
            "device",
            next(self.model.parameters()).device,
        )
        self.entity_types = {
            entity_id: self.base_trainer._entity_types(entity_id)
            for entity_id in self.entities
        }

    def score_candidates(self, sample) -> dict[str, float]:
        if hasattr(self.base_trainer, "_current_candidate_scores"):
            return self.base_trainer._current_candidate_scores(sample)
        if hasattr(self.base_trainer, "_current_candidate_scores_batch"):
            batch_fn = self.base_trainer._current_candidate_scores_batch
            parameters = inspect.signature(batch_fn).parameters
            if "candidate_pools" in parameters:
                batch_scores = batch_fn([sample], [list(self.entities)])
            else:
                batch_scores = batch_fn([sample])
            if isinstance(batch_scores, list):
                if not batch_scores:
                    raise ValueError("_current_candidate_scores_batch returned an empty list.")
                return batch_scores[0]
            if isinstance(batch_scores, tuple):
                if not batch_scores:
                    raise ValueError("_current_candidate_scores_batch returned an empty tuple.")
                return batch_scores[0]
            if isinstance(batch_scores, dict):
                first_value = next(iter(batch_scores.values()), None)
                if isinstance(first_value, dict):
                    return first_value
            raise TypeError(
                f"Unsupported return type from _current_candidate_scores_batch: {type(batch_scores)!r}"
            )
        raise AttributeError(
            "LMCATICTrainer exposes neither _current_candidate_scores nor "
            "_current_candidate_scores_batch."
        )

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
        batch = self._move_batch_to_device(batch)
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
            relation_history = torch.tensor([sample.relation_history], dtype=torch.float32, device=self.device)
            subject_ids = torch.tensor([self.entity_to_idx[sample.quadruple.subject]], dtype=torch.long, device=self.device)
            relation_ids = torch.tensor([self.relation_to_idx[sample.quadruple.relation]], dtype=torch.long, device=self.device)
            subject_neighbor_ids, subject_neighbor_deltas = self.base_trainer._pad_neighbors(
                [sample.subject_neighbors],
                [sample.extra.get("subject_neighbor_deltas", [])],
            )
            subject_neighbor_ids = subject_neighbor_ids.to(self.device)
            subject_neighbor_deltas = subject_neighbor_deltas.to(self.device)
            subject_embed, _ = self.model.encode_entities(
                prompts=[sample.subject_prompt],
                relation_histories=relation_history,
                entity_ids=subject_ids,
                neighbor_ids=subject_neighbor_ids,
                neighbor_deltas=subject_neighbor_deltas,
            )
            subject_embed = subject_embed.repeat(len(candidate_ids), 1)
            object_ids = torch.tensor([self.entity_to_idx[candidate] for candidate in candidate_ids], dtype=torch.long, device=self.device)
            repeated_histories = relation_history.repeat(len(candidate_ids), 1)
            object_neighbor_ids, object_neighbor_deltas = self.base_trainer._pad_neighbors(
                [self.entity_neighbor_cache.get(candidate, []) for candidate in candidate_ids],
                [self.entity_delta_cache.get(candidate, []) for candidate in candidate_ids],
            )
            object_neighbor_ids = object_neighbor_ids.to(self.device)
            object_neighbor_deltas = object_neighbor_deltas.to(self.device)
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
                torch.zeros((len(candidate_ids), 0), dtype=torch.long, device=self.device),
                torch.zeros((len(candidate_ids), 0), dtype=torch.float32, device=self.device),
                torch.zeros((len(candidate_ids), 0), dtype=torch.bool, device=self.device),
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
            torch.tensor(padded_ids, dtype=torch.long, device=self.device),
            torch.tensor(padded_deltas, dtype=torch.float32, device=self.device),
            torch.tensor(padded_mask, dtype=torch.bool, device=self.device),
        )

    def save_generator_checkpoint(self, name: str) -> None:
        self.base_trainer._save_checkpoint(name)

    def load_generator_checkpoint(self, name: str) -> None:
        path = Path(self.base_config.checkpoint_dir) / name
        if not path.exists():
            return
        try:
            self.base_trainer._load_checkpoint(name)
            return
        except Exception:
            payload = torch.load(path, map_location=self.device, weights_only=False)
            if isinstance(payload, dict):
                if "model" in payload:
                    self.model.load_state_dict(payload["model"])
                    return
                if "model_state_dict" in payload:
                    self.model.load_state_dict(payload["model_state_dict"])
                    return
            self.model.load_state_dict(payload)

    def _move_batch_to_device(self, payload):
        if torch is None:
            return payload
        if isinstance(payload, torch.Tensor):
            return payload.to(self.device)
        if isinstance(payload, dict):
            return {key: self._move_batch_to_device(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [self._move_batch_to_device(value) for value in payload]
        if isinstance(payload, tuple):
            return tuple(self._move_batch_to_device(value) for value in payload)
        return payload


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
