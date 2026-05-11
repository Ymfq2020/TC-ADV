from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

from tc_adv.training.trainer import TCADVTrainer


@dataclass(frozen=True)
class _Quadruple:
    subject: str
    relation: str
    object: str
    timestamp: int
    split: str = "test"
    is_inductive: bool = False


@dataclass
class _Sample:
    quadruple: _Quadruple
    subject_prompt: str = "subject"
    object_prompt: str = "object"
    relation_history: list[float] = field(default_factory=lambda: [0.2, 0.4])
    subject_neighbors: list[str] = field(default_factory=lambda: ["N1"])
    object_neighbors: list[str] = field(default_factory=lambda: ["O1"])
    subject_types: tuple[str, ...] = ("type=S",)
    object_types: tuple[str, ...] = ("type=O",)
    subject_neighbor_relations: list[str] = field(default_factory=lambda: ["R_prev"])
    object_neighbor_relations: list[str] = field(default_factory=lambda: ["R_obj"])
    extra: dict[str, list[float]] = field(
        default_factory=lambda: {
            "subject_neighbor_deltas": [1.0],
            "object_neighbor_deltas": [2.0],
        }
    )


def _sample(subject: str = "S", relation: str = "R", obj: str = "O", timestamp: int = 10):
    return _Sample(quadruple=_Quadruple(subject=subject, relation=relation, object=obj, timestamp=timestamp))


class _FakeEvaluator:
    def __init__(self):
        self.calls = []

    def evaluate(self, predictions):
        self.calls.append(predictions)
        return SimpleNamespace(to_dict=lambda: {"MRR": 1.0, "Hits@10": 1.0})


class _FakeGenerator:
    supports_gradient = False

    def __init__(self, samples):
        dataset = SimpleNamespace(samples=samples)
        self.train_dataset = dataset
        self.valid_dataset = dataset
        self.test_dataset = dataset
        self.filtered_targets = {}
        self.seen_samples = []

    def score_candidates(self, sample):
        self.seen_samples.append(sample)
        if sample.quadruple.timestamp == 1:
            return {"O1": 0.9, "X": 0.1}
        if sample.quadruple.timestamp == 2:
            return {"O2": 0.8, "X": 0.2}
        return {"O": 0.9, "X": 0.1}

    def load_generator_checkpoint(self, name: str) -> None:
        self.loaded = name


def _build_trainer(samples):
    trainer = object.__new__(TCADVTrainer)
    trainer.generator = _FakeGenerator(samples)
    trainer.output_dir = Path("/tmp")
    trainer.runtime_device = None
    trainer.temperature = 1.0
    trainer.trm = None
    trainer.ecm = None
    trainer.config = SimpleNamespace(
        tc_adv=SimpleNamespace(
            fusion=SimpleNamespace(gamma=0.7),
            trainer=SimpleNamespace(topk_fake_candidates=2),
        ),
        metadata={},
    )
    trainer.base_config = SimpleNamespace(
        model=SimpleNamespace(tgn_time_window_days=3, tgn_neighbor_size=4),
        ontology_keys=["type"],
    )
    trainer.generator.base_trainer = SimpleNamespace(
        bie_records={},
        _entity_prompt=lambda entity_id: f"prompt::{entity_id}",
    )
    evaluator = _FakeEvaluator()
    trainer.bridge = SimpleNamespace(create_filtered_evaluator=lambda filtered: evaluator, validate=lambda: None)
    trainer._load_checkpoint_pair = lambda checkpoint_name: None
    return trainer, evaluator


def test_evaluate_with_noise_does_not_mutate_frozen_quadruple():
    sample = _sample(timestamp=10)
    trainer, _ = _build_trainer([sample])
    original_quadruple = sample.quadruple
    trainer.evaluate_with_noise(split="test", sigma=1.0)
    assert sample.quadruple == original_quadruple


def test_multistep_eval_uses_real_future_queries_per_horizon():
    samples = [
        _sample(obj="O1", timestamp=1),
        _sample(obj="O2", timestamp=2),
    ]
    trainer, evaluator = _build_trainer(samples)
    trainer._resolve_lmca_types = lambda: SimpleNamespace(
        TemporalQuadruple=_Quadruple,
        ProcessedSample=_Sample,
        empty_bie_record=lambda entity_id: SimpleNamespace(attributes={}),
        relation_history_vector=lambda history, relation, timestamp, window_size=16: [float(len(history)), float(timestamp)],
        sample_temporal_neighbors=lambda history, subject, obj, timestamp, window_days, max_neighbors: (
            [],
            [],
            [],
            [],
            [],
            [],
        ),
        extract_entity_types=lambda record, ontology_keys: ("type=UNKNOWN",),
    )
    trainer.evaluate_multi_step(split="test", max_steps=2)
    assert len(evaluator.calls) == 2
    step1_predictions = evaluator.calls[0]
    step2_predictions = evaluator.calls[1]
    assert [row["timestamp"] for row in step1_predictions] == [1, 2]
    assert [row["gold"] for row in step1_predictions] == ["O1", "O2"]
    assert [row["timestamp"] for row in step2_predictions] == [2]
    assert [row["gold"] for row in step2_predictions] == ["O2"]


def test_multistep_rollout_rebuilds_neighbors_relations_and_deltas():
    samples = [
        _sample(obj="O1", timestamp=1),
        _sample(obj="O2", timestamp=2),
    ]
    trainer, _ = _build_trainer(samples)

    def _resolve_lmca_types():
        def empty_bie_record(entity_id):
            return SimpleNamespace(attributes={})

        def relation_history_vector(history, relation, timestamp, window_size=16):
            return [float(len(history)), float(timestamp)]

        def sample_temporal_neighbors(history, subject, obj, timestamp, window_days, max_neighbors):
            predicted = [quad.object for quad in history if quad.subject == subject and quad.timestamp <= timestamp]
            relations = [quad.relation for quad in history if quad.subject == subject and quad.timestamp <= timestamp]
            deltas = [float(timestamp - quad.timestamp) for quad in history if quad.subject == subject and quad.timestamp <= timestamp]
            return (
                predicted[-max_neighbors:],
                relations[-max_neighbors:],
                deltas[-max_neighbors:],
                [subject],
                [f"{relation}__inverse" for relation in relations[-1:]] if relations else [],
                [0.0] if relations else [],
            )

        return SimpleNamespace(
            TemporalQuadruple=_Quadruple,
            ProcessedSample=_Sample,
            empty_bie_record=empty_bie_record,
            relation_history_vector=relation_history_vector,
            sample_temporal_neighbors=sample_temporal_neighbors,
            extract_entity_types=lambda record, ontology_keys: ("type=UNKNOWN",),
        )

    trainer._resolve_lmca_types = _resolve_lmca_types
    trainer.evaluate_multi_step(split="test", max_steps=2)
    future_query = trainer.generator.seen_samples[-1]
    assert future_query.quadruple.timestamp == 2
    assert "O1" in future_query.subject_neighbors
    assert "R" in future_query.subject_neighbor_relations
    assert 1.0 in future_query.extra["subject_neighbor_deltas"]


def test_discriminator_fallback_supervises_fused_probability_only():
    trainer, _ = _build_trainer([_sample()])
    trainer._violation_probabilities = lambda sample, candidate_ids, requires_grad: {
        "p_trm": [0.2] if candidate_ids == ["O"] else [0.9],
        "p_ecm": [0.8] if candidate_ids == ["O"] else [0.1],
        "p_fake": [0.3] if candidate_ids == ["O"] else [0.7],
    }
    loss = trainer._discriminator_step_fallback(_sample(), ["X"])
    assert abs(loss - (0.3**2 + (1.0 - 0.7) ** 2)) < 1e-9
