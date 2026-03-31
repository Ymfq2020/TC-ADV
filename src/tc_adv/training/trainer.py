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
