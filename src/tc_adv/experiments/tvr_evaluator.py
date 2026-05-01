"""Offline Temporal Violation Rate (TVR) Evaluator.

Implements the offline rule-based scanner described in Chapter 4.5.2 to evaluate
Top-3 predictions for temporal boundary constraints and local evolution continuity.
"""

import json
from pathlib import Path
from collections import defaultdict

from tc_adv.discriminators.trm import infer_bandwidth, gaussian_kde_score, normalize_activity_score
from tc_adv.bridge.lmca_tic import LMCATICBridge
from tc_adv.config.loader import load_tcadv_config


class TVREvaluator:
    def __init__(self, config_path: str, w: int = 3, threshold: float = 0.05):
        self.config = load_tcadv_config(config_path)
        self.bridge = LMCATICBridge()
        self.base_config = self.bridge.load_experiment_config(self.config.lmca_experiment_config)
        self.w = w
        self.threshold = threshold
        
        # Load dataset to build history
        self.train_dataset = self.bridge.create_dataset(self.base_config.processed_dir, "train")
        self.valid_dataset = self.bridge.create_dataset(self.base_config.processed_dir, "valid")
        self.test_dataset = self.bridge.create_dataset(self.base_config.processed_dir, "test")
        
        self.entity_timestamps = defaultdict(list)
        self.entity_relations = defaultdict(list)
        
        self._build_index()

    def _build_index(self):
        # Build history from train/valid to establish bounds and local contexts
        for sample in self.train_dataset.samples + self.valid_dataset.samples:
            q = sample.quadruple
            self.entity_timestamps[q.subject].append(q.timestamp)
            self.entity_timestamps[q.object].append(q.timestamp)
            
            # Record local history ordered by timestamp
            self.entity_relations[q.subject].append((q.timestamp, q.relation))
            self.entity_relations[q.object].append((q.timestamp, q.relation))

        # Sort and precalculate bandwidths for KDE
        self.entity_bandwidths = {}
        self.entity_score_max = {}
        for entity_id, values in self.entity_timestamps.items():
            ordered = sorted(values)
            self.entity_timestamps[entity_id] = ordered
            
            # Sort relations by time
            self.entity_relations[entity_id].sort(key=lambda x: x[0])
            
            bw = infer_bandwidth(ordered, "auto")
            self.entity_bandwidths[entity_id] = bw
            
            max_score = max(gaussian_kde_score(ordered, ts, bw, 1e-5) for ts in ordered) if ordered else 1e-5
            self.entity_score_max[entity_id] = max_score

    def _check_activity_window(self, entity_id: str, timestamp: int) -> bool:
        """Rule 1: Entity Active Time Window. Returns True if violated."""
        timestamps = self.entity_timestamps.get(entity_id, [])
        if len(timestamps) < 3:
            # Fallback for sparse entities: [first - margin, last + margin]
            if not timestamps:
                return False  # Can't evaluate
            first, last = timestamps[0], timestamps[-1]
            margin = (last - first) * 0.5 if last > first else 10.0 # arbitrary fallback margin
            return timestamp < (first - margin) or timestamp > (last + margin)
            
        bw = self.entity_bandwidths.get(entity_id, 1.0)
        max_score = self.entity_score_max.get(entity_id, 1e-5)
        raw = gaussian_kde_score(timestamps, timestamp, bw, 1e-5)
        norm = normalize_activity_score(raw, max_score + 1e-5, 1e-5)
        
        return norm < self.threshold

    def _check_evolution_continuity(self, entity_id: str, timestamp: int, candidate_relation: str) -> bool:
        """Rule 2: Local Evolution Continuity. Returns True if violated."""
        history = self.entity_relations.get(entity_id, [])
        # Get up to w relations before timestamp
        recent_relations = [rel for ts, rel in history if ts < timestamp][-self.w:]
        if not recent_relations:
            return False  # No history to violate
            
        # If candidate relation is not in recent window, check semantic overlap.
        if candidate_relation not in recent_relations:
            # Fallback heuristic ontology check: prefix matching (e.g. 'Make_Statement' matches 'Make_Appeal')
            cand_prefix = candidate_relation.split('_')[0] if '_' in candidate_relation else candidate_relation
            if not any(rel.startswith(cand_prefix) for rel in recent_relations):
                return True
        return False

    def evaluate_predictions(self, predictions_path: str) -> dict:
        path = Path(predictions_path)
        if not path.exists():
            raise FileNotFoundError(f"Predictions not found: {path}")
            
        with path.open("r", encoding="utf-8") as f:
            predictions = [json.loads(line) for line in f]
            
        violations = {
            "trm_violations": 0,
            "ecm_violations": 0,
            "total_violations": 0,
            "total_scanned": 0
        }
        
        for p in predictions:
            subject = p["subject"]
            timestamp = p["timestamp"]
            scores = p["scores"]
            top3 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for candidate, _ in top3:
                violations["total_scanned"] += 1
                
                # Check Subject and Candidate
                trm_viol = self._check_activity_window(subject, timestamp) or self._check_activity_window(candidate, timestamp)
                
                # Check continuity for candidate relation
                # Only check candidate side for simplicity as it's the predicted entity
                ecm_viol = self._check_evolution_continuity(candidate, timestamp, p["relation"])
                
                if trm_viol: violations["trm_violations"] += 1
                if ecm_viol: violations["ecm_violations"] += 1
                if trm_viol or ecm_viol: violations["total_violations"] += 1
                
        tvr = violations["total_violations"] / max(violations["total_scanned"], 1)
        return {
            "TVR": tvr,
            "trm_violation_rate": violations["trm_violations"] / max(violations["total_scanned"], 1),
            "ecm_violation_rate": violations["ecm_violations"] / max(violations["total_scanned"], 1),
            "total_scanned": violations["total_scanned"]
        }
