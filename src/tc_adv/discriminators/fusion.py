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
