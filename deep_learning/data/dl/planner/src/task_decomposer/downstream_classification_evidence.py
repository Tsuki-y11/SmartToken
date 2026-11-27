"""Explicit classification evidence for downstream all-zero compatibility.

This is a sidecar proposal only. It does not add or change any frozen TaskPlan
V1.0 field.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


VALID_STATUS = frozenset({"EXECUTED", "NOT_EXECUTED", "EXECUTION_ERROR"})


def validate_classification_evidence(
    plan: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> dict[str, str]:
    """Classify empty dimensions without inventing a positive label.

    Required sidecar fields:
    - request_id
    - classifier_status: EXECUTED | NOT_EXECUTED | EXECUTION_ERROR
    - dimensions_known: boolean
    - all_zero: boolean

    `EXECUTED + dimensions_known + all_zero` is the only compatibility condition
    for a legal eight-dimension all-zero result. Everything else remains an
    explicit downstream error.
    """

    request_id = evidence.get("request_id")
    if request_id != plan.get("request_id"):
        raise ValueError("classification evidence request_id mismatch")
    status = evidence.get("classifier_status")
    if status not in VALID_STATUS:
        raise ValueError("invalid classifier_status")
    dimensions_known = evidence.get("dimensions_known")
    all_zero = evidence.get("all_zero")
    if not isinstance(dimensions_known, bool) or not isinstance(all_zero, bool):
        raise ValueError("dimensions_known and all_zero must be boolean")

    empty_dimensions = not plan.get("dimensions")
    if status == "EXECUTION_ERROR":
        return {
            "status": "ERROR",
            "reason": "MODEL_EXECUTION_ERROR",
            "downstream_action": "REJECT_EXECUTION",
        }
    if status == "NOT_EXECUTED" or not dimensions_known:
        return {
            "status": "UNKNOWN",
            "reason": "CLASSIFIER_NOT_EXECUTED_OR_LABELS_UNKNOWN",
            "downstream_action": "REJECT_UNTIL_CLASSIFICATION",
        }
    if empty_dimensions and all_zero:
        return {
            "status": "OK",
            "reason": "LEGAL_ALL_ZERO_EXECUTED",
            "downstream_action": "ACCEPT_WITH_SIDE_EVIDENCE",
        }
    if empty_dimensions:
        return {
            "status": "ERROR",
            "reason": "EMPTY_DIMENSIONS_WITHOUT_ALL_ZERO_EVIDENCE",
            "downstream_action": "REJECT",
        }
    if all_zero:
        return {
            "status": "ERROR",
            "reason": "EVIDENCE_PLAN_MISMATCH",
            "downstream_action": "REJECT",
        }
    return {
        "status": "OK",
        "reason": "POSITIVE_DIMENSIONS_PRESENT",
        "downstream_action": "NORMAL_PATH",
    }
