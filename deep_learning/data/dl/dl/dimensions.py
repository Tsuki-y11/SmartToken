"""Standalone v3 capability vocabulary and strict annotation provenance.

An explicit empty sequence is a known negative annotation. ``None`` means no
annotation exists. ML annotations are declared predictions, never human truth.
"""
from __future__ import annotations

import numpy as np


LABELS = ('agents', 'coding', 'reasoning', 'knowledge', 'math', 'multilingual',
          'multimodal', 'instruction_following')
LABEL_SOURCES = ('human', 'ml', 'dataset_proxy', 'missing')
LABEL_FEATURE_NAMES = LABELS + tuple(f'known_{label}' for label in LABELS) + (
    'labels_available', 'source_human', 'source_ml', 'source_dataset_proxy', 'source_missing')
LABEL_POLICY = 'capability_requirements_human_ml_or_partial_dataset_proxy_v1'
LABEL_POLICY_VERSION = '0.5.1'


def label_schema() -> dict:
    """Return fresh JSON-safe metadata; artifact reload requires exact equality."""
    return {'version': 1, 'labels': list(LABELS), 'sources': list(LABEL_SOURCES),
            'feature_names': list(LABEL_FEATURE_NAMES), 'policy': LABEL_POLICY,
            'label_policy_version': LABEL_POLICY_VERSION,
            'missing_policy': 'None_is_unannotated;empty_list_is_known_negative',
            'provenance_policy': 'present_requires_human_ml_or_dataset_proxy;absent_requires_missing',
            'mask_policy': 'human_ml_default_all_known;dataset_proxy_requires_explicit_mask;positive_subset_known'}


def validate_label_schema(schema: object) -> None:
    if schema != label_schema():
        raise ValueError('saved v3 label schema does not match vocabulary, order, or policy')


def validate_label_policy_version(value: object = None) -> None:
    """Legacy rows may omit policy; any attested taxonomy must match definitions."""
    if value is not None and (not isinstance(value, str) or value != LABEL_POLICY_VERSION):
        raise ValueError(f'unsupported label policy version; expected {LABEL_POLICY_VERSION}')


def _labels(value: object, field: str) -> list[str]:
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise ValueError(f'{field} must be a sequence of labels')
    if isinstance(value, np.ndarray) and value.ndim != 1:
        raise ValueError(f'{field} must be one-dimensional')
    labels = list(value)
    if any(not isinstance(label, str) for label in labels):
        raise ValueError(f'{field} must contain string labels')
    if len(set(labels)) != len(labels) or set(labels) - set(LABELS):
        raise ValueError(f'{field} contains duplicate or unknown labels')
    return labels


def normalize_annotation(dimensions: object = None, label_source: object = None, *,
                         dimension_mask: object = None) -> tuple[list[str] | None, str, list[str]]:
    """Validate annotations; a mask lists dimensions whose positives/negatives are known."""
    if dimensions is None:
        if label_source is not None and (not isinstance(label_source, str) or label_source != 'missing'):
            raise ValueError('absent dimensions require label_source missing or None')
        if dimension_mask is not None:
            raise ValueError('absent dimensions require absent dimension_mask')
        return None, 'missing', []
    labels = _labels(dimensions, 'dimensions')
    if not isinstance(label_source, str) or label_source not in ('human', 'ml', 'dataset_proxy'):
        raise ValueError('present dimensions require explicit label_source human, ml or dataset_proxy')
    if dimension_mask is None:
        if label_source == 'dataset_proxy':
            raise ValueError('dataset_proxy requires explicit dimension_mask')
        mask = list(LABELS)
    else:
        mask = _labels(dimension_mask, 'dimension_mask')
    if set(labels) - set(mask):
        raise ValueError('positive dimensions must be included in dimension_mask')
    return labels, label_source, mask


def normalize_dimensions(dimensions: object = None, label_source: object = None) -> tuple[list[str] | None, str]:
    labels, source, _mask = normalize_annotation(dimensions, label_source)
    return labels, source


def encode_dimensions(dimensions: object = None, label_source: object = None, *,
                      dimension_mask: object = None) -> np.ndarray:
    """Eight positives, eight known bits, availability and four provenance bits."""
    labels, source, mask = normalize_annotation(dimensions, label_source, dimension_mask=dimension_mask)
    present = labels is not None
    chosen = set(labels or [])
    return np.asarray([float(label in chosen) for label in LABELS] +
                      [float(label in mask) for label in LABELS] + [float(present)] +
                      [float(source == name) for name in LABEL_SOURCES], dtype=np.float32)
