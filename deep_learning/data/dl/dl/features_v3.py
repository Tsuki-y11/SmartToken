"""V2 rendered-text features plus row-specific v3 capability annotations.

Only text features may be deduplicated/cached by prompt. The annotation matrix
is always built from each record, including distinct annotations on equal text.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from deep_learning.dl.dimensions import LABEL_FEATURE_NAMES, encode_dimensions
from deep_learning.dl.features_v2 import FeatureConfigV2, encode_prompt_features


def annotation_features(row_count: int, *, dimensions: Sequence | None = None,
                        label_sources: Sequence | None = None,
                        dimension_masks: Sequence | None = None) -> np.ndarray:
    def column(values, name):
        if values is None:
            return [None] * row_count
        if isinstance(values, (str, bytes)):
            raise ValueError(f'{name} must contain one value per row')
        values = list(values)
        if len(values) != row_count:
            raise ValueError(f'{name} length does not match prompt rows')
        return values

    labels = column(dimensions, 'dimensions')
    sources = column(label_sources, 'label_sources')
    masks = column(dimension_masks, 'dimension_masks')
    return np.asarray([encode_dimensions(value, source, dimension_mask=mask)
                       for value, source, mask in zip(labels, sources, masks)],
                      dtype=np.float32).reshape(row_count, len(LABEL_FEATURE_NAMES))


def encode_labeled_features(prompts: Sequence[str], encoder: object, *,
                            dimensions: Sequence | None = None,
                            label_sources: Sequence | None = None,
                            dimension_masks: Sequence | None = None,
                            use_label_features: bool = True,
                            tokenizer: object | None = None,
                            config: FeatureConfigV2 | None = None,
                            batch_size: int = 128) -> np.ndarray:
    prompts = list(prompts)
    annotations = annotation_features(len(prompts), dimensions=dimensions,
                                     label_sources=label_sources, dimension_masks=dimension_masks)
    text = encode_prompt_features(prompts, encoder, tokenizer=tokenizer, config=config, batch_size=batch_size)
    return np.concatenate((text, annotations), axis=1) if use_label_features else text
