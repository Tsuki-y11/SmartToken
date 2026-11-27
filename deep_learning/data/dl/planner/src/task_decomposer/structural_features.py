"""Structural features for should_split classification."""

from __future__ import annotations

import re
from typing import Any, List

import numpy as np


CUE_PATTERN = re.compile(
    r"首先|然后|接着|其次|最后|再|同时|分别|另外|此外|并且|"
    r"第[一二三四五六七八九十\d]+步|first|second|third|fourth|then|next|"
    r"after that|finally|lastly|meanwhile|at the same time|also|additionally",
    re.IGNORECASE,
)


def extract_structural_features(values: List[Any]) -> np.ndarray:
    """Return prompt length, cue count, and punctuation count."""

    rows: List[List[float]] = []
    for value in values:
        text = str(value)
        rows.append(
            [
                float(len(text)),
                float(len(CUE_PATTERN.findall(text))),
                float(sum(char in text for char in "，。；、！？")),
            ]
        )
    return np.asarray(rows, dtype=float)
