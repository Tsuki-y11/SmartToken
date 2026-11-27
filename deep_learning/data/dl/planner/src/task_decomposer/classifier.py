"""Eight-dimension classifier inference over the saved Data A model bundle."""

from __future__ import annotations
from submission_compat import load_trusted_joblib

import math
from pathlib import Path
from typing import Any, Dict, Mapping, Protocol

import joblib
import numpy as np

from .schema import DIMENSIONS, DIMENSION_SET


class ClassificationProtocol(Protocol):
    def classify(self, text: str, context: str = "") -> Dict[str, Any]:
        """Return dimensions and real model confidence values."""


class TaskClassifier:
    """Load a joblib bundle and expose a small inference-only API."""

    def __init__(
        self,
        model_path: str | Path,
        threshold: float | Mapping[str, float] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"classifier model bundle not found: {self.model_path}")

        bundle = load_trusted_joblib(self.model_path)
        if not isinstance(bundle, dict):
            raise ValueError("classifier model bundle must be a dictionary")

        label_cols = bundle.get("label_cols")
        if not isinstance(label_cols, list) or set(label_cols) != DIMENSION_SET:
            raise ValueError("classifier bundle label columns do not match the eight dimensions")
        if len(label_cols) != len(set(label_cols)):
            raise ValueError("classifier bundle label columns must be unique")

        self.vectorizer = bundle.get("vectorizer")
        self.reducer = bundle.get("reducer")
        self.estimators = bundle.get("estimators")
        self.arch = bundle.get("arch", "logreg")
        self.embedding_model = None
        if self.arch == "sentence_transformer":
            embedding_model_name = bundle.get("embedding_model_name")
            if not isinstance(embedding_model_name, str) or not embedding_model_name:
                raise ValueError("sentence-transformer bundle is missing embedding_model_name")
            from sentence_transformers import SentenceTransformer

            # The frozen V5 bundle refers to a project-local fine-tuned encoder.
            # Resolve relative to repository root, never download/substitute a public E5.
            local_encoder = Path(embedding_model_name)
            if not local_encoder.is_absolute():
                local_encoder = Path(__file__).resolve().parents[3] / local_encoder
            if not local_encoder.is_dir() or not (local_encoder / "model.safetensors").is_file():
                raise FileNotFoundError(
                    "ORIGINAL_ENCODER_REQUIRED: copy the frozen model.safetensors into "
                    f"{local_encoder}; public E5 substitutes are forbidden"
                )
            self.embedding_model = SentenceTransformer(str(local_encoder))
        elif not hasattr(self.vectorizer, "transform"):
            raise ValueError("classifier bundle is missing a transformable vectorizer")
        if not isinstance(self.estimators, dict):
            raise ValueError("classifier bundle is missing estimators")

        self.label_cols = list(label_cols)
        if isinstance(threshold, Mapping):
            invalid = set(threshold) - DIMENSION_SET
            if invalid:
                raise ValueError(f"invalid threshold dimensions: {sorted(invalid)}")
            self.thresholds = {dimension: 0.5 for dimension in DIMENSIONS}
            self.thresholds.update(
                {dimension: float(value) for dimension, value in threshold.items()}
            )
        else:
            if threshold is None:
                bundle_thresholds = bundle.get("thresholds")
                if isinstance(bundle_thresholds, Mapping):
                    self.thresholds = {
                        dimension: float(bundle_thresholds.get(dimension, 0.5))
                        for dimension in DIMENSIONS
                    }
                else:
                    self.thresholds = {dimension: 0.5 for dimension in DIMENSIONS}
            else:
                threshold_value = float(threshold)
                self.thresholds = {dimension: threshold_value for dimension in DIMENSIONS}

        for dimension, value in self.thresholds.items():
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"threshold for {dimension} must be between 0 and 1")

    def classify(self, text: str, context: str = "") -> Dict[str, Any]:
        """Classify one instruction and return selected dimensions and confidences."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("text to classify must be a non-empty string")
        if not isinstance(context, str):
            raise ValueError("context must be a string")

        model_text = context + "\nUSER:\n" + text
        if self.arch == "sentence_transformer":
            features = self.embedding_model.encode(
                [model_text],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            features = self.vectorizer.transform([model_text])
            if self.reducer is not None:
                features = self.reducer.transform(features)

        probabilities: Dict[str, float] = {}
        dimensions: list[str] = []
        for dimension in self.label_cols:
            estimator = self.estimators.get(dimension)
            if estimator is None:
                # Unknown dimensions are omitted, never represented as fake 0.0.
                continue

            probability = float(estimator.predict_proba(features)[0, 1])
            if not math.isfinite(probability):
                raise ValueError(f"model produced a non-finite probability for {dimension}")
            probability = min(1.0, max(0.0, probability))
            probabilities[dimension] = round(probability, 6)
            if probability >= self.thresholds[dimension]:
                dimensions.append(dimension)

        return {
            "dimensions": dimensions,
            "dimension_confidence": {
                dimension: probabilities[dimension] for dimension in dimensions
            },
        }
