"""Independent supervised research heads and a lazy frozen-model adapter.

Artifacts use joblib and must come from a trusted local source. Hash verification
detects corruption; it does not make an untrusted pickle safe to deserialize.
No optional encoder or frozen torch dependency is imported at module import.
"""
from __future__ import annotations
from submission_compat import load_trusted_joblib

from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .data import LABELS, SUPPORTED_POLICY_VERSION, normalized_prompt_sha256, normalized_task_context_sha256, validate_records


ARTIFACT_VERSION = 1
FEATURE_SETTINGS = {"analyzer": "char", "ngram_range": (2, 5), "min_df": 1,
                    "max_features": 50000, "sublinear_tf": True, "norm": "l2"}


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encoder_directory(path: str | Path) -> Path:
    local = Path(path).expanduser().resolve()
    if not local.is_dir():
        raise ValueError("encoder_path must be an existing local model directory")
    return local


def _encoder_hash(path: Path) -> str:
    files = sorted((file for file in path.rglob("*") if file.is_file()),
                   key=lambda file: file.relative_to(path).as_posix())
    if not files:
        raise ValueError("encoder_path must be a nonempty local model directory")
    digest = hashlib.sha256()
    for file in files:
        digest.update(_json_bytes([file.relative_to(path).as_posix(), _sha256_file(file)]))
    return digest.hexdigest()


def _label_text(text: str, context: str = "") -> str:
    if not isinstance(text, str) or not isinstance(context, str):
        raise ValueError("text and context must be strings")
    return context + "\nUSER:\n" + text


def _boundary_text(left: str, right: str) -> str:
    if not isinstance(left, str) or not isinstance(right, str):
        raise ValueError("left and right must be strings")
    return left + "\nBOUNDARY:\n" + right


class _Features:
    def __init__(self, encoder_path=None):
        self.vectorizer = TfidfVectorizer(**FEATURE_SETTINGS)
        self.encoder_path = str(_encoder_directory(encoder_path)) if encoder_path is not None else None
        self.encoder_sha256 = _encoder_hash(Path(self.encoder_path)) if self.encoder_path else None
        self._encoder = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def _local_encoder(self):
        if self._encoder is None:
            local = _encoder_directory(self.encoder_path)
            if _encoder_hash(local) != self.encoder_sha256:
                raise ValueError("local encoder hash differs from the trained artifact")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("local embeddings require sentence-transformers and its optional dependencies") from exc
            self._encoder = SentenceTransformer(str(local), local_files_only=True, trust_remote_code=False)
        return self._encoder

    def fit(self, texts):
        self.vectorizer.fit(texts)
        return self

    def transform(self, texts):
        lexical = self.vectorizer.transform(texts)
        if self.encoder_path is None:
            return lexical
        embedding = np.asarray(self._local_encoder().encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float64)
        if embedding.ndim != 2 or embedding.shape[0] != len(texts) or not np.isfinite(embedding).all():
            raise ValueError("local encoder returned invalid features")
        return hstack([lexical, csr_matrix(embedding)], format="csr")


def _support(targets):
    positive = int(np.sum(targets))
    return {"positive": positive, "negative": len(targets) - positive, "total": len(targets)}


def _require_binary(name, targets):
    support = _support(targets)
    if not support["positive"] or not support["negative"]:
        raise ValueError(f"{name} requires positive and negative reviewed training support")


def _select_threshold(targets, scores):
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    def objective(threshold):
        predicted = scores >= threshold
        tp = int(np.sum(predicted & (targets == 1)))
        fp = int(np.sum(predicted & (targets == 0)))
        fn = int(np.sum(~predicted & (targets == 1)))
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        return f1, -abs(float(threshold) - 0.5), float(threshold)
    return float(max(candidates, key=objective))


class _BinaryHead:
    def __init__(self, estimator, calibrator=None):
        self.estimator = estimator
        self.calibrator = calibrator

    def scores(self, features):
        if self.calibrator is None:
            return self.estimator.predict_proba(features)[:, 1]
        raw = self.estimator.decision_function(features).reshape(-1, 1)
        return self.calibrator.predict_proba(raw)[:, 1]


def _fit_head(name, train_x, targets, validation_x, validation_targets):
    _require_binary(name, targets)
    estimator = LogisticRegression(C=4.0, solver="liblinear", max_iter=2000, random_state=0)
    estimator.fit(train_x, targets)
    head = _BinaryHead(estimator)
    calibration = {"training_support": _support(targets), "validation_support": _support(validation_targets),
                   "status": "unavailable_validation_classes", "threshold_source": "default_0.5"}
    threshold = 0.5
    if len(np.unique(validation_targets)) == 2:
        calibrator = LogisticRegression(C=1000000.0, solver="lbfgs", max_iter=2000, random_state=0)
        calibrator.fit(estimator.decision_function(validation_x).reshape(-1, 1), validation_targets)
        head.calibrator = calibrator
        threshold = _select_threshold(validation_targets, head.scores(validation_x))
        calibration.update(status="sigmoid_validation", threshold_source="validation_f1",
                           caution="calibration and threshold use the same validation split; evaluate on independent test data")
    return head, threshold, calibration


def _examples(records, split):
    labels, splits, boundaries = [], [], []
    for record in records:
        if record["split"] != split:
            continue
        labels.append((_label_text(record["original_prompt"]), record["direct_dimensions"]))
        labels.extend((_label_text(task["instruction"], task["context"]), task["dimensions"])
                      for task in record["tasks"])
        splits.append((record["original_prompt"], int(record["should_split"])))
        boundaries.extend((_boundary_text(item["left"], item["right"]), int(item["is_boundary"]))
                          for item in record["boundaries"])
    return labels, splits, boundaries


def train_models(records, directory, encoder_path=None) -> dict:
    """Fit only reviewed train rows; calibrate/select thresholds on validation.

    Every training head requires both classes. Missing validation classes leave
    that head explicitly uncalibrated at threshold 0.5. No test rows are accepted.
    The output directory must be absent or empty to avoid replacing experiments.
    """
    records = validate_records(records)
    output = Path(directory).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("model output directory must be absent or empty")
    features = _Features(encoder_path)
    train = _examples(records, "train")
    validation = _examples(records, "validation")
    if not train[0]:
        raise ValueError("training records are required")
    targets = {label: np.asarray([int(label in labels) for _, labels in train[0]]) for label in LABELS}
    targets["split"] = np.asarray([label for _, label in train[1]])
    targets["boundary"] = np.asarray([label for _, label in train[2]])
    for name, values in targets.items():
        _require_binary(name, values)
    # The vocabulary and optional encoder features never see validation at fit.
    corpus = [text for family in train for text, _ in family]
    features.fit(corpus)
    train_features = [features.transform([text for text, _ in family]) for family in train]
    validation_features = [features.transform([text for text, _ in family]) if family else None for family in validation]
    heads, thresholds, calibration = {}, {}, {}
    for label in LABELS:
        val_y = np.asarray([int(label in labels) for _, labels in validation[0]])
        heads[label], thresholds[label], calibration[label] = _fit_head(
            label, train_features[0], targets[label], validation_features[0], val_y,
        )
    gates, gate_thresholds = {}, {}
    for index, name in ((1, "split"), (2, "boundary")):
        val_y = np.asarray([label for _, label in validation[index]])
        gates[name], gate_thresholds[name], calibration[name] = _fit_head(
            name, train_features[index], targets[name], validation_features[index], val_y,
        )
    feature_metadata = {"type": "char_tfidf_local_embeddings" if features.encoder_path else "char_tfidf",
                        "settings": json.loads(_json_bytes(FEATURE_SETTINGS)),
                        "fit_split": "train", "vocabulary_size": len(features.vectorizer.vocabulary_),
                        "label_preprocessing": "context_newline_USER_newline_text_v1",
                        "boundary_preprocessing": "left_newline_BOUNDARY_newline_right_v1",
                        "encoder_path": features.encoder_path, "encoder_sha256": features.encoder_sha256}
    metadata = {
        "artifact_version": ARTIFACT_VERSION, "model_id": "hybrid_supervised_logistic_v1",
        "created_at": datetime.now(timezone.utc).isoformat(), "label_policy_version": records[0]["label_policy_version"],
        "labels": list(LABELS), "thresholds": thresholds, "split_threshold": gate_thresholds["split"],
        "boundary_threshold": gate_thresholds["boundary"], "features": feature_metadata,
        "training": {"label_heads": 8, "estimator": "independent_LogisticRegression", "fit_split": "train",
                     "test_records_consumed": 0, "random_state": 0},
        "calibration": calibration,
        "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "data": {"sha256": hashlib.sha256(_json_bytes(records)).hexdigest(),
                 "leakage_checks": ["request_id", "group", "normalized_prompt", "normalized_task_context", "normalized_boundary"],
                 "independence_caution": "group and exact-normalization checks do not establish full historical independence",
                 "splits": {split: {"records": sum(record["split"] == split for record in records),
                                     "group_ids": sorted({record["group_id"] for record in records if record["split"] == split})}
                            for split in ("train", "validation")},
                 "sources": [{"request_id": record["request_id"], "group_id": record["group_id"], "split": record["split"],
                              "record_sha256": hashlib.sha256(_json_bytes(record)).hexdigest(),
                              "prompt_sha256": hashlib.sha256(record["original_prompt"].encode("utf-8")).hexdigest(),
                              "normalized_prompt_sha256": normalized_prompt_sha256(record["original_prompt"]),
                              "normalized_task_context_sha256": [normalized_task_context_sha256(record["original_prompt"])] + [
                                  normalized_task_context_sha256(task["instruction"], task["context"]) for task in record["tasks"]]}
                             for record in records]},
    }
    bundle = {"artifact_version": ARTIFACT_VERSION, "features": features, "heads": heads, "gates": gates}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hybrid-models-", dir=output.parent) as temporary:
        staging = Path(temporary)
        joblib.dump(bundle, staging / "models.joblib", compress=3)
        (staging / "metadata.json").write_bytes(_json_bytes(metadata))
        manifest = {"artifact_version": ARTIFACT_VERSION, "files": {
            filename: _sha256_file(staging / filename) for filename in ("models.joblib", "metadata.json")}}
        (staging / "manifest.json").write_bytes(_json_bytes(manifest))
        output.mkdir(exist_ok=True)
        for filename in ("models.joblib", "metadata.json", "manifest.json"):
            os.replace(staging / filename, output / filename)
    return _artifact_metadata(output, metadata, manifest)


def _artifact_metadata(directory, metadata, manifest):
    return {**metadata, "model_sha256": manifest["files"]["models.joblib"],
            "metadata_sha256": manifest["files"]["metadata.json"],
            "manifest_sha256": _sha256_file(directory / "manifest.json")}


def _probability(value, name):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid {name} probability")
    return value


class TaskModels:
    """A trusted local fitted feature transform, eight heads and two gate heads."""
    def __init__(self, bundle, metadata):
        self._features = bundle["features"]
        if metadata.get("coverage_training") and "label_features" not in bundle:
            raise ValueError("coverage candidate is missing its label feature transform")
        self._label_features = bundle.get("label_features", self._features)
        self._heads = bundle["heads"]
        self._gates = bundle["gates"]
        self.metadata = metadata
        self.thresholds = {label: _probability(metadata["thresholds"][label], label) for label in LABELS}
        self.split_threshold = _probability(metadata["split_threshold"], "split threshold")
        self.boundary_threshold = _probability(metadata["boundary_threshold"], "boundary threshold")
        self._lock = threading.RLock()

    @classmethod
    def load(cls, directory, *, encoder_path=None):
        """Verify local artifact hashes before loading trusted joblib contents."""
        directory = Path(directory).expanduser().resolve()
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest["artifact_version"] != ARTIFACT_VERSION:
                raise ValueError("unsupported model artifact version")
            for filename in ("models.joblib", "metadata.json"):
                if _sha256_file(directory / filename) != manifest["files"][filename]:
                    raise ValueError(f"artifact hash mismatch: {filename}")
            metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
            if metadata["artifact_version"] != ARTIFACT_VERSION or metadata["labels"] != list(LABELS):
                raise ValueError("incompatible model metadata")
            if metadata.get("label_policy_version") != SUPPORTED_POLICY_VERSION:
                raise ValueError(f"unsupported artifact label policy; expected {SUPPORTED_POLICY_VERSION}")
            bundle = load_trusted_joblib(directory / "models.joblib")
            if bundle["artifact_version"] != ARTIFACT_VERSION or set(bundle["heads"]) != set(LABELS) or set(bundle["gates"]) != {"split", "boundary"}:
                raise ValueError("incomplete model artifact")
            if encoder_path is not None:
                relocated = _encoder_directory(encoder_path)
                relocated_hash = _encoder_hash(relocated)
                transforms = [bundle["features"]]
                if "label_features" in bundle:
                    transforms.append(bundle["label_features"])
                for features in transforms:
                    if features.encoder_path is None:
                        raise ValueError("this artifact was trained without a local encoder")
                    if relocated_hash != features.encoder_sha256:
                        raise ValueError("relocated encoder hash differs from the trained artifact")
                    features.encoder_path, features._encoder = str(relocated), None
            return cls(bundle, _artifact_metadata(directory, metadata, manifest))
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("missing or malformed local model artifact") from exc

    def local_encoder(self):
        """Share the verified encoder with the token predictor without reloading it."""
        with self._lock:
            return self._features._local_encoder()

    def score_labels(self, text, context=""):
        with self._lock:
            if self._label_features is not self._features and self._label_features.encoder_path:
                if self._label_features.encoder_sha256 != self._features.encoder_sha256:
                    raise ValueError("coverage candidate encoder differs from the original gate encoder")
                self._label_features._encoder = self._features._local_encoder()
            features = self._label_features.transform([_label_text(text, context)])
            return {label: _probability(self._heads[label].scores(features)[0], label) for label in LABELS}

    def score_split(self, prompt):
        if not isinstance(prompt, str):
            raise ValueError("prompt must be a string")
        with self._lock:
            return _probability(self._gates["split"].scores(self._features.transform([prompt]))[0], "split")

    def score_boundary(self, left, right):
        with self._lock:
            return _probability(self._gates["boundary"].scores(self._features.transform([_boundary_text(left, right)]))[0], "boundary")


class FrozenModels:
    """Lazy adapter preserving frozen research backend preprocessing and weights."""
    split_threshold = 0.5
    boundary_threshold = 0.5

    def __init__(self, backend="v5", *, accept_unverified_encoder=False):
        if backend not in {"v5", "candidate_c", "best8", "hybrid_h1"}:
            raise ValueError(f"unknown classifier backend: {backend}")
        self.backend = backend
        self.accept_unverified_encoder = accept_unverified_encoder
        self._classifier = None
        self._lock = threading.RLock()

    def _load(self):
        with self._lock:
            if self._classifier is None:
                module = importlib.import_module("deep_learning.planner.src.task_decomposer.research_backends")
                self._classifier = module.build_classifier(self.backend, accept_unverified_encoder=self.accept_unverified_encoder)
            return self._classifier

    @property
    def thresholds(self):
        classifier = self._load()
        return {label: _probability(classifier.thresholds[label], label + " threshold") for label in LABELS}

    @property
    def metadata(self):
        metadata = {"model_id": self.backend, "backend": self.backend, "frozen": True,
                    "split_model": "unavailable", "boundary_model": "unavailable"}
        if self._classifier is not None:
            metadata.update(model_sha256=getattr(self._classifier, "model_sha256", ""),
                            preprocessing_version=getattr(self._classifier, "preprocessing_version", "unknown"))
        return metadata

    def score_labels(self, text, context=""):
        if not isinstance(text, str) or not isinstance(context, str):
            raise ValueError("text and context must be strings")
        with self._lock:
            classifier = self._load()
            if hasattr(classifier, "predict_all_dimensions"):
                scores = classifier.predict_all_dimensions(text, context)
            else:
                # Original V5 only exposes selected-label classify(). Reuse its
                # exact frozen feature path to recover *all* probabilities,
                # without changing thresholds, weights or original source code.
                model_text = context + "\nUSER:\n" + text
                if classifier.arch == "sentence_transformer":
                    features = classifier.embedding_model.encode(
                        [model_text], normalize_embeddings=True, show_progress_bar=False,
                    )
                else:
                    features = classifier.vectorizer.transform([model_text])
                    if classifier.reducer is not None:
                        features = classifier.reducer.transform(features)
                if set(classifier.estimators) != set(LABELS) or any(
                    classifier.estimators[label] is None for label in LABELS
                ):
                    raise ValueError("frozen classifier must provide all eight dimension estimators")
                scores = {label: classifier.estimators[label].predict_proba(features)[0, 1] for label in LABELS}
            if set(scores) != set(LABELS):
                raise ValueError("frozen classifier must return all eight dimension scores")
            return {label: _probability(scores[label], label) for label in LABELS}

    def score_split(self, prompt):
        return None

    def score_boundary(self, left, right):
        return None
