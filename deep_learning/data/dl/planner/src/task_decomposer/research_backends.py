"""Selectable frozen research classifiers for LY planning integration.

The official production backend remains V5. Candidate C, Best8, and Hybrid-H1
are research-only and never share probabilities or retrain heads.
"""
from __future__ import annotations
from submission_compat import load_trusted_joblib

import hashlib
import json
import math
import threading
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import sentencepiece as spm
import torch
from scipy.sparse import csr_matrix, hstack

from .classifier import TaskClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[3]
V5_PATH = PROJECT_ROOT / "planner/models/v5/model_bundle.joblib"
CANDIDATE_C_PATH = (
    PROJECT_ROOT / "planner/experiments/shadow_candidates_v3/candidate_c_selective_ab.joblib"
)
CANDIDATE_C_VECTORIZER_PATH = (
    PROJECT_ROOT / "planner/experiments/shadow_candidates_v3/candidate_b_e5_char_tfidf.joblib"
)
BEST8_DIR = PROJECT_ROOT / "planner/experiments/best8_v14_v5"
BEST8_SPM_PATH = BEST8_DIR / "tokenizer_from_json_research.model"
BEST8_SPM_SHA256 = "9104bbc7a7761d39b75d18f646ed28359e94d5fa1513000659e874bbb2d79c92"

LABEL_COLS = [
    "coding",
    "math",
    "reasoning",
    "knowledge",
    "multimodal",
    "multilingual",
    "agents",
    "instruction_following",
]

HYBRID_H1_SOURCE = {
    "coding": "candidate_c",
    "math": "best8",
    "reasoning": "candidate_c",
    "knowledge": "best8",
    "multimodal": "best8",
    "multilingual": "best8",
    "agents": "best8",
    "instruction_following": "best8",
}


BEST8_V14_LABELS = frozenset(
    {
        "agents",
        "coding",
        "knowledge",
        "multilingual",
        "instruction_following",
    }
)
BEST8_V5_LABELS = frozenset({"reasoning", "math", "multimodal"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class BaseResearchClassifier:
    model_id = "research"
    preprocessing_version = "unknown"
    model_sha256 = ""
    encoder_status = "UNKNOWN"

    def predict_all_dimensions(self, text: str, context: str = "") -> dict[str, float]:
        raise NotImplementedError

    def classify(self, text: str, context: str = "") -> dict[str, Any]:
        scores = self.predict_all_dimensions(text, context)
        dimensions = [label for label in LABEL_COLS if scores[label] >= self.thresholds[label]]
        return {
            "dimensions": dimensions,
            "dimension_confidence": {
                label: round(float(scores[label]), 6) for label in dimensions
            },
        }


class V5Classifier(TaskClassifier):
    model_id = "v5"
    preprocessing_version = "context_newline_user_text_e5_v1"
    encoder_status = "VERIFIED_LOCAL_V5"

    def __init__(self) -> None:
        super().__init__(V5_PATH)
        self.model_sha256 = sha256_file(V5_PATH)


class CandidateCClassifier(BaseResearchClassifier):
    model_id = "candidate_c"
    preprocessing_version = "context_newline_user_text_e5_plus_char_tfidf_v1"
    encoder_status = "VERIFIED_LOCAL_V5_FOR_E5"

    def __init__(self) -> None:
        self.bundle = load_trusted_joblib(CANDIDATE_C_PATH)
        vectorizer_bundle = load_trusted_joblib(CANDIDATE_C_VECTORIZER_PATH)
        self.vectorizer = vectorizer_bundle["vectorizer"]
        self.thresholds = {
            label: float(self.bundle["thresholds"][label]) for label in LABEL_COLS
        }
        self.feature_types = dict(self.bundle["feature_types"])
        self.model_sha256 = sha256_file(CANDIDATE_C_PATH)
        self._v5 = V5Classifier()
        self._lock = threading.RLock()

    def predict_all_dimensions(self, text: str, context: str = "") -> dict[str, float]:
        model_text = context + "\nUSER:\n" + text
        with self._lock:
            embedding = np.asarray(
                self._v5.embedding_model.encode(
                    [model_text],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            scores: dict[str, float] = {}
            for label in LABEL_COLS:
                if self.feature_types[label] == "e5_embedding":
                    features = embedding
                else:
                    features = hstack(
                        [csr_matrix(embedding), self.vectorizer.transform([model_text])]
                    ).tocsr()
                value = float(self.bundle["models"][label].predict_proba(features)[0, 1])
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"invalid Candidate C score for {label}")
                scores[label] = value
            return scores


class Best8Classifier(BaseResearchClassifier):
    model_id = "best8"
    preprocessing_version = (
        "v14_reconstructed_spm_prompt_only_128_and_v5_original_fast_tokenizer_512"
    )
    encoder_status = "RESEARCH_V14_INPUT_PATH_CONFIRMED_INDEPENDENT_CACHE_MISSING"

    def __init__(self, *, accept_unverified_encoder: bool = False) -> None:
        if not accept_unverified_encoder:
            raise PermissionError(
                "Best8 is research-only: pass explicit research acknowledgement"
            )
        self.registry = json.loads(
            (BEST8_DIR / "exp0102_champion_registry.json").read_text(encoding="utf-8")
        )
        self.research = load_trusted_joblib(BEST8_DIR / "research_head_bank_RESEARCH_ONLY.joblib")
        self.v5_bundle = load_trusted_joblib(BEST8_DIR / "v5_model_bundle.joblib")
        if sha256_file(BEST8_SPM_PATH) != BEST8_SPM_SHA256:
            raise ValueError("Best8 reconstructed SentencePiece SHA256 mismatch")
        self.sentencepiece = spm.SentencePieceProcessor(
            model_proto=BEST8_SPM_PATH.read_bytes()
        )
        self.research_sha256 = sha256_file(
            BEST8_DIR / "research_head_bank_RESEARCH_ONLY.joblib"
        )
        self.v5_sha256 = sha256_file(BEST8_DIR / "v5_model_bundle.joblib")
        self.model_sha256 = hashlib.sha256(
            (self.research_sha256 + self.v5_sha256).encode("utf-8")
        ).hexdigest()
        self.thresholds = {
            label: float(self.research["heads"][label]["threshold"])
            if label in BEST8_V14_LABELS
            else float(self.v5_bundle["thresholds"][label])
            for label in LABEL_COLS
        }
        self._v5 = V5Classifier()
        self._lock = threading.RLock()

    def _embed(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        encoder = self._v5.embedding_model
        v5_embedding = np.asarray(
            encoder.encode([text], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        # V14 historical training used a reconstructed SentencePiece path at
        # 128 tokens, not the original V5 FastTokenizer path.
        token_ids = (
            [0]
            + self.sentencepiece.encode(text, out_type=int)[:126]
            + [2]
        )
        auto_model = encoder[0].auto_model
        device = next(auto_model.parameters()).device
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        # Historical V14 cache: BOS + 126 SentencePiece IDs + EOS; mean/normalize.
        # Never call SentenceTransformer.encode() for this V14 research path.
        with torch.no_grad():
            hidden = auto_model(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state
            research_embedding = hidden.mean(dim=1)
            research_embedding = torch.nn.functional.normalize(
                research_embedding, p=2, dim=1
            )
        research_embedding = np.asarray(
            research_embedding.cpu().numpy(), dtype=np.float32
        )[0]
        return research_embedding, v5_embedding

    def predict_all_dimensions(self, text: str, context: str = "") -> dict[str, float]:
        del context  # Best8 historical heads were trained on prompt-only text.
        research_embedding, v5_embedding = self._embed(text)
        word = self.research["word_vectorizer"].transform([text])
        char = self.research["char_vectorizer"].transform([text])
        lexical = hstack([word, char], format="csr")
        embedding = csr_matrix(research_embedding.astype(np.float64))
        early = hstack([lexical, 2.0 * embedding], format="csr")
        scores: dict[str, float] = {}
        for label in LABEL_COLS:
            if label in BEST8_V5_LABELS:
                value = float(
                    self.v5_bundle["estimators"][label].predict_proba(
                        np.asarray(v5_embedding, dtype=np.float64)
                    )[0, 1]
                )
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(f"invalid Best8 V5 score for {label}")
                scores[label] = value
                continue

            head = self.research["heads"][label]
            family = head["family"]
            if family == "F0_EMBED_LR":
                value = head["estimators"]["emb"].predict_proba(embedding)[0, 1]
            elif family == "F1_EARLY_LR":
                value = head["estimators"]["early"].predict_proba(early)[0, 1]
            elif family == "F2_LATE_FUSION":
                alpha = float(head["alpha"])
                value = alpha * head["estimators"]["emb"].predict_proba(embedding)[0, 1] + (
                    1.0 - alpha
                ) * head["estimators"]["lex"].predict_proba(lexical)[0, 1]
            else:
                raise ValueError(f"unknown Best8 family for {label}: {family}")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid Best8 score for {label}")
            scores[label] = value
        return scores

    def model_mapping(self) -> dict[str, dict[str, Any]]:
        return {
            label: {
                "source": (
                    "V14_HISTORICAL_RESEARCH_ONLY"
                    if label in BEST8_V14_LABELS
                    else "OFFICIAL_V5_FROZEN"
                ),
                "head_family": (
                    self.research["heads"][label]["family"]
                    if label in BEST8_V14_LABELS
                    else "V5_E5_LOGREG"
                ),
                "input": (
                    "prompt_only_reconstructed_spm_128_mean_normalize_plus_lexical"
                    if label in BEST8_V14_LABELS
                    else "prompt_text_original_length_v5_embedding"
                ),
                "threshold": float(self.thresholds[label]),
                "artifact_sha256": (
                    self.research_sha256
                    if label in BEST8_V14_LABELS
                    else self.v5_sha256
                ),
            }
            for label in LABEL_COLS
        }


class HybridH1Classifier(BaseResearchClassifier):
    model_id = "hybrid_h1"
    preprocessing_version = "per_dimension_source_specific_v1"
    encoder_status = "RESEARCH_UNVERIFIED_BEST8_ENCODER_PARITY"

    def __init__(self, *, accept_unverified_encoder: bool = False) -> None:
        self.candidate_c = CandidateCClassifier()
        self.best8 = Best8Classifier(accept_unverified_encoder=accept_unverified_encoder)
        self.thresholds = {
            label: (
                self.candidate_c.thresholds[label]
                if HYBRID_H1_SOURCE[label] == "candidate_c"
                else self.best8.thresholds[label]
            )
            for label in LABEL_COLS
        }
        self.model_sha256 = hashlib.sha256(
            (
                self.candidate_c.model_sha256 + self.best8.model_sha256
            ).encode("utf-8")
        ).hexdigest()

    def predict_all_dimensions(self, text: str, context: str = "") -> dict[str, float]:
        candidate_scores = self.candidate_c.predict_all_dimensions(text, context)
        best8_scores = self.best8.predict_all_dimensions(text, context)
        return {
            label: (
                candidate_scores[label]
                if HYBRID_H1_SOURCE[label] == "candidate_c"
                else best8_scores[label]
            )
            for label in LABEL_COLS
        }

    def model_mapping(self) -> dict[str, dict[str, Any]]:
        return {
            label: {
                "source": HYBRID_H1_SOURCE[label],
                "threshold": float(self.thresholds[label]),
                "artifact_sha256": (
                    self.candidate_c.model_sha256
                    if HYBRID_H1_SOURCE[label] == "candidate_c"
                    else self.best8.model_sha256
                ),
            }
            for label in LABEL_COLS
        }


def build_classifier(
    backend: str = "v5",
    *,
    accept_unverified_encoder: bool = False,
) -> Any:
    if backend == "v5":
        return V5Classifier()
    if backend == "candidate_c":
        return CandidateCClassifier()
    if backend == "best8":
        return Best8Classifier(accept_unverified_encoder=accept_unverified_encoder)
    if backend == "hybrid_h1":
        return HybridH1Classifier(accept_unverified_encoder=accept_unverified_encoder)
    raise ValueError(f"unknown classifier backend: {backend}")
