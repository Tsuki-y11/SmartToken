"""Frozen sentence embeddings shared by the XGBoost and DL experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


DEFAULT_ENCODER = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_ENCODER_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
SEGMENT_CHARS = 96
FEATURE_STRATEGY = "first_last_96_chars_separate_v1"
TEXT_LIMIT = SEGMENT_CHARS * 2


def shorten_prompt(text: str, limit: int = 1024) -> str:
    """Retain both ends of long prompts for a bounded sentence encoder input."""
    if limit < 2:
        raise ValueError("limit must be at least 2")
    if len(text) <= limit:
        return text
    first = limit // 2
    return text[:first] + "\n...\n" + text[-(limit - first):]


def encode_prompt_vectors(
    prompts: Sequence[str], encoder: object, batch_size: int = 128, show_progress_bar: bool = False,
    segment_chars: int = SEGMENT_CHARS,
) -> np.ndarray:
    """Encode both ends separately so the encoder's 128-token limit retains the tail."""
    segments = []
    counts = []
    for prompt in prompts:
        parts = [prompt] if len(prompt) <= segment_chars else [prompt[:segment_chars], prompt[-segment_chars:]]
        segments.extend(parts)
        counts.append(len(parts))
    vectors = np.asarray(encoder.encode(
        segments, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=show_progress_bar), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(segments):
        raise ValueError("sentence encoder returned an invalid embedding matrix")
    result = np.empty((len(prompts), vectors.shape[1]), dtype=np.float32)
    offset = 0
    for index, count in enumerate(counts):
        vector = vectors[offset:offset + count].mean(axis=0)
        if count > 1:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector /= norm
        result[index] = vector
        offset += count
    return result


def _fingerprint(prompts: Sequence[str], encoder_id: str, limit: int, encoder_revision: str) -> str:
    digest = hashlib.sha256(f"{encoder_id}\0{encoder_revision}\0{limit}\0{FEATURE_STRATEGY}\0".encode())
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def load_embedding_cache(
    prompts: Sequence[str], output_dir: str | Path, encoder_id: str, text_limit: int = TEXT_LIMIT,
    encoder_revision: str = DEFAULT_ENCODER_REVISION,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject a stale cache even if its row count happens to be unchanged."""
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
    fingerprint = _fingerprint(prompts, encoder_id, text_limit, encoder_revision)
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("embedding cache does not match prompts, order, encoder, or text limit")
    return np.load(output_dir / "embeddings.npy"), np.load(output_dir / "row_index.npy")


def build_embedding_cache(
    prompts: Sequence[str],
    encoder: object,
    output_dir: str | Path,
    batch_size: int = 128,
    encoder_id: str | None = None,
    text_limit: int = TEXT_LIMIT,
    encoder_revision: str = DEFAULT_ENCODER_REVISION,
) -> tuple[np.ndarray, np.ndarray]:
    """Encode unique rendered prompts once and map results back to row order."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts = list(prompts)
    if not prompts or any(not isinstance(prompt, str) for prompt in prompts):
        raise ValueError("prompts must be a nonempty sequence of strings")
    if text_limit < 2:
        raise ValueError("text limit must be at least 2")
    encoder_id = encoder_id or encoder.__class__.__name__
    fingerprint = _fingerprint(prompts, encoder_id, text_limit, encoder_revision)
    metadata_path = output_dir / "metadata.json"
    embedding_path = output_dir / "embeddings.npy"
    index_path = output_dir / "row_index.npy"
    if metadata_path.exists() and embedding_path.exists() and index_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return np.load(embedding_path), np.load(index_path)
    row_index, unique_prompts = pd.factorize(pd.Series(prompts), sort=False)
    embeddings = encode_prompt_vectors(
        unique_prompts, encoder, batch_size=batch_size, show_progress_bar=True,
        segment_chars=min(SEGMENT_CHARS, text_limit // 2))
    if embeddings.ndim != 2 or len(embeddings) != len(unique_prompts):
        raise ValueError("sentence encoder returned an invalid embedding matrix")
    row_index = np.asarray(row_index, dtype=np.int32)
    np.save(embedding_path, embeddings)
    np.save(index_path, row_index)
    metadata_path.write_text(
        json.dumps({"fingerprint": fingerprint, "encoder_id": encoder_id, "encoder_revision": encoder_revision,
                    "text_limit": text_limit, "feature_strategy": FEATURE_STRATEGY,
                    "row_count": len(prompts), "unique_prompts": len(unique_prompts), "dimensions": embeddings.shape[1]}, indent=2),
        encoding="utf-8",
    )
    return embeddings, row_index
