"""Precompute frozen text features without fitting any prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from deep_learning.dl.features import DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION, build_embedding_cache


def prepare_features(
    data_dir: str | Path,
    cache_dir: str | Path,
    *,
    encoder: object | None = None,
    encoder_id: str = DEFAULT_ENCODER,
    encoder_revision: str = DEFAULT_ENCODER_REVISION,
    device: str = "cpu",
) -> dict[str, int]:
    """Cache each unique rendered prompt once; leave labels and splits untouched."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(encoder_id, device=device, revision=encoder_revision)
    prompts = pd.read_parquet(data_dir / "records.parquet", columns=["prompt"])["prompt"].tolist()
    embeddings, row_index = build_embedding_cache(
        prompts, encoder, cache_dir, encoder_id=encoder_id, encoder_revision=encoder_revision)
    return {"rows": len(row_index), "unique_prompts": len(embeddings), "dimensions": embeddings.shape[1]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--encoder-id", default=DEFAULT_ENCODER)
    parser.add_argument("--encoder-revision", default=DEFAULT_ENCODER_REVISION)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    print(json.dumps(prepare_features(args.data_dir, args.cache_dir, encoder_id=args.encoder_id,
                                     encoder_revision=args.encoder_revision, device=args.device)))
