"""Offline inference: one prompt to separate estimates for selected models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from deep_learning.dl.features import FEATURE_STRATEGY, TEXT_LIMIT, encode_prompt_vectors
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor


class TokenPredictor:
    def __init__(self, network: ModelAwareRegressor, model_ids: Sequence[str], encoder: object, device: str = "cpu"):
        if len(model_ids) != network.num_models or len(set(model_ids)) != len(model_ids):
            raise ValueError("model IDs do not match the trained network")
        self.network = network.to(device).eval()
        self.model_ids = list(model_ids)
        self.model_to_index = {name: index for index, name in enumerate(self.model_ids)}
        self.encoder = encoder
        self.device = device

    def predict(self, text: str, model_ids: Sequence[str] | None = None) -> dict[str, int]:
        """Estimate output tokens separately for every requested model ID."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prompt text must be nonempty")
        requested = self.model_ids if model_ids is None else list(model_ids)
        unknown = sorted(set(requested) - set(self.model_to_index))
        if unknown:
            raise ValueError(f"unknown model IDs: {unknown}")
        if not requested:
            return {}
        vectors = encode_prompt_vectors([text], self.encoder)
        if vectors.shape != (1, self.network.embedding_dim):
            raise ValueError("encoder dimension does not match the trained network")
        repeated = torch.as_tensor(np.repeat(vectors, len(requested), axis=0), device=self.device)
        indices = torch.as_tensor([self.model_to_index[name] for name in requested], dtype=torch.long, device=self.device)
        lengths = torch.full((len(requested),), float(len(text)), device=self.device)
        with torch.inference_mode():
            log_predictions = self.network(repeated, indices, lengths).cpu().numpy()
        values = np.expm1(np.clip(log_predictions, 0, math.log1p(MAX_OUTPUT_TOKENS)))
        return {name: max(1, int(round(float(value)))) for name, value in zip(requested, values)}

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = "cpu", encoder: object | None = None) -> "TokenPredictor":
        """Load a locally saved model; the caller may inject a preloaded encoder."""
        artifact_dir = Path(artifact_dir)
        metadata = json.loads((artifact_dir / "model_config.json").read_text(encoding="utf-8"))
        if metadata.get("feature_strategy") != FEATURE_STRATEGY or metadata.get("text_limit") != TEXT_LIMIT:
            raise ValueError("saved model uses a different prompt encoding strategy")
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(metadata["encoder_id"], device=device,
                                          revision=metadata["encoder_revision"])
        network = ModelAwareRegressor(
            embedding_dim=metadata["embedding_dim"],
            num_models=len(metadata["model_ids"]),
            hidden_dim=metadata["hidden_dim"],
        )
        state = torch.load(artifact_dir / "model.pt", map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        return cls(network, metadata["model_ids"], encoder, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    predictor = TokenPredictor.load(args.artifact_dir, device=args.device)
    print(json.dumps(predictor.predict(args.text, args.model_ids), indent=2, ensure_ascii=False))
