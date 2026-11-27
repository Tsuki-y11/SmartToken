"""Offline inference for the separately versioned v2 model artifacts."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from deep_learning.dl.features_v2 import FeatureConfigV2, encode_prompt_features
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor


class TokenPredictorV2:
    predictor_version = 'model_aware_v2'

    def __init__(self, network: ModelAwareRegressor, model_ids: Sequence[str], encoder: object,
                 *, tokenizer: object | None = None, config: FeatureConfigV2 | None = None, device: str = 'cpu'):
        if len(model_ids) != network.num_models or len(set(model_ids)) != len(model_ids):
            raise ValueError('model IDs do not match the trained network')
        self.network = network.to(device).eval()
        self.model_ids = list(model_ids)
        self.model_to_index = {name: index for index, name in enumerate(model_ids)}
        self.encoder, self.tokenizer = encoder, tokenizer
        self.config, self.device = config or FeatureConfigV2(), device

    def features(self, text: str) -> np.ndarray:
        vectors = encode_prompt_features([text], self.encoder, tokenizer=self.tokenizer, config=self.config)
        if vectors.shape != (1, self.network.embedding_dim):
            raise ValueError('v2 encoder feature dimension does not match the trained network')
        return vectors

    def predict(self, text: str, model_ids: Sequence[str] | None = None) -> dict[str, int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('prompt text must be nonempty')
        requested = self.model_ids if model_ids is None else list(model_ids)
        unknown = sorted(set(requested) - set(self.model_to_index))
        if unknown:
            raise ValueError(f'unknown model IDs: {unknown}')
        if not requested:
            return {}
        vectors = torch.as_tensor(np.repeat(self.features(text), len(requested), axis=0), device=self.device)
        indices = torch.as_tensor([self.model_to_index[name] for name in requested], dtype=torch.long, device=self.device)
        lengths = torch.full((len(requested),), float(len(text)), device=self.device)
        with torch.inference_mode():
            logs = self.network(vectors, indices, lengths)
            predictions = torch.expm1(logs.clamp(0, math.log1p(MAX_OUTPUT_TOKENS))).cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError('network returned nonfinite predictions')
        return {name: max(1, int(round(float(value)))) for name, value in zip(requested, predictions)}

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = 'cpu', encoder: object | None = None,
             tokenizer: object | None = None, local_files_only: bool = True) -> 'TokenPredictorV2':
        artifact_dir = Path(artifact_dir)
        metadata = json.loads((artifact_dir / 'model_config.json').read_text(encoding='utf-8'))
        if metadata.get('architecture') != cls.predictor_version:
            raise ValueError('saved model is not a model_aware_v2 artifact')
        config = FeatureConfigV2(**metadata['feature_config'])
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(config.encoder_id, revision=config.encoder_revision,
                                          device=device, local_files_only=local_files_only)
        network = ModelAwareRegressor(metadata['embedding_dim'], len(metadata['model_ids']), metadata['hidden_dim'])
        network.load_state_dict(torch.load(artifact_dir / 'model.pt', map_location='cpu', weights_only=True))
        return cls(network, metadata['model_ids'], encoder, tokenizer=tokenizer, config=config, device=device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--model-id', action='append', dest='model_ids')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    print(json.dumps(TokenPredictorV2.load(args.artifact_dir, device=args.device).predict(args.text, args.model_ids), indent=2))
