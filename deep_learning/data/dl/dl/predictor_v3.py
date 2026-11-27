"""Opt-in v3 inference with capability features and strict artifact metadata."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from deep_learning.dl.dimensions import LABEL_FEATURE_NAMES, normalize_annotation, validate_label_schema
from deep_learning.dl.features_v2 import FeatureConfigV2
from deep_learning.dl.features_v3 import encode_labeled_features
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor
from deep_learning.dl.predictor_v2 import TokenPredictorV2


def load_configuration(artifact_dir: str | Path) -> dict:
    metadata = json.loads((Path(artifact_dir) / 'model_config.json').read_text(encoding='utf-8'))
    if metadata.get('architecture') != 'model_aware_v3':
        raise ValueError('saved model is not a model_aware_v3 artifact')
    validate_label_schema(metadata.get('label_schema'))
    if type(metadata.get('use_label_features')) is not bool:
        raise ValueError('v3 artifact requires a boolean use_label_features policy')
    expected = len(LABEL_FEATURE_NAMES) if metadata['use_label_features'] else 0
    if metadata.get('label_dimensions') != expected:
        raise ValueError('v3 label schema feature dimensions do not match ablation policy')
    if metadata.get('embedding_dim') != metadata.get('text_feature_dimensions', -1) + expected:
        raise ValueError('v3 feature dimensions do not match saved schema')
    FeatureConfigV2(**metadata['feature_config'])
    return metadata


class TokenPredictorV3(TokenPredictorV2):
    predictor_version = 'model_aware_v3'

    def __init__(self, network: ModelAwareRegressor, model_ids: Sequence[str], encoder: object,
                 *, tokenizer: object | None = None, config: FeatureConfigV2 | None = None,
                 device: str = 'cpu', use_label_features: bool = True):
        super().__init__(network, model_ids, encoder, tokenizer=tokenizer, config=config, device=device)
        if type(use_label_features) is not bool:
            raise ValueError('use_label_features must be boolean')
        self.use_label_features = use_label_features

    def features(self, text: str, *, dimensions=None, label_source=None, dimension_mask=None) -> np.ndarray:
        vectors = encode_labeled_features([text], self.encoder, tokenizer=self.tokenizer,
                                          config=self.config, dimensions=[dimensions],
                                          label_sources=[label_source], dimension_masks=[dimension_mask],
                                          use_label_features=self.use_label_features)
        if vectors.shape != (1, self.network.embedding_dim):
            raise ValueError('v3 encoder feature dimension does not match the trained network')
        return vectors

    def predict(self, text: str, model_ids: Sequence[str] | None = None, *,
                dimensions=None, label_source=None, dimension_mask=None) -> dict[str, int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError('prompt text must be nonempty')
        normalize_annotation(dimensions, label_source, dimension_mask=dimension_mask)
        if isinstance(model_ids, (str, bytes)):
            raise ValueError('model IDs must be a sequence')
        requested = self.model_ids if model_ids is None else list(model_ids)
        if any(not isinstance(name, str) for name in requested):
            raise ValueError('model IDs must be strings')
        unknown = sorted(set(requested) - set(self.model_to_index))
        if unknown:
            raise ValueError(f'unknown model IDs: {unknown}')
        if not requested:
            return {}
        features = self.features(text, dimensions=dimensions, label_source=label_source,
                                 dimension_mask=dimension_mask)
        vectors = torch.as_tensor(np.repeat(features, len(requested), axis=0), device=self.device)
        indices = torch.as_tensor([self.model_to_index[name] for name in requested], dtype=torch.long,
                                  device=self.device)
        lengths = torch.full((len(requested),), float(len(text)), device=self.device)
        with torch.inference_mode():
            logs = self.network(vectors, indices, lengths)
            predictions = torch.expm1(logs.clamp(0, math.log1p(MAX_OUTPUT_TOKENS))).cpu().numpy()
        if not np.isfinite(predictions).all():
            raise ValueError('network returned nonfinite predictions')
        return {name: max(1, int(round(float(value)))) for name, value in zip(requested, predictions)}

    @classmethod
    def load(cls, artifact_dir: str | Path, device: str = 'cpu', encoder: object | None = None,
             tokenizer: object | None = None, local_files_only: bool = True) -> 'TokenPredictorV3':
        metadata = load_configuration(artifact_dir)
        config = FeatureConfigV2(**metadata['feature_config'])
        if encoder is None:
            from sentence_transformers import SentenceTransformer
            encoder = SentenceTransformer(config.encoder_id, revision=config.encoder_revision,
                                          device=device, local_files_only=local_files_only)
        network = ModelAwareRegressor(metadata['embedding_dim'], len(metadata['model_ids']), metadata['hidden_dim'])
        network.load_state_dict(torch.load(Path(artifact_dir) / 'model.pt', map_location='cpu', weights_only=True))
        return cls(network, metadata['model_ids'], encoder, tokenizer=tokenizer, config=config, device=device,
                   use_label_features=metadata['use_label_features'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--model-id', action='append', dest='model_ids')
    parser.add_argument('--dimensions', type=json.loads, help='JSON array; omit for absent labels')
    parser.add_argument('--label-source', choices=('human', 'ml', 'dataset_proxy', 'missing'))
    parser.add_argument('--dimension-mask', type=json.loads, help='JSON array of known dimensions')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    predictor = TokenPredictorV3.load(args.artifact_dir, device=args.device)
    print(json.dumps(predictor.predict(args.text, args.model_ids, dimensions=args.dimensions,
                                       label_source=args.label_source, dimension_mask=args.dimension_mask), indent=2))
