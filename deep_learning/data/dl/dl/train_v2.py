"""Train on train/valid only; evaluate a selected saved v2 artifact explicitly."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from deep_learning.dl.features_v2 import (FeatureConfigV2, STRUCTURE_FEATURE_NAMES, build_feature_cache,
                            load_feature_cache)
from deep_learning.dl.metrics import evaluate_predictions
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor
from deep_learning.dl.predictor_v2 import TokenPredictorV2
from deep_learning.dl.train import validate_question_splits


def mixed_token_loss(predicted_log: torch.Tensor, target_log: torch.Tensor,
                     raw_loss_weight: float = 0.0) -> torch.Tensor:
    """Log Huber control plus optional original-token MAE divided by 1000."""
    if not math.isfinite(raw_loss_weight) or raw_loss_weight < 0:
        raise ValueError('raw_loss_weight must be finite and nonnegative')
    loss = F.smooth_l1_loss(predicted_log, target_log, beta=0.5)
    if raw_loss_weight:
        original = torch.expm1(predicted_log.clamp(0, math.log1p(MAX_OUTPUT_TOKENS)))
        loss = loss + raw_loss_weight * torch.abs(original - torch.expm1(target_log)).mean() / 1000.0
    return loss


def _device(device: str) -> str:
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device not in ('cpu', 'cuda'):
        raise ValueError('device must be auto, cpu, or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return device


def _read_data(data_dir: Path):
    rows = pd.read_parquet(data_dir / 'records.parquet')
    catalog = json.loads((data_dir / 'catalog.json').read_text(encoding='utf-8'))
    model_ids = [item['model_id'] for item in catalog]
    if not model_ids or len(set(model_ids)) != len(model_ids):
        raise ValueError('catalog must contain distinct model IDs')
    if not {'train', 'valid'}.issubset(set(rows['split'])):
        raise ValueError('train and validation rows are required')
    if not set(rows['split']).issubset({'train', 'valid', 'test'}):
        raise ValueError('unknown split labels in records')
    if set(rows['model_id']) - set(model_ids):
        raise ValueError('records contain model IDs outside the catalog')
    if rows['prompt'].map(lambda text: not isinstance(text, str) or not text.strip()).any():
        raise ValueError('records contain empty or invalid prompts')
    validate_question_splits(rows)
    return rows, catalog, model_ids


def prepare_feature_cache(data_dir: str | Path, cache_dir: str | Path, *, device: str = 'auto',
                          encoder: object | None = None, tokenizer: object | None = None,
                          config: FeatureConfigV2 | None = None, batch_size: int = 128):
    config = config or FeatureConfigV2()
    rows, _catalog, _model_ids = _read_data(Path(data_dir))
    prompts = rows['prompt'].tolist()
    if (Path(cache_dir) / 'metadata.json').exists():
        return load_feature_cache(prompts, cache_dir, config=config)
    if encoder is None:
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer(config.encoder_id, revision=config.encoder_revision, device=_device(device),
                                      local_files_only=True)
    return build_feature_cache(prompts, encoder, cache_dir, tokenizer=tokenizer, config=config, batch_size=batch_size)


class _FeatureRows(Dataset):
    """Gather cached unique features per batch without a multi-GB row-expanded copy."""
    def __init__(self, rows: pd.DataFrame, features: np.ndarray, row_index: np.ndarray,
                 model_ids: list[str], mask: np.ndarray):
        mapping = {name: index for index, name in enumerate(model_ids)}
        selected = rows.loc[mask]
        targets = selected['output_tokens'].to_numpy(dtype=np.float32)
        if not np.isfinite(targets).all() or (targets <= 0).any():
            raise ValueError('output token labels must be finite and positive')
        self.features = features
        self.row_index = row_index[mask]
        self.model_indices = selected['model_id'].map(mapping).to_numpy(dtype=np.int64)
        self.lengths = selected['prompt'].str.len().to_numpy(dtype=np.float32)
        self.targets_log = np.log1p(targets)

    def __len__(self):
        return len(self.row_index)

    def __getitem__(self, index):
        return (np.array(self.features[self.row_index[index]], dtype=np.float32, copy=True),
                self.model_indices[index], self.lengths[index], self.targets_log[index])


def _loader(dataset: _FeatureRows, batch_size: int, shuffle: bool, device: str) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=device == 'cuda')


def _predict_original(network: ModelAwareRegressor, loader: DataLoader, device: str) -> np.ndarray:
    network.eval()
    chunks = []
    with torch.inference_mode():
        for vectors, models, lengths, _targets in loader:
            logs = network(vectors.to(device), models.to(device), lengths.to(device))
            values = torch.expm1(logs.clamp(0, math.log1p(MAX_OUTPUT_TOKENS)))
            chunks.append(values.round().clamp(min=1).cpu().numpy())
    predictions = np.concatenate(chunks).astype(np.float64)
    if not np.isfinite(predictions).all():
        raise RuntimeError('nonfinite evaluation predictions')
    return predictions


def _file_hash(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return 'unknown'


def train_predictor_v2(
    data_dir: str | Path, output_dir: str | Path, *, cache_dir: str | Path,
    epochs: int = 12, patience: int = 3, lr: float = 0.001, hidden_dim: int = 128,
    raw_loss_weight: float = 0.0, seed: int = 42, device: str = 'auto', batch_size: int = 1024,
    config: FeatureConfigV2 | None = None, encoder: object | None = None, tokenizer: object | None = None,
) -> dict:
    """Select checkpoint by validation MAE in tokens. This never evaluates test."""
    if min(epochs, patience, batch_size, hidden_dim) < 1 or not math.isfinite(lr) or lr <= 0:
        raise ValueError('epochs, patience, batch size, hidden size, and lr must be positive')
    if not math.isfinite(raw_loss_weight) or raw_loss_weight < 0:
        raise ValueError('raw_loss_weight must be finite and nonnegative')
    data_dir, output_dir, cache_dir = Path(data_dir), Path(output_dir), Path(cache_dir)
    # Refuse to mix old test results/checkpoints into a new candidate run.
    if any((output_dir / name).exists() for name in ('model.pt', 'model_config.json', 'test_metrics.json')):
        raise ValueError('output directory already contains an artifact; use a new candidate directory')
    config = config or FeatureConfigV2()
    device = _device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    rows, catalog, model_ids = _read_data(data_dir)
    train_mask = rows['split'].eq('train').to_numpy()
    valid_mask = rows['split'].eq('valid').to_numpy()
    if set(rows.loc[train_mask, 'model_id']) != set(model_ids):
        raise ValueError('each model requires training rows')
    features, row_index = prepare_feature_cache(data_dir, cache_dir, device=device, encoder=encoder,
                                              tokenizer=tokenizer, config=config)
    train_set = _FeatureRows(rows, features, row_index, model_ids, train_mask)
    valid_set = _FeatureRows(rows, features, row_index, model_ids, valid_mask)
    train_loader = _loader(train_set, batch_size, True, device)
    valid_loader = _loader(valid_set, batch_size, False, device)
    network = ModelAwareRegressor(features.shape[1], len(model_ids), hidden_dim=hidden_dim).to(device)
    with torch.no_grad():
        for index in range(len(model_ids)):
            network.model_head.weight[index, -1] = float(train_set.targets_log[train_set.model_indices == index].mean())
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    best_mae, best_epoch, best_state, stale = math.inf, 0, None, 0
    history = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        network.train()
        total, count = 0.0, 0
        for vectors, models, lengths, target_log in train_loader:
            vectors, models, lengths, target_log = (value.to(device) for value in (vectors, models, lengths, target_log))
            optimizer.zero_grad(set_to_none=True)
            loss = mixed_token_loss(network(vectors, models, lengths), target_log, raw_loss_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=10.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(vectors)
            count += len(vectors)
        predictions = _predict_original(network, valid_loader, device)
        mae = float(np.abs(predictions - rows.loc[valid_mask, 'output_tokens'].to_numpy(dtype=np.float64)).mean())
        history.append({'epoch': epoch, 'train_loss': total / count, 'validation_mae': mae})
        (output_dir / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(history[-1]), flush=True)
        if mae < best_mae:
            best_mae, best_epoch, stale = mae, epoch, 0
            best_state = copy.deepcopy({name: value.detach().cpu() for name, value in network.state_dict().items()})
            torch.save(best_state, output_dir / 'model.pt')
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    network.load_state_dict(best_state)
    cache_metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    configuration = {
        'architecture': 'model_aware_v2', 'feature_config': asdict(config),
        'encoder_id': config.encoder_id, 'encoder_revision': config.encoder_revision,
        'embedding_dim': int(features.shape[1]), 'semantic_dimensions': cache_metadata['semantic_dimensions'],
        'structure_dimensions': len(STRUCTURE_FEATURE_NAMES), 'structure_feature_names': list(STRUCTURE_FEATURE_NAMES),
        'feature_cache_fingerprint': cache_metadata['fingerprint'], 'model_ids': model_ids,
        'hidden_dim': hidden_dim, 'lr': lr, 'raw_loss_weight': raw_loss_weight,
        'seed': seed, 'batch_size': batch_size, 'max_epochs': epochs, 'patience': patience,
        'best_epoch': best_epoch, 'best_validation_mae': best_mae,
        'checkpoint_selection': 'minimum_validation_mae_original_tokens_rounded_minimum_1',
        'dimension_signal': 'none; caller-provided dimension is grouping metadata only',
        'records_sha256': _file_hash(data_dir / 'records.parquet'), 'catalog_sha256': _file_hash(data_dir / 'catalog.json'),
        'versions': {name: _version(name) for name in ('torch', 'numpy', 'pandas', 'sentence-transformers', 'transformers')},
    }
    (output_dir / 'model_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    valid_rows = rows.loc[valid_mask, ['source', 'dataset', 'original_query', 'model_id', 'output_tokens']].reset_index(drop=True)
    predictions = _predict_original(network, valid_loader, device)
    report = evaluate_predictions(valid_rows, predictions, catalog=catalog)
    report.update(best_epoch=best_epoch, best_validation_mae=best_mae)
    (output_dir / 'validation_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    valid_rows.assign(predicted_output_tokens=predictions).to_csv(output_dir / 'validation_predictions.csv', index=False)
    return report


def evaluate_saved_predictor(data_dir: str | Path, artifact_dir: str | Path, *, cache_dir: str | Path,
                             device: str = 'auto', encoder: object | None = None,
                             tokenizer: object | None = None, batch_size: int = 1024) -> dict:
    """Explicit held-out evaluation of a selected checkpoint, never called by training."""
    artifact_dir = Path(artifact_dir)
    if (artifact_dir / 'test_metrics.json').exists() or (artifact_dir / 'test_predictions.csv').exists():
        raise ValueError('test results already exist; this artifact has already been evaluated')
    device = _device(device)
    configuration = json.loads((artifact_dir / 'model_config.json').read_text(encoding='utf-8'))
    if configuration.get('architecture') != 'model_aware_v2':
        raise ValueError('saved model is not a model_aware_v2 artifact')
    config = FeatureConfigV2(**configuration['feature_config'])
    rows, catalog, model_ids = _read_data(Path(data_dir))
    if model_ids != configuration['model_ids']:
        raise ValueError('evaluation catalog order differs from trained model IDs')
    expected = configuration.get('records_sha256')
    if expected is not None and _file_hash(Path(data_dir) / 'records.parquet') != expected:
        raise ValueError('evaluation records differ from the saved training data')
    features, row_index = load_feature_cache(rows['prompt'].tolist(), cache_dir, config=config)
    if configuration.get('feature_cache_fingerprint') != json.loads((Path(cache_dir) / 'metadata.json').read_text(encoding='utf-8'))['fingerprint']:
        raise ValueError('evaluation cache differs from the saved training features')
    test_mask = rows['split'].eq('test').to_numpy()
    if not test_mask.any():
        raise ValueError('test rows are required for explicit held-out evaluation')
    # Reload the saved network independently; an encoder is only needed for
    # arbitrary prompt inference, while cached test features need no encoder.
    network = ModelAwareRegressor(configuration['embedding_dim'], len(model_ids), configuration['hidden_dim']).to(device)
    network.load_state_dict(torch.load(artifact_dir / 'model.pt', map_location='cpu', weights_only=True))
    dataset = _FeatureRows(rows, features, row_index, model_ids, test_mask)
    predictions = _predict_original(network, _loader(dataset, batch_size, False, device), device)
    test_rows = rows.loc[test_mask, ['source', 'dataset', 'original_query', 'model_id', 'output_tokens']].reset_index(drop=True)
    report = evaluate_predictions(test_rows, predictions, catalog=catalog)
    (artifact_dir / 'test_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    test_rows.assign(predicted_output_tokens=predictions).to_csv(artifact_dir / 'test_predictions.csv', index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--raw-loss-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=1024)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if args.prepare_only:
        features, index = prepare_feature_cache(args.data_dir, args.cache_dir, device=args.device)
        print(json.dumps({'unique_prompts': len(features), 'rows': len(index), 'dimensions': features.shape[1]}))
    elif args.evaluate_only:
        report = evaluate_saved_predictor(args.data_dir, args.output_dir, cache_dir=args.cache_dir,
                                         device=args.device, batch_size=args.batch_size)
        print(json.dumps(report['overall'], indent=2))
    else:
        report = train_predictor_v2(args.data_dir, args.output_dir, cache_dir=args.cache_dir, epochs=args.epochs,
                                    patience=args.patience, lr=args.lr, hidden_dim=args.hidden_dim,
                                    raw_loss_weight=args.raw_loss_weight, seed=args.seed, device=args.device,
                                    batch_size=args.batch_size)
        print(json.dumps(report['overall'], indent=2))


if __name__ == '__main__':
    main()
