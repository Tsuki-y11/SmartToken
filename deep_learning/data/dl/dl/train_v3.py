"""Train opt-in v3 on real token targets; explicitly evaluate a selected artifact.

Label-free ablation uses exactly the same records, split masks and random seed.
Text cache reuse is safe: annotation features are gathered per record at batch
time, never deduplicated by prompt. Public unannotated rows remain unannotated.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import unicodedata
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep_learning.dl.dimensions import (LABELS, LABEL_FEATURE_NAMES, LABEL_SOURCES, label_schema,
                           validate_label_policy_version)
from deep_learning.dl.features_v2 import FeatureConfigV2, STRUCTURE_FEATURE_NAMES, load_feature_cache
from deep_learning.dl.features_v3 import annotation_features
from deep_learning.dl.metrics import evaluate_predictions
from deep_learning.dl.model import ModelAwareRegressor
from deep_learning.dl.predictor_v3 import load_configuration
from deep_learning.dl.train_v2 import (_FeatureRows, _device, _file_hash, _loader, _predict_original,
                         _read_data as _read_v2_data, _version, mixed_token_loss,
                         prepare_feature_cache as _prepare_text_cache)


def _annotations(rows: pd.DataFrame) -> np.ndarray:
    def values(field):
        if field not in rows:
            return None
        # Pandas 3 string columns decode parquet null as float NaN. Preserve
        # null/default provenance for absent annotations at this storage boundary;
        # the public encoder still rejects numeric NaN supplied as provenance.
        column = rows[field]
        if field == 'label_source':
            column = column.astype(object).where(column.notna(), None)
        return column.tolist()
    return annotation_features(len(rows), dimensions=values('dimensions'),
                               label_sources=values('label_source'), dimension_masks=values('dimension_mask'))


def _read_data(data_dir: Path):
    rows, catalog, model_ids = _read_v2_data(data_dir)
    if 'label_policy_version' in rows:
        policies = rows['label_policy_version'].astype(object).where(rows['label_policy_version'].notna(), None)
        for policy in policies:
            validate_label_policy_version(policy)
    parent_fields = ('group_id', 'request_id', 'source_prompt', 'source_example_id')
    dependent = np.zeros(len(rows), dtype=bool)
    for field in ('depends_on', 'input_refs'):
        if field not in rows:
            continue

        def needs_parent(value):
            if value is None or (np.isscalar(value) and pd.isna(value)):
                return False
            if not isinstance(value, (list, tuple, np.ndarray)):
                raise ValueError(f'{field} must contain lists when supplied')
            if field == 'depends_on':
                return bool(len(value))
            return any(isinstance(reference, str) and reference.endswith('.output') for reference in value)

        dependent |= rows[field].map(needs_parent).to_numpy(dtype=bool)
    has_parent = np.zeros(len(rows), dtype=bool)
    for field in parent_fields:
        if field in rows:
            has_parent |= rows[field].notna().to_numpy()
    if (dependent & ~has_parent).any():
        raise ValueError('dependent records require a whole-example parent group_id, request_id, source_prompt or source_example_id')
    # Different dependent tasks may have different original_query fields while
    # still belonging to one request. Honor every available parent identifier.
    for field in parent_fields + ('original_query', 'prompt'):
        if field not in rows:
            continue
        selected = rows.loc[rows[field].notna(), [field, 'split']].copy()
        if selected[field].map(lambda value: not isinstance(value, str) or not value.strip()).any():
            raise ValueError(f'{field} grouping values must be nonempty strings when provided')
        keys = selected[field].map(lambda value: ' '.join(unicodedata.normalize('NFKC', value).casefold().split()))
        if (selected.groupby(keys)['split'].nunique() > 1).any():
            raise ValueError(f'{field} group appears in multiple splits')
    try:
        targets = rows['output_tokens'].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError('output token targets must be numeric') from exc
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError('output token targets must be finite and positive')
    annotations = _annotations(rows)  # Validate all provenance before cache/network operations.
    return rows, catalog, model_ids, annotations


def _annotation_metadata(rows: pd.DataFrame, annotations: np.ndarray) -> dict:
    source_counts, known_counts, split_hashes = {}, {}, {}
    for split in ('train', 'valid', 'test'):
        mask = rows['split'].eq(split).to_numpy()
        source_counts[split] = {source: int(annotations[mask, -len(LABEL_SOURCES) + index].sum())
                                for index, source in enumerate(LABEL_SOURCES)}
        known_counts[split] = {label: int(annotations[mask, len(LABELS) + index].sum())
                               for index, label in enumerate(LABELS)}
        digest = hashlib.sha256()
        selected = rows.loc[mask]
        for start in range(0, len(selected), 1024):
            encoded = selected.iloc[start:start + 1024].to_json(orient='records').encode('utf-8')
            digest.update(len(encoded).to_bytes(8, 'big'))
            digest.update(encoded)
        split_hashes[split] = digest.hexdigest()
    counts = source_counts['train']
    return {'label_source_counts': source_counts, 'dimension_known_counts': known_counts,
            'split_records_sha256': split_hashes,
            'annotation_sha256': hashlib.sha256(annotations.tobytes()).hexdigest(),
            'label_evidence': {'has_labeled_training_rows': sum(counts[source] for source in LABEL_SOURCES[:-1]) > 0,
                               'human_training_rows': counts['human'],
                               'ml_predicted_training_rows': counts['ml'],
                               'dataset_proxy_training_rows': counts['dataset_proxy'],
                               'missing_training_rows': counts['missing'],
                               'label_effectiveness_established': False,
                               'interpretation': 'source counts disclose availability, not label quality or benefit'}}


class _LabeledRows(_FeatureRows):
    def __init__(self, rows, features, row_index, model_ids, mask, annotations, use_label_features):
        super().__init__(rows, features, row_index, model_ids, mask)
        self.annotations = annotations[mask] if use_label_features else None

    def __getitem__(self, index):
        vectors, model, length, target = super().__getitem__(index)
        if self.annotations is not None:
            vectors = np.concatenate((vectors, self.annotations[index]))
        return vectors, model, length, target


def _write_report(rows, mask, predictions, catalog, output_dir, name, annotation_metadata):
    selected = rows.loc[mask].reset_index(drop=True)
    report = evaluate_predictions(selected, predictions, catalog=catalog)
    report.update({key: annotation_metadata[key] for key in ('label_source_counts', 'dimension_known_counts', 'label_evidence')})
    (output_dir / f'{name}_metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    selected.assign(predicted_output_tokens=predictions).to_csv(output_dir / f'{name}_predictions.csv', index=False)
    return report


def prepare_feature_cache(data_dir: str | Path, cache_dir: str | Path, **kwargs):
    _read_data(Path(data_dir))
    return _prepare_text_cache(data_dir, cache_dir, **kwargs)


def train_predictor_v3(data_dir: str | Path, output_dir: str | Path, *, cache_dir: str | Path,
                       epochs: int = 12, patience: int = 3, lr: float = 0.001, hidden_dim: int = 128,
                       raw_loss_weight: float = 0.0, seed: int = 42, device: str = 'auto', batch_size: int = 1024,
                       config: FeatureConfigV2 | None = None, encoder: object | None = None,
                       tokenizer: object | None = None, use_label_features: bool = True) -> dict:
    """Select by validation token MAE only. Training never evaluates test rows."""
    for name, value in (('epochs', epochs), ('patience', patience), ('batch_size', batch_size),
                        ('hidden_dim', hidden_dim)):
        if type(value) is not int or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(raw_loss_weight) or raw_loss_weight < 0:
        raise ValueError('lr must be positive and loss weight nonnegative, both finite')
    if type(use_label_features) is not bool:
        raise ValueError('use_label_features must be boolean')
    data_dir, output_dir, cache_dir = Path(data_dir), Path(output_dir), Path(cache_dir)
    if any((output_dir / name).exists() for name in ('model.pt', 'model_config.json', 'test_metrics.json')):
        raise ValueError('output directory already contains an artifact; use a new candidate directory')
    rows, catalog, model_ids, annotations = _read_data(data_dir)
    train_mask, valid_mask = (rows['split'].eq(split).to_numpy() for split in ('train', 'valid'))
    if set(rows.loc[train_mask, 'model_id']) != set(model_ids):
        raise ValueError('each model requires training rows')
    config, device = config or FeatureConfigV2(), _device(device)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    features, row_index = _prepare_text_cache(data_dir, cache_dir, encoder=encoder, tokenizer=tokenizer,
                                             config=config, device=device)
    datasets = [_LabeledRows(rows, features, row_index, model_ids, mask, annotations, use_label_features)
                for mask in (train_mask, valid_mask)]
    train_set, valid_set = datasets
    train_loader, valid_loader = (_loader(dataset, batch_size, shuffle, device)
                                  for dataset, shuffle in ((train_set, True), (valid_set, False)))
    label_dimensions = len(LABEL_FEATURE_NAMES) if use_label_features else 0
    network = ModelAwareRegressor(features.shape[1] + label_dimensions, len(model_ids), hidden_dim).to(device)
    with torch.no_grad():
        for index in range(len(model_ids)):
            network.model_head.weight[index, -1] = float(train_set.targets_log[train_set.model_indices == index].mean())
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=1e-4)
    best_mae, best_epoch, best_state, stale, history = math.inf, 0, None, 0, []
    output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        network.train()
        total, count = 0.0, 0
        for values in train_loader:
            vectors, models, lengths, targets = (value.to(device) for value in values)
            optimizer.zero_grad(set_to_none=True)
            loss = mixed_token_loss(network(vectors, models, lengths), targets, raw_loss_weight)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(vectors)
            count += len(vectors)
        predictions = _predict_original(network, valid_loader, device)
        mae = float(np.abs(predictions - rows.loc[valid_mask, 'output_tokens'].to_numpy(dtype=np.float64)).mean())
        history.append({'epoch': epoch, 'train_loss': total / count, 'validation_mae': mae})
        print(json.dumps(history[-1]), flush=True)
        if mae < best_mae:
            best_mae, best_epoch, stale = mae, epoch, 0
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in network.state_dict().items()})
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError('training did not produce a checkpoint')
    network.load_state_dict(best_state)
    annotation_metadata = _annotation_metadata(rows, annotations)
    cache_metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    configuration = {
        'architecture': 'model_aware_v3', 'feature_config': asdict(config), 'label_schema': label_schema(),
        'use_label_features': use_label_features, 'label_dimensions': label_dimensions,
        'embedding_dim': features.shape[1] + label_dimensions, 'text_feature_dimensions': features.shape[1],
        'semantic_dimensions': cache_metadata['semantic_dimensions'],
        'structure_dimensions': len(STRUCTURE_FEATURE_NAMES), 'structure_feature_names': list(STRUCTURE_FEATURE_NAMES),
        'feature_cache_fingerprint': cache_metadata['fingerprint'], 'model_ids': model_ids,
        'hidden_dim': hidden_dim, 'lr': lr, 'raw_loss_weight': raw_loss_weight,
        'seed': seed, 'batch_size': batch_size, 'max_epochs': epochs, 'patience': patience,
        'best_epoch': best_epoch, 'best_validation_mae': best_mae,
        'checkpoint_selection': 'minimum_validation_mae_original_tokens_rounded_minimum_1',
        'records_sha256': _file_hash(data_dir / 'records.parquet'), 'catalog_sha256': _file_hash(data_dir / 'catalog.json'),
        'versions': {name: _version(name) for name in ('torch', 'numpy', 'pandas', 'sentence-transformers', 'transformers')},
        **annotation_metadata,
    }
    torch.save(best_state, output_dir / 'model.pt')
    (output_dir / 'model_config.json').write_text(json.dumps(configuration, indent=2), encoding='utf-8')
    (output_dir / 'training_history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    predictions = _predict_original(network, valid_loader, device)
    return _write_report(rows, valid_mask, predictions, catalog, output_dir, 'validation', annotation_metadata)


def evaluate_saved_predictor(data_dir: str | Path, artifact_dir: str | Path, *, cache_dir: str | Path,
                             device: str = 'auto', encoder: object | None = None,
                             tokenizer: object | None = None, batch_size: int = 1024) -> dict:
    """Reload independently and evaluate immutable held-out rows once per artifact."""
    data_dir, artifact_dir, cache_dir = Path(data_dir), Path(artifact_dir), Path(cache_dir)
    if any((artifact_dir / name).exists() for name in ('test_metrics.json', 'test_predictions.csv')):
        raise ValueError('test results already exist; this artifact has already been evaluated')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    configuration = load_configuration(artifact_dir)
    rows, catalog, model_ids, annotations = _read_data(data_dir)
    for field, filename in (('records_sha256', 'records.parquet'), ('catalog_sha256', 'catalog.json')):
        if not configuration.get(field) or _file_hash(data_dir / filename) != configuration[field]:
            raise ValueError(f'evaluation {filename.split(".")[0]} differ from the saved training data')
    if model_ids != configuration['model_ids']:
        raise ValueError('evaluation catalog order differs from trained model IDs')
    annotation_metadata = _annotation_metadata(rows, annotations)
    if any(configuration.get(key) != annotation_metadata[key] for key in
           ('annotation_sha256', 'split_records_sha256', 'label_source_counts', 'dimension_known_counts')):
        raise ValueError('evaluation annotations or split records differ from saved training data')
    features, row_index = load_feature_cache(rows['prompt'].tolist(), cache_dir,
                                            config=FeatureConfigV2(**configuration['feature_config']))
    cache_metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    if cache_metadata['fingerprint'] != configuration.get('feature_cache_fingerprint'):
        raise ValueError('evaluation cache differs from saved training features')
    test_mask = rows['split'].eq('test').to_numpy()
    if not test_mask.any():
        raise ValueError('test rows are required for explicit held-out evaluation')
    device = _device(device)
    network = ModelAwareRegressor(configuration['embedding_dim'], len(model_ids), configuration['hidden_dim']).to(device)
    network.load_state_dict(torch.load(artifact_dir / 'model.pt', map_location='cpu', weights_only=True))
    dataset = _LabeledRows(rows, features, row_index, model_ids, test_mask, annotations,
                           configuration['use_label_features'])
    predictions = _predict_original(network, _loader(dataset, batch_size, False, device), device)
    return _write_report(rows, test_mask, predictions, catalog, artifact_dir, 'test', annotation_metadata)


def _load_local_encoder(path, device):
    """Verify the frozen manifest before loading an explicitly local encoder."""
    from deep_learning.scripts.download_encoder import prepare
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError('A local encoder directory is required; training never downloads an encoder')
    manifest_path = Path(__file__).resolve().parents[1] / 'artifacts/encoder-manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    prepare(path, manifest, source=path)
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path), local_files_only=True, trust_remote_code=False, device=_device(device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--encoder-path', type=Path,
                        help='verified local frozen encoder; omitted retains the existing cache-based behavior')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--raw-loss-weight', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--no-label-features', action='store_true', help='omit all capability, mask and source features')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    encoder = _load_local_encoder(args.encoder_path, args.device) if args.encoder_path is not None else None
    if args.prepare_only:
        features, index = prepare_feature_cache(args.data_dir, args.cache_dir, device=args.device, encoder=encoder)
        print(json.dumps({'unique_prompts': len(features), 'rows': len(index), 'text_dimensions': features.shape[1]}))
    elif args.evaluate_only:
        print(json.dumps(evaluate_saved_predictor(args.data_dir, args.output_dir, cache_dir=args.cache_dir,
                                                  device=args.device, batch_size=args.batch_size, encoder=encoder)['overall'], indent=2))
    else:
        report = train_predictor_v3(args.data_dir, args.output_dir, cache_dir=args.cache_dir,
                                     epochs=args.epochs, patience=args.patience, lr=args.lr,
                                     hidden_dim=args.hidden_dim, raw_loss_weight=args.raw_loss_weight,
                                     seed=args.seed, device=args.device, batch_size=args.batch_size,
                                     use_label_features=not args.no_label_features, encoder=encoder)
        print(json.dumps(report['overall'], indent=2))


if __name__ == '__main__':
    main()
