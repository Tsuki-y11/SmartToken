"""Bounded, validation-selected 50-head experiment; historical test is explicit.

Run validation first, then a separate ``--phase test`` invocation. The first
phase never predicts test rows. Proxy-label results are research ablations;
only no-label weights qualify for live deployment with unaligned ML labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep_learning.dl.dimensions import LABEL_SOURCES
from deep_learning.dl.features_v2 import FeatureConfigV2, STRUCTURE_FEATURE_NAMES, load_feature_cache
from deep_learning.dl.metrics import evaluate_predictions
from deep_learning.dl.model import MAX_OUTPUT_TOKENS, ModelAwareRegressor
from deep_learning.dl.predictor_v3 import TokenPredictorV3, load_configuration
from deep_learning.dl.train_v2 import _device, _file_hash, _loader, _predict_original
from deep_learning.dl.train_v3 import (_LabeledRows, _annotation_metadata, _annotations,
                         _load_local_encoder, _read_data, train_predictor_v3)


CANONICAL_CATALOG = Path(__file__).resolve().parents[1] / 'data/dl/catalog.json'
PROTOCOL_VERSION = 'fifty_model_validation_v1'
FIXED = {'epochs': 20, 'patience': 4, 'batch_size': 1024, 'lr': 0.001}
HISTORICAL_TEST_NOTICE = ('Already-exposed historical test; not a new blind test. '
                          'No test metric participates in candidate selection or tuning.')


def _json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def validate_records(rows, catalog):
    """Enforce exact catalog identity, complete splits and whole-example isolation."""
    canonical = json.loads(CANONICAL_CATALOG.read_text(encoding='utf-8'))
    expected = [item['model_id'] for item in canonical]
    ids = [item['model_id'] for item in catalog]
    if len(ids) != 50 or len(set(ids)) != 50 or ids != expected:
        raise ValueError('catalog must contain the exact fifty historical IDs in canonical order')
    required = {'model_id', 'prompt', 'original_query', 'split', 'output_tokens', 'source'}
    if required - set(rows.columns):
        raise ValueError(f'missing required records fields: {sorted(required - set(rows.columns))}')
    if set(rows['split']) != {'train', 'valid', 'test'}:
        raise ValueError('original train/valid/test splits are required')
    for split in ('train', 'valid', 'test'):
        if set(rows.loc[rows['split'].eq(split), 'model_id']) != set(ids):
            raise ValueError(f'{split} model coverage is missing exact catalog IDs')
    for field in ('group_id', 'request_id', 'source_prompt', 'source_example_id', 'original_query', 'prompt'):
        if field not in rows:
            continue
        selected = rows.loc[rows[field].notna(), [field, 'split']]
        if selected[field].map(lambda x: not isinstance(x, str) or not x.strip()).any():
            raise ValueError(f'{field} grouping values must be nonempty strings')
        keys = selected[field].map(lambda x: ' '.join(unicodedata.normalize('NFKC', x).casefold().split()))
        if (selected.groupby(keys)['split'].nunique() > 1).any():
            raise ValueError(f'{field} group appears in multiple original splits')
    targets = rows['output_tokens'].to_numpy(dtype=np.float64)
    if not np.isfinite(targets).all() or (targets <= 0).any():
        raise ValueError('historical output targets must be finite and positive')
    _annotations(rows)
    return ids


def fit_baselines(rows, structure, model_ids):
    """Fit fixed alpha=1 per-model raw-token ridge and constants on train only."""
    values = np.asarray(structure, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(rows) or not np.isfinite(values).all():
        raise ValueError('structure features must be a finite matrix aligned with records')
    train = rows['split'].eq('train').to_numpy()
    if not train.any() or set(rows.loc[train, 'model_id']) != set(model_ids):
        raise ValueError('training coverage must include every exact catalog ID')
    selected = rows.loc[train]
    x = values[train]
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack([(x - center) / scale, np.ones(len(x))])
    penalty = np.eye(design.shape[1])
    penalty[-1, -1] = 0
    weights = {}
    for model in model_ids:
        mask = selected['model_id'].eq(model).to_numpy()
        a = design[mask]
        y = selected.loc[mask, 'output_tokens'].to_numpy(dtype=np.float64)
        weights[model] = np.linalg.solve(a.T @ a + penalty, a.T @ y).tolist()
    common = {'model_ids': list(model_ids), 'fit_split': 'train', 'training_rows': int(train.sum())}
    return {
        'mean': {**common, 'kind': 'mean', 'values': selected.groupby('model_id')['output_tokens'].mean().to_dict()},
        'median': {**common, 'kind': 'median', 'values': selected.groupby('model_id')['output_tokens'].median().to_dict()},
        'structure_ridge': {**common, 'kind': 'structure_ridge', 'alpha': 1.0,
                            'target_units': 'original_tokens', 'center': center.tolist(),
                            'scale': scale.tolist(), 'weights': weights},
    }


def predict_baseline(model, rows, structure):
    if set(rows['model_id']) - set(model['model_ids']):
        raise ValueError('unknown model outside baseline catalog')
    if model['kind'] in ('mean', 'median'):
        return rows['model_id'].map(model['values']).to_numpy(dtype=np.float64)
    x = np.asarray(structure, dtype=np.float64)
    if x.shape != (len(rows), len(model['center'])) or not np.isfinite(x).all():
        raise ValueError('structure feature shape differs from fitted ridge')
    design = np.column_stack([(x - model['center']) / model['scale'], np.ones(len(rows))])
    weights = np.asarray([model['weights'][name] for name in rows['model_id']], dtype=np.float64)
    return np.clip(np.sum(design * weights, axis=1), 1, MAX_OUTPUT_TOKENS)


def metric_report(rows, predictions, *, catalog=None):
    """Original-token errors, with macro model weighting and signed/tail errors."""
    report = evaluate_predictions(rows.reset_index(drop=True), predictions, catalog=catalog)
    error = np.asarray(predictions, dtype=np.float64) - rows['output_tokens'].to_numpy(dtype=np.float64)
    report['overall']['absolute_error_p90'] = float(np.quantile(np.abs(error), 0.9))
    report['overall']['absolute_error_p95'] = float(np.quantile(np.abs(error), 0.95))
    report['macro_model_mae'] = float(np.mean([item['mae'] for item in report['by_model'].values()]))
    report['macro_model_bias'] = float(np.mean([item['bias'] for item in report['by_model'].values()]))
    for model, indices in rows.reset_index(drop=True).groupby('model_id').indices.items():
        report['by_model'][str(model)]['absolute_error_p90'] = float(np.quantile(np.abs(error[indices]), 0.9))
    return report


def baseline_metric_report(rows, predictions, *, catalog=None):
    """Retain continuous baseline evidence and disclose the native integer rule."""
    report = metric_report(rows, predictions, catalog=catalog)
    report['output_quantization'] = 'continuous_baseline'
    quantized = np.clip(np.rint(np.asarray(predictions, dtype=np.float64)), 1, MAX_OUTPUT_TOKENS)
    report['deployment_quantized'] = metric_report(rows, quantized, catalog=catalog)
    report['deployment_quantized']['output_quantization'] = 'round_nearest_even_clip_1_1000000'
    return report


def validate_released_provenance(configuration, context, model_ids, annotations):
    """Cache identity never substitutes for original records/split/label identity."""
    expected = {key: context[key] for key in
                ('records_sha256', 'catalog_sha256', 'feature_cache_fingerprint', 'split_records_sha256')}
    expected['model_ids'] = list(model_ids)
    expected.update({key: annotations[key] for key in
                     ('annotation_sha256', 'label_source_counts', 'dimension_known_counts')})
    for field, value in expected.items():
        if value is None or configuration.get(field) != value:
            raise ValueError(f'released checkpoint provenance identity differs or is missing: {field}')
    return {'verified': True, 'records_sha256': context['records_sha256'],
            'checked_fields': expected,
            'policy': 'exact_records_catalog_splits_and_annotations_no_cache_only_equivalence'}


def initial_matrix():
    return [dict(FIXED, use_label_features=labels, hidden_dim=width, raw_loss_weight=raw, seed=42)
            for labels in (False, True) for width in (64, 128) for raw in (0.0, 0.2, 0.5)]


def repeat_matrix(entries):
    pairs = {}
    for entry in entries:
        config = entry.get('config', {})
        if entry['kind'] == 'neural' and config.get('seed') == 42:
            key = (config['hidden_dim'], config['raw_loss_weight'])
            pairs.setdefault(key, {})[config['use_label_features']] = entry['validation']['overall']['mae']
    complete = [(sum(scores.values()) / 2, width, raw) for (width, raw), scores in pairs.items()
                if set(scores) == {False, True}]
    if not complete:
        raise ValueError('a complete labels-on/off validation pair is required')
    _, width, raw = min(complete)
    return [dict(FIXED, hidden_dim=width, raw_loss_weight=raw, seed=seed, use_label_features=labels)
            for seed in (7, 2026) for labels in (False, True)]


def select_candidate(entries, *, deployable_only=True):
    eligible = []
    for entry in entries:
        if entry['kind'] not in ('neural', 'released'):
            continue
        config = entry['config']
        if entry['kind'] == 'neural' and config.get('seed') != 42:
            continue  # Repeat seeds measure stability, never add selection attempts.
        if deployable_only and config.get('use_label_features') is not False:
            continue
        mae = float(entry['validation']['overall']['mae'])
        if not math.isfinite(mae):
            raise ValueError('candidate validation MAE must be finite')
        eligible.append(entry)
    if not eligible:
        raise ValueError('no validation-selected deployable candidate')
    return min(eligible, key=lambda e: (e['validation']['overall']['mae'],
                                      e['kind'] != 'released', e['candidate_id']))


def artifact_identity(artifact_dir):
    path = Path(artifact_dir).resolve()
    names = ['baseline.json'] if (path / 'baseline.json').exists() else ['model_config.json', 'model.pt']
    if any(not (path / name).is_file() for name in names):
        raise ValueError('artifact is incomplete; config and trained weights are required')
    return {'path': str(path), 'files_sha256': {name: _file_hash(path / name) for name in names}}


def _verify_identity(identity):
    for name, expected in identity['files_sha256'].items():
        if not expected or _file_hash(Path(identity['path']) / name) != expected:
            raise ValueError(f'artifact identity changed: {identity["path"]}/{name}')


def freeze_selection(entries, path, *, context=None):
    path = Path(path)
    if path.exists():
        raise ValueError('selection is already frozen; use a new experiment directory')
    selected = select_candidate(entries)
    research = select_candidate(entries, deployable_only=False)
    final = {selected['candidate_id'], research['candidate_id']}
    final.update(e['candidate_id'] for e in entries if e['kind'] in ('baseline', 'released'))
    repeat = [e for e in entries if e['kind'] == 'neural' and e['config'].get('seed') in (7, 2026)]
    final.update(e['candidate_id'] for e in repeat)
    paired_keys = {(e['config']['hidden_dim'], e['config']['raw_loss_weight']) for e in repeat}
    final.update(e['candidate_id'] for e in entries if e['kind'] == 'neural' and
                 (e['config']['hidden_dim'], e['config']['raw_loss_weight']) in paired_keys)
    for entry in entries:
        _verify_identity(entry['artifact'])
    manifest = {'protocol_version': PROTOCOL_VERSION, 'selection_split': 'valid',
                'selection_metric': 'overall_mae_original_tokens',
                'tie_policy': 'retain_released_then_lexicographic_id',
                'selected_candidate_id': selected['candidate_id'],
                'research_winner_candidate_id': research['candidate_id'],
                'deployable_policy': 'omit_all_label_features',
                'repeat_seed_policy': 'stability_only_no_reselection',
                'final_test_candidate_ids': sorted(final), 'candidates': entries,
                'historical_test_notice': HISTORICAL_TEST_NOTICE, 'context': context or {}}
    manifest['manifest_sha256'] = _digest(manifest)
    _json(path, manifest, exclusive=True)
    return manifest


def verify_frozen_selection(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    body = {key: value for key, value in manifest.items() if key != 'manifest_sha256'}
    if manifest.get('manifest_sha256') != _digest(body):
        raise ValueError('frozen selection manifest hash changed')
    for entry in manifest['candidates']:
        _verify_identity(entry['artifact'])
    return True


def permitted_test_candidates(path, requested=None):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    permitted = manifest['final_test_candidate_ids']
    selected = permitted if requested is None else list(requested)
    if set(selected) - set(permitted):
        raise ValueError('test evaluation requested a candidate outside the frozen selection')
    verify_frozen_selection(path)
    return selected


def label_provenance_audit(rows):
    annotations = _annotations(rows)
    metadata = _annotation_metadata(rows, annotations)
    counts = metadata['label_source_counts']['train']
    return {**metadata, 'training_source_counts': counts,
            'ml_label_effectiveness_established': False,
            'live_ml_source_unseen_in_training': counts['ml'] == 0,
            'safe_live_policy': 'omit_all_label_features',
            'limitation': 'Dataset-proxy labels are automatic partial supervision, not human gold or learned ML labels. '
                          'Proxy-label validation cannot establish benefit from live classifier labels/source channels.',
            'low_cost_future_alignment': 'Freeze classifier version and predictions for original train/valid/test prompts; '
                'preserve all splits; retrain matched labels-on/off pairs with ML source on train and select on valid. '
                'Alternatively predeclare and validate a source-channel ablation before deployment. '
                'Neither optional extension is performed or claimed by this bounded experiment.'}


def audit_predictor_coverage(predictor, model_ids):
    if len(model_ids) != 50 or list(predictor.model_ids) != list(model_ids):
        raise ValueError('predictor coverage differs from exact historical catalog order')
    heads = predictor.network.model_head.weight.detach().cpu().numpy()
    distinct = len({hashlib.sha256(row.tobytes()).hexdigest() for row in heads})
    if len(heads) != 50 or distinct != 50:
        raise ValueError('fifty distinct trained model heads are required')
    probes = []
    for prompt in ('Explain why unit tests are useful in two sentences.',
                   'Write Python code to sort a list; explain complexity and include one test.'):
        values = predictor.predict(prompt)
        if list(values) != list(model_ids) or any(type(value) is not int or value < 1 for value in values.values()):
            raise ValueError('predictor did not produce valid forecasts for exact fifty IDs')
        count = len(set(values.values()))
        if count < 2:
            raise ValueError('all model forecasts are identical; per-model coverage audit failed')
        probes.append({'prompt': prompt, 'predictions': values, 'distinct_forecast_count': count})
    return {'exact_model_count': 50, 'distinct_head_count': distinct,
            'prediction_method': 'native_model_aware_v3_heads_no_aliases_or_rescaling', 'probes': probes}


def _checkpoint_predictions(artifact_dir, rows, features, row_index, ids, split, device):
    config = load_configuration(artifact_dir)
    if config['model_ids'] != ids:
        raise ValueError('checkpoint model IDs differ from exact catalog order')
    if features.shape[1] != config['text_feature_dimensions']:
        raise ValueError('checkpoint text dimensions differ from frozen features')
    mask = rows['split'].eq(split).to_numpy()
    network = ModelAwareRegressor(config['embedding_dim'], 50, config['hidden_dim']).to(device)
    network.load_state_dict(torch.load(Path(artifact_dir) / 'model.pt', map_location='cpu', weights_only=True))
    dataset = _LabeledRows(rows, features, row_index, ids, mask, _annotations(rows), config['use_label_features'])
    return _predict_original(network, _loader(dataset, 1024, False, device), device)


def _candidate_id(config):
    return f'neural_labels-{int(config["use_label_features"])}_width-{config["hidden_dim"]}_raw-{config["raw_loss_weight"]:g}_seed-{config["seed"]}'


def _context(data_dir, cache_dir, rows, catalog):
    metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    return {'records_sha256': _file_hash(data_dir / 'records.parquet'),
            'catalog_sha256': _file_hash(data_dir / 'catalog.json'),
            'canonical_ids_sha256': _digest([item['model_id'] for item in catalog]),
            'cache_metadata_sha256': _file_hash(cache_dir / 'metadata.json'),
            'feature_cache_fingerprint': metadata['fingerprint'],
            'feature_cache_file_sha256': metadata['file_sha256'],
            'split_counts': rows['split'].value_counts().to_dict(),
            'split_records_sha256': _annotation_metadata(rows, _annotations(rows))['split_records_sha256']}


def audit_released_candidates(data_dir, output_dir, *, cache_dir):
    """Audit an existing protocol without changing training, selection or results.

    This also supplies an explicit provenance receipt for an already-running
    v1 experiment started before the stricter release-admission gate existed.
    It predicts no held-out test rows and never retrains any candidate.
    """
    data_dir, output_dir, cache_dir = map(Path, (data_dir, output_dir, cache_dir))
    protocol_path = output_dir / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    rows, catalog, ids, annotations = _read_data(data_dir)
    validate_records(rows, catalog)
    context = _context(data_dir, cache_dir, rows, catalog)
    if context != protocol['context']:
        raise ValueError('provenance audit data or frozen split/cache identity differs from protocol')
    metadata = _annotation_metadata(rows, annotations)
    released = {}
    for name, identity in protocol['released_artifacts'].items():
        _verify_identity(identity)
        configuration = load_configuration(identity['path'])
        if configuration['use_label_features'] != (name == 'labels'):
            raise ValueError('released checkpoint label-feature identity differs from protocol role')
        released[name] = {'artifact': identity,
                          **validate_released_provenance(configuration, context, ids, metadata)}
    report = {'verified': True, 'audit_version': 'released_provenance_v1',
              'protocol_sha256': _file_hash(protocol_path), 'context': context, 'released': released,
              'candidate_results_modified': False, 'selection_modified': False,
              'test_predictions_performed': False}
    path = output_dir / 'released_provenance_audit.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != report:
            raise ValueError('existing released provenance audit differs; do not overwrite evidence')
    else:
        _json(path, report, exclusive=True)
    # Preserve all original scores. The sidecar makes comparison quantization
    # explicit even for the immutable v1 process already running in memory.
    if all((output_dir / f'baseline_{name}/baseline.json').exists()
           for name in ('mean', 'median', 'structure_ridge')):
        features, row_index = load_feature_cache(rows['prompt'].tolist(), cache_dir)
        valid = rows['split'].eq('valid').to_numpy()
        selected = rows.loc[valid].reset_index(drop=True)
        structure = np.asarray(features[row_index[valid], -len(STRUCTURE_FEATURE_NAMES):], dtype=np.float64)
        comparisons = {}
        for name in ('mean', 'median', 'structure_ridge'):
            model = json.loads((output_dir / f'baseline_{name}/baseline.json').read_text(encoding='utf-8'))
            predictions = predict_baseline(model, selected, structure)
            comparisons[name] = baseline_metric_report(selected, predictions, catalog=catalog)
        _json(output_dir / 'baseline_quantization_audit.json', {
            'selection_modified': False, 'original_candidate_scores_preserved': True,
            'split': 'valid', 'baselines': comparisons})
    return report


def finalize_selection(output_dir, *, device='cpu', encoder=None):
    """Idempotently stage a frozen winner; never retrain or select again.

    A transient encoder/filesystem failure may be retried. An actual failed
    coverage audit still raises, and no successful deployment receipt is made.
    """
    output_dir = Path(output_dir)
    verify_frozen_selection(output_dir / 'selection.json')
    manifest = json.loads((output_dir / 'selection.json').read_text(encoding='utf-8'))
    selected = next(entry for entry in manifest['candidates']
                    if entry['candidate_id'] == manifest['selected_candidate_id'])
    if selected['config'].get('use_label_features') is not False:
        raise ValueError('frozen deployment policy requires no-label native predictor weights')
    if any(entry['kind'] == 'released' for entry in manifest['candidates']):
        audit_path = output_dir / 'released_provenance_audit.json'
        if not audit_path.exists():
            raise ValueError('run the explicit provenance audit before finalizing this frozen experiment')
        proof = json.loads(audit_path.read_text(encoding='utf-8'))
        if not proof.get('verified') or proof.get('context') != manifest['context']:
            raise ValueError('released provenance audit differs from frozen experiment identity')
        if proof.get('protocol_sha256') != _file_hash(output_dir / 'protocol.json'):
            raise ValueError('protocol identity changed after released provenance audit')
        audited = [item['artifact'] for item in proof['released'].values()]
        for entry in manifest['candidates']:
            if entry['kind'] == 'released' and entry['artifact'] not in audited:
                raise ValueError('a frozen released artifact has no matching provenance audit')
    destination = output_dir / 'selected'
    destination.mkdir(exist_ok=True)
    receipt_path = output_dir / 'selected_artifact.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        if (receipt.get('selected_candidate_id') != manifest['selected_candidate_id'] or
                receipt.get('selection_manifest_sha256') != manifest['manifest_sha256'] or
                receipt.get('source_artifact') != selected['artifact']):
            raise ValueError('existing staging receipt differs from frozen selection')
        _verify_identity(receipt['selected_artifact'])
    else:
        for filename in ('model.pt', 'model_config.json'):
            shutil.copyfile(Path(selected['artifact']['path']) / filename, destination / filename)
    configuration = load_configuration(destination)
    predictor = TokenPredictorV3.load(destination, device=_device(device), encoder=encoder)
    audit = audit_predictor_coverage(predictor, configuration['model_ids'])
    _json(output_dir / 'selected_coverage.json', audit)
    receipt = {'selected_candidate_id': selected['candidate_id'],
               'research_winner_candidate_id': manifest['research_winner_candidate_id'],
               'selected_artifact': artifact_identity(destination),
               'source_artifact': selected['artifact'], 'selection_manifest_sha256': manifest['manifest_sha256'],
               'feature_policy': 'omit_all_label_features', 'historical_test_evaluated': False}
    _json(receipt_path, receipt)
    return receipt


def _save_validation(entry, predictions, rows, catalog, directory):
    entry['validation'] = (baseline_metric_report(rows, predictions, catalog=catalog)
                           if entry['kind'] == 'baseline' else metric_report(rows, predictions, catalog=catalog))
    if entry['kind'] != 'baseline':
        entry['validation']['output_quantization'] = 'round_nearest_even_clip_1_1000000'
    entry['deployable'] = entry['kind'] != 'baseline' and entry['config'].get('use_label_features') is False
    _json(directory / 'validation_report.json', entry['validation'])
    return entry


def _write_leaderboard(entries, output_dir):
    ordered = sorted(entries, key=lambda e: (e['validation']['overall']['mae'], e['candidate_id']))
    mean = next(e['validation']['overall']['mae'] for e in entries if e['candidate_id'] == 'baseline_mean')
    table = []
    for rank, entry in enumerate(ordered, 1):
        report = entry['validation']
        table.append({'research_rank': rank, 'candidate_id': entry['candidate_id'], 'kind': entry['kind'],
                      'deployable': entry['deployable'], 'selection_eligible_seed': entry['kind'] != 'neural' or entry['config']['seed'] == 42,
                      'mae': report['overall']['mae'], 'macro_model_mae': report['macro_model_mae'],
                      'bias': report['overall']['bias'], 'absolute_error_p90': report['overall']['absolute_error_p90'],
                      'output_quantization': report.get('output_quantization'),
                      'deployment_quantized_mae': report.get('deployment_quantized', report)['overall']['mae'],
                      'gain_vs_mean_percent': 100 * (mean - report['overall']['mae']) / mean if mean else None,
                      **entry['config']})
    _json(output_dir / 'validation_leaderboard.json', ordered)
    pd.DataFrame(table).to_csv(output_dir / 'validation_leaderboard.csv', index=False)


def run_validation(data_dir, output_dir, *, cache_dir, released_labels, released_no_labels,
                   device='auto', encoder=None):
    """Train bounded candidates, freeze validation selection and stage chosen weights."""
    data_dir, output_dir, cache_dir = map(Path, (data_dir, output_dir, cache_dir))
    if (output_dir / 'selection.json').exists():
        raise ValueError('selection already frozen; use --phase finalize to retry staging or --phase test for the frozen evaluation')
    rows, catalog, ids, annotations = _read_data(data_dir)
    validate_records(rows, catalog)
    device = _device(device)
    features, row_index = load_feature_cache(rows['prompt'].tolist(), cache_dir)
    context = _context(data_dir, cache_dir, rows, catalog)
    annotation_metadata = _annotation_metadata(rows, annotations)
    # Reject incompatible released training partitions before admitting any
    # release to the protocol, validation leaderboard or deployment selection.
    for label, directory in ((True, released_labels), (False, released_no_labels)):
        configuration = load_configuration(directory)
        if configuration['use_label_features'] != label:
            raise ValueError('released checkpoint label-feature provenance differs from declared role')
        validate_released_provenance(configuration, context, ids, annotation_metadata)
    protocol = {'version': PROTOCOL_VERSION, 'context': context, 'initial_matrix': initial_matrix(),
                'paired_repeats': {'seeds': [7, 2026], 'selection': 'minimum_mean_paired_labels_on_off_validation_mae'},
                'baseline_ridge_alpha': 1.0, 'test_notice': HISTORICAL_TEST_NOTICE,
                'deployment_gate': 'no_label_features_only_until_matched_live_ml_validation',
                'released_artifacts': {'labels': artifact_identity(released_labels),
                                       'no_labels': artifact_identity(released_no_labels)}}
    protocol_path = output_dir / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
            raise ValueError('resume data/cache/released artifacts differ from frozen protocol')
    else:
        _json(protocol_path, protocol, exclusive=True)
    _json(output_dir / 'label_provenance.json', label_provenance_audit(rows))
    valid = rows['split'].eq('valid').to_numpy()
    valid_rows = rows.loc[valid].reset_index(drop=True)
    structure = np.asarray(features[row_index, -len(STRUCTURE_FEATURE_NAMES):], dtype=np.float64)
    baselines = fit_baselines(rows, structure, ids)
    entries = []
    for name, model in baselines.items():
        directory = output_dir / f'baseline_{name}'
        _json(directory / 'baseline.json', model)
        predictions = predict_baseline(model, valid_rows, structure[valid])
        entry = {'candidate_id': f'baseline_{name}', 'kind': 'baseline', 'config': {},
                 'artifact': artifact_identity(directory)}
        entries.append(_save_validation(entry, predictions, valid_rows, catalog, directory))
    cache_fingerprint = context['feature_cache_fingerprint']
    for label, directory in ((True, Path(released_labels)), (False, Path(released_no_labels))):
        config = load_configuration(directory)
        if config['use_label_features'] != label or config.get('feature_cache_fingerprint') != cache_fingerprint:
            raise ValueError('released checkpoint labels or frozen feature cache identity differ')
        destination = output_dir / ('released_labels' if label else 'released_no_labels')
        entry = {'candidate_id': destination.name, 'kind': 'released',
                 'config': {key: config.get(key) for key in ('use_label_features', 'hidden_dim', 'raw_loss_weight', 'seed')},
                 'artifact': artifact_identity(directory)}
        predictions = _checkpoint_predictions(directory, rows, features, row_index, ids, 'valid', device)
        entries.append(_save_validation(entry, predictions, valid_rows, catalog, destination))
    _write_leaderboard(entries, output_dir)

    def train(config):
        directory = output_dir / _candidate_id(config)
        if (directory / 'model.pt').exists() and (directory / 'model_config.json').exists():
            saved = load_configuration(directory)
            for key, expected in {**config, 'records_sha256': context['records_sha256'],
                                  'feature_cache_fingerprint': cache_fingerprint}.items():
                field = 'max_epochs' if key == 'epochs' else key
                if saved.get(field) != expected:
                    raise ValueError(f'resumed candidate config differs: {field}')
        else:
            train_predictor_v3(data_dir, directory, cache_dir=cache_dir, device=device, encoder=encoder, **config)
        predictions = _checkpoint_predictions(directory, rows, features, row_index, ids, 'valid', device)
        entry = {'candidate_id': directory.name, 'kind': 'neural', 'config': config,
                 'artifact': artifact_identity(directory)}
        entries.append(_save_validation(entry, predictions, valid_rows, catalog, directory))
        _write_leaderboard(entries, output_dir)
        print(json.dumps({'completed': directory.name, 'validation_mae': entries[-1]['validation']['overall']['mae']}), flush=True)

    for config in initial_matrix():
        train(config)
    repeats = repeat_matrix(entries)
    _json(output_dir / 'paired_repeat_configuration.json', repeats)
    for config in repeats:
        train(config)
    audit_released_candidates(data_dir, output_dir, cache_dir=cache_dir)
    freeze_selection(entries, output_dir / 'selection.json', context=context)
    return finalize_selection(output_dir, device=device, encoder=encoder)


def run_historical_test(data_dir, output_dir, *, cache_dir, device='auto'):
    """Evaluate only the immutable preselected list; never change the winner."""
    data_dir, output_dir, cache_dir = map(Path, (data_dir, output_dir, cache_dir))
    selection_path = output_dir / 'selection.json'
    permitted = permitted_test_candidates(selection_path)
    manifest = json.loads(selection_path.read_text(encoding='utf-8'))
    if (output_dir / 'historical_test_report.json').exists():
        raise ValueError('historical test report already exists; no repeated test phase')
    audit_released_candidates(data_dir, output_dir, cache_dir=cache_dir)
    rows, catalog, ids, _ = _read_data(data_dir)
    validate_records(rows, catalog)
    if _context(data_dir, cache_dir, rows, catalog) != manifest['context']:
        raise ValueError('records, original splits or frozen features changed after selection')
    features, row_index = load_feature_cache(rows['prompt'].tolist(), cache_dir)
    test = rows['split'].eq('test').to_numpy()
    test_rows = rows.loc[test].reset_index(drop=True)
    structure = np.asarray(features[row_index[test], -len(STRUCTURE_FEATURE_NAMES):], dtype=np.float64)
    reports = {}
    for entry in manifest['candidates']:
        name = entry['candidate_id']
        if name not in permitted:
            continue
        path = Path(entry['artifact']['path'])
        if entry['kind'] == 'baseline':
            model = json.loads((path / 'baseline.json').read_text(encoding='utf-8'))
            predictions = predict_baseline(model, test_rows, structure)
        else:
            predictions = _checkpoint_predictions(path, rows, features, row_index, ids, 'test', _device(device))
        reports[name] = (baseline_metric_report(test_rows, predictions, catalog=catalog)
                         if entry['kind'] == 'baseline' else metric_report(test_rows, predictions, catalog=catalog))
        if entry['kind'] != 'baseline':
            reports[name]['output_quantization'] = 'round_nearest_even_clip_1_1000000'
        _json(output_dir / 'final_test' / f'{name}.json', reports[name])
    report = {'historical_test_notice': HISTORICAL_TEST_NOTICE,
              'selection_manifest_sha256': manifest['manifest_sha256'],
              'selected_candidate_id': manifest['selected_candidate_id'],
              'research_winner_candidate_id': manifest['research_winner_candidate_id'],
              'evaluated_candidate_ids': sorted(reports), 'candidates': reports,
              'selection_changed_after_test': False}
    _json(output_dir / 'historical_test_report.json', report, exclusive=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('validation', 'audit', 'finalize', 'test'), required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--released-labels', type=Path)
    parser.add_argument('--released-no-labels', type=Path)
    parser.add_argument('--encoder-path', type=Path)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    args = parser.parse_args()
    if args.phase == 'validation':
        if args.released_labels is None or args.released_no_labels is None:
            parser.error('validation requires both fixed released checkpoints')
        encoder = _load_local_encoder(args.encoder_path, args.device) if args.encoder_path else None
        result = run_validation(args.data_dir, args.output_dir, cache_dir=args.cache_dir,
                                released_labels=args.released_labels, released_no_labels=args.released_no_labels,
                                device=args.device, encoder=encoder)
    elif args.phase == 'audit':
        result = audit_released_candidates(args.data_dir, args.output_dir, cache_dir=args.cache_dir)
    elif args.phase == 'finalize':
        audit_released_candidates(args.data_dir, args.output_dir, cache_dir=args.cache_dir)
        encoder = _load_local_encoder(args.encoder_path, args.device) if args.encoder_path else None
        result = finalize_selection(args.output_dir, device=args.device, encoder=encoder)
    else:
        result = run_historical_test(args.data_dir, args.output_dir, cache_dir=args.cache_dir, device=args.device)
    print(json.dumps(result if args.phase != 'test' else {
        'selected_candidate_id': result['selected_candidate_id'],
        'evaluated_candidate_ids': result['evaluated_candidate_ids'],
        'notice': HISTORICAL_TEST_NOTICE}, indent=2))


if __name__ == '__main__':
    main()
