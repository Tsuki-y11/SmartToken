"""Auditable, partial dataset-category proxies, never human capability Gold."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from deep_learning.dl.dimensions import LABELS, encode_dimensions

MAPPING_VERSION = 'dataset_proxy_0.5.1_v2'
REFERENCES = {
    'llmrouterbench': 'https://github.com/ynulihao/LLMRouterBench/blob/main/evaluation/README.md',
    'xroutebench': 'https://huggingface.co/datasets/ulab-ai/xRouteBench/blob/main/README.md',
    'korbench': 'https://github.com/KOR-Bench/KOR-Bench',
}

# These are project-specific interpretations of benchmark tasks. They establish
# only coarse positive priors, never evidence that other capabilities are absent.
CATEGORY_RULES = {
    'llmrouterbench': {
        'aime': ['reasoning', 'math'], 'arc-agi': ['reasoning'],
        'arcc': ['reasoning', 'knowledge'], 'arenahard': None,
        'arenahard_coding': ['coding'], 'arenahard_creative_writing': None,
        'arenahard_math': ['reasoning', 'math'], 'bbh': ['reasoning'],
        'emorynlp': ['reasoning'], 'finqa': ['reasoning', 'knowledge', 'math'],
        'gpqa': ['reasoning', 'knowledge'], 'hle': ['knowledge'],
        'humaneval': ['coding'], 'kandk': ['reasoning'], 'korbench': ['reasoning'],
        'livecodebench': ['coding', 'reasoning'], 'livemathbench': ['reasoning', 'math'],
        'math500': ['reasoning', 'math'], 'mathbench': ['math'], 'mbpp': ['coding'],
        'medqa': ['knowledge'], 'meld': ['reasoning'], 'mmlupro': ['reasoning', 'knowledge'],
        'simpleqa': ['knowledge'], 'swe-bench': ['coding'], 'tau2': ['agents'],
        'winogrande': ['reasoning'],
    },
    'xroutebench': {
        'aime_2020_2024': ['reasoning', 'math'], 'arc_challenge': ['reasoning', 'knowledge'],
        'boolq': ['knowledge'], 'commonsense_qa': ['reasoning', 'knowledge'],
        'gsm8k': ['math'], 'hellaswag': ['reasoning'], 'human_eval': ['coding'],
        'math': ['reasoning', 'math'], 'mbpp': ['coding'], 'mmlu': ['knowledge'],
        'mmlu_pro': ['reasoning', 'knowledge'], 'openbook_qa': ['reasoning', 'knowledge'],
        'squad': ['knowledge'],
    },
}

def _ordered(labels):
    return [label for label in LABELS if label in labels]


def map_category(source: str, dataset: str) -> dict:
    labels = CATEGORY_RULES.get(source, {}).get(dataset)
    positives = _ordered(labels) if labels else None
    return dict(dimensions=positives, dimension_mask=list(positives) if positives else None,
                label_source='dataset_proxy' if positives else 'missing',
                label_policy_version='0.5.1', mapping_version=MAPPING_VERSION,
                mapping_evidence=f'category:{source}/{dataset}' if positives else 'unmapped_or_mixed_category',
                mapping_reference=REFERENCES.get(source, ''),
                mapping_scope='partial_positive_prior_not_human_gold')


def annotate_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if not {'source', 'dataset', 'prompt'}.issubset(rows.columns):
        raise ValueError('source, dataset and prompt columns are required')
    if set(rows.columns) & {'dimensions', 'dimension_mask', 'label_source'}:
        raise ValueError('refuse to overwrite existing annotation columns')
    lookup = {(source, dataset): map_category(source, dataset)
              for source, dataset in rows[['source', 'dataset']].drop_duplicates().itertuples(index=False, name=None)}
    for entry in lookup.values():
        encode_dimensions(entry['dimensions'], entry['label_source'], dimension_mask=entry['dimension_mask'])
    annotations = []
    for source, dataset, prompt in rows[['source', 'dataset', 'prompt']].itertuples(index=False, name=None):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a nonempty rendered string')
        entry = dict(lookup[source, dataset])
        # Benchmark wrappers are not user-authored extra constraints under
        # annotation policy 0.5.1. Category-only mapping cannot infer those.
        annotations.append(entry)
    output = rows.copy()
    frame = pd.DataFrame(annotations, index=rows.index)
    for column in frame:
        output[column] = frame[column]
    return output


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_proxy_dataset(data_dir: str | Path, output_dir: str | Path) -> dict:
    source, output = Path(data_dir), Path(output_dir)
    if output.exists():
        raise ValueError('output directory exists; choose a new path')
    rows = pd.read_parquet(source/'records.parquet')
    annotated = annotate_rows(rows)
    # Exact preservation includes real token targets and original split IDs.
    pd.testing.assert_frame_equal(annotated[rows.columns], rows)
    category_rows = []
    for (src, dataset), group in annotated.groupby(['source', 'dataset'], sort=True):
        entry = map_category(src, dataset)
        category_rows.append(dict(source=src, dataset=dataset, records=len(group),
                                  category_dimensions=entry['dimensions'],
                                  proxy_records=int(group['label_source'].eq('dataset_proxy').sum())))
    positives = collections.Counter(label for values in annotated['dimensions'] for label in (values or []))
    report = dict(mapping_version=MAPPING_VERSION, label_policy_version='0.5.1',
                  source_records_sha256=_sha(source/'records.parquet'),
                  source_catalog_sha256=_sha(source/'catalog.json'), records=len(rows),
                  proxy_records=int(annotated['label_source'].eq('dataset_proxy').sum()),
                  missing_records=int(annotated['label_source'].eq('missing').sum()),
                  split_counts=rows['split'].value_counts().to_dict(),
                  positive_records={label: positives[label] for label in LABELS},
                  source_counts=annotated['label_source'].value_counts().to_dict(),
                  categories=category_rows, references=REFERENCES,
                  unchanged_columns=list(rows.columns),
                  caveats=['Partial approximate positive labels, other dimensions unknown, not human Gold.',
                           'Category proxies are task/source priors, not verified per-prompt capabilities.',
                           'No multimodal from text-only MELD or serialized ARC grids; KOR is not Korean.',
                           'No multilingual assigned solely because prompt is English or Chinese.',
                           'No output_tokens, predictions, scores or answers used by mapping.',
                           'No instruction_following inferred from benchmark wrappers; user constraints need separate annotation.',
                           'Use original splits and a no-label ablation; full-question results do not validate dependent subtasks.'])
    output.mkdir(parents=True)
    annotated.to_parquet(output/'records.parquet', index=False)
    shutil.copy2(source/'catalog.json', output/'catalog.json')
    rules = dict(version=MAPPING_VERSION, categories=CATEGORY_RULES,
                 references=REFERENCES)
    (output/'mapping_rules.json').write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding='utf8')
    report.update(records_sha256=_sha(output/'records.parquet'), rules_sha256=_sha(output/'mapping_rules.json'))
    (output/'mapping_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_proxy_dataset(args.data_dir, args.output_dir), ensure_ascii=False, indent=2))
