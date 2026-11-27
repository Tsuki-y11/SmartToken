"""Versioned frozen first/mean/last token-window features and prompt structure."""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from deep_learning.dl.features import DEFAULT_ENCODER, DEFAULT_ENCODER_REVISION

FEATURE_STRATEGY = 'uniform_token_windows_first_mean_last_structure_v2'
STRUCTURE_FEATURE_NAMES = (
    'log_chars', 'log_words', 'log_lines', 'log_digits', 'log_questions',
    'log_code_fences', 'log_bullets', 'log_headings', 'log_brackets',
    'digit_ratio', 'alpha_ratio', 'whitespace_ratio', 'cjk_ratio',
    'punctuation_ratio', 'newline_ratio', 'log_max_line_chars',
    'log_token_constraints', 'log_max_token_constraint',
    'log_word_constraints', 'log_max_word_constraint',
    'log_char_constraints', 'log_max_char_constraint',
)


@dataclass(frozen=True)
class FeatureConfigV2:
    strategy: str = FEATURE_STRATEGY
    encoder_id: str = DEFAULT_ENCODER
    encoder_revision: str = DEFAULT_ENCODER_REVISION
    window_tokens: int = 120
    max_windows: int = 8

    def __post_init__(self):
        if self.strategy != FEATURE_STRATEGY:
            raise ValueError('unsupported v2 feature strategy')
        if self.window_tokens < 1 or not 1 <= self.max_windows <= 8:
            raise ValueError('window_tokens must be positive and max_windows between 1 and 8')
        if not self.encoder_id or not self.encoder_revision:
            raise ValueError('encoder identity and pinned revision are required')


def prompt_windows(text: str, tokenizer: object, config: FeatureConfigV2) -> list[str]:
    """Uniformly sample bounded contiguous token windows, including both ends."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    count = min(config.max_windows, max(1, math.ceil(len(tokens) / config.window_tokens)))
    starts = np.linspace(0, max(0, len(tokens) - config.window_tokens), count, dtype=np.int64)
    windows = []
    for start in starts:
        selected = tokens[int(start):int(start) + config.window_tokens]
        window = tokenizer.decode(selected, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        # Decoding a subword boundary can retokenize to more tokens. Shrink until
        # the exact string handed to SentenceTransformer respects the budget.
        while len(tokenizer.encode(window, add_special_tokens=False)) > config.window_tokens:
            selected = selected[:-1]
            window = tokenizer.decode(selected, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        windows.append(window)
    return windows


def structure_features(text: str) -> np.ndarray:
    """Only quantities derivable from the complete rendered prompt at inference."""
    length = max(1, len(text))
    digits = sum(char.isdigit() for char in text)
    alpha = sum(char.isalpha() for char in text)
    whitespace = sum(char.isspace() for char in text)
    cjk = sum('\u4e00' <= char <= '\u9fff' for char in text)
    punctuation = sum(not char.isalnum() and not char.isspace() for char in text)
    counts = [len(text), len(text.split()), text.count('\n') + 1, digits,
              text.count('?') + text.count('？'), text.count('```'),
              len(re.findall(r'(?m)^\s*(?:[-*+]\s|\d+[.)]\s)', text)),
              len(re.findall(r'(?m)^\s*#{1,6}\s', text)),
              sum(text.count(char) for char in '{}[]')]
    values = [math.log1p(value) for value in counts]
    values += [digits / length, alpha / length, whitespace / length, cjk / length,
               punctuation / length, text.count('\n') / length,
               math.log1p(max((len(line) for line in text.splitlines()), default=0))]
    for unit in (r'tokens?\b|个?\s*(?:令牌|tokens?\b)', r'words?\b|个?\s*(?:单词|字词)', r'characters?\b|chars?\b|个?\s*字(?:符)?'):
        constraints = [min(float(number.replace(',', '')), 1e9)
                       for number in re.findall(r'(\d[\d,]{0,12})\s*(?:' + unit + r')', text, re.IGNORECASE)]
        values.extend((math.log1p(len(constraints)), math.log1p(max(constraints, default=0))))
    return np.asarray(values, dtype=np.float32)


def encode_prompt_features(
    prompts: Sequence[str], encoder: object, *, tokenizer: object | None = None,
    config: FeatureConfigV2 | None = None, batch_size: int = 128,
    show_progress_bar: bool = False, prompt_batch_size: int = 256,
) -> np.ndarray:
    config = config or FeatureConfigV2()
    prompts = list(prompts)
    if not prompts or any(not isinstance(text, str) for text in prompts):
        raise ValueError('prompts must be a nonempty sequence of strings')
    tokenizer = tokenizer or getattr(encoder, 'tokenizer', None)
    if tokenizer is None:
        raise ValueError('a tokenizer is required for v2 features')
    special = tokenizer.num_special_tokens_to_add(pair=False)
    if config.window_tokens + special > getattr(encoder, 'max_seq_length', 128):
        raise ValueError('window token budget exceeds encoder max_seq_length including special tokens')
    result = []
    for offset in range(0, len(prompts), prompt_batch_size):
        subset = prompts[offset:offset + prompt_batch_size]
        windows = [prompt_windows(text, tokenizer, config) for text in subset]
        flattened = [part for parts in windows for part in parts]
        vectors = np.asarray(encoder.encode(
            flattened, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=show_progress_bar), dtype=np.float32)
        if vectors.ndim != 2 or len(vectors) != len(flattened) or not np.isfinite(vectors).all():
            raise ValueError('sentence encoder returned invalid embeddings')
        position = 0
        for text, parts in zip(subset, windows):
            current = vectors[position:position + len(parts)]
            result.append(np.concatenate((current[0], current.mean(axis=0), current[-1], structure_features(text))))
            position += len(parts)
    return np.asarray(result, dtype=np.float32)


def feature_fingerprint(prompts: Sequence[str], config: FeatureConfigV2) -> str:
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode())
    digest.update(json.dumps(STRUCTURE_FEATURE_NAMES).encode())
    for text in prompts:
        encoded = text.encode('utf-8')
        digest.update(len(encoded).to_bytes(8, 'big'))
        digest.update(encoded)
    return digest.hexdigest()


def cache_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_feature_cache(prompts: Sequence[str], cache_dir: str | Path, *, config: FeatureConfigV2 | None = None):
    config = config or FeatureConfigV2()
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / 'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('fingerprint') != feature_fingerprint(prompts, config):
        raise ValueError('v2 feature cache does not match prompt order, encoder revision, or feature configuration')
    checksums = metadata.get('file_sha256', {})
    for filename in ('features.npy', 'row_index.npy'):
        if checksums.get(filename) != cache_file_sha256(cache_dir / filename):
            raise ValueError(f'v2 feature cache integrity check failed: {filename}')
    features = np.load(cache_dir / 'features.npy', mmap_mode='r')
    row_index = np.load(cache_dir / 'row_index.npy')
    if (features.shape != (metadata['unique_prompts'], metadata['dimensions']) or
            row_index.shape != (len(prompts),) or
            (len(row_index) and (row_index.min() < 0 or row_index.max() >= len(features)))):
        raise ValueError('v2 feature cache has invalid matrix or row index')
    if features.dtype != np.float32 or row_index.dtype != np.int32:
        raise ValueError('v2 feature cache has invalid matrix or row index dtype')
    if not np.isfinite(features).all():
        raise ValueError('v2 feature cache contains nonfinite features')
    return features, row_index


def build_feature_cache(
    prompts: Sequence[str], encoder: object, cache_dir: str | Path, *, tokenizer: object | None = None,
    config: FeatureConfigV2 | None = None, batch_size: int = 128,
):
    config = config or FeatureConfigV2()
    prompts = list(prompts)
    if not prompts or any(not isinstance(text, str) for text in prompts):
        raise ValueError('prompts must be a nonempty sequence of strings')
    cache_dir = Path(cache_dir)
    paths = [cache_dir / name for name in ('metadata.json', 'features.npy', 'row_index.npy')]
    if any(path.exists() for path in paths):
        if not all(path.exists() for path in paths):
            raise ValueError('incomplete v2 feature cache; use a separate empty cache directory')
        return load_feature_cache(prompts, cache_dir, config=config)
    lookup, unique, index = {}, [], []
    for text in prompts:
        if text not in lookup:
            lookup[text] = len(unique)
            unique.append(text)
        index.append(lookup[text])
    features = encode_prompt_features(unique, encoder, tokenizer=tokenizer, config=config,
                                     batch_size=batch_size, show_progress_bar=True)
    row_index = np.asarray(index, dtype=np.int32)
    metadata = {'fingerprint': feature_fingerprint(prompts, config), 'feature_config': asdict(config),
                'row_count': len(prompts), 'unique_prompts': len(unique), 'dimensions': features.shape[1],
                'semantic_dimensions': features.shape[1] - len(STRUCTURE_FEATURE_NAMES),
                'structure_dimensions': len(STRUCTURE_FEATURE_NAMES), 'structure_feature_names': list(STRUCTURE_FEATURE_NAMES)}
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(paths[1], features)
    np.save(paths[2], row_index)
    metadata['file_sha256'] = {path.name: cache_file_sha256(path) for path in paths[1:]}
    paths[0].write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return features, row_index
