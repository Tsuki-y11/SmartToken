"""Normalize published model-execution records and prevent question leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from deep_learning.dl.catalog import LLMROUTER_IDS, XROUTE_GENERIC, build_catalog, canonical_model_id


COLUMNS = (
    "source", "dataset", "source_split", "model_id", "raw_model_name",
    "prompt", "original_query", "input_tokens", "output_tokens",
)


def _text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _finish(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=COLUMNS)
    rows = rows.loc[:, COLUMNS].copy()
    for column in ("prompt", "original_query", "raw_model_name", "dataset"):
        rows[column] = rows[column].map(_text)
    rows["input_tokens"] = pd.to_numeric(rows["input_tokens"], errors="coerce")
    rows["output_tokens"] = pd.to_numeric(rows["output_tokens"], errors="coerce")
    valid = (
        rows["prompt"].ne("") & rows["original_query"].ne("")
        & rows["model_id"].ne("")
        & rows["input_tokens"].gt(0) & rows["output_tokens"].gt(0)
    )
    rows = rows.loc[valid].copy()
    rows["input_tokens"] = rows["input_tokens"].astype("int32")
    rows["output_tokens"] = rows["output_tokens"].astype("int32")
    return rows.reset_index(drop=True)


def normalize_xroute_rows(frame: pd.DataFrame, source_split: str) -> pd.DataFrame:
    """Map xRouteBench's llmrouter_generic schema to the common columns."""
    required = {"task_name", "query", "model_name", "input_tokens", "output_tokens"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"xRouteBench missing columns: {sorted(missing)}")
    names = frame["model_name"].map(_text)
    accepted = {canonical_model_id(name) for name in XROUTE_GENERIC}
    rows = pd.DataFrame({
        "source": "xroutebench",
        "dataset": frame["task_name"],
        "source_split": source_split,
        "model_id": names.map(lambda name: canonical_model_id(name) if name else ""),
        "raw_model_name": names,
        "prompt": frame["query"],
        "original_query": frame["query"],
        "input_tokens": frame["input_tokens"],
        "output_tokens": frame["output_tokens"],
    })
    return _finish(rows.loc[rows["model_id"].isin(accepted)])


def validate_xroute_coverage(model_ids: set[str]) -> None:
    """Reject incomplete releases before writing the fixed historical catalog."""
    expected = {canonical_model_id(name) for name in XROUTE_GENERIC}
    missing = expected - set(model_ids)
    if missing:
        raise ValueError(f"missing published xRoute models: {sorted(missing)}")


def normalize_llmrouter_file(path: str | Path) -> pd.DataFrame:
    """Read one result JSON, accepting only the published 33 model IDs."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    name = document.get("model_name") or path.parent.name
    model_id = canonical_model_id(name)
    if model_id not in LLMROUTER_IDS or document.get("demo") is True:
        return pd.DataFrame(columns=COLUMNS)
    records = document.get("records") or []
    if not isinstance(records, list):
        raise ValueError(f"records must be a list: {path}")
    dataset = document.get("dataset_name") or path.parents[2].name
    source_split = document.get("split") or path.parents[1].name
    rows = pd.DataFrame([
        {
            "source": "llmrouterbench",
            "dataset": dataset,
            "source_split": source_split,
            "model_id": model_id,
            "raw_model_name": name,
            "prompt": record.get("prompt"),
            "original_query": record.get("origin_query"),
            "input_tokens": record.get("prompt_tokens"),
            "output_tokens": record.get("completion_tokens"),
        }
        for record in records if isinstance(record, dict)
    ], columns=COLUMNS)
    return _finish(rows)


def _question_key(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def group_split(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign a deterministic 70/15/15 split by original question text."""
    rows = frame.copy()
    def assign(question: str) -> str:
        key = f"{seed}\0{_question_key(question)}".encode("utf-8")
        fraction = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64
        return "train" if fraction < 0.70 else "valid" if fraction < 0.85 else "test"
    rows["split"] = rows["original_query"].map(assign)
    return rows


def prepare_dataset(raw_dir: str | Path, output_dir: str | Path, seed: int = 42) -> dict:
    """Build one clean table, 50-model catalog, and auditable summary."""
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_frames = []
    for source_split in ("train", "test"):
        path = raw_dir / f"{source_split}.parquet"
        raw_frames.append(normalize_xroute_rows(pd.read_parquet(path), source_split))
    validate_xroute_coverage(set(pd.concat(raw_frames, ignore_index=True)["model_id"]))
    catalog = build_catalog(XROUTE_GENERIC)
    files = sorted((raw_dir / "llmrouterbench" / "extracted" / "bench-release").rglob("*.json"))
    if not files:
        raise FileNotFoundError("LLMRouterBench result JSON files were not found")
    frames = raw_frames + [normalize_llmrouter_file(path) for path in files]
    rows = pd.concat(frames, ignore_index=True)
    before_dedup = len(rows)
    rows = rows.drop_duplicates(subset=["source", "dataset", "model_id", "prompt", "original_query"], keep="first")
    rows = group_split(rows, seed=seed)
    model_ids = set(rows["model_id"])
    absent = sorted({item["model_id"] for item in catalog} - model_ids)
    if absent:
        raise ValueError(f"catalog models with no valid records: {absent}")
    rows.to_parquet(output_dir / "records.parquet", index=False)
    (output_dir / "catalog.json").write_text(json.dumps(catalog, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "seed": seed,
        "catalog_models": len(catalog),
        "records": len(rows),
        "duplicates_removed": before_dedup - len(rows),
        "source_counts": rows["source"].value_counts().to_dict(),
        "split_counts": rows["split"].value_counts().to_dict(),
        "unique_questions": rows["original_query"].map(_question_key).nunique(),
        "per_model_counts": rows["model_id"].value_counts().sort_index().to_dict(),
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(args.raw_dir, args.output_dir, args.seed), indent=2, ensure_ascii=False))
