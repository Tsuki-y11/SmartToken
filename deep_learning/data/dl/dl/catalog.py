"""Published historical model pools for offline token prediction."""

from __future__ import annotations


LLMROUTER_SMALL = (
    "DeepHermes-3-Llama-3-8B-Preview",
    "DeepSeek-R1-0528-Qwen3-8B",
    "DeepSeek-R1-Distill-Qwen-7B",
    "Fin-R1",
    "GLM-Z1-9B-0414",
    "Intern-S1-mini",
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-8B-UltraMedical",
    "Llama-3.1-Nemotron-Nano-8B-v1",
    "MiMo-7B-RL-0530",
    "MiniCPM4.1-8B",
    "NVIDIA-Nemotron-Nano-9B-v2",
    "OpenThinker3-7B",
    "Qwen2.5-Coder-7B-Instruct",
    "Qwen3-8B",
    "cogito-v1-preview-llama-8B",
    "gemma-2-9b-it",
    "glm-4-9b-chat",
    "granite-3.3-8b-instruct",
    "internlm3-8b-instruct",
)

# The archive calls the published GPT-5-medium entry "gpt-5".
LLMROUTER_FLAGSHIP = (
    "claude-sonnet-4",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gpt-5-chat",
    "gpt-5",
    "qwen3-235b-a22b-2507",
    "qwen3-235b-a22b-thinking-2507",
    "deepseek-v3-0324",
    "deepseek-v3.1-terminus",
    "deepseek-r1-0528",
    "glm-4.6",
    "kimi-k2-0905",
    "intern-s1",
)

XROUTE_GENERIC = (
    "cogito-v2-1-671b",
    "deepseek-v3.1",
    "gemma-2-9b-it",
    "gpt-oss-120b",
    "gpt-oss-20b",
    "llama-3-8b-instruct-lite",
    "llama-3.3-70b-instruct-turbo",
    "llama-4-maverick",
    "llama3-70b-instruct",
    "mistral-7b-instruct-v0.3",
    "mistral-small-3-24b-instruct",
    "mixtral-8x22b-instruct-v0.1",
    "mixtral-8x7b-instruct-v0.1",
    "qwen2.5-7b-instruct",
    "qwen2.5-7b-instruct-turbo",
    "qwen3-coder-next",
    "qwen3-next-80b-a3b-instruct",
    "rnj-1-instruct",
)

LLMROUTER_IDS = frozenset(name.casefold() for name in (*LLMROUTER_SMALL, *LLMROUTER_FLAGSHIP))


def canonical_model_id(name: str) -> str:
    """Merge capitalization variants only; version suffixes remain distinct."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model name must be a nonempty string")
    return name.strip().casefold()


def build_catalog(xroute_names: list[str] | tuple[str, ...]) -> list[dict]:
    """Return the case-insensitive union of published and xRoute model IDs."""
    by_id: dict[str, dict] = {}
    for pool, names in (("llmrouter_small", LLMROUTER_SMALL), ("llmrouter_flagship", LLMROUTER_FLAGSHIP), ("xroute", xroute_names)):
        for name in names:
            model_id = canonical_model_id(name)
            row = by_id.setdefault(model_id, {"model_id": model_id, "display_name": name, "pools": []})
            if pool not in row["pools"]:
                row["pools"].append(pool)
    return [by_id[model_id] for model_id in sorted(by_id)]
