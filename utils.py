"""
Shared utilities: dataset loading, answer extraction, token counting, I/O.
"""

import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

import config


# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_gsm8k(split: str = "train", n: int | None = None, seed: int = config.SEED):
    """Load GSM8K and optionally sub-sample *n* examples (deterministic)."""
    ds = load_dataset(config.DATASET_NAME, "main", split=split)
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n))
    return ds


def extract_gold_answer(answer_text: str) -> str:
    """Pull the numeric answer after #### from the GSM8K ground-truth field."""
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return _normalise_number(match.group(1).strip())
    return ""


def extract_pred_answer(model_output: str) -> str:
    """Pull the numeric answer after #### from a model generation."""
    match = re.search(r"####\s*(.+)", model_output)
    if match:
        return _normalise_number(match.group(1).strip())
    return ""


def _normalise_number(s: str) -> str:
    """Remove commas and trailing whitespace so '1,200' == '1200'."""
    return s.replace(",", "").strip()


def is_correct(pred: str, gold: str) -> bool:
    """Compare two numeric answer strings."""
    return pred == gold


# ── Token counting ───────────────────────────────────────────────────────────

_tokenizer_cache: dict[str, AutoTokenizer] = {}


def count_tokens(text: str, model_name: str = config.STUDENT_MODEL) -> int:
    """Count tokens using the given model's tokenizer (cached)."""
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = AutoTokenizer.from_pretrained(model_name)
    return len(_tokenizer_cache[model_name].encode(text))


# ── I/O ──────────────────────────────────────────────────────────────────────

def save_json(data, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(data) if isinstance(data, list) else '-'} records -> {path}")


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


# ── Formatting helpers ───────────────────────────────────────────────────────

def format_train_sample(question: str, cot: str) -> str:
    """Format a question + CoT into the instruction-following template
    used during fine-tuning.  Matches Llama-3 chat format."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Solve the following math problem. Show your work, then give the "
        f"final answer after ####.\n\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{cot}<|eot_id|><|end_of_text|>"
    )


def format_eval_prompt(question: str) -> str:
    """Build the inference-time prompt (no answer included)."""
    return (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Solve the following math problem. Show your work, then give the "
        f"final answer after ####.\n\n{question}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
