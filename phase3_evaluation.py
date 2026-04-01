#!/usr/bin/env python3
"""
Phase 3 — Evaluation
=====================
Run every fine-tuned student model on the GSM8K test set and record
accuracy, token usage, and inference speed.

Usage:
    python phase3_evaluation.py              # evaluate all levels
    python phase3_evaluation.py --level 2    # evaluate one level
"""

import argparse
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import config
from utils import (
    count_tokens,
    extract_gold_answer,
    extract_pred_answer,
    format_eval_prompt,
    is_correct,
    load_gsm8k,
    save_json,
)


def load_student(level: int):
    """Load the base model + LoRA adapter for a given level."""
    adapter_path = config.model_path(level)
    print(f"  Loading base model … {config.STUDENT_MODEL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(adapter_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@torch.inference_mode()
def evaluate_level(level: int, dataset) -> list[dict]:
    """Generate answers for every test question and score them."""
    print(f"\n{'='*60}")
    print(f"  Evaluating Level {level} ({config.LEVEL_NAMES[level]})")
    print(f"{'='*60}")

    model, tokenizer = load_student(level)
    records = []

    for i, row in enumerate(dataset):
        prompt = format_eval_prompt(row["question"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        t0 = time.perf_counter()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=config.EVAL_MAX_NEW_TOKENS,
            temperature=config.EVAL_TEMPERATURE,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        generated_ids = output_ids[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        gold = extract_gold_answer(row["answer"])
        pred = extract_pred_answer(generated_text)
        cot_tokens = len(generated_ids)

        records.append({
            "question": row["question"],
            "gold_answer": gold,
            "pred_answer": pred,
            "correct": is_correct(pred, gold),
            "cot": generated_text,
            "cot_tokens": cot_tokens,
            "latency_ms": round(latency_ms, 2),
            "level": level,
        })

        if (i + 1) % 50 == 0:
            acc_so_far = sum(r["correct"] for r in records) / len(records)
            print(f"  [{i+1}/{len(dataset)}]  running accuracy: {acc_so_far:.3f}")

    acc = sum(r["correct"] for r in records) / len(records)
    avg_tok = sum(r["cot_tokens"] for r in records) / len(records)
    avg_ms = sum(r["latency_ms"] for r in records) / len(records)
    print(f"\n  -- Level {level} results --")
    print(f"     Accuracy      : {acc:.4f}  ({sum(r['correct'] for r in records)}/{len(records)})")
    print(f"     Avg CoT tokens: {avg_tok:.1f}")
    print(f"     Avg latency   : {avg_ms:.1f} ms")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Evaluation")
    parser.add_argument("--level", type=int, default=None,
                        help="Evaluate only this level (0-4). Default: all.")
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else config.LEVELS
    dataset = load_gsm8k(config.DATASET_SPLIT_TEST)
    print(f"Loaded {len(dataset)} test samples from GSM8K.")

    all_records = {}
    for level in levels:
        records = evaluate_level(level, dataset)
        save_json(records, config.eval_path(level))
        all_records[level] = records

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"{'Level':<10} {'Accuracy':>10} {'Avg Tokens':>12} {'Avg ms':>10}")
    print(f"{'-'*42}")
    for level in levels:
        recs = all_records[level]
        n = len(recs)
        acc = sum(r["correct"] for r in recs) / n
        tok = sum(r["cot_tokens"] for r in recs) / n
        ms = sum(r["latency_ms"] for r in recs) / n
        print(f"{level:<10} {acc:>10.4f} {tok:>12.1f} {ms:>10.1f}")

    print(f"\nPhase 3 complete.")


if __name__ == "__main__":
    main()
