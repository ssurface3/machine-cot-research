#!/usr/bin/env python3
"""
Phase 1 — Data Synthesis
========================
Use the Teacher model (Qwen2.5-72B-Instruct via vLLM) to generate
5 compression levels of CoT for each GSM8K training question.

Usage:
    # Generate all levels:
    python phase1_data_synthesis.py

    # Generate a single level:
    python phase1_data_synthesis.py --level 2

    # Quick test with 10 samples:
    python phase1_data_synthesis.py --level 0 --num-samples 10
"""

import argparse
import time

from vllm import LLM, SamplingParams

import config
from prompts import build_prompt
from utils import extract_gold_answer, extract_pred_answer, load_gsm8k, save_json


def create_teacher() -> LLM:
    """Load the teacher model with tensor parallelism across all GPUs."""
    print(f"Loading teacher model: {config.TEACHER_MODEL}")
    print(f"  Quantization : {config.TEACHER_QUANTIZATION}")
    print(f"  Tensor-parallel : {config.TEACHER_TENSOR_PARALLEL} GPUs")
    return LLM(
        model=config.TEACHER_MODEL,
        quantization=config.TEACHER_QUANTIZATION,
        tensor_parallel_size=config.TEACHER_TENSOR_PARALLEL,
        max_model_len=config.TEACHER_MAX_MODEL_LEN,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
    )


def generate_level(llm: LLM, dataset, level: int) -> list[dict]:
    """
    Ask the teacher to solve every question in *dataset* at the given
    compression *level*.  Returns a list of record dicts.
    """
    prompts = [build_prompt(level, row["question"]) for row in dataset]
    sampling = SamplingParams(
        temperature=config.TEACHER_TEMPERATURE,
        top_p=config.TEACHER_TOP_P,
        max_tokens=config.TEACHER_MAX_TOKENS,
    )

    print(f"\n> Generating Level {level} ({config.LEVEL_NAMES[level]}) "
          f"for {len(prompts)} questions …")
    t0 = time.perf_counter()
    outputs = llm.chat(prompts, sampling_params=sampling)
    elapsed = time.perf_counter() - t0
    print(f"  Done in {elapsed:.1f}s  ({elapsed / len(prompts):.2f}s / question)")

    records = []
    for row, output in zip(dataset, outputs):
        generated = output.outputs[0].text
        gold = extract_gold_answer(row["answer"])
        pred = extract_pred_answer(generated)
        records.append({
            "question": row["question"],
            "gold_answer": gold,
            "cot": generated,
            "pred_answer": pred,
            "correct": pred == gold,
            "level": level,
        })

    correct = sum(r["correct"] for r in records)
    print(f"  Teacher accuracy: {correct}/{len(records)} "
          f"({100 * correct / len(records):.1f}%)")
    return records


def main():
    parser = argparse.ArgumentParser(description="Phase 1: Data Synthesis")
    parser.add_argument("--level", type=int, default=None,
                        help="Generate only this level (0-4). Default: all.")
    parser.add_argument("--num-samples", type=int, default=config.NUM_TRAIN_SAMPLES,
                        help="Number of GSM8K training samples to use.")
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else config.LEVELS
    dataset = load_gsm8k("train", n=args.num_samples)
    print(f"Loaded {len(dataset)} training samples from GSM8K.")

    llm = create_teacher()

    for level in levels:
        records = generate_level(llm, dataset, level)
        save_json(records, config.synth_path(level))

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
