#!/usr/bin/env python3
"""
Phase 2 — Fine-Tuning (Student)
================================
Fine-tune Llama-3-8B with QLoRA on each compression level's synthetic data.

Usage:
    # Fine-tune all levels:
    python phase2_finetune.py

    # Fine-tune a single level:
    python phase2_finetune.py --level 2

    # Quick smoke-test (64 samples, 1 epoch):
    python phase2_finetune.py --level 0 --max-samples 64 --epochs 1
"""

import argparse

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

import config
from utils import format_train_sample, load_json


def build_dataset(level: int, max_samples: int | None = None) -> Dataset:
    """Load Phase-1 records and convert to HF Dataset with a 'text' column."""
    records = load_json(config.synth_path(level))
    # Only keep samples where the teacher got the right answer
    records = [r for r in records if r["correct"]]
    if max_samples:
        records = records[:max_samples]

    texts = [
        format_train_sample(r["question"], r["cot"])
        for r in records
    ]
    print(f"  Level {level}: {len(texts)} correct samples for training")
    return Dataset.from_dict({"text": texts})


def train_one_level(level: int, max_samples: int | None, epochs: int):
    """QLoRA fine-tune on a single compression level."""
    print(f"\n{'='*60}")
    print(f"  Fine-tuning Level {level} ({config.LEVEL_NAMES[level]})")
    print(f"{'='*60}")

    # ── Tokenizer ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        config.STUDENT_TOKENIZER, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── 4-bit quantised base model ───────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.STUDENT_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA adapters ────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ── Dataset ──────────────────────────────────────────────────────────
    dataset = build_dataset(level, max_samples)

    # ── Training args ────────────────────────────────────────────────────
    output_dir = config.model_path(level)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=config.TRAIN_GRAD_ACCUM,
        learning_rate=config.TRAIN_LR,
        warmup_ratio=config.TRAIN_WARMUP_RATIO,
        weight_decay=config.TRAIN_WEIGHT_DECAY,
        fp16=config.TRAIN_FP16,
        bf16=config.TRAIN_BF16,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=config.SEED,
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=config.TRAIN_MAX_SEQ_LEN,
    )
    trainer.train()

    # ── Save adapter weights ─────────────────────────────────────────────
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  Saved adapter -> {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Phase 2: Fine-Tuning")
    parser.add_argument("--level", type=int, default=None,
                        help="Train only this level (0-4). Default: all.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples (for smoke tests).")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    args = parser.parse_args()

    levels = [args.level] if args.level is not None else config.LEVELS

    for level in levels:
        train_one_level(level, args.max_samples, args.epochs)

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
