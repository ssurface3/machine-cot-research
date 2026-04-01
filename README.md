# Machine-Centric Chain-of-Thought: Efficiency-Accuracy Pareto Frontier

Can LLMs reason just as well with compressed, non-human-readable
Chain-of-Thought? This project systematically measures accuracy
degradation across 5 levels of CoT compression, from full English to
extreme token minimisation, to find the breaking point.

Runs 100% locally on consumer GPUs (7x RTX 2080 Ti). No API calls.

---

## Prerequisites

- Python 3.10+
- CUDA-capable GPUs (tested on 7x RTX 2080 Ti, 11 GB each)
- ~60 GB free disk space (model weights + outputs)

## Installation

```bash
git clone https://github.com/<your-username>/machine-cot-research.git
cd machine-cot-research
pip install -r requirements.txt
```

## Project Structure

```
config.py                 -- All settings: models, paths, hyperparameters
prompts.py                -- Prompt templates for 5 compression levels
utils.py                  -- Dataset loading, answer extraction, I/O helpers
phase1_data_synthesis.py  -- Teacher model generates CoT at each level
phase2_finetune.py        -- QLoRA fine-tune student model on each level
phase3_evaluation.py      -- Run GSM8K test set, measure accuracy/tokens/speed
phase4_analysis.py        -- Compute metrics, generate paper-ready figures
run_all.sh                -- Run the full pipeline with one command
requirements.txt
data/                     -- Auto-populated by HuggingFace datasets
outputs/                  -- Synthetic data and evaluation JSON files
models/                   -- Saved LoRA adapter weights
figures/                  -- Generated plots (PNG)
```

## Compression Levels

| Level | Name       | Description                                  |
|-------|------------|----------------------------------------------|
| 0     | Verbose    | Full natural-language, step-by-step           |
| 1     | Concise    | Bullet points, no filler words                |
| 2     | Symbolic   | Pure math equations and variable assignments  |
| 3     | Shorthand  | Single-letter codes, operator symbols only    |
| 4     | Extreme    | Absolute minimum tokens (target: 15 or fewer) |

## How to Run

### Smoke test (verify setup works before committing hours of compute)

```bash
python phase1_data_synthesis.py --level 0 --num-samples 10
python phase2_finetune.py --level 0 --max-samples 64 --epochs 1
python phase3_evaluation.py --level 0
python phase4_analysis.py
```

### Full pipeline

```bash
bash run_all.sh
```

Or run phases one at a time:

```bash
# Phase 1: Teacher generates CoT for 5000 questions x 5 levels (~3-5 hours)
python phase1_data_synthesis.py

# Phase 2: Fine-tune student model on each level (~1-2 hours per level)
python phase2_finetune.py

# Phase 3: Evaluate all 5 student models on GSM8K test set (~1-2 hours)
python phase3_evaluation.py

# Phase 4: Compute metrics and generate figures (seconds)
python phase4_analysis.py
```

Each phase supports a `--level N` flag to process a single level (0-4).

### Expected runtime

| Phase   | What it does                         | Time estimate    |
|---------|--------------------------------------|------------------|
| Phase 1 | 25,000 teacher generations           | 3-5 hours        |
| Phase 2 | 5 QLoRA fine-tuning runs             | 6-10 hours total |
| Phase 3 | 5 x 1,319 test inferences           | 1-2 hours        |
| Phase 4 | Aggregation and plotting             | < 1 minute       |

## Outputs

After a full run you will have:

- `outputs/synth_level_{0-4}.json` -- Teacher-generated CoT per level
- `outputs/eval_level_{0-4}.json` -- Student predictions with per-question metrics
- `outputs/results.json` -- Aggregated summary (accuracy, tokens, latency, efficiency)
- `figures/pareto_frontier.png` -- Accuracy vs. token count (Pareto curve)
- `figures/degradation_curve.png` -- Accuracy bar chart across levels
- `figures/efficiency_score.png` -- Accuracy-per-token by level
- `figures/latency.png` -- Inference speed by level

## Configuration

All tuneable parameters are in `config.py`:

- `TEACHER_MODEL` / `TEACHER_TENSOR_PARALLEL` -- teacher model and GPU count
- `STUDENT_MODEL` -- student architecture
- `NUM_TRAIN_SAMPLES` -- how many GSM8K training examples to use
- `LORA_R`, `TRAIN_LR`, `TRAIN_EPOCHS` -- fine-tuning hyperparameters
- `TEACHER_TEMPERATURE`, `TEACHER_MAX_TOKENS` -- generation settings

If Qwen-72B does not fit on your hardware, set:

```python
TEACHER_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
TEACHER_TENSOR_PARALLEL = 2
```

## License

MIT
