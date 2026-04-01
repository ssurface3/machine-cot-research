"""
Central configuration for the Machine-CoT research project.
All paths, model names, hyperparameters, and hardware settings live here.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
MODEL_DIR = ROOT_DIR / "models"
FIGURE_DIR = ROOT_DIR / "figures"

# ── Hardware ─────────────────────────────────────────────────────────────────
NUM_GPUS = 7  # 7x RTX 2080 Ti
GPU_MEMORY_GB = 11  # per GPU

# ── Teacher model (Phase 1) ─────────────────────────────────────────────────
TEACHER_MODEL = "Qwen/Qwen2.5-72B-Instruct"
TEACHER_QUANTIZATION = "awq"  # 4-bit quantization for vLLM
TEACHER_TENSOR_PARALLEL = NUM_GPUS
TEACHER_MAX_MODEL_LEN = 4096
TEACHER_TEMPERATURE = 0.3
TEACHER_TOP_P = 0.95
TEACHER_MAX_TOKENS = 1024

# ── Student model (Phase 2) ──────────────────────────────────────────────────
STUDENT_MODEL = "meta-llama/Meta-Llama-3-8B"
STUDENT_TOKENIZER = STUDENT_MODEL

# ── Dataset ──────────────────────────────────────────────────────────────────
DATASET_NAME = "openai/gsm8k"
DATASET_SPLIT_TRAIN = "train"
DATASET_SPLIT_TEST = "test"
NUM_TRAIN_SAMPLES = 5000  # from ~7.5k train set
SEED = 42

# ── Compression levels ───────────────────────────────────────────────────────
LEVELS = [0, 1, 2, 3, 4]
LEVEL_NAMES = {
    0: "verbose",
    1: "concise",
    2: "symbolic",
    3: "shorthand",
    4: "extreme",
}

# ── QLoRA fine-tuning (Phase 2) ──────────────────────────────────────────────
LORA_R = 64
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]

TRAIN_BATCH_SIZE = 4
TRAIN_GRAD_ACCUM = 4
TRAIN_EPOCHS = 3
TRAIN_LR = 2e-4
TRAIN_MAX_SEQ_LEN = 2048
TRAIN_WARMUP_RATIO = 0.03
TRAIN_WEIGHT_DECAY = 0.01
TRAIN_FP16 = True
TRAIN_BF16 = False  # 2080 Ti doesn't support bf16

# ── Evaluation (Phase 3) ─────────────────────────────────────────────────────
EVAL_BATCH_SIZE = 16
EVAL_MAX_NEW_TOKENS = 512
EVAL_TEMPERATURE = 0.0  # greedy decoding for reproducibility

# ── Output file naming ──────────────────────────────────────────────────────
def synth_path(level: int) -> Path:
    return OUTPUT_DIR / f"synth_level_{level}.json"

def model_path(level: int) -> Path:
    return MODEL_DIR / f"student_level_{level}"

def eval_path(level: int) -> Path:
    return OUTPUT_DIR / f"eval_level_{level}.json"

RESULTS_PATH = OUTPUT_DIR / "results.json"
