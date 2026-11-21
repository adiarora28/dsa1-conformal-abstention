"""Single source of truth for settings shared by all three tasks.

Change values here, never inside a task notebook, so every task sees the same setup.
"""
from __future__ import annotations

MODELS = {
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-8b": "Qwen/Qwen3-8B",
}
# Default numerical precision per model (guide Step 0.3). 8B runs in 4-bit to fit free GPUs.
DEFAULT_PRECISION = {"qwen3-1.7b": "fp16", "qwen3-4b": "fp16", "qwen3-8b": "4bit"}

MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
]  # 135 + 265 + 173 + 100 + 272 = 945 test questions

# Question pools (guide Step 0.2)
TRIVIAQA_N_PROBE = 2000
TRIVIAQA_N_POOL = 3000
MMLU_N_NONMEDICAL = 2000  # optional calibration domain for Task 2's DS-CP check on Qwen3
POOL_SEED = 0

# Splits (guide Step 0.7)
SPLIT_SEEDS = [0, 1, 2, 3, 4]
N_CAL = 1000
N_TEST_TRIVIA = 1000
N_TEST_BIOASQ = 1000

# Generation (guide Step 0.3)
MAX_NEW_TOKENS = 32
N_SAMPLES = 10
SAMPLE_TEMPERATURE = 1.0
SAMPLE_TOP_P = 0.95
HIDDEN_LAYER_FRACTIONS = (0.25, 0.5, 1.0)  # early, middle, last

# Evaluation
TARGETS = (0.05, 0.10, 0.20)
DELTA = 0.05

# Auxiliary models
NLI_MODEL = "microsoft/deberta-large-mnli"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# DS-CP domain classifier: XGBoost settings reported by Lin et al. (arXiv 2510.05566, Appendix C)
DSCP_XGB_PARAMS = dict(
    max_depth=3,
    n_estimators=50,
    learning_rate=0.2,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=5,
    reg_lambda=10,
    random_state=42,
    n_jobs=1,
)
