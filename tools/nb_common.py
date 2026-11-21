"""Shared cells for the notebook builder."""

SETUP = r'''# --- Setup (implementation guide, section 2). Run once per session. ------------------------------
import os, sys, subprocess, random, json
from pathlib import Path

REPO_URL = "https://github.com/adiarora28/dsa1-conformal-abstention.git"   # <- set once for the team
ON_KAGGLE = Path("/kaggle").exists()
REPO_DIR = Path("/kaggle/working/dsa1-conformal-abstention") if ON_KAGGLE else Path.cwd().resolve().parent
if ON_KAGGLE and not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)], check=True)
if ON_KAGGLE:  # upgrade only what Kaggle's image may lack; never reinstall torch
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56",
                    "accelerate", "bitsandbytes", "sentence-transformers", "xgboost"], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))
os.chdir(REPO_DIR)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
try:
    import torch
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    print("PyTorch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
          "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
except ImportError:
    print("PyTorch not installed (this notebook can run on CPU)")

from dsa1 import config as C
from dsa1.cache_io import link_cache_roots
CACHE_ROOT = link_cache_roots("/kaggle/input", "/kaggle/working/cache_merged") if ON_KAGGLE else Path("cache")
print("CACHE_ROOT:", CACHE_ROOT)
if ON_KAGGLE:  # bring outputs of earlier notebooks (attached as inputs) into this working copy
    import shutil
    Path("results").mkdir(exist_ok=True)
    for p in Path("/kaggle/input").rglob("semantic_entropy"):
        if p.is_dir():
            shutil.copytree(p, "results/semantic_entropy", dirs_exist_ok=True)
    for name in ["hand_check.csv", "failure_cases.csv"]:
        hits = sorted(Path("/kaggle/input").rglob(name))
        if hits and not (Path("results") / name).exists():
            shutil.copy(hits[0], Path("results") / name)
            print("copied", hits[0])
for p in sorted(Path(CACHE_ROOT).glob("*/*")):
    print("  ", p.relative_to(CACHE_ROOT))
'''

SCORES_ROOT = r'''# Score tables written by Task 1 (attach Sai Akhil's notebook output as an input on Kaggle).
# Until they exist, token-probability scores are computed directly from the cache so you can start now.
from dsa1 import experiments as E

def find_scores_root():
    for p in Path("/kaggle/input").rglob("scores") if ON_KAGGLE else []:
        if p.is_dir() and any(p.glob("qwen3-*/*.parquet")):
            return p
    return Path("results/scores")

SCORES_ROOT = find_scores_root()
print("SCORES_ROOT:", SCORES_ROOT)

def get_scores(model, dataset):
    path = Path(SCORES_ROOT) / model / f"{dataset}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    print(f"[{model}/{dataset}] no Task 1 score table yet - using token probability from the cache")
    return E.base_scores(CACHE_ROOT, model, dataset)
'''

SPLITS = r'''from dsa1.splits import load_splits
SPLITS = load_splits("data/splits/splits.json")
SEEDS = [int(s) for s in SPLITS["seeds"]]
TEST_KEYS = {"triviaqa_pool": "test_triviaqa", "bioasq": "test_bioasq", "mmlu_med": "test_mmlu_med"}
print("seeds:", SEEDS, "| sizes (seed 0):", {k: len(v) for k, v in SPLITS["seeds"]["0"].items()})
'''

REFS = """## 10. References

- Bibliography numbers refer to `DSA 1 - Bibliography.xlsx`.
- Datasets: TriviaQA (Joshi et al., 2017), BioASQ Task B 14b (Krithara et al., 2023), MMLU (Hendrycks et al., 2021).
- Models: Qwen3 (Qwen Team, 2025); DeBERTa-large-MNLI (He et al., 2021); all-MiniLM-L6-v2 (Reimers & Gurevych, 2019).
- Code: shared `dsa1` package in this repository; LM-Polygraph (Vashurin et al., 2025) used for cross-checks;
  DS-CP reference code https://github.com/zhexiaolin/CP (Lin et al., 2026).
"""
