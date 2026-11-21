"""Step 0.5 - export 100 answers (50 TriviaQA, 50 BioASQ, Qwen3-4B) for manual label checking.

Usage:
    python scripts/hand_check_sheet.py --cache-root cache
Fill the column `human_correct` (1/0) in results/hand_check.csv, then compute agreement in a notebook.
"""
import argparse

import pandas as pd

import _path  # noqa: F401
from dsa1.cache_io import cache_dir, load_records

ap = argparse.ArgumentParser()
ap.add_argument("--cache-root", default="cache")
ap.add_argument("--model", default="qwen3-4b")
args = ap.parse_args()

parts = []
for ds in ["triviaqa_pool", "bioasq"]:
    rec = load_records(cache_dir(args.cache_root, args.model, ds))
    # 25 automatically-correct and 25 automatically-wrong answers per dataset
    for flag in (True, False):
        sub = rec[rec["correct_em"] == flag]
        parts.append(sub.sample(min(25, len(sub)), random_state=0))
sheet = pd.concat(parts)[["qid", "dataset", "question", "aliases", "answer", "correct_em", "f1"]]
sheet["aliases"] = sheet["aliases"].apply(lambda a: " | ".join(a[:10]))
sheet["human_correct"] = ""
sheet.to_csv("results/hand_check.csv", index=False)
print("wrote results/hand_check.csv with", len(sheet), "rows")
