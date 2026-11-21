"""Step 0.2 - build the fixed question tables once and commit them.

Usage:
    python scripts/prepare_questions.py --bioasq-json data/raw/training14b.json
"""
import argparse

import _path  # noqa: F401
from dsa1 import data

ap = argparse.ArgumentParser()
ap.add_argument("--bioasq-json", required=True, help="BioASQ Task B training JSON (manual download)")
ap.add_argument("--root", default="data/questions")
ap.add_argument("--skip-mmlu-other", action="store_true", help="skip the optional non-medical MMLU pool")
args = ap.parse_args()

tables = {
    "triviaqa": data.load_triviaqa(),
    "bioasq": data.load_bioasq(args.bioasq_json),
    "mmlu_med": data.load_mmlu_medical(),
}
if not args.skip_mmlu_other:
    tables["mmlu_other"] = data.load_mmlu_nonmedical()
for name, df in tables.items():
    path = data.save_questions(df, name, args.root)
    extra = df["role"].value_counts().to_dict() if name == "triviaqa" else ""
    print(f"{name:10s} {len(df):6d} questions -> {path} {extra}")
if len(tables["mmlu_med"]) != 945:
    print("WARNING: expected 945 MMLU medical questions, got", len(tables["mmlu_med"]))
