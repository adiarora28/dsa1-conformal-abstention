"""Step 0.7 - write data/splits/splits.json (five seeds) and the precision-shift subset for Task 3.

Usage:
    python scripts/make_splits.py
"""
import json

import numpy as np

import _path  # noqa: F401
from dsa1 import data, splits

tq = data.read_questions("triviaqa")
bio = data.read_questions("bioasq")
mm = data.read_questions("mmlu_med")
sp = splits.make_splits(tq, bio, mm)
path = splits.save_splits(sp)
print("wrote", path)
for s, d in sp["seeds"].items():
    print(s, {k: len(v) for k, v in d.items()})

# Task 3 precision shift: 500 TriviaQA test questions of seed 0 + 500 BioASQ questions, fixed once.
rng = np.random.default_rng(123)
s0 = sp["seeds"]["0"]
subset = {
    "triviaqa_pool": sorted(rng.choice(s0["test_triviaqa"], 500, replace=False).tolist()),
    "bioasq": sorted(rng.choice(bio["qid"].tolist(), 500, replace=False).tolist()),
}
with open("data/splits/precision_subset.json", "w") as f:
    json.dump(subset, f, indent=1)
print("wrote data/splits/precision_subset.json")
