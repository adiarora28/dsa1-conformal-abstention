"""The five seeded splits shared by every notebook (guide Step 0.7).

splits.json layout:
{
  "probe": [qid, ...],                       # fixed TriviaQA probe-training set (Task 1)
  "seeds": {
    "0": {"cal": [...], "test_triviaqa": [...], "test_bioasq": [...], "test_mmlu_med": [...]},
    ...
  }
}
Labelled target questions never appear in "cal".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def make_splits(trivia: pd.DataFrame, bioasq: pd.DataFrame, mmlu_med: pd.DataFrame,
                seeds=C.SPLIT_SEEDS, n_cal: int = C.N_CAL, n_test_trivia: int = C.N_TEST_TRIVIA,
                n_test_bioasq: int = C.N_TEST_BIOASQ) -> dict:
    probe = trivia.loc[trivia["role"] == "probe", "qid"].tolist()
    pool = np.array(trivia.loc[trivia["role"] == "pool", "qid"].tolist())
    bio = np.array(bioasq["qid"].tolist())
    mmlu = mmlu_med["qid"].tolist()
    if n_cal + n_test_trivia > len(pool):
        raise ValueError("TriviaQA pool is smaller than calibration + test")
    out = {"probe": probe, "seeds": {}}
    for s in seeds:
        rng = np.random.default_rng(s)
        perm = rng.permutation(len(pool))
        cal = pool[perm[:n_cal]].tolist()
        test_t = pool[perm[n_cal:n_cal + n_test_trivia]].tolist()
        k = min(n_test_bioasq, len(bio))
        test_b = bio[rng.choice(len(bio), size=k, replace=False)].tolist()
        out["seeds"][str(s)] = {
            "cal": cal,
            "test_triviaqa": test_t,
            "test_bioasq": test_b,
            "test_mmlu_med": list(mmlu),  # all 945; the calibration redraw provides the variation
        }
    check_splits(out)
    return out


def check_splits(splits: dict) -> None:
    probe = set(splits["probe"])
    for s, d in splits["seeds"].items():
        cal, tt = set(d["cal"]), set(d["test_triviaqa"])
        assert not cal & tt, f"seed {s}: calibration and TriviaQA test overlap"
        assert not cal & probe and not tt & probe, f"seed {s}: probe questions leak into cal/test"
        assert all(q.startswith("tqa_") for q in cal), f"seed {s}: non-TriviaQA question in calibration"


def save_splits(splits: dict, path: str | Path = "data/splits/splits.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=1))
    return path


def load_splits(path: str | Path = "data/splits/splits.json") -> dict:
    splits = json.loads(Path(path).read_text())
    check_splits(splits)
    return splits


def mixture(trivia_test: list[str], bio_test: list[str], frac_bio: float, n: int = 1000,
            seed: int = 0) -> list[str]:
    """Task 2 graded shift: a test set of n questions with a share frac_bio of BioASQ questions."""
    rng = np.random.default_rng(seed)
    k_bio = int(round(frac_bio * n))
    k_tqa = n - k_bio
    if k_bio > len(bio_test) or k_tqa > len(trivia_test):
        raise ValueError("not enough questions for this mixture")
    b = rng.choice(bio_test, size=k_bio, replace=False).tolist() if k_bio else []
    t = rng.choice(trivia_test, size=k_tqa, replace=False).tolist() if k_tqa else []
    return t + b
