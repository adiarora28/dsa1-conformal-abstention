"""Load the three benchmarks into one question-table format (guide Step 0.2).

Open-ended tables (TriviaQA, BioASQ) have columns:
    qid, dataset, question, aliases (list[str]), role
Multiple-choice tables (MMLU) have columns:
    qid, dataset, question, choices (list[str]), answer_idx (int), subject, role

Question tables are written once to data/questions/*.parquet by scripts/prepare_questions.py and
committed, so every member and every model uses exactly the same questions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------------------- TriviaQA
def load_triviaqa(n_probe: int = C.TRIVIAQA_N_PROBE, n_pool: int = C.TRIVIAQA_N_POOL,
                  seed: int = C.POOL_SEED) -> pd.DataFrame:
    """Draw a fixed probe-training set and a calibration/test pool from the TriviaQA validation split."""
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    df = ds.to_pandas()
    df = df.drop_duplicates("question_id").reset_index(drop=True)

    def aliases(ans) -> list[str]:
        out = list(ans.get("normalized_aliases", [])) + list(ans.get("aliases", []))
        for k in ("normalized_value", "value"):
            if ans.get(k):
                out.append(ans[k])
        return sorted(set(a for a in out if a))

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))[: n_probe + n_pool]
    sel = df.iloc[idx].reset_index(drop=True)
    out = pd.DataFrame({
        "qid": "tqa_" + sel["question_id"].astype(str),
        "dataset": "triviaqa",
        "question": sel["question"].str.strip(),
        "aliases": [aliases(a) for a in sel["answer"]],
        "role": ["probe"] * n_probe + ["pool"] * n_pool,
    })
    return out


# ----------------------------------------------------------------------------- BioASQ
def _flatten(x) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    out: list[str] = []
    for item in x:
        out.extend(_flatten(item))
    return out


def load_bioasq(json_path: str | Path) -> pd.DataFrame:
    """Factoid questions from the BioASQ Task B training JSON (downloaded manually after registration)."""
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for q in raw["questions"]:
        if q.get("type") != "factoid":
            continue
        aliases = sorted(set(a.strip() for a in _flatten(q.get("exact_answer")) if a and a.strip()))
        if not aliases:
            continue
        rows.append({
            "qid": "bio_" + str(q["id"]),
            "dataset": "bioasq",
            "question": q["body"].strip(),
            "aliases": aliases,
            "role": "target",
        })
    df = pd.DataFrame(rows).drop_duplicates("qid").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- MMLU
def load_mmlu(subjects: list[str] | None = None, exclude: list[str] | None = None,
              n: int | None = None, seed: int = C.POOL_SEED, split: str = "test",
              dataset_name: str = "mmlu_med", role: str = "target") -> pd.DataFrame:
    """MMLU questions, either a list of subjects or everything except `exclude` (optionally subsampled)."""
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split=split)
    df = ds.to_pandas()
    if subjects is not None:
        df = df[df["subject"].isin(subjects)]
    if exclude is not None:
        df = df[~df["subject"].isin(exclude)]
    df = df.reset_index(drop=True)
    df["row"] = np.arange(len(df))
    if n is not None and n < len(df):
        rng = np.random.default_rng(seed)
        df = df.iloc[np.sort(rng.choice(len(df), size=n, replace=False))].reset_index(drop=True)
    return pd.DataFrame({
        "qid": [f"{dataset_name}_{s}_{r}" for s, r in zip(df["subject"], df["row"])],
        "dataset": dataset_name,
        "question": df["question"].str.strip(),
        "choices": [list(c) for c in df["choices"]],
        "answer_idx": df["answer"].astype(int),
        "subject": df["subject"],
        "role": role,
    })


def load_mmlu_medical() -> pd.DataFrame:
    return load_mmlu(subjects=C.MMLU_MEDICAL_SUBJECTS, dataset_name="mmlu_med", role="target")


def load_mmlu_nonmedical(n: int = C.MMLU_N_NONMEDICAL) -> pd.DataFrame:
    return load_mmlu(exclude=C.MMLU_MEDICAL_SUBJECTS, n=n, dataset_name="mmlu_other", role="calibration")


# ----------------------------------------------------------------------------- I/O
QUESTION_FILES = {
    "triviaqa": "triviaqa.parquet",
    "bioasq": "bioasq.parquet",
    "mmlu_med": "mmlu_med.parquet",
    "mmlu_other": "mmlu_other.parquet",
}


def save_questions(df: pd.DataFrame, name: str, root: str | Path = "data/questions") -> Path:
    path = Path(root) / QUESTION_FILES[name]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_questions(name: str, root: str | Path = "data/questions") -> pd.DataFrame:
    df = pd.read_parquet(Path(root) / QUESTION_FILES[name])
    for col in ("aliases", "choices"):
        if col in df.columns:
            df[col] = df[col].apply(list)
    return df
