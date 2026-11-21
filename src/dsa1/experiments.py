"""Glue code used by the task notebooks, so the notebooks stay short and every task computes
things the same way.

Score tables (one per model and dataset) are the hand-over format between tasks:
    results/scores/<model>/<dataset>.parquet
    columns: qid, dataset, err (1 = wrong, exact match), err_f1, u_tokprob, u_se, u_se_lik, u_probe,
             subject (MMLU only), cost columns (seconds per question)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .cache_io import cache_dir, load_hidden, load_records
from .rules import coin_threshold, crc_threshold, evaluate_threshold, naive_threshold, weighted_plugin_threshold
from .scores import fit_probe, mcq_tokprob_uncertainty, option_entropy, probe_uncertainty, tokprob_uncertainty
from .weights import Weights

OPEN_DATASETS = ("triviaqa_pool", "bioasq")
MCQ_DATASETS = ("mmlu_med", "mmlu_other")


# ----------------------------------------------------------------------------- Task 1: score tables
def base_scores(cache_root, model_key: str, dataset: str, precision: str | None = None,
                suffix: str = "", se_root: str | Path = "results/semantic_entropy") -> pd.DataFrame:
    """Scores that need no training: token probability and semantic entropy (or option entropy).

    Semantic entropy is read from <se_root>/<model>/<dataset>.parquet, written by
    scripts/compute_semantic_entropy.py (cache folders on Kaggle inputs are read-only).
    """
    path = cache_dir(cache_root, model_key, dataset, precision)
    if suffix:
        path = path.parent / (path.name + suffix)
    rec = load_records(path)
    df = pd.DataFrame({"qid": rec["qid"], "dataset": dataset, "err": (~rec["correct_em"].astype(bool)).astype(int)})
    if dataset in MCQ_DATASETS:
        df["u_tokprob"] = rec["option_probs"].apply(mcq_tokprob_uncertainty)
        df["u_se"] = rec["option_probs"].apply(option_entropy)  # MMLU replacement for semantic entropy
        df["u_se_lik"] = df["u_se"]
        df["subject"] = rec["subject"]
        df["t_greedy"] = rec["t_forward"]
        df["t_samples"] = 0.0
        df["t_hidden"] = 0.0
        df["t_nli"] = 0.0
        df["err_f1"] = df["err"]
    else:
        df["err_f1"] = (~rec["correct_f1"].astype(bool)).astype(int)
        df["u_tokprob"] = rec["token_logprobs"].apply(tokprob_uncertainty)
        for c in ("t_greedy", "t_samples", "t_hidden"):
            df[c] = rec[c]
        se_path = Path(se_root) / model_key / f"{dataset}{suffix}.parquet"
        if se_path.exists():
            se = pd.read_parquet(se_path)
            keep = [c for c in ("qid", "u_se", "u_se_lik", "n_clusters", "t_nli") if c in se.columns]
            df = df.merge(se[keep], on="qid", how="left")
        else:
            df["u_se"] = np.nan
            df["u_se_lik"] = np.nan
            df["t_nli"] = np.nan
    # an empty answer has infinite token uncertainty; keep it finite and maximal for ranking
    finite = df["u_tokprob"].replace(np.inf, np.nan)
    df["u_tokprob"] = df["u_tokprob"].replace(np.inf, (finite.max() if finite.notna().any() else 0) + 1)
    return df


def select_probe(cache_root, model_key: str, positions=("prompt", "answer"), layers=(0, 1, 2),
                 Cs=(0.01, 0.1, 1.0), seed: int = 0):
    """Choose layer, position and C by 5-fold CV on the TriviaQA probe set only; refit the best probe."""
    path = cache_dir(cache_root, model_key, "triviaqa_probe")
    rec = load_records(path)
    y = (~rec["correct_em"].astype(bool)).astype(int).to_numpy()
    rows, best = [], None
    for pos in positions:
        for layer in layers:
            _, X = load_hidden(path, pos, layer, qids=rec["qid"].tolist())
            probe, auc = fit_probe(X, y, Cs=Cs, seed=seed)
            rows.append({"position": pos, "layer_slot": layer, "cv_auroc": auc,
                         "C": probe.named_steps["clf"].C})
            if best is None or auc > best[0]:
                best = (auc, probe, pos, layer)
    _, probe, pos, layer = best
    return probe, pos, layer, pd.DataFrame(rows).sort_values("cv_auroc", ascending=False)


def add_probe_scores(df: pd.DataFrame, probe, cache_root, model_key: str, dataset: str,
                     position: str, layer: int, precision: str | None = None, suffix: str = "") -> pd.DataFrame:
    path = cache_dir(cache_root, model_key, dataset, precision)
    if suffix:
        path = path.parent / (path.name + suffix)
    _, X = load_hidden(path, position, layer, qids=df["qid"].tolist())
    out = df.copy()
    out["u_probe"] = probe_uncertainty(probe, X)
    return out


def save_score_table(df: pd.DataFrame, model_key: str, dataset: str, root: str | Path = "results/scores") -> Path:
    path = Path(root) / model_key / f"{dataset}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_score_table(model_key: str, dataset: str, root: str | Path = "results/scores") -> pd.DataFrame:
    return pd.read_parquet(Path(root) / model_key / f"{dataset}.parquet")


def rows_for(df: pd.DataFrame, qids: list[str]) -> pd.DataFrame:
    """Rows of a score table in the order of `qids` (raises if any is missing)."""
    idx = df.set_index("qid")
    missing = [q for q in qids if q not in idx.index]
    if missing:
        raise KeyError(f"{len(missing)} qids missing, e.g. {missing[:3]}")
    return idx.loc[qids].reset_index()


# ----------------------------------------------------------------------------- embeddings
def load_embeddings(table: str, root: str | Path = "data/embeddings") -> dict:
    z = np.load(Path(root) / f"{table}.npz", allow_pickle=False)
    return dict(zip(z["qids"].astype(str), z["emb"]))


def embed_rows(emb: dict, qids: list[str]) -> np.ndarray:
    return np.stack([emb[q] for q in qids])


# ----------------------------------------------------------------------------- Task 2
def shift_eval(u_cal, err_cal, u_test, err_test, W: Weights, alphas=C.TARGETS) -> list[dict]:
    """Task 2: one weighting, the shared weighted plug-in rule, all targets."""
    rows = []
    for a in alphas:
        t = weighted_plugin_threshold(u_cal, err_cal, a, w=W.w, lam=W.lam)
        ev = evaluate_threshold(t, u_test, err_test)
        rows.append({"method": W.method, "target": a, **ev,
                     "gap": ev["error_among_answered"] - a, "ess": W.ess,
                     "domain_auc": W.domain_auc, "lam": W.lam})
    return rows


# ----------------------------------------------------------------------------- Task 3
RULES = {
    "naive": lambda u, e, a, d: naive_threshold(u, e, a),
    "crc": lambda u, e, a, d: crc_threshold(u, e, a),
    "coin": lambda u, e, a, d: coin_threshold(u, e, a, delta=d),
}


def rule_eval(u_cal, err_cal, u_test, err_test, alphas=C.TARGETS, delta: float = C.DELTA,
              rules=("naive", "crc", "coin")) -> list[dict]:
    rows = []
    for r in rules:
        for a in alphas:
            t = RULES[r](u_cal, err_cal, a, delta)
            ev = evaluate_threshold(t, u_test, err_test)
            err_v = ev["error_among_answered"]
            rows.append({"rule": r, "target": a, **ev,
                         "violated": bool(not np.isnan(err_v) and err_v > a)})
    return rows


def cost_per_question(df: pd.DataFrame, score: str) -> pd.Series:
    """GPU seconds per question needed to produce a score (from the cache timings)."""
    def col(name):
        return df[name].fillna(0) if name in df.columns else 0.0
    t = col("t_greedy")
    if score.startswith("u_se"):
        t = t + col("t_samples") + col("t_nli")
    if score == "u_probe":
        t = t + col("t_hidden")
    return t


# ----------------------------------------------------------------------------- confusion matrices and demo tables
def abstention_confusion(u_test, err_test, t: float) -> dict:
    """2 x 2 counts for an answer-or-abstain decision: answered/abstained × right/wrong."""
    u = np.asarray(u_test, dtype=float)
    e = np.asarray(err_test).astype(int)
    ans = u <= t
    return {
        "answered_right": int((ans & (e == 0)).sum()),
        "answered_wrong": int((ans & (e == 1)).sum()),
        "abstained_right": int((~ans & (e == 0)).sum()),
        "abstained_wrong": int((~ans & (e == 1)).sum()),
    }


def detection_confusion(u_test, err_test) -> dict:
    """Task 1: flag as many answers as there are wrong ones (the highest uncertainties); 2 x 2 counts."""
    u = np.asarray(u_test, dtype=float)
    e = np.asarray(err_test).astype(int)
    k = int(e.sum())
    flagged = np.zeros(len(u), dtype=bool)
    if k:
        flagged[np.argsort(-u, kind="stable")[:k]] = True
    return {
        "wrong_flagged": int((flagged & (e == 1)).sum()),
        "wrong_missed": int((~flagged & (e == 1)).sum()),
        "right_flagged": int((flagged & (e == 0)).sum()),
        "right_kept": int((~flagged & (e == 0)).sum()),
    }


def plot_confusions(panels: list[tuple[str, dict]], kind: str, path: str | Path, suptitle: str = ""):
    """One 2 x 2 heatmap per panel. kind = 'abstention' or 'detection'."""
    import matplotlib.pyplot as plt

    if kind == "abstention":
        keys = [["answered_wrong", "answered_right"], ["abstained_wrong", "abstained_right"]]
        rows, cols = ["answered", "abstained"], ["actually wrong", "actually right"]
    else:
        keys = [["wrong_flagged", "right_flagged"], ["wrong_missed", "right_kept"]]
        rows, cols = ["flagged uncertain", "not flagged"], ["actually wrong", "actually right"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 3.2), squeeze=False)
    for ax, (title, counts) in zip(axes[0], panels):
        m = np.array([[counts[k] for k in r] for r in keys])
        ax.imshow(m, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(m[i, j]), ha="center", va="center",
                        color="white" if m[i, j] > m.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(cols)
        ax.set_yticks([0, 1]); ax.set_yticklabels(rows)
        ax.set_title(title)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    return fig
