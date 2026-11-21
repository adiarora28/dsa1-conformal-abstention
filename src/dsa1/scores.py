"""Uncertainty scores (Task 1). Convention: a HIGHER value means the answer is MORE LIKELY WRONG.

V1 token probability  -> u_tokprob
V2 semantic entropy   -> u_se (discrete) and u_se_lik (likelihood-weighted); MMLU: option entropy
V3 hidden-state probe -> u_probe
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from .grading import normalize_answer


# ----------------------------------------------------------------------------- V1
def tokprob_uncertainty(token_logprobs: Sequence[float]) -> float:
    """Minus the length-normalised log-probability of the greedy answer (empty answer = maximal)."""
    if len(token_logprobs) == 0:
        return float("inf")
    return float(-np.mean(token_logprobs))


def mcq_tokprob_uncertainty(option_probs: Sequence[float]) -> float:
    """MMLU: minus the log-probability of the predicted option."""
    p = float(np.max(option_probs))
    return float(-np.log(max(p, 1e-12)))


def option_entropy(option_probs: Sequence[float]) -> float:
    """MMLU replacement for semantic entropy: entropy of the renormalised option probabilities."""
    p = np.asarray(option_probs, dtype=float)
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


# ----------------------------------------------------------------------------- V2
EntailFn = Callable[[list[str], list[str]], list[bool]]


def cluster_by_meaning(question: str, answers: list[str], entails: EntailFn) -> list[int]:
    """Greedy bidirectional-entailment clustering (Kuhn et al. 2023; Farquhar et al. 2024).

    An answer joins the first existing cluster whose first member it entails AND is entailed by.
    Answers equal after normalisation join without an NLI call. Each NLI input is
    "question + answer", as in Farquhar et al.
    """
    labels: list[int] = []
    reps: list[int] = []  # index of each cluster's first answer
    for i, a in enumerate(answers):
        assigned = None
        na = normalize_answer(a)
        for c, r in enumerate(reps):
            if na and na == normalize_answer(answers[r]):
                assigned = c
                break
        if assigned is None and reps:
            prem = [f"{question} {answers[r]}" for r in reps] + [f"{question} {a}"] * len(reps)
            hyp = [f"{question} {a}"] * len(reps) + [f"{question} {answers[r]}" for r in reps]
            ent = entails(prem, hyp)
            k = len(reps)
            for c in range(k):
                if ent[c] and ent[k + c]:
                    assigned = c
                    break
        if assigned is None:
            reps.append(i)
            assigned = len(reps) - 1
        labels.append(assigned)
    return labels


def discrete_semantic_entropy(labels: Sequence[int]) -> float:
    if len(labels) == 0:
        return float("nan")
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def likelihood_semantic_entropy(labels: Sequence[int], seq_logprobs: Sequence[float]) -> float:
    """Entropy over clusters with cluster mass = sum of sequence probabilities, renormalised."""
    labels = np.asarray(labels)
    lp = np.asarray(seq_logprobs, dtype=float)
    cl = np.unique(labels)
    cl_lp = np.array([np.logaddexp.reduce(lp[labels == c]) for c in cl])
    cl_lp = cl_lp - np.logaddexp.reduce(cl_lp)
    p = np.exp(cl_lp)
    return float(-(p * cl_lp).sum())


class NLIEntailment:
    """DeBERTa-MNLI entailment checker with a small cache. Callable as entails(premises, hypotheses)."""

    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 64):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        id2label = {int(k): v.upper() for k, v in self.model.config.id2label.items()}
        self.entail_id = [k for k, v in id2label.items() if v.startswith("ENTAIL")][0]
        self.batch_size = batch_size
        self.cache: dict[tuple[str, str], bool] = {}

    def __call__(self, premises: list[str], hypotheses: list[str]) -> list[bool]:
        todo = [(p, h) for p, h in zip(premises, hypotheses) if (p, h) not in self.cache]
        for s in range(0, len(todo), self.batch_size):
            part = todo[s:s + self.batch_size]
            enc = self.tok([p for p, _ in part], [h for _, h in part], return_tensors="pt",
                           padding=True, truncation=True, max_length=256).to(self.device)
            with self.torch.no_grad():
                pred = self.model(**enc).logits.argmax(-1).cpu().tolist()
            for pair, y in zip(part, pred):
                self.cache[pair] = (y == self.entail_id)
        return [self.cache[(p, h)] for p, h in zip(premises, hypotheses)]


def semantic_entropy_table(records: pd.DataFrame, entails: EntailFn) -> pd.DataFrame:
    """Compute u_se, u_se_lik and the number of clusters for every row of an open-ended records table."""
    rows = []
    for _, r in records.iterrows():
        samples = list(r["samples"])
        labels = cluster_by_meaning(r["question"], samples, entails)
        rows.append({
            "qid": r["qid"],
            "u_se": discrete_semantic_entropy(labels),
            "u_se_lik": likelihood_semantic_entropy(labels, r["sample_logprob_sum"]),
            "n_clusters": int(len(set(labels))),
            "cluster_labels": labels,
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- V3
def fit_probe(X: np.ndarray, y_wrong: np.ndarray, Cs=(0.01, 0.1, 1.0), cv: int = 5, seed: int = 0):
    """Logistic-regression probe with scaling inside each CV fold. Returns (best_pipeline, best_cv_auroc)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    grid = GridSearchCV(pipe, {"clf__C": list(Cs)}, scoring="roc_auc",
                        cv=StratifiedKFold(cv, shuffle=True, random_state=seed), n_jobs=-1)
    grid.fit(X, y_wrong)
    return grid.best_estimator_, float(grid.best_score_)


def probe_uncertainty(probe, X: np.ndarray) -> np.ndarray:
    """Predicted probability that the answer is wrong."""
    return probe.predict_proba(X)[:, 1]
