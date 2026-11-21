"""Evaluation metrics shared by all tasks. Convention: higher uncertainty = more likely wrong."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def auroc_wrong(u, err) -> float:
    """AUROC for detecting wrong answers (positive class = wrong)."""
    err = np.asarray(err).astype(int)
    if err.min() == err.max():
        return float("nan")
    return float(roc_auc_score(err, np.asarray(u, dtype=float)))


def _rejection_auc(order: np.ndarray, correct: np.ndarray) -> float:
    """Area under accuracy-of-kept vs rejection-rate, rejecting in the given order (first = rejected first)."""
    n = len(correct)
    kept_sorted = correct[order[::-1]]          # most trusted first
    acc_kept = np.cumsum(kept_sorted) / np.arange(1, n + 1)  # accuracy when keeping the top j
    acc_by_rejection = acc_kept[::-1]           # rejection rate i/n keeps n - i questions
    r = np.arange(n) / n
    return float(np.trapezoid(acc_by_rejection, r) if hasattr(np, "trapezoid") else np.trapz(acc_by_rejection, r))


def prediction_rejection_ratio(u, err) -> float:
    """PRR (Vashurin et al. 2025): (AUC_score - AUC_random) / (AUC_oracle - AUC_random)."""
    u = np.asarray(u, dtype=float)
    correct = 1 - np.asarray(err, dtype=float)
    n = len(u)
    order_score = np.argsort(-u, kind="stable")             # most uncertain rejected first
    order_oracle = np.argsort(correct, kind="stable")        # wrong answers rejected first
    auc_s = _rejection_auc(order_score, correct)
    auc_o = _rejection_auc(order_oracle, correct)
    auc_r = float(correct.mean() * (n - 1) / n)               # constant accuracy under random rejection
    if np.isclose(auc_o, auc_r):
        return float("nan")
    return (auc_s - auc_r) / (auc_o - auc_r)


def risk_coverage_curve(u, err):
    """Error among answered vs coverage when answering the k most confident questions, k = 1..n."""
    u = np.asarray(u, dtype=float)
    err = np.asarray(err, dtype=float)
    order = np.argsort(u, kind="stable")
    e = err[order]
    k = np.arange(1, len(u) + 1)
    return k / len(u), np.cumsum(e) / k


def aurc(u, err) -> float:
    """Area under the risk-coverage curve (lower is better)."""
    cov, risk = risk_coverage_curve(u, err)
    return float(risk.mean())


def paired_bootstrap_auroc_diff(err, u1, u2, n_boot: int = 1000, seed: int = 0) -> dict:
    """AUROC(u1) - AUROC(u2) on the same questions, with a 95% percentile bootstrap interval."""
    rng = np.random.default_rng(seed)
    err = np.asarray(err).astype(int)
    u1, u2 = np.asarray(u1, dtype=float), np.asarray(u2, dtype=float)
    diffs = []
    n = len(err)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if err[idx].min() == err[idx].max():
            continue
        diffs.append(roc_auc_score(err[idx], u1[idx]) - roc_auc_score(err[idx], u2[idx]))
    diffs = np.array(diffs)
    return {
        "diff": auroc_wrong(u1, err) - auroc_wrong(u2, err),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "significant": bool(np.percentile(diffs, 2.5) > 0 or np.percentile(diffs, 97.5) < 0),
    }


def mean_std(values) -> str:
    v = np.asarray([x for x in values if not np.isnan(x)], dtype=float)
    if len(v) == 0:
        return "n/a"
    return f"{v.mean():.3f} ± {v.std(ddof=1) if len(v) > 1 else 0.0:.3f}"
