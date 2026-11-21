"""Threshold rules: a question is ANSWERED when its uncertainty u <= t, otherwise the model abstains.

Every rule takes calibration uncertainties `u` and errors `err` (1 = wrong answer) and returns t.
t = -inf means "abstain on everything" (no threshold meets the target).

Task 3 compares:
  naive_threshold   V1  empirical error among answered <= alpha
  crc_threshold     V2  conformal risk control on the loss 1[answered and wrong]
  coin_threshold    V3  Clopper-Pearson upper bound on error among answered <= alpha (COIN)
Task 2 uses:
  weighted_plugin_threshold  the naive rule with calibration weights and a regulariser mass lambda
"""
from __future__ import annotations

import numpy as np
from scipy.stats import beta

NO_ANSWER = -np.inf


def _sweep(u: np.ndarray, err: np.ndarray, w: np.ndarray | None = None):
    """Candidate thresholds (unique sorted u) with weighted answered mass N, weighted errors K, raw counts."""
    u = np.asarray(u, dtype=float)
    err = np.asarray(err, dtype=float)
    w = np.ones_like(u) if w is None else np.asarray(w, dtype=float)
    order = np.argsort(u, kind="stable")
    u_s, e_s, w_s = u[order], err[order], w[order]
    last = np.r_[np.nonzero(np.diff(u_s))[0], len(u_s) - 1]
    N = np.cumsum(w_s)[last]
    K = np.cumsum(w_s * e_s)[last]
    n_raw = (np.arange(len(u_s)) + 1)[last]
    k_raw = np.cumsum(e_s)[last]
    return u_s[last], N, K, n_raw, k_raw


def naive_threshold(u, err, alpha: float, min_answered: int = 20) -> float:
    t, N, K, n, k = _sweep(u, err)
    ok = (n >= min_answered) & (k / n <= alpha)
    return float(t[ok].max()) if ok.any() else NO_ANSWER


def weighted_plugin_threshold(u, err, alpha: float, w=None, lam: float = 0.0, min_answered: int = 20) -> float:
    """Largest t with (sum_i w_i 1[u_i<=t] err_i + lam) / (sum_i w_i 1[u_i<=t] + lam) <= alpha.

    The lam term counts as one extra answered-and-wrong pseudo-question of weight lam. It is our
    adaptation of DS-CP's point mass at infinity to the error-among-answered setting: it makes the
    rule more conservative when few calibration questions resemble the target domain.
    With w = 1 and lam = 0 this is exactly naive_threshold.
    """
    t, N, K, n, k = _sweep(u, err, w)
    risk = (K + lam) / (N + lam)
    ok = (n >= min_answered) & (risk <= alpha)
    return float(t[ok].max()) if ok.any() else NO_ANSWER


def crc_threshold(u, err, alpha: float) -> float:
    """Conformal risk control (Angelopoulos et al.) with loss L_i(t) = 1[u_i <= t] * err_i, bound B = 1.

    Picks the largest t with  n/(n+1) * (k_t / n) + 1/(n+1) <= alpha,  i.e. (k_t + 1)/(n + 1) <= alpha.
    This controls the expected share of ALL questions that are answered and wrong, not the error
    among answered questions.
    """
    u = np.asarray(u)
    n_total = len(u)
    t, N, K, n, k = _sweep(u, err)
    ok = (k + 1) / (n_total + 1) <= alpha
    return float(t[ok].max()) if ok.any() else NO_ANSWER


def clopper_pearson_upper(k: np.ndarray, n: np.ndarray, delta: float) -> np.ndarray:
    """One-sided (1 - delta) upper confidence bound on a binomial proportion k / n."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    ub = np.ones_like(k)
    m = k < n
    ub[m] = beta.ppf(1 - delta, k[m] + 1, n[m] - k[m])
    return ub


def coin_threshold(u, err, alpha: float, delta: float = 0.05, mode: str = "fixed_sequence",
                   min_answered: int = 50) -> float:
    """COIN-style rule: the Clopper-Pearson upper bound on error among answered must be <= alpha.

    mode="fixed_sequence": scan thresholds from the most confident (starting once at least
        `min_answered` questions are answered) and stop at the first failure. Testing in a
        pre-set order controls the error of picking a bad threshold among many candidates.
    mode="max": largest t whose bound passes, with no multiplicity control (report as a sensitivity check).
    Check COIN's paper (Wang et al., AAAI 2026) for its exact search before the final runs and keep
    the mode that matches it.
    """
    t, N, K, n, k = _sweep(u, err)
    ub = clopper_pearson_upper(k, n, delta)
    passed = (ub <= alpha) & (n >= min_answered)
    if mode == "max":
        return float(t[passed].max()) if passed.any() else NO_ANSWER
    if mode != "fixed_sequence":
        raise ValueError(mode)
    start = np.argmax(n >= min_answered) if (n >= min_answered).any() else len(n)
    best = NO_ANSWER
    for i in range(start, len(t)):
        if ub[i] <= alpha:
            best = float(t[i])
        else:
            break
    return best


def evaluate_threshold(t: float, u_test, err_test) -> dict:
    """Coverage and realised error among answered questions on a labelled test set."""
    u_test = np.asarray(u_test, dtype=float)
    err_test = np.asarray(err_test, dtype=float)
    answered = u_test <= t
    n_ans = int(answered.sum())
    return {
        "threshold": t,
        "coverage": n_ans / len(u_test),
        "n_answered": n_ans,
        "error_among_answered": float(err_test[answered].mean()) if n_ans else float("nan"),
        "answered_and_wrong_rate": float((answered & (err_test > 0)).mean()),
    }
