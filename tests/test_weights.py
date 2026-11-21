import numpy as np

from dsa1 import weights


def test_weighted_quantile_reduces_to_split_conformal():
    s = np.arange(1, 101, dtype=float)
    q = weights.weighted_conformal_quantile(s, np.ones(100), lam=1.0, alpha=0.1)
    # standard split conformal: ceil((n+1)(1-alpha)) = 91st smallest score
    assert q == 91.0
    assert weights.weighted_conformal_quantile(s, np.ones(100), lam=1e6, alpha=0.1) == np.inf


def test_tfidf_weights_upweight_target_like_questions():
    cal = ["who won the world cup"] * 50 + ["which gene causes cystic fibrosis disease"] * 10
    tgt = ["which gene is linked to disease x"] * 40
    w = weights.tfidf_weights(cal, tgt, clip_quantile=None)
    assert w.w[50:].mean() > w.w[:50].mean()
    assert w.domain_auc > 0.8
    assert 1 <= w.ess <= 60


def test_dscp_weights_and_shift_distance():
    rng = np.random.default_rng(0)
    cal = rng.normal(size=(300, 8))
    tgt = rng.normal(loc=0.8, size=(200, 8))
    cal /= np.linalg.norm(cal, axis=1, keepdims=True)
    tgt /= np.linalg.norm(tgt, axis=1, keepdims=True)
    w = weights.dscp_weights(cal, tgt)
    assert w.lam == w.w.max()
    assert w.domain_auc > 0.6
    assert weights.embedding_shift_distance(cal, tgt) > weights.embedding_shift_distance(cal, cal[:150])


def test_conformal_pvalues():
    u_cal = np.linspace(0, 1, 99)
    p = weights.conformal_pvalues(u_cal, np.array([2.0, -1.0]))
    assert np.isclose(p[0], 1 / 100) and np.isclose(p[1], 1.0)
