import numpy as np

from dsa1 import metrics, rules


def toy(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n)
    err = (rng.uniform(size=n) < u).astype(int)  # error probability equals the uncertainty
    return u, err


def test_naive_matches_definition():
    u, err = toy()
    t = rules.naive_threshold(u, err, 0.10)
    ans = u <= t
    assert err[ans].mean() <= 0.10
    assert rules.weighted_plugin_threshold(u, err, 0.10) == t  # w = 1, lam = 0


def test_rules_ordering_and_targets():
    u, err = toy()
    t_naive = rules.naive_threshold(u, err, 0.10)
    t_coin = rules.coin_threshold(u, err, 0.10, delta=0.05)
    t_max = rules.coin_threshold(u, err, 0.10, delta=0.05, mode="max")
    assert t_coin <= t_naive and t_max <= t_naive  # the bound is more conservative
    t_crc = rules.crc_threshold(u, err, 0.10)
    ev = rules.evaluate_threshold(t_crc, u, err)
    assert ev["answered_and_wrong_rate"] <= 0.10 + 1e-9
    assert ev["error_among_answered"] > 0.10  # CRC bounds a different quantity


def test_coin_valid_out_of_sample():
    viol = 0
    for s in range(200):
        uc, ec = toy(500, seed=s)
        ut, et = toy(20000, seed=1000 + s)
        t = rules.coin_threshold(uc, ec, 0.10, delta=0.05)
        r = rules.evaluate_threshold(t, ut, et)["error_among_answered"]
        viol += (not np.isnan(r)) and r > 0.10
    assert viol / 200 <= 0.08


def test_no_answer_when_infeasible():
    u = np.linspace(0, 1, 100)
    err = np.ones(100)
    assert rules.coin_threshold(u, err, 0.05) == rules.NO_ANSWER
    ev = rules.evaluate_threshold(rules.NO_ANSWER, u, err)
    assert ev["coverage"] == 0 and np.isnan(ev["error_among_answered"])


def test_lambda_makes_rule_more_conservative():
    u, err = toy()
    assert rules.weighted_plugin_threshold(u, err, 0.10, lam=50) <= rules.weighted_plugin_threshold(u, err, 0.10)


def test_metrics():
    u, err = toy()
    assert 0.7 < metrics.auroc_wrong(u, err) < 0.9
    assert np.isclose(metrics.prediction_rejection_ratio(err.astype(float), err), 1.0, atol=1e-6)
    rng = np.random.default_rng(1)
    assert abs(metrics.prediction_rejection_ratio(rng.uniform(size=len(u)), err)) < 0.1
    cov, risk = metrics.risk_coverage_curve(u, err)
    assert cov[-1] == 1.0 and np.isclose(risk[-1], err.mean())
    b = metrics.paired_bootstrap_auroc_diff(err, u, rng.uniform(size=len(u)), n_boot=200)
    assert b["significant"] and b["diff"] > 0
