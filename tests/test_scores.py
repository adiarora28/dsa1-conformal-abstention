import numpy as np

from dsa1 import scores


def fake_entails(prem, hyp):
    # two answers "mean the same" when their last word matches (stand-in for the NLI model)
    return [p.split()[-1].lower() == h.split()[-1].lower() for p, h in zip(prem, hyp)]


def test_clustering_and_entropy():
    labels = scores.cluster_by_meaning("Q?", ["Paris", "paris", "city Paris", "Lyon", "Rome"], fake_entails)
    assert labels == [0, 0, 0, 1, 2]
    se = scores.discrete_semantic_entropy(labels)
    p = np.array([3, 1, 1]) / 5
    assert np.isclose(se, -(p * np.log(p)).sum())
    assert scores.discrete_semantic_entropy([0] * 10) == 0.0


def test_likelihood_se_single_cluster_is_zero():
    assert np.isclose(scores.likelihood_semantic_entropy([0, 0, 0], [-1.0, -2.0, -3.0]), 0.0)


def test_tokprob_and_mcq():
    assert scores.tokprob_uncertainty([-0.1, -0.3]) == 0.2
    assert scores.tokprob_uncertainty([]) == float("inf")
    assert np.isclose(scores.option_entropy([1, 1, 1, 1]), np.log(4))
    assert scores.mcq_tokprob_uncertainty([0.5, 0.5, 0, 0]) == -np.log(0.5)


def test_probe_learns_signal():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 16))
    y = (X[:, 0] + 0.3 * rng.normal(size=400) > 0).astype(int)
    probe, auc = scores.fit_probe(X, y)
    assert auc > 0.9
    assert scores.probe_uncertainty(probe, X).shape == (400,)
