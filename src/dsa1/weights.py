"""Calibration weights under domain shift (Task 2).

V1 uniform       : w_i = 1, lambda = 1 (standard split conformal)
V2 TF-IDF        : logistic-regression domain classifier on TF-IDF question features, cross-fitted,
                   weights clipped at a quantile (classical weighted conformal, Tibshirani et al. 2019)
V3 DS-CP         : XGBoost domain classifier on all-MiniLM-L6-v2 embeddings, lambda = max_i w_i
                   (Lin et al., arXiv 2510.05566, Algorithm 1 and Appendix C)

Density ratio from a domain classifier (W=0 calibration, W=1 target):
    r(x) = p(x) / (1 - p(x)) * P(W=0) / P(W=1) = p / (1 - p) * n_cal / n_target
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as C


@dataclass
class Weights:
    w: np.ndarray          # one weight per calibration question
    lam: float             # mass placed on the "infinite score" point (DS-CP regulariser)
    domain_auc: float      # cross-validated AUC of the domain classifier = shift-size measure
    method: str

    @property
    def ess(self) -> float:
        return effective_sample_size(self.w)


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum())


def _ratio(p: np.ndarray, n_cal: int, n_tgt: int, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return p / (1 - p) * (n_cal / n_tgt)


def uniform_weights(n_cal: int) -> Weights:
    return Weights(w=np.ones(n_cal), lam=1.0, domain_auc=float("nan"), method="uniform")


def _cv_domain_auc(clf_factory, X, y, seed: int, cv: int = 5) -> float:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    p = cross_val_predict(clf_factory(), X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def tfidf_weights(cal_texts: list[str], tgt_texts: list[str], clip_quantile: float | None = 0.99,
                  cv: int = 5, seed: int = 0, lam: str | float = "max") -> Weights:
    """V2: cross-fitted logistic-regression domain classifier on TF-IDF features."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    texts = list(cal_texts) + list(tgt_texts)
    y = np.r_[np.zeros(len(cal_texts)), np.ones(len(tgt_texts))]

    def factory():
        return make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
            LogisticRegression(max_iter=2000, C=1.0),
        )

    p = cross_val_predict(factory(), texts, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                          method="predict_proba")[:, 1]
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(y, p))
    w = _ratio(p[: len(cal_texts)], len(cal_texts), len(tgt_texts))
    if clip_quantile is not None:
        w = np.minimum(w, np.quantile(w, clip_quantile))
    lam_v = float(w.max()) if lam == "max" else float(lam)
    return Weights(w=w, lam=lam_v, domain_auc=auc, method="tfidf")


def embed_texts(texts: list[str], model_name: str = C.EMBED_MODEL, batch_size: int = 128) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)


def dscp_weights(cal_emb: np.ndarray, tgt_emb: np.ndarray, gamma: float = 1.0, seed: int = 42,
                 cross_fit: bool = False, cv: int = 5) -> Weights:
    """V3 DS-CP: XGBoost domain classifier in embedding space; lambda = gamma * max_i w_i.

    gamma = 1 is the paper's default (its Appendix B ablates gamma in {0, .25, .5, 1, 2, 5}).
    cross_fit=False follows the paper (classifier fitted on all calibration + target embeddings).
    """
    from xgboost import XGBClassifier

    X = np.vstack([cal_emb, tgt_emb])
    y = np.r_[np.zeros(len(cal_emb)), np.ones(len(tgt_emb))]
    params = dict(C.DSCP_XGB_PARAMS)
    params["random_state"] = seed

    def factory():
        return XGBClassifier(**params, eval_metric="logloss")

    if cross_fit:
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        p = cross_val_predict(factory(), X, y, cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
                              method="predict_proba")[:, 1][: len(cal_emb)]
    else:
        clf = factory().fit(X, y)
        p = clf.predict_proba(cal_emb)[:, 1]
    w = _ratio(p, len(cal_emb), len(tgt_emb))
    auc = _cv_domain_auc(factory, X, y, seed=seed, cv=cv)
    return Weights(w=w, lam=float(gamma * w.max()), domain_auc=auc, method="dscp")


def embedding_shift_distance(cal_emb: np.ndarray, tgt_emb: np.ndarray) -> float:
    """Mean cosine distance between calibration and target questions (embeddings are unit-norm)."""
    return float(1.0 - cal_emb.mean(axis=0) @ tgt_emb.mean(axis=0))


def weighted_conformal_quantile(scores: np.ndarray, w: np.ndarray, lam: float, alpha: float) -> float:
    """(1 - alpha)-quantile of sum_i w_i/(W+lam) delta_{S_i} + lam/(W+lam) delta_inf  (DS-CP Step 5).

    With w = 1 and lam = 1 this is the standard split-conformal quantile.
    Returns +inf when the mass below every finite score is too small (the set is the whole label space).
    """
    scores = np.asarray(scores, dtype=float)
    w = np.asarray(w, dtype=float)
    order = np.argsort(scores, kind="stable")
    s, ws = scores[order], w[order]
    cdf = np.cumsum(ws) / (ws.sum() + lam)
    idx = np.searchsorted(cdf, 1 - alpha - 1e-12, side="left")
    return float(s[idx]) if idx < len(s) else float("inf")


def conformal_pvalues(u_cal: np.ndarray, u_test: np.ndarray) -> np.ndarray:
    """Shift alarm (Task 2, SConU-style): p = (1 + #{u_cal >= u_test}) / (n + 1) for each test question."""
    u_cal = np.sort(np.asarray(u_cal, dtype=float))
    n = len(u_cal)
    ge = n - np.searchsorted(u_cal, np.asarray(u_test, dtype=float), side="left")
    return (1 + ge) / (n + 1)
