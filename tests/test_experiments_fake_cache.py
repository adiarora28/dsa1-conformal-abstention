"""End-to-end check of the CPU pipeline on a fake cache with the exact on-disk format of generate.py."""
import numpy as np
import pandas as pd

from dsa1 import experiments as E
from dsa1.cache_io import cache_dir, load_hidden, load_records
from dsa1.weights import tfidf_weights, uniform_weights


def _write(path, records, hp, ha, shard=0):
    path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(path / f"records_{shard:05d}.parquet", index=False)
    np.savez_compressed(path / f"hidden_{shard:05d}.npz", qids=np.array([r["qid"] for r in records]),
                        h_prompt=np.stack(hp).astype(np.float16), h_answer=np.stack(ha).astype(np.float16))


def _fake_open(root, dataset, n, prefix, seed, shift=0.0):
    rng = np.random.default_rng(seed)
    recs, hp, ha = [], [], []
    for i in range(n):
        wrong = rng.uniform() < 0.3 + shift
        conf = rng.normal(-0.2 if not wrong else -1.0, 0.3)
        lp = [min(-1e-3, conf + 0.05 * rng.normal()) for _ in range(3)]
        h = rng.normal(size=(3, 8)) + (1.5 if wrong else 0.0) * np.eye(3, 8)
        recs.append({"qid": f"{prefix}_{i}", "dataset": "x", "question": f"question {prefix} {i}",
                     "aliases": ["a"], "answer": "a", "token_logprobs": lp, "correct_em": not wrong,
                     "correct_f1": not wrong, "f1": float(not wrong), "samples": ["a"] * 3,
                     "sample_logprob_sum": [-1.0] * 3, "t_greedy": 0.1, "t_samples": 0.5, "t_hidden": 0.05})
        hp.append(h)
        ha.append(h)
    # two shards to exercise concatenation
    _write(cache_dir(root, "qwen3-4b", dataset), recs[: n // 2], hp[: n // 2], ha[: n // 2], 0)
    _write(cache_dir(root, "qwen3-4b", dataset), recs[n // 2:], hp[n // 2:], ha[n // 2:], 1)


def test_pipeline(tmp_path):
    root = tmp_path / "cache"
    _fake_open(root, "triviaqa_probe", 300, "tqa_p", 0)
    _fake_open(root, "triviaqa_pool", 400, "tqa", 1)
    _fake_open(root, "bioasq", 200, "bio", 2, shift=0.3)

    rec = load_records(cache_dir(root, "qwen3-4b", "triviaqa_pool"))
    assert len(rec) == 400 and isinstance(rec.loc[0, "token_logprobs"], list)
    q, X = load_hidden(cache_dir(root, "qwen3-4b", "triviaqa_pool"), "answer", 1, qids=["tqa_5", "tqa_1"])
    assert list(q) == ["tqa_5", "tqa_1"] and X.shape == (2, 8)

    probe, pos, layer, table = E.select_probe(root, "qwen3-4b")
    assert table["cv_auroc"].max() > 0.7

    scores = {}
    for ds in ("triviaqa_pool", "bioasq"):
        df = E.base_scores(root, "qwen3-4b", ds)
        df = E.add_probe_scores(df, probe, root, "qwen3-4b", ds, pos, layer)
        E.save_score_table(df, "qwen3-4b", ds, root=tmp_path / "scores")
        scores[ds] = E.load_score_table("qwen3-4b", ds, root=tmp_path / "scores")
    t = scores["triviaqa_pool"]
    assert {"u_tokprob", "u_probe", "err"} <= set(t.columns)

    cal = E.rows_for(t, [f"tqa_{i}" for i in range(200)])
    test = scores["bioasq"]
    rows = E.rule_eval(cal["u_tokprob"], cal["err"], test["u_tokprob"], test["err"])
    assert len(rows) == 9 and {r["rule"] for r in rows} == {"naive", "crc", "coin"}

    W = tfidf_weights(cal_texts=[f"question tqa {i}" for i in range(200)],
                      tgt_texts=[f"question bio {i}" for i in range(200)], clip_quantile=0.99)
    for weights in (uniform_weights(200), W):
        rows = E.shift_eval(cal["u_tokprob"], cal["err"], test["u_tokprob"], test["err"], weights)
        assert len(rows) == 3


def test_link_cache_roots(tmp_path):
    from dsa1.cache_io import link_cache_roots
    _fake_open(tmp_path / "input" / "dsa1-cache-a" / "cache", "triviaqa_pool", 20, "tqa", 0)
    _fake_open(tmp_path / "input" / "dsa1-cache-b" / "cache", "bioasq", 20, "bio", 1)
    root = link_cache_roots(tmp_path / "input", tmp_path / "merged")
    assert len(load_records(cache_dir(root, "qwen3-4b", "triviaqa_pool"))) == 20
    assert len(load_records(cache_dir(root, "qwen3-4b", "bioasq"))) == 20


def test_confusions(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    u = np.array([0.1, 0.2, 0.8, 0.9])
    e = np.array([0, 1, 0, 1])
    c = E.abstention_confusion(u, e, 0.5)
    assert c == {"answered_right": 1, "answered_wrong": 1, "abstained_right": 1, "abstained_wrong": 1}
    d = E.detection_confusion(u, e)
    assert d["wrong_flagged"] + d["wrong_missed"] == 2 and d["wrong_flagged"] + d["right_flagged"] == 2
    E.plot_confusions([("x", c)], "abstention", tmp_path / "a.png")
    E.plot_confusions([("y", d)], "detection", tmp_path / "b.png")
    assert (tmp_path / "a.png").exists() and (tmp_path / "b.png").exists()
