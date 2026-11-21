import json

import pandas as pd
import pytest

from dsa1 import data, splits


def test_bioasq_parsing(tmp_path):
    raw = {"questions": [
        {"id": "a1", "type": "factoid", "body": "Which gene?", "exact_answer": [["BRCA1", "brca-1"], ["TP53"]]},
        {"id": "a2", "type": "yesno", "body": "Is it?", "exact_answer": "yes"},
        {"id": "a3", "type": "factoid", "body": "Which drug?", "exact_answer": ["aspirin"]},
    ]}
    p = tmp_path / "b.json"
    p.write_text(json.dumps(raw))
    df = data.load_bioasq(p)
    assert list(df["qid"]) == ["bio_a1", "bio_a3"]
    assert set(df.loc[0, "aliases"]) == {"BRCA1", "brca-1", "TP53"}


def _tables():
    tq = pd.DataFrame({"qid": [f"tqa_{i}" for i in range(60)], "role": ["probe"] * 10 + ["pool"] * 50})
    bio = pd.DataFrame({"qid": [f"bio_{i}" for i in range(30)]})
    mm = pd.DataFrame({"qid": [f"mmlu_med_x_{i}" for i in range(20)]})
    return tq, bio, mm


def test_splits_disjoint_and_reproducible():
    tq, bio, mm = _tables()
    a = splits.make_splits(tq, bio, mm, seeds=[0, 1], n_cal=20, n_test_trivia=20, n_test_bioasq=15)
    b = splits.make_splits(tq, bio, mm, seeds=[0, 1], n_cal=20, n_test_trivia=20, n_test_bioasq=15)
    assert a == b
    s0 = a["seeds"]["0"]
    assert not set(s0["cal"]) & set(s0["test_triviaqa"])
    assert a["seeds"]["0"]["cal"] != a["seeds"]["1"]["cal"]


def test_split_check_catches_leak():
    tq, bio, mm = _tables()
    sp = splits.make_splits(tq, bio, mm, seeds=[0], n_cal=20, n_test_trivia=20, n_test_bioasq=15)
    sp["seeds"]["0"]["cal"].append("bio_1")
    with pytest.raises(AssertionError):
        splits.check_splits(sp)


def test_mixture():
    m = splits.mixture([f"t{i}" for i in range(100)], [f"b{i}" for i in range(100)], 0.25, n=40)
    assert len(m) == 40 and sum(x.startswith("b") for x in m) == 10
