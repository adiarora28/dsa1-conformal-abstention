"""Task 2 - MiniLM embeddings of every question (CPU is fine, GPU is faster).

Usage:
    python scripts/embed_questions.py
Writes data/embeddings/<table>.npz with arrays qids and emb.
"""
from pathlib import Path

import numpy as np

import _path  # noqa: F401
from dsa1 import data
from dsa1.weights import embed_texts

out = Path("data/embeddings")
out.mkdir(parents=True, exist_ok=True)
for name in ["triviaqa", "bioasq", "mmlu_med", "mmlu_other"]:
    try:
        df = data.read_questions(name)
    except FileNotFoundError:
        print("skip", name)
        continue
    emb = embed_texts(df["question"].tolist())
    np.savez_compressed(out / f"{name}.npz", qids=np.array(df["qid"].tolist(), dtype=str), emb=emb.astype(np.float32))
    print(name, emb.shape)
