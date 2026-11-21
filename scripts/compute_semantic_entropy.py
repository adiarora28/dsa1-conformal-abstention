"""Task 1 V2 - semantic entropy for one model and one open-ended cache folder (GPU recommended).

Usage:
    python scripts/compute_semantic_entropy.py --model qwen3-4b --dataset bioasq --cache-root CACHE_ROOT
Writes results/semantic_entropy/<model>/<dataset>.parquet (qid, u_se, u_se_lik, n_clusters, t_nli).
"""
import argparse
import time
from pathlib import Path

import _path  # noqa: F401
from dsa1 import config as C
from dsa1.cache_io import cache_dir, load_records
from dsa1.scores import NLIEntailment, semantic_entropy_table

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(C.MODELS))
ap.add_argument("--dataset", required=True, choices=["triviaqa_pool", "bioasq"])
ap.add_argument("--precision", default=None)
ap.add_argument("--cache-root", default="cache")
ap.add_argument("--out-root", default="results/semantic_entropy")
args = ap.parse_args()

rec = load_records(cache_dir(args.cache_root, args.model, args.dataset, args.precision))
nli = NLIEntailment(C.NLI_MODEL)
t0 = time.perf_counter()
se = semantic_entropy_table(rec, nli)
se["t_nli"] = (time.perf_counter() - t0) / len(se)
out = Path(args.out_root) / args.model / f"{args.dataset}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
se.to_parquet(out, index=False)
print("wrote", out)
print(se[["u_se", "u_se_lik", "n_clusters"]].describe())
