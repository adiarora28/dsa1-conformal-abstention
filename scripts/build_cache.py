"""Steps 0.3-0.6 - generate one part of the shared answer cache on a GPU (resumable).

Examples (run from the repository root):
    python scripts/build_cache.py --model qwen3-4b --dataset triviaqa_pool --limit 20   # smoke test
    python scripts/build_cache.py --model qwen3-4b --dataset triviaqa_probe            # greedy + hidden only
    python scripts/build_cache.py --model qwen3-4b --dataset bioasq
    python scripts/build_cache.py --model qwen3-4b --dataset mmlu_med
    python scripts/build_cache.py --model qwen3-4b --dataset bioasq --precision 4bit \
        --no-samples --qids-file data/splits/precision_subset.json                    # Task 3 precision shift
"""
import argparse
import json

import _path  # noqa: F401
from dsa1 import config as C
from dsa1 import data
from dsa1.cache_io import cache_dir
from dsa1.generate import load_model, run_mcq, run_open

DATASETS = {
    # cache name : (question table, role filter, open-ended?, samples by default?)
    "triviaqa_probe": ("triviaqa", "probe", True, False),
    "triviaqa_pool": ("triviaqa", "pool", True, True),
    "bioasq": ("bioasq", None, True, True),
    "mmlu_med": ("mmlu_med", None, False, False),
    "mmlu_other": ("mmlu_other", None, False, False),
}

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, choices=list(C.MODELS))
ap.add_argument("--dataset", required=True, choices=list(DATASETS))
ap.add_argument("--precision", default=None, help="fp16 | bf16 | fp32 | 8bit | 4bit (default from config)")
ap.add_argument("--cache-root", default="cache")
ap.add_argument("--limit", type=int, default=None, help="only the first N not-yet-done questions")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--sample-batch-size", type=int, default=4)
ap.add_argument("--no-samples", action="store_true")
ap.add_argument("--qids-file", default=None, help="JSON {cache name: [qids]} to restrict the questions")
args = ap.parse_args()

table, role, is_open, samples_default = DATASETS[args.dataset]
df = data.read_questions(table)
if role:
    df = df[df["role"] == role].reset_index(drop=True)
if args.qids_file:
    keep = set(json.load(open(args.qids_file))[args.dataset])
    df = df[df["qid"].isin(keep)].reset_index(drop=True)

model, tok, precision = load_model(args.model, args.precision)
out = cache_dir(args.cache_root, args.model, args.dataset, precision)
if args.qids_file:
    out = out.parent / (out.name + "_subset")
print(f"{len(df)} questions -> {out}")
if is_open:
    n_samples = 0 if (args.no_samples or not samples_default) else C.N_SAMPLES
    run_open(df, model, tok, out, model_key=args.model, precision=precision, n_samples=n_samples,
             batch_size=args.batch_size, sample_batch_size=args.sample_batch_size, limit=args.limit)
else:
    run_mcq(df, model, tok, out, model_key=args.model, precision=precision,
            batch_size=args.batch_size, limit=args.limit)
print("done:", out)
