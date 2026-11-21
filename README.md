# DSA1 – Conformal abstention for small open LLMs under domain shift

Shared code for the three modeling tasks (Action Learning, MSc Data Science & Analytics, EPITA).
The step-by-step guide lives in the team doc *"DSA 1 – Modeling Tasks: Step-by-Step Implementation Guide"*.

| Task | Owner | Notebook |
| --- | --- | --- |
| 1 – Uncertainty scores | Sai Akhil Singade | `notebooks/SaiAkhilSingade_Uncertainty_Scores.ipynb` |
| 2 – Domain-shift correction | Aditya Arora | `notebooks/AdityaArora_Domain_Shift_Correction.ipynb` |
| 3 – Threshold rule and reliability | Shivam Vivek Rai | `notebooks/ShivamVivekRai_Threshold_Rule_Reliability.ipynb` |
| Shared cache (GPU) | everyone, own part | `notebooks/00_build_cache.ipynb` |
| Final comparison | everyone | `notebooks/DSA1_Final_Comparison.ipynb` |

## Layout

```
src/dsa1/        shared package: config, data, prompts, generate (GPU), grading, scores, weights, rules, metrics, experiments
scripts/         command-line steps (prepare questions, splits, cache, semantic entropy, embeddings, hand check, smoke test)
notebooks/       the five notebooks above (built by tools/build_notebooks.py)
data/questions/  fixed question tables (committed)
data/splits/     splits.json + precision_subset.json (committed)
data/embeddings/ MiniLM question embeddings (committed, ~10 MB)
results/         metrics, figures, score tables, probes
slides/          DSA1_Modeling_Defense.pptx – defense slide template (fill on Day 5, export PDF on Day 6)
tests/           unit tests and an end-to-end test on a fake cache (CPU, ~5 s)
```

## One-time setup (one member, on a laptop or Colab, CPU is enough)

```bash
git clone https://github.com/adiarora28/dsa1-conformal-abstention.git && cd dsa1-conformal-abstention
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # laptop only; Kaggle/Colab already have torch
pip install -r requirements.txt
python -m pytest -q tests                                   # 23 tests should pass
# BioASQ: register at participants-area.bioasq.org, download the Task B training set (14b), put the JSON in data/raw/
python scripts/prepare_questions.py --bioasq-json data/raw/training14b.json
python scripts/make_splits.py
python scripts/embed_questions.py
git add data/questions data/splits data/embeddings && git commit -m "Fixed questions, splits, embeddings" && git push
```

## Shared cache (each member, Kaggle GPU) – see notebooks/00_build_cache.ipynb

```bash
python scripts/smoke_test.py --model qwen3-1.7b                                   # 5-minute check
python scripts/build_cache.py --model qwen3-4b --dataset triviaqa_pool --limit 48  # timing
python scripts/build_cache.py --model qwen3-4b --dataset triviaqa_pool             # full, resumable
```
Datasets: `triviaqa_probe`, `triviaqa_pool` (Sai Akhil) · `bioasq` (Aditya) · `mmlu_med`, `mmlu_other` (Shivam).
Models in order: `qwen3-4b`, `qwen3-1.7b`, `qwen3-8b`. Publish your `/kaggle/working/cache` as a private Kaggle
Dataset `dsa1-cache-<part>` shared with the team; notebooks merge all attached parts automatically.

Then: semantic entropy (Sai Akhil, GPU) and precision-shift runs (Shivam, GPU):
```bash
python scripts/compute_semantic_entropy.py --model qwen3-4b --dataset bioasq --cache-root <CACHE_ROOT>
python scripts/build_cache.py --model qwen3-4b --dataset bioasq --precision 4bit --no-samples \
       --qids-file data/splits/precision_subset.json
```

## Hand-over between tasks

Task 1 writes `results/scores/<model>/<dataset>.parquet` (columns `qid, err, u_tokprob, u_se, u_probe, ...`).
On Kaggle, Tasks 2 and 3 add Task 1's notebook as an input and pick the tables up automatically; until then they
fall back to token probability computed from the cache, so nobody waits.

## Conventions

- Every uncertainty score: higher = more likely wrong. `err = 1` means a wrong answer (normalised exact match).
- A question is answered when `u <= t`; `t = -inf` means abstain on everything.
- Never use target-domain labels anywhere except to measure realised error.

## Before submission

```bash
pip freeze | grep -E "^(torch|transformers|accelerate|bitsandbytes|datasets|sentence-transformers|scikit-learn|xgboost|scipy|numpy|pandas|pyarrow)==" > requirements-lock.txt
```
Run each notebook top to bottom on Kaggle (Save Version → Save & Run All), make it public, test the link in an
incognito window, and list it in `DSA1_Notebook_Links.txt`.
