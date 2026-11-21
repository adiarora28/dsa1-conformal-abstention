"""Build the notebook templates in notebooks/. Run: python tools/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

from nb_common import REFS, SCORES_ROOT, SETUP, SPLITS

OUT = Path(__file__).resolve().parents[1] / "notebooks"


def nb(cells):
    book = nbf.v4.new_notebook()
    book.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    book.cells = [nbf.v4.new_markdown_cell(c[1]) if c[0] == "md" else nbf.v4.new_code_cell(c[1]) for c in cells]
    return book


# =============================================================================== 00 cache
CACHE = [
("md", """# 00 – Build the shared answer cache (GPU, Kaggle)

Guide Step 0. Every member runs this notebook on **their own Kaggle account** for their part of the cache.

| Owner | Datasets to generate | Extra |
| --- | --- | --- |
| Sai Akhil Singade | `triviaqa_probe`, `triviaqa_pool` | semantic entropy for all models (section 5) |
| Aditya Arora | `bioasq` | – |
| Shivam Vivek Rai | `mmlu_med`, `mmlu_other` | precision-shift runs (section 6) |

**Kaggle settings:** Accelerator = GPU T4 x2 (or P100), Internet = On, Persistence = Files.
Fixed scope: Qwen3-4B and Qwen3-1.7B with 10 samples, Qwen3-8B greedy answers only (no semantic entropy for 8B). Order: 4B first (baseline), then 1.7B, then 8B. Each run is resumable."""),
("code", SETUP),
("code", r'''ME = "Sai Akhil"   # <- "Sai Akhil" | "Aditya" | "Shivam"
OWNER_DATASETS = {"Sai Akhil": ["triviaqa_probe", "triviaqa_pool"],
                  "Aditya": ["bioasq"],
                  "Shivam": ["mmlu_med", "mmlu_other"]}
OUT_ROOT = Path("/kaggle/working/cache") if ON_KAGGLE else Path("cache")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

import shutil, subprocess, time
def run(cmd):
    print(">>", cmd)
    subprocess.run(cmd, shell=True, check=True)

# Resume: copy what earlier versions of YOUR cache dataset already contain (attach it as an input first).
for p in Path(CACHE_ROOT).glob("*/*"):
    if p.name in OWNER_DATASETS[ME]:
        dst = OUT_ROOT / p.parent.name / p.name
        if not dst.exists():
            shutil.copytree(p.resolve(), dst)
            print("resumed from input:", dst)'''),
("md", "## 1. Five-minute smoke test (do this before any long run)"),
("code", 'run("python scripts/smoke_test.py --model qwen3-1.7b")'),
("md", "## 2. Timing check: 48 questions per model, then an estimate for your whole part"),
("code", r'''from dsa1.cache_io import cache_dir, load_records
from dsa1.data import read_questions
MODEL = "qwen3-4b"
SIZES = {"triviaqa_probe": 2000, "triviaqa_pool": 3000, "bioasq": len(read_questions("bioasq")),
         "mmlu_med": 945, "mmlu_other": 2000}
total = 0.0
for model, extra in [("qwen3-4b", ""), ("qwen3-1.7b", ""), ("qwen3-8b", " --no-samples")]:
    for ds in OWNER_DATASETS[ME]:
        run(f"python scripts/build_cache.py --model {model} --dataset {ds} --cache-root {OUT_ROOT} --limit 48{extra}")
        rec = load_records(cache_dir(OUT_ROOT, model, ds))
        per_q = rec[[c for c in ["t_greedy", "t_samples", "t_hidden", "t_forward"] if c in rec]].sum(axis=1).mean()
        hours = per_q * SIZES[ds] / 3600
        total += hours
        print(f"{model} {ds}: {per_q:.2f} s/question -> about {hours:.1f} GPU hours for {SIZES[ds]} questions")
print(f"Total for your part: about {total:.1f} GPU hours. Post this number in the team chat.")'''),
("md", """## 3. Full runs

Kaggle stops a session after about 12 hours. The script skips questions already saved, so rerun this cell
in a new session after re-attaching your dataset (section 4) and running the resume cell above."""),
("code", r'''# Fixed scope: Qwen3-4B and Qwen3-1.7B with 10 samples (semantic entropy); Qwen3-8B greedy answers only
for model, extra in [("qwen3-4b", ""), ("qwen3-1.7b", ""), ("qwen3-8b", " --no-samples")]:
    for ds in OWNER_DATASETS[ME]:
        run(f"python scripts/build_cache.py --model {model} --dataset {ds} --cache-root {OUT_ROOT}{extra}")'''),
("md", """## 4. Publish your part as a Kaggle Dataset

1. Click **Save Version → Save & Run All** (or Quick Save if outputs are already in `/kaggle/working`).
2. Open the saved version → **Output** → **New Dataset** → name it `dsa1-cache-<your part>`
   (`triviaqa`, `bioasq` or `mmlu`), keep it **Private**, then **Settings → Sharing** → add your two teammates.
3. Next time, add a new version of the same dataset instead of creating a new one.

CLI alternative (needs your `kaggle.json` API token as a Kaggle secret):
```
kaggle datasets init -p /kaggle/working/cache        # first time: edit id/title in dataset-metadata.json
kaggle datasets create -p /kaggle/working/cache --dir-mode zip
kaggle datasets version -p /kaggle/working/cache -m "qwen3-8b bioasq" --dir-mode zip
```
Teammates then attach all three `dsa1-cache-*` datasets; `link_cache_roots` merges them automatically."""),
("md", "## 5. Sai Akhil only – semantic entropy (GPU, after all three cache parts are attached)"),
("code", r'''if ME == "Sai Akhil":
    for model in ["qwen3-4b", "qwen3-1.7b"]:   # fixed scope: no semantic entropy for Qwen3-8B
        for ds in ["triviaqa_pool", "bioasq"]:
            run(f"python scripts/compute_semantic_entropy.py --model {model} --dataset {ds} --cache-root {CACHE_ROOT}")
    run(f"python scripts/hand_check_sheet.py --cache-root {CACHE_ROOT}")   # 100 answers to label by hand'''),
("md", "## 6. Shivam only – precision shift for Task 3 (Qwen3-4B in 8-bit and 4-bit, greedy only)"),
("code", r'''if ME == "Shivam":
    for prec in ["8bit", "4bit"]:
        for ds in ["triviaqa_pool", "bioasq"]:
            run(f"python scripts/build_cache.py --model qwen3-4b --dataset {ds} --precision {prec} "
                f"--no-samples --qids-file data/splits/precision_subset.json --cache-root {OUT_ROOT}")'''),
("code", 'run("pip freeze | grep -E \\"^(torch|transformers|accelerate|bitsandbytes|datasets|sentence-transformers|scikit-learn|xgboost|scipy|numpy|pandas|pyarrow)==\\" > /kaggle/working/requirements-lock.txt; cat /kaggle/working/requirements-lock.txt")'),
]

# =============================================================================== Task 1
T1 = [
("md", """# Task 1 – Uncertainty Scores
**Owner:** Sai Akhil Singade · DSA1 · Action Learning

## 1. Introduction
**Research context.** #1 Vashurin et al. 2025 (LM-Polygraph benchmark, PRR), #2 Farquhar et al. 2024
(semantic entropy), #3 Kossen et al. 2024 (hidden-state probes and their out-of-distribution behaviour),
#4 Shelmanov et al. 2025 (hidden-state detectors may memorise topics).

**Objectives.** Compare three uncertainty scores for detecting wrong answers of Qwen3-1.7B/4B/8B, in domain
(TriviaQA) and under shift (BioASQ, MMLU medical), and measure their compute cost.

**Research question.** Which score best ranks wrong answers above right ones on TriviaQA, and how much of
that ranking survives on BioASQ and MMLU medical?

**Hypothesis.** The probe is strongest in domain but loses most under shift; semantic entropy transfers best;
token probability is cheapest and in between.

**Convention.** Every score is an *uncertainty*: higher = more likely wrong. Positive class = wrong answer."""),
("md", "## 2. Setup"),
("code", SETUP),
("code", SPLITS),
("md", "## 3. Data loading & exploration"),
("code", r'''from dsa1 import experiments as E
from dsa1.cache_io import cache_dir, load_records
MODELS = list(C.MODELS)
DATASETS = ["triviaqa_pool", "bioasq", "mmlu_med"]
base, rows = {}, []
for m in MODELS:
    for ds in DATASETS:
        try:
            df = E.base_scores(CACHE_ROOT, m, ds)
        except FileNotFoundError:
            print("missing:", m, ds)
            continue
        base[(m, ds)] = df
        rows.append({"model": m, "dataset": ds, "n": len(df), "accuracy (EM)": 1 - df["err"].mean(),
                     "accuracy (F1>=0.5)": 1 - df["err_f1"].mean(), "semantic entropy ready": df["u_se"].notna().mean()})
accuracy = pd.DataFrame(rows)
accuracy.to_csv("results/metrics/task1_accuracy.csv", index=False)
accuracy'''),
("code", r'''# Five examples per open-ended domain (Qwen3-4B), including wrong ones
for ds in ["triviaqa_pool", "bioasq"]:
    rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", ds))
    display(rec.sample(5, random_state=0)[["question", "answer", "aliases", "correct_em", "f1"]])'''),
("code", r'''# Hand-check agreement (Step 0.5): fill results/hand_check.csv, column human_correct = 1/0
hc = Path("results/hand_check.csv")
if hc.exists():
    h = pd.read_csv(hc)
    h = h[h["human_correct"].astype(str).str.strip().isin(["0", "1"])]
    if len(h):
        agree = (h["human_correct"].astype(int) == h["correct_em"].astype(int)).mean()
        print(f"Exact-match labels agree with manual labels on {agree:.1%} of {len(h)} answers")
        display(h[h["human_correct"].astype(int) != h["correct_em"].astype(int)].head(10))'''),
("md", """## 4. Preprocessing
Splits come from `data/splits/splits.json` (five seeds). Answers are normalised SQuAD-style before exact match.
The probe trains only on the fixed 2,000-question TriviaQA probe set, which is disjoint from every
calibration and test set."""),
("code", r'''for (m, ds), df in base.items():
    for s in SEEDS:
        missing = set(SPLITS["seeds"][str(s)][TEST_KEYS[ds]]) - set(df["qid"])
        assert not missing, f"{m}/{ds} seed {s}: {len(missing)} test questions missing from cache"
print("all test questions are in the cache")'''),
("md", """## 5. Model definition
- Qwen3-1.7B and 4B in float16, Qwen3-8B in 4-bit NF4; thinking disabled; greedy answer for all, plus 10 samples (T=1.0, top-p 0.95) for 1.7B and 4B.
- **V1 token probability:** minus the mean token log-probability of the greedy answer (MMLU: minus log-prob of the chosen option).
- **V2 semantic entropy:** 10 samples clustered by bidirectional entailment with DeBERTa-large-MNLI (computed on GPU by
  `scripts/compute_semantic_entropy.py`); MMLU uses the entropy of the four option probabilities.
- **V3 probe:** logistic regression on one hidden state; position (last prompt / last answer token), layer
  (25/50/100% depth) and C chosen by 5-fold CV on the probe set only."""),
("code", r'''import joblib
probes = {}
for m in MODELS:
    try:
        probe, pos, layer, table = E.select_probe(CACHE_ROOT, m)
    except FileNotFoundError:
        print("no probe set for", m)
        continue
    probes[m] = (probe, pos, layer)
    joblib.dump(probe, f"results/probe_{m}.joblib")
    print(m, "-> best:", pos, "token, layer slot", layer)
    display(table.assign(model=m))'''),
("md", "## 6. Running the variations (apply the probe, write the score tables used by Tasks 2 and 3)"),
("code", r'''for (m, ds), df in list(base.items()):
    if m in probes:
        probe, pos, layer = probes[m]
        df = E.add_probe_scores(df, probe, CACHE_ROOT, m, ds, pos, layer)
    else:
        df = df.assign(u_probe=np.nan)
    base[(m, ds)] = df
    E.save_score_table(df, m, ds)
print("score tables written to results/scores/<model>/<dataset>.parquet")'''),
("md", """Optional cross-check with LM-Polygraph (guide Task 1, Steps 2–3): run its maximum-sequence-probability and
semantic-entropy estimators on 50 TriviaQA questions with Qwen3-4B, save `results/polygraph_check.csv` with
columns `qid, polygraph_msp, polygraph_se`, then run the next cell. If LM-Polygraph does not run with Qwen3,
record it as a technical challenge."""),
("code", r'''pc = Path("results/polygraph_check.csv")
if pc.exists():
    from scipy.stats import spearmanr
    chk = pd.read_csv(pc).merge(base[("qwen3-4b", "triviaqa_pool")], on="qid")
    print("Spearman token prob vs LM-Polygraph MSP:", spearmanr(chk["u_tokprob"], -chk["polygraph_msp"]).statistic)
    print("Spearman semantic entropy vs LM-Polygraph:", spearmanr(chk["u_se"], chk["polygraph_se"]).statistic)'''),
("md", "## 7. Evaluation – AUROC and PRR, mean ± std over five seeds"),
("code", r'''from dsa1.metrics import auroc_wrong, prediction_rejection_ratio, paired_bootstrap_auroc_diff, mean_std
SCORES = ["u_tokprob", "u_se", "u_probe"]
rows = []
for (m, ds), df in base.items():
    for s in SEEDS:
        sub = E.rows_for(df, SPLITS["seeds"][str(s)][TEST_KEYS[ds]])
        for sc in SCORES:
            if sub[sc].isna().any():
                continue
            rows.append({"model": m, "dataset": ds, "seed": s, "score": sc,
                         "auroc": auroc_wrong(sub[sc], sub["err"]),
                         "prr": prediction_rejection_ratio(sub[sc], sub["err"])})
res = pd.DataFrame(rows)
res.to_csv("results/metrics/task1_auroc_prr.csv", index=False)
for metric in ["auroc", "prr"]:
    print(metric.upper(), "(MMLU uses all 945 questions in every seed, so its std is 0)")
    display(res.pivot_table(index="score", columns=["model", "dataset"], values=metric, aggfunc=mean_std))'''),
("code", r'''# Is the difference between two scores real? Paired bootstrap on seed 0, Qwen3-4B
rows = []
for ds in DATASETS:
    if ("qwen3-4b", ds) not in base:
        continue
    sub = E.rows_for(base[("qwen3-4b", ds)], SPLITS["seeds"]["0"][TEST_KEYS[ds]])
    for a, b in [("u_se", "u_tokprob"), ("u_probe", "u_tokprob"), ("u_probe", "u_se")]:
        if sub[[a, b]].isna().any().any():
            continue
        rows.append({"dataset": ds, "comparison": f"{a} - {b}", **paired_bootstrap_auroc_diff(sub["err"], sub[a], sub[b])})
pd.DataFrame(rows)'''),
("code", r'''# Cost per score (GPU seconds per question, from the cache timings)
rows = []
for (m, ds), df in base.items():
    for sc in SCORES:
        rows.append({"model": m, "dataset": ds, "score": sc, "gpu_s_per_question": E.cost_per_question(df, sc).mean(),
                     "extra_generations": 10 if sc.startswith("u_se") and ds != "mmlu_med" else 0})
cost = pd.DataFrame(rows)
cost.to_csv("results/metrics/task1_cost.csv", index=False)
cost.pivot_table(index="score", columns=["model", "dataset"], values="gpu_s_per_question").round(3)'''),
("md", "## 8. Comparison & analysis"),
("code", r'''# Grouped bars: AUROC by score and domain, one panel per model
fig, axes = plt.subplots(1, len(MODELS), figsize=(5 * len(MODELS), 4), sharey=True)
for ax, m in zip(np.atleast_1d(axes), MODELS):
    g = res[res["model"] == m].groupby(["dataset", "score"])["auroc"].agg(["mean", "std"]).unstack()
    if g.empty:
        continue
    g["mean"].plot.bar(yerr=g["std"], ax=ax, rot=0, capsize=3)
    ax.set_title(m); ax.set_ylabel("AUROC (wrong answer)"); ax.set_ylim(0.4, 1.0); ax.axhline(0.5, ls=":", c="grey")
plt.tight_layout(); plt.savefig("results/figures/task1_auroc_bars.png", dpi=150); plt.show()'''),
("code", r'''# ROC curves, Qwen3-4B, seed 0
from sklearn.metrics import roc_curve
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, ds in zip(axes, DATASETS):
    if ("qwen3-4b", ds) not in base:
        continue
    sub = E.rows_for(base[("qwen3-4b", ds)], SPLITS["seeds"]["0"][TEST_KEYS[ds]])
    for sc in SCORES:
        if sub[sc].isna().any():
            continue
        fpr, tpr, _ = roc_curve(sub["err"], sub[sc])
        ax.plot(fpr, tpr, label=f"{sc} ({auroc_wrong(sub[sc], sub['err']):.2f})")
    ax.plot([0, 1], [0, 1], ":", c="grey"); ax.set_title(f"Qwen3-4B – {ds}")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate (wrong answers caught)"); ax.legend()
plt.tight_layout(); plt.savefig("results/figures/task1_roc_4b.png", dpi=150); plt.show()'''),
("code", r'''# Deeper analysis 1 – transfer gap: in-domain AUROC minus shifted AUROC (positive = worse under shift)
piv = res.pivot_table(index=["model", "score", "seed"], columns="dataset", values="auroc")
gap = pd.DataFrame({"to BioASQ": piv["triviaqa_pool"] - piv.get("bioasq"),
                    "to MMLU med": piv["triviaqa_pool"] - piv.get("mmlu_med")}).reset_index()
gsum = gap.groupby(["model", "score"])[["to BioASQ", "to MMLU med"]].agg(["mean", "std"])
display(gsum.round(3))
fig, ax = plt.subplots(figsize=(8, 4))
for i, (key, row) in enumerate(gsum.iterrows()):
    ax.errorbar(i, row[("to BioASQ", "mean")], yerr=row[("to BioASQ", "std")], fmt="o", capsize=3)
ax.set_xticks(range(len(gsum))); ax.set_xticklabels([f"{m}\n{s}" for m, s in gsum.index], rotation=60, fontsize=8)
ax.axhline(0, c="grey", ls=":"); ax.set_ylabel("AUROC drop TriviaQA → BioASQ"); ax.set_title("Transfer gap per score and model")
plt.tight_layout(); plt.savefig("results/figures/task1_transfer_gap.png", dpi=150); plt.show()'''),
("code", r'''# AUROC versus compute cost (efficiency vs performance, as in the implementation guide)
merged = res.groupby(["model", "dataset", "score"])["auroc"].mean().reset_index().merge(cost, on=["model", "dataset", "score"])
fig, ax = plt.subplots(figsize=(7, 4))
for sc, g in merged[merged["dataset"] == "bioasq"].groupby("score"):
    ax.scatter(g["gpu_s_per_question"], g["auroc"], label=sc, s=60)
    for _, r in g.iterrows():
        ax.annotate(r["model"].replace("qwen3-", ""), (r["gpu_s_per_question"], r["auroc"]), fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("GPU seconds per question (log)"); ax.set_ylabel("AUROC on BioASQ"); ax.legend()
ax.set_title("Quality vs cost under shift"); plt.tight_layout(); plt.savefig("results/figures/task1_cost_quality.png", dpi=150); plt.show()'''),
("code", r'''# Deeper analysis 2 – failure cases (Qwen3-4B, seed 0): export a sheet to label by hand
rows = []
for ds in ["triviaqa_pool", "bioasq"]:
    sub = E.rows_for(base[("qwen3-4b", ds)], SPLITS["seeds"]["0"][TEST_KEYS[ds]])
    rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", ds)).set_index("qid")
    for sc in SCORES:
        if sub[sc].isna().any():
            continue
        lo, hi = sub[sc].quantile(0.2), sub[sc].quantile(0.8)
        cw = sub[(sub[sc] <= lo) & (sub["err"] == 1)]
        ur = sub[(sub[sc] >= hi) & (sub["err"] == 0)]
        for kind, part in [("confident_but_wrong", cw), ("uncertain_but_right", ur)]:
            for q in part.sample(min(50, len(part)), random_state=0)["qid"]:
                rows.append({"dataset": ds, "score": sc, "kind": kind, "qid": q, "question": rec.loc[q, "question"],
                             "answer": rec.loc[q, "answer"], "aliases": " | ".join(list(rec.loc[q, "aliases"])[:8]), "category": ""})
fc = pd.DataFrame(rows)
fc_path = Path("results/failure_cases.csv")
if fc_path.exists() and pd.read_csv(fc_path)["category"].notna().any():
    print("results/failure_cases.csv already has labels - not overwritten")
else:
    fc.to_csv(fc_path, index=False)
print("Label column 'category' with: grading/alias error, partly correct, entity confusion, ambiguous question, knowledge gap")
fc.groupby(["dataset", "score", "kind"]).size().unstack()'''),
("code", r'''# After labelling: category counts, and how often the scores fail on the same questions
fc = pd.read_csv("results/failure_cases.csv")
if fc["category"].notna().any():
    display(fc.dropna(subset=["category"]).pivot_table(index="category", columns=["dataset", "score", "kind"],
                                                         values="qid", aggfunc="count", fill_value=0))
for ds in ["triviaqa_pool", "bioasq"]:
    sets = {sc: set(g["qid"]) for sc, g in fc[(fc["dataset"] == ds) & (fc["kind"] == "confident_but_wrong")].groupby("score")}
    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = sets[keys[i]], sets[keys[j]]
            print(f"{ds}: overlap of confident-but-wrong sets {keys[i]} vs {keys[j]}: {len(a & b) / max(1, len(a | b)):.0%}")'''),
("md", "### Confusion matrix and demo (for the defense)"),
("code", r'''# Confusion matrix (defense guide): flag as many BioASQ answers as there are wrong ones, by each score
sub = E.rows_for(base[("qwen3-4b", "bioasq")], SPLITS["seeds"]["0"]["test_bioasq"])
panels, rows = [], []
for sc in SCORES:
    if sub[sc].isna().any():
        continue
    c = E.detection_confusion(sub[sc], sub["err"])
    panels.append((sc, c)); rows.append({"score": sc, **c})
pd.DataFrame(rows).to_csv("results/metrics/task1_confusion.csv", index=False)
E.plot_confusions(panels, "detection", "results/figures/task1_confusion.png", "Qwen3-4B · BioASQ · seed 0")
plt.show()
pd.DataFrame(rows)'''),
("code", r'''# Demo – inference on test samples: the model's own answers to three BioASQ test questions, with all scores
pd.set_option("display.max_colwidth", 80)
rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", "bioasq")).set_index("qid")
lo, hi = sub["u_tokprob"].quantile(0.2), sub["u_tokprob"].quantile(0.8)
cases = {"confident and right": sub[(sub["u_tokprob"] <= lo) & (sub["err"] == 0)],
         "uncertain and wrong (caught)": sub[(sub["u_tokprob"] >= hi) & (sub["err"] == 1)],
         "confident but wrong (missed)": sub[(sub["u_tokprob"] <= lo) & (sub["err"] == 1)]}
rows = []
for case, part in cases.items():
    if len(part) == 0:
        continue
    q = part.sample(1, random_state=1).iloc[0]
    r = rec.loc[q["qid"]]
    rows.append({"case": case, "question": r["question"], "model answer": r["answer"],
                 "reference": " | ".join(list(r["aliases"])[:3]), "correct": not bool(q["err"]),
                 **{sc: round(float(q[sc]), 3) for sc in SCORES if not pd.isna(q[sc])}})
demo = pd.DataFrame(rows)
demo.to_csv("results/metrics/task1_demo_examples.csv", index=False)
demo'''),
("md", """## 9. Discussion
*Write after the results are in. Answer each point in 2–3 sentences.*
- Which score is best in domain, and is the difference significant?
- Which score transfers best to BioASQ and MMLU medical? Does the probe's in-domain advantage survive?
- What do the failure categories say: score failures or model failures (shared confident errors)?
- Cost: is semantic entropy's quality worth 10 extra generations?
- Limitations: exact-match labels (quote the hand-check agreement), single model family, no semantic entropy for Qwen3-8B (4-bit sampling exceeded the free GPU budget).
- Link to papers #1–#4, and the MVP implication: which score the assistant should use."""),
("md", REFS),
]

# =============================================================================== Task 2
T2 = [
("md", """# Task 2 – Domain-Shift Correction
**Owner:** Aditya Arora · DSA1 · Action Learning

## 1. Introduction
**Research context.** #8 Lin et al. 2026 (DS-CP), #9 Wang N.-S. et al. 2025 (weights from a domain classifier),
#10 Hu et al. 2026 (conformal guarantees fail under prompt shift), #7 Wang Z. et al. 2025 (detecting shifted
questions); Tibshirani et al. 2019 (weighted conformal).

**Objective.** Measure how far the realised error among answered questions drifts from the target when thresholds
are calibrated on TriviaQA and applied to BioASQ and MMLU medical, and how much of that drift reweighting removes.

**Research question.** With thresholds calibrated on TriviaQA, how large is the gap between realised and target error
under shift, and how much do TF-IDF and DS-CP weights reduce it?

**Hypothesis.** Standard calibration exceeds the target on BioASQ; both weightings narrow the gap; DS-CP narrows it most.

**Shared rule.** All variations use the same rule; only the weights change:
answer if u ≤ t, with t the largest value such that (Σ wᵢ·1[uᵢ≤t]·errᵢ + λ) / (Σ wᵢ·1[uᵢ≤t] + λ) ≤ target.
λ is DS-CP's mass at infinity (λ = 1 for uniform weights, λ = max wᵢ for weighted methods), adapted from
coverage to error among answered questions."""),
("md", "## 2. Setup"),
("code", SETUP),
("code", SCORES_ROOT),
("code", SPLITS),
("md", "## 3. Data loading & exploration"),
("code", r'''from dsa1 import data
QTEXT = {}
for t in ["triviaqa", "bioasq", "mmlu_med", "mmlu_other"]:
    try:
        q = data.read_questions(t)
        QTEXT.update(dict(zip(q["qid"], q["question"])))
    except FileNotFoundError:
        pass
if not Path("data/embeddings/triviaqa.npz").exists():
    subprocess.run([sys.executable, "scripts/embed_questions.py"], check=True)
EMB = {}
for t in ["triviaqa", "bioasq", "mmlu_med", "mmlu_other"]:
    if Path(f"data/embeddings/{t}.npz").exists():
        EMB.update(E.load_embeddings(t))
print(len(QTEXT), "questions,", len(EMB), "embeddings")

MODELS, ALL = [], {}
for m in C.MODELS:
    try:
        ALL[m] = pd.concat([get_scores(m, ds) for ds in ["triviaqa_pool", "bioasq", "mmlu_med"]]).set_index("qid")
        MODELS.append(m)
    except FileNotFoundError as e:
        print("skip", m, e)
SCORES = [s for s in ["u_tokprob", "u_se", "u_probe"] if any(ALL[m][s].notna().all() for m in MODELS if s in ALL[m])]
print("models:", MODELS, "| scores:", SCORES)
pd.DataFrame({m: ALL[m].groupby("dataset")["err"].mean() for m in MODELS}).rename_axis("error rate")'''),
("md", """## 4. Preprocessing – test sets with graded shift
Per seed: mixtures of 1,000 questions with 0/25/50/75/100% BioASQ (rest TriviaQA test), each MMLU medical subject,
and all 945 MMLU medical questions. Weights see only the questions of each test set, never their labels."""),
("code", r'''from dsa1.splits import mixture
MM = data.read_questions("mmlu_med")
def test_sets(seed):
    d = SPLITS["seeds"][str(seed)]
    sets = {f"mix_{int(f * 100):03d}": mixture(d["test_triviaqa"], d["test_bioasq"], f, n=1000, seed=seed)
            for f in [0, 0.25, 0.5, 0.75, 1.0]}
    sets["mmlu_all"] = d["test_mmlu_med"]
    for subj in C.MMLU_MEDICAL_SUBJECTS:
        sets[f"mmlu_{subj}"] = MM.loc[MM["subject"] == subj, "qid"].tolist()
    return sets
{k: len(v) for k, v in test_sets(0).items()}'''),
("md", """## 5. Model definition – the three weightings
- **V1 uniform:** wᵢ = 1, λ = 1 (standard split conformal).
- **V2 TF-IDF:** logistic regression on word 1–2-gram TF-IDF, 5-fold cross-fitted, weights clipped at the 99th percentile, λ = max wᵢ.
- **V3 DS-CP:** XGBoost on all-MiniLM-L6-v2 embeddings with the paper's settings (Appendix C), λ = max wᵢ (Algorithm 1)."""),
("code", r'''from dsa1.weights import uniform_weights, tfidf_weights, dscp_weights, embedding_shift_distance, weighted_conformal_quantile
METHODS = ["uniform", "tfidf", "dscp"]
def make_weights(cal_q, tgt_q, method, **kw):
    if method == "uniform":
        return uniform_weights(len(cal_q))
    if method == "tfidf":
        return tfidf_weights([QTEXT[q] for q in cal_q], [QTEXT[q] for q in tgt_q], **kw)
    if method == "dscp":
        return dscp_weights(E.embed_rows(EMB, cal_q), E.embed_rows(EMB, tgt_q), **kw)
    raise ValueError(method)'''),
("md", """## 6. Running the variations

### 6a. Reproduction first: DS-CP coverage on MMLU (90% target)
**Exact reproduction:** clone the authors' code (`https://github.com/zhexiaolin/CP`), run it for one or two models
on the logits they reuse from LLM-Uncertainty-Bench, and confirm standard CP under-covers on some subject pairs while
DS-CP stays closer to 90%. Then run the cell below to check that *our* `weighted_conformal_quantile` + `dscp_weights`
give the same qualitative result with Qwen3 (calibrate on non-medical MMLU, test on each medical subject)."""),
("code", r'''from dsa1.cache_io import cache_dir, load_records
def lac_scores(rec):  # nonconformity = 1 - probability of the correct option (LAC, as in the paper)
    P = np.stack(rec["option_probs"].values)
    return P, 1 - P[np.arange(len(P)), rec["answer_idx"].to_numpy()]
rows = []
try:
    rec_o = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", "mmlu_other"))
    rec_m = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", "mmlu_med"))
    _, S_cal = lac_scores(rec_o)
    for subj in C.MMLU_MEDICAL_SUBJECTS:
        r = rec_m[rec_m["subject"] == subj]
        P_t, S_t = lac_scores(r)
        W = dscp_weights(E.embed_rows(EMB, rec_o["qid"].tolist()), E.embed_rows(EMB, r["qid"].tolist()))
        for name, w, lam in [("standard CP", np.ones(len(S_cal)), 1.0), ("DS-CP", W.w, W.lam)]:
            q = weighted_conformal_quantile(S_cal, w, lam, alpha=0.10)
            rows.append({"subject": subj, "method": name, "coverage": float((S_t <= q).mean()),
                         "avg_set_size": float(((1 - P_t) <= q).sum(axis=1).mean()), "ess": float(w.sum() ** 2 / (w ** 2).sum())})
    repro = pd.DataFrame(rows)
    repro.to_csv("results/metrics/task2_dscp_reproduction.csv", index=False)
    display(repro.pivot_table(index="subject", columns="method", values=["coverage", "avg_set_size"]).round(3))
except FileNotFoundError as e:
    print("mmlu_other cache not available yet:", e)'''),
("md", "### 6b. Main experiment: 3 weightings × 3 targets × all test sets × 5 seeds × scores × models"),
("code", r'''import time
rows = []
for s in SEEDS:
    cal_q = SPLITS["seeds"][str(s)]["cal"]
    for name, tgt_q in test_sets(s).items():
        t0 = time.perf_counter()
        Ws = {m: make_weights(cal_q, tgt_q, m) for m in METHODS}
        t_w = time.perf_counter() - t0
        dist = embedding_shift_distance(E.embed_rows(EMB, cal_q), E.embed_rows(EMB, tgt_q))
        for model in MODELS:
            cal, tst = ALL[model].loc[cal_q], ALL[model].loc[tgt_q]
            for sc in SCORES:
                if cal[sc].isna().any() or tst[sc].isna().any():
                    continue
                for m in METHODS:
                    for r in E.shift_eval(cal[sc].values, cal["err"].values, tst[sc].values, tst["err"].values, Ws[m]):
                        rows.append({"seed": s, "test_set": name, "model": model, "score": sc,
                                     "shift_dist": dist, "dscp_auc": Ws["dscp"].domain_auc,
                                     "tfidf_auc": Ws["tfidf"].domain_auc, "weight_time_s": t_w, **r})
    print("seed", s, "done")
res = pd.DataFrame(rows)
res.to_csv("results/metrics/task2_results.csv", index=False)
res.head()'''),
("md", "## 7. Evaluation – signed gap (realised − target) and coverage, mean ± std over seeds"),
("code", r'''from dsa1.metrics import mean_std
main = res[(res["model"] == "qwen3-4b") & (res["score"] == "u_tokprob")]
display(main.pivot_table(index=["test_set"], columns=["target", "method"], values="gap", aggfunc=mean_std))
display(main.pivot_table(index=["test_set"], columns=["target", "method"], values="coverage", aggfunc=mean_std))
# 95% t-interval over the five seeds: does the gap exclude zero?
from scipy.stats import t as student_t
g = main.groupby(["test_set", "target", "method"])["gap"]
summary = pd.DataFrame({"mean_gap": g.mean(), "half_width": g.std() * student_t.ppf(0.975, g.count() - 1) / np.sqrt(g.count())})
summary["excludes_zero"] = (summary["mean_gap"].abs() > summary["half_width"])
summary.round(3)'''),
("md", "## 8. Comparison & analysis"),
("code", r'''# Key figure – gap versus shift size (target 10%), Qwen3-4B, token probability
d = main[main["target"] == 0.10]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, x, lab in [(axes[0], "dscp_auc", "domain-classifier AUC (MiniLM + XGBoost)"), (axes[1], "shift_dist", "mean cosine distance")]:
    for m, gm in d.groupby("method"):
        agg = gm.groupby("test_set").agg(x=(x, "mean"), gap=("gap", "mean"), sd=("gap", "std")).sort_values("x")
        ax.errorbar(agg["x"], agg["gap"], yerr=agg["sd"], marker="o", capsize=3, label=m)
    ax.axhline(0, c="grey", ls=":"); ax.set_xlabel(lab); ax.set_ylabel("realised − target error (answered)")
    ax.legend(); ax.set_title("Gap vs shift size, target 10%")
plt.tight_layout(); plt.savefig("results/figures/task2_gap_vs_shift.png", dpi=150); plt.show()'''),
("code", r'''# Coverage versus gap (what does each method pay for its correction?)
fig, ax = plt.subplots(figsize=(6, 4))
for m, gm in d[d["test_set"].isin(["mix_100", "mmlu_all"])].groupby("method"):
    ax.scatter(gm["coverage"], gm["gap"], label=m, alpha=0.7)
ax.axhline(0, c="grey", ls=":"); ax.set_xlabel("coverage (share answered)"); ax.set_ylabel("gap"); ax.legend()
plt.tight_layout(); plt.savefig("results/figures/task2_coverage_vs_gap.png", dpi=150); plt.show()'''),
("code", r'''# Weight histograms on 100% BioASQ, seed 0
cal_q = SPLITS["seeds"]["0"]["cal"]; tgt_q = test_sets(0)["mix_100"]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
for ax, m in zip(axes, ["tfidf", "dscp"]):
    W = make_weights(cal_q, tgt_q, m)
    ax.hist(np.log10(W.w + 1e-12), bins=50); ax.set_title(f"{m}: ESS = {W.ess:.0f} of {len(W.w)}, AUC = {W.domain_auc:.3f}")
    ax.set_xlabel("log10 weight")
plt.tight_layout(); plt.savefig("results/figures/task2_weight_hist.png", dpi=150); plt.show()'''),
("code", r'''# Ablations on 100% BioASQ: TF-IDF clipping and DS-CP lambda (gamma × max weight), token probability, Qwen3-4B
rows = []
for s in SEEDS:
    cal_q = SPLITS["seeds"][str(s)]["cal"]; tgt_q = test_sets(s)["mix_100"]
    cal, tst = ALL["qwen3-4b"].loc[cal_q], ALL["qwen3-4b"].loc[tgt_q]
    variants = {f"tfidf clip={c}": make_weights(cal_q, tgt_q, "tfidf", clip_quantile=c) for c in [None, 0.99, 0.95]}
    variants.update({f"dscp gamma={g}": make_weights(cal_q, tgt_q, "dscp", gamma=g) for g in [0.5, 1.0, 2.0]})
    for name, W in variants.items():
        for r in E.shift_eval(cal["u_tokprob"].values, cal["err"].values, tst["u_tokprob"].values, tst["err"].values, W):
            rows.append({"seed": s, "variant": name, **r})
abl = pd.DataFrame(rows)
abl.to_csv("results/metrics/task2_ablations.csv", index=False)
abl.pivot_table(index="variant", columns="target", values=["gap", "coverage", "ess"], aggfunc="mean").round(3)'''),
("code", r'''# Shift alarm (SConU-style): share of test questions with conformal p-value < 0.05, vs share of BioASQ
from dsa1.weights import conformal_pvalues
rows = []
for s in SEEDS:
    cal_q = SPLITS["seeds"][str(s)]["cal"]
    for name, tgt_q in test_sets(s).items():
        if not name.startswith("mix_"):
            continue
        p = conformal_pvalues(ALL["qwen3-4b"].loc[cal_q, "u_tokprob"].values, ALL["qwen3-4b"].loc[tgt_q, "u_tokprob"].values)
        rows.append({"seed": s, "share_bioasq": int(name[-3:]) / 100, "flagged": float((p < 0.05).mean())})
alarm = pd.DataFrame(rows)
agg = alarm.groupby("share_bioasq")["flagged"].agg(["mean", "std"])
plt.errorbar(agg.index, agg["mean"], yerr=agg["std"], marker="o", capsize=3); plt.axhline(0.05, ls=":", c="grey")
plt.xlabel("share of BioASQ questions"); plt.ylabel("share flagged (p < 0.05)"); plt.title("Shift alarm")
plt.savefig("results/figures/task2_shift_alarm.png", dpi=150); plt.show()'''),
("md", "### Confusion matrix and demo (for the defense)"),
("code", r'''# Confusion matrix: answered/abstained × right/wrong on 100% BioASQ, seed 0, target 10%, each weighting
from dsa1.rules import weighted_plugin_threshold
cal_q = SPLITS["seeds"]["0"]["cal"]; tgt_q = test_sets(0)["mix_100"]
cal, tst = ALL["qwen3-4b"].loc[cal_q], ALL["qwen3-4b"].loc[tgt_q]
panels, rows, thresholds = [], [], {}
for m in METHODS:
    W = make_weights(cal_q, tgt_q, m)
    t = weighted_plugin_threshold(cal["u_tokprob"].values, cal["err"].values, 0.10, w=W.w, lam=W.lam)
    thresholds[m] = t
    c = E.abstention_confusion(tst["u_tokprob"], tst["err"], t)
    panels.append((m, c)); rows.append({"method": m, "threshold": t, **c})
pd.DataFrame(rows).to_csv("results/metrics/task2_confusion.csv", index=False)
E.plot_confusions(panels, "abstention", "results/figures/task2_confusion.png", "Qwen3-4B · BioASQ · target 10% · seed 0")
plt.show()
pd.DataFrame(rows)'''),
("code", r'''# Demo – inference on test samples: five BioASQ questions, the model's answer, and each method's decision
from dsa1.cache_io import cache_dir, load_records
pd.set_option("display.max_colwidth", 70)
rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", "bioasq")).set_index("qid")
pick = tst.sample(5, random_state=3)
demo = pd.DataFrame({
    "question": [rec.loc[q, "question"] for q in pick.index],
    "model answer": [rec.loc[q, "answer"] for q in pick.index],
    "correct": (pick["err"] == 0).values,
    "uncertainty": pick["u_tokprob"].round(3).values,
    **{m: np.where(pick["u_tokprob"] <= thresholds[m], "answer", "route to staff") for m in METHODS},
})
demo.to_csv("results/metrics/task2_demo_examples.csv", index=False)
demo'''),
("md", """## 9. Discussion
- How large is the gap without correction on BioASQ and MMLU medical, at each target?
- Does weighting reduce it? Does DS-CP beat the TF-IDF classifier? At what coverage cost?
- What do effective sample size and the weight histograms say about overlap between TriviaQA and BioASQ?
- Did the reproduction match the DS-CP paper qualitatively?
- Limitations: λ-adaptation to error among answered is ours; exact-match labels; one model family.
- MVP: which correction to use and whether the shift alarm can drive monitoring and recalibration."""),
("md", REFS),
]

# =============================================================================== Task 3
T3 = [
("md", """# Task 3 – Threshold Rule and Reliability
**Owner:** Shivam Vivek Rai · DSA1 · Action Learning

## 1. Introduction
**Research context.** #6 Wang Z. et al. 2026 (COIN), #5 Abbasi Yadkori et al. 2024 (conformal abstention);
Angelopoulos et al. 2024 (conformal risk control); Geifman & El-Yaniv 2017 (risk-coverage curves);
Doula & Mühlhäuser 2026 (inference-time changes break split-conformal guarantees).

**Objective.** Find the rule for setting the abstention threshold that keeps the error among answered questions
below 5%, 10% and 20%, and what it costs in coverage, in domain and under shift.

**Research question.** Standard conformal guarantees concern coverage, not error among answered questions. Which
rule actually keeps that error below the target?

**Hypothesis.** The naive rule often violates the target; conformal risk control bounds a different quantity;
COIN meets the target most reliably at some cost in coverage."""),
("md", "## 2. Setup"),
("code", SETUP),
("code", SCORES_ROOT),
("code", SPLITS),
("md", "## 3. Data loading & exploration"),
("code", r'''MODELS, ALL = [], {}
for m in C.MODELS:
    try:
        ALL[m] = pd.concat([get_scores(m, ds) for ds in ["triviaqa_pool", "bioasq", "mmlu_med"]]).set_index("qid")
        MODELS.append(m)
    except FileNotFoundError as e:
        print("skip", m, e)
SCORES = [s for s in ["u_tokprob", "u_se", "u_probe"] if any(ALL[m][s].notna().all() for m in MODELS if s in ALL[m])]
POOL = ALL["qwen3-4b"].index[ALL["qwen3-4b"]["dataset"] == "triviaqa_pool"].tolist()
print("models:", MODELS, "| scores:", SCORES, "| TriviaQA pool:", len(POOL))
pd.DataFrame({m: 1 - ALL[m].groupby("dataset")["err"].mean() for m in MODELS}).rename_axis("accuracy")'''),
("md", "## 4. Preprocessing\nCalibration = 1,000 TriviaQA questions per seed; tests = TriviaQA test, BioASQ, MMLU medical. Uniform weights throughout."),
("md", """## 5. Model definition – the three rules (see `src/dsa1/rules.py`)
- **V1 naive:** largest t with k_t / n_t ≤ α (n_t ≥ 20).
- **V2 conformal risk control:** largest t with (k_t + 1)/(n + 1) ≤ α; bounds answered-and-wrong over **all** questions.
- **V3 COIN:** largest t, scanned in a fixed order from the most confident, with Clopper-Pearson upper bound on k_t / n_t ≤ α at δ = 0.05.

The unit tests check each rule on a toy problem with a known answer."""),
("code", 'subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_rules_metrics.py"], check=True)'),
("md", "## 6. Running the variations – 5 seeds + 100 random re-splits"),
("code", r'''TESTS = {"triviaqa_pool": "test_triviaqa", "bioasq": "test_bioasq", "mmlu_med": "test_mmlu_med"}
rows = []
def run_rules(model, sc, cal_q, test_sets, tag):
    cal = ALL[model].loc[cal_q]
    for ds, tq in test_sets.items():
        tst = ALL[model].loc[tq]
        for r in E.rule_eval(cal[sc].values, cal["err"].values, tst[sc].values, tst["err"].values):
            rows.append({"model": model, "score": sc, "test": ds, **tag, **r})

for model in MODELS:
    for sc in SCORES:
        if ALL[model][sc].isna().any():
            continue
        for s in SEEDS:
            d = SPLITS["seeds"][str(s)]
            run_rules(model, sc, d["cal"], {ds: d[k] for ds, k in TESTS.items()}, {"split": f"seed{s}", "n_cal": 1000})
rng = np.random.default_rng(2026)
BIO = SPLITS["seeds"]["0"]["test_bioasq"]; MED = SPLITS["seeds"]["0"]["test_mmlu_med"]
for b in range(100):
    perm = rng.permutation(POOL)
    for model in MODELS:
        for sc in SCORES:
            if ALL[model][sc].isna().any():
                continue
            run_rules(model, sc, list(perm[:1000]), {"triviaqa_pool": list(perm[1000:2000]), "bioasq": BIO, "mmlu_med": MED},
                      {"split": f"resplit{b}", "n_cal": 1000})
res = pd.DataFrame(rows)
res.to_csv("results/metrics/task3_results.csv", index=False)
print(len(res), "rows")'''),
("md", "## 7. Evaluation – violation frequency, realised error, coverage"),
("code", r'''from dsa1.metrics import mean_std
main = res[(res["model"] == "qwen3-4b") & (res["score"] == "u_tokprob")]
viol = main.groupby(["test", "rule", "target"]).agg(
    violation_freq=("violated", "mean"), realised=("error_among_answered", "mean"),
    coverage=("coverage", "mean"), answered_and_wrong=("answered_and_wrong_rate", "mean"),
    abstain_all=("n_answered", lambda v: float((v == 0).mean())), n_answered=("n_answered", "mean"))
# sampling noise of the realised error on a finite test set: sqrt(a(1-a)/n_answered)
viol["test_noise_sd"] = np.sqrt(viol.index.get_level_values("target") * (1 - viol.index.get_level_values("target")) / viol["n_answered"].clip(lower=1))
viol.round(3)'''),
("code", r'''# Compute cost per answered question (GPU seconds), by model and score, 10% target, COIN, in domain
rows = []
for model in MODELS:
    for sc in SCORES:
        c = E.cost_per_question(ALL[model].reset_index(), sc).mean()
        sub = res[(res["model"] == model) & (res["score"] == sc) & (res["rule"] == "coin") & (res["target"] == 0.10)]
        for ds, g in sub.groupby("test"):
            rows.append({"model": model, "score": sc, "test": ds, "gpu_s_per_question": c,
                         "coverage": g["coverage"].mean(), "gpu_s_per_answered": c / max(g["coverage"].mean(), 1e-9)})
cost = pd.DataFrame(rows); cost.to_csv("results/metrics/task3_cost.csv", index=False); cost.round(3)'''),
("md", "## 8. Comparison & analysis"),
("code", r'''# Key figure – risk-coverage curves with each rule's threshold marked (Qwen3-4B, token probability, seed 0, target 10%)
from dsa1.metrics import risk_coverage_curve, aurc
from dsa1.rules import naive_threshold, crc_threshold, coin_threshold, evaluate_threshold
d = SPLITS["seeds"]["0"]; cal = ALL["qwen3-4b"].loc[d["cal"]]
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (ds, key) in zip(axes, TESTS.items()):
    tst = ALL["qwen3-4b"].loc[d[key]]
    cov, risk = risk_coverage_curve(tst["u_tokprob"], tst["err"])
    ax.plot(cov, risk, c="k", lw=1, label=f"risk-coverage (AURC {aurc(tst['u_tokprob'], tst['err']):.3f})")
    for name, fn in [("naive", naive_threshold), ("crc", crc_threshold), ("coin", coin_threshold)]:
        ev = evaluate_threshold(fn(cal["u_tokprob"].values, cal["err"].values, 0.10), tst["u_tokprob"], tst["err"])
        if ev["n_answered"]:
            ax.scatter(ev["coverage"], ev["error_among_answered"], s=80, label=name, zorder=3)
    ax.axhline(0.10, ls="--", c="grey"); ax.set_title(f"Qwen3-4B – {ds}"); ax.set_xlabel("coverage"); ax.set_ylabel("error among answered")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("results/figures/task3_risk_coverage.png", dpi=150); plt.show()'''),
("code", r'''# Realised error versus target per rule (points = mean over 105 splits, bars = std)
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, ds in zip(axes, TESTS):
    g = main[main["test"] == ds].groupby(["rule", "target"])["error_among_answered"].agg(["mean", "std"]).reset_index()
    for i, (r, gr) in enumerate(g.groupby("rule")):
        ax.errorbar(gr["target"] + (i - 1) * 0.004, gr["mean"], yerr=gr["std"], fmt="o", capsize=3, label=r)
    ax.plot([0, 0.25], [0, 0.25], ":", c="grey"); ax.set_title(ds); ax.set_xlabel("target"); ax.legend()
axes[0].set_ylabel("realised error among answered")
plt.tight_layout(); plt.savefig("results/figures/task3_realised_vs_target.png", dpi=150); plt.show()'''),
("code", r'''# Deeper analysis 1 – model size: coverage at each target vs cost (COIN, token probability, in domain and BioASQ)
g = res[(res["score"] == "u_tokprob") & (res["rule"] == "coin")].groupby(["model", "test", "target"])["coverage"].mean().reset_index()
g = g.merge(cost[cost["score"] == "u_tokprob"][["model", "test", "gpu_s_per_question"]], on=["model", "test"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, ds in zip(axes, ["triviaqa_pool", "bioasq"]):
    for tgt, gt in g[g["test"] == ds].groupby("target"):
        ax.plot(gt["gpu_s_per_question"], gt["coverage"], marker="o", label=f"target {tgt:.0%}")
        for _, r in gt.iterrows():
            ax.annotate(r["model"].replace("qwen3-", ""), (r["gpu_s_per_question"], r["coverage"]), fontsize=8)
    ax.set_title(ds); ax.set_xlabel("GPU s / question"); ax.set_ylabel("coverage with COIN"); ax.legend()
plt.tight_layout(); plt.savefig("results/figures/task3_model_size.png", dpi=150); plt.show()'''),
("code", r'''# Deeper analysis 2 – calibration size: violation frequency per rule for n_cal = 250, 500, 1000 (100 re-splits, Qwen3-4B)
rows = []
rng = np.random.default_rng(7)
for n_cal in [250, 500, 1000]:
    for b in range(100):
        perm = rng.permutation(POOL)
        cal = ALL["qwen3-4b"].loc[list(perm[:n_cal])]; tst = ALL["qwen3-4b"].loc[list(perm[1000:2000])]
        for r in E.rule_eval(cal["u_tokprob"].values, cal["err"].values, tst["u_tokprob"].values, tst["err"].values):
            rows.append({"n_cal": n_cal, **r})
cs = pd.DataFrame(rows)
cs.to_csv("results/metrics/task3_calibration_size.csv", index=False)
p = cs.pivot_table(index="n_cal", columns=["target", "rule"], values="violated", aggfunc="mean")
display(p.round(3))
p.xs(0.10, axis=1, level="target").plot(marker="o"); plt.ylabel("violation frequency (target 10%)")
plt.title("Calibration size vs reliability"); plt.savefig("results/figures/task3_calibration_size.png", dpi=150); plt.show()'''),
("code", r'''# Deeper analysis 3 – precision shift: thresholds from the float16 cache applied to 8-bit and 4-bit answers
from dsa1.cache_io import cache_dir, load_records
from dsa1.grading import normalize_answer
from dsa1.rules import naive_threshold, crc_threshold, coin_threshold
sub = json.load(open("data/splits/precision_subset.json"))
cal = ALL["qwen3-4b"].loc[SPLITS["seeds"]["0"]["cal"]]
rows = []
for ds in ["triviaqa_pool", "bioasq"]:
    ref_rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", ds)).set_index("qid")
    for prec in ["fp16", "8bit", "4bit"]:
        try:
            if prec == "fp16":
                q = ALL["qwen3-4b"].loc[sub[ds]].reset_index(); changed = 0.0
            else:
                q = E.base_scores(CACHE_ROOT, "qwen3-4b", ds, precision=prec, suffix="_subset")
                rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", ds, prec).parent / f"{ds}_subset").set_index("qid")
                changed = float(np.mean([normalize_answer(rec.loc[k, "answer"]) != normalize_answer(ref_rec.loc[k, "answer"]) for k in rec.index]))
        except FileNotFoundError:
            print("missing", prec, ds); continue
        for name, fn in [("naive", naive_threshold), ("crc", crc_threshold), ("coin", coin_threshold)]:
            for a in C.TARGETS:
                ev = evaluate_threshold(fn(cal["u_tokprob"].values, cal["err"].values, a), q["u_tokprob"], q["err"])
                rows.append({"dataset": ds, "precision": prec, "rule": name, "target": a, "accuracy": 1 - q["err"].mean(),
                             "answers_changed": changed, **ev})
prec_tab = pd.DataFrame(rows); prec_tab.to_csv("results/metrics/task3_precision_shift.csv", index=False)
prec_tab.pivot_table(index=["dataset", "precision"], columns=["rule", "target"], values="error_among_answered").round(3)'''),
("md", "### Confusion matrix and demo (for the defense)"),
("code", r'''# Confusion matrix: answered/abstained × right/wrong per rule, target 10%, seed 0, in domain and on BioASQ
from dsa1.rules import naive_threshold, crc_threshold, coin_threshold
d0 = SPLITS["seeds"]["0"]; cal = ALL["qwen3-4b"].loc[d0["cal"]]
RULE_FNS = {"naive": naive_threshold, "crc": crc_threshold, "coin": coin_threshold}
thresholds = {r: fn(cal["u_tokprob"].values, cal["err"].values, 0.10) for r, fn in RULE_FNS.items()}
rows = []
for ds, key in [("triviaqa_pool", "test_triviaqa"), ("bioasq", "test_bioasq")]:
    tst = ALL["qwen3-4b"].loc[d0[key]]
    panels = []
    for r, t in thresholds.items():
        c = E.abstention_confusion(tst["u_tokprob"], tst["err"], t)
        panels.append((r, c)); rows.append({"test": ds, "rule": r, "threshold": t, **c})
    E.plot_confusions(panels, "abstention", f"results/figures/task3_confusion_{ds}.png", f"Qwen3-4B · {ds} · target 10% · seed 0")
    plt.show()
pd.DataFrame(rows).to_csv("results/metrics/task3_confusion.csv", index=False)
pd.DataFrame(rows)'''),
("code", r'''# Demo – inference on test samples: five BioASQ questions, the model's answer, and each rule's decision at 10%
from dsa1.cache_io import cache_dir, load_records
pd.set_option("display.max_colwidth", 70)
rec = load_records(cache_dir(CACHE_ROOT, "qwen3-4b", "bioasq")).set_index("qid")
tst = ALL["qwen3-4b"].loc[d0["test_bioasq"]]
pick = tst.sample(5, random_state=3)
demo = pd.DataFrame({
    "question": [rec.loc[q, "question"] for q in pick.index],
    "model answer": [rec.loc[q, "answer"] for q in pick.index],
    "correct": (pick["err"] == 0).values,
    "uncertainty": pick["u_tokprob"].round(3).values,
    **{r: np.where(pick["u_tokprob"] <= t, "answer", "route to staff") for r, t in thresholds.items()},
})
demo.to_csv("results/metrics/task3_demo_examples.csv", index=False)
demo'''),
("md", """## 9. Discussion
- How often does each rule violate its target, in domain and under shift? Is COIN's violation rate near δ in domain?
- What does conformal risk control guarantee here, and why does it differ from error among answered?
- Coverage cost of COIN's guarantee; infeasible targets on BioASQ.
- Model size and calibration size: what is worth paying for?
- Precision shift: do 8-bit and 4-bit answers keep the float16 threshold's promise?
- MVP: which rule routes questions to staff, at which target."""),
("md", REFS),
]

# =============================================================================== final comparison
FINAL = [
("md", """# DSA1 – Final cross-task comparison (all members)
Reads Task 2's result table (attach Aditya's notebook output) and answers the research question:
does a better score or a shift correction reduce the gap more? Target 10%, Qwen3-4B."""),
("code", SETUP),
("code", r'''hits = list(Path("/kaggle/input").rglob("task2_results.csv")) if ON_KAGGLE else []
path = hits[0] if hits else Path("results/metrics/task2_results.csv")
res = pd.read_csv(path)
from dsa1.metrics import mean_std
for test_set in ["mix_100", "mmlu_all"]:
    d = res[(res["model"] == "qwen3-4b") & (res["target"] == 0.10) & (res["test_set"] == test_set)]
    print(test_set, "– gap (realised − target) | coverage")
    display(d.pivot_table(index="score", columns="method", values="gap", aggfunc=mean_std))
    display(d.pivot_table(index="score", columns="method", values="coverage", aggfunc=mean_std))'''),
("md", """Read the table: down the `uniform` column = effect of a better score alone; along the `u_tokprob` row = effect of
a shift correction alone; bottom-right = both. Pick the MVP configuration (smallest gap at acceptable coverage and cost)."""),
]

OUT.mkdir(exist_ok=True)
for name, cells in [("00_build_cache.ipynb", CACHE),
                    ("SaiAkhilSingade_Uncertainty_Scores.ipynb", T1),
                    ("AdityaArora_Domain_Shift_Correction.ipynb", T2),
                    ("ShivamVivekRai_Threshold_Rule_Reliability.ipynb", T3),
                    ("DSA1_Final_Comparison.ipynb", FINAL)]:
    nbf.write(nb(cells), OUT / name)
    print("wrote", OUT / name)
