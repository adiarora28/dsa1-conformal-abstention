"""Generate the shared answer cache on a GPU (guide Steps 0.3, 0.4 and 0.6).

For every open-ended question (TriviaQA, BioASQ) the cache stores:
  greedy answer, its token ids and per-token log-probabilities (from the raw logits),
  N sampled answers with their summed log-probabilities, correctness labels,
  hidden states at the last prompt token and the last answer token for three layers,
  and wall-clock timings for each stage.
For MMLU it stores the four option probabilities, the predicted letter, labels, hidden states at the
last prompt token and at the predicted letter, and timings.

Results are written in shards of `flush_every` questions; a rerun skips questions already saved, so
a disconnected Kaggle session resumes where it stopped.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from . import config as C
from .grading import clean_generation, grade
from .prompts import letter_token_ids, mcq_messages, open_messages, render


# ----------------------------------------------------------------------------- model loading
def load_model(model_key: str, precision: str | None = None, device_map: str = "auto"):
    """Load a Qwen3 model and tokenizer. precision: fp16 | bf16 | fp32 | 8bit | 4bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    name = C.MODELS[model_key]
    precision = precision or C.DEFAULT_PRECISION[model_key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw: dict = {"device_map": device_map}
    if precision in ("fp16", "bf16", "fp32"):
        kw["dtype"] = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    elif precision == "8bit":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "4bit":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
        )
    else:
        raise ValueError(f"unknown precision {precision}")
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok, precision


def _eos_ids(model, tok) -> set[int]:
    ids = set()
    gc_eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(gc_eos, int):
        ids.add(gc_eos)
    elif gc_eos:
        ids.update(gc_eos)
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    if tok.pad_token_id is not None:
        ids.add(tok.pad_token_id)
    return ids


def layer_indices(model) -> list[int]:
    """Indices into `hidden_states` (0 = embeddings) for the early, middle and last layer."""
    n = model.config.num_hidden_layers
    return [max(1, min(n, int(round(f * n)))) for f in C.HIDDEN_LAYER_FRACTIONS]


# ----------------------------------------------------------------------------- generation
@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], *, do_sample: bool, n: int = 1,
                   max_new_tokens: int = C.MAX_NEW_TOKENS, temperature: float = C.SAMPLE_TEMPERATURE,
                   top_p: float = C.SAMPLE_TOP_P) -> list[list[dict]]:
    """Return, per prompt, a list of n generations: {ids, logprobs, text}.

    Log-probabilities come from the raw logits (output_logits=True), so they are the model's own
    probabilities, not the top-p-filtered sampling distribution.
    """
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kw = dict(max_new_tokens=max_new_tokens, return_dict_in_generate=True, output_logits=True,
              pad_token_id=tok.pad_token_id)
    if do_sample:
        kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0, num_return_sequences=n)
    else:
        kw.update(do_sample=False, temperature=None, top_p=None, top_k=None, num_return_sequences=1)
        n = 1
    out = model.generate(**enc, **kw)
    prompt_len = enc["input_ids"].shape[1]
    seqs = out.sequences[:, prompt_len:]
    steps = []
    for t, lg in enumerate(out.logits):
        lp = torch.log_softmax(lg.float(), dim=-1)
        steps.append(lp.gather(1, seqs[:, t:t + 1]).squeeze(1))
    logprobs = torch.stack(steps, dim=1).cpu().numpy() if steps else np.zeros((seqs.shape[0], 0))
    seqs = seqs.cpu().numpy()
    if not np.isfinite(logprobs).all():
        raise FloatingPointError("non-finite log-probabilities: try precision='fp32' or 'bf16'")
    eos = _eos_ids(model, tok)
    results: list[list[dict]] = [[] for _ in prompts]
    for row in range(seqs.shape[0]):
        ids, lps = [], []
        for tid, lp in zip(seqs[row].tolist(), logprobs[row].tolist()):
            if tid in eos:
                break
            ids.append(int(tid))
            lps.append(float(lp))
        text = tok.decode(ids, skip_special_tokens=True)
        results[row // n].append({"ids": ids, "logprobs": lps, "text": text})
    return results


@torch.no_grad()
def hidden_states_batch(model, tok, prompt_ids: list[list[int]], answer_ids: list[list[int]],
                        layers: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Hidden states at the last prompt token and the last answer token, shape (B, n_layers, D).

    Sequences are built from token ids (prompt ids + generated ids), not re-tokenised text, so the
    positions are exact.
    """
    seqs = [p + a for p, a in zip(prompt_ids, answer_ids)]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    input_ids = torch.full((len(seqs), L), pad, dtype=torch.long)
    attn = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, L - len(s):] = torch.tensor(s)
        attn[i, L - len(s):] = 1
    out = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device),
                output_hidden_states=True)
    hs = out.hidden_states
    last_answer = L - 1
    h_prompt, h_answer = [], []
    for i, a in enumerate(answer_ids):
        last_prompt = L - 1 - len(a)
        h_prompt.append(torch.stack([hs[l][i, last_prompt] for l in layers]))
        h_answer.append(torch.stack([hs[l][i, last_answer] for l in layers]))
    to_np = lambda xs: torch.stack(xs).float().cpu().numpy().astype(np.float16)
    return to_np(h_prompt), to_np(h_answer)


# ----------------------------------------------------------------------------- shards
def _done_qids(out_dir: Path) -> set[str]:
    done: set[str] = set()
    for f in sorted(out_dir.glob("records_*.parquet")):
        done.update(pd.read_parquet(f, columns=["qid"])["qid"].tolist())
    return done


def _write_shard(out_dir: Path, records: list[dict], h_prompt: list, h_answer: list) -> None:
    k = len(list(out_dir.glob("records_*.parquet")))
    pd.DataFrame(records).to_parquet(out_dir / f"records_{k:05d}.parquet", index=False)
    np.savez_compressed(out_dir / f"hidden_{k:05d}.npz",
                        qids=np.array([r["qid"] for r in records]),
                        h_prompt=np.stack(h_prompt), h_answer=np.stack(h_answer))


def write_run_config(out_dir: Path, **extra) -> None:
    import transformers

    cfg = {
        "max_new_tokens": C.MAX_NEW_TOKENS, "n_samples": C.N_SAMPLES,
        "temperature": C.SAMPLE_TEMPERATURE, "top_p": C.SAMPLE_TOP_P,
        "hidden_layer_fractions": C.HIDDEN_LAYER_FRACTIONS,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "python": platform.python_version(),
    }
    cfg.update(extra)
    (out_dir / "run_config.json").write_text(json.dumps(cfg, indent=1, default=str))


# ----------------------------------------------------------------------------- runners
def run_open(df: pd.DataFrame, model, tok, out_dir: str | Path, *, model_key: str, precision: str,
             n_samples: int = C.N_SAMPLES, batch_size: int = 8, sample_batch_size: int = 4,
             flush_every: int = 100, limit: int | None = None) -> Path:
    """Greedy answer (+ samples) + hidden states for an open-ended question table."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, model=model_key, precision=precision, n_samples_run=n_samples)
    layers = layer_indices(model)
    done = _done_qids(out_dir)
    todo = df[~df["qid"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    records, hp_buf, ha_buf = [], [], []
    for start in tqdm(range(0, len(todo), batch_size), desc=f"{model_key}:{out_dir.name}"):
        chunk = todo.iloc[start:start + batch_size]
        prompts = [render(tok, open_messages(q)) for q in chunk["question"]]
        prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

        t0 = time.perf_counter()
        greedy = [g[0] for g in generate_batch(model, tok, prompts, do_sample=False)]
        t_greedy = (time.perf_counter() - t0) / len(chunk)

        samples: list[list[dict]] = [[] for _ in prompts]
        t_samples = 0.0
        if n_samples:
            t0 = time.perf_counter()
            for s in range(0, len(prompts), sample_batch_size):
                part = generate_batch(model, tok, prompts[s:s + sample_batch_size], do_sample=True, n=n_samples)
                for j, gens in enumerate(part):
                    samples[s + j] = gens
            t_samples = (time.perf_counter() - t0) / len(chunk)

        t0 = time.perf_counter()
        hp, ha = hidden_states_batch(model, tok, prompt_ids, [g["ids"] for g in greedy], layers)
        t_hidden = (time.perf_counter() - t0) / len(chunk)

        for i, (_, row) in enumerate(chunk.iterrows()):
            ans = clean_generation(greedy[i]["text"])
            lab = grade(ans, row["aliases"])
            s_texts = [clean_generation(g["text"]) for g in samples[i]]
            records.append({
                "qid": row["qid"], "dataset": row["dataset"], "question": row["question"],
                "aliases": list(row["aliases"]), "answer_raw": greedy[i]["text"], "answer": ans,
                "answer_token_ids": greedy[i]["ids"], "token_logprobs": greedy[i]["logprobs"],
                "n_answer_tokens": len(greedy[i]["ids"]),
                "correct_em": lab["correct_em"], "f1": lab["f1"], "correct_f1": lab["correct_f1"],
                "samples": s_texts,
                "sample_logprob_sum": [float(np.sum(g["logprobs"])) for g in samples[i]],
                "sample_n_tokens": [len(g["ids"]) for g in samples[i]],
                "sample_correct_em": [grade(t, row["aliases"])["correct_em"] for t in s_texts],
                "t_greedy": t_greedy, "t_samples": t_samples, "t_hidden": t_hidden,
                "model": model_key, "precision": precision,
            })
            hp_buf.append(hp[i])
            ha_buf.append(ha[i])
        if len(records) >= flush_every:
            _write_shard(out_dir, records, hp_buf, ha_buf)
            records, hp_buf, ha_buf = [], [], []
    if records:
        _write_shard(out_dir, records, hp_buf, ha_buf)
    return out_dir


@torch.no_grad()
def run_mcq(df: pd.DataFrame, model, tok, out_dir: str | Path, *, model_key: str, precision: str,
            batch_size: int = 8, flush_every: int = 100, limit: int | None = None) -> Path:
    """Option probabilities + hidden states for an MMLU question table (no sampling)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_run_config(out_dir, model=model_key, precision=precision)
    layers = layer_indices(model)
    letter_ids = letter_token_ids(tok)
    first_letter_id = [ids[0] for ids in letter_ids]
    done = _done_qids(out_dir)
    todo = df[~df["qid"].isin(done)]
    if limit is not None:
        todo = todo.head(limit)
    records, hp_buf, ha_buf = [], [], []
    for start in tqdm(range(0, len(todo), batch_size), desc=f"{model_key}:{out_dir.name}"):
        chunk = todo.iloc[start:start + batch_size]
        prompts = [render(tok, mcq_messages(q, c)) for q, c in zip(chunk["question"], chunk["choices"])]
        prompt_ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]

        t0 = time.perf_counter()
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        logits = model(**enc).logits[:, -1, :].float()
        lse = torch.stack([torch.logsumexp(logits[:, ids], dim=-1) for ids in letter_ids], dim=-1)
        probs = torch.softmax(lse, dim=-1).cpu().numpy()
        pred = probs.argmax(axis=1)
        hp, ha = hidden_states_batch(model, tok, prompt_ids, [[first_letter_id[p]] for p in pred], layers)
        t_forward = (time.perf_counter() - t0) / len(chunk)

        for i, (_, row) in enumerate(chunk.iterrows()):
            records.append({
                "qid": row["qid"], "dataset": row["dataset"], "subject": row["subject"],
                "question": row["question"], "choices": list(row["choices"]),
                "answer_idx": int(row["answer_idx"]), "option_probs": probs[i].tolist(),
                "pred_idx": int(pred[i]), "correct_em": bool(pred[i] == row["answer_idx"]),
                "t_forward": t_forward, "model": model_key, "precision": precision,
            })
            hp_buf.append(hp[i])
            ha_buf.append(ha[i])
        if len(records) >= flush_every:
            _write_shard(out_dir, records, hp_buf, ha_buf)
            records, hp_buf, ha_buf = [], [], []
    if records:
        _write_shard(out_dir, records, hp_buf, ha_buf)
    return out_dir
