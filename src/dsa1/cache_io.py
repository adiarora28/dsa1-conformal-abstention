"""Read the shared answer cache. Task notebooks only read; they never regenerate answers.

Layout (one folder per model, precision and dataset):
    <cache_root>/<model>_<precision>/<dataset>/records_00000.parquet
    <cache_root>/<model>_<precision>/<dataset>/hidden_00000.npz
    <cache_root>/<model>_<precision>/<dataset>/run_config.json
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C


def cache_dir(cache_root: str | Path, model_key: str, dataset: str, precision: str | None = None) -> Path:
    precision = precision or C.DEFAULT_PRECISION[model_key]
    return Path(cache_root) / f"{model_key}_{precision}" / dataset


def load_records(path: str | Path) -> pd.DataFrame:
    files = sorted(Path(path).glob("records_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no records in {path}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.drop_duplicates("qid", keep="last").reset_index(drop=True)
    list_cols = ["aliases", "answer_token_ids", "token_logprobs", "samples", "sample_logprob_sum",
                 "sample_n_tokens", "sample_correct_em", "choices", "option_probs"]
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: list(x) if x is not None else [])
    return df


def load_hidden(path: str | Path, position: str = "answer", layer: int = 1,
                qids: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Hidden-state matrix for one position ('prompt' or 'answer') and one layer slot (0, 1 or 2).

    Layer slots follow config.HIDDEN_LAYER_FRACTIONS (0 = early, 1 = middle, 2 = last).
    Returns (qids, X) with X as float32; if `qids` is given, rows follow that order.
    """
    key = {"prompt": "h_prompt", "answer": "h_answer"}[position]
    all_q, all_x = [], []
    for f in sorted(Path(path).glob("hidden_*.npz")):
        z = np.load(f, allow_pickle=False)
        all_q.append(z["qids"].astype(str))
        all_x.append(z[key][:, layer, :])
    q = np.concatenate(all_q)
    X = np.concatenate(all_x).astype(np.float32)
    _, first = np.unique(q[::-1], return_index=True)  # keep the last write of a qid
    keep = np.sort(len(q) - 1 - first)
    q, X = q[keep], X[keep]
    if qids is not None:
        pos = {k: i for i, k in enumerate(q)}
        missing = [k for k in qids if k not in pos]
        if missing:
            raise KeyError(f"{len(missing)} qids missing from hidden states, e.g. {missing[:3]}")
        idx = np.array([pos[k] for k in qids])
        q, X = q[idx], X[idx]
    return q, X


def subset(df: pd.DataFrame, qids: list[str]) -> pd.DataFrame:
    """Rows of a records table in the order of `qids` (raises if any are missing)."""
    out = df.set_index("qid").reindex(qids)
    if out.index.isna().any() or out.isna().all(axis=1).any():
        missing = out[out.isna().all(axis=1)].index.tolist()
        raise KeyError(f"{len(missing)} qids missing from records, e.g. {missing[:3]}")
    return out.reset_index()


def link_cache_roots(search_root: str | Path = "/kaggle/input",
                     target: str | Path = "/kaggle/working/cache_merged") -> Path:
    """Merge cache folders from several attached Kaggle datasets into one root via symlinks.

    Each member publishes their own part (for example dsa1-cache-triviaqa, dsa1-cache-bioasq,
    dsa1-cache-mmlu). This finds every <model>_<precision>/<dataset>/ folder that holds records and
    links it under `target`, so all notebooks use one CACHE_ROOT. If a folder appears twice, the copy
    with more record shards wins.
    """
    target = Path(target)
    best: dict[tuple[str, str], Path] = {}
    for rec in Path(search_root).rglob("records_00000.parquet"):
        ds_dir = rec.parent
        model_dir = ds_dir.parent.name
        if not model_dir.startswith("qwen3-"):
            continue
        key = (model_dir, ds_dir.name)
        n = len(list(ds_dir.glob("records_*.parquet")))
        if key not in best or n > len(list(best[key].glob("records_*.parquet"))):
            best[key] = ds_dir
    for (model_dir, ds), src in best.items():
        dst = target / model_dir / ds
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink() or dst.exists():
            dst.unlink() if dst.is_symlink() else None
        if not dst.exists():
            dst.symlink_to(src, target_is_directory=True)
    return target
