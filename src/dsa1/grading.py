"""Answer normalisation and correctness labels (guide Step 0.5).

Main label: normalised exact match against any alias.
Secondary label: token F1 >= 0.5 against the best alias. It is stored separately and never replaces
the main label.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Iterable

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation)


def clean_generation(text: str) -> str:
    """Keep the first non-empty line of a model answer and strip simple wrappers."""
    if text is None:
        return ""
    text = text.replace("<think>", "").replace("</think>", "")
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = lines[0] if lines else ""
    for prefix in ("final answer:", "answer:", "the answer is"):
        if first.lower().startswith(prefix):
            first = first[len(prefix):].strip()
    return first.strip().strip('"').strip("'").rstrip(".").strip()


def normalize_answer(s: str) -> str:
    """SQuAD-style normalisation: lower case, remove punctuation and articles, collapse whitespace."""
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, aliases: Iterable[str]) -> bool:
    pred = normalize_answer(prediction)
    if not pred:
        return False
    return any(pred == normalize_answer(a) for a in aliases if a)


def _f1(pred_tokens: list[str], gold_tokens: list[str]) -> float:
    common = Counter(pred_tokens) & Counter(gold_tokens)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision = same / len(pred_tokens)
    recall = same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: str, aliases: Iterable[str]) -> float:
    pred = normalize_answer(prediction).split()
    if not pred:
        return 0.0
    best = 0.0
    for a in aliases:
        gold = normalize_answer(a).split()
        if gold:
            best = max(best, _f1(pred, gold))
    return best


def grade(prediction: str, aliases: Iterable[str], f1_threshold: float = 0.5) -> dict:
    """Return the main label (exact match) and the secondary token-F1 label for one answer."""
    aliases = list(aliases)
    f1 = token_f1(prediction, aliases)
    return {
        "correct_em": bool(exact_match(prediction, aliases)),
        "f1": float(f1),
        "correct_f1": bool(f1 >= f1_threshold),
    }
