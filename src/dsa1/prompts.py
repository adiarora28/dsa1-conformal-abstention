"""Prompts shared by every model and dataset (guide Step 0.3).

The same short-answer prompt is used for TriviaQA and BioASQ so that only the questions change
between domains. Qwen3's thinking mode is always disabled with enable_thinking=False.
"""
from __future__ import annotations

SYSTEM_OPEN = (
    "You are a question answering assistant. Answer with the shortest possible answer: "
    "a name, a number or a short phrase. Do not explain."
)
SYSTEM_MCQ = "You are a question answering assistant. Answer with the letter of the correct option only."
LETTERS = ["A", "B", "C", "D"]


def open_messages(question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_OPEN},
        {"role": "user", "content": f"Question: {question}\nAnswer:"},
    ]


def mcq_messages(question: str, choices: list[str]) -> list[dict]:
    options = "\n".join(f"{L}. {c}" for L, c in zip(LETTERS, choices))
    return [
        {"role": "system", "content": SYSTEM_MCQ},
        {"role": "user", "content": f"Question: {question}\n{options}\nAnswer:"},
    ]


def render(tokenizer, messages: list[dict]) -> str:
    """Chat-template text ending where the assistant's answer starts, with thinking disabled."""
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def letter_token_ids(tokenizer) -> list[list[int]]:
    """Token ids that can start each option letter (with and without a leading space).

    Option probabilities sum over both variants, so the result does not depend on whether the
    model emits "A" or " A" first.
    """
    out = []
    for L in LETTERS:
        ids = set()
        for variant in (L, " " + L):
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if len(toks) == 1:
                ids.add(toks[0])
        if not ids:
            raise ValueError(f"letter {L} is not a single token for this tokenizer")
        out.append(sorted(ids))
    return out
