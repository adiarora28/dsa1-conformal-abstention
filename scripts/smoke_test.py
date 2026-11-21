"""Five-minute GPU check before any long run: loads a model and answers a few questions of each type.

Usage:
    python scripts/smoke_test.py --model qwen3-1.7b
"""
import argparse

import torch

import _path  # noqa: F401
from dsa1 import config as C
from dsa1.generate import generate_batch, load_model
from dsa1.prompts import letter_token_ids, open_messages, render

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen3-1.7b", choices=list(C.MODELS))
ap.add_argument("--precision", default=None)
args = ap.parse_args()

print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
model, tok, precision = load_model(args.model, args.precision)
print("loaded", args.model, precision, "layers:", model.config.num_hidden_layers)
qs = ["Which planet is known as the Red Planet?", "What gene is mutated in cystic fibrosis?"]
prompts = [render(tok, open_messages(q)) for q in qs]
if "<think>\n\n</think>" not in prompts[0]:
    print("WARNING: empty think block not found - check that enable_thinking=False reaches the chat template")
for q, g in zip(qs, generate_batch(model, tok, prompts, do_sample=False)):
    print(f"Q: {q}\nA: {g[0]['text']!r}  ({len(g[0]['ids'])} tokens, mean logprob "
          f"{sum(g[0]['logprobs']) / max(1, len(g[0]['logprobs'])):.3f})")
s = generate_batch(model, tok, prompts[:1], do_sample=True, n=5)[0]
print("5 samples:", [x["text"] for x in s])
print("letter token ids:", letter_token_ids(tok))
print("GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else "-")
