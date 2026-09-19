"""Nano FEL: natural-language query -> structured {layers, where, relations}.

    python nanofel.py "Give me all oil discoveries."
    python nanofel.py --eval data/nanofel/val.jsonl      # exact-match on held-out set
    echo "..." | python nanofel.py                         # one query per stdin line
"""
import argparse
import json
import os
import sys

import tiktoken
import torch

from model import GPT, GPTConfig
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "nanofel"))
from prepare import fmt  # noqa: E402
from normalize import normalize, Unparseable  # noqa: E402


def load(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    return model.eval().to(device)


@torch.no_grad()
def predict(model, enc, question, device, max_new=192):
    idx = torch.tensor([enc.encode_ordinary(fmt(question))], device=device)
    n0 = idx.shape[1]
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.config.block_size:])
        nxt = logits[:, -1, :].argmax(-1, keepdim=True)  # greedy: structured output wants determinism
        if nxt.item() == enc.eot_token:
            break
        idx = torch.cat([idx, nxt], 1)
    out = enc.decode(idx[0, n0:].tolist()).strip()
    try:
        return json.loads(out), out
    except json.JSONDecodeError:
        return None, out


def same_query(pred, want):
    """True when pred differs from want only by clause order / parenthesization."""
    if not pred or pred.get("layers") != want["layers"] or pred.get("relations") != want["relations"]:
        return False
    try:
        return [normalize(w) for w in pred["where"]] == [normalize(w) for w in want["where"]]
    except (Unparseable, TypeError, KeyError):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--ckpt", default="out-nanofel/ckpt.pt")
    ap.add_argument("--eval", help="val.jsonl path; prints exact-match accuracy")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    a = ap.parse_args()

    enc = tiktoken.get_encoding("gpt2")
    model = load(a.ckpt, a.device)

    if a.eval:
        recs = [json.loads(l) for l in open(a.eval, encoding="utf-8")]
        hit = norm_hit = tot = 0
        for r in recs:
            for q in (r["text"], r["source_text"]):
                pred, raw = predict(model, enc, q, a.device)
                tot += 1
                if pred == r["meta"]:
                    hit += 1
                    norm_hit += 1
                else:
                    print(f"MISS: {q}\n  want {json.dumps(r['meta'], ensure_ascii=False)}\n  got  {raw}")
                    norm_hit += same_query(pred, r["meta"])
        print(f"exact match: {hit}/{tot} = {hit/tot:.1%}   style-normalized: {norm_hit}/{tot} = {norm_hit/tot:.1%}")
        return

    queries = [a.query] if a.query else [l.strip() for l in sys.stdin if l.strip()]
    for q in queries:
        pred, raw = predict(model, enc, q, a.device)
        print(json.dumps(pred, ensure_ascii=False) if pred is not None else f"INVALID JSON: {raw}")


if __name__ == "__main__":
    main()
