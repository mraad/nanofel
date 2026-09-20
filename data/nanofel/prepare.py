"""FELN.json -> train.bin / val.bin for nanoGPT (GPT-2 BPE).

Each record has `text` and `source_text` (two phrasings) that map to one `meta`.
Both phrasings become training pairs; the split is by record so no meta leaks
into validation. If `extra.jsonl` exists next to this file (generated
`{text, meta}` lines), it is normalized to gold style and added to train only,
minus BETWEEN clauses, any record whose meta appears in the gold validation
split, and any record filtering on a (layer, column) gold never uses. Format per pair:

    Q: <question>\nA: <compact json meta><|endoftext|>
"""
import json
import os
import random
import re
import sys

import numpy as np
import tiktoken

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize import normalize, Unparseable  # noqa: E402

HERE = os.path.dirname(__file__)
VAL_FRAC = 0.1
COL = re.compile(r"(\w+) (?:=|<>|>=|<=|>|<|LIKE|ILIKE|BETWEEN)")


def fmt(question, meta=None):
    s = f"Q: {question}\nA:"
    if meta is not None:
        s += " " + json.dumps(meta, ensure_ascii=False, separators=(",", ":"))
    return s


if __name__ == "__main__":
    recs = json.load(open(os.path.join(HERE, "FELN.json"), encoding="utf-8"))
    random.Random(1337).shuffle(recs)
    n_val = int(len(recs) * VAL_FRAC)
    val, train = recs[:n_val], recs[n_val:]

    pairs = [(q, r["meta"]) for r in train for q in (r["text"], r["source_text"])]
    extra_path = os.path.join(HERE, "extra.jsonl")
    if os.path.exists(extra_path):
        val_metas = {json.dumps(r["meta"], sort_keys=True) for r in val}
        gold_cols = {(l, c) for r in recs for l, w in zip(r["meta"]["layers"], r["meta"]["where"]) for c in COL.findall(w)}
        kept = dropped = 0
        for line in open(extra_path, encoding="utf-8"):
            r = json.loads(line)
            m = r["meta"]
            if (json.dumps(m, sort_keys=True) in val_metas or any("BETWEEN" in w for w in m["where"])
                    or any((l, c) not in gold_cols for l, w in zip(m["layers"], m["where"]) for c in COL.findall(w))):
                dropped += 1; continue  # column the catalog no longer has (e.g. medium, paly_slides) shows up as absent from gold
            try:
                r["meta"]["where"] = [normalize(w) for w in r["meta"]["where"]]
            except Unparseable:
                dropped += 1; continue
            pairs.append((r["text"], r["meta"])); kept += 1
        print(f"extra: kept {kept}, dropped {dropped}")
    random.Random(1337).shuffle(pairs)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def encode(ps):
        ids, lens = [], []
        for q, meta in ps:
            t = enc.encode_ordinary(fmt(q, meta)) + [eot]
            ids += t
            lens.append(len(t))
        return np.array(ids, dtype=np.uint16), lens

    train_ids, lens = encode(pairs)
    val_ids, _ = encode([(q, r["meta"]) for r in val for q in (r["text"], r["source_text"])])
    train_ids.tofile(os.path.join(HERE, "train.bin"))
    val_ids.tofile(os.path.join(HERE, "val.bin"))
    with open(os.path.join(HERE, "val.jsonl"), "w", encoding="utf-8") as f:
        for r in val:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train: {len(pairs)} pairs, {len(train_ids):,} tokens; val: {len(val)} gold recs, {len(val_ids):,} tokens")
    print(f"pair tokens: max {max(lens)}, mean {sum(lens)/len(lens):.0f}")
