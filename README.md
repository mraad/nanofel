# Nano FEL

A GPT-2 fine-tune that turns a natural-language question about North Sea
oil and gas layers into a structured FELN query spec (Find Existing Location
with N layers). Built on [nanoGPT](https://github.com/karpathy/nanoGPT):
`model.py`, `train.py` and `configurator.py` are vendored verbatim from
nanoGPT commit `3adf61e` under its MIT license (see `LICENSE`).

```
Q: Which gas wells in Norway are within 2 kilometers of an oil pipeline?
A: {"layers":["Wells","Pipelines"],
    "where":["content_type = cast(2 as SMALLINT) and (country = 'NO')","PipelinesType = cast(4 as SMALLINT)"],
    "relations":["withinDistance 2 kilometers"]}
```

`layers[0]` is the target layer; `where[i]` is the SQL filter for `layers[i]`
(empty string = no filter); `relations[i]` is the spatial relation between
`layers[0]` and `layers[i+1]`. Column names, coded domains and casting rules
come from the layer catalog docs of the NorthSea ArcGIS project (Wells,
Pipelines, Discoveries; Norwegian Petroleum Directorate and neighbouring
agencies' public data).

## Files

| Path | Purpose |
|---|---|
| `data/nanofel/FELN.json` | 1000 records: `text` (casual phrasing), `source_text` (canonical phrasing), `meta` (target). |
| `data/nanofel/extra.jsonl` | Optional, not tracked: 15,996 generated `{text, meta}` lines from a grammar sampler plus LLM paraphrases (separate, private generator). Train only. |
| `data/nanofel/normalize.py` | Rewrites generated where-clauses into gold surface style (ILIKE columns, subtype-first, parenthesization). `python normalize.py FELN.json` asserts gold is a fixed point (99.4%; the rest is gold's own inconsistency). |
| `data/nanofel/prepare.py` | Shuffles with seed 1337, holds out 100 gold records, emits `train.bin`, `val.bin`, `val.jsonl`. Both phrasings of a record go to the same split. Adds `extra.jsonl` to train after normalizing, dropping BETWEEN clauses and any meta that appears in the gold val split. |
| `config/finetune_nanofel.py` | Training config (GPT-2 BPE, block 256, lr 3e-5, dropout 0.1, 3000 iters, keeps best-val checkpoint). |
| `nanofel.py` | Local inference and eval. Greedy decoding, stops at `<\|endoftext\|>`, parses JSON. Eval prints exact match and a style-normalized match (same query up to clause order and parens). |
| `data/nanofel/val.jsonl` | The 100 held-out gold records, input to `nanofel.py --eval`. |
| `out-nanofel/ckpt.pt` | Shipped v2 weights, optimizer state stripped (gitignored, ~1.4 GB). |
| `out-nanofel/ckpt-v1.pt` | v1 weights (gold only, 79%), kept locally for comparison. |

Training pair format is `Q: <question>\nA: <compact json><|endoftext|>`, one
after another in a flat token stream, so nanoGPT's random-window sampler needs
no changes.

## Train (remote GPU box, 2x RTX PRO 6000 Blackwell)

```bash
rsync -az --exclude .git --exclude .venv --exclude out-nanofel ./ GPU_HOST:~/nanofel/
ssh GPU_HOST
cd ~/nanofel
uv venv --python 3.12 .venv && uv pip install -r requirements.txt   # torch cu130 works on Blackwell
# optional: drop generated {text, meta} lines into data/nanofel/extra.jsonl (+15.6k pairs)
.venv/bin/python data/nanofel/prepare.py
nohup .venv/bin/torchrun --standalone --nproc_per_node=2 train.py config/finetune_nanofel.py > train.log 2>&1 &
```

Any `{"text": ..., "meta": {...}}` JSONL works as extra data as long as the
meta uses the same layers and columns; `normalize.py` takes care of surface
style.

Override the base model or output dir from the command line, e.g.
`--init_from=gpt2-large --out_dir=out-nanofel-large`.

About 82 ms/iter for gpt2-medium; 3000 iters is under ten minutes including
checkpoint writes. Strip the optimizer before copying home:

```bash
.venv/bin/python -c "
import torch
c = torch.load('out-nanofel/ckpt.pt', map_location='cpu', weights_only=False)
torch.save({k: c[k] for k in ('model','model_args','iter_num','best_val_loss','config')}, 'out-nanofel/nanofel.pt')"
```

```bash
scp GPU_HOST:~/nanofel/out-nanofel/nanofel.pt out-nanofel/ckpt.pt
```

## Run (local, Mac MPS)

```bash
uv venv --python 3.12 .venv && uv pip install torch numpy tiktoken
.venv/bin/python nanofel.py "Give me all oil discoveries."
echo "wells within 5 feet of a condensate pipeline" | .venv/bin/python nanofel.py
.venv/bin/python nanofel.py --eval data/nanofel/val.jsonl      # prints every miss, then accuracy
```

Flags: `--ckpt` (default `out-nanofel/ckpt.pt`), `--device` (auto: mps > cuda > cpu).
Output is one JSON object per line, or `INVALID JSON: ...` if the model emits
something unparsable.

## Results

Exact match on the 100 held-out gold records, both phrasings each (200
queries). Same checkpoint gives the same numbers on the GPU box (cuda) and a Mac (mps).

| Run | Train pairs | Base model | Best val loss (step) | Exact match |
|---|---|---|---|---|
| v1 | 1,800 (gold only) | gpt2-medium | 0.308 (400) | 158/200 = 79.0% |
| v1 | 1,800 (gold only) | gpt2-large | 0.311 (200) | 158/200 = 79.0% |
| **v2 (shipped)** | 17,372 (gold + extra) | gpt2-medium | 0.387 (2600) | **194/200 = 97.0%** |

Val loss is token-level over question + answer and is not comparable across
runs with different question styles; exact match is the metric to trust.
The gpt2-large result shows v1 was data-limited, not capacity-limited.

### Where v1 missed (42 of 200)

| Cause | Queries | Example |
|---|---|---|
| ILIKE wildcard placement: "ends with" as `'%x%'` or `'x%'` instead of `'%x'` | 12 | `production_licence ILIKE '001%'` for "ending in '001'" |
| Invented a filter on an unfiltered target layer | 7 | "Show me wells ..." got `content_type = cast(1 as SMALLINT)` |
| Dropped or garbled a second clause | 7 | "in Denmark" lost `country = 'DK'` |
| Copied a literal wrong (name, year off by one) | 7 | `'%Munin%'` for 'Midgard'; `1972` for "after 1970" |
| AND-clause order or parentheses | 5 | `(b) and (a)` for gold `(a) and (b)` |
| Wrong subtype code | 4 | `content_type = 7` for 'gas/condensate shows' (11) |

Every category traces to sparse coverage in 900 gold records: 37 "ends with"
examples, 22 unfiltered targets, 43 numeric casts, 17 dates. The same model
size at the same data with gpt2-large did not move, so it was data, not
capacity.

### What v2 changed

The extra data has 1,488 unfiltered targets, 1,297 numeric casts, 927 dates
and several paraphrases per pattern. Normalizing it into gold style mattered
as much as adding it: without `normalize.py` the model would learn two
competing surface forms and lose exact match on gold. The style-normalized
metric equals the exact metric for v2, so none of the remaining misses are
formatting.

Remaining 6 misses in v2:

- "after 1970": emits `entry_date > '1970-01-01'` where gold wants
  `>= '1971-01-01'` (2 queries). Semantically close, different convention.
- Clause order with a leading value set, e.g. `(status = A or status = B) and
  (field_label = X)`: emits the field_label first, or drops the outer parens
  (2 queries).
- Drops the discovery subtype when it co-occurs with a hydrocarbon-type value
  set (2 queries).

## To go past 97%

- Year conventions: gold encodes "after 1970" as `>= '1971-01-01'` and
  "before 2008" as `< '2008-01-01'`. Add a few hundred generated date examples
  in that convention (the extra data uses BETWEEN ranges, which prepare drops).
- Value set plus another clause on one layer (`(a or b) and (c)`) is rare in
  both sets; generate more.
- Loss on answer tokens only: mask the `Q:` tokens in `train.py` so capacity
  is not spent predicting paraphrased questions. Not needed to reach 95%, so
  not done.

## Known limits

- Column and domain coverage is whatever the training data contains. The OKF
  docs were used to understand the targets and to pick the ILIKE columns for
  the normalizer, not fed to the model.
- The model copies unfamiliar column words: "depth > 350 meters" gives
  `depth > cast(350.0 as DOUBLE PRECISION)`; say "water depth" for
  `water_depth`. A downstream column allowlist catches these.
- Greedy decoding, no grammar constraint, no output validation against the
  schema. Add a schema check downstream if invalid columns matter.

## Credits

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy: model,
  training loop and configurator, MIT licensed.
- GPT-2 weights from OpenAI via Hugging Face `transformers`.
