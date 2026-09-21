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
| `data/nanofel/FELN.json` | 3000 records (2026-09-20 regen; was 1000): `text` (casual phrasing), `source_text` (canonical phrasing), `meta` (target). |
| `data/nanofel/extra.jsonl` | Optional, not tracked: 15,996 generated `{text, meta}` lines = `cat feln-dsl/data/{train,val,test_syn}.jsonl` (grammar sampler over the old `Layers.json` plus LLM paraphrases, from the sibling `feln-dsl` project). Train only. |
| `data/nanofel/normalize.py` | Rewrites generated where-clauses into gold surface style (subtype-first, parenthesization). `python normalize.py FELN.json` asserts gold is a fixed point (100% on the 2026-09-20 gold). |
| `data/nanofel/prepare.py` | Shuffles with seed 1337, holds out 10% of gold records (300), emits `train.bin`, `val.bin`, `val.jsonl`. Both phrasings of a record go to the same split. Adds `extra.jsonl` to train after normalizing, dropping BETWEEN clauses, any meta that appears in the gold val split, and any row filtering on a (layer, column) gold never uses (catches columns removed from the catalog). |
| `config/finetune_nanofel.py` | Training config (GPT-2 BPE, block 256, lr 3e-5, dropout 0.1, 3000 iters, keeps best-val checkpoint). |
| `nanofel.py` | Local inference and eval. Greedy decoding, stops at `<\|endoftext\|>`, parses JSON. Eval prints exact match and a style-normalized match (same query up to clause order and parens). |
| `data/nanofel/val.jsonl` | The 300 held-out gold records, input to `nanofel.py --eval`. |
| `out-nanofel/ckpt.pt` | Shipped v4 weights, optimizer state stripped (gitignored, ~1.4 GB). |
| `out-nanofel/ckpt-v3.pt`, `ckpt-v2.pt`, `ckpt-v1.pt` | v3 (new gold only), v2 (old gold + extra, ILIKE era), v1 (old gold only), kept locally for comparison. v5 (longer run, no gain) stays on the GPU box in `out-nanofel-v5/`. |

Training pair format is `Q: <question>\nA: <compact json><|endoftext|>`, one
after another in a flat token stream, so nanoGPT's random-window sampler needs
no changes.

## Train (remote GPU box, RTX PRO 6000 Blackwell)

```bash
rsync -az --exclude .git --exclude .venv --exclude out-nanofel ./ GPU_HOST:~/nanofel/
ssh GPU_HOST
cd ~/nanofel
uv venv --python 3.12 .venv && uv pip install -r requirements.txt   # torch cu130 works on Blackwell
# optional: drop generated {text, meta} lines into data/nanofel/extra.jsonl
.venv/bin/python data/nanofel/prepare.py
# one GPU (v3: gc3 device 0):
CUDA_VISIBLE_DEVICES=0 nohup .venv/bin/python train.py config/finetune_nanofel.py > train.log 2>&1 &
# two GPUs (v1/v2; v4: gc3 devices 0 and 2, device 1 was taken):
CUDA_VISIBLE_DEVICES=0,2 nohup .venv/bin/torchrun --standalone --nproc_per_node=2 train.py config/finetune_nanofel.py > train.log 2>&1 &
# three GPUs (v5): accumulation steps must divide by the world size, so 3 -> 48 seqs / iter
CUDA_VISIBLE_DEVICES=0,1,2 nohup .venv/bin/torchrun --standalone --nproc_per_node=3 train.py config/finetune_nanofel.py \
  --out_dir=out-nanofel-v5 --gradient_accumulation_steps=3 --max_iters=6000 --lr_decay_iters=6000 > train-v5.log 2>&1 &
```

`gradient_accumulation_steps = 2` with `batch_size = 16` gives 32 sequences
(8,192 tokens) per iteration in both cases: nanoGPT divides the accumulation
steps by the world size under DDP, so one process does both micro-steps.
`train.py` asserts `gradient_accumulation_steps % world_size == 0`, hence
the override for three GPUs.

Any `{"text": ..., "meta": {...}}` JSONL works as extra data as long as the
meta uses the same layers and columns; `normalize.py` takes care of surface
style.

Override the base model or output dir from the command line, e.g.
`--init_from=gpt2-large --out_dir=out-nanofel-large`.

About 83 ms/iter on two GPUs, 97 ms/iter on one, for gpt2-medium; 3000
iters is under ten minutes either way including compile and checkpoint writes
(v3: 8 minutes wall on one GPU; v4: 5 minutes on two). Strip the optimizer before copying home:

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

Exact match on the held-out gold records, both phrasings each. Same checkpoint
gives the same numbers on the GPU box (cuda) and a Mac (mps). v1/v2 were
scored on the old 100-record val split; v3 on the new 300-record split
(different gold, see below), so only rows on the same split are directly
comparable.

| Run | Gold | Train pairs | Best val loss (step) | Val split | Exact match |
|---|---|---|---|---|---|
| v1 | old (1,000) | 1,800 (gold only) | 0.308 (400) | old, 200 queries | 158/200 = 79.0% |
| v1, gpt2-large | old (1,000) | 1,800 (gold only) | 0.311 (200) | old, 200 queries | 158/200 = 79.0% |
| v2 | old (1,000) | 17,372 (gold + extra) | 0.387 (2600) | old, 200 queries | 194/200 = 97.0% |
| v2 | old (1,000) | 17,372 (gold + extra) | 0.387 (2600) | new, 600 queries, `ILIKE`→`LIKE` on output | 322/600 = 53.7% |
| v3 | new (3,000) | 5,400 (gold only) | 0.319 (1000) | new, 600 queries | 529/600 = 88.2% |
| **v4 (shipped)** | new (3,000) | 18,297 (gold + extra) | 0.322 (3000) | new, 600 queries | **534/600 = 89.0%** |
| v5 | new (3,000) | 18,297 (gold + extra) | 0.326 (1800) | new, 600 queries | 532/600 = 88.7% |

All runs gpt2-medium unless noted. Val loss is token-level over question +
answer and is not comparable across runs with different question styles;
exact match is the metric to trust. The gpt2-large result shows v1 was
data-limited, not capacity-limited. Style-normalized match equals exact
match for v2, v3, v4 and v5: no remaining miss is formatting.

### The 2026-09-20 data update (v3, v4)

`FELN.json`, `Layers.json` and the OKF docs were regenerated in the NorthSea
project. What changed for this model:

- 3,000 gold records instead of 1,000; only 94 of the old metas survive, one
  old question text. Every record now carries a filter on the target layer
  (old gold had 22 unfiltered targets).
- `LIKE` everywhere. The old catalog hints said `ILIKE` on 11 name columns;
  the new hints say `LIKE`, and gold follows. `normalize.py` no longer
  rewrites `LIKE` to `ILIKE`, and with that the new gold is a 100% fixed
  point (old gold: 99.4%). A v2 checkpoint is only usable against the new
  gold with `ILIKE`→`LIKE` on its output.
- Columns `medium`, `paly_slides`, `old_wdss` are gone from `Layers.json` and
  from gold; `content` is still in the layer but no longer appears in gold.
- The subtype hints dropped the layer noun: `PipelinesType = 4` is now the
  hint for `'oil'`, not `'oil pipelines'`. The canonical `source_text` follows
  the hints, so "List gas/condensate. The returned wells must be within 15
  kilometers of oil." is gold for `Wells` near oil *pipelines*, with nothing
  in the text saying pipelines. 2,154 of the 2,502 multi-layer gold records
  name fewer layers than their meta has. That is the dominant v3 miss.
- v3 is gold only: 2,700 records × 2 phrasings = 5,400 pairs, 474k tokens.
  3,000 iters is ~50 epochs; best val loss came at step 1000 and the run
  kept that checkpoint.
- v4 adds the v2 extra data back (rebuilt from `feln-dsl`, which is where it
  came from). It was generated against the old `Layers.json`, so `prepare.py`
  now drops rows on columns gold no longer uses: 3,099 of 15,996 dropped
  (`medium`, `paly_slides`, `old_wdss`, `content`, `doc_by_licensee`, the
  Discoveries `source`), 12,897 kept, 18,297 pairs, 1.37M tokens. Val loss was
  still falling at step 3000 (0.3216), where the run ended.
- The v2 checkpoint scores 53.7% on the new val split (with `ILIKE`→`LIKE`
  applied to its output; 230 of its 278 misses are the layer choice or order
  described above, which the old gold always spelled out). Retraining on the
  new gold takes that to 88.2%.

### Where v3 and v4 missed (71 and 66 of 600)

v4 fixes the where-clause misses that the extra data covers and leaves the
layer misses untouched: 45 queries miss in both, 26 only in v3, 21 only in
v4. v4 columns below, v3 in parentheses.

| Cause | Queries | Example |
|---|---|---|
| Secondary layer not named in the question, model picked another layer | 40 (41) | "Get all oil discoveries within 5 kilometers of gas." wants `Wells`, got `Pipelines` |
| Secondary layer order (3-layer queries) or target/secondary swapped | 12 (11) | "Show oil pipelines … from gas wells and … from gas discoveries" wants `[Pipelines, Discoveries, Wells]`, got `[Pipelines, Wells, Discoveries]` |
| Wrong subtype code, `oil/condensate` (12) vs `oil/condensate shows` (15) or `oil/gas` (5) vs `oil/gas shows` (10) | 8 (7) | "oil/condensate wells" got `content_type = cast(15 as SMALLINT)` |
| Wrong column for a literal (`included_in_discovery_name` for `discovery_name`, `field_name` for `field_label`) | 4 (9) | "discovery name ends with 'Vigdis'" got `included_in_discovery_name LIKE '%Vigdis'` |
| Literal copied wrong, dropped clause, literal on the wrong layer | 2 (3) | |
| LIKE wildcard placement | 0 (1) | `'%Goliat%'` for "ending with 'Goliat'" |

The layer misses are the new gold's ambiguity: 34 of the 52 are on
`source_text` phrasings, 18 on `text`, and both styles omit the layer noun for
secondary layers. Splitting the 600 queries by whether the question mentions
at least as many layer nouns as the meta has layers: when every layer is named
v3 gets 192/207 = 92.8% and v4 197/207 = 95.2%; when not, both get 337/393 =
85.8%. The ceiling with the layer misses left in is 552/600 = 92%. The v1
miss classes (wildcard placement, invented filter on an unfiltered target,
dropped clause, wrong year) are gone or down to one query; three times the
gold covered those.

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

### v5: longer run, same data (2026-09-21)

The NorthSea `FELN.json`, `Layers.json` and OKF docs were unchanged since
the 2026-09-20 regen (same md5, `train.bin` bit-identical to v4), so v5 only
tests the "val loss was still falling at step 3000" lead from v4: three
GPUs, 48 sequences (12k tokens) per iteration, 6,000 iterations, same lr
schedule stretched to 6,000. About 100 ms/iter plus 36 s per checkpoint
write (six saves); 14.5 minutes wall.

Val loss bottomed at step 1800 (0.3258, about 86k sequences seen, v4 saw
96k in 3000 steps) and drifted up to 0.343 by step 6000 while train loss
kept falling, so the best-val checkpoint is the step-1800 one. It scores
532/600 = 88.7%: 60 misses shared with v4, 6 fixed, 8 new, all in the
existing classes (three of the eight new ones are the unnamed secondary
layer, one a secondary-layer order swap, three a column choice
(`included_in_discovery_name` for `discovery_name`, `field_name` for
`field_label`), one a year off by one). Wrong-layer misses are 40 in both.
Same data, same result: the model is at the data's ceiling, not
under-trained. v4 stays shipped.

Per-run curves (val loss at each 200-step eval) are in `train-v4.log` and
`train-v5.log` on the GPU box; miss logs in `eval-v4.log`, `eval-v5.log`.

Split by question style and shape, v4 → v5:

| Split | v4 | v5 |
|---|---|---|
| `text` (casual) | 267/300 = 89.0% | 263/300 = 87.7% |
| `source_text` (canonical) | 267/300 = 89.0% | 269/300 = 89.7% |
| 1-layer | 90/94 = 95.7% | 92/94 = 97.9% |
| 2-layer | 313/358 = 87.4% | 307/358 = 85.8% |
| 3-layer | 131/148 = 88.5% | 133/148 = 89.9% |
| all layers named in the question | 186/196 = 94.9% | 182/196 = 92.9% |
| some layer unnamed | 348/404 = 86.1% | 350/404 = 86.6% |

### Lessons from v1 to v5

- The ceiling is the data, not compute. gpt2-large (v1) and 2× steps at
  1.5× batch on three GPUs (v5) each moved exact match by two queries or
  less. The 40 wrong-layer misses are questions that never name the
  secondary layer; the fix is in the NorthSea subtype hints and a regen of
  `FELN.json`, not in training.
- Coverage beats everything else. v1 → v2 (+18 points) and v3 → v4 came from
  adding generated pairs for the sparse patterns, and only after
  `normalize.py` put them in gold surface style; two competing surface forms
  would have cost exact match on gold.
- Token val loss is not the metric. It is noisy at `eval_iters = 40`
  (±0.005), it is not comparable across question styles, and v5 had a worse
  val loss with better 1-layer and 3-layer accuracy. Score exact match on the
  held-out gold; treat the style-normalized match as a formatting check
  (it has equalled exact match since v2).
- "Val loss still falling" at the end of a run is not a lead unless the drop
  is well above the eval jitter. v4's last-200-step drop was 0.004; v5 showed
  the minimum is around 90k sequences (about 5 epochs of the 18k pairs)
  whatever the schedule, and a larger batch reaches it sooner then overfits.
- Check the data before retraining. md5 on the source files and a diff of
  `train.bin` took seconds and turned "retrain on the latest data" into a
  controlled experiment; without that, v5 would have read as a data effect.
- Split the misses before spending compute. A layer / order / subtype code /
  column / literal breakdown of the eval log, kept per run, says whether a
  change moved anything or just reshuffled the same 60 queries.
- Checkpoint writes are the wall-clock tax: 36 s per 4.2 GB save on the
  GPU box, six saves in v5. Keep `always_save_checkpoint = False` and strip
  the optimizer before copying.

## To go past 89%

- Layer nouns in the questions: 52 of 66 misses are a secondary layer the
  question never names, and more data did not move them (v3 → v4). Restore
  the layer noun in the NorthSea subtype hints (`'oil pipelines'`, not
  `'oil'`) and regenerate `FELN.json`, or accept that these are ambiguous and
  score them as such.
- Regenerate `extra.jsonl` against the new `Layers.json` (`feln-dsl`
  `gen.py` + `rephrase.py`) so the 3,099 dropped rows come back on current
  columns. Worth a point or two on the named-layer queries, not more.
- Longer run: tried as v5 (2× steps, 1.5× batch). No gain; val loss bottoms
  around 0.32–0.33 whatever the schedule. Do not spend more compute here.
- `oil/condensate` (12) vs `oil/condensate shows` (15) and `oil/gas` (5) vs
  `oil/gas shows` (10) is the one remaining code confusion; a few hundred
  generated pairs contrasting them would settle it.
- Loss on answer tokens only: mask the `Q:` tokens in `train.py` so capacity
  is not spent predicting paraphrased questions. v2 reached 97% without it,
  so not done.

## Known limits

- Column and domain coverage is whatever the training data contains. The OKF
  docs were used to understand the targets (and, before the 2026-09-20 regen,
  to pick the ILIKE columns for the normalizer), not fed to the model.
- The model copies unfamiliar column words: "depth > 350 meters" gives
  `depth > cast(350.0 as DOUBLE PRECISION)`; say "water depth" for
  `water_depth`. A downstream column allowlist catches these.
- Greedy decoding, no grammar constraint, no output validation against the
  schema. Add a schema check downstream if invalid columns matter.

## Credits

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy: model,
  training loop and configurator, MIT licensed.
- GPT-2 weights from OpenAI via Hugging Face `transformers`.
