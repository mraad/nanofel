# v5 retrain (2026-09-21)

Data: NorthSea FELN.json / Layers.json / okf unchanged since v4 (md5 match). v5 = longer run on 3 GPUs.

- [x] Review implementation, confirm data identical
- [x] rsync repo -> gc3, prepare.py
- [x] Train v5: devices 0,1,2, grad_accum 3, 6000 iters, out-nanofel-v5
- [x] Track train.log (val loss per 200 steps)
- [x] Eval v5 on val.jsonl, classify misses vs v4
- [x] Strip optimizer, copy ckpt home: skipped, v5 (88.7%) < v4 (89.0%), v4 stays shipped
- [x] Update README (results table, v5 section), config if v5 ships
- [ ] Commit (not requested)
