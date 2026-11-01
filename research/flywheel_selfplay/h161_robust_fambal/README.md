# H161 — Robust Per-Context Normalized/Huber Loss + Family-Balanced Sampling on `r882` Continuation

**Status: preregistered (`PREREGISTERED_NOT_EXECUTED`); not yet run.** See [PREREG.md](PREREG.md).

- **Hypothesis**: Does normalizing each context's training loss by its detached fit-fold `r882` median scale and applying Huber ($\delta = 1.0$) (`CLIP`), drawing equal-family minibatches with gradient accumulation (`FAMBAL`, `2` microbatches of `8` distinct families per update), or combining both (`CLIP_FAMBAL`) on `r882` continuation (`40` AdamW updates/cell, `lr = 1e-4`, `wd = 1e-4`) produce checkpoints admitted by the `H148` 5-fold family cross-fit selector (`gain > +0.001 + 1e-12` and 95% CI LCB `> 0`) and improve out-of-fold equal-family utility over `r882`?
- **Harness**: Merged `H148` 5-fold family cross-fit (`1,219` contexts / `205` families real-S3 `DL-Train`, `fold_seed = 161140`, `fit 3 / select 1 / assess 1` folds, seeds `161101–161103`, `boot_seed = 161141`) + `H151` per-arm L2 drift logging and conditional post-hoc `CTRL_DRIFT` gate.
- **Files**:
  - [PREREG.md](PREREG.md) — prospective protocol and decision gates
  - [fold_manifest.json](fold_manifest.json) — 5-fold family partition (`fold_seed = 161140`)
  - [manifest.json](manifest.json) — SHA-256 input, asset, and executor pins
  - [runtime.py](runtime.py) — input audit, freeze guard (`DOPE = c51a56d0`), and `per_context_loss` / `robust_loss`
  - [science.py](science.py) — 5-fold partition, `batch_plan`, `fambal_batch_plan`, `fold_median_scale`, `choose`, `summarize`, and `decide`
  - [runner.py](runner.py) — single-attempt wave executor and `--smoke` constructor check
  - [monitor.py](monitor.py) — `nice 19` resource supervisor (`<= 4` threads, `RSS < 6 GB`)
  - [test_h161.py](test_h161.py) — `10` data-free unit tests
