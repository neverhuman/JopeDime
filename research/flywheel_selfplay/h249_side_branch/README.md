# Wave H249: Zero-Init Low-Rank Residual Side Branch on Frozen `r882` Adapter (`h249-side-branch-20261103`)

- **Status**: `PREREGISTERED` (pre-execution; `0` outcomes read)
- **Preregistration**: [`PREREG.md`](PREREG.md)
- **Fold Manifest**: [`fold_manifest.json`](fold_manifest.json) (`fold_seed = 249140`, seeds `249101–249103`)
- **Schedules**: [`schedules.json`](schedules.json)
- **Provenance**: [`provenance.json`](provenance.json)
- **Manifest**: [`manifest.json`](manifest.json)
- **Metrics**: [`metrics.json`](metrics.json)
- **Source**: [`treatment.py`](treatment.py), [`analysis.py`](analysis.py), [`runner.py`](runner.py), [`monitor.py`](monitor.py), [`test_h249.py`](test_h249.py)

## Overview

Wave **H249** tests a function-preserving capacity expansion of the frozen `r882` adapter (`29,211` parameters across `47` tensors, including `16` 2-D weight matrices): a zero-initialised low-rank residual side branch $\Delta W = (B A) \cdot (\alpha / r)$ (`A ~ Kaiming-uniform`, `B = 0`, $\alpha = r$) trained with AdamW (`lr = 1e-4`, `betas = (0.9, 0.999)`, decoupled `weight_decay = 1e-2` on the branch only) while all `47` `r882` adapter tensors remain frozen (`requires_grad = False`), compared against the exact H170 `CTRL` full-adapter continuation (`CONTROL`) across `3` seeds (`249101–249103`, fold seed `249140`, `15` cells/arm, `40` updates/cell, `1,800` total updates) on `xbabe2` CPU (`taskset -c 52-55`, `nice -n 19`, H239 A1 cooperative resource guard).

- **Arms (`Holm m = 2`)**:
  - `CONTROL`: exact H170 `CTRL` full-adapter continuation (`29,211` trainable parameters).
  - `SIDE4` (**primary**): rank $r = 4$, $\alpha = 4.0$ low-rank residual side branch on all `16` 2-D adapter matrices (`6,232` trainable parameters; `r882` adapter frozen).
  - `SIDE16` (**secondary**): rank $r = 16$, $\alpha = 16.0$ low-rank residual side branch on all `16` 2-D adapter matrices (`24,928` trainable parameters; `r882` adapter frozen).
