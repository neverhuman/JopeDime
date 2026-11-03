# H282: frozen RERANK8_VETO transfer to unseen retained-checkpoint pools

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any GPU run.

**What it tests.** The three frozen H278 RERANK8_VETO models (no refit) are applied to the DL-Val91 top-8 pools of 7
retained flywheel checkpoints that no RERANK wave has used: r782, r785, r859, r2750, r2795, r3100 and r4330.

**How.**
1. Each checkpoint's pool is re-scored on xbabe1's GPU with H281's exact writer-path scorer, and must reproduce its
   stored utility exactly.
2. The frozen pick is compared against each checkpoint's own slot-0 pick.

## Files

| File | Contents |
|---|---|
| [eligibility.py](eligibility.py), [eligible.txt](eligible.txt) | the pre-outcome eligibility check (strict key/shape load, completed round json) and its result |
| [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | step 1, xbabe1 GPU (H281's scorer, with round and sha256 as arguments) |
| [transfer.py](transfer.py), [analysis.py](analysis.py) | step 2, xbabe2 CPU (H278's topk/freeze, H281's fetch/panel; the primary rule) |
| [test_h282.py](test_h282.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
