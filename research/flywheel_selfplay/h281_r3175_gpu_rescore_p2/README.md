# H281: GPU re-score of r3175 and H278's P2 adoption gate

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any GPU run.

## What this checks

**Background.** H278 closed P2_BLOCKED: r3175's stored DL-Val91 utility 1.1389572 re-scores on CPU as 1.1366829 (90/91
contexts).

**What H281 does.**
1. Re-scores r3175 on xbabe1's RTX 4090, replicating the writer's own scoring path.
2. Only if the re-score is exact, evaluates H278's unchanged P2 rule, using the frozen models exported in H278 (no
   refit).

The H278 exploratory readout has already been seen, so this is a confirmation of a fixed rule.

## Files

| File | Contents |
|---|---|
| [gpu_rescore.py](gpu_rescore.py) | step 1, xbabe1 GPU (forward-only replica of the writer's dev scoring) |
| [p2.py](p2.py) | step 2, xbabe2 CPU (H278's P2 with H278's code, models fetched by S3 VersionId) |
| [device_pool_cache.py](device_pool_cache.py), [dope_freeze_guard.py](dope_freeze_guard.py) | the writer's helpers, byte-identical from `486b709c3` |
| [test_h281.py](test_h281.py) | data-free controls |
| [manifest.json](manifest.json) | pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
