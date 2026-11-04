# H288: confirmatory, powered, GPU-exact fresh test on ODD cohort v3

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any fresh context is built or scored.

**What it tests.** The frozen BAG_SWITCH and BAG_SWITCH_NESTED (primary), RERANK8_VETO (secondary replication) and,
if exported in time, rd_argon's H287 seed ensembles (ENS_MEAN, ENS_VOTE2, ENS_VOTE3, CLIP; exported before the prereg). Each is compared against r3175's own slot-0 pick on 300 fresh
ODD families (cohort v3, sha256 `e247983d…`), disjoint from H283, H285's 60 and H286.

**How.**
- **Scoring:** GPU-exact, after an in-process exact r3175 regression.
- **Rule:** tail-aware; the sign test p < 0.05 on 3/3 seeds is part of the primary rule.
- **Nulls:** two, both of which must fail.

## Files

| File | Contents |
|---|---|
| [cohort_v3.py](cohort_v3.py) | the outcome-blind cohort v3 (H286's rule; ODD; H285's 60 excluded) |
| [build_fresh.py](build_fresh.py), [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | H286's build and GPU path |
| [evaluate.py](evaluate.py), [analysis.py](analysis.py) | the arms, both nulls, the tail-aware rule, and the secondary and pooled readouts |
| [h286_always_harmed.json](h286_always_harmed.json) | H286's 18 always-harmed VETO families, for the descriptive overlap |
| [test_h288.py](test_h288.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
