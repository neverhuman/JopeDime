# H286: powered, disjoint fresh replication (cohort v2)

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any fresh context is built or scored.

**What it tests.** It replicates H283's fresh-cohort test at a larger scale on 200 new EVEN families (cohort v2, sha256
`0de1b383…`), disjoint from H283's 60. The arms are:
- the primary H278 frozen RERANK8_VETO;
- rd_argon's H284 frozen BAG_SWITCH and BAG_SWITCH_NESTED, as pre-declared arms;
- a random-valid-slot null.

**Readouts.** Secondary sign-test and normalized-gain readouts, an H283+H286 pooled readout, and descriptive r782/r785
pools.

## Files

| File | Contents |
|---|---|
| [cohort_v2.py](cohort_v2.py) | the outcome-blind cohort v2 |
| [build_fresh.py](build_fresh.py) | H283's context builder (a `--wave` flag only) |
| [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | the xbabe1 GPU step: r3175 primary; r782/r785 descriptive |
| [evaluate.py](evaluate.py), [analysis.py](analysis.py) | the arms, the null, the rule, and the secondary, pooled and descriptive readouts |
| [test_h286.py](test_h286.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
