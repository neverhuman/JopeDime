# H292: one clean confirmation of H287 ENS_VOTE2 and CLIP on fresh cohort v4

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any v4 context is built or scored.

**What it tests.** Two pre-declared primaries, Holm-corrected: rd_argon's frozen H287 ENS_VOTE2 and CLIP. They are
compared against r3175's own slot-0 pick on 295 fresh ODD families (cohort v4, sha256 `7f7ff09d…`, which uses up the
qualifying ODD reserve), GPU-exact. Both nulls must fail. BAG_SWITCH, ENS_MEAN, ENS_VOTE3 and RERANK8_VETO are
descriptive only.

## Files

| File | Contents |
|---|---|
| [regress_h288.py](regress_h288.py) | step 1b: the H288 ENS_VOTE2 cell reproduced bit-exactly from S3-published artifacts (PASS) |
| [cohort_v4.py](cohort_v4.py) | H288's cohort rule on its 649-family reserve; disjointness asserted |
| [build_fresh.py](build_fresh.py), [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | H288's build and GPU path (unchanged) with the admission wait |
| [evaluate.py](evaluate.py), [analysis.py](analysis.py) | the arms, the nulls, the Holm rule, and the pooled H288+H292 readout |
| [test_h292.py](test_h292.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
