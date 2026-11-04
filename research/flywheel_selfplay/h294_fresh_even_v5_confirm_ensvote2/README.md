# H294: fresh EVEN v5 confirmation of H287 ENS_VOTE2

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any v5 context is built or scored.

**What it tests.** One primary: rd_argon's frozen H287 ENS_VOTE2, compared against r3175's own slot-0 pick on 358 fresh
EVEN families (cohort v5, sha256 `ea5364ad…`), GPU-exact. It uses H292's rule with GOV's pre-declared invalid-pick
**non-inferiority** clause, and both nulls must fail. ENS_VOTE2+HARMVETO (H291), CLIP and BAG_SWITCH are descriptive.

## Files

| File | Contents |
|---|---|
| [regress_h292.py](regress_h292.py) | step 1b: the H292 ENS_VOTE2 cell reproduced bit-exactly from S3-published artifacts (PASS) |
| [cohort_v5.py](cohort_v5.py) | H286's rule on the EVEN half beyond H283's 60 and H286's 480 scanned; disjointness asserted |
| [build_fresh.py](build_fresh.py), [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh) | H292's build and GPU path (unchanged) |
| [evaluate.py](evaluate.py), [analysis.py](analysis.py) | the arms (including the H291 HARMVETO port), the nulls, the rule and the pooled readout |
| [test_h294.py](test_h294.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
