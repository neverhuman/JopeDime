# H283: frozen RERANK8_VETO on a fresh regression cohort (GPU-exact)

**Status: complete. Verdict: NOT_TRANSFERRED_FRESH** under the pre-declared rule. There is **no serving or promotion
change**.

**The result.** On 29 fresh families, which no flywheel round, RERANK fit or selection had ever touched, the frozen
H278 models lean the right way but the evidence is too weak to pass:
- **Gains:** +0.0172, +0.0129 and +0.0004 by seed (median **+0.0129**).
- **Head-to-head:** W/T/L by family 15/8/6, 12/15/2 and 14/10/5; Elo 1659, 1811 and 1679.
- **Safety:** harmed families stay at or below 21%, and there are 0 invalid picks.
- **The null:** the random-valid-slot null comes out negative, median −0.1136.
- **Why it fails:** with 29 families of very spread-out utility, each seed's 95% interval is about ±0.15. So **no
  seed's lower bound is above 0**, and part 1 of the rule fails.

This is **an underpowered null, not evidence of harm.**

**Coverage gap.** Only 29 of the 60 assigned EVEN families built. The other 31 S1 contexts (all P3 `train_nonpool`
roles) have **empty candidate pools**.

## Run record

- **Prereg:** published on `origin/v1/next` as `b8724c84` at 23:16:17Z, before any fresh context was built or scored
  (same-agent review receipt sha256 `a1915ed6…`).
- **Shared cohort:** published at 23:09Z (sha256 `12d2b1f3…`).
- **GPU step:** xbabe1 RTX 4090, 23:16:51–23:17:01Z, under `~/qualification.lock`.
- **Evaluation:** xbabe2 CPU, cores 44–47.
- **Integrity:** 0 neural updates. The r3175 copy (`1d6ea900…`) and DOPE `c51a56d0` were unchanged. The writer (PID
  1402163) was OK before, after the regression and after the fresh phase, and was never touched; neither was the
  shadow observer.

## Hypothesis

The frozen H278 RERANK8_VETO models (no refit; seeds 278101–278103) beat r3175's own slot-0 pick on a fresh
regression cohort, scored GPU-exact on a read-only r3175 copy.

## Setup

1. **The shared fresh cohort.** [fresh_cohort.py](fresh_cohort.py) produced `fresh_cohort_v1.json` (sha256
   `12d2b1f3…`; S3 VersionId `IFA1R_jO…`; `~/jope_plan/ns/receipts/`) from metadata only.
   - **Source:** catalog v246 regression.
   - **Excluded:** every dataset, catalog family key and coarse corpus family of the 1,780-context corpus, the
     1,219/91 packet and the writer input (358 datasets, 43 coarse-family matches); P3 heldout / sealed_excluded / dev
     datasets (268); name overlaps (4); sealed ids.
   - **Counts:** 2,729 eligible and 1,972 buildable families. Families are ordered by sha256(family key); the EVEN
     half goes to H283 (first 60) and the ODD half to H285.
2. **Contexts.** [build_fresh.py](build_fresh.py) uses the corpus's own construction:
   - the train-only table is the teacher JTT1 with its test rows stripped;
   - the pool is the S1 `summary.tar.gz` accepts;
   - the features come from H58's own functions on CPU.

   Before the prereg, the build was verified **bit-exact on 10 DL-Val91 contexts** (tables, pools, features, priors).
   **29 of 60 families built**; 31 had empty pools.
3. **GPU scoring.** [gpu_rescore.py](gpu_rescore.py) is H281's writer-path scorer. In one process it ran the r3175
   DL-Val91 regression first (**exact: 91/91 utilities and picks, 1.1389572403748425**), then the 29 fresh contexts.
   Peak VRAM was 28 MB.
4. **Evaluation.** [evaluate.py](evaluate.py):
   - it reproduced H281's r3175 panel bit-exactly first;
   - top-8 packets are built with the H278 `topk` builder;
   - the three frozen models were fetched by S3 VersionId with each sha256 enforced;
   - the null is SEEDED_RANDOM_VALID_SLOT (seeds 283101–283103);
   - the statistics are in [analysis.py](analysis.py): family bootstrap with 10,000 draws, seed 283150; ties
     (|d| ≤ 1e-4) are not wins.

## Outcome (29 fresh families, 1 context each)

| Arm − r3175 slot 0 | Per-seed family-mean gain | Per-seed family-median | Per-seed CI95 | Median over seeds [pooled CI95] | W/T/L by seed | Elo (slot 0 = 1500) | Harmed share (loss > 0.01) | Invalid (slot 0 / arm) | Changed picks |
|---|---|---|---|---|---|---|---|---|---|
| **FROZEN_VETO** | +0.0172, +0.0129, +0.0004 | +0.0025, 0, 0 | [−0.110, +0.142], [−0.143, +0.152], [−0.175, +0.159] | **+0.0129 [−0.141, +0.141]** | 15/8/6, 12/15/2, 14/10/5 | 1659, 1811, 1679 | 20.7%, 6.9%, 17.2% | 0/0 each | 21, 15, 19 |
| Random valid slot (null) | +0.0446, −0.1136, −0.1346 | +0.0029, 0, 0 | [−0.105, +0.187], [−0.311, +0.065], [−0.424, +0.076] | −0.1136 [−0.300, +0.084] | 16/3/10, 14/3/12, 14/2/13 | 1582, 1527, 1513 | 34.5%, 41.4%, 41.4% | 0/0 each | 29 each |

- **Absolute levels:** r3175's slot-0 fresh utility mean is **1.4858** (report only; not comparable with
  1.1389572). The top-8 oracle is 1.7529. FROZEN_VETO's context means are 1.5030, 1.4987 and 1.4862.
- **Harmed families (FROZEN_VETO):** `srcfile:fish` on all three seeds; `03088c09`, `4ef6fccb` and `83367a7f` on two;
  `0d6229e1`, `47f0ab26` and `7aa5fbc1` on one. Lists are in [metrics.json](metrics.json).

### Primary rule

| Part | FROZEN_VETO | Null |
|---|---|---|
| 1. LCB95 > 0 on 3/3 seeds | **fail** (all lower bounds < 0) | fail |
| 2. Median > +0.010 | pass (+0.0129) | fail |
| 3. Harmed ≤ 35% every seed | pass | fail |
| 4. Invalid not above slot 0 | pass | pass |
| 5. Null does not pass | yes | — |

## Verdict

**NOT_TRANSFERRED_FRESH.**
- The frozen models' fresh-cohort gain is small and positive in median (+0.0129). Wins outnumber losses on every seed
  (15:6, 12:2, 14:5), harm is low, and the random null is clearly worse.
- But the spread across 29 families leaves the effect statistically unresolved.
- The +0.039–0.069 lifts measured on DL-Val91 and DL-Train **did not reproduce at that size** on fresh pools, and the
  seed-3 gain is ~0.

## Next step

1. **Pool the fresh halves.** Combine H283 (EVEN, GPU) with H285 (ODD, CPU) and H284's reuse of these pools,
   pre-declaring a pooled analysis, to reach about 60 or more families.
   - The 31 unbuilt EVEN families point to a fix: choose **`train_fit` contexts with non-empty pools** in a cohort v2.
     This is outcome-blind: pool size is metadata, not utility.
2. **Use a variance-aware rule.** Fresh utilities are far more spread than DL-Val91's, with a top-8 oracle of 1.75
   against a slot 0 of 1.49. A sign-based or stratified test, pre-declared, would use the W/T/L signal that the
   mean-based LCB ignores.
3. **No promotion** on this evidence. Shadow stays shadow.

## Deviations

- **31 of 60 assigned families did not build.** Their S1 summaries hold 0 accepts (P3 `train_nonpool`). The
  pre-declared minimum-pool rule excluded them, and the 25-family floor was met.
- **Pool fitness:** the extractor takes the GP record's own `fitness`, as the corpus DL03 pools do, rather than
  `dataset_builder.py`'s `constrained_fitness`. The two differ only on failed-gate candidates. This was established by
  regression before the prereg and is stated in it.
- **Code fix before the prereg:** the run-wrapper compute-stop epoch was corrected to 1793758500 (02:15Z).

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 10 KB, holding the cohort, build, GPU, regression, frozen and null summaries, the
    fresh-pool pointers and the S3 manifest;
  - [PREREG.md](PREREG.md), [fresh_cohort.py](fresh_cohort.py), [build_fresh.py](build_fresh.py),
    [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh), [evaluate.py](evaluate.py),
    [analysis.py](analysis.py), [test_h283.py](test_h283.py) (5 data-free tests) and [manifest.json](manifest.json).
- **Fresh pools for H284:** `~/jope_plan/ns/receipts/fresh_pools_H283.json` points to
  `…/H283/fresh_pools/`, which holds `fresh_top8_packets.npz` (X, top-8 costs and order; no picks),
  `fresh-r3175-gpu-costs.npz` and `fresh-build.tar.gz` (rows, pools, tables, features).
- **S3:**
  - `…/H283/attempt-001/` holds `rescore.json`, `evaluate.json`, the panels, the picks, the lock log and
    `s3_manifest.json` (VersionId `GUw9UKy4…`);
  - `…/H283/prereg-b8724c84/` holds the sources and the review receipt;
  - `…/H283/cohort/` holds `fresh_cohort_v1.json` and its builder.
