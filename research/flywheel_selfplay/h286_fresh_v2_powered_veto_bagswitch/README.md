# H286: powered, disjoint fresh replication (cohort v2)

**Status: complete. Verdict: NOT_TRANSFERRED_FRESH**, for RERANK8_VETO (primary) and for both pre-declared H284 arms
(BAG_SWITCH and BAG_SWITCH_NESTED). There is **no serving or promotion change**. H283's verdict stands.

**The result.** With 200 fresh families, about 7× H283's power, all three frozen rerankers **gain in mean** over r3175's
slot 0, but **none passes the rule**:

| Arm | Gain by seed | Median gain [pooled CI] | Seeds with lower bound > 0 |
|---|---|---|---|
| RERANK8_VETO | +0.045 / +0.038 / +0.060 | +0.045 | 1/3 (seed 3) |
| BAG_SWITCH | +0.069 / +0.063 / +0.036 | +0.063 [+0.019, +0.117] | 2/3 |
| BAG_SWITCH_NESTED | — | +0.051 | 1/3 |

**Why the means overstate it.** The gains are tail-driven:
- every arm's **median family gain is 0** (most families keep slot 0's choice or tie);
- the random-valid-slot **null also gains +0.016** in mean, from the heavy right tail of fresh utilities.

The rule correctly fails the null too (harmed share 36–40%).

**Arm by arm:**
- **BAG_SWITCH is the strongest and safest.** It has the lowest harm (3.5–10.5% of families) and fewer invalid picks
  than slot 0. Its sign test is p = 0.004 on seed 278102.
- **RERANK8_VETO** has p = 0.0003 on seed 278103.
- **No arm** reaches sign-test p < 0.05 on 3/3 seeds.

## Run record

- **Prereg:** published on `origin/v1/next` as `1e3585c3` at 23:30:47Z, before any fresh context was built or scored
  (same-agent review receipt sha256 `54f21350…`).
- **Cohort v2:** published at 23:25Z.
- **GPU step:** xbabe1, 23:32:07–23:32:42Z, under `~/qualification.lock`, with a 3 GiB memory cap per process.
- **Evaluation:** xbabe2 CPU, cores 44–47.
- **Integrity:** 0 neural updates. DOPE `c51a56d0` was unchanged in every process, and the checkpoint copies kept their
  hashes. The writer (PID 1402163) was OK at every check and was never touched; neither was the shadow observer.

## Hypothesis

On fresh cohort v2 (200 EVEN families with non-empty pools, disjoint from H283's 60, from H285's ODD half and from
everything any flywheel round, RERANK fit or selection has seen), frozen H278 RERANK8_VETO beats r3175's slot 0, scored
GPU-exact.

## Setup

1. **Cohort v2.** [cohort_v2.py](cohort_v2.py) builds it from metadata only: sha256 `0de1b383…`, S3 VersionId
   `S7duiC8O…`, `~/jope_plan/ns/receipts/fresh_cohort_v2_H286.json`.
   - **Source:** `fresh_cohort_v1` EVEN, in the same order, minus H283's 60.
   - **Qualification:** S1 contexts with at least 2 accepts (a line count only), preferring `train_fit`.
   - **Receipts:** 480 scanned; 154 had no buildable context and 126 had fewer than 2 accepts.
   - **Result:** 200 families, 60 `train_fit` and 140 `train_nonpool`, all with 1,152-candidate pools.
   - **All 200 built**, compared with 29/60 in H283.
2. **Build and score.** These follow H283's path:
   - [build_fresh.py](build_fresh.py): the builder, bit-exact on DL-Val91;
   - [gpu_rescore.py](gpu_rescore.py): H283's scorer with the checkpoint sha256 and round as arguments. The r3175
     DL-Val91 regression was **exact (91/91, 1.1389572403748425)** in-process before the fresh contexts.
3. **Arms.**
   - **RERANK8_VETO (primary):** the H278 frozen models (seeds 278101–278103).
   - **BAG_SWITCH and BAG_SWITCH_NESTED (pre-declared):** rd_argon's H284 frozen export (seeds 279101–279103). Its
     manifest was on `v1/next` at commit `972d3011`, 23:25:55Z, before the prereg. Each of the 30 files was fetched by
     VersionId with sha256 enforced, and scored with H284's own `score_frozen_bagged_cell`.
   - **Null:** SEEDED_RANDOM_VALID_SLOT (seeds 286101–286103).
   - **Statistics:** [analysis.py](analysis.py) applies H283's rule unchanged, with bootstrap seed 286150 (10,000
     draws) and an 80-family floor. Ties (|d| ≤ 1e-4) are not wins.
4. **Regression.** H281's r3175 panel was reproduced bit-exactly before scoring.

## Outcome (200 fresh families, 1 context each; r3175 slot-0 fresh mean 1.3386, top-8 oracle 1.5618)

| Arm − r3175 slot 0 | Family-mean gain by seed | Per-seed CI95 | Median over seeds [pooled CI95] | W/T/L by seed | Elo | Harmed share | Invalid (slot 0 / arm) | Changed picks | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **RERANK8_VETO** | +0.0451, +0.0381, +0.0595 | [−0.008, +0.101], [−0.013, +0.093], **[+0.007, +0.116]** | **+0.0451** [−0.005, +0.102] | 64/88/48, 41/127/32, 75/87/38 | 1550, 1543, 1618 | 19%, 12%, 16% | 4/3 each | 114, 75, 116 | NOT_TRANSFERRED_FRESH |
| **BAG_SWITCH** | +0.0692, +0.0630, +0.0356 | **[+0.023, +0.126], [+0.019, +0.117]**, [−0.027, +0.100] | **+0.0630 [+0.019, +0.117]** | 44/124/32, 27/163/10, 32/144/24 | 1555, 1672, 1550 | 10.5%, 3.5%, 8.5% | 4/2, 4/2, 4/4 | 77, 37, 58 | NOT_TRANSFERRED_FRESH |
| **BAG_SWITCH_NESTED** | +0.0692, +0.0514, +0.0356 | **[+0.023, +0.126]**, [−0.009, +0.114], [−0.027, +0.100] | +0.0514 [+0.003, +0.114] | 44/124/32, 33/153/14, 32/144/24 | 1555, 1649, 1550 | 10.5%, 5%, 8.5% | 4/2, 4/3, 4/4 | 77, 47, 58 | NOT_TRANSFERRED_FRESH |
| Random valid slot (null) | +0.0155, +0.0082, +0.0537 | all include 0 | +0.0155 [−0.041, +0.098] | 94/12/94, 94/11/95, 93/8/99 | 1500, 1498, 1489 | 35.5%, 39%, 40% | 4/0 each | 200 each | fails |

- **Family-median gains are 0 for every arm on every seed.**

**Secondary (pre-declared; cannot flip a verdict):**
- **Sign test** (one-sided, wins:losses), seeds in order 278101 / 278102 / 278103:

  | Arm | Seed 278101 | Seed 278102 | Seed 278103 | 3/3? |
  |---|---|---|---|---|
  | RERANK8_VETO | 64:48, p 0.078 | 41:32, p 0.175 | 75:38, **p 0.0003** | no |
  | BAG_SWITCH | 44:32, p 0.103 | 27:10, **p 0.004** | 32:24, p 0.175 | no |
  | BAG_SWITCH_NESTED | 44:32, p 0.103 | 33:14, **p 0.004** | 32:24, p 0.175 | no |

- **Normalized gain** (arm − slot 0) / (oracle − slot 0): 167 families kept and 33 dropped (zero gap). The median is
  **0** for every arm and seed, so most families are unchanged.

**Harmed families (RERANK8_VETO).** 38, 24 and 32 by seed. **18 are harmed on all three seeds**, including
`srcfile:rdataset…` (3 families), `srcfile:house-sa…`, `srcfile:treasury…` and `legacy:ab21f465…`. The full lists are in
[metrics.json](metrics.json).

**POOLED, H283 + H286 (pre-declared, descriptive; "pooled, H283 portion already seen"; 229 families).** RERANK8_VETO
+0.0416, +0.0349 and +0.0520 by seed, with per-seed CIs [−0.007, +0.093], [−0.013, +0.085] and **[+0.001, +0.105]**.
The median is +0.0416 [−0.006, +0.094]. **The rule is not met** (lower bound above 0 on 1/3 seeds). The null median is
+0.019 and it fails.

**DESCRIPTIVE distinct pools.** Frozen VETO against each checkpoint's own slot 0, on the same 200 contexts. Each
checkpoint's DL-Val91 GPU regression was exact.

| Checkpoint | Gain by seed | Per-seed CI95 | Median [pooled CI95] | W/T/L by seed | Invalid (slot 0 / arm) |
|---|---|---|---|---|---|
| r782 | +0.0848, +0.0594, +0.0599 | **[+0.024, +0.158]**, [−0.005, +0.136], **[+0.011, +0.114]** | +0.0599 **[+0.016, +0.137]** | 62/92/46, 42/121/37, 61/93/46 | 4/2, 4/3, 4/2 |
| r785 | +0.0594, +0.0291, +0.0595 | **[+0.010, +0.118]**, [−0.021, +0.084], **[+0.013, +0.112]** | +0.0594 **[+0.007, +0.106]** | 57/96/47, 38/125/37, 65/92/43 | 4/2, 4/3, 4/2 |

Both are positive with a pooled lower bound above 0, but each fails seed 278102.

## Verdict

**NOT_TRANSFERRED_FRESH** for RERANK8_VETO, BAG_SWITCH and BAG_SWITCH_NESTED.

**What the larger sample adds over H283.** Powered at 200 fresh families, the frozen rerankers show a **consistent
positive mean** on fresh pools, about +0.04 to +0.06, close to their DL-Val91 size:
- every arm on every seed;
- the descriptive r782 and r785 pools too.

**Why it still fails the gate.**
- **One seed misses each time:** each arm has a seed whose interval includes 0.
- **The lift is tail-driven:** the median family is unchanged, and even the random null gains in mean.

**Takeaway.** **BAG_SWITCH is the most promising candidate**: the largest median gain (+0.063), a pooled CI above 0,
the lowest harm and the fewest invalid picks. It still misses 3/3.

## Next step

1. **Pool H286 with H285.** Its ODD half on the CPU path, under a pre-declared pooled rule, for about 400 families.
2. **Use a tail-aware metric.** Pre-register one, for example a trimmed-mean or rank-based family gain, or the sign
   test as the primary. Mean-based bootstrap intervals are dominated by a few large fresh wins and losses.
3. **Look at the persistent harms.** Inspect the 18 always-harmed families (the `srcfile:rdataset…` group, house, treasury)
   for a systematic failure mode, such as candidate-type bias, before any shadow adoption.
4. **No promotion.** BAG_SWITCH is the candidate to carry into the next powered gate.

## Deviations

- **None to the rule.**
- **The BAG arms were added** under the assignment's pre-declared condition: the export was on `v1/next` before the
  prereg.
- **The H283 component of the pooled readout** reuses H283's own null seeds (283101–283103), as stated in the
  prereg's label.
- **The descriptive r782/r785 runs completed** within the lock hold, at 10–11 s each.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): 29 KB, holding every arm's summary and secondary readouts, the null, the pooled and
    descriptive readouts, GPU records, harmed lists for VETO and pointers;
  - [PREREG.md](PREREG.md), [cohort_v2.py](cohort_v2.py), [build_fresh.py](build_fresh.py),
    [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh), [evaluate.py](evaluate.py),
    [analysis.py](analysis.py), [test_h286.py](test_h286.py) (6 data-free tests) and [manifest.json](manifest.json).
- **Fresh pools:** `~/jope_plan/ns/receipts/fresh_pools_H286.json` points to `…/H286/fresh_pools/`, which holds the top-8
  packets (no picks), the r3175, r782 and r785 GPU cost vectors, and `fresh-build.tar.gz`.
- **S3:**
  - `…/H286/attempt-001/` holds `evaluate.json` (all picks), the panels, the `rescore.json` files, the lock log and
    `s3_manifest.json` (VersionId `7PC3uKHG…`);
  - `…/H286/prereg-1e3585c3/` holds the prereg sources;
  - `…/H286/cohort/` holds cohort v2.
