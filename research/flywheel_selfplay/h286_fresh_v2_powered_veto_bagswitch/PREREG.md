# H286: powered, disjoint fresh replication (cohort v2): frozen RERANK8_VETO, plus the H284 frozen BAG_SWITCH arms

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2325_opus_h286.md` (Grok Bot H-wave pass for Jepson,
2026-11-03 23:25Z, decided). **Claim:** `~/jope_plan/ns/receipts/H286_claim.txt`.

**Scope.** H286 is a larger, **disjoint** fresh replication, not a retry of H283. H283's 60 families are excluded, and
H283's verdict (NOT_TRANSFERRED_FRESH) stands as published.

**Integrity.** Zero neural updates. Checkpoint **copies** only. DOPE `c51a56d0` is verified before and after. The
writer (PID 1402163) and the shadow observer (PID 2849436) are never touched. Sealed data, P3
sealed_excluded/heldout, and DL-Test/holdout are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any fresh context is built or any
fresh forward pass is run**.

## Hypothesis

On fresh cohort v2 (200 EVEN families with non-empty pools, disjoint from H283's 60, from H285's ODD half and from
everything any flywheel round, RERANK fit or selection has seen), frozen H278 RERANK8_VETO beats r3175's slot 0,
scored GPU-exact.

## Cohort v2 (metadata only; published before this prereg)

**The file.** `fresh_cohort_v2_H286.json`, sha256 **`0de1b3839cb3201d6c8a05aafec134e28e119ae6c233588b41d6169adb3e7b11`**:
- local copy: `~/jope_plan/ns/receipts/fresh_cohort_v2_H286.json`;
- S3: `…/H286/cohort/`, VersionId `S7duiC8OFJoMp9Bhs8cnGc6E3.K3IhWW`;
- built by [cohort_v2.py](cohort_v2.py).

**Construction.**
- **Starting point:** `fresh_cohort_v1.json` (sha256 `12d2b1f3…`), EVEN half only, in the same sha256(family key)
  order, minus H283's 60.
- **Candidate contexts:** the train-side S1-prod contexts of each family's catalog datasets.
- **Qualification:** a context qualifies with **at least 2 accepts**, counted as a line count only; no record is
  parsed and no fitness is read.
- **Choice per family:** `train_fit` first, then the smallest catalog id, then the smallest context sha256.
- **Size:** the first 200 qualifying families.

**Receipts.** 480 EVEN families were scanned: 60 were H283's, 154 had no buildable context and 126 had fewer than 2
accepts. All 200 taken contexts have 1,152 accepts (60 `train_fit`, 140 `train_nonpool`). The build and scoring cost
is small (H283 built 60 in about 20 s and scored 29 on the GPU in about 10 s), so no smaller prefix is needed.

## Construction, scoring and evaluation

**Contexts.** [build_fresh.py](build_fresh.py) is H283's builder unchanged (only a `--wave` flag): the train-only
teacher table, the S1 pool and H58 features. It is bit-exact on DL-Val91. A pool with fewer than 8 candidates does not
build.

**GPU (xbabe1).** [gpu_rescore.py](gpu_rescore.py) is H283's scorer, with the checkpoint sha256 and round taken as
arguments, as in H282. [run_gpu.sh](run_gpu.sh) runs as follows.
- **Lock:** it waits for `flock -n ~/qualification.lock`, retrying every 5 minutes and never stealing it. If the lock
  is not acquired by **03:00Z**, the wave closes **BLOCKED_GPU_LOCK**.
- **Memory:** each process runs under a 3 GiB memory cap. If MemAvailable is below 8 GiB before a process starts,
  the step stops.
- **PRIMARY run (r3175):** the in-process DL-Val91 regression must reproduce **1.1389572403748425 on 91/91**; only then
  are the fresh contexts scored.
- **DESCRIPTIVE runs (r782 and r785):** these are H282's genuinely distinct pools, scored from read-only copies on the
  same path, each after its own exact DL-Val91 regression. They run only if the primary is COMPLETE and at least 15
  minutes remain, and they are **not part of any verdict**.

**Evaluation (xbabe2 CPU).** [evaluate.py](evaluate.py) and [analysis.py](analysis.py) are H283's code, with null
seeds 286101–286103, bootstrap seed 286150 (10,000 draws) and an 80-family floor.
- **Regression first:** H281's r3175 GPU panel must be reproduced bit-exactly.
- **Packets:** top-8 packets are built with the H278 `topk` builder.

### Pre-declared arms

**RERANK8_VETO (primary).** The H278 frozen models, seeds 278101–278103, from `…/H278/export/export_manifest.json`
(VersionId `S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, sha256 `2fce784a…`):

| Seed | n_est | Strict margin | Ranker sha256 / VersionId | Veto sha256 / VersionId |
|---|---:|---:|---|---|
| 278101 | 200 | 0.1 | `7a690df843d8ac78…` / `yZXnwTfHBK3.U1bZDl1fB35GWifsev8l` | `1d099ac06298b460…` / `z.DQomoEdHXaz96vLF0o37jux3l2KpH.` |
| 278102 | 50 | 0.1 | `2a2a8b3b5dd767f6…` / `bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1` | `9df7a85a622b51ae…` / `4b9vDz.S3com.E6j7dGIukcfJ70kb4IF` |
| 278103 | 100 | 0.1 | `ae406a1c416cd780…` / `CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq` | `db7283f3c8da2112…` / `uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I` |

**BAG_SWITCH and BAG_SWITCH_NESTED (added).** These are rd_argon's H284 frozen exports.
- **Why they are in:** H284's `export_manifest.json` was on `origin/v1/next` before this prereg (commit `972d3011`,
  23:25:55Z; file sha256 `b570d3af…`).
- **Models:** seeds 279101–279103, each a ranker, a veto and 8 bags (30 files). Each file is fetched by its S3 key and
  VersionId, with sha256 enforced.
- **Scoring:** H284's own `freeze_bagged.score_frozen_bagged_cell` is applied unchanged to the same top-8 features
  (H284's `topk.py`, `veto.py` and `h265_core.py` are byte-identical to H278's).
- **Fetch check:** fetching and scoring were smoke-tested on random inputs, with no outcome read.
- **Pairing and verdicts:** the H284 seeds are paired with the 3 seed slots by index. Each BAG arm gets **the same rule
  and its own verdict**; the BAG arms do not change the RERANK8_VETO verdict.

**Null.** SEEDED_RANDOM_VALID_SLOT, seeds 286101–286103, picks uniformly over the valid slots 1..7.

### PRIMARY rule (H283's rule, unchanged; all must hold)

1. The per-seed family-bootstrap LCB95 of the family-mean gain is > 0 on **3/3 seeds**.
2. The median across the 3 seeds is **> +0.010**.
3. Harmed families (loss > 0.01) are **≤ 35%** on every seed.
4. Invalid (−2.0) picks are not above slot 0's.
5. The null **fails** the same rule.

Ties (|d| ≤ 1e-4) are not wins.

**Verdicts.** **FRESH_TRANSFER_PASS**, **NOT_TRANSFERRED_FRESH**, **INSUFFICIENT_FRESH** (fewer than 80 families
built) or **BLOCKED_GPU_LOCK**.

### Secondary, pooled and descriptive readouts (none can flip a verdict)

**SECONDARY (pre-declared, reported):**
- **(a)** a one-sided exact sign test on non-tied families, p < 0.05 on 3/3 seeds;
- **(b)** the normalized gain (arm − slot 0) / (oracle_top8 − slot 0) per family, with families whose gap is ≤ 1e-4
  dropped (count reported); median and CI per seed.

**POOLED (pre-declared, descriptive).** H283's 29 families plus the H286 families, with the same rule, labelled
**"pooled, H283 portion already seen"**.

**DESCRIPTIVE.** Frozen VETO against r782's and r785's own slot 0 on the same contexts.

**Reporting.** Per seed: family-mean and family-median gain, CI95, W/T/L, harmed families, Elo (slot 0 = 1500),
invalid and changed picks. Also the median and pooled CI95 across seeds, and r3175's absolute fresh slot-0 mean (report
only). No AUC, and no speed verdicts.

**Pools.** The fresh top-8 packets (no picks) are published to S3 with VersionIds, and
`~/jope_plan/ns/receipts/fresh_pools_H286.json` is written.

## Resources and deadlines

- **Hosts:**
  - xbabe2: cores 44–47, nice 19, numeric threads 1 (LightGBM `n_jobs` 1), RSS ≤ 4 GiB.
  - xbabe1: nice 10, ≤ 4 threads, host RSS ≤ 3 GiB (memory cap), abort if MemAvailable < 8 GiB. The checkpoint copies
    are deleted at close.
  - Never xbabe0 or xbabe3. Single attempt, no scientific retry.
- **Deadlines:** cohort v2 by 00:10Z (published 23:25Z); prereg by 00:30Z; **compute stop 03:00Z**; closeout by 03:45Z
  on Nov 4.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H286/` with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows (`GIT_COMMITTED`).
