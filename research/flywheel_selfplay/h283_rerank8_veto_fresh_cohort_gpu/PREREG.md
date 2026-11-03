# H283: frozen RERANK8_VETO on a fresh regression cohort (GPU-exact)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t2305_opus_h283.md` (Grok Bot H-wave pass for Jepson,
2026-11-03 23:05Z, decided). **Claim:** `~/jope_plan/ns/receipts/H283_claim.txt`.

Zero neural updates. Checkpoint **copies** only: r3175 (`1d6ea900…`). DOPE `c51a56d0` is verified before and after.
The writer (PID 1402163) and the shadow observer (PID 2849436), their outputs and custody are never touched. There is
no promotion or serving change. The sealed bucket, sealed-matching ids, P3 sealed_excluded and heldout families, and
DL-Test/holdout data are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any fresh context is built or scored**.

## Why

Every RERANK8_VETO result so far (H270–H282) scores on DL-Train (out of cluster) or on the reused DL-Val91 pools. H283
measures the frozen models on **fresh contexts**: families that no flywheel round, RERANK fit or selection has touched.

## Hypothesis

The frozen H278 RERANK8_VETO models (no refit; seeds 278101–278103) beat r3175's own slot-0 pick on the fresh
regression cohort's EVEN half, scored GPU-exact on a read-only r3175 copy.

## The fresh cohort (decided before any outcome)

**The file.** `fresh_cohort_v1.json`, sha256 **`12d2b1f37bf0e3c48ee61a24949b4f504251fe56b9b26b3081398623982f323c`**:
- local copy: `~/jope_plan/ns/receipts/fresh_cohort_v1.json`;
- S3: `…/H283/cohort/fresh_cohort_v1.json`, VersionId `IFA1R_jOLs0Dw7HfbyMzuAxVgfsvv_Ps`;
- built by [fresh_cohort.py](fresh_cohort.py) from metadata only (no fitness, no `final.json`, no outcome);
- shared with H284 and H285; rd_astra uses this file and does not build its own.

**Source.** Catalog v246, regression, origin real, datasets bucket only. The family key is the dedupe fingerprint,
falling back to the provisional `srcfile:<stem>`.

**Exclusion receipts (record counts):**

| Reason | Records |
|---|---:|
| Not regression | 1,248 |
| Dataset in the 1,780-context corpus, the 1,219/91 packet or the writer input | 358 |
| P3 heldout / sealed_excluded / dev (from the S1-prod input lists) | 268 |
| Shares a coarse P3 family with the flywheel corpus | 43 |
| Name shared with a seen dataset | 4 |
| Family has an excluded member | 1 |

The catalog's 14 sealed-matching ids and any BeyondArena/sealed text match are also excluded; none remain in
regression after the above.

**Buildable.** A family is buildable if it has an admitted S1-prod context (the GP runs that produced every corpus
pool) on the train side (P3 `train_fit` or `train_nonpool`), with a `summary.tar.gz` pool and a `teacher.jtt1.gz`.

**Counts:** 2,729 eligible families, of which 1,972 are buildable (964 EVEN, 1,008 ODD).

**Order.** Families are sorted by sha256(family key). Even first nibble goes to H283; odd goes to H285 (rd_astra, CPU).
H283 takes the **first 60 buildable EVEN families**, with one context per family: the smallest catalog id, then the
smallest context sha256.

**Known limit.** The coarse corpus grouping (`fam-…`) can merge semantically related datasets. Seen-overlap is excluded
by dataset, catalog key, coarse P3 family, source URL/file, normalised name and data hash, but semantic near-relatives
outside those keys may remain. The frozen DOPE encoder's own pre-training data is upstream and not controlled here.

## Context construction (the corpus's own builder; [build_fresh.py](build_fresh.py))

- **Table:** the S1 context's teacher JTT1 with the test rows stripped.
- **Pool:** the `summary.tar.gz` accepts, extracted as `dataset_builder.py` does. Each candidate's fitness is the GP
  record's own `fitness`, which is what the corpus DL03 pools carry.
- **Features:** H58 `prepare_cache.py`'s own functions on CPU with the frozen network (dope629, lm48_z, tokens,
  base_prior_cost, sos_prior, proj16).
- **Regression (already run, on corpus contexts only, before this prereg).** On 10 DL-Val91 contexts, the rebuilt
  table bytes, pool arrays (program, fitness, gate, program sha256) and H58 cache arrays are all **bit-exact**.
- **Minimum pool size:** contexts with fewer than 8 candidates do not build (K = 8). **INSUFFICIENT_FRESH if fewer than
  25 families build.**

## Scoring (xbabe1 GPU; [gpu_rescore.py](gpu_rescore.py), [run_gpu.sh](run_gpu.sh))

**The scorer.** H281's writer-path scorer, unchanged on its DL-Val91 path: GPU tokens, proj16 in float64, PyTorch
defaults, batch 1, `no_grad`, eval mode.

**The run.**
- It starts under `flock -n ~/qualification.lock`, retrying every 5 minutes and never stealing it.
- VRAM must be free ≥ 1.5 GiB need + 3 GiB, with the allocator capped at 2 GiB.
- **Regression first, in the same process.** r3175 must re-score DL-Val91 to 1.1389572403748425, with 91/91
  utilities and picks. Only then are the fresh contexts scored, with the same `predict` path.

**Abort conditions.** Any CUDA OOM or writer error aborts the run, with no retry. If the lock is never acquired by
02:15Z, the wave closes **BLOCKED_GPU_LOCK**.

## Evaluation (xbabe2 CPU; [evaluate.py](evaluate.py), [analysis.py](analysis.py))

- **Code reuse.** H278 code is imported unchanged (sha-pinned): `topk.load_packets` builds each fresh top-8 packet
  from the GPU costs with the H278-shipped `topk` builder, and `freeze.frozen_picks` applies each model. H281's
  `fetch_models` (S3 VersionId, sha256 enforced) and `panel` are reused unchanged.
- **Regression first:** H281's r3175 GPU panel must be reproduced bit-exactly.
- **Frozen models (no refit)** come from `…/H278/export/export_manifest.json` (VersionId
  `S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, sha256 `2fce784a…`):

| Seed | n_est | Strict margin | Ranker sha256 / VersionId | Veto sha256 / VersionId |
|---|---:|---:|---|---|
| 278101 | 200 | 0.1 | `7a690df843d8ac78…` / `yZXnwTfHBK3.U1bZDl1fB35GWifsev8l` | `1d099ac06298b460…` / `z.DQomoEdHXaz96vLF0o37jux3l2KpH.` |
| 278102 | 50 | 0.1 | `2a2a8b3b5dd767f6…` / `bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1` | `9df7a85a622b51ae…` / `4b9vDz.S3com.E6j7dGIukcfJ70kb4IF` |
| 278103 | 100 | 0.1 | `ae406a1c416cd780…` / `CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq` | `db7283f3c8da2112…` / `uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I` |

- **Descriptive null:** SEEDED_RANDOM_VALID_SLOT (seeds 283101–283103) picks uniformly over the valid (gated) slots
  1..7, or slot 0 if none is valid.

### Primary rule

**All five parts must hold, else NOT_TRANSFERRED_FRESH.** A family is one context.

1. The per-seed family-bootstrap LCB95 of the family-mean gain is > 0 on **3/3 seeds** (10,000 draws, seed 283150).
2. The **median across the 3 seeds** of the family-mean gain is **> +0.010**.
3. **Harmed families** (gain < −0.01) are **≤ 35%** of families on every seed.
4. FROZEN_VETO makes **no more invalid (−2.0) picks than slot 0** on any seed.
5. The random null does **not** pass parts 1–4.

Ties (|d| ≤ 1e-4) are not wins.

**Verdicts.** FRESH_TRANSFER_PASS, NOT_TRANSFERRED_FRESH, **INSUFFICIENT_FRESH** (fewer than 25 families build) or
**BLOCKED_GPU_LOCK**.

**Reporting.** Per seed: family-mean and family-median gain, CI95, W/T/L, harmed families, Elo (slot 0 = 1500; H278's
`bt_two`), invalid and changed picks. Also the median and pooled CI95 across the 3 seeds, and r3175's absolute slot-0
fresh utility mean (report only, not compared with 1.1389572). No AUC, and no speed verdicts.

## Pools for H284

After scoring, the fresh top-8 packets are published to S3 with VersionIds:
- the features X, top-8 costs and top-8 order, with no picks;
- `fresh_pools_H283.json` in `~/jope_plan/ns/receipts/`.

## Resources and deadlines

- **Hosts:**
  - xbabe1: nice 10, ≤ 4 threads, outputs < 100 MB. The r3175 copy is deleted at close.
  - xbabe2: taskset 44–47, nice 19, RSS ≤ 4 GiB; paused if MemAvailable < 12 GiB or swap-in > 2,000 pages/s.
  - Never xbabe0 or xbabe3. Single attempt, no scientific retry.
- **Deadlines:** cohort file by 23:40Z (published 23:09Z); prereg by 23:59Z; **compute stop 02:15Z**; closeout by
  03:00Z on Nov 4.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H283/` with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows (`GIT_COMMITTED`).
