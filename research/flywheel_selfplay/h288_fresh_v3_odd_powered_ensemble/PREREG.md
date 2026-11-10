# H288: confirmatory, powered, GPU-exact fresh test on ODD cohort v3, with a tail-aware rule

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0005_opus_h288.md` (Grok Bot H-wave pass for Jepson,
2026-11-04 00:05Z, decided). **Claim:** `~/jope_plan/ns/receipts/H288_claim.txt`.

**Integrity.** Zero neural updates. Checkpoint **copies** only. DOPE `c51a56d0` is verified before and after. The
writer (PID 1402163) and the shadow observer (PID 2849436) are never touched. Sealed data, P3
sealed_excluded/heldout, and DL-Test/holdout are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any fresh context is built or any
fresh forward pass is run**.

## Why

H286 (`9a79996f`) gave every frozen arm a +0.04 to +0.07 mean gain on 200 fresh EVEN families, but:
- one seed's LCB95 crossed 0 for each arm;
- the family-median gain was 0;
- the null gained +0.016.

H285's ODD half used CPU pools, and its random valid-slot null passed. **The ODD half has not yet had a clean,
powered, GPU-exact test.** H288 is that test, with a pre-declared **tail-aware** rule.

## Hypothesis

On fresh cohort v3 (300 ODD families with non-empty pools, disjoint from H285's 60 ODD prefix, H283's 60, H286's 200
and everything any flywheel round, RERANK fit or selection has seen), frozen BAG_SWITCH beats r3175's slot 0,
GPU-exact. A deterministic seed ensemble from H287 (ENS_MEAN, ENS_VOTE2 or ENS_VOTE3) passes the rule where single seeds do not.

## Cohort v3 (metadata only; published before this prereg)

**The file.** `fresh_cohort_v3_H288.json`, sha256 **`e247983d0412f37ad28ba3553decee1f4f6a096bd461caefafca58fdc858eac3`**:
- local copy: `~/jope_plan/ns/receipts/fresh_cohort_v3_H288.json`;
- S3: `…/H288/cohort/`, VersionId `jSdAj.ZnUf6gUlcKeAWaPWfhvhe2bHsI`;
- built by [cohort_v3.py](cohort_v3.py), which is H286's `cohort_v2.py` with only the half (ODD), the excluded prior
  prefix (H285's 60) and the size (300) changed.

**Construction.** `fresh_cohort_v1` (sha256 `12d2b1f3…`), ODD half, in sha256(family key) order, minus H285's 60.
H286's qualification rule is unchanged: at least 2 accepts (a line count only), one context per family, `train_fit`
preferred.

**Receipts.** 692 ODD families were scanned after H285's 60: 219 had no buildable context, 173 had fewer than 2
accepts, and **649 beyond the first 300 are reserved, untouched, for a later wave**. All 300 taken contexts have
1,152-candidate pools (89 `train_fit`, 211 `train_nonpool`). Build and scoring cost is small (H286: 200 built in about
1 minute, scored in 13 s), so no smaller prefix is needed.

## Construction and scoring (H286's path)

- **Build:** [build_fresh.py](build_fresh.py), H286's builder unchanged: the train-only teacher table, the S1 pool and
  H58 features. It is bit-exact on DL-Val91.
- **Score:** [gpu_rescore.py](gpu_rescore.py), H286's scorer unchanged. [run_gpu.sh](run_gpu.sh) runs as follows.
  - **Lock:** it waits for `flock -n ~/qualification.lock`, retrying every 5 minutes and never stealing it. If the lock
    is not acquired by **04:00Z**, the wave closes **BLOCKED_GPU_LOCK**.
  - **Memory:** the run is one process under a 3 GiB memory cap, and needs MemAvailable ≥ 8 GiB to start.
  - **Regression first, in-process:** r3175 must reproduce DL-Val91 **1.1389572403748425 on 91/91**; only then are the
    fresh contexts scored.
- **Evaluate:** [evaluate.py](evaluate.py) and [analysis.py](analysis.py).
  - **Regression first:** H281's r3175 GPU panel must be reproduced bit-exactly.
  - **Packets:** top-8 packets are built with the H278-shipped `topk` builder.
  - **Smoke:** an outcome-free run on H286's already-published contexts reproduced H286's VETO panel and arm gains
    exactly.

## Arms (all frozen, no refit; every file fetched by S3 VersionId with sha256 enforced)

- **PRIMARY: BAG_SWITCH and BAG_SWITCH_NESTED.** rd_argon's H284 export (seeds 279101–279103; `export_manifest.json`
  sha256 `b570d3af…`, commit `972d3011`), scored with H284's own `score_frozen_bagged_cell`.
- **SECONDARY REPLICATION: RERANK8_VETO.** The H278 export (seeds 278101–278103; VersionId
  `S_PoMNCZ0jiAYru1vEligisesVuKDuPf`, sha256 `2fce784a…`):

| Seed | n_est | Strict margin | Ranker sha256 / VersionId | Veto sha256 / VersionId |
|---|---:|---:|---|---|
| 278101 | 200 | 0.1 | `7a690df843d8ac78…` / `yZXnwTfHBK3.U1bZDl1fB35GWifsev8l` | `1d099ac06298b460…` / `z.DQomoEdHXaz96vLF0o37jux3l2KpH.` |
| 278102 | 50 | 0.1 | `2a2a8b3b5dd767f6…` / `bQcNpCmstPNy42QQHcO5F8fcZk_n3aZ1` | `9df7a85a622b51ae…` / `4b9vDz.S3com.E6j7dGIukcfJ70kb4IF` |
| 278103 | 100 | 0.1 | `ae406a1c416cd780…` / `CnrMHiKmlTjj2_05GxmVK9MI3qkeprvq` | `db7283f3c8da2112…` / `uuDUbHpR0CBuTMhf0.e3JSM.xpyc4q4I` |

- **H287 ENSEMBLE ARMS: ENS_MEAN, ENS_VOTE2, ENS_VOTE3 and CLIP.**
  - **Why they are in:** rd_argon's H287 `export_manifest.json` was on `origin/v1/next` before this prereg (commit
    `213b30ad`, 00:47:46Z; file sha256 `874693a7…`; H287 verdict EXPORTED_ADVANCE, advancing arm ENS_MEAN).
  - **Models:** the ensemble spec (VersionId `Wg0jPfW1mYwuaEawsCU2gVn5PkGWGCCX`, sha256 `32ca19f3…`) and all 60 member
    files (6 members × ranker, veto and 8 bags). Each is fetched by S3 key and VersionId with sha256 enforced.
  - **Scoring:** rd_argon's own `ensemble.score_all_h287_arms`, unchanged. H287's `topk`, `veto`, `h265_core` and
    `bagged` are byte-identical to H284's and H278's.
  - **Fetch check:** fetching and scoring were smoke-tested on random inputs, with no outcome read.
  - **Not included:** H287 also exports ENS_MEAN_PROPOSER and CLIP_ENS_VOTE2. They are **not** arms here; the
    assignment named ENS_MEAN, ENS_VOTE2, ENS_VOTE3 and CLIP.
  - **Consistency check:** H287's per-seed BAG_SWITCH picks are compared with H284's (descriptive).

**Nulls.** Both must fail for any PASS to count:
- **SEEDED_RANDOM_VALID_SLOT** (seeds 288101–288103): uniform over the valid slots 1..7;
- **SEEDED_RANDOM_SLOT_NO_LABELS** (seeds 288201–288203): uniform over top-8 slots 0..7, using no validity
  information.

## PRIMARY rule, per arm (all must hold; [analysis.py](analysis.py))

1. The per-seed family-bootstrap LCB95 of the family-mean gain is > 0 on **3/3 seeds** (10,000 draws, bootstrap seed
   288150).
2. The median across seeds of the family-mean gain is **> +0.010**.
3. A **one-sided exact sign test on non-tied families gives p < 0.05 on 3/3 seeds.** This is new versus H286 and adds
   tail resistance.
4. Harmed families (loss > 0.01) are **≤ 35%** on every seed.
5. Invalid (−2.0) picks are not above slot 0's on any seed.
6. **Both nulls fail** parts 1–5.

Ties (|d| ≤ 1e-4) are not wins.

**Verdicts, per arm:** **FRESH_TRANSFER_PASS**, **NOT_TRANSFERRED_FRESH**, **INSUFFICIENT_FRESH** (fewer than 80
families built) or **BLOCKED_GPU_LOCK**. The wave's headline is the PRIMARY arms' verdicts.

**Seeds.** For the seeded arms, the "seeds" are the 3 model seeds; their per-seed bootstrap uses seed 288150 with
common draws, as in H283 and H286. **ENS_MEAN, ENS_VOTE2 and ENS_VOTE3 are deterministic single policies.** For them, **"3 seeds" means 3 bootstrap seeds
(288150, 288151, 288152) on the same picks**, so W/T/L, the sign test, harm and invalid picks are identical across the
three. CLIP has 3 refit seeds and is treated like the seeded arms.

## Secondary, pooled and reporting

**SECONDARY (cannot flip a verdict):**
- the 10%-trimmed family-mean gain, with a CI (2,000 bootstrap draws);
- the normalized gain (arm − slot 0) / (oracle_top8 − slot 0), with families whose gap is ≤ 1e-4 dropped; median and
  CI;
- the overlap of the always-harmed families (harmed on all 3 seeds) with H286's 18 always-harmed VETO families
  ([h286_always_harmed.json](h286_always_harmed.json)), by source type. This is descriptive: the cohorts are disjoint.

**POOLED (pre-declared, descriptive).** H286's EVEN v2 (200) plus H288's ODD v3 (300), about 500 families, under the
same rule for every arm and null. Every arm and null is re-scored on H286's packets, and H286's VETO panel must be
reproduced. It is labelled **"pooled, H286 portion already seen"**.

**Reporting.** Per seed and per arm: family-mean and family-median gain, CI95, W/T/L, harmed families, Elo (slot 0 =
1500), invalid and changed picks. Also the median and CI across seeds, and the absolute slot-0 fresh mean. No AUC, and
no speed verdicts.

**Pools.** The fresh top-8 packets (no picks) are published to S3 with VersionIds, and
`~/jope_plan/ns/receipts/fresh_pools_H288.json` is written.

## Resources and deadlines

- **Hosts:**
  - xbabe2: cores 44–47, nice 19, numeric threads 1, RSS ≤ 4 GiB.
  - xbabe1: nice 10, ≤ 4 threads, host RSS ≤ 3 GiB (memory cap), abort if MemAvailable < 8 GiB, scratch < 20 GiB. The
    r3175 copy is deleted at close.
  - Never xbabe0 or xbabe3. Single attempt, no scientific retry.
- **Deadlines:** cohort v3 by 00:45Z (published 00:04Z); prereg by 01:30Z; **compute stop 04:00Z**; closeout by 04:45Z.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H288/` with VersionIds.
- **Git:** the README, a metrics JSON under 50 KB and the index rows (`GIT_COMMITTED`).
