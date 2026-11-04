# H292: one clean confirmation of H287 ENS_VOTE2 and CLIP on fresh cohort v4 (Holm, 2 primaries, GPU-exact)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0125_opus_h292.md` (Grok Bot H-wave pass for Jepson,
2026-11-04 01:25Z, decided). **Claim:** `~/jope_plan/ns/receipts/H292_claim.txt`.

**Integrity.** Zero neural updates. Checkpoint **copies** only. DOPE `c51a56d0` is untouched (`dope_freeze_guard`). The
writer (PID 1402163), the shadow observer (PID 2849436) and the foreign Blender render are never touched. Sealed data,
P3 sealed_excluded, and DL-Test/holdout are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any v4 context is built or scored**.

## Why

In H288 (`1f681ae1`), the H287 arms passed on 300 fresh ODD families (ENS_VOTE2 +0.0854 [+0.040, +0.138]; CLIP
+0.0604, +0.0741, +0.0790), and both nulls failed. The red-team raised six points:
- seven arms were tested with no multiplicity correction;
- CLIP's seeds 2–3 are bit-identical to BAG_SWITCH's, so its pass rests on its seed-1 refit;
- CLIP was exploratory in H287;
- ENS's "3 seeds" are bootstrap seeds on one policy;
- the family-median gain is 0;
- there is an ODD/EVEN level shift.

H292 is **one clean confirmation**: two pre-declared primaries, Holm-corrected, on a disjoint fresh cohort.

## Hypothesis

On fresh cohort v4, the frozen H287 arms ENS_VOTE2 and CLIP (the only two primaries, Holm-corrected) beat r3175's slot 0,
GPU-exact, with LCB > 0 and a sign test on every seed, while both nulls fail.

## Step 1: regressions (done before v4 was touched, or in-process)

- **(a) r3175 DL-Val91 in-process:** **1.1389572403748425 on 91/91** utilities and picks must be reproduced before
  any v4 context, in the same GPU process (H288's `gpu_rescore.py`, unchanged).
- **(b) The published H288 cell: PASSED.** [regress_h288.py](regress_h288.py) fetched H288's published
  `fresh-build.tar.gz` and `fresh-r3175-gpu-costs.npz` by S3 VersionId (sha256 enforced). It rebuilt the packets and
  scored ENS_VOTE2 with H287's scorer. It reproduced **+0.08537299387889595, CI95 [0.039552121289789376,
  0.13829149681075167] and W/T/L 60/221/19 bit-exactly** (bootstrap seed 288150).
- **Evaluator smoke:** an outcome-free run of [evaluate.py](evaluate.py) on H288's already-seen v3 data reproduced
  H288's ENS_VOTE2, CLIP and BAG_SWITCH gains exactly. Its pooled overlap guard tripped, as designed.

## Step 2: cohort v4 (metadata only; published before this prereg)

**The file.** `fresh_cohort_v4_H292.json`, sha256 **`7f7ff09da91b5ab8338a43d62a553b0fefe2cda59e00a8d48e80e49c57266aba`**:
- local copy: `~/jope_plan/ns/receipts/fresh_cohort_v4_H292.json`;
- S3: `…/H292/cohort/`, VersionId `Ie7wT7_grve0Jo3OgIxdB2aYyi6EgqZD`;
- built by [cohort_v4.py](cohort_v4.py), which is H288's `cohort_v3.py` rule unchanged, applied only to the **649 ODD
  families H288 reserved**, in their existing order.

**Result.**
- **Every qualifying reserved family is taken: 295, not 300** (174 had no buildable context and 180 had fewer than 2
  accepts).
- 87 `train_fit` and 208 `train_nonpool`, all with 1,152-candidate pools.
- **0 qualifying ODD families remain reserved**: the ODD reserve is exhausted.

**Disjointness** from H283, H285, H286 and H288 is **asserted in code**. The v1 exclusions remove everything any flywheel
round, RERANK fit or selection has seen.

## Arms (all frozen, no refit; every file by S3 VersionId with sha256 enforced; each scored with its exporter's scorer)

**PRIMARIES: ENS_VOTE2 and CLIP only.** Both come from rd_argon's H287 export (commit `213b30ad`; manifest sha256
`874693a7…`; ensemble spec VersionId `Wg0jPfW1mYwuaEawsCU2gVn5PkGWGCCX`, sha256 `32ca19f3…`), scored with
`ensemble.score_all_h287_arms`.
- **ENS_VOTE2** is a **deterministic single policy**. Its 3 "seeds" are **bootstrap seeds 292150, 292151 and 292152 on
  the same picks** (labelled single-policy).
- **CLIP** uses its 3 refit seeds (members 279101–279103, refit 287101–287103), with bootstrap seed 292150 common draws.
  Whether its seeds 2–3 equal BAG_SWITCH's picks is recorded.

**SECONDARY / DESCRIPTIVE (cannot flip a verdict):**
- BAG_SWITCH (H284 export, `score_frozen_bagged_cell`);
- ENS_MEAN and ENS_VOTE3 (H287);
- RERANK8_VETO (H278 export).

**NULLS:**
- random valid slot (seeds 292101–292103);
- label-free random slot over 0..7 (seeds 292201–292203).

## Rule ([analysis.py](analysis.py)), per primary

**Seed-level tests.** For each "seed" there is a 10,000-draw family bootstrap of the family-equal mean gain, a one-sided
bootstrap p (P(B ≤ 0)), and a one-sided exact sign test on non-tied families. Ties (|d| ≤ 1e-4) are never wins.

**Holm over the 2 primaries** (one-sided family-wise α = 0.05):
- **Arm p-value:** the maximum over seeds of max(bootstrap p, sign p). This is intersection-union: every seed must pass.
- **Order:** arms are tested in ascending order of their p (ties: ENS_VOTE2 first).
  - The **first tested uses level 0.025** (97.5% one-sided intervals).
  - The second uses 0.05.
  - The procedure **stops at the first non-confirmation**.

**An arm tested at level a is CONFIRMED_FRESH iff all of the following hold:**
1. The level-a LCB (the a quantile of the bootstrap) is > 0 on **3/3 seeds**.
2. The median across seeds of the family-mean gain is **> +0.010**.
3. The sign test p is < a on **3/3 seeds**.
4. Harmed families (loss > 0.01) are **≤ 35%** on every seed.
5. Invalid (−2.0) picks are not above slot 0's.
6. **Both nulls fail** criteria 1–5 at the lenient level 0.05.

Otherwise the arm is **NOT_CONFIRMED_FRESH**. If fewer than 80 families build, both are **INSUFFICIENT_FRESH**.

**After the verdict.** A CONFIRMED arm becomes a **shadow-only candidate**. There is no serving or promotion change in
this wave.

**Reporting.** Per arm and seed: family-equal mean, family-median gain, W/T/L, harmed families, Elo (MM Bradley–Terry,
slot 0 = 1500), invalid and changed picks. Also the median of seeds with a pooled CI (10,000-draw family bootstrap),
the always-harmed families by source group (watching `srcfile:rdataset*`), and r3175's absolute fresh slot-0 mean (report
only).

**POOLED (descriptive only).** H288 v3 plus H292 v4 (about 595 ODD families), with every arm and null re-scored on
H288's published packets and the same Holm procedure. It is labelled "pooled, H288 portion already seen".

## Build and score

H288's [build_fresh.py](build_fresh.py) and [gpu_rescore.py](gpu_rescore.py) are used **unchanged** (byte-identical,
tested). [run_gpu.sh](run_gpu.sh) folds in H288's declared admission wait.
- **Admission:** every 5 minutes until 04:00Z, it runs only if free VRAM is ≥ 4.5 GiB, MemAvailable is ≥ 8 GiB and
  `flock -n ~/qualification.lock` succeeds (never stolen). It then runs one process under a 3 GiB memory cap.
- **Refusals:** if the scorer itself refuses with NOT_ADMITTED_VRAM, before any forward pass, the refusal is recorded
  and the wait resumes into the next attempt directory. That is not a scientific retry.
- **Single scientific attempt.**

**Lock note (declared deviation from the assignment's wording).** The assignment names `~/jope_plan/qualification.lock`.
On xbabe1, `lslocks` shows that file **held by the live writer (PIDs 1402163/1440487) for its whole lifetime**. Flocking
it could never succeed, and contending for the writer's lock is out of bounds. As in H281–H288, the GPU step uses
`~/qualification.lock`; the writer and its lock are never touched.

**Cleanup.** The xbabe1 r3175 copy is deleted after scoring, with its sha256 `1d6ea900…` verified first.

## Resources and deadlines

- **Hosts:**
  - xbabe2: cores 44–47, nice 19, numeric threads 1, aggregate threads ≤ 4, RSS ≤ 4 GiB; the evaluator aborts if
    MemAvailable < 10 GiB.
  - The GPU is used only for the short scoring step under the lock.
  - Never xbabe0 or xbabe3.
- **Deadlines:** cohort v4 by 01:55Z (published 01:29Z); prereg by 02:15Z; compute stop 04:00Z; closeout by 04:30Z.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H292/` with VersionIds.
- **Git:** this PREREG, the README, `metrics.json` under 50 KB and the index rows.
