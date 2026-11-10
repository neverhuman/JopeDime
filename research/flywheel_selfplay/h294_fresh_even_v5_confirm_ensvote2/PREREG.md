# H294: fresh EVEN v5 confirmation of H287 ENS_VOTE2 (single primary; GOV invalid-pick non-inferiority)

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0158_opus_h294.md` (Grok Bot H-wave pass for Jepson,
2026-11-04 01:58Z, decided). **Claim:** `~/jope_plan/ns/receipts/H294_claim.txt`.

**Integrity.** Zero neural updates. Checkpoint **copies** only. DOPE `c51a56d0` is untouched (`dope_freeze_guard`). The
writer (PID 1402163), the observer (PID 2849436) and every foreign job are never touched. Sealed data, P3
sealed_excluded, and DL-Test/holdout are never opened.

**Publication.** This prereg is pushed to `origin/v1/next` (non-force) **before any v5 context is built or scored**.

## Why, and the GOV decision

In H292 (`f9e31c57`), ENS_VOTE2 passed every statistical part on fresh ODD v4 (295 families: +0.0665 [+0.021,
+0.122], W/T/L 50/228/17, harm 5.4%). It failed **only** "invalid picks not above slot 0" (7 against 6).

**GOV's decision**, made before any EVEN-v5 data is built: the invalid-pick clause becomes a pre-declared
**non-inferiority** rule:
- **Pass iff** (arm_invalid − slot0_invalid) ≤ max(2, ⌈0.01 × n_families⌉) **and** the exact one-sided McNemar test on
  discordant invalid picks gives **p ≥ 0.05**.
- **Scope:** because the tolerance was chosen after seeing H292, it counts **only on a new fresh cohort**. **H292 stays
  NOT_CONFIRMED_FRESH.**
- **Smoke readout:** in this evaluator's outcome-free smoke on H292's already-published data, ENS_VOTE2 would pass the
  new clause (excess 1, tolerance 3, McNemar p 0.5). That is exactly why it cannot count there.

## Hypothesis

On a new, disjoint fresh EVEN cohort v5, the frozen H287 ENS_VOTE2 (export `213b30ad`) beats r3175's slot 0, GPU-exact,
under the H292 rule with the non-inferiority invalid-pick clause above.

## Step 1: regressions

- **(a) r3175 DL-Val91 in-process:** **1.1389572403748425 on 91/91** must be reproduced in the same GPU process before
  any v5 context (H292's `gpu_rescore.py`, unchanged).
- **(b) The published H292 cell: PASSED.** [regress_h292.py](regress_h292.py) fetched H292's published artifacts by
  S3 VersionId (sha256 enforced) and reproduced the ENS_VOTE2 cell (bootstrap seed 292150) **bit-exactly**:
  +0.06647400236769202, CI95 [0.0056759322267907735, 0.15047979374400253], W/T/L 50/228/17.
- **(c) The secondary HARMVETO port:** [evaluate.py](evaluate.py) reproduces the published H291 Part B ENS_VOTE2+HARMVETO
  seed gains on H286 EVEN (0.0023750521973936573, 0.02714368976363186, 0.03312056683143756) **bit-exactly**, as
  verified in the smoke. The same check is repeated at run time.

## Step 2: cohort v5 (metadata only; published before this prereg)

**The file.** `fresh_cohort_v5_H294.json`, sha256 **`ea5364ad8d224719300a7932b0d8418709793405c1668dc581aac59302f7d0c9`**:
- local copy: `~/jope_plan/ns/receipts/fresh_cohort_v5_H294.json`;
- S3: `…/H294/cohort/`, VersionId `zfXFbXPDS1P2CSxGixOrykepkzkWvFNT`;
- built by [cohort_v5.py](cohort_v5.py), which is H292's `cohort_v4.py` structure with **H286's qualification rule
  unchanged** (at least 2 accepts, one context per family, `train_fit` preferred, sha256(family key) order), applied to
  the **EVEN** half of `fresh_cohort_v1`.

**Construction.** It skips H283's 60 and **H286's 480 scanned EVEN families** (qualifying or not, up to H286's 200th),
then takes **all remaining qualifying families**.

**Result.**
- **358 families** (104 `train_fit`, 254 `train_nonpool`; all with 1,152-candidate pools). 210 had no buildable context
  and 220 had fewer than 2 accepts.
- The EVEN reserve is now exhausted.
- **Disjointness** from H283, H285, H286, H288 and H292 is **asserted in code**.
- The floor of 100 is met.

## Arms (all frozen, no refit; every file by S3 VersionId with sha256 enforced; exporters' own scorers)

**ONE PRIMARY: ENS_VOTE2** (H287 `ensemble.score_all_h287_arms`). It is a **deterministic single policy**: its 3
"seeds" are **bootstrap seeds 294150, 294151 and 294152 on the same picks** (labelled single-policy).

**SECONDARY / DESCRIPTIVE (cannot flip the verdict):**
- **ENS_VOTE2+HARMVETO:** rd_argon's H291 frozen harm-veto (fit lock at commit `d8bc373b80`, `fit_lock.json` sha256
  `3d054619…`; the 3 models for seeds 291101–291103 are fetched by VersionId from H291's `s3_manifest.json`, with
  sha256 enforced against the fit lock). It is applied exactly as H291's `_score_frozen_arm` does:
  - base picks = `compute_vote_picks(member picks, 2)` (verified identical to H287's ENS_VOTE2 in the smoke);
  - 644-column features from `build_context_harm_features_from_arrays`;
  - a switch is kept iff p_harm is below the seed's locked τ (0.135, 0.165 and 0.135).
- **CLIP:** 3 refit seeds.
- **BAG_SWITCH:** H287's per-seed picks, which equal H284's (verified in H288).

**NULLS:** random valid slot (294101–294103) and label-free random slot over 0..7 (294201–294203).

## Rule ([analysis.py](analysis.py); one-sided α = 0.05)

The per-seed statistics are H292's `analysis.summarize`, unchanged: the family-equal mean, a 10,000-draw family
bootstrap, a one-sided exact sign test on non-tied families, harmed share, invalid counts and Elo (MM Bradley–Terry,
slot 0 = 1500).

**CONFIRMED_FRESH iff all hold:**
1. The one-sided 95% bootstrap LCB (the 0.05 quantile) is > 0 on **3/3 seeds**.
2. The median gain across seeds is **> +0.010**.
3. The sign test p is < 0.05 on **3/3 seeds**.
4. Harmed families (loss > 0.01) are **≤ 35%** on every seed.
5. **Invalid non-inferiority** holds on every seed: the excess is ≤ max(2, ⌈0.01 n⌉), which is **4 for n = 358**, and
   the McNemar p is ≥ 0.05.
6. **Both nulls fail** parts 1–5.

Ties (|d| ≤ 1e-4) are never wins.

**Verdicts.** Otherwise **NOT_CONFIRMED_FRESH**. Fewer than 100 families built means **UNDERPOWERED** (no verdict).
A **CONFIRMED** result makes ENS_VOTE2 a **shadow-only candidate**. There is no serving or promotion change.

**Reporting.** Family-equal mean, median of seeds with a pooled CI (10,000-draw family bootstrap), family-median gain,
W/T/L, harmed families, invalid counts for both arms (with the McNemar discordants), Elo, and always-harmed families by
source group (watching `srcfile:rdataset*`). The descriptive strict clause (invalid not above slot 0) is also reported.

**POOLED (descriptive only).** H288 v3 + H292 v4 + H294 v5 (about 950 families), with every arm and null re-scored on
the published H288 and H292 packets. It is labelled "pooled, H288 and H292 portions already seen".

## Build and score

H292's [build_fresh.py](build_fresh.py) and [gpu_rescore.py](gpu_rescore.py) are used **unchanged** (byte-identical,
tested). [run_gpu.sh](run_gpu.sh) is H292's with only the data path and the stop time (05:00Z) changed.
- **Admission:** a 5-minute admission wait for ≥ 4.5 GiB free VRAM, MemAvailable ≥ 8 GiB and `flock -n
  ~/qualification.lock` (never stolen).
- **Process:** one process under a 3 GiB memory cap.
- **Single scientific attempt.**

**Lock note (declared again).** `~/jope_plan/qualification.lock` is held by the live writer for its whole lifetime
(`lslocks`), so the GPU step uses `~/qualification.lock`, as in H281–H292.

**Cleanup.** The xbabe1 r3175 copy is deleted after scoring, with its sha256 `1d6ea900…` verified first.

## Resources and deadlines

- **Hosts:**
  - xbabe2: cores 44–47, nice 19, numeric threads 1, aggregate ≤ 4, RSS ≤ 4 GiB; the evaluator aborts if
    MemAvailable < 10 GiB.
  - The GPU is used only for the scoring step.
  - Never xbabe0 or xbabe3.
- **Deadlines:** cohort v5 by 02:35Z (published 02:00Z); prereg by 02:55Z; closeout by 05:30Z.

## Custody

- **S3:** `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H294/` with VersionIds.
- **Git:** this PREREG, the README, `metrics.json` under 50 KB and the index rows.
