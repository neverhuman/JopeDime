# H199 — BLEND_025 on the flywheel native first-argmin protocol, 205-family pooled cross-fit, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hwave_t1300_h199_rdopus.md`. **Claim:**
`~/jope_plan/ns/receipts/H199_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h199-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. On CANDIDATE
it is written up with a **shadow-only adoption note** (frozen DOPE, no serving change) for the `flywheel_gp_fused`
owner. This file is merged before any outcome is opened.

## Question

H195 confirmed **BLEND_025 = r882 + 0.25·(GBM_cal − r882)** on H119 Track-2 U4: +0.0137 [+0.0056, +0.0259], Elo
1523.1, 3/3 seeds. Its pooled-A native first-argmin readout was descriptively +0.042 [+0.013, +0.084].
**H135** (COMPLETE, NO_ADVANCE) tested fixed BLEND_025 on the flywheel's own 91-context native protocol: median
−0.0041, CI [−0.029, +0.0099], Elo 1496. With only 19 families, it was underpowered.

**H199 asks: does BLEND_025 lift r882 on the flywheel NATIVE protocol when powered by the 205-family pooled
cross-fit on DL-Train families?** This PR also corrects the H195 README "next step", which wrongly called that
question open; it cites H135.

## Endpoint (primary)

The endpoint is the flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (the merged H139 `native`):
- the **full-pool first argmin** of the arm's cost;
- utility **−fitness when valid, −2.0 when invalid**;
- no U4 clipping, and revealed slots are not privileged.

The runner checks on every scored context that `native` equals H124's first-argmin `legacy` utility. **Exact ties
(identical picks) are zero gain.**

## Arms (no λ search; the dose is report-only)

| Arm | Cost | Role |
|---|---|---|
| R882 | H124 baseline (live, kernel-active), 0 updates | reference (Elo 1500); DL-Val native 91/91 bit-exact first |
| **BLEND_025** | `r882 + 0.25·(GBM_cal − r882)` | **primary (Holm m = 1)** |
| BLEND_015, BLEND_035 | λ = 0.15 and 0.35, same GBM_cal | report-only dose |
| BLEND_025_SHUF | the GBM fit on within-family permuted FIT targets | report-only placebo (seed 199101 only) |

## Design (the H195 machinery, minimal diff)

This wave copies the H195 runner, analysis and monitor. The diff from H195:
- new seeds;
- the native primary, with U4 report-only;
- the λ dose arms;
- the NOKERNEL arms and the H184 dependency removed;
- the B lock hashing the native B panel.

**Seeds and folds.** Seeds **199101–199103**, with fresh independent fold seeds **199141–199143**. None of these
seeds appears elsewhere in the repository; the check is recorded in the PR. Partitions use H124 `folds` on the 205
DL-Train families (1,219 contexts) and are pinned in [fold_manifest.json](fold_manifest.json). Pairwise best-match
Jaccard is 0.16–0.17, which is chance level.

**Model and fitting.** Unchanged from H195 / H129:
- sample seed 199140, 192 programs per context;
- the HistGBM recipe;
- cells A = k, B = k+1, FIT = the other three;
- per seed and type, 5 outer fits (exclude {A, B}) and 10 inner fits (exclude {A, B, j});
- positive-slope calibration on the inner models' out-of-fold FIT predictions;
- SHUF on seed 199101 only, with stream seed `sha256(199160, …)`;
- **60 fits**.

**Stage 1 lock.** `stage1_lock.json` is written exclusively and fsynced **before any A or DL-Val scoring**. It holds
every cell's calibration, the fit prediction hashes and the SHA-256 of the BLEND_025 native B panel. Nothing is
selected on B.

**Stage 2.** Pooled A: each context is scored once per seed with its own cell's outer model and calibration.

## Statistics and decision

For each arm against R882 on pooled-A native utility:
- per-seed equal-family gain and the median over seeds;
- a **10,000-draw paired family-cluster bootstrap** (seed **199150**), with the same family draws for every seed,
  reduced over seeds on axis 0, giving the CI95;
- positive seeds;
- harm: families whose three-seed mean delta is < −0.01;
- −2.0 pick counts;
- the H129 p against +0.001 (report-only);
- canonical Bradley–Terry **Elo against r882 (= 1500)**: 400 iterations, prior 0.01, tie margin 1e-4, family-mean
  win matrices, 1,000 draws, seed 199151.

**Decision (BLEND_025 only, Holm m = 1).** The result is **CANDIDATE: BLEND_025** iff all four gates hold:
1. **LCB95 > 0**;
2. **median > +0.001**;
3. **3/3 seeds positive**;
4. **harm ≤ 10%**.

Anything else is **NO_ADVANCE**.

**Report-only.** U4 (replication of H195, with Elo), the λ 0.15/0.35 dose, SHUF, and −2.0 counts.

**DL-Val flywheel91 (descriptive).** First, **R882 native 91/91 must equal the pinned contract bit-exact (mean
1.1377941898164525)**, or the run stops. Then every arm reports, using the mean calibrated outer-model prior over
its cells:
- the per-context mean native utility (the H135 statistic);
- the equal-family mean;
- U4;
- −2.0 counts.

These figures are compared with H135 descriptively. No gate is applied, because there are 19 families.

## Read boundary and frozen state

- **Reads:** the real-S3-derived DL-Train packet (1,219 contexts, 205 families) and the 91 DL-Val contexts (for
  regression and descriptive use only). This is the H124 pinned packet, `selected.json` `981960ad…`, per
  `S3_DATASET_CATALOG.md`.
- **Never read:** sealed, holdout or meta-test payloads, or the live loop. The r882 checkpoint is a pinned copy
  (`0084479a…`), checked before and after.
- **DOPE stays `c51a56d0`, checked before and after.** There are 0 optimizer updates.

## Host, resources and the single-resume rule

**Host: xbabe1 only for execution and output** (xbabe2 is excluded by the assignment). Inputs are at
`~/jope-data/h199/inputs`, a symlink to the H195 staging. Its hash list (`2f7a3642…`, 2,628 files) is identical to
the H195 preflight record.

**Guards.** The monitor and runner guards are those of H195:
- nice 19;
- OMP 4, with every other pool at 1;
- worker-tree CPU ≤ 4.0 cores;
- RSS < 6 GB;
- MemAvailable ≥ 16 GiB;
- load and host CPU ≤ 85%;
- xbabe1 free ≥ 40 GB;
- root < 500 MB;
- a 4 h deadline.

The H195 run of the same 60 fits took 1 h 44 min with a 170 MB root.

**Single resume.** If and only if the monitor stops the attempt on a resource guard, one `--resume` on the same root
is allowed. It reuses only hash-verified fits and writes `*-resume` stage directories and `stage1_lock-resume.json`.
A second stop closes H199 **INCOMPLETE** with no verdict.

## Custody

The run root goes create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H199/<run_id>/`, with a manifest. The closeout
commits `metrics.json` (< 100 KB), a README (hypothesis, setup, median and CI over 3 seeds, family-grouped
validation, Elo, verdict and next step) and an index row. On CANDIDATE it also adds a shadow-only adoption note for
the `flywheel_gp_fused` owner.
