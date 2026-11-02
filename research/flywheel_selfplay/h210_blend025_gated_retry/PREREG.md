# H210 — H203 retry: BLEND_025 gated by an outcome-free harm signal, xbabe1 CPU, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/gbw_2335_rd_opus.md`. **Claim:**
`~/jope_plan/ns/receipts/H210_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h210-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. On CANDIDATE
it is written up with a **shadow-only adoption note** (frozen DOPE, no serving change) for the `flywheel_gp_fused`
owner. **This prereg merges before anything runs**, and rd_opus does not merge it.

## Why a retry

[H203](../h203_blend025_gated/README.md) asked this exact question and closed **INCOMPLETE with no outcome
computed**. Its run and its single resume ran on xbabe2 by owner deviation (n1918). Both were stopped by one-minute
host-load spikes from other jobs:
- load1 120.9 and 154.6 against a limit of 108.8;
- 15 of 45 fits done, no stage-1 lock, nothing scored.

H210 is the same design with three changes:
1. **Fresh seeds and folds.** H203's single resume is spent, and its fits are not reused.
2. **xbabe1 only**, the host H203 was preregistered for.
3. **A load gate matched to xbabe1** (see Budget).

## Question

H199 found that **BLEND_025 = r882 + 0.25·(GBM_cal − r882)** raises mean native first-argmin utility over r882:
+0.0434 [+0.0097, +0.0903], 3/3 seeds, Elo 1514.0. But it **harms 24.9% of families**, against a 10% cap. The blend
changes the native utility in 31.6% of seed-context outcomes (1,155/3,657; 584/1,219 = 47.9% of contexts in at least
one seed). These are utility changes; H199 recorded no candidate IDs:
- 71 families win by more than 0.01 (mean +0.195);
- 51 families lose by more than 0.01 (mean −0.099).

**H210 asks: can an outcome-free signal decide per context when to trust the blend's switch, so that the gated
blend keeps a native gain with harm ≤ 10%?** When the signal says "risky", the gated arm **falls back to r882's own
pick**.

## Endpoint (as H199 and H203)

The endpoint is the flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (the merged H139 `native`):
- full-pool first argmin;
- utility −fitness when valid, −2.0 when invalid.

The runner checks it on every scored context against H124's first-argmin utility. **Exact ties are zero gain.**

## Gate (outcome-free; registered before any outcome; identical to H203)

**Primary signal: `zgap`.** Let `a` be r882's first argmin and `b` the blend's. Then:
- `zgap = (r882[b] − r882[a]) / std(r882)`;
- `zgap = 0` when the picks agree;
- `zgap = +inf` for a constant r882.

The signal uses **costs only**: no fitness, validity or utility of any candidate, revealed slots included.

**Gate rule.** The gate is open (use the blend) iff `zgap ≤ τ`. Otherwise the arm uses r882's cost vector.

**τ grid.** The grid has seven entries:
- −inf (closed: always r882);
- the FIT-quantiles q10, q25, q50, q75 and q90 of `zgap` among switching FIT contexts;
- +inf (the ungated blend).

**τ selection, FIT folds only, per cell.** Each FIT context is scored using the inner models' out-of-fold calibrated
GBM, the same predictions used for calibration. Selection maximizes the FIT equal-family native gain over r882,
subject to **FIT harm ≤ 5%** (half the A cap). Ties go to the smaller τ. The closed gate (gain 0, harm 0) is always
feasible, so an unsafe cell falls back to r882. **No A, B or DL-Val outcome is used to choose τ.**

**Report-only gate:** `GATED_025_DIS`. This arm uses the same selection rule, with H129's outcome-free
`disagreement` (1 − Spearman(GBM, r882) over the unrevealed slots) in place of `zgap`.

## Arms (λ fixed at 0.25; no λ search)

| Arm | Cost | Role |
|---|---|---|
| R882 | H124 baseline (live, kernel-active), 0 updates | reference (Elo 1500); DL-Val native 91/91 bit-exact first |
| **GATED_025** | blend if `zgap ≤ τ_cell`, else r882 | **primary (Holm m = 1)** |
| BLEND_025 | ungated blend | report-only (replicates H199 on fresh folds) |
| GATED_025_DIS | blend if `disagreement ≤ τ_cell`, else r882 | report-only |

## Design (the merged H203 machinery, minimal diff)

H210 copies the gov-reviewed H203 prereg head (`c2d67cb4`): runner, analysis and monitor, including the
**xbabe1-only host admission**, checked before any side effect. The diff from that head:
- new seeds and folds;
- the load gate (monitor and runner, below);
- the 93.8% disk rule on xbabe1;
- paths and names.

**Seeds and folds.** Seeds **210101–210103**, with fresh fold seeds **210141–210143**, sample seed 210140 and boot
seed 210150. None of these appears elsewhere in the repository (checked with `git grep` against origin/main).
Partitions use H124 `folds` on the 205 DL-Train families (1,219 contexts) and are pinned in
[fold_manifest.json](fold_manifest.json). Mean best-match Jaccard is 0.166 against H203's folds and 0.160 against
H199's, which is chance level.

**Model and fitting.** Unchanged from H195, H199 and H203:
- the H129 HistGBM recipe and 192 programs per context;
- cells A = k, B = k+1, FIT = the other three;
- per seed, 5 outer fits (exclude {A, B}) and 10 inner fits (exclude {A, B, j});
- positive-slope calibration from the inner models' out-of-fold FIT predictions;
- **45 fits**.

**Stage 1 lock.** `stage1_lock.json` is written exclusively and fsynced **before any A or DL-Val scoring**. Per cell
it holds:
- the calibration;
- the chosen τ for both gates, with the full FIT grid (gain, harm and open rate per τ);
- the fit prediction hashes;
- the BLEND_025 native B-panel hash.

**Stage 2.** Pooled A: each context is scored once per seed with its own cell's outer model, calibration and τ.

## Statistics and decision

Each arm is compared against R882 on pooled-A native utility:
- per-seed equal-family gain and the median over seeds;
- a **10,000-draw paired family-cluster bootstrap** (seed 210150), with the same family draws for every seed,
  reduced over seeds on axis 0, giving the CI95;
- the **H129 one-sided bootstrap p against +0.001**;
- harm: families whose three-seed mean delta is < −0.01;
- positive seeds;
- −2.0 pick counts;
- canonical Bradley–Terry **Elo against r882 (= 1500)**: 400 iterations, prior 0.01, tie margin 1e-4, 1,000 draws,
  seed 210151;
- gate open rates (overall and among switching contexts) and per-cell τ, FIT gain and FIT harm.

**Decision (GATED_025 only, Holm m = 1).** The result is **CANDIDATE: GATED_025** iff all three gates hold:
1. **LCB95 > 0**;
2. **p < 0.05**;
3. **harm ≤ 10%**.

Anything else is **NO_ADVANCE**.

**Report-only.** BLEND_025, GATED_025_DIS, U4 and Elo on U4.

**DL-Val flywheel91 (descriptive, 19 families).** First, **R882 native 91/91 must equal the pinned contract bit-exact
(1.1377941898164525)**, or the run stops. Then each of the 15 cells applies its own outer prior and τ, and
per-context utilities are averaged over the cells.

## Read boundary and frozen state

- **Reads:** the real-S3-derived DL-Train packet (1,219 contexts, 205 families; H124 pinned `selected.json`
  `981960ad…`) and the 91 DL-Val contexts (for regression and descriptive use only).
- **Never read:** sealed, holdout or meta-test payloads, or the live loop. The r882 checkpoint is a pinned copy
  (`0084479a…`), checked before and after.
- **DOPE stays `c51a56d0`, checked before and after.** There are 0 optimizer updates.

## Budget, load gate, disk, GPU and host

- **Host: xbabe1 CPU only.** Admission is checked before any read, write or launch, and xbabe2, xbabe0 and xbabe3
  are refused. **No GPU**: HistGBM and the r882 forward run on CPU. xbabe1's RTX 4090 belongs to the live flywheel
  and the H205 claim. **No GPU claim.**
- **Load gate, matched to xbabe1.** xbabe1 has 102 online cores, so 85% is a load of 86.7. On 2026-11-02 between
  02:40 and 06:30Z, its sar load1 ranged from 5 to 34 (mean 11.2).
  - **Admission** (monitor, before the claim): load1 ≤ 50% and load5 ≤ 50% of cores, host CPU ≤ 50%, and disk used
    < 93.5%. A refusal creates no claim or output, so the attempt is not consumed.
  - **Runtime** (monitor, every 0.5 s): stop when **load5 > 85%** of cores, or when **load1 > 85% or host CPU > 85%
    continuously for 300 s**. A dip below the limit restarts the 300 s clock.
  - **Fit boundaries** (runner, before every fit and stage): load5 ≤ 85% of cores.
  - Unchanged and instantaneous: MemAvailable ≥ 16 GiB, worker-tree CPU ≤ 4.0 cores (10 s window, OMP 4, every other
    pool 1), aggregate RSS < 6 GB, nice 19, and a **4 h deadline**.

  Rationale: H210 is nice 19 and capped at 4 cores, so a short spike from other jobs is not caused by it, and H210
  yields to it. Sustained overload still ends the attempt.
- **Disk.**
  - Runtime stop at **free < 40 GB or user-visible used ≥ 93.8%** (lanes_t1615), in both monitor and runner.
  - Admission needs used < 93.5%. xbabe1 was at 93.18% (127 GB free) at prereg.
  - The output root must stay < 500 MB; about 130 MB is expected. Outputs go to `~/jope-data/h210/`.
- **Inputs.** `~/jope-data/h210/inputs` on xbabe1 will be a symlink to the verified H195 staging. Its hash list is
  `2f7a3642…` (2,628 files), and it is checked before launch.
- **Budget.** 45 HistGBM fits at about 2 min each on xbabe1, plus the r882 forwards and the DL-Val regression first.
  The expected wall time is **about 1 h 30 min**.
- **Single resume.** If and only if a resource guard stops the attempt, one `--resume` is allowed. It reuses only
  hash-verified fits. A second stop closes H210 **INCOMPLETE** with no verdict.

## Custody

The prereg files go create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H210/prereg/`. The run root goes to
`.../H210/<run_id>/`, with a manifest. The closeout commits `metrics.json` (< 100 KB), a README (hypothesis, setup,
median and CI over 3 seeds, family-grouped validation, Elo, verdict and next step) and an index row. On CANDIDATE it
also adds a shadow-only adoption note.
