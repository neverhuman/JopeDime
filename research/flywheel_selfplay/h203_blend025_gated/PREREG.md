# H203 — BLEND_025 gated by an outcome-free harm signal, flywheel native protocol, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/n1615_rd_opus.md`, which follows `~/nudges/lanes_t1615.md`.
**Claim:** `~/jope_plan/ns/receipts/H203_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h203-opus`, under the
H-wave exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. On CANDIDATE
it is written up with a **shadow-only adoption note** (frozen DOPE, no serving change) for the `flywheel_gp_fused`
owner. **This prereg merges before anything runs**, and rd_opus does not merge it.

## Question

H199 found that **BLEND_025 = r882 + 0.25·(GBM_cal − r882)** raises mean native first-argmin utility over r882:
+0.0434 [+0.0097, +0.0903], 3/3 seeds, Elo 1514.0. But it **harms 24.9% of families**, against a 10% cap. The blend
changes the native utility in 31.6% of seed-context outcomes (1,155/3,657; 584/1,219 = 47.9% of contexts in at least
one seed). These are utility changes; H199 recorded no candidate IDs:
- 71 families win by more than 0.01 (mean +0.195);
- 51 families lose by more than 0.01 (mean −0.099).

Harm rises with λ, and no fixed λ meets the cap.

**H203 asks: can an outcome-free signal decide per context when to trust the blend's switch, so that the gated
blend keeps a native gain with harm ≤ 10%?** When the signal says "risky", the gated arm **falls back to r882's own
pick**.

## Endpoint (as H199)

The endpoint is the flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (the merged H139 `native`):
- full-pool first argmin;
- utility −fitness when valid, −2.0 when invalid.

The runner checks it on every scored context against H124's first-argmin utility. **Exact ties are zero gain.**

## Gate (outcome-free; registered before any outcome)

**Primary signal: `zgap`.** This measures how strongly r882 disfavours the blend's pick. Let `a` be r882's first
argmin and `b` the blend's. Then:
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

## Design (the H199 machinery, minimal diff)

This wave copies the H199 runner, analysis and monitor. That includes the reviewed **xbabe1-only host admission**,
checked before any side effect. The diff from H199:
- new seeds;
- 45 fits, with no SHUF;
- λ = 0.25 only;
- the gate statistics, FIT-only τ selection and gated arms;
- an inf-safe JSON encoder for τ = ±inf.

**Seeds and folds.** Seeds **203101–203103**, with fresh fold seeds **203141–203143**, sample seed 203140 and boot
seed 203150. None of these appears elsewhere in the repository (checked with `git grep`). Partitions use H124
`folds` on the 205 DL-Train families (1,219 contexts) and are pinned in [fold_manifest.json](fold_manifest.json).
Pairwise best-match Jaccard is about 0.15–0.17, which is chance level.

**Model and fitting.** Unchanged from H195 and H199:
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
- a **10,000-draw paired family-cluster bootstrap** (seed 203150), with the same family draws for every seed,
  reduced over seeds on axis 0, giving the CI95;
- the **H129 one-sided bootstrap p against +0.001**;
- harm: families whose three-seed mean delta is < −0.01;
- positive seeds;
- −2.0 pick counts;
- canonical Bradley–Terry **Elo against r882 (= 1500)**: 400 iterations, prior 0.01, tie margin 1e-4, 1,000 draws,
  seed 203151;
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

## Budget, disk, GPU and host

- **Host: xbabe1 CPU only.** Admission is checked before any read, write or launch, and xbabe2, xbabe0 and xbabe3
  are refused.
- **No GPU needed.** HistGBM is CPU-only and the r882 forward runs on CPU, as in H195 and H199. Under lanes_t1615,
  the xbabe1 RTX 4090 is held by the live flywheel anyway. **No GPU claim.**
- **Budget.** The 45 HistGBM fits take about 2 min each, with the r882 forwards and the DL-Val regression first.
  The expected wall time is **about 1 h 30 min** (H199: 60 fits in 1 h 52 min).
- **Guards:**
  - nice 19;
  - OMP 4, with every other pool at 1;
  - worker-tree CPU ≤ 4.0 cores;
  - RSS < 6 GB;
  - MemAvailable ≥ 16 GiB;
  - load and host CPU ≤ 85%;
  - a **4 h deadline**.
- **Disk.** The guard is xbabe1 free ≥ 40 GB (90 GB free at prereg). The output root must stay < 500 MB; about
  130 MB is expected, since H199 used 171 MB for 60 fits. Outputs go to `~/jope-data/h203/`.
- **Inputs.** `~/jope-data/h203/inputs` on xbabe1 is a symlink to the verified H195 staging. Its hash list is
  `2f7a3642…` (2,628 files).
- **Single resume.** If and only if a resource guard stops the attempt, one `--resume` is allowed. It reuses only
  hash-verified fits. A second stop closes H203 **INCOMPLETE** with no verdict.

## Custody

The run root goes create-only to
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H203/<run_id>/`, with a manifest. The closeout
commits `metrics.json` (< 100 KB), a README (hypothesis, setup, median and CI over 3 seeds, family-grouped validation,
Elo, verdict and next step) and an index row. On CANDIDATE it also adds a shadow-only adoption note.
