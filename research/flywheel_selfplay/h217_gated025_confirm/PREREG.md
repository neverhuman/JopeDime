# H217 — fresh-fold confirmation of H210 GATED_025 (frozen gate), xbabe2 CPU, 0 updates

**Owner:** rd_opus [OPUS]. **Assignment:** `~/nudges/hw_t0330_opus_h217.md` (H217 reserved for rd_opus). **Claim:**
`~/jope_plan/ns/receipts/H217_claim.txt`. **Lane:** `~/JopeDime/.agent/worktrees/h217-opus`, under the H-wave
exemption at DECISIONS line 73.

This is a diagnostic only. It cannot advance a model, a serving release or a live flywheel checkpoint. DOPE stays
`c51a56d0`, and nothing in the live loop changes. **This prereg merges before anything runs.** Per hw_t0330, rd_opus
squash-merges it after the exact-head checks.

## Why

[H210](../h210_blend025_gated_retry/README.md) found **CANDIDATE: GATED_025**:
- median native gain +0.0140 [+0.0022, +0.0355];
- harm 3.4% (7/205);
- 3/3 seeds positive;
- Elo 1505.6.

Its caveats are what H217 addresses:
- the result comes from one set of fold partitions;
- DL-Val flywheel91 moved only +0.0010 (descriptive);
- U4's CI crossed 0.

**H217 asks: does the frozen GATED_025 recipe, re-run on fresh partitions with no retuning, keep a native gain over
r882 with harm ≤ 10% in every seed? And does it lift the flywheel91 plateau?**

## Frozen recipe (identical to H210; no retuning)

The science is H210's, byte-for-byte. A test compares the function bodies of `zgap`, `disagreement`, `tau_grid`,
`gated`, `choose_tau` and `blend` with H210's.
- **Blend.** BLEND_025 = r882 + 0.25·(GBM_cal − r882) in cost space (H129). λ is fixed.
- **Gate.** `zgap = (r882[b] − r882[a]) / std(r882)`, where `a` is r882's first argmin and `b` the blend's. It is 0
  when the picks agree and +inf for a constant r882. The signal uses costs only. The gate is open iff `zgap ≤ τ`;
  otherwise the arm uses r882's cost vector.
- **τ procedure.**
  - **Grid:** −inf, the FIT-quantiles q10/q25/q50/q75/q90 of `zgap` among switching FIT contexts, and +inf.
  - **Selection:** chosen per cell on FIT folds only, maximizing the FIT equal-family native gain subject to FIT
    harm ≤ 5%. Ties go to the smaller τ.
  - **Not carried over:** no τ value from H210.
  - **No outcome leakage:** no A, B or DL-Val outcome is used to choose τ.
- **Model.** The H129 HistGBM recipe, 192 programs per context and positive-slope calibration from the inner models'
  out-of-fold FIT predictions, as in H195, H199, H203 and H210.
- **Endpoint.** The flywheel protocol `fixed-dev-cost-argmin-negative-fitness/v1` (the merged H139 `native`): full-pool
  first argmin, utility −fitness when valid and −2.0 when invalid. The runner checks it on every scored context
  against H124's first-argmin utility. **Exact ties are zero gain.**

## Arms (no new arm except the H129 placebo)

| Arm | Cost | Role |
|---|---|---|
| R882 | H124 baseline (live, kernel-active), 0 updates | reference (Elo 1500); DL-Val native 91/91 bit-exact first |
| **GATED_025** | blend if `zgap ≤ τ_cell`, else r882 | **primary and co-primary** |
| GATED_025_DIS | blend if H129 `disagreement ≤ τ_cell`, else r882 (same τ rule) | placebo, report-only |
| BLEND_025 | the ungated blend that GATED_025 switches to | component, report-only (not a decision arm) |

## Fresh partitions

- **Seeds and folds.** Seeds **217101–217103**, with fold seeds **217141–217143**, one distinct partition per seed.
  The sample seed is 217140, the boot seed 217150, the Elo seed 217151 and the plateau seed 217152. None appears as a
  number anywhere on origin/main (checked with `git grep -w`).
- **Partitions.** Built with H124 `folds` on the 205 DL-Train families (1,219 contexts), cyclic cells A = k, B = k+1,
  FIT = the other three. They are pinned in [fold_manifest.json](fold_manifest.json).
- **Overlap.**
  - Within H217, pairwise mean best-match Jaccard is 0.149–0.182.
  - Against H210's three partitions, the maximum is 0.183. Against H203's, it is 0.179.
  - Every H217 partition differs from every earlier one, and the code refuses a reused partition.
- **What the seeds share.** The 192-program training subsample per context (sample seed 217140) and the deterministic
  r882 costs are shared by all three seeds, as in H210. The pinned H124 loader's own `split.json` grouping (fold seed
  217141) is recorded but not used for any cell.

## Fits, lock and scoring (as H210)

- **Fits.** Per seed, 5 outer fits (exclude {A, B}) and 10 inner fits (exclude {A, B, j}): **45 fits**.
- **Stage-1 lock.** `stage1_lock.json` is written exclusively and fsynced **before any A or DL-Val scoring**. Per cell
  it holds:
  - the calibration;
  - the chosen τ for both gates, with the full FIT grid;
  - the fit prediction hashes;
  - the BLEND_025 native B-panel hash.
- **Stage 2.** Pooled A: each context is scored once per seed with its own cell's outer model, calibration and τ.
- **DL-Val.** First, **R882 native 91/91 must equal the pinned contract bit-exact (1.1377941898164525)**, or the run
  stops. Then each of the 15 cells applies its own outer prior and τ to the 91 contexts, and per-context utilities are
  averaged over the cells.

## Statistics and decision

**Primary (GATED_025 vs R882, pooled-A native, 205 families).**
- The statistic is the median over the 3 seeds of the per-seed equal-family gain.
- **CI95** comes from a 10,000-draw paired family-cluster bootstrap (seed 217150), with the same family draws for
  every seed, reduced over seeds on axis 0.
- Harm counts families whose three-seed mean delta is < −0.01.
- **CONFIRMED: GATED_025** iff all three hold:
  1. **LCB95 > 0**;
  2. **3/3 seeds positive**;
  3. **harm ≤ 10%**.

  Anything else is **NOT_CONFIRMED**. The H129 one-sided p against +0.001 is reported, not gated.

**Co-primary: the flywheel91 plateau (GATED_025 vs R882 on DL-Val, 91 contexts, 19 families).**
- The statistic is the context-mean native utility gain, the arm's 91-context mean minus r882's 1.1377942.
- **CI95** comes from a 10,000-draw family-cluster bootstrap (seed 217152): families are resampled with replacement,
  and each drawn family's contexts are pooled.
- **PLATEAU_WIN** iff **gain > +0.002 AND LCB95 > 0**. A tie, or a gain at or below +0.002, is **not** a win
  (**NO_PLATEAU_WIN**).

**Combined verdict.** The verdict is reported as `<primary> + <co-primary>`, for example
`CONFIRMED: GATED_025 + NO_PLATEAU_WIN`. The two decisions are separate, each with its own preregistered rule; there
is no multiplicity adjustment between them. CONFIRMED earns a shadow-only note at most. A PLATEAU_WIN claim needs
both.

**Reported, not gated.**
- Elo against r882 (= 1500): canonical Bradley–Terry, 400 iterations, prior 0.01, tie margin 1e-4, 1,000 draws, seed
  217151.
- The same contrasts and the plateau statistic for GATED_025_DIS and BLEND_025.
- U4 contrasts and U4 Elo.
- −2.0 pick counts.
- Gate open rates (overall and among switching contexts).
- Per-cell τ, FIT gain and FIT harm.

Only Elo and utility are reported: no AUC and no speed numbers.

## Read boundary and frozen state

- **Reads.** The real-S3-derived DL-Train packet (1,219 contexts, 205 families; H124 pinned `selected.json`
  `981960ad…`) and the 91 DL-Val contexts (regression and co-primary).
- **Never read.** Sealed, holdout or meta-test payloads, and the live loop. The r882 checkpoint is a pinned copy
  (`0084479a…`), checked before and after.
- **Frozen state.** DOPE stays `c51a56d0`, and its checksum is recorded before and after. There are 0 optimizer
  updates and no adapter changes.

## Host, guards and budget (operational changes from H210)

- **Host: xbabe2 CPU only** (hw_t0330). Admission is checked before any read, write or launch, and xbabe1, xbabe0 and
  xbabe3 are refused.
  - **No GPU, no GPU claim.** HistGBM and the r882 forward run on CPU, and there is no `~/qualification.lock`.
  - Nice 19, OMP 4, every other pool 1, torch 1 thread.
- **Effective CPU allocation (fixes H210 E1).** Every load and CPU limit is sized from `os.sched_getaffinity`: 102
  CPUs on xbabe2, of 128 online. Host CPU is the mean busy share of those CPUs.
- **Admission** (monitor, before the claim). A refusal creates no claim and does not consume the attempt. It needs:
  - load1 ≤ 51 and load5 ≤ 51 (50%);
  - host CPU ≤ 50%;
  - **MemAvailable ≥ 24 GiB**;
  - disk free ≥ 220 GB (amendment A1; originally used < 93.7%).
- **Source binding before the claim (fixes H210 D1).** The monitor runs `runner.py --check-launch`, which checks the
  host, the PREREG hash, the HEAD and the bound blobs at both the reviewed head and the merged commit. It does this
  before writing the claim. A failure there creates no claim.
- **Runtime stops** (monitor, every 0.5 s), on any of:
  - **load5 > 86.7 (85%)**;
  - load1 > 86.7 or host CPU > 85%, either **continuously for 300 s**;
  - **MemAvailable < 20 GiB** (xbabe2 swap is full);
  - disk free < 200 GB (amendment A1; originally free < 40 GB or used ≥ 93.8%);
  - aggregate RSS ≥ 6 GB;
  - worker tree > 4.5 cores over a 30 s window. H210 tripped a 4.0-core, 10 s cap at 4.0136 from OpenMP helper
    threads.

  The runner repeats the load5, memory, disk and deadline checks before every fit and stage.
- **One absolute deadline (fixes H210 E1).** The attempt claim records claim time + 4 h. A resume inherits that
  deadline from the claim and is refused once it has passed. The worker receives it and stops at it. No clock restarts.
- **Disk and inputs.**
  - xbabe2 was at 93.53% used (508 GB free) at prereg, and at 93.92% (477 GB free) when A1 was recorded.
  - The output root must stay < 500 MB; about 180 MB is expected under `~/jope-data/h217/`.
  - `~/jope-data/h217/inputs` is a symlink to the verified H195 staging (hash list `2f7a3642…`, 2,628 files), checked
    before launch.
- **Budget.** 45 HistGBM fits at about 2 min each, plus the r882 forwards and the DL-Val regression first. The
  expected wall time is **about 1 h 30 min to 2 h**.
- **Single resume.** If and only if a resource guard stops the attempt, one `--resume` is allowed, before the
  attempt's absolute deadline. It reuses only hash-verified fits. A second stop, or reaching the deadline, closes H217
  **INCOMPLETE** with no verdict.

## Custody

- **Prereg.** The prereg files go create-only to
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H217/prereg-<head>/`.
- **Run.** The run root goes to `.../H217/<run_id>/`, with a manifest.
- **Closeout.** It commits:
  - `metrics.json` (< 100 KB);
  - a README (hypothesis, setup, outcome, metrics, verdict and next step, with caveats);
  - `custody.json`;
  - an index row.

## Amendment A1 (pre-outcome): disk rule on xbabe2

- **When and who.** Recorded 2026-11-02 at about 11:05Z by the H-wave owner (rd_opus), under hw_t0400. This was
  before any H217 claim, root, read, fit or outcome; no H217 attempt had started.
- **Why.** xbabe2's root is a 7.86 TB disk. At 93.9% used it still has about 477 GB free, so the 93.8% percentage rule
  (written for xbabe1's 1.87 TB disk, where 6.2% is about 116 GB) refused a launch with ample space.
- **Change, xbabe2 only.**
  - The runtime disk stop is **free < 200 GB**, in both the monitor and the runner. It replaces "free < 40 GB or
    used ≥ 93.8%".
  - Admission needs **free ≥ 220 GB**, a 20 GB margin above the stop. It replaces "used < 93.7%".
- **Unchanged.** The 93.8% rule stays for xbabe1, where the shadow logger writes. Nothing else changes: science,
  seeds, folds, arms, decision rules, the load gate, memory, RSS, the core cap, the deadline and the single resume.
