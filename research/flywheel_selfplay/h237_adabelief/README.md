# H237 — AdaBelief (and RAdam) for the r882 adapter continuation, xbabe2 CPU (amendment A1)

**Status: COMPLETE. Verdict: NO_ADVANCE.** Neither AdaBelief nor RAdam beats r882 on DL-Val91. The strict selector
kept r882 in every ABEL and CONTROL cell, and in 14 of 15 RADAM cells, so every median gain is exactly 0 (Elo 1500).
No model, checkpoint, DOPE or serving state changed.

## Hypothesis

**H237 (hw_t0030) asks: is the plateau partly an adaptive-step-size artefact?** AdaBelief (Zhuang et al. 2020)
scales each step by the variance of the gradient around its EMA (the "belief") instead of the raw second moment. At
the incumbent's 40-update budget, does that lift DL-Val91 utility over r882 by more than +0.002? RAdam (Liu et al.
2020) is the secondary. See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #2074 (`4fe28441`), including host amendment A1 (see Deviations).
- **Arms.** Only the optimizer differs. All arms share the CTRL loss, H156 batches, lr 1e-4, betas (0.9, 0.999),
  decoupled wd 1e-4, clipping at 1.0 and 40 constant updates.
  - CONTROL: AdamW (exact H170 CTRL).
  - **ABEL** (primary): inline AdaBelief with ε = 1e-16 inside s, bias correction and no rectification.
  - RADAM (secondary): `torch.optim.RAdam` with decoupled wd. Holm m = 2.
- **Machinery.** Merged H234/H232 with the A1/A2 fixes. Fresh per-seed family folds 237141–237143, seeds
  237101–237103, 15 cells, **1,800 updates**.
- **Decision.** On DL-Val91, an arm ADVANCEs iff its median gain against r882 is > +0.002, its family LCB95 is > 0
  and its Holm p is < 0.05.
- **Frozen state.**
  - Native91 bit-exact (1.1377941898164525).
  - Only the adapter trains: 47 tensors, 29,211 parameters.
  - DOPE `c51a56d0` checked after every update.
  - The audit refused 0 opens; there were no sealed, holdout or live-loop reads.

## Outcome (Elo: r882 = 1500)

**Decision panel: DL-Val91, selected policies.**

| Contrast | Median gain | Per-seed | CI95 | Seeds > 0 | Holm p | Elo |
|---|---:|---|---|---:|---:|---:|
| **ABEL − r882 (primary)** | **0.00000** | 0 / 0 / 0 | **[0, 0]** | 0/3 | 1.00 | 1500.0 |
| RADAM − r882 | 0.00000 | 0 / 0 / −0.00049 | [0, 0] | 0/3 | 1.00 | 1500.0 |
| CONTROL − r882 | 0.00000 | 0 / 0 / 0 | [0, 0] | 0/3 | — | 1500.0 |

Both arms are **NO_ADVANCE**: a tie is no advance. Against the plateau best (1.1389572, r3175), every arm's DL-Val91
context mean is r882's 1.13779 (−0.00116). The TRAIN out-of-fold panel (1,219 contexts, 205 families) is also exactly
0 for every arm.

**Harm** (families whose three-seed mean delta is < −0.01):

| Panel | CONTROL | ABEL | RADAM |
|---|---:|---:|---:|
| Selected (DL-Val and OOF) | 0 | 0 | 0 |
| DL-Val, unconstrained | 3/19 | 2/19 | 1/19 |

**Selector behaviour.**
- The only trained-step selection in 45 arm-cells was RADAM at step 40 in A2-237103.
- The best B-step was nonzero in 7 cells for CONTROL, 6 for ABEL and 13 for RADAM, but never cleared the gate (gain
  > .001 with a bootstrap LCB > 0) except that one RADAM cell.

**Descriptive unconstrained best (B-best step, no gates; cannot advance).**

| Contrast | DL-Val gain [CI95] | DL-Val Elo |
|---|---|---:|
| CONTROL − r882 | −0.0054 [−0.0127, +0.0001] | 1479.9 |
| ABEL − r882 | −0.0022 [−0.0056, +0.0006] | 1499.1 |
| RADAM − r882 | −0.0017 [−0.0056, +0.0003] | 1492.7 |
| ABEL − CONTROL | +0.0013 [+0.0002, +0.0032] | |
| RADAM − CONTROL | +0.0031 [−0.0015, +0.0104] | |

**Drift.** Median step-40 parameter L2 drift is CONTROL 0.375, ABEL 0.586 and RADAM 0.028.
- **AdaBelief moves the adapter about 1.6× further than AdamW,** as expected: consistent gradients get larger steps.
- **RAdam barely moves it.** Rectification makes its early steps un-adapted momentum updates at lr 1e-4.

## Interpretation (red-team per hw_t2200)

- **The primary contrast is against r882, and it is an exact tie.** The selector found no ABEL step that reliably
  beats r882 on the held-out B families.
- **AdaBelief's larger steps did not find better adapters.** Its best-step readout is still below r882 (−0.0022).
  It is above CONTROL (+0.0013, CI above 0), but only because AdamW degrades r882 more (−0.0054). That is
  descriptive, not an advance.
- **RAdam's small drift explains its near-r882 behaviour.** It is not evidence of a better optimizer.
- **The pattern holds.** AdamW, one-cycle, clipping, Lion, AdaBelief and RAdam all fail to beat r882 at 40 updates.
  The adapter continuation from r882 is not limited by its adaptive step-size rule.

## Verdict

**NO_ADVANCE** for ABEL and RADAM on the DL-Val91 decision panel. It is a diagnostic: 1,800 updates on a read-only
r882 copy, DOPE frozen, no serving promotion.

## Next step

- **Stop the optimizer line for this continuation.** Six optimizer variants (H170, H197, H232, H237) all tie with or
  lose to r882.
- **A lift needs new signal.** That means new data or families, or a better base, not a different update rule.

## Deviations

- **Amendment A1 (pre-outcome, host only; `hw_t0036_opus_h237_a1_host.md`, sha256 `2d2cb87a…`).** The assignment
  named xbabe1, but its MemAvailable was 16–17 GiB, below the 22 GiB admission gate. The xbabe1 pre-merge smoke was
  refused at admission and never ran.
  - The wave moved to xbabe2, `taskset` 48–51, with load5 pause > 108 and resume ≤ 100, and free ≥ 300 GB.
  - Inputs were re-pinned to a hash-identical r882 copy.
  - Recorded in the PREREG and the claim receipt before any run. No science change.

## Run record

- **Host.** xbabe2 CPU, unit `h237-run`, nice 19, 1 math thread.
- **Smokes.** Pre-merge `smoke-001` and merged-head `smoke-002` both passed.
- **Attempt.** Admitted 00:38:43Z (load5 54.6, MemAvailable 73.7 GB, free 778 GB), PASS at 01:23:18Z: 2,666 s, peak
  RSS 3.35 GB, peak 3 threads.
- **The new memory guard worked.** MemAvailable twice dipped to about 19.7 GiB (load5 73–76). The monitor paused only
  its own process group, for 24 s in total, then resumed at ≥ 22 GiB. H234's hard 20 GiB rule would have ended the
  run there.
- **Single attempt, no retry.**

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H237/attempt-001/`. It holds the archive of
  the run root, logs, smokes, launch files and claim; the full metrics; and `s3_manifest.json`.
- **Prereg.** `.../H237/prereg-26fb142e/`.
- **In git.** [metrics.json](metrics.json) is a compact extract that names the full file's SHA-256.
- **Receipts.** `~/jope_plan/ns/receipts/H237_claim.txt`.

**Code.** [runner.py](runner.py), [treatment.py](treatment.py) (AdaBelief, RAdam), [analysis.py](analysis.py),
[monitor.py](monitor.py) and [test_h237.py](test_h237.py) (15 data-free tests), executed at `4fe28441`.
