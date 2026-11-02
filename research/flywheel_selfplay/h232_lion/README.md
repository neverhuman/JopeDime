# H232 — Lion optimizer for the r882 adapter continuation, xbabe1 CPU

**Status: COMPLETE. Verdict: NO_ADVANCE.** Neither Lion arm beats r882 on DL-Val91. LION is −0.0005
[−0.0018, +0.0004] with 0/3 seeds positive, and LION_LR3 is 0. The AdamW CONTROL fell back to r882 in all 15 cells.
Lion's only advantage is descriptive: in the unconstrained readout it degrades r882 less than CONTROL does. No model,
checkpoint, DOPE or serving state changed.

## Hypothesis

No earlier H-wave tried a sign-momentum optimizer for the adapter. **H232 (hw_t2120) asks:** at the incumbent's
40-update budget, does Lion (Chen et al. 2023; sign of interpolated momentum, decoupled weight decay) find adapter
updates that lift DL-Val91 utility over r882 by more than +0.002? See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #2059 (`cf1e5788`), plus two pre-outcome amendments. Both came from constructor smokes that stopped
  before any data read:
  - A1 (#2060) re-pinned the H146 runtime after the #2006 refactor;
  - A2 (#2061) released the early `research` namespace binding that #2006 introduced.

  The run executed `2b1f0942`.
- **Arms.**
  - CONTROL is the exact merged H170 CTRL recipe: AdamW, lr 1e-4, wd 1e-4, betas (0.9, 0.999).
  - **LION** is the primary: lr 1e-5, betas (0.9, 0.99), decoupled wd 3e-4.
  - LION_LR3: lr 3.33e-5.
  - All settings are constant over the 40 updates. Holm m = 2 over the Lion arms.
- **Machinery.** Merged H170, unchanged except the optimizer:
  - 1,219 DL-Train contexts in 205 families, 5 family folds (fold seed 232140);
  - seeds 232101–232103, 15 cells, **1,800 updates**;
  - H156 batches and loss, clipping at 1.0;
  - the strict inner selector (earliest best of steps 0/10/20/30/40; gain > .001 and bootstrap LCB > 0, else r882);
  - locks fsynced before assessment.
- **Decision panel.** DL-Val91 (91 contexts, 19 families): each seed averages its five selected policies. An arm
  ADVANCEs iff the median seed gain against r882 is > +0.002, the family LCB95 is > 0 and Holm p < 0.05.
- **Frozen state.**
  - Native91 r882 was bit-exact first (1.1377941898164525).
  - Only the adapter trains: 47 tensors, 29,211 parameters, on a read-only r882 copy (`0084479a…`).
  - DOPE `c51a56d0` was checked after every update, and its asset files were re-hashed OK after the run.
  - The audit refused 0 opens; there were no sealed, holdout or live-loop reads.

## Outcome (Elo: r882 = 1500)

**Decision panel: DL-Val91, selected policies.**

| Contrast | Median gain | Per-seed | CI95 | Seeds > 0 | Holm p | Elo [CI95] |
|---|---:|---|---|---:|---:|---|
| **LION − r882 (primary)** | **−0.00049** | −0.00050 / 0 / −0.00049 | **[−0.00184, +0.00038]** | 0/3 | 1.00 | 1499.1 [1485.4, 1507.3] |
| LION_LR3 − r882 | 0.00000 | −0.00025 / 0 / 0 | [0, 0] | 0/3 | 1.00 | 1500.0 |
| CONTROL − r882 | 0.00000 | 0 / 0 / 0 | [0, 0] | 0/3 | — | 1500.0 |
| LION − CONTROL (descriptive) | −0.00049 | | [−0.00184, +0.00038] | 0/3 | — | |

Both Lion arms are **NO_ADVANCE**: the gain is not above +0.002, the LCB is not above 0 and Holm p is 1.0.

**Family out-of-fold panel (1,219 contexts, 205 families, report-only).** LION −0.00027 [−0.00067, 0], Elo 1498.6.
LION_LR3 and CONTROL are both 0.

**Harm** (families whose three-seed mean delta is < −0.01):

| Panel | CONTROL | LION | LION_LR3 |
|---|---:|---:|---:|
| DL-Val, selected | 0/19 | 0/19 | 0/19 |
| OOF, selected | 0/205 | 2/205 | 0/205 |
| DL-Val, unconstrained | 4/19 | 1/19 | 4/19 |

**Selector behaviour.**
- The strict selector kept r882 in almost every cell. It chose a trained step only 3 times for LION (step 40 in
  cells A1-232101, A1-232103 and A3-232101) and once for LION_LR3 (step 10 in A3-232101). It never chose one for
  CONTROL.
- The best B-step was nonzero in 8 cells for CONTROL, 11 for LION and 13 for LION_LR3, but mostly below the
  acceptance gate.

**Descriptive unconstrained best (B-best step, no gates; cannot advance).**

| Contrast | DL-Val gain [CI95] | OOF gain [CI95] | DL-Val Elo |
|---|---|---|---:|
| CONTROL − r882 | −0.0043 [−0.0107, +0.0004] | −0.0004 [−0.0127, +0.0155] | 1491.8 |
| LION − r882 | −0.0018 [−0.0056, +0.0006] | +0.0002 [−0.0005, +0.0008] | 1498.2 |
| LION_LR3 − r882 | −0.0103 [−0.0256, −0.0005] | −0.0022 [−0.0136, +0.0124] | 1488.1 |
| LION − CONTROL | +0.0034 [−0.0011, +0.0096] | +0.0006 [−0.0153, +0.0128] | |

**Drift.** Median step-40 parameter L2 drift is CONTROL 0.376, LION 0.049 and LION_LR3 0.161. At lr 1e-5, Lion
moves the adapter about 8× less than AdamW.

## Interpretation (red-team per hw_t2200)

- **The primary contrast is LION against r882, and it fails.** The median is slightly negative and no seed is
  positive.
- **LION's apparent edge over CONTROL is descriptive only and does not advance.** In the unconstrained readout LION
  is +0.0034 over CONTROL, because the AdamW continuation degrades r882 (−0.0043 on DL-Val, consistent with
  H219/H220/H228), while Lion at lr/10 barely moves the adapter.
- **Lion at lr/3 is clearly worse.** LION_LR3's unconstrained DL-Val is −0.0103, with a CI that excludes 0.
- **The pattern matches H170.** Under the strict selector, every continuation recipe from r882 collapses to r882 at
  this budget. A smaller, sign-based step does not find a better adapter; it only does less harm.

## Verdict

**NO_ADVANCE** for LION and LION_LR3 on the DL-Val91 decision panel. It is a diagnostic: 1,800 optimizer updates
on a read-only r882 copy, DOPE frozen, no serving promotion.

## Next step

- **Drop the optimizer-swap line for the r882 adapter continuation.** AdamW, one-cycle (H170), clipping (H197) and
  now Lion all fail to beat r882 at 40 updates.
- **Look beyond the update rule for a lift.** It needs a different signal or data (for example, new families, or
  the outcome-free gate line H210/H217/H223), not a different optimizer.
- **If Lion is revisited,** it should be at lr ≤ 1e-5 with a longer budget. Only its low drift kept harm low.

## Run record

- **Host.** xbabe1 CPU, unit `h232-run`, taskset cores 44–47, nice 19, 1 math thread.
- **Smokes.** Smokes 001 and 002 stopped before any read (A1 and A2, above). Smoke-003 passed.
- **Attempt.** Admitted 21:40:00Z at load5 6.5, PASS at 22:08:44Z: 1,719 s, 0 load pauses, peak RSS 3.36 GB,
  peak 3 threads.
- **Memory guard.** MemAvailable bottomed at 20.44 GiB. The guard (< 20 GiB) is a hard stop of the worker's own
  process group only; hw_t2200 asked for a pause, which could not be applied mid-run without a restart. It did not
  fire.
- **Single attempt, no retry.**

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H232/attempt-001/`:
  - the archive of the run root, logs, smokes, launch files, claim and probe;
  - the full 232 KB `metrics.json`;
  - `s3_manifest.json`.
- **Prereg.** `.../H232/prereg-57d56c21/`.
- **In git.** [metrics.json](metrics.json) is a compact extract that names the full file's SHA-256.
- **Receipts.** `~/jope_plan/ns/receipts/H232_claim.txt`.

**Code.** [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py), [monitor.py](monitor.py)
and [test_h232.py](test_h232.py) (14 data-free tests), executed at `2b1f0942`.
