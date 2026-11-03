# H235 — heteroscedastic beta-NLL head for the r882 adapter continuation, xbabe2 CPU

**Status: COMPLETE. Verdict: NO_ADVANCE.** Neither beta-NLL arm beats r882 on DL-Val91. BNLL_MIX is −0.00145
[−0.0033, +0.0001] and BNLL_ADD is −0.00147, both with 0/3 seeds positive. The learned variance barely moved in 40
updates (step-40 log-variance about +0.03), so the down-weighting mechanism was hardly engaged. No model,
checkpoint, DOPE or serving state changed.

## Hypothesis

Noisy contexts may dominate the adapter's regression loss. **H235 (hw_t2300) asks:** does a learned per-candidate
variance, trained with beta-NLL (Seitzer et al. 2022, β = 0.5, stop-gradient on σ^(2β)), down-weight them and lift
DL-Val91 utility over r882 by more than +0.002, at the incumbent's 40-update budget? See [PREREG.md](PREREG.md).

## Setup

- **Prereg.** PR #2070 (`6855922e`), executed at that commit.
- **Variance head.**
  - One `Linear(32, 1)` (33 parameters) on the adapter's per-candidate query embedding, initialized at zero and
    clamped to [−6, 6].
  - Scoring uses the mean output only, so update-0 predictions are r882's bit for bit. A probe confirmed this before
    the run, and the Native91 gate confirmed it in the run.
- **Arms.** Identical H170 CTRL AdamW for all.
  - CONTROL uses the CTRL loss.
  - **BNLL_MIX** is the primary: CTRL's smooth-L1 regression term is replaced by beta-NLL, and the 0.1·KL term is
    kept.
  - BNLL_ADD is the secondary: CTRL + 0.5·beta-NLL. Holm m = 2.
- **Machinery.** Merged H234/H232 with the A1/A2 fixes. Fresh per-seed family folds 235141–235143, seeds
  235101–235103, 15 cells, **1,800 updates**.
- **Decision.** On DL-Val91, an arm ADVANCEs iff its median gain against r882 is > +0.002, its family LCB95 is > 0
  and its Holm p is < 0.05.
- **Frozen state.**
  - Native91 bit-exact (1.1377941898164525).
  - DOPE `c51a56d0` checked after every update.
  - The audit refused 0 opens; there were no sealed, holdout or live-loop reads.

## Outcome (Elo: r882 = 1500)

**Decision panel: DL-Val91, selected policies.**

| Contrast | Median gain | Per-seed | CI95 | Seeds > 0 | Holm p | Elo [CI95] |
|---|---:|---|---|---:|---:|---|
| **BNLL_MIX − r882 (primary)** | **−0.00145** | 0 / −0.00145 / −0.00163 | **[−0.00326, +0.00008]** | 0/3 | 1.00 | 1489.0 [1463.3, 1507.3] |
| BNLL_ADD − r882 | −0.00147 | 0 / −0.00147 / −0.00176 | [−0.00330, 0] | 0/3 | 1.00 | 1485.4 [1463.3, 1500.0] |
| CONTROL − r882 | 0.00000 | 0 / 0 / −0.00050 | [0, 0] | 0/3 | — | 1500.0 |
| BNLL_MIX − CONTROL | −0.00114 | | [−0.00259, 0] | 0/3 | — | |

Both arms are **NO_ADVANCE**: the gain is negative, the LCB is not above 0 and Holm p is 1.0.

**Against the plateau best (1.1389572, r3175; descriptive).** The median DL-Val91 context means are BNLL_MIX
1.13628 (−0.00268), BNLL_ADD 1.13626 (−0.00270) and CONTROL/r882 1.13779 (−0.00116).

**Family out-of-fold panel (report-only).** BNLL_MIX −0.00120 and BNLL_ADD −0.00120, with 0/3 seeds positive.

**Harm** (families whose three-seed mean delta is < −0.01):

| Panel | CONTROL | BNLL_MIX | BNLL_ADD |
|---|---:|---:|---:|
| DL-Val, selected | 0/19 | 0/19 | 0/19 |
| OOF, selected | 2/205 | 9/205 | 14/205 |
| DL-Val, unconstrained | 3/19 | 3/19 | 3/19 |

**Selector behaviour.**
- The strict selector chose a trained step in only 5 of 45 arm-cells:
  - BNLL_MIX: step 20 in A0-235103 and step 10 in A2-235102;
  - BNLL_ADD: step 40 in A0-235103 and step 10 in A2-235102;
  - CONTROL: step 10 in A4-235103.
- Those beta-NLL picks drive the negative DL-Val medians.

**Descriptive unconstrained best (B-best step, no gates; cannot advance).**

| Contrast | DL-Val gain [CI95] | DL-Val Elo |
|---|---|---:|
| CONTROL − r882 | −0.0106 [−0.0271, +0.0002] | 1488.1 |
| BNLL_MIX − r882 | −0.0032 [−0.0083, +0.0006] | 1497.3 |
| BNLL_ADD − r882 | −0.0034 [−0.0087, +0.0005] | 1482.6 |
| BNLL_MIX − CONTROL | +0.0066 [+0.0003, +0.0179] | |

**Variance head and drift.**
- Step-40 log-variance averaged +0.027 for BNLL_MIX and +0.027 for BNLL_ADD over 15 cells, with a range of −0.003
  to +0.038. So σ² stayed within about 4% of 1.
- Median adapter drift was CONTROL 0.375, BNLL_MIX 0.268 and BNLL_ADD 0.279.

## Interpretation (red-team per hw_t2200)

- **The primary contrast is against r882, and it fails.** Both beta-NLL arms are slightly negative, and the arms they
  selected harm more OOF families than CONTROL (9 and 14, against 2).
- **The hypothesis was barely tested.** In 40 updates at lr 1e-4, the zero-initialized head moved σ² by about 3%. So
  BNLL_MIX behaved essentially as CTRL with its Huber term replaced by half squared error, not as a variance-weighted
  loss.
- **The BNLL edge over CONTROL is descriptive only.** It exists only in the unconstrained readout (+0.0066), and comes
  from CONTROL's larger degradation (−0.0106) and lower drift, not from any lift over r882.
- **Same pattern as H170, H197 and H232.** At this budget, every continuation recipe from r882 either falls back to
  r882 or loses to it.

## Verdict

**NO_ADVANCE** for BNLL_MIX and BNLL_ADD on the DL-Val91 decision panel. It is a diagnostic: 1,800 updates on a
read-only r882 copy, DOPE frozen, no serving promotion.

## Next step

- **A fair test of heteroscedastic weighting needs the variance to actually learn.** Options include a higher head
  learning rate, or warm-starting the head on r882's residuals before the continuation, within a preregistered
  budget. H235 as run does not show that variance weighting helps or hurts.
- **Loss-shape changes do not beat r882 at 40 updates.** That covers LambdaLoss (H234, not run to completion) and
  beta-NLL; together with the optimizer swaps (H170/H232), the evidence points away from loss and optimizer tweaks of
  the adapter continuation.

## Run record

- **Host.** xbabe2 CPU, unit `h235-run`, taskset cores 40–43 (avoiding H231 and H233), nice 19, 1 math thread.
- **Smokes.** The pre-merge smoke passed after one admission refusal at load5 56.9; the merged-head smoke passed.
- **Attempt.** Admitted 23:21:38Z (load5 44.9, MemAvailable 82 GB), PASS at 23:56:14Z: 2,032 s, 0 load pauses, peak
  RSS 3.37 GB, peak 3 threads.
- **Single attempt, no retry.**
- **Prereg custody.** The 12 prereg objects uploaded, but the helper's readback failed on a missing temp directory
  after the host reboot. All 12 were re-verified by version-pinned readback, and `s3_manifest-v2.json` supersedes the
  empty first manifest; nothing was overwritten.

## Custody

- **S3 (create-only, version-pinned readback):**
  `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H235/attempt-001/`. It holds the archive of
  the run root (including `*-varhead.npz`), logs, smokes, launch files, claim and probe; the full metrics; and
  `s3_manifest.json`.
- **Prereg.** `.../H235/prereg-d1ecc309/` (see `s3_manifest-v2.json`).
- **In git.** [metrics.json](metrics.json) is a compact extract that names the full file's SHA-256.
- **Receipts.** `~/jope_plan/ns/receipts/H235_claim.txt`.

**Code.** [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py), [monitor.py](monitor.py)
and [test_h235.py](test_h235.py) (17 data-free tests), executed at `6855922e`.
