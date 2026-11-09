# H260: recall@4 objectives for the r882 adapter continuation

**Status: complete. Verdict: NO_ADVANCE.** The rule was preregistered in [PREREG.md](PREREG.md) and published on
`origin/v1/next` as `7438c10d` at 15:39:02Z (PR #2071 split; same-agent review receipt sha256 `6f6aaa44…`) before
anything ran. [AMENDMENT-A1](AMENDMENT-A1.md) (`feacfff4`, watcher decision before any assessment) moved only the
deadlines: the compute stop to 18:45Z and the closeout to 19:30Z. The single attempt ran on xbabe2 CPU from 15:40:25Z to
16:31:25Z, inside the original 17:15Z stop that the running guard still enforced: **1,800 optimizer updates** over
15 cells × 3 arms × 40.

**Training for recall@4 does not lift TOP4_PICK.**
- **The decision.** On DL-Val91 TOP4_PICK, both objectives tie or lose against TOP4_PICK@r882 (1.2403601682250796,
  reproduced exactly):
  - T4_HINGE − r882 = −0.00013 [−0.0005, 0];
  - T4_PL − r882 = 0 [0, 0];
  - 0/3 seeds positive, Holm p 1.0 for both.

  The single-pick guard passes (≥ −0.002), but the TOP4 gates fail.
- **Recall@4 did not move.** r882 puts an oracle-best candidate in its predicted top-4 on 6.6% of DL-Val91 contexts and
  0.9% of TRAIN contexts. After 40 updates every arm is at 4.0–4.6% on DL-Val91 and 1.0–1.2% on TRAIN; the recall
  objectives are no better than CONTROL.
- **The aux losses mostly act as brakes.** At step 40, both objectives lose TOP4 utility like CONTROL does (about −0.011).
  They lose less single-pick utility: T4_PL − CONTROL = +0.018 [+0.002, +0.041] (3/3 seeds), and their drift from r882
  is lower (0.32 and 0.35 against 0.37).

## Hypothesis

**What is known.** H258 showed that the deployment-relevant readout is best-of-revealed: TOP4_PICK@r882 = 1.2404 on
DL-Val91 against 1.1378 for single-pick r882. Every adapter continuation so far was trained and selected for the top-1
argmin.

**The question.** Does training the adapter so the oracle-best candidate lands in its predicted top-4 lift DL-Val91
TOP4_PICK over TOP4_PICK@r882 by more than +0.002, at the same 4-reveal budget?

## Setup

- **Arms.** All use the exact H170 CTRL AdamW, batches, data order, clip 1.0 and 40 updates from a read-only r882 copy.
  Constants were frozen before any outcome:
  - **CONTROL:** exact H170 CTRL loss.
  - **T4_HINGE (primary):** CTRL + 0.5 · mean relu(ĉ(oracle best) − ĉ(predicted rank 4) + m), with m = 0.05 × the std of
    r882's predicted costs over the cell's FIT-fold candidates (m = 1.38–1.52 across cells).
  - **T4_PL (secondary):** CTRL + 0.5 · −log of the exact Plackett–Luce probability that the oracle best is among the
    first 4 draws from softmax(−ĉ), τ = 1.
  - **Oracle labels:** the minimum gated cost among each minibatch context's 24 candidates, from FIT-fold contexts only.
  - **Hypothesis family.** T4_HINGE and T4_PL against r882, with Holm m = 2.
- **Readout.** The exact H258 TOP4_PICK: reveal the 4 lowest-cost candidates when top_zgap < 0.25 and pick the best
  revealed gated cost, else pick argmin cost.
- **Design.**
  - **Folds and cells.** One fresh fold seed 260140, seeds 260101–260103, 15 cells per arm, merged H256/H247
    machinery.
  - **Selector.** The strict inner selector on SELECT TOP4_PICK (gain > .001 and family LCB95 > 0, else r882),
    locked before assessment.
  - **Decision.** On DL-Val91 TOP4_PICK: median gain > +0.002, LCB95 > 0, Holm p < .05, 3/3 seeds positive, and
    single-pick within −0.002.
- **Checks.**
  - Native91 single-pick was bit-exact (1.1377941898164525), and TOP4_PICK@r882 reproduced H258's 1.2403601682250796
    exactly before any TRAIN outcome opened.
  - DOPE `c51a56d0` was unchanged before and after.
- **Resources.** xbabe2 cores 44–47, nice 19; peak RSS 3.36 GB, 3 threads; 3,062 s with no pause and no resume.

## Outcome

### Decision panel (DL-Val91 TOP4_PICK; Elo with TOP4_PICK@r882 = 1500)

| Contrast | Per-seed gain | Median [CI95] | Seeds > 0 | Holm p | Elo [CI95] | Single-pick median gain |
|---|---|---|---:|---:|---|---|
| **T4_HINGE − r882** | −0.00013, −0.00183, 0 | **−0.00013 [−0.0005, 0]** | 0/3 | 1.0 | 1495 [1480, 1500] | −0.00002 (guard passes) |
| **T4_PL − r882** | 0, −0.00046, 0 | **0.0000 [0, 0]** | 0/3 | 1.0 | 1500 | 0 (guard passes) |
| CONTROL − r882 | −0.00039, −0.00056, 0 | −0.00039 [−0.0008, 0] | 0/3 | — | 1483 [1465, 1500] | −0.00057 |
| T4_HINGE − CONTROL | +0.00026, −0.00127, 0 | 0.0000 [−0.0004, 0] | 1/3 | — | — | — |
| T4_PL − CONTROL | +0.00039, +0.00009, 0 | +0.00009 [0, +0.0003] | 2/3 | — | — | — |

- **Selections.** The strict selector kept r882 in 40 of 45 decisions. It accepted step 10 for CONTROL in 2 cells,
  steps 10 and 30 for T4_HINGE in 2 cells, and step 10 for T4_PL in 1 cell, with SELECT gains of +0.005 to +0.007 and
  LCB ≤ +0.0009. Each accepted cell scored at or below TOP4_PICK@r882 on DL-Val91. Seed 260103 kept r882 everywhere.
- **Strict TRAIN out-of-fold** (1,219 contexts): TOP4_PICK T4_HINGE and T4_PL − r882 = 0 (CI within ±0.0014);
  single-pick T4_HINGE −0.0041 [−0.0113, 0], T4_PL 0.
- **References.** TOP4_PICK@r3175 = 1.2403226556046079 (H258, not recomputed). The family-weighted TOP4_PICK@r882 is
  1.252000.

### Recall@4 of the oracle best (share of contexts)

| Policy | DL-Val91 strict | DL-Val91 step 40 | TRAIN OOF strict | TRAIN OOF step 40 |
|---|---|---|---|---|
| r882 | 0.066 | 0.066 | 0.009 | 0.009 |
| CONTROL | 0.062–0.066 | 0.040 | 0.009 | 0.010–0.011 |
| T4_HINGE | 0.064–0.066 | 0.044–0.046 | 0.009 | 0.010–0.011 |
| T4_PL | 0.062–0.066 | 0.046 | 0.009 | 0.011–0.012 |

### Unconstrained diagnostics (descriptive; before any strict fallback)

| Readout | CONTROL − r882 | T4_HINGE − r882 | T4_PL − r882 | T4_HINGE − CONTROL | T4_PL − CONTROL |
|---|---|---|---|---|---|
| **Step 40, DL-Val91 TOP4_PICK** | −0.0111 [−0.0167, −0.0052] | −0.0113 [−0.0184, −0.0039] | −0.0107 [−0.0178, −0.0042] | −0.0002 [−0.0043, +0.0031], 1/3 | +0.0002 [−0.0038, +0.0036], 2/3 |
| Step 40, DL-Val91 single-pick | −0.0329 [−0.0714, −0.0035] | −0.0259 [−0.0575, −0.0019] | −0.0150 [−0.0362, 0] | +0.0126 [0, +0.0345], 3/3 | **+0.0179 [+0.0024, +0.0410], 3/3** |
| Step 40, TRAIN OOF TOP4_PICK | −0.0073 [−0.0149, −0.0019] | −0.0097 [−0.0230, −0.0005] | −0.0082 [−0.0152, −0.0001] | +0.0001 [−0.0087, +0.0057], 2/3 | −0.0010 [−0.0038, +0.0050], 1/3 |
| Step 40, TRAIN OOF single-pick | −0.0337 [−0.0610, −0.0081] | −0.0251 [−0.0526, −0.0013] | −0.0255 [−0.0515, −0.0008] | +0.0062 [−0.0040, +0.0151], 3/3 | +0.0073 [−0.0002, +0.0153], 3/3 |
| Best SELECT step, DL-Val91 TOP4_PICK | −0.0024 [−0.0043, −0.0003] | −0.0023 [−0.0044, −0.0007] | −0.0020 [−0.0041, −0.0004] | +0.0001 [−0.0011, +0.0010] | +0.0003 [−0.0006, +0.0013] |
| Step-40 TOP4 Elo, DL-Val91 [CI95] | 1421 [1375, 1467] | 1434 [1400, 1471] | 1443 [1414, 1465] | — | — |
| Step-40 TOP4 harm families (DL-Val, per seed) | 13, 14, 14 | 13, 14, 12 | 14, 14, 13 | — | — |
| Step-40 L2 drift from r882 (median [min, max]) | 0.372 [0.364, 0.389] | 0.349 [0.325, 0.356] | 0.323 [0.309, 0.333] | — | — |

### The H152 gate-cliff context #27 (`fam-9f67f144…`), reported separately and not excluded

r882 scores +1.1227 single-pick and +1.1285 TOP4_PICK there.

| Mixture | TOP4_PICK per seed | Single-pick |
|---|---|---|
| Strict, all arms | 1.1284–1.1285 | — |
| Step 40, CONTROL | 1.1268–1.1278 | 1.1227 in all seeds (no flip) |
| Step 40, T4_HINGE | 1.1278 in all seeds | 1.1227 in all seeds |
| Step 40, T4_PL | 1.1247–1.1259 | 1.1227 in all seeds |

## Reading

- **The minibatch recall target does not transfer to the full pool.** Every minibatch carries the 4 lowest-fitness
  candidates plus 20 random ones, so its oracle best is always among a handful of strong candidates. On the full pool,
  r882's top-4 contains the oracle best for only 6.6% (DL-Val91) and 0.9% (TRAIN) of contexts, and 40 updates of
  either objective did not raise that. TOP4_PICK's value comes from revealing a good-enough candidate, not the oracle
  best, and neither objective improved that either.
- **The aux losses act as a brake on drift, not as a ranking signal.** Both objectives cut drift (0.32–0.35 against
  0.37) and reduce the single-pick damage at step 40 (T4_PL roughly halves it: −0.015 against −0.033; T4_PL − CONTROL
  +0.018, 3/3 seeds). Their step-40 TOP4_PICK loss
  is the same as CONTROL's.
- **TOP4_PICK is harder to move than single-pick.** CONTROL's step-40 TOP4 loss (−0.011) is a third of its single-pick
  loss (−0.033), because revealing 4 candidates absorbs most single-pick mistakes. The same buffering leaves little
  room for a continuation to gain.
- **The strict selector protected the decision.** It accepted 5 of 45 cells, all small SELECT wins that did not carry
  over to DL-Val91.

## Verdict

**NO_ADVANCE.**
- T4_HINGE and T4_PL tie or lose TOP4_PICK@r882 on DL-Val91: −0.00013 [−0.0005, 0] and 0 [0, 0], 0/3 seeds,
  Holm p 1.0.
- No serving or live-flywheel change. DOPE `c51a56d0` is unchanged.

## Next step

**Do not tune these objectives.** Neither moved full-pool recall@4.

**If recall@4 stays the target, train it on the full candidate pool**, against the frozen r882 top-k on each FIT
context, rather than on the H156 minibatch. The minibatch's built-in top-4 makes its boundary uninformative.

**Treat TOP4_PICK@r882 = 1.2404 as the incumbent to beat.** Every continuation so far has lost to it at step 40.

No follow-up recall@4 wave on the minibatch objective is proposed.

## Files and custody

- **In this directory:**
  - [metrics.json](metrics.json): the compact readout, 88 KB, with `dope_before`/`dope_after`, the strict, unconstrained
    and step-40 panels for TOP4_PICK and single-pick, recall@4, drift, margins, the cliff context and the full-metrics
    pointer. The TRAIN OOF panels carry harm-family counts; the full lists are in S3;
  - [PREREG.md](PREREG.md) and [AMENDMENT-A1.md](AMENDMENT-A1.md);
  - [runner.py](runner.py), [treatment.py](treatment.py), [analysis.py](analysis.py) and [monitor.py](monitor.py);
  - [test_h260.py](test_h260.py): 15 data-free tests. Its coverage, test-inventory and source-classification
    registrations are deferred until `v1/next` is rebased on main, because the freeze forbids CI-config edits;
  - [schedules.json](schedules.json), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json) and
    [provenance.json](provenance.json).
- **S3, create-only with sha256 and VersionId readback:**
  - `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H260/attempt-001/` holds:
    - the full `metrics.json` (VersionId `YY351n1U…`);
    - the resource summary and the attempt claim;
    - `custody-attempt-001.tar.gz`, which contains every cell JSON, the per-cell selection locks, the adapter
      snapshots (`.npz`), the panels, the stdout, the resource samples and both smokes;
    - `s3_manifest.json` (sha256 `e291356d…`, VersionId `_i4ToPQ4…`).
  - The prereg sources, smokes, the review receipt and AMENDMENT-A1 are under `H260/prereg-7438c10d/` (20 objects).
- **Not in Git.** No checkpoint or snapshot.
