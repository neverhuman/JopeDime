# H260: recall@4 objectives for the r882 adapter continuation

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before anything runs.

## Hypothesis

**What is known.** H258 showed that the deployment-relevant readout is best-of-revealed: TOP4_PICK@r882 scores
1.2404 on DL-Val91 against 1.1378 for single-pick r882. Every adapter continuation so far was trained and selected for
the top-1 argmin and gave no lift.

**What is new.** Training the adapter for recall@4, so the oracle-best candidate lands in the predicted top-4 that the
4-reveal flywheel actually reveals. The question is whether that lifts DL-Val91 TOP4_PICK over TOP4_PICK@r882.

## Setup

- **Arms.** All use the exact H170 CTRL AdamW, batches, data order, clip 1.0 and 40 updates from a read-only r882 copy:
  - **CONTROL:** exact H170 CTRL loss.
  - **T4_HINGE (primary):** CTRL + 0.5 × a hinge on the predicted rank-4 boundary, margin 0.05 × the FIT-fold r882
    cost std.
  - **T4_PL (secondary):** CTRL + 0.5 × −log of the exact Plackett–Luce probability that the oracle best is in the
    first 4 draws (τ = 1).
- **Hypothesis family.** T4_HINGE and T4_PL against r882, with Holm m = 2.
- **Design.** One fresh fold seed 260140, seeds 260101–260103, 15 cells per arm, the strict inner selector on SELECT
  TOP4_PICK utility, and selections locked before assessment.
- **Decision.** It is made on DL-Val91 TOP4_PICK (exact H258): median gain > +0.002, LCB95 > 0, Holm p < .05,
  3/3 seeds positive, and single-pick not worse than r882 by more than 0.002.
- **Also reported:** recall@4, single-pick utility, the TRAIN out-of-fold panels, the step-40 diagnostic, ARM − CONTROL,
  harm families, Elo and the cliff context #27.
- **Host.** xbabe2 CPU cores 44–47, nice 19, under the H239 A1 memory guard. The compute stop is 17:15Z.

## Files

| File | Contents |
|---|---|
| [runner.py](runner.py) | the study runner |
| [treatment.py](treatment.py) | the hinge@4, Plackett–Luce top-4 and CTRL losses |
| [analysis.py](analysis.py) | the exact TOP4_PICK readout, selection and inference |
| [monitor.py](monitor.py) | the A1 guard |
| [test_h260.py](test_h260.py) | data-free controls |
| [schedules.json](schedules.json) | the frozen settings |
| [manifest.json](manifest.json) | input, asset and source pins |
| [fold_manifest.json](fold_manifest.json) | the folds |
| [provenance.json](provenance.json) | the real-S3 closure |
| [metrics.json](metrics.json) | the metrics schema placeholder |
