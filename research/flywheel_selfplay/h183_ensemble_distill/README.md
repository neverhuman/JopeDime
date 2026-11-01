# H183: distill ten fresh adapters into one

Preregistration only; no scientific outcomes or optimizer updates yet.

The experiment asks whether a single 29,211-parameter student can retain ENS10 utility. RENS10 averages the same ten teacher initial states with zero updates, testing whether any benefit comes from random adapter structure. DIST10 is primary; DIST10_80 is secondary; Holm m=2 excludes diagnostics.

Use 1,219 real TRAIN contexts/205 families, fresh five-fold family partition, three seeds, FIT-only teachers and soft targets, and frozen cached DOPE features. Persist every member/student A candidate score and choice outside Git. A FIT-only timing pilot can select the first three outer folds; it never removes seeds or uses A outcomes. The full grid costs 7,800 actual updates because the 40-step student is the prefix of the 80-step trajectory.

See [PREREG.md](PREREG.md) for fixed losses, seeds, bootstrap, strict candidate gate and resource limits. Native r882 must reproduce all 91 baseline utility and candidate-cost bytes before training. No model is promoted by this study.

Validation: data-free controls, Ruff, five targeted jope-records verifiers and constructor-only smoke; exact-head cross-review precedes the single guarded CPU run. Results, independent replay and versioned S3 custody will be added in a separate closeout.
