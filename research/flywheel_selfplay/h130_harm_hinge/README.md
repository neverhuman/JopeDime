# H130 — Harm-constrained adapter fine-tuning

**Preregistration PR ready; not run.** H130 tests fit-fold family harm penalties in the loss against frozen `r882` per-family baselines. The preregistration is [PREREG.md](PREREG.md); machine-readable frozen protocol is [manifest.json](manifest.json). No training or assessment scoring has occurred.

The three paired arms are uniform mean-loss `CONTROL`, `HINGE`, and top-20%-family `CVAR`, each at seeds 94–96 with 40 updates per cell. Fold assignment uses seed 130094; the compact assignment and SHA-256 are in [fold_manifest_130094.json](fold_manifest_130094.json). Baselines use fit families only. Primary outcomes compare HINGE and CVAR with unchanged `r882` using paired family-cluster bootstrap, Holm correction, a 0.001 LCB threshold, nonnegative family median gain, and at most 10% harmed families. See the preregistration for exact formulas and resource gates.

`run_h130_cpu.py` is adapted from the H128 CPU runner. `--synthetic-smoke` exercises finite gradients without data or optimizer updates. Real execution requires the preregistration source merged into `origin/main`, is CPU-only, and writes local custody receipts under `/home/ubuntu/jope-rescue-20260921/h130-harm-hinge/`. No serving promotion is implied by this study.
