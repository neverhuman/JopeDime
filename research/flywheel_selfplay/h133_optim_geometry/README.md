# H133 — optimizer geometry / step size vs harm

**Status: INCOMPLETE; attempt 013 stopped at the baseline regression gate before treatment.** A6 is merged. H133 retains the frozen CONTROL, LOWLR_LONG, and SGDM design, fold seed `133100`, and seeds `133, 134, 138`; no arm result or verdict is available.

The single post-merge CPU launch reproduced paired predictions, utility, CONTROL fit loss, and fit gradients exactly, but the required r882 family utility was `1.1377942` and the observed value was `1.1255361`. The worker stopped with zero optimizer updates and zero completed cells. This mismatch does not support an admission or refutation. No retry or expectation change is authorized.

See [PREREG.md](PREREG.md), [AMENDMENT_A6.md](AMENDMENT_A6.md), [manifest.json](manifest.json), and [metrics.json](metrics.json). The detailed [attempt 013 closeout](CLOSEOUT_A6_013.md) records the gate result. Receipts and resource logs are preserved in H133 custody; the consolidated receipt hash is recorded in `manifest.json`.
