# H216: random-init lottery with B-locked strict-margin selection

**Status: PREREGISTERED, NOT RUN.** Authority: `~/nudges/hw_t0300_argon_h216.md` (`rd_argon`).

Tests whether selecting a single untrained `adapter_10tok` initialization (`K = 16` fresh constructor initializations per replicate seed, `0` optimizer updates) on the reserved `B` split using a strict bootstrap-margin rule (`10,000`-draw paired family-bootstrap median gain over `r882` on `B` has 10th-percentile `> +0.005`, else keep `r882`) yields held-out utility lift on `A` across the `1,219` `DL-Train` contexts (`205` families, `H183` 5-fold family partition `FIT`/`B`/`A`).

See [PREREG.md](PREREG.md), [manifest.json](manifest.json), [fold_manifest.json](fold_manifest.json), and [provenance.json](provenance.json).
