# H303: ENS_VOTE2 with fully nested hyperparameters, plus a concentration audit

**Status: preregistered, not executed.** [PREREG.md](PREREG.md) is the frozen protocol. It is published on `v1/next`
before any H303 fit and before any outer-test score.

**What it tests.** H302's ENS_VOTE2 used TAU, the member vote threshold and the 2-of-3 agreement as constants tuned on
all of DL-Train. H303 chooses them inside each H302 outer_train only, by an inner 4-fold family CV over a declared
18-config grid. It then refits, freezes and scores outer_test once. It also audits whether the gain is concentrated in a
few families.

## Files

| File | Contents |
|---|---|
| [run_h303.py](run_h303.py) | the nested fit phase (inner CV, choice, refit, fit_lock) and outer-test scoring, reusing H302's code |
| [evaluate_h303.py](evaluate_h303.py) | the H302 picks regression, H302's statistics, the concentration audit and the decision |
| [test_h303.py](test_h303.py), [manifest.json](manifest.json) | data-free controls; pins |
| [metrics.json](metrics.json) | the metrics schema placeholder |
