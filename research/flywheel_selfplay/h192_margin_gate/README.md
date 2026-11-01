# H192 — strict-margin (confidence-gated) adapter selection, 0 updates

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Gate.** Keep the anchor-only base pick unless r882's full pick wins by a normalized cost margin > τ.
  - τ = 0 is R882 exactly.
  - The τ grid is the FIT-quantiles of the margin among disagreeing contexts.
  - τ\* is chosen on B, locked and fsynced, then scored on A (MGATE, m = 1).
- **Folds and inference.** Seeds 192101–192103 use different fold seeds, 192141–192143. Reported: the H177 pooled
  selector, the median gain with a paired family bootstrap (axis 0), Elo against r882, −2.0 counts, override rates
  and τ\* stability.
- **Code.**
  - [runner.py](runner.py) and [treatment.py](treatment.py).
  - [analysis.py](analysis.py): imports the merged H188 helpers.
  - [monitor.py](monitor.py).
  - [test_h192.py](test_h192.py): 8 data-free tests.
