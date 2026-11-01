# H188 — kernel-input ablation on the r882 adapter (0 updates)

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Arms.** Three kernel-input changes on the trained r882 adapter, each changing exactly one input:
  - KIN_ANCHOR: similarity from the frozen anchor's features;
  - KIN_RANKRES: rank-normalized revealed residuals;
  - KIN_CLIPRES: Huber-winsorized revealed residuals.

  Only the kernel term is recomputed; the kernel replay is gated bit for bit against the live code.
- **Selection.** H177's pooled-B selector rule with m = 3, then a lock, then pooled-A confirmation, with seeds
  188101–188103 as fold reshuffles.
- **Reported.** Elo against r882, a 10,000-draw paired family bootstrap (axis 0) and −2.0 counts.
- **Disclosed.** At 0 updates, pooled A replicates pooled B exactly.
- **Code.**
  - [runner.py](runner.py) and [treatment.py](treatment.py).
  - [analysis.py](analysis.py).
  - [monitor.py](monitor.py).
  - [test_h188.py](test_h188.py): 10 data-free tests.
