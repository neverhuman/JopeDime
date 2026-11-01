# H195 — prospective confirmation of H129 BLEND_025 on fresh folds (kernel-active r882), 0 updates

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Question.** Does H129's BLEND_025 (`r882 + 0.25·(GBM − r882)`, calibrated HistGBM prior) beat r882 on U4 on
  held-out families, under three independent fold partitions? H134 never reached a readout.
- **Premise correction.** H129's r882 was already kernel-active, because the H124 baseline reveals slots 0..3 to the
  live adapter. BLEND_025 is therefore the exact H129 specification. BLEND_025_NOKERNEL (KERNEL summand removed, via
  the merged H184 decomposition, proved bit-exact) is descriptive, as are BLEND_025_SHUF (within-family permuted
  targets) and R882_NOKERNEL.
- **Design.**
  - Seeds 195101–195103, with fold seeds 195141–195143.
  - Cells: A = k, B = k+1, FIT = the other three.
  - The GBM and its calibration come from FIT only: 5 outer and 10 inner fits per seed and type, 60 fits in total.
  - The stage-1 lock is fsynced before any A scoring.
  - The primary contrast is BLEND_025 against R882 (m = 1). CANDIDATE requires LCB95 > 0, p < 0.05 against +0.001,
    and harm ≤ 10%.
- **Code.**
  - [runner.py](runner.py): imports the merged H139, H124 and H184 modules, byte-pinned.
  - [analysis.py](analysis.py).
  - [monitor.py](monitor.py).
  - [test_h195.py](test_h195.py): 11 data-free tests.
