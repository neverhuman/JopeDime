# H203 — BLEND_025 gated by an outcome-free harm signal, flywheel native protocol, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Question.** H199's BLEND_025 lifts mean native utility but harms 24.9% of families. Can an outcome-free signal
  decide per context when to trust the blend's switch, falling back to r882 otherwise, so that the native gain is
  kept with harm ≤ 10%?
- **Gate.**
  - `zgap = (r882[blend pick] − r882[r882 pick]) / std(r882)`, computed from costs only.
  - τ is chosen per cell on FIT folds only: maximize FIT native gain subject to FIT harm ≤ 5%. A closed gate falls
    back to r882.
  - H129's `disagreement` gate is report-only.
- **Design.**
  - The H199 machinery (xbabe1-only admission), with seeds 203101–203103 and fold seeds 203141–203143.
  - 45 FIT-only fits.
  - The stage-1 lock (calibration and τ) is fsynced before A scoring.
- **Decision.** GATED_025 against R882 (m = 1). CANDIDATE iff LCB95 > 0, p < 0.05 and harm ≤ 10%.
- **Budget.** xbabe1 CPU, about 1 h 30 min, root about 130 MB. No GPU.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
  [test_h203.py](test_h203.py) (26 data-free tests).
