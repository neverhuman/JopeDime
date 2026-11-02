# H210 — H203 retry: BLEND_025 gated by an outcome-free harm signal, xbabe1 CPU, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Why.** [H203](../h203_blend025_gated/README.md) closed INCOMPLETE with no outcome computed. Both of its
  attempts on xbabe2 were stopped by one-minute load spikes from other jobs.
- **Question.** H199's BLEND_025 lifts mean native utility but harms 24.9% of families. Can an outcome-free signal
  decide per context when to trust the blend's switch, falling back to r882 otherwise, so that the native gain is
  kept with harm ≤ 10%?
- **Gate (as H203).**
  - `zgap = (r882[blend pick] − r882[r882 pick]) / std(r882)`, computed from costs only.
  - τ is chosen per cell on FIT folds only: maximize FIT native gain subject to FIT harm ≤ 5%. A closed gate falls
    back to r882.
  - H129's `disagreement` gate is report-only.
- **Design.** The merged H203 machinery (xbabe1-only admission), with fresh seeds 210101–210103 and fold seeds
  210141–210143. There are 45 FIT-only fits, and the stage-1 lock (calibration and τ) is fsynced before A scoring.
- **Decision.** GATED_025 against R882 (m = 1). CANDIDATE iff LCB95 > 0, p < 0.05 and harm ≤ 10%.
- **Load gate (new, matched to xbabe1's 102 cores).**
  - Admission needs load1, load5 and host CPU ≤ 50%, and disk used < 93.5%.
  - The run stops on load5 > 85%, or on load1 or host CPU > 85% sustained for 300 s.
  - The disk stop is at used ≥ 93.8% or free < 40 GB.
- **Budget.** xbabe1 CPU, about 1 h 30 min, root about 130 MB. No GPU.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
  [test_h210.py](test_h210.py) (29 data-free tests).
