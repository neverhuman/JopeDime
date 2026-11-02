# H217 — fresh-fold confirmation of H210 GATED_025 (frozen gate), xbabe2 CPU, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Why.** [H210](../h210_blend025_gated_retry/README.md) found CANDIDATE: GATED_025 on one set of fold partitions:
  native +0.0140 [+0.0022, +0.0355], harm 3.4%, 3/3 seeds, Elo 1505.6. DL-Val flywheel91 moved only +0.0010
  (descriptive), and U4's CI crossed 0.
- **Question.** Does the frozen GATED_025 recipe, re-run on fresh partitions with no retuning, keep a native gain over
  r882 with harm ≤ 10% in every seed, and does it lift the flywheel91 plateau?
- **Frozen recipe.** H210's blend, `zgap` gate and FIT-only τ procedure: the same grid, a 5% FIT harm target and
  r882 fallback. A test checks that their function bodies are identical to H210's. No τ is carried over.
- **Arms.**
  - GATED_025 is the primary.
  - The H129 disagreement gate (GATED_025_DIS) is the placebo.
  - The ungated BLEND_025 is reported as the component GATED_025 switches to.
- **Fresh partitions.** Seeds 217101–217103 with fold seeds 217141–217143, one distinct partition per seed. Best-match
  Jaccard is at most 0.183 against H210's partitions and at most 0.179 against H203's.
- **Decision.**
  - **Primary.** CONFIRMED: GATED_025 iff LCB95 > 0, 3/3 seeds positive and harm ≤ 10%, over 205 families with a
    10k paired family-cluster bootstrap.
  - **Co-primary.** PLATEAU_WIN iff the DL-Val 91-context context-mean gain over r882 (1.1377942) is > +0.002 with a
    family-bootstrap LCB95 > 0. A tie is not a win.
  - Elo and utility only.
- **Host and guards (operational fixes from H210).**
  - xbabe2 CPU, nice 19, no GPU.
  - Limits are sized from the effective CPU affinity (102 of 128).
  - The source is bound before the claim.
  - One absolute deadline: claim time + 4 h, inherited by any resume.
  - Stop at MemAvailable < 20 GiB (swap is full) or disk ≥ 93.8%.
- **Budget.** 45 HistGBM fits, about 1 h 30 min to 2 h; root about 180 MB.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
  [test_h217.py](test_h217.py) (34 data-free tests).
