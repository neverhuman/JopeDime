# H223 — bagged blend component inside the frozen GATED_025 recipe, xbabe2 CPU, 0 updates

**Status: preregistered; not yet run.** It runs only after this prereg merges. See [PREREG.md](PREREG.md).

- **Why.** [H217](../h217_gated025_confirm/README.md) confirmed GATED_025 (native +0.0216, harm 4.4%, Elo 1504.8),
  but its DL-Val flywheel91 gain, +0.00145, missed the +0.002 plateau margin. The ungated blend gained +0.0367 with
  25.9% harm.
- **Hypothesis.** A single HistGBM per context is high-variance. Averaging three members before the 0.25 blend should
  lower harm, let the FIT-only `zgap` τ open the gate more often, and lift DL-Val past +0.002.
- **Bag.** Each of the 45 fit units has three members on one design matrix (135 fits):
  - member 0 is H217's unweighted fit, tested bit-identical, and is also GATED_025's component;
  - members 1–2 are family-cluster bootstrap refits, using family multiplicities as sample weights.
- **Arms.**
  - GATED_BAG3 is the primary.
  - GATED_025 is the frozen reference; GATED_BAG3 − GATED_025 is reported paired.
  - GATED_BAG3_DIS (H129 disagreement on the bagged blend) is the placebo.
  - BLEND_BAG3 and BLEND_025 are the ungated components.
- **Fresh partitions.** Seeds 223101–223103 with fold seeds 223141–223143. Best-match Jaccard is at most 0.194 against
  H203, H210, H217 and H218; the code refuses anything over 0.20.
- **Decision.**
  - **Primary.** CANDIDATE: GATED_BAG3 iff LCB95 > 0, 3/3 seeds positive and harm ≤ 10% (205 families, 10k paired
    family-cluster bootstrap).
  - **Co-primary.** PLATEAU_WIN iff the DL-Val flywheel91 gain over r882 (1.1377942) is > +0.002 with a
    family-bootstrap LCB95 > 0. A tie is not a win.
  - Elo and utility only.
- **Host and guards (hw_t1325).** xbabe2 CPU, nice 19, OMP 7. Stops: worker tree > 8 cores, RSS ≥ 4 GiB, load5 > 60,
  MemAvailable < 20 GiB or free disk < 300 GB. One absolute deadline: the earlier of claim + 4 h and 17:25Z.
- **Budget.** 135 HistGBM fits, about 3 h. Closeout by 18:00Z, or INCOMPLETE with evidence.
- **Code.** [runner.py](runner.py), [analysis.py](analysis.py), [monitor.py](monitor.py) and
  [test_h223.py](test_h223.py) (37 data-free tests).
