# H160 — trajectory-subspace projected continuation of r882

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Basis.** The top-k principal directions of the deltas along the accepted adapter chain R95 → … → R256 →
  R782 → R785 → R859 → R882. k = 2 holds 92.6% of the delta energy and k = 4 holds 98.5%.
- **Arms.** Each continues r882 for 40 AdamW updates (lr 1e-4):
  - CTRL: unconstrained;
  - **PROJ_K2** and **PROJ_K4**: constrained to r882 + span(U_k);
  - CTRL_DRIFT: unconstrained, with the learning rate scaled so its drift matches PROJ_K4's.
- **Harness.** Merged H148 family cross-fit: fit on 3 folds, select on 1, assess on 1, over 205 TRAIN families,
  with seeds 160101–160103. The strict selector picks a step only on a gain > +0.001 with a lower bound > 0;
  otherwise it keeps step 0.
- **Reported.** Gain against r882 with a family-bootstrap CI, Elo against r882 = 1500, the number of non-zero
  selections, and DL-Val as descriptive output.
- **Code.**
  - [run.py](run.py) reuses the pinned H146 executor, as H148 does.
  - [analysis.py](analysis.py) holds the basis, projection, H148 harness and decision.
  - [monitor.py](monitor.py) is the hard-stop supervisor.
  - [test_h160.py](test_h160.py) has 11 data-free tests.
  - [preflight.json](preflight.json) is the zero-forward smoke.
