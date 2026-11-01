# H199 — BLEND_025 on the flywheel native first-argmin protocol, 205-family pooled cross-fit, 0 updates

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Question.** Does H195's confirmed BLEND_025 lift r882 on the flywheel's own native protocol (full-pool first
  argmin, −fitness valid / −2.0 invalid)? It is tested on the 205-family pooled cross-fit rather than H135's
  19-family flywheel91.
- **Design.**
  - The H195 machinery, with fresh seeds 199101–199103 and fold seeds 199141–199143.
  - 60 FIT-only fits.
  - The stage-1 lock is fsynced before A scoring.
- **Decision.** The primary is BLEND_025 against R882 (m = 1). CANDIDATE iff LCB95 > 0, median > +0.001, 3/3 seeds
  positive and harm ≤ 10%.
- **Report-only.** U4, the λ 0.15/0.35 dose, SHUF, −2.0 counts and the flywheel91 DL-Val comparison with H135.
- **Code.**
  - [runner.py](runner.py), [analysis.py](analysis.py) and [monitor.py](monitor.py): copied from H195, minimal diff.
  - [test_h199.py](test_h199.py): 20 data-free tests, including xbabe1-only host admission before any side effect.
