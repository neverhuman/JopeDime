# H134 — prospective confirmation of H129 BLEND_025 vs r882

**Status: preregistered (CPU); no scoring yet.** See [PREREG.md](PREREG.md).

- **Primary.** One arm, BLEND_025 = r882 + 0.25·(GBM − r882), with λ fixed.
- **Fresh draw.** New family fold 134141, sampling seed 134142 and model seeds 141–143.
- **Gates.** All are required:
  - median seed gain > +0.001;
  - family-bootstrap LCB95 > 0;
  - all three seeds > 0;
  - harm ≤ 10%;
  - trimmed mean (dropping the 10 largest-|gain| families) > 0.
- **Elo.** Reported against r882 = 1500.
- **Code.** [runner.py](runner.py) imports the merged H129 runner (sha `e34d4a3f…`) and rebinds only the seeds and
  fold. [test_h134.py](test_h134.py) has 8 synthetic controls.
- **Caveat.** This re-splits the same DL-Train families, so it tests robustness, not new data.

## Amendment A-resume

The preregistered disk guard aborted the first attempt after 39/45 fits, before any outcome existed. Owner decision
(a) in `rd_opus_t1715` authorizes exactly one sha-verified resume of the 6 missing fits.
[AMENDMENT_A_RESUME.md](AMENDMENT_A_RESUME.md) records it, and [resume.py](resume.py) is the launcher. The protocol
and gates are unchanged.
