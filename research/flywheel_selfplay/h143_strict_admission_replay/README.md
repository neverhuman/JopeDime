# H143 — shadow replay of the strict-margin admission rule

**Status: preregistered (CPU, zero optimizer updates, offline); not yet run.** See [PREREG.md](PREREG.md).

- **What it does.** Replays all 1,756 logged admissions (rounds 775–2530) from the per-round receipts' per-context
  val utilities. Coverage is complete.
- **Arms.**
  - S_LIVE: must reproduce every logged action and r882.
  - R_STRICT: family-bootstrap median gain > +0.001, ties rejected.
  - R_STRICT_NOINV: R_STRICT with invalid (−2) contexts excluded.
- **Seeds.** 143101–143103.
- **DL-Train scoring.** Each final incumbent is scored against R882 with a family-bootstrap CI and Elo.
- **Verdict.** RULE_HELPS, RULE_NEUTRAL, RULE_HURTS or INCOMPLETE_COVERAGE.
- **Limitation.** The candidate stream is the logged one; counterfactual candidates cannot be reproduced offline.
- **Code.** [runner.py](runner.py) has the stages `replay` and `dltrain`. [test_h143.py](test_h143.py) has 12
  synthetic controls.
