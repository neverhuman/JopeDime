# B06 scalar utility receipt preparation

Native measurements executed at `c89194ef384def6e0be3aacc08a4c495047d3760` using the corrected raw CV evaluator. This preparation is private; normal compiled parity passed for all 602 cases. Full sanitizers passed for all 602 cases. Card after-suite and exact-head publication lifecycle remain pending.

Primary policy: `Penalize{0.05,0.0,true}`; Legacy remains informational. Eight pinned TRAIN-only datasets, five seeds, equal 72-attempt budgets, caps15/4. Reporting uses a distinct internal training cut after selection/refit; outer-test and sealed data remain outside the worker.

| Gate | Measured status |
| --- | --- |
| G-A | fail |
| G-B | pass |
| G-C | fail |
| G-D | pass |
| G-E | pass |

Decision: keep-experimental. Default/Baseline promotion is not authorized.

Failed promotion gates:

- G-A: Raw predictive-quality gate failed at unchanged thresholds.
  Noninferior on every dataset: True; strict improvements: 1/2; valid ten-percent M3 improvement: False.
- G-C: Actual fault, hard-reject or invalid-offspring excess failed an unchanged soundness limit.
  Witness: `{"dataset": "golden-toy-107", "difference": 0.04186248073959938, "maximum_allowed": 0.02, "metric": "M4_domain_rate"}`.
  Witness: `{"dataset": "golden-toy-107", "difference": 0.022222222222222223, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-229-pwlinear", "difference": 0.01111111111111111, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-695-chatfield-4", "difference": 0.022187981510015407, "maximum_allowed": 0.02, "metric": "M4_domain_rate"}`.
  Witness: `{"dataset": "pmlb-695-chatfield-4", "difference": 0.01666666666666667, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-first-principles-supernovae-zg", "difference": 0.02222222222222222, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-agridat-bancroft-peanut-uniformity", "difference": 0.024966294298921415, "maximum_allowed": 0.02, "metric": "M4_domain_rate"}`.
  Witness: `{"dataset": "rdatasets-agridat-bancroft-peanut-uniformity", "difference": 0.016666666666666663, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-stevedata-chile88", "difference": 0.021451271186440676, "maximum_allowed": 0.02, "metric": "M4_domain_rate"}`.
  Witness: `{"dataset": "rdatasets-stevedata-chile88", "difference": 0.01666666666666667, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-wooldridge-barium", "difference": 0.01666666666666667, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.

The paired JSON binds profile/policy/cohort/fold/cut/source identities, exact actual work and M1–M10, including null thresholds. Search objectives remain Penalize while M1/M2/M3 report raw CV. The retained negative exact-bit regression and corrected result establish that distinction. Prior source6e observations remain historical; these fresh results do not transplant their quality timings or source labels.
Normal compiled evidence: 602 cases (240 winners, 302 manual controls, 60 actual create/change/remove phases), 19 batches, 1806 source identities, 344 retained artifact hashes. Complete Rust/Python/g++17/clang++20 F/Q/S/P and fitted g++17/clang++20 P/Q match exactly. Full ASan/UBSan passed all 602 complete and fitted cases in g++17/clang++20: 76 compiles, 76 runs, 152 commands bounded at the original 240 seconds, 2408 case/compiler comparisons. G-B passed. This preparation does not authorize publication or promotion.

Exact parity is qualified on the recorded x86_64 Linux/glibc host. It does not establish a portable twin for PlatformLibm capabilities. Compiler/interpreter identities are hash-bound in the paired JSON.
