# B05 scalar utility receipt preparation

Native measurements executed at `c89194ef384def6e0be3aacc08a4c495047d3760` using the corrected raw CV evaluator. This preparation is private; normal compiled parity passed for all 602 cases. Full sanitizers passed for all 602 cases. Card after-suite and exact-head publication lifecycle remain pending.

Primary policy: `Penalize{0.05,0.0,true}`; Legacy remains informational. Eight pinned TRAIN-only datasets, five seeds, equal 72-attempt budgets, caps15/4. Reporting uses a distinct internal training cut after selection/refit; outer-test and sealed data remain outside the worker.

| Gate | Measured status |
| --- | --- |
| G-A | fail |
| G-B | pass |
| G-C | fail |
| G-D | fail |
| G-E | pass |

Decision: keep-experimental. Default/Baseline promotion is not authorized.

Failed promotion gates:

- G-A: Raw predictive-quality gate failed at unchanged thresholds.
  Noninferior on every dataset: False; strict improvements: 0/2; valid ten-percent M3 improvement: False.
  Witness: `{"M1_delta": -0.008021421138834639, "dataset": "pmlb-695-chatfield-4", "minimum_noninferior_delta": -0.004921895946144945}`.
  Witness: `{"M1_delta": -0.0024038470547516066, "dataset": "pmlb-first-principles-supernovae-zg", "minimum_noninferior_delta": -0.0006447633521545359}`.
- G-C: Actual fault, hard-reject or invalid-offspring excess failed an unchanged soundness limit.
  Witness: `{"dataset": "golden-toy-107", "difference": 0.02777777777777778, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "golden-toy-911", "difference": 0.019444444444444445, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-229-pwlinear", "difference": 0.04722222222222222, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-695-chatfield-4", "difference": 0.027777777777777783, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "pmlb-first-principles-supernovae-zg", "difference": 0.03888888888888888, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-agridat-bancroft-peanut-uniformity", "difference": 0.024999999999999994, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-stevedata-chile88", "difference": 0.01666666666666667, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
  Witness: `{"dataset": "rdatasets-wooldridge-barium", "difference": 0.030555555555555558, "maximum_allowed": 0.01, "metric": "M6_invalid_offspring_rate"}`.
- G-D: Actual average bloat, artifact or timing cost exceeded an unchanged promotion limit.
  Witness: `{"maximum_allowed_B": 4.745, "mean_A": 4.745, "mean_B": 4.7825, "metric": "M7_elite_mean_nodes"}`.
  Witness: `{"maximum_allowed_B": 2.985, "mean_A": 2.985, "mean_B": 2.99, "metric": "M7_elite_mean_depth"}`.

The paired JSON binds profile/policy/cohort/fold/cut/source identities, exact actual work and M1–M10, including null thresholds. Search objectives remain Penalize while M1/M2/M3 report raw CV. The retained negative exact-bit regression and corrected result establish that distinction. Prior source6e observations remain historical; these fresh results do not transplant their quality timings or source labels.
Normal compiled evidence: 602 cases (240 winners, 302 manual controls, 60 actual create/change/remove phases), 19 batches, 1806 source identities, 344 retained artifact hashes. Complete Rust/Python/g++17/clang++20 F/Q/S/P and fitted g++17/clang++20 P/Q match exactly. Full ASan/UBSan passed all 602 complete and fitted cases in g++17/clang++20: 76 compiles, 76 runs, 152 commands bounded at the original 240 seconds, 2408 case/compiler comparisons. G-B passed. This preparation does not authorize publication or promotion.

Exact parity is qualified on the recorded x86_64 Linux/glibc host. It does not establish a portable twin for PlatformLibm capabilities. Compiler/interpreter identities are hash-bound in the paired JSON.
