# GP language red-team matrix

Observed on remote main `1f352bb0b740bffad7f88fc36583659981da59e5` (B20 core merged).

The executable cases are in [`gp_language_tricks.rs`](../../crates/feat-math/tests/gp_language_tricks.rs). They use existing public scalar-program APIs; no shared determinism harness is edited. Emission and execution require no model or LLM service.

## Initial unchanged-main audit

27 constructions × nine rows = 243 mathematical reference checks. Python and Rust each match all 243 evaluator words exactly. Every C++17 source fails its required `-Wall -Werror` compilation; no C++ execution or three-language PASS is claimed. The complete sweep exits 101 and retains all compiler stderr.

## Corrected dependency validation

At the committed Lane B emitter correction `1e8cf7aa6413e83ad2c6d28a4198c57bf95ef97b`, both Lane A tests pass (zero ignored/excluded): all 27 constructions compile and execute in all three languages. All 729 backend output words match the evaluator, and all 243 independent reference checks pass. The correction belongs to Lane B; this matrix adds no product or shared harness changes. Publication remains dependent on that repair landing and the final local/hosted gates.

The 25 legacy expression spellings parse to bounded ASTs and reach canonical fixed points. The two versioned inverse-link trees use the public typed `ScalarExpressionProgram` API; their descriptive `probit_v2(c0)` and `logit_v2(c0)` labels do not establish support in the frozen legacy parser. Every program preserves its binary codec bytes through decoding. Each backend is emitted twice with identical source bytes. Runtime input words are supplied after compilation, preventing input-dependent constant folding. The drivers exercise the emitted scalar stage rather than substituting a handwritten formula.

Python executes the unmodified recipe via `python3 -c` and `compile`/`exec` in a separate module namespace. Rust compiles with `rustc --edition=2021 -O`; its original CLI main is renamed only to attach the numeric probe. The maintained matrix requires both `g++` and `clang++` to compile the unmodified C++ recipe plus a numeric probe with `-std=c++17 -Wall -Werror -O2 -ffp-contract=off -fno-fast-math`. The recorded initial dependency run above used g++; the fused candidate adds the independent clang++ lane and needs its own execution receipt. The stage values are checked, not merely the presence of source strings.

| Trick | Tree witness | Python bits | Rust bits | C++17 strict |
| --- | --- | --- | --- | --- |
| ols_predictor | `2*c0+1` | PASS | PASS | PASS |
| ridge_closed_form | `(8/(4+1))*c0+1` | PASS | PASS | PASS |
| lasso_orthogonal | `sign(c1)*max(abs(c1)-0.25,0)*c0` | PASS | PASS | PASS |
| elastic_net_orthogonal | `sign(c1)*max(abs(c1)-0.25,0)/(1+0.5)*c0` | PASS | PASS | PASS |
| glm_logistic | `sigmoid(c0)` | PASS | PASS | PASS |
| glm_log_link | `exp(c0)` | PASS | PASS | PASS |
| hinge | `max(c0-0.5,0)` | PASS | PASS | PASS |
| linear_spline | `c0+0.5*max(c0-0.5,0)` | PASS | PASS | PASS |
| cubic_spline | `c0+0.5*max(c0-0.5,0)^3` | PASS | PASS | PASS |
| nested_decision_tree | `(1-step(c0,0))*2+step(c0,0)*((1-step(c1,0.5))*3+step(c1,0.5)*4)` | PASS | PASS | PASS |
| boosted_stumps | `0.5+0.25*step(c0,0)-0.5*step(c1,0.5)` | PASS | PASS | PASS |
| rbf_kernel | `exp(-0.5*((c0-c2)^2+(c1-c3)^2))` | PASS | PASS | PASS |
| polynomial_kernel | `(1+c0*c2+c1*c3)^3` | PASS | PASS | PASS |
| standardize_frozen_train_state | `(c0-0.5)/2` | PASS | PASS | PASS |
| minmax_frozen_train_state | `(c0+2)/4` | PASS | PASS | PASS |
| log1p_transform | `log1p(abs(c0))` | PASS | PASS | PASS |
| softmax_component | `exp(c0-max(c0,c1))/(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))` | PASS | PASS | PASS |
| log_sum_exp | `max(c0,c1)+log(exp(c0-max(c0,c1))+exp(c1-max(c0,c1)))` | PASS | PASS | PASS |
| clamp | `clip(c0,-0.5,0.5)` | PASS | PASS | PASS |
| where_finite_branches | `step(c0,0)*c1+(1-step(c0,0))*c2` | PASS | PASS | PASS |
| interactions | `c0*c1+c2*c3` | PASS | PASS | PASS |
| rational_function | `(c0+c1)/(1+c2*c2)` | PASS | PASS | PASS |
| box_cox_lambda2 | `(c0^2-1)/2` | PASS | PASS | PASS |
| box_cox_lambda0 | `log(c0)` | PASS | PASS | PASS |
| glm_logit_link | `log(c0/(1-c0))` | PASS | PASS | PASS |
| glm_probit_link_v2 | `probit_v2(c0)` | PASS | PASS | PASS |
| glm_logit_link_v2 | `logit_v2(c0)` | PASS | PASS | PASS |

## Reference and fitting scope

The mathematical reference functions are independent algebraic implementations, not calls back into the interpreter. Their bounded roundoff check was fixed before the sweep at `5e-14 * max(1, abs(reference))`. Backend parity is a separate exact word comparison and has no tolerance. Probit instead uses its approved Acklam relative-error contract `1.2e-9 * abs(reference)`, with exact zero at p=0.5. Its nine oracle values are `sqrt(2)*erfinv(2*p-1)`, evaluated independently at 100 decimal digits for p=1/16 through 9/16. These checks do not qualify all probabilities or libm platforms.

The OLS witness is an affine predictor with supplied coefficients; this test does not fit them. The ridge witness includes the centered one-dimensional closed-form ratio from supplied moments. Lasso and elastic-net demonstrate the orthogonal-design soft-threshold closed form only. The matrix does not claim a general correlated-design lasso/elastic-net optimizer as a finite scalar tree. Standardization and min-max witnesses use supplied, frozen training-state constants; this proof does not estimate them or establish training-fold isolation. The emitted artifact is a complete training recipe, but the probe verifies its scalar feature stage, not the subsequent fitted ridge state or prediction.

Splines cover a linear hinge basis and one cubic truncated-power basis; RBF and polynomial kernels cover two fixed coordinates. Decision-tree and boosted-stump witnesses cover bounded finite leaves and exact threshold neighbors, not an arbitrary-depth forest. Softmax is one fixed-width component. This finite corpus does not establish universal grammar coverage.

## Findings and next focused repairs

### GP-LANG-001 — all scalar C++ recipes rejected by the required compiler flags

`Model::fit` puts the scalar shape guard and label loop on one line, producing `-Wmisleading-indentation`. Two unused `static` helpers (`sanitize` and `finish_stage`) produce `-Wunused-function` on the raw scalar route. Both become errors under the requested flags. This is a real emitted-source defect, independent of all 27 arithmetic trees. No warning suppression or relaxed compiler command is proposed.

A narrow proposed repair is prepared outside the checkout: normalize only versioned scalar C++ source formatting and make the two helper definitions inline. The original generator is frozen; the unfrozen scalar recipe path is currently claimed by Lane B. Lane B confirmed at 10:40:56Z that it owns and is implementing the generator repair; Lane A will not duplicate it. The earlier ownership question needs no operator action. Proposed changes must preserve every v1 source byte and mathematical behavior, then pass the same strict compiler and word checks. This repair and regression belong in their own small PR.

### GP-LANG-002 — no lazy conditional selection node

Arithmetic selection evaluates both branches. With `c0=-1`, `c2=c3=NaN`, `(1-step(c0,0))*2+step(c0,0)*clip(c1,c2,c3)` refuses because the inactive clip branch has invalid bounds; a lazy conditional would return 2. `where(c0,c1,c2)` is not parseable. The finite-leaf witness above is therefore not proof of short-circuit if-then-else. A future fix needs an explicitly versioned lazy-selection contract and all three lowerings; the legacy eager arithmetic contract must remain unchanged.

### Protected arithmetic boundaries

The legacy power node clamps the exponent to [-8,8]; `c0^9` at c0=2 produces 256 with POW_EXP_CLAMPED, not the ninth-degree value 512. The ninth-degree polynomial itself is representable using nine multiplied leaves. Box-Cox/general polynomial construction must not silently choose a guarded power when it means an unrestricted exponent. This is an existing protected-operation contract, not an evaluator/emitter mismatch.

Stabilized log-sum-exp composition still inherits the historical exp argument clamp [-30,30]. At logits [0,-100], its small positive result differs from `log1p(exp(-100))`; EXP_ARG_CLAMPED is reported. The ordinary-range PASS does not certify large logit gaps, broad RBF radii or unbounded mathematical exp. Correcting that contract requires a versioned primitive rather than changing legacy guards.

### Resource limits

Trees are limited to 64 nodes and depth 16. A 65-leaf sum is explicitly refused; no tree is silently truncated. Expanding ensemble size needs a separately bounded representation/composition design, not removal of admission limits.

## Evidence

Initial raw generated sources, runtime input words, compiler stdout/stderr and matrix words are retained outside Git under `/home/ubuntu/jope-evidence/kernel2/bt/GP_LANGUAGE_REDTEAM/main-1f352bb0/audit-02/`. `execution-binding.json` records the main SHA, test source hash, command and nonpassing exit. The first 25-case sweep is retained as audit-01. No expected/golden file, tolerance, frozen identity input, v1 operator, runtime source or Lane B harness was changed.

The passing dependency execution and its raw generated programs/words are separately retained under `GP_LANGUAGE_REDTEAM/dependency-1e8cf7aa/focused-01/` (59.32 seconds test execution). Initial failures are not deleted or relabeled.

## Fused independent proofs

This release PR also carries [nested expression conversion](gp-expression-conversion.md) from Lane C's `e1ed1076388fe46614e21ca9be8015e8f8b3cc98` and the [independent PRESS oracle](ridge-press-oracle.md) from `cde9ba2c77ee50e7db53c44a192e0ac84a904432`. Their original assertions and numerical bounds are retained. Nested conversion now also requires strict g++ and clang++ compilation; only the test driver's indentation changes. These are separate acceptance scopes, not extra mathematical cases in the 27-trick matrix.

The first fused execution passes both trick tests (113.81 seconds) with 972 exact words across Python, Rust, g++ and clang++. Both PRESS tests pass: 81 designs, 1,296 independent deleted-row solves, and 405 prefix objectives/extensions. The full e-graph selection passes 22 tests. This is focused evidence, not a completed PR tier or hosted qualification.

The first strict imported-conversion run exposed another defect: the legacy Symbolic expression recipe included an unused `static max_num` helper. Its first subtraction witness failed under `-Wall -Werror`; the earlier proof lacked those flags. On the committed Lane B correction `c7a8127948ed059191b769e8f7860493effe6154`, both conversion tests now pass: all nine complete recipes execute in Python, Rust, g++ and clang++, matching 72 explicit expected output words with unchanged strict compiler flags. The command took 53.15 seconds including compilation; test execution took 28.12 seconds. This is focused dependency evidence, not final PR qualification or a merged-repair claim. Original failing logs remain retained under `GP_LANGUAGE_REDTEAM/fusion-focused-01/`; the successful run is under `dependency-c7-strict-conversion/`.

The maintained `math_expr::evolution::v2::egraph::tests::all_libm_calls_preserve_literal_shape_including_tanh` test protects the B20 exclusion of constant libm folding. It checks eleven functions, five literal values including both signed zeros, and direct versus arithmetic-produced arguments (110 shapes). Each enclosing call must remain a call, its e-class must contain no literal, and evaluation bits/faults must remain identical. The simplifier is still test-only; this test does not establish runtime search wiring. Adding exactly Lane C's four-line unauthorized constant-Tanh fold changes 22 passing maintained tests to 21 passes and a failure in the new test. The control and mutant compile successfully; their source hashes, commands and raw logs are in `GP_LANGUAGE_REDTEAM/tanh-maintained-mutation/result.json`. Mutation source and logs remain outside Git; production folding rules are unchanged.
