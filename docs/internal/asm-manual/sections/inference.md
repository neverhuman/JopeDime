# Complete fitted inference and language support

At source `68f56404c3e199e06413bfd645d3fe0803d7b96f`, merged unchanged as main `a75e543d2` (same source tree), the legacy `jope emit` and `jope pipeline run --emit` paths produce type 3. `jope fit` computes a fit and reports scores but does not provide a saved fitted-artifact output flag. The pending product/winner packets, which are not part of merged main, expose type 4 from complete benchmark/demo fitted state. Do not advertise a legacy recipe file as the exact fitted winner.

This chapter and the supplied tiny fixture are **maintainer-private**. The DSL, private recipe manifest, discovery metadata and qualification references do not belong in a public inference bundle. The public allowlist described below is deliberately narrower than the manual's working directory.

## Deployment support: definition, priority and status

The owner's rule for this chapter, relayed on 2026-09-23 at 22:27:11 UTC, sets what may be called deployment support. It is a requirement for export work, not a description of the current source:

- **A complete fitted pipeline.** A deployable artifact carries the whole fitted pipeline of the selected model: its ordered transforms, fitted dictionaries, target mappings, classes and their order, and its numerical policy.
- **The actual winner, without refit.** The backend executes the selected winner from its saved fitted state. Refitting a recipe, even the same recipe on the same rows, is not deployment support.
- **Unsupported stays visible.** When a backend cannot represent the winner, that backend's result is "winner + unsupported". The winner is never replaced by another recipe that happens to export.
- **Priority.** Python and Rust first, then C++17 and C++20, then JavaScript.
- **Inside the original window.** Export work uses the original campaign window and stops before its reporting reserve ([§](#backend-numerical-policies-public-metadata-and-the-export-clock)).

By this rule, no route at merged main `a75e543d2` provides deployment support yet. Status as of 2026-09-23, about 23:55 UTC:

| Route | What it executes | Status |
|---|---|---|
| Legacy recipe emitters (`jope emit`, `jope pipeline run --emit`) | Generated source that refits a recipe from a training CSV. None of the 26 registered benchmark recipes can be emitted ([§](#coverage-by-recipe-and-backend)). | **available** for recipe refit only, which the fourteen-command private example run qualified on its fixtures ([§](#measured-execution-of-the-private-examples)). This is not deployment support. |
| Private fitted C++17 library (`emit_fitted_cpp17`) | Scalar-state C++ generated from a saved `BoundModel`, without refit | **implemented but unqualified** as a deployment route. At `a75e543d2` only the research crate `crates/jope-campaigns` calls it; the `jope` CLI does not. Its admission record, `crates/jope-campaigns/manifests/advanced-v3/fitted-cpp-v3-admission.v1.json`, is marked `test_adaptive: true` and `benchmark_valid: false`. |
| Public fitted bundles for Python/Rust, C++17/C++20 and JavaScript, driven by `benchmark export-winner` and `benchmark qualify-export` | The selected winner's saved fitted state, lowered for the supported subset, without refit | **pending integration**. These exist only as stopped custody patch views, absent from merged main ([§](#example-c-exact-fitted-winner-export-after-pending-integration)). No winner has been exported or qualified. |
| CUDA inference | Nothing yet | **proposed**, and deferred until the backends above qualify. |

## Language interfaces at the frozen source

| Interface | Input and output | Coverage and operational meaning |
|---|---|---|
| Native Rust `feat_math`/advanced-v3 | Typed recipe or authenticated `BoundModel`; numeric or raw-table prediction boundary | The benchmark uses the raw-table boundary and preserves complete private fitted state. The native roster is task-specific. |
| Legacy Python emitter | Generated single-file Python with `Model` fit/predict and CSV CLI | Numerical recipe execution; host `libm`. Supports the emitter's legacy program set, not every advanced-v3 state. |
| Legacy Rust emitter | Generated standard-library Rust with fit/predict and CSV CLI | Numerical recipe execution; source emitted with Python as a pair. Compiler/runtime and fixture qualification remain required. |
| Legacy C++ emitter | One C++ source, C ABI, numeric input, fit/predict; external CSV driver for examples | Ridge/ElasticNet heads and the emitter's admitted transform subset. C++17 and C++20 use this same source with different standard flags. |
| Legacy JavaScript emitter | ES2020 numeric fit/predict API; external CSV driver | Ridge/ElasticNet heads and its admitted transform subset. `Math` may differ from native platform libm. |
| Existing private fitted C++ library | `supports_fitted_cpp17`, `emit_fitted_cpp17` over `BoundModel` | Broader scalar-state emitter exists at `68f56404` and merged main `a75e543d2`. Its source gate is not the pending public bundle gate or proof of winner parity. Multiclass probability matrices are refused by its scalar ABI. |
| Pending public fitted Python/Rust | `predict_raw` over ordered raw rows with string/null cells | Complete selected-state lowering for the supported subset below; no model refit. |
| Pending public fitted C++17/C++20 | Same generated C++ inference core plus `codec.json` and Python `ctypes` raw-table boundary | Public gate currently restricts this broader emitter to regression ridge and binary logit ridge. Inference therefore needs a Python 3 `ctypes` host beside the compiled library (the pending packet names a Python 3.10 `ctypes` typed-table host); it is not a self-contained C++ raw-table parser. |
| Pending public fitted JavaScript | `predictRaw(rows)` over raw rows; Node/CommonJS qualification harness | Linear and multiclass subset below; a declared ECMAScript numerical policy and tolerance are used. |
| CUDA | No qualified public fitted-export route in this delivery sequence | Deferred until Python/Rust, the one C++ emitter under both standards and JavaScript are qualified. A training GPU does not establish CUDA inference parity. |

Source anchors: `crates/jope-cli/src/kernel.rs`, `crates/jope-pipeline/src/emit.rs`, `crates/feat-math/src/{transpile,emit_py,emit_rs,emit_cpp,emit_js}.rs`, `crates/feat-math/src/advanced_v3/fitted_cpp.rs`, and the pending `advanced_v3/fitted_portable.rs` plus `crates/jope-bench/src/panel/inference.rs`, which are absent from merged main `a75e543d2` and exist only as stopped, unintegrated custody patch views ([§](catalogue.md#pending-source-references)).

## Complete registered-recipe matrix

### Coverage by recipe and backend

The table covers all 26 task-compatible native benchmark recipes. **Yes means the source admits the recipe/head under the additional state restrictions; it is not a measured parity result.** Every row has native Rust fit/save/reload support in the benchmark contract. Public fitted columns come from the pending product packet, which is not in merged main `a75e543d2`; they are unverified against merged source and require actual selected-state qualification. The private C++ column describes the head gate that exists at `a75e543d2` only. Legacy `jope emit` has no column because it can emit none of the 26 recipes: each carries an `input` stage (`crates/jope-bench/src/panel/methods.rs:20-21`, `:50`), which the legacy planner rejects (`crates/feat-math/src/spec.rs:383-389`, `:592`). External adapters are covered in [§](#external-winners).

| Task | Registered method ID | Private C++ scalar head gate | Public Python/Rust fitted | Public C++17/20 fitted | Public JavaScript fitted |
|---|---|---|---|---|---|
| Regression | `rust.ridge` | Yes | Yes | Yes | Yes |
| Regression | `rust.ridge_gcv` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.elasticnet` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.histgbm` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.histgbm2` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.ksmooth` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.ensemble` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.krr` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.gpr` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.hurdle2` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.stack` | Yes | Unsupported | Unsupported | Unsupported |
| Regression | `rust.tab_boost` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.extra_trees` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.gam_boost` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.pls` | Yes | Yes | Unsupported | Unsupported |
| Regression | `rust.portfolio` (`core_v1`, `preset=core`) | Yes, members checked | Unsupported | Unsupported | Unsupported |
| Regression | `rust.gpr.rbf_linear` | Yes | Unsupported | Unsupported | Unsupported |
| Binary | `rust.logit_ridge` | Yes | Yes | Yes | Yes |
| Binary | `rust.tab_boost` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.extra_trees` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.gam_boost` | Yes | Yes | Unsupported | Unsupported |
| Binary | `rust.portfolio` | Yes, members checked | Yes | Unsupported | Unsupported |
| Multiclass | `rust.multiclass_ridge` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_ovr_logistic` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_diagonal_lda` | Unsupported matrix ABI | Yes | Unsupported | Yes |
| Multiclass | `rust.multiclass_nearest_centroid` | Unsupported matrix ABI | Yes | Unsupported | Yes |

This yields static public coverage of 14/26 recipes for Python/Rust, 2/26 for each C++ standard and 6/26 for JavaScript. The existing private C++ head gate admits 22/26 (every regression and binary recipe); it is neither the public bundle gate nor a parity result. Those counts are not percentages of datasets, winners or actual successful parity runs. Python/Rust's twelve unsupported registered recipes are legacy regression heads outside the public fitted lowering. Generated programs can introduce additional unsupported state even when their final head appears in a supported row.

The registry records aliases separately: regression ridge has the historical alias `logistic`, histogram boosting has `gbm`, the ensemble has `ensemble_ridge_gbm`, and binary logit ridge has `logistic`. Aliases do not consume extra fit slots. `gpr.rbf_linear` remains a distinct configuration. Do not infer task compatibility from an ambiguous alias; use task plus exact registered recipe identity.

Source anchors: `crates/jope-bench/src/panel/methods.rs::rust_methods`, `advanced_v3/fitted_cpp.rs::{supports_fitted_cpp17,check_head}`, pending `advanced_v3/fitted_portable.rs::{fitted_portable_unsupported,fitted_javascript_unsupported,fitted_public_cpp_unsupported}` and `advanced_v3/fitted_portable/scalar.rs::supports`.

### State restrictions beyond the head table

The pending public lowering validates the full model binding and fitted input dimensions. It admits IID rows and fitted IID row order only. Historical state, other model forms, target mappings, nonempty general transform lists and calibration are currently unsupported. Categorical encoding must be the admitted fitted one-hot mode. Missing-aware state is limited to the supported tabular-boosting and extra-tree implementations. Invalid coefficients, schema bindings or dimensions are errors, not legitimate unsupported fallbacks.

The benchmark's registered recipes use numeric median imputation, categorical one-hot encoding, missing indicators, `min_count=1` and `max_onehot=256`. Their fitted dictionaries and imputation statistics are part of the state. A deployment must submit columns in the recorded order and use the recorded raw string/null representation. An unseen category must follow the fitted input policy; neither the host nor caller should invent a new category mapping. Binary probability outputs include the two original class names and order; multiclass outputs include every original class column.

The existing private C++ library has broader checks for target mappings, transforms, calibration and ordered/raw input than the public fitted gate. Its broader admission must not be used to skip public privacy, raw-boundary or numerical qualification. The selected public subset is a deliberate current boundary, not evidence that every other native model is intrinsically unexportable.

### External winners

XGBoost, LightGBM, CatBoost and scikit-learn are executable benchmark adapters for the admitted tasks. Their native fitted artifact retains the estimator pickle and preprocessing/reload evidence. The pending public winner-export helper reports standalone Python, Rust, C++17, C++20 and JavaScript export of that external library state as **typed unsupported**. It preserves the external winner, all losses and native fitted artifact. Installing a library in Python does not make its pickle a dependency-free Python inference bundle.

## Private worked examples and their measured execution

Examples A and B are runbooks that can be executed at merged main. Their runbook summary:

| Runbook part | Examples A and B |
|---|---|
| Purpose and status | Show a legacy recipe fit, five-language recipe emission with refit and a bit comparison (A), and the shape of the legacy pipeline search (B). **available**: all fourteen packet commands passed on 23 September 2026 ([§](#measured-execution-of-the-private-examples)). |
| Prerequisites and resources | A checksum-verified immutable CLI (`J`), the reviewed source checkout (`R`) for the drivers and helpers, `python3`, `rustc`, `g++` and `node`, and a new durable private output directory (`A`) outside Git and `/tmp`. Numerical work needs its own allocation; the executed run declared one CPU, 4 GiB memory, no swap or GPU, no network and 300 seconds in total. |
| Manifest-creation command | None. The packet `examples/commands.json` is a committed, hand-prepared file, and no committed runner executes it; the 23 September run used a private controller retained in custody. `examples/execution-receipt.json` points from the packet to that run's receipts. |
| Execution | The listings below, or the argument arrays of `examples/commands.json` with their placeholders filled in. |
| Artifacts and success | For A: `fit-report.json`, `emit-report.json`, five `pred-*.txt` files, `predictions.bits` and `recipe-parity.json`. For B: `pipeline-result.json` and the pipeline's `receipt.json`. Success means every command exits zero, the validator finds all five prediction files with exactly eight finite, bit-identical values, and receipt verification passes. |
| Failure and resume | A nonzero exit is a failure; keep its stderr. The packet has no resume step: rerun it into a new output directory. |
| Limitations | Recipe refits on one synthetic fixture and one legacy pipeline fixture only. No saved-fitted-state export, neural ASM, panel or training qualification. |

### Example A: a tiny private recipe, fit and predictions

#### The synthetic fixture

The supplied `examples/gp-synthetic.json` contains 24 explicitly synthetic rows with two integer, non-null features and target `3*x0 − 2*x1 + 5`: 16 training rows, four validation rows and four test rows. It is a maintainer example, not one of the frozen thirty datasets or the official ten generated fixtures. `train.csv` contains the same sixteen training rows as headerless `x0,x1,target`; `predict.csv` contains the eight validation/test feature rows with targets removed.

One training row is `-12,-5,-21`; the corresponding typed JSON row is:

```json
{"id":0,"features":[-12,-5],"target":-21,"split":"train","group":null,"timestamp":null,"series":null}
```

#### Fit and emit the recipe

Set `R` to the repository root, `E` to `$R/docs/internal/asm-manual/examples`, `J` to a checksum-verified immutable CLI and `A` to a new durable private output directory. Execute numerical work only after its allocation is available. Executed on 23 September 2026: all fourteen packet commands of Examples A and B passed ([§](#measured-execution-of-the-private-examples)). The listings are reproducible instructions, not receipts; `mkdir` and the two `cat` concatenations are setup steps outside the fourteen:

```sh
mkdir "$A"
"$J" fit --input "$E/gp-synthetic.json" \
  --genome '(pipe X (standardize) (ridge))' --json > "$A/fit-report.json"
"$J" emit --genome '(pipe X (standardize) (ridge))' \
  --lang all --output "$A/recipe" --json > "$A/emit-report.json"
```

`fit-report.json` records training/holdout R² for this numeric legacy path. The holdout prefers validation rows; the fallback is test rows if validation cannot be loaded. Do not use this convenience path to establish the frozen-panel leakage or classification protocol. It does not persist a complete fitted artifact. `emit-report.json` lists language source sizes, paths and hashes, and each emitted program below fits its own model from `train.csv`.

#### Run and compare the emitted programs

```sh
python3 "$A/recipe/model.py" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-python.txt"
rustc -O "$A/recipe/model.rs" -o "$A/recipe/model-rs"
"$A/recipe/model-rs" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-rust.txt"

cat "$A/recipe/model.cpp" "$R/deploy/ci/emitters/driver.cpp" > "$A/recipe/program.cpp"
g++ -std=c++17 -O3 -ffp-contract=off -fno-fast-math \
  "$A/recipe/program.cpp" -o "$A/recipe/model-cpp17"
"$A/recipe/model-cpp17" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-cpp17.txt"
g++ -std=c++20 -O3 -ffp-contract=off -fno-fast-math \
  "$A/recipe/program.cpp" -o "$A/recipe/model-cpp20"
"$A/recipe/model-cpp20" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-cpp20.txt"

cat "$A/recipe/model.js" "$R/deploy/ci/emitters/driver.js" > "$A/recipe/program.js"
node "$A/recipe/program.js" "$E/train.csv" "$E/predict.csv" "$A/recipe/pred-js.txt"
python3 "$R/deploy/ci/emitters/compare-bits.py" "$A/recipe" > "$A/predictions.bits"
python3 "$E/validate-outputs.py" --recipe-output "$A/recipe" > "$A/recipe-parity.json"
```

The supplied `validate-outputs.py` requires all five prediction files to exist, each with eight finite values, and checks their complete bit patterns before producing a success receipt. The helper compares every existing `pred-*.txt`; by itself it does not enforce that the required five files were produced. Record each command's nonzero exit as failure and retain its stderr. A comparison of only one surviving file cannot qualify the language matrix. The observed exact bit agreement on this ridge fixture does not imply exact JavaScript agreement for arbitrary transcendental operations.

`examples/commands.json` supplies the exact argument arrays and placeholders for a maintainer runner. Compiler and interpreter versions, source hashes, fixture hashes, commands, exit codes and wall/CPU measurements belong in the execution receipt. Preparation and execution receipts are retained separately: `commands.json` keeps its original `prepared, not executed` status field as a historical record, and the execution is recorded in [§](#measured-execution-of-the-private-examples). The packet is not edited after execution. Instead, `examples/execution-receipt.json` points from it to the execution: it names the packet and its SHA-256, the fourteen-command run of 23 September 2026, and the SHA-256 of each execution receipt under `data/example-acceptance/`.

### Example B: actual legacy pipeline/search shape

The existing pipeline takes a locally cached catalog dataset directory with `metadata.json` or `manifest.json` and verified files. A bare CSV and the synthetic JSON above are not that catalog format. Use the committed regression fixture for a reproducible offline command:

```sh
"$J" pipeline run \
  --dataset "$R/datasets/test/pmlb-229-pwlinear" \
  --budget evaluations=4,seconds=60 --seed 7 --rng-seed 7 --cuts 1 \
  --emit rust,python,cpp17,cpp20,js --policy skip \
  --output "$A/pipeline" --json > "$A/pipeline-result.json"
"$J" pipeline verify-receipt --receipt "$A/pipeline/receipt.json" --json
```

At this budget the command measures four legacy candidates on one prepared cut: generation zero's bootstrap genome `(pipe X (ridge))` plus three grammar-random draws. No evolution runs: generation zero has `POPULATION = 12` slots, and offspring are bred only in later generations, which need an evaluation budget above twelve (the default budget is `evaluations=64,seconds=300`). The command then selects the lowest mean validation RMSE, emits supported recipe sources and writes a pipeline receipt. Sources: `crates/jope-pipeline/src/search.rs:75-80`, `:466-484`; `crates/jope-pipeline/src/select.rs:18`; `crates/jope-pipeline/src/cli.rs:38`. `--policy skip` explicitly records the absent neural proposal stage; `auto` also records a skip and `require` fails closed in this legacy route. It is not the qualified neural ASM arm.

The legacy budget is per cut, combines exploration/comparison attempts and active seconds, and is nonpreemptive for an in-flight fit. Multiple cuts can consume multiple per-cut allowances. It is not the twenty-worker panel's hard isolated 30-second fit and original global deadline contract. Preserve the recorded `deadline_overrun` and stage counts rather than applying official panel claims to this command.

The emitted winner may not be covered by C++ or JavaScript. The receipt retains exclusions and their reasons; it never replaces the selected recipe solely to make all requested languages emit. Reusing the same C++ source under `-std=c++17` and `-std=c++20` establishes the intended single-emitter structure. The maintained `deploy/ci/emitters.sh` and `deploy/ci/e2e-smoke.sh` provide larger qualification workflows; running them builds and executes models and requires its own resource allocation.

### Measured execution of the private examples

#### What passed and what was measured

On 23 September 2026, all **14 unchanged example commands passed**, with zero failed, unavailable, timed-out or unstarted commands. The original command packet and static-only preparation receipts are retained unchanged as historical inputs; this execution is separate evidence.

The immutable CLI came from reviewed source `68f56404c3e199e06413bfd645d3fe0803d7b96f`; merged main `a75e543d2166017cb0139e0a251d51d9a0c98e08` has the identical source tree `176b75bc36d76c7ce4eaeb7234677ec013468889`. The executable SHA256 was `eab2d5b1b05629020aab3d540d51eb24ba0db18ff0917da188f89a34b6dc5e2b`. The original fourteen-command packet SHA256 was `f34771938934d1360cbe054689393198a53929479ba119f5a0faf2b0da7a170a`.

The private synthetic fixture contains **24 rows: 16 training, four validation and four test**. The recipe `(pipe X (standardize) (ridge))` achieved training R² **0.9995230651** and validation R² **0.9918641493**. Its emitted Python, Rust, C++17, C++20 and JavaScript programs each fitted from the same training CSV and produced **eight finite predictions with identical IEEE-754 bits**. A separate validator required all five output files and all eight rows, so a missing backend could not pass by omission.

The packet also ran the separately identified committed **PMLB 229 pwlinear** legacy pipeline fixture (`1cb97b1d7314d43d`). It completed four successful candidate measurements in one generation with no failed measurements (`stopped_by: evaluations`), selected the bootstrap genome `(pipe X (ridge))`, reported validation RMSE **0.0775685574** and R² **0.8151398564**, and passed receipt verification. Its policy stage was explicitly skipped. This fixture is not the synthetic twenty-four-row example and this invocation is not the official thirty-dataset campaign.

#### Resources, accounting and scope limits

| Measurement | Recorded value |
|---|---|
| Controller elapsed time | 10.407138 seconds |
| Launcher elapsed time | 10.506140 seconds |
| Exact systemd unit CPU | 9.382049 CPU-seconds |
| Fourteen-command child CPU subtotal | 2.952963 CPU-seconds |
| Python | 3.12.3 |
| Rust | 1.97.1 |
| G++ | 13.3.0 |
| Node.js | 26.1.0 |

The unit CPU includes overlapping command/controller work; **do not add it to those subtotals**. The separately granted allocation declared one CPU, 4 GiB memory, zero swap/GPU, network denial and 300 seconds total, with 30-second numerical-command and 60-second compiler limits. The shared exclusive CPU lock was held and released; all command process groups drained. CPU affinity `[0]` and the dedicated unit cgroup were observed. A live property readback raced unit collection and returned unloaded defaults, so independent readback of memory/swap/quota/network/device limits is **not established**. The unit reported 5,332,992 bytes peak memory and zero swap, while cumulative reaped-child maximum RSS was 168,808 KiB; retain these different accounting domains without claiming a reconciled total peak.

This result qualifies the documented **recipe refits and legacy pipeline example**. It does not qualify exact saved-fitted-state export, neural ASM consumption, an ASM under-sixty-second warm demonstration, GPU training or the full panel. No Cargo build or download ran. Full command outputs, tool/input/source/output hashes and the original unit journal are retained in the private `agent-maintainer-manual/examples-run/` evidence directory.

Evidence pins: `HANDOFF.json` SHA256 `a05e8a0e9f9430b0c71ff23c57e5e29532a61bbe118daac3fe358b45f2423346`; `audit.json` SHA256 `363fbe62a706b44030205da78411243ba206ad2366d546a277babe966b802c91`; `result.json` SHA256 `71395a038feeef768bd6c69bb43dcabbfced20333d989cd1161c07213f4f68d6`. Byte-identical copies of these three files are in `data/example-acceptance/`, and `examples/execution-receipt.json` records the same pins beside the packet's hash.

## Example C: exact fitted winner export after pending integration

This runbook is the route intended to meet the rule of [§](#deployment-support-definition-priority-and-status). Its commands exist only in stopped custody patch views, and none of it has been executed. Its runbook summary:

| Runbook part | Example C: exact fitted winner export |
|---|---|
| Purpose and status | Export the selected winner's saved fitted state to Python, Rust, C++17, C++20 and JavaScript, and qualify each backend against the native predictions, without refit. **pending integration**: `benchmark export-winner` and `benchmark qualify-export` are absent from merged main `a75e543d2`, and the pending view compiles them only with the `asm` feature on Linux. |
| Prerequisites and resources | The same integrated immutable installed CLI, approved plan and plan ID, manifest, development root and journal as the campaign, with complete confirmation; a pinned tools file ([§](#tool-inventory-qualification-order-and-isolation)); and the original campaign's remaining allowance. Each job has one thread and at most the original 30-second fit limit, inside the campaign's existing supervisor and worker cgroup. |
| Manifest-creation command | None of its own: the export consumes the campaign's existing plan, manifest and journal. No manifest-creation command exists at `a75e543d2`, or in the pending export view, for the tools file; it must be prepared from the installed bytes, in the campaign `Runtime` schema. |
| Execution | The three commands of [§](#export-commands-and-winner-selection), with backends qualified in the order Python, Rust, C++17, C++20, JavaScript. |
| Artifacts and success | The semantic prepared identity, the private `prepared.json`, and a public bundle root with each validated bundle and its public manifest. A backend succeeds only through an executed qualification that meets its numerical policy; `all_backends_qualified` is true only when all five pass. |
| Failure and resume | A compiler absence, timeout, crash or parity failure blocks the later backends, and a failed comparison stays `failed_parity`. Started work without a finished receipt stays unknown until shared terminal recovery. Repeating the command cannot give an expired campaign a new allowance, and report recovery is read-only. |
| Limitations | Static public coverage is 14 of 26 recipes for Python/Rust, 2 for each C++ standard and 6 for JavaScript ([§](#coverage-by-recipe-and-backend)). External-library winners are typed unsupported. Fold-zero parity does not prove the other four folds. No separate post-panel validation stage exists. |

### Export commands and winner selection

Use the final integrated installed CLI, original admitted plan and journal, unchanged panel/development inputs and exact selected recipe. The pending front door, absent from merged main `a75e543d2`, accepts a dataset and an existing fitted fold, rather than a replacement recipe:

```sh
"$J" benchmark export-winner \
  --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" \
  --journal "$JOURNAL" --dataset "$DATASET_ID" --fold 0
"$J" benchmark qualify-export \
  --prepared "$PREPARED_EXPORT" --prepared-sha256 "$PREPARED_EXPORT_ID" \
  --tools "$TOOLS" --tools-sha256 "$TOOLS_FILE_SHA256"
"$J" benchmark report \
  --plan "$PLAN" --plan-sha256 "$PLAN_ID" --manifest "$M" --datasets "$D" \
  --journal "$JOURNAL"
```

The first command returns the semantic prepared identity, private `prepared.json` and public bundle root. It chooses the smallest complete five-fold mean loss, breaking ties by method identity. Incomplete confirmation has no final winner. Choose another existing fold only when that exact fitted state is the intended qualification target; fold-zero parity does not prove the other four states.

### Tool inventory, qualification order and isolation

The tools file uses the existing private `Runtime` schema keyed by command name, normally `python3`, `rustc`, `g++` and `node`. Each entry pins the physical executable and SHA256, exact version, empty argument prefix, required toolchain/package artifacts, admitted environment and unavailable reason or null. Use the real compiler executable, not an unpinned selector such as a mutable rustup shim. Tool inventory must be prepared from actual installed bytes; an illustrative JSON snippet with invented digests is not admissible.

Qualification order is Python, Rust, the same C++ emitter under C++17 then C++20, then JavaScript. Unsupported settles that backend's inventory without replacing the winner. A compiler absence, crash, timeout or parity failure prevents silently proceeding as though that backend qualified. The complete probability matrix, row order and class order are compared, not just the winning class or a rounded score.

The child receives generated inference files and unlabeled raw prediction rows. Expected predictions, original scoring labels and dataset-root access stay with the controller. Each job uses the existing bounded worker supervisor. Report recovery checks source/tool/input/artifact/command/receipt identities and runs no compiler or predictor. Started work without a finished receipt remains unknown until shared terminal recovery; it is not a successful retry.

### Backend numerical policies, public metadata and the export clock

| Backend | Public files | Pinned numerical comparison |
|---|---|---|
| Python | `model.py` | Exact IEEE-754 bits, host platform libm. |
| Rust | `model.rs` | Exact IEEE-754 bits, host platform libm. |
| C++17 / C++20 | `model.cpp`, `codec.json`, `boundary.py` | Exact IEEE-754 bits through the typed Python/ctypes boundary; no fast-math/FMA contraction drift. |
| JavaScript | `model.js` | Absolute/relative `1e-12` policy with explicit ECMAScript Math identity. |

Necessary public metadata records the task, ordered columns/classes, fitted raw-cell/input policy, emitter/template hashes, runtime, numerical policy and file checksums. The initial bundle is `emitted-unqualified`. `all_backends_qualified` becomes true only after five actual successful qualifications. An operation can complete with explicit unsupported outcomes while that all-backends flag remains false.

These jobs use the original campaign's remaining execution allowance; they do not open a new export clock. For the official two-hour panel, no new export job may start after original `t0 + 6900 seconds`, and the last five minutes remain reporting reserve. If confirmation consumes the window, preserve unattempted export statuses. A future separately bounded post-panel validation stage would need its own explicit identity and cost accounting; the current front door does not implement one.

## Future work and acceptance boundaries

Priority gaps are actual full-state parity for supported winners, broader fitted-state lowering, a typed multiclass C++ boundary, qualified external-library export where feasible, and only later CUDA. Each extension needs exact native-state replay, missing/unknown-category tests, numerical policy fixed before scoring, stale-evidence refusal and public-bundle canaries. New operators additionally need ablation and compatibility evidence; adding an emitter branch alone does not establish reusable learning value.

Two existing emitter options are not exposed by any command. The legacy emitters' default math policy, `platform-libm-v1`, lowers transcendentals to the host math library, so exact parity does not extend to JavaScript or CUDA. The reserved `portable-scalar-kernel-v1` policy makes the Python/Rust pair fail closed until every emitted transcendental is routed through the portable kernel (`crates/feat-math/src/transpile.rs:51-80`, `:176-187`), while the C++ and JavaScript emitters accept it and substitute portable math helpers (`crates/feat-math/src/emit_cpp.rs:48-52`, `:72-81`; `crates/feat-math/src/emit_js.rs:37-45`); only in-source tests select it. `EmitOptions.fast_runtime` exists, but neither `jope emit` nor `jope pipeline run --emit` exposes it (`crates/jope-cli/src/kernel.rs:78-81`; `crates/jope-pipeline/src/emit.rs:6-7`, `:122`).

Retain native winners even when every requested language is unsupported. Report the full loss/ranking and export gap together. Performance claims need measured creation/prediction time on the actual artifact and hardware; binary size, source-line count, emission throughput or a passed static support function cannot substitute for those measurements.
