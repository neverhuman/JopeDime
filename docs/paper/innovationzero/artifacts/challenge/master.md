# The General Regressor Challenge

## A neutral, first-principles master prompt and submission contract

**Deliverable:** one self-contained regression algorithm, implemented entirely in safe Rust, designed without data experiments, frozen once, and ranked by Elo across thousands of unknown datasets. **The final column of every complete tabular dataset is always the target.**

The participant is given only a mathematical building-block allowlist, an input/output contract, and competition rules. No existing system architecture, fitted artifact, model recipe, recommended family, preprocessing strategy, parameter setting, performance result, or benchmark identity is supplied.

**Use:** give a fresh model the complete text between `BEGIN MASTER PROMPT` and `END MASTER PROMPT`. The organizer appendix records the red-team findings and evaluation controls; do not include that appendix in the participant prompt. Any common campaign card must be settled before collecting submissions.

This specification calls for working source code. It does not ask the participant to choose a configuration from an existing model catalog. Mathematical constructions written by the participant are allowed when their implementation uses only the permitted fundamentals.

---

<!-- BEGIN MASTER PROMPT -->

# Challenge: invent one general-purpose regressor

## 1. Objective

Write the strongest general-purpose regression algorithm you can derive from first principles using only the permitted fundamentals below.

You must deliver **one complete, self-contained implementation**. It will be frozen and evaluated, without your involvement, on **thousands of unknown regression datasets**. The competition ranks algorithms by **Elo**, based on their comparative predictive performance on held-out data.

**Your algorithm must be 100% Rust. The last column of a complete dataset is always the single numeric target; all preceding columns are features.** This column position is a fixed contract, not something to discover from values or names. At prediction time, the evaluator removes the target column entirely and supplies features only.

You will receive no datasets, dataset descriptions, dataset encodings, benchmark examples, existing implementations, reference architectures, recommended model families, performance tables, or feedback about what works.

Your algorithm must be your final answer. Make all design decisions now. Supply code that can actually fit a model and produce predictions.

## 2. What “one algorithm” means

One submission is one immutable implementation of a learning procedure. If a complete table has `C` columns, then `P = C - 1` is the feature count:

\[
T_{train}=[X_{train}\mid y_{train}],\qquad
\operatorname{fit}(T_{train},seed)\rightarrow M,
\qquad
\operatorname{predict}(M,X_{new})\rightarrow\hat y.
\]

The same source code is used on every dataset. Each dataset receives a newly created instance and independently fitted state. No information persists between datasets.

The algorithm may estimate parameters and make general decisions from its current training data. Such behavior must be fully specified in the submitted source. A rule based on the current training matrix is part of the algorithm; a rule that recognizes a particular benchmark and selects a special answer is prohibited.

“One algorithm” does not mean one numerical coefficient or one fixed prediction formula. It means one complete, predetermined learning procedure with one public fitting interface and one prediction interface. You choose its internal organization. Every internal computation must be implemented from the allowlist and charged to the same resource budget.

Do not provide several alternatives, ask the evaluator to select among submissions, defer a design choice, or request a later tuning conversation.

## 3. Design-time restrictions

You must design the algorithm by reasoning, without empirical experimentation.

Symbolic derivations and reasoning about API or numerical edge cases are allowed. Executing a candidate or choosing it from observed predictive performance is not.

### 3.1 Data and feedback you must not use

You must not:

- Obtain, inspect, download, reconstruct, or request any real dataset for this task.
- Use dataset embeddings, summaries, examples, labels, statistics, or benchmark identities supplied outside this prompt.
- Generate synthetic datasets to evaluate, train, tune, or compare your design.
- Run any candidate on real, synthetic, toy, or memorized tabular data.
- Read existing benchmark results or use a leaderboard to choose the design.
- Train or fine-tune yourself, another model, or a surrogate to help solve this challenge.
- Ask another model, agent, service, or person to produce, evaluate, or select a candidate.
- Browse the web, retrieve source code, inspect another implementation, or use an external codebase.
- Import pretrained weights, fitted parameters, data-derived tables, or previously optimized configurations.

You may use general mathematical knowledge already available to you and derive whatever method you believe will work. You do not need to invent new mathematics. You do need to write the submitted implementation yourself from the allowed fundamentals.

### 3.2 Uniform authoring policy

The default authoring mode is **reasoning only**: no tools, browsing, code execution, data access, or experiments. Reason through your implementation and submit it once.

An organizer-issued campaign card may instead enable **compilation only** for every entrant in that campaign. In that mode, each entrant receives the same offline compiler service, pinned compiler, interface check, maximum of three compilation requests, and maximum of two compiler-feedback revisions before final freeze. A model's private or provider-specific tool access does not grant additional permission. Do not mix these two authoring modes within one model comparison.

Permitted compilation feedback is limited to syntax, type, ownership, and linking/interface diagnostics from your own source. You may not execute the regressor, call its constructor, fitting or prediction methods, run numerical probes, measure algorithm speed, construct a test dataset, or inspect unrelated files.

The same prohibition applies during compilation: no dataset generation, candidate evaluation, empirical tuning, or learned-artifact construction through constant evaluation, macros, build scripts, or compiler output. Ordinary compile-time calculation of fixed mathematical constants and storage sizes is allowed. No external source or dependency may be loaded.

Do not claim to have compiled or tested anything you did not compile or test. Post-submission correctness checks are performed by the organizer after your source is frozen and supply no feedback for revising it.

### 3.3 Later fitting is allowed

After the submission is frozen, the evaluator will run your algorithm on training data. That is the intended use of your code and is allowed.

The prohibition is on **empirical feedback during your design process**, not on the submitted algorithm learning from training rows when it is later evaluated.

All fitted statistics, learned parameters, internally selected settings, and stopping decisions must depend only on the current fitting call's training inputs, its supplied seed, and constants present in your frozen source. They must never depend on outer test targets or on results from another dataset.

## 4. Closed-world permission rule

Your implementation may use only:

1. The scalar values, arithmetic, comparisons, and mathematical functions in Section 5.
2. The ordinary data structures and control flow in Section 6.
3. The narrow standard-library facilities in Section 7.
4. Functions that you implement yourself by composing those permitted fundamentals.
5. The input arrays and seed supplied through the public interface.

**Everything else is forbidden unless expressly listed.**

A function you implement from permitted operations is allowed even if that function is not named in this prompt. The allowlist limits the underlying building blocks; it does not limit the mathematical algorithms you may derive from them.

An external function is not allowed merely because you could have implemented it from the fundamentals. You must actually supply the implementation.

No open-source dependencies or borrowed implementations are permitted. The standard compiler/runtime and the explicitly listed standard-library utilities are the sole infrastructure exception. They may provide elementary arithmetic, storage, and control facilities; they may not supply your learning algorithm.

## 5. Complete mathematical-fundamentals allowlist

These permissions describe elementary capabilities. They do not provide a numerical-method library. Where a facility is not directly supplied by the permitted Rust standard library, write it yourself using the permitted operations.

**Every listed fundamental is available; using every fundamental is not required.** Choose whatever composition you judge best. Availability is not a recommendation about the algorithm's structure.

### 5.1 Values and literals

| Fundamental | Permission |
|---|---|
| Real scalars | IEEE-754 binary64 (`f64`) |
| Integer scalars | Rust fixed-width signed/unsigned integers and `usize`/`isize` for indexing and bookkeeping |
| Boolean scalars | `true`, `false` |
| Numeric literals | Finite integer and real constants written explicitly in source |
| Mathematical constants | Finite constants derived from mathematics, or the standard `f64` mathematical constants |
| Missing-value representation | `f64::NAN` is the input missing-value marker; it is not a valid prediction |
| Infinity values | May be used as internal sentinels; final predictions must be finite |
| Strings | Error messages and source-level metadata only; not an external data or code channel |

Constants must serve a general numerical or algorithmic purpose. Encoded datasets, pretrained coefficient arrays, benchmark lookup tables, compressed model weights, and executable payloads are forbidden regardless of their representation.

### 5.2 Scalar arithmetic and comparisons

| Group | Permitted operations |
|---|---|
| Arithmetic | Addition, subtraction, multiplication, division, remainder, unary sign |
| Magnitude | Absolute value, minimum, maximum, clamping |
| Comparison | `<`, `<=`, `>`, `>=`, `==`, `!=`; total floating ordering for sorting |
| Boolean logic | Conjunction, disjunction, negation |
| Rounding | Floor and nearest-integer rounding; other rounding rules may be composed from the listed operations |
| Numeric conversion | Explicit checked or ordinary numeric casts, with validity handled in your code |
| Numeric inspection | Finite/NaN/infinite tests, sign inspection, floating bit representation |
| Integer bookkeeping | Checked, saturating, and wrapping arithmetic; bitwise operations; shifts and rotations |

No hidden numerical safeguards are supplied. You are responsible for division by zero, invalid domains, integer overflow, floating overflow, underflow, and loss of numerical precision. If you use a protected version of an operation, its rule must be visible in your code.

### 5.3 Complete scalar-function inventory

The following table explicitly names every scalar function in the fundamental inventory. The formulas describe the mathematical operation, not a prescribed numerical approximation or a prescribed feature transformation.

| Fundamental | Mathematical meaning | Permitted implementation |
|---|---|---|
| `abs(x)` | Absolute value | `f64::abs` or your own arithmetic |
| `sqrt(x)` | Square root on its real domain | `f64::sqrt` or your own arithmetic |
| `log1p(x)` | Natural logarithm of `1 + x` | `f64::ln_1p` or your own arithmetic |
| `sin(x)` | Sine, argument in radians | `f64::sin` or your own arithmetic |
| `cos(x)` | Cosine, argument in radians | `f64::cos` or your own arithmetic |
| `tanh(x)` | Hyperbolic tangent | `f64::tanh` or your own arithmetic |
| `sigmoid(x)` | `1 / (1 + exp(-x))` | Implement in your source from permitted operations |
| `min(a,b)` | Smaller of two real values | `f64::min` or comparisons |
| `max(a,b)` | Larger of two real values | `f64::max` or comparisons |
| `clip(x,lo,hi)` | Clamp to a closed interval with ordered bounds | `f64::clamp` or comparisons |
| `step(x)` | Indicator of a threshold crossing | Implement with a comparison; document the threshold and equality convention |
| `exp(x)` | Natural exponential | `f64::exp` or your own arithmetic |
| `log(x)` | Natural logarithm on its real domain | `f64::ln` or your own arithmetic |
| `asinh(x)` | Inverse hyperbolic sine | `f64::asinh` or your own arithmetic |
| `atan(x)` | Inverse tangent | `f64::atan` or your own arithmetic |
| `erf(x)` | Error function: \(2/\sqrt{\pi}\int_0^x e^{-t^2}\,dt\) | Implement your own approximation from permitted operations |
| `sign(x)` | Negative, zero, or positive sign represented numerically | Comparisons and literals; state your zero convention |
| `floor(x)` | Greatest integer no larger than `x` | `f64::floor` or your own arithmetic |
| `round(x)` | Nearest-integer rounding | `f64::round`, or your own stated tie convention |

Also permitted are unary negation and binary addition, subtraction, multiplication, division, and power. For power, `f64::powf` and `f64::powi` are allowed. Numeric inspection, casts, bit access, and integer bookkeeping remain allowed as listed in Section 5.2.

This is the complete direct mathematical-function boundary. Do not assume that every method on Rust's numeric types is authorized. Functions beyond this list may be constructed in your own source from permitted operations; an additional standard-library or external numerical call is not automatically allowed.

No special-function library, distribution library, numerical integration library, symbolic engine, or automatic differentiation library is available. In particular, permission to use the mathematical error function does not authorize an external implementation or an unlisted compiler-version-specific method.

You decide and implement numerical guards, approximation accuracy, missing-value handling, and invalid-domain behavior. There are no hidden clipping constants or supplied protected-operation rules. These choices must be visible in the frozen source and produce finite predictions on successful calls.

### 5.4 Scalar access and row summaries

The available access and summary fundamentals are:

| Fundamental | Meaning |
|---|---|
| Current scalar | A numeric value currently being processed |
| Indexed feature column | Any feature coordinate selected by a valid index `0 <= j < P` |
| Row sum | Sum over the row's feature values |
| Row mean | Arithmetic mean over the row's feature values |
| Row L1 magnitude | Sum of absolute feature values |
| Row L2 magnitude | Square root of the sum of squared feature values |
| Row minimum | Minimum feature value |
| Row maximum | Maximum feature value |
| Row standard deviation | Square root of the mean squared deviation from the row mean |

Implement these yourself with arrays, loops, and permitted scalar operations. If you use a summary, define how it treats missing features, an empty feature set, and nonfinite intermediate results. The definitions above describe the usual finite, nonempty case.

**The target is not a feature. A feature-row summary must never include the final target column of a training table.** Training labels may be used inside `fit` to learn the model, as specified in Section 9; they may not be smuggled into a predictor input or a feature formula that requires an unavailable label at inference.

### 5.5 Arrays, reductions, ordering, and basic linear algebra

You may build the following computations yourself from the permitted scalars, arrays, and loops:

| Fundamental construction | Permission |
|---|---|
| Vector/matrix storage | Dense arrays, index arrays, and sparse representations implemented in your code |
| Element access | Read, write, copy, gather, scatter, slice, concatenate, reshape by index, transpose |
| Elementwise maps | Any formula composed of permitted scalar operations |
| Reductions | Sum, product, minimum, maximum, count, and combinations of those operations |
| Ordering | Sort, select, partition, binary search, and stable index permutations |
| Summary calculations | Counts, moments, ranks, quantiles, and other quantities you explicitly derive and implement |
| Vector arithmetic | Addition, scaling, dot products, outer products, and distances you implement |
| Matrix arithmetic | Addition, multiplication, transposition, and other formulas you implement |
| Numerical routines | Any finite routine you write entirely from the permitted primitives |

There is no supplied matrix inverse, factorization, solver, optimizer, fitting routine, model selector, or regression estimator. You may implement whatever mathematics you need yourself. Calls to external equivalents are prohibited.

This table is a permission to compose arithmetic, not a suggestion that any particular construction belongs in your solution.

### 5.6 Deterministic pseudorandomness

You may implement a pseudorandom generator using the permitted integer operations and the supplied seed. You may derive additional numeric draws using permitted scalar mathematics.

You may not use operating-system entropy, random-device files, a random library, wall-clock time, hardware identifiers, process identifiers, or an unspecified global random source.

All random choices must be reproducible from the supplied seed and the current training inputs. Repeated predictions from the same fitted state must not resample the model.

## 6. Complete programming-fundamentals allowlist

**All participant-authored algorithm code must be safe Rust, edition 2021, in one source file named `submission.rs`.** This includes preprocessing, scalar-function approximations, numerical routines, learning, prediction, state management, and every helper. No part may be implemented in, generated as, or delegated to another language.

The approved Rust compiler and standard runtime are infrastructure. Their ordinary platform implementation of an explicitly permitted primitive does not make your submission a mixed-language algorithm. This exception does not authorize your own foreign linkage or access to any additional library.

Permitted language facilities:

- Functions and methods defined in the submitted source.
- Closures defined in that source, used only with permitted operations.
- Local variables, immutable constants, and owned fitted state.
- Structs, enums, tuples, fixed-size arrays, slices, references, and generics.
- `if`, `else`, `match`, `for`, `while`, `loop`, `break`, `continue`, and `return`.
- Recursion, provided it terminates and respects the resource budget.
- Ordinary traits and trait implementations written in the same source file.
- Bounds-checked indexing and safe mutable access.
- `Option` and `Result` for explicit state and error handling.
- Private helper functions and private modules written inline in the same file.
- Built-in derives listed in Section 7 and ordinary lint, `inline`, and `forbid(unsafe_code)` attributes.
- Deterministic internal decisions based on the current training inputs.

All learning behavior must be present in the file. There may be no unfinished function, TODO, placeholder, stub, missing module, or instruction to install another component.

No `unsafe` code, inline assembly, foreign function interface, custom linkage, dynamic loading, subprocess, executable generation, external source inclusion, global mutable state, or hidden runtime callback is permitted. Feature flags, target-dependent implementations, custom compiler features, and participant-defined macros are unavailable. Use ordinary functions and closures for reusable computations.

The submitted implementation must be single-threaded. You may not start threads, use asynchronous workers, spawn processes, or invoke accelerator APIs.

## 7. Exact standard-library boundary

The following standard-library facilities are permitted only for the functions described here:

| Facility | Allowed purpose |
|---|---|
| Primitive numeric and Boolean types | The operations listed in Section 5 |
| `Vec`, fixed arrays, slices, `Box` | Owned numeric storage, safe indexing, allocation, copying, iteration |
| `Option`, `Result` | State and error handling |
| `String`, `str` | Error messages and nonexecuting identifiers written in the source |
| Iterators and ranges | In-process traversal and aggregation over permitted values |
| `std::cmp` | Comparisons and ordering |
| `std::mem::swap`, `replace`, `take`, `size_of` | Safe storage management and size accounting |
| `std::collections::BTreeMap`, `BTreeSet`, `BinaryHeap`, `VecDeque` | In-memory numeric/index data structures for the current fitting instance |
| `std::convert`, ordinary conversion traits | Safe type conversion and ownership conversion |
| `std::ops` and ordinary operator traits | Implementing the permitted operations on your own in-file types |
| `std::fmt` | Formatting your own error messages and the listed `Debug` output only |
| `Clone`, `Copy`, `Default`, `Debug`, `PartialEq`, `Eq`, `PartialOrd`, `Ord` derives | Boilerplate for your own types |
| `vec!`, `format!`, `matches!` | Numeric allocation, error formatting, and ordinary pattern checks |

Common collection methods such as allocation, reserve, resize, push, pop, insert, remove, iteration, slicing, copying, sorting, selection, and binary search are allowed. These methods may organize your current numeric state; they may not obtain outside information or invoke prohibited capabilities.

Ordinary prelude traits needed for these facilities, including `Sized`, `Iterator`, `IntoIterator`, `FromIterator`, `From`, `Into`, `TryFrom`, `TryInto`, `AsRef`, and `AsMut`, are allowed for these same purposes. In-memory records may contain your own permitted structs and enums. A permitted module name does not authorize unrelated capabilities from that module.

Use prelude or `std` paths. A compiler-supplied `core` or `alloc` equivalent is allowed only when it denotes the exact same permitted facility; a different path does not expand the capability boundary.

All standard-library capabilities not covered above are unavailable. In particular:

- No filesystem, network, environment-variable, process, time, thread, or OS-interface modules.
- No random hashing infrastructure that obtains a hidden seed; use permitted deterministic structures.
- No `include!`, `include_str!`, `include_bytes!`, `env!`, `option_env!`, external module files, build scripts, or procedural macros from another package.
- No external crates, packages, vendored code, downloaded source, dynamic libraries, WebAssembly modules, or model files.
- No arbitrary standard-library escape hatch just because it is not a third-party package.

The judge owns file loading, serialization, timing, resource enforcement, scoring, and result reporting. Those facilities are not exposed to your algorithm.

## 8. Required callable interface and target-column contract

Include this trait declaration exactly, define a public `Regressor` struct with your own private state, and implement `RegressorApi` for it:

```rust
pub trait RegressorApi: Sized {
    fn new(seed: u64) -> Self;

    fn fit(
        &mut self,
        train: &[f64],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<(), String>;

    fn predict(
        &self,
        features: &[f64],
        n_rows: usize,
        n_features: usize,
    ) -> Result<Vec<f64>, String>;
}
```

This is a valid Rust trait declaration, not a completed submission. Your file must also contain the concrete `pub struct Regressor`, the complete `impl RegressorApi for Regressor`, and all helpers. The evaluator imports `submission::{Regressor, RegressorApi}`. There is no additional target array, file path, callback, or context object in the API.

### 8.1 The final column is always the target

Every complete numeric table uses this exact layout:

| Position | Role |
|---|---|
| Columns `0` through `C - 2`, when present | Input features, in their supplied order |
| Column `C - 1` | The one regression target, always |

The training argument `train` contains the complete training table in row-major order. For training row `i`:

```text
C = n_cols
P = C - 1
feature(i, j) = train[i * C + j]          for 0 <= j < P
target(i)     = train[i * C + (C - 1)]
```

`n_cols` therefore **includes** the target column. You must separate its role before computing feature preprocessing, row summaries, or predictor inputs. Do not infer a different target from column statistics, headings, values, or apparent predictiveness. Do not treat the last feature as a second target.

### 8.2 Training input guarantees

- `n_rows >= 1` and `n_cols >= 1`.
- `train.len()` equals `n_rows * n_cols`, checked without integer overflow.
- The feature count is exactly `P = n_cols - 1`.
- A one-column training table is valid: it contains targets and zero features. Define fitting and prediction for this case without indexing a nonexistent feature.
- Every value in the final column is finite. NaN or infinity in a target is an invalid input.
- Feature entries may be finite or NaN. NaN means missing; positive and negative infinity are invalid feature inputs.
- Entire feature columns or feature rows may be missing. Repeated rows, repeated columns, constants, signed zeros, and single-row training sets are valid.
- Feature and target values are not guaranteed to be centered, scaled, bounded to a small interval, or expressed in comparable units.
- No field names, dataset identifier, category vocabulary, provenance label, task description, or held-out row count is supplied to `fit`.
- The evaluator passes only the assigned training rows. You do not receive the full unsplit dataset.

All input columns have already been represented numerically under one common evaluation contract. There are no strings to parse and no task-specific schema to request. Any organizer-side conversion of nonnumeric source fields must be fixed and applied without held-out-label leakage.

### 8.3 Prediction inputs never contain a target column

The complete source dataset still follows the final-column-target rule. Before calling `predict`, the evaluator removes that column and keeps its held-out values outside the participant process.

`features` is a row-major matrix with exactly `P` columns. Its entry `(i,j)` is `features[i * P + j]`. The argument `n_features` equals the feature count learned at fitting, **not** the training table's total column count.

- No target values, target placeholders, zero-filled target column, or NaN-filled target column are present.
- Do not remove the last column again: every supplied prediction column is a feature.
- `n_features` must equal the fitted `P`; otherwise return an error.
- `features.len()` must equal `n_rows * n_features`, using checked multiplication.
- Feature values follow the same finite-or-NaN convention as at fitting. Infinite feature inputs are invalid.
- `n_rows` may be zero. An already fitted model returns an empty vector for a valid empty batch.
- When `P = 0`, the feature slice is empty even for a nonempty prediction batch. `n_rows` still specifies the number of required outputs.
- Return exactly `n_rows` finite `f64` predictions, in input row order and original target units. Any target transformation your algorithm uses must be undone before returning.

If an input violates the contract, return `Err(String)`. Do not silently truncate, invent columns, reuse a stale shape, or guess a different interpretation. A valid but numerically unsuccessful fitting or prediction call may return an explicit error; it then counts as an algorithm failure, not a successful prediction.

### 8.4 State lifecycle and determinism

`new(seed)` creates an unfitted instance and records the supplied seed. Predicting before a successful fit must return an error, including for an empty prediction batch.

`fit` replaces any earlier fitted state. Each call starts from the original instance seed and the current training table alone. It must not quietly continue an earlier task, concatenate old rows, or inherit an advanced random stream. If fitting fails, the instance becomes unfitted; subsequent prediction must not use partial or stale state.

`predict` must not update learned parameters or use persistent interior mutability. Repeated calls do not change the model or resample its random choices.

A row's prediction must depend on that row and the fitted state, not on the presence, order, number, or content of other prediction rows. The evaluator may split predictions into batches or permute their order. Batch composition must not drive normalization, adaptation, parameter selection, or other learned behavior.

The same source, seed, training rows, and pinned runtime must reproduce the same fitted behavior. Harmless floating-roundoff differences under equivalent batching are assessed with the common conformance tolerance:

\[
|a-b| \leq 10^{-10}\max(1,|a|,|b|).
\]

This tolerance is only a numerical check, not permission to implement batch-dependent behavior. It is not an Elo draw threshold. Cross-compiler bitwise identity is not required.

## 9. What you may learn at evaluation time

Inside `fit`, you may compute any general learning procedure that you implemented using the permitted fundamentals. You may examine the current training features and labels, calculate training-only quantities, and determine the model state.

Any internal partition, parameter estimation, stopping criterion, representation, or adaptive choice must be completely specified in the source before evaluation. If you create an internal validation partition, its held-out labels may influence only the internal decisions your code explicitly defines; they do not give access to the evaluator's outer test data.

All learned preprocessing and all learned target or feature statistics must come from the current outer training partition. If your algorithm relies on internal validation to compare choices, fit the statistics for each such comparison on its internal training rows. Do not let that comparison's held-out labels enter its fitted predictor or preprocessing. After making the predetermined decision, your procedure may fit its final state using all outer training rows.

Inside `predict`, you receive only new feature rows. There is no test-target argument. You may not infer or request test labels through filenames, environment state, ordering conventions, hidden files, network requests, prior calls, or other side channels.

You are free to make mathematically motivated design choices. This prompt does not recommend any architecture, feature transformation, optimization method, model family, or hyperparameter value.

## 10. Resource contract

The same limits apply to every submission:

| Resource | Limit |
|---|---:|
| Source file | At most 262,144 UTF-8 bytes |
| Execution | Single-threaded CPU; one physical core; no GPU |
| Process-tree memory | 8 GiB, including inputs and fitted state |
| Fit budget | 600 seconds elapsed and 600 CPU seconds per fitting job, including construction and participant-side input validation |
| Prediction budget | 60 seconds elapsed and 60 CPU seconds for the entire held-out partition |
| External data or services | None |
| Persistent cross-job storage | None |

These limits are challenge defaults. If the arena supplies an explicit resource card with this prompt, that card replaces only this table and is identical for every participant. Do not assume additional resources. The stable compiler version, target, flags, and hardware are pinned by the organizer for the entire campaign; no nightly feature is available.

Common cards may define resources, enable the compilation-only authoring mode, and pin the arena scoring protocol. They may not relax the final-column target rule, Rust-only implementation, dependency boundary, no-experiment rule, one-submission rule, or train/test isolation. Resolve any conflicting packet before authoring; there is no entrant-specific exception.

There is no guaranteed small row count or feature count. The hidden panel spans varied shapes. You must design a finite procedure with sensible time and memory behavior and explain its complexity. No algorithm can guarantee completion for every imaginable array; the evaluator uses a prespecified panel and applies the same limits to all entrants.

All internal work counts, including `new`, input validation, allocations, calculations used to select settings, unsuccessful numerical attempts, multiple passes, and final fitting. There is no separate unmetered preprocessing or tuning budget. Prediction input validation and every prediction batch count toward the single aggregate prediction allowance. The supervisor also bounds cleanup; destructors may not perform hidden learning or evaluation work.

The judge controls elapsed-time enforcement. You cannot read a clock. You may choose deterministic work limits based on the current shape and your frozen source, but you cannot exploit timeouts as a normal model-selection mechanism.

Do not claim your code meets a timing limit based on experiments. You are not permitted to conduct those experiments during design.

## 11. Elo evaluation on thousands of unknown datasets

The evaluator will fit your frozen implementation independently on each task's training partition and compare its predictions on held-out rows against other algorithms.

The primary task outcome is based on held-out regression error. The default error metric is root mean squared error (RMSE) in original target units:

\[
\operatorname{RMSE}=\sqrt{\frac{1}{m}\sum_{i=1}^{m}(\hat y_i-y_i)^2},\qquad m\geq1.
\]

Lower error is better. A common scoring card supplied before authoring may specify the arena's already established regression metric instead. The organizer must use one frozen definition for every entrant. Elo is calculated from comparative task outcomes; it is not an average of raw errors across unrelated target units.

The arena converts pairwise outcomes into Elo ratings under one frozen scoring protocol. Dataset identities, hidden values, opponent implementations, and results are not disclosed to you during the challenge.

Unless the common scoring card specifies otherwise, lower finite error wins, exactly equal finite errors draw, a valid successful result beats an algorithm failure, and two algorithm failures draw. The card fixes any alternative tie tolerance and the aggregation of repeated splits/seeds before authoring. No failure may be discarded to improve a rating. An infrastructure error is repaired uniformly and is not treated as evidence of algorithm quality.

Rules of the game:

1. One source submission is frozen before hidden evaluation begins.
2. The identical source is used for every dataset. Only the fitted state changes.
3. All entrants receive the same training and held-out row assignments for a task, the same numerical input representation, and the same allowed resources.
4. The evaluator supplies the same prescribed seeds to every entrant. If multiple seeds are used, all declared seeds count; nobody chooses a favorable result afterward.
5. No submission is manually specialized, repaired, or replaced after hidden results become available.
6. A failed fit, invalid output, nonfinite prediction, memory failure, or timeout is a failure under the arena's frozen policy. Difficult tasks are not silently omitted.
7. A compile failure or disallowed dependency makes the submission invalid.
8. Training error, rhetorical claims, source length, novelty, and explanation quality do not increase predictive Elo.
9. Elo is the competition outcome, not an objective your algorithm can read or optimize against at runtime.
10. You receive no score feedback with which to refine your answer.

Aim for broad competitive performance across unknown datasets. You cannot assume the panel resembles a particular public benchmark or a particular application domain.

## 12. Prohibited shortcuts

The following are disqualifying even when disguised inside otherwise valid code:

- Calling a pretrained or externally implemented predictor.
- Wrapping an existing library and describing the wrapper as original.
- Copying or embedding an external implementation, including vendored source.
- Shipping a large constant array that encodes fitted model knowledge or dataset-specific answers.
- Identifying benchmark datasets by dimensions, values, hashes, or fingerprints and switching to prepared answers.
- Sharing fitted state, predictions, scores, or caches between datasets.
- Accessing hidden labels or information from another contestant.
- Returning different results depending on prediction batch composition.
- Requiring human selection, extra training instructions, dependency installation, or a future experiment to finish the algorithm.
- Presenting pseudocode or an incomplete implementation as executable code.
- Exploiting the runner, scorer, compiler, resource monitor, or data loader.

A general shape-dependent computational rule is allowed. A lookup table targeting recognizable datasets is not. The distinction is whether the rule implements a general learning procedure or encodes prior benchmark knowledge.

## 13. Required final answer

Return exactly three sections, in this order.

### 13.1 `submission.rs`

One Rust code block containing the entire source file. It must:

- Include the exact `RegressorApi` trait and implement it fully for a public `Regressor` struct.
- Contain all algorithmic code and helpers.
- Use only the allowed fundamentals.
- Have no external dependency or missing file.
- Compile as a library without a `main` function.
- Include `#![forbid(unsafe_code)]` at the crate root.
- Interpret the final training column as the target and accept prediction features without a target column.
- Return explicit errors for invalid input and invalid lifecycle calls.
- Implement an actual learning procedure and finite predictions.

Do not put several alternative implementations in this block or hide a choice behind a compile-time flag.

### 13.2 `submission.json`

One valid JSON block with exactly the following fields:

```json
{
  "schema_version": "general-regressor-challenge/v2",
  "algorithm_name": "Your name for the algorithm",
  "language": "rust",
  "edition": "2021",
  "target_column": "last",
  "prediction_input": "features_only",
  "external_dependencies": [],
  "unsafe_code": false,
  "authoring_mode": "reasoning_only",
  "data_accessed_during_design": false,
  "experiments_run_during_design": false,
  "external_code_used": false,
  "compiled_during_design": false,
  "single_frozen_implementation": true,
  "fit_complexity": "State the asymptotic cost of your actual implementation",
  "predict_complexity": "State the asymptotic cost of your actual implementation",
  "memory_complexity": "State the asymptotic memory cost of your actual implementation"
}
```

Set `authoring_mode` to the common campaign's `reasoning_only` or `compilation_only`, and set `compiled_during_design` truthfully. The other compliance fields must be truthful as well; an assertion cannot make a prohibited action compliant. Do not invent a source hash; the evaluator computes it. In the complexity fields define training rows `n`, features `P`, prediction rows `m`, and any additional internal work parameter your bound depends on.

### 13.3 `design.md`

Provide a clear explanation of at most 2,500 words:

1. The mathematical learning procedure your code implements.
2. Why you expect it to be broadly competitive without assuming a particular dataset family.
3. What is learned from training data and what remains fixed.
4. The objective or principle governing fitted parameters and any internal decisions.
5. How your implementation handles missing values, degenerate inputs, zero features, numerical stability, and varying units.
6. The dominant computational and memory costs, including all internal selection work.
7. Determinism, isolation of fitted state, correct target-column handling, and prediction without labels.
8. Important weaknesses and cases where you expect the algorithm to perform poorly.
9. A precise statement of any compilation performed and confirmation that no data experiments were run.

Explain the design and its justification. Do not provide an experimental proposal, an invitation to tune it later, or unsupported claims of measured superiority.

## 14. Final instruction

**Produce your single best complete general regressor now, using only the allowed fundamentals and 100% Rust. The final training column is always the target. Prediction inputs contain features only. Your one implementation will be frozen and ranked by Elo across thousands of unknown datasets, with no design-time data experiments or later revision.**

<!-- END MASTER PROMPT -->

---

# Organizer appendix: making the challenge fair and executable

## O0. Red-team findings and corrections

This appendix is for the organizer. The participant-facing challenge above deliberately gives no candidate design, existing approach, suggested model family, tuned setting, or performance clue.

| Finding in the earlier draft or foreseeable attack | Why it matters | Resolution in the final prompt |
|---|---|---|
| Target-column placement was unspecified in the callable contract | An implementation could fit against the wrong column or assume a separate label source | `fit` receives the complete table; `C - 1` is always the target; `P = C - 1` is always the feature count |
| A complete training row could be used as a feature row | Row summaries or preprocessing could incorporate the answer and produce a misleading apparent fit | Feature operations exclude the target; label use is confined to learning inside `fit` |
| “The last column is the target” could be applied a second time during inference | The final actual feature could be discarded, or a dummy label could leak through the interface | Prediction receives exactly `P` feature columns and no target or target placeholder |
| Semicolon-only function declarations appeared inside an inherent `impl` block | The advertised interface was not valid Rust syntax | A valid, exact trait contract is paired with a required concrete implementation |
| Several available scalar functions and row summaries were only implicit | Different entrants could assume different mathematical capabilities | All 19 scalar functions, arithmetic operations, indexed access, and seven row summaries are explicitly enumerated |
| “Use all fundamentals” could be read as a requirement to include every operation | That would constrain the algorithm arbitrarily and encourage irrelevant calculations | The whole inventory is available; use of each individual operation is optional |
| “Rust only” could still allow a wrapper around another implementation | A wrapper could import an external learner while superficially satisfying the language rule | Every participant-authored computational component must be safe Rust; foreign calls and external source are forbidden |
| An unrestricted standard-library allowance could expose time, files, entropy, or processes | These enable nondeterminism, side channels, or extra work outside the stated model | A purpose-limited library allowlist closes the capability boundary |
| Optional compilation depended on the model's available tools | Some entrants could receive much more iteration or hidden execution than others | One authoring mode is selected for the whole campaign; compiler access and revisions are equal and bounded |
| “No execution” could be bypassed through constant evaluation or build machinery | A nominal compile could run a synthetic experiment or construct fitted artifacts | The no-experiment rule applies during compilation as well as ordinary execution |
| Refit could inherit an earlier random stream or partial model | State from previous calls could change the supposedly independent algorithm | Refit resets to the original instance seed and current training data; failed refit invalidates state |
| Constructor work and repeated prediction calls could escape phase budgets | Effective compute could differ even with identical nominal limits | Construction and validation count toward fitting; prediction has one aggregate allowance |
| Exact bit equality under every batching pattern could reject harmless roundoff | A numerical implementation could be rejected for a difference unrelated to information leakage | One fixed conformance tolerance is stated; intentionally batch-dependent prediction remains forbidden |
| Degenerate shapes and missing values had incomplete semantics | Entrants and drivers could disagree on otherwise valid inputs | Zero features, zero prediction rows, missing features, finite targets, and invalid shapes are explicit |
| A single “Elo” label leaves ties, failures, repeated splits, and rating order underspecified | The ranking can change without changing the submitted algorithms | The organizer freezes the actual scoring protocol before authoring; all scheduled outcomes are recorded |
| Silent fixes or selective omission after evaluating a candidate | Such intervention changes what is being measured and can inflate ratings | Source freezes before any numerical execution; no post-score repair or omission is permitted |

**Remaining limits of the claim:** this is a test of task-time reasoning without data experiments, not proof that a foundation model never encountered a mathematical idea or dataset during pretraining. Source review and isolation enforce observable rules; a self-reported compliance manifest alone does not prove provenance. Elo is relative to the chosen tasks, opponents, resources, and scoring protocol; it does not establish universal optimality.

## O1. Preserve the neutral information boundary

The participant packet is the master prompt plus, if needed, the arena's common resource/scoring card. It must not contain project source, a repository link, an existing algorithm, an architecture diagram, a default model collection, tuned settings, prior results, or advice about which approaches succeed.

The mathematical catalog is a low-level permission contract. It is not a literal export grammar or an assertion that every composite program can be lowered by a particular existing exporter. The submission is standalone source, and its numerical routines must be implemented by the participant. Do not present this as an unchanged implementation of an existing language profile.

Do not append the previous system's recipes under an “examples,” “baselines,” “inspiration,” or “helpful context” heading. Do not give models a benchmark sample as a convenience. The interface declaration is enough to explain the required output.

Use fresh sessions. The organizer should not use a conversation that has already exposed a model to existing approaches, results, or dataset details. Disable retrieval, web access, connected accounts, automatic repository context, and automatic delegation. Select the default reasoning-only mode or the uniformly available compilation-only mode before authoring starts.

This control limits task-time exposure. It cannot establish that a foundation model never encountered public numerical ideas or public datasets during pretraining. Describe the experiment accurately as no task-time data or empirical feedback.

## O2. Keep your existing Elo arena authoritative

The user-selected outcome is Elo over thousands of unknown datasets. This specification does not replace the arena with an average normalized error score or introduce a second primary leaderboard.

Before issuing prompts, pin the existing arena's:

- Task error metric and direction.
- How repeated folds and seeds become a dataset-level result.
- Pairwise win/loss and tie rules, including numerical tolerance if any.
- Failure handling, including one-sided failure and double failure.
- Pairing schedule or comparison graph.
- Rating algorithm, initialization, update/fitting parameters, and any regularization.
- Dataset/task weights and treatment of related datasets.
- Whether ratings are computed in one batch or by an ordered update sequence.
- Bootstrap or other uncertainty method, if used.

These choices belong to one immutable `scoring-card.json`. Use the arena's actual definitions; do not invent a replacement Elo variant. If the existing arena has unresolved rules, settle them before candidate collection. If it already has frozen rules, preserve them. None of this requires revealing datasets, algorithms, or comparative performance to participants.

The default participant text says RMSE because a task-level error measure is needed to interpret the objective. If the actual arena uses another metric, state it in the common scoring card before issuing the prompt. Likewise, state the actual resource limits if they differ. Record the resulting packet bytes once and give those identical bytes to every entrant. No card may change the fixed target location, language, dependency, isolation, or no-experiment rules.

Compute the task metric consistently and with numerically robust arithmetic. A scorer overflow or adapter bug must not be mislabeled as a contestant's nonfinite prediction. Keep the prediction-conformance tolerance separate from the scoring card's tie definition. Make all draw, failure, and repeated-seed aggregation decisions before seeing results.

Elo is a relative rating conditioned on the opponent pool, task panel, and scoring protocol. Report those identities with every rating. Do not compare ratings from different panels or opponent pools as if they shared an absolute scale.

## O3. Single-submission protocol

1. Freeze the prompt, interface, authoring mode, runtime image, resource card, and scoring card.
2. Commit to the hidden dataset collection and splits before reading candidate source or predictions. Keep the collection private.
3. Give every model the identical packet and the same maximum number of submission attempts.
4. Use one attempt per model as the primary challenge. If repeated independent attempts are desired, preregister the same count for all models and report them all; do not choose the best hidden score.
5. Retain the full visible authoring transcript, model identifier/revision, generation settings, time, and any compilation diagnostics. Do not request private internal reasoning traces. Predeclare a common output budget large enough for the allowed source, plus comparable authoring limits supported by the tested model interfaces; record unavoidable differences.
6. Extract exactly one `submission.rs`, one JSON manifest, and one rationale from each response.
7. Freeze each submission's raw bytes and SHA-256 before any numerical execution.
8. Apply the same source and interface admission checks.
9. Run organizer-only correctness checks, then hidden evaluation. Return no data-dependent feedback to the author.
10. Publish ratings only after all primary submissions are frozen and their scheduled tasks finish or receive explicit failure outcomes.

A compilation-only repair window may occur before step 7 only in the common compilation-only mode. It allows at most three compilation requests and two feedback revisions; these are one authoring attempt, not three scored candidates. Feedback must contain only compiler/interface diagnostics. No constructor/fit/predict execution, model quality, runtime, memory, hidden input shape, or dataset-specific failure may influence that repair. A source rejected for incomplete output is not silently continued beyond the common output budget.

After freezing, a corrected source file is a new entrant in a new declared round. Preserve the original failed submission and results. Never overwrite the submitted source or silently repair it.

## O4. Minimal implementation layout

The paths below are proposed new challenge files. They are not claims about files already present in a repository.

```text
challenges/general-regressor-v2/
  README.md
  master-prompt.md
  interface.md
  allowed-fundamentals.json
  resource-card.json
  scoring-card.json
  campaign.json
  schemas/
    submission.schema.json
    receipt.schema.json
  harness/
    build.sh
    driver.rs
    source-audit.md
    container.lock.json
  conformance/
    cases.json
    expected-contracts.json
  entrants/
    <entrant-id>/
      submission.rs
      submission.json
      design.md
      authorship-receipt.json
      source.sha256
      admission.json
  results/
    <campaign-id>/
      index.json
      task-receipts.jsonl
      ratings.json
      ratings.csv
      report.md
```

Keep hidden datasets, row assignments, held-out labels, and detailed predictions in private evaluation storage. The public tree may contain immutable references and hashes, subject to the arena's privacy and release policy. It must not become an unintended path from participant compilation into the hidden panel.

Each file has one responsibility:

| File | Responsibility |
|---|---|
| `master-prompt.md` | Exact participant-visible prompt, with no project-specific advice |
| `allowed-fundamentals.json` | Machine-readable inventory of permitted numerical, language, and library capabilities |
| `interface.md` | Exact public signatures, dimensions, lifecycle, and error behavior |
| `resource-card.json` | Actual common hardware/process limits; overrides the prompt defaults only when distributed with it |
| `scoring-card.json` | Exact existing arena metric, comparison, failure, and Elo configuration |
| `campaign.json` | Frozen identities of prompt, cards, runtime, entrants, private task collection, splits, and seeds |
| `driver.rs` | Array transport and calls to the submitted API; no learned preprocessing or contestant-specific repair |
| `source-audit.md` | Review rules for permitted dependencies, capabilities, and encoded external knowledge |
| `admission.json` | Compilation/interface/source audit outcome before numerical benchmarking |
| `task-receipts.jsonl` | One explicit outcome for every scheduled entrant/task/fold/seed job |
| `ratings.json` | Ratings derived by the frozen arena with its inputs and version attached |

Git stores source, cards, manifests, and compact receipts. Large private inputs and predictions are referenced by content identity, not copied into the participant package.

## O5. Build and execution contract

### O5.1 Offline source admission

Compile every entrant with the same pinned compiler and flags. An example library build is:

```bash
rustc --edition=2021 \
  --crate-name submission \
  --crate-type rlib \
  -C opt-level=3 \
  -C codegen-units=1 \
  -D unsafe-code \
  submission.rs \
  -o libsubmission.rlib
```

The compiler version and target are pinned in the runtime image, not assumed from the host. Do not use Cargo to resolve participant dependencies. The source has no external crate requirements.

Link the organizer's Rust driver against the produced library. It imports `submission::{Regressor, RegressorApi}`, supplies arrays and a seed, calls `Regressor::new`, `fit`, and `predict`, and emits only the protocol result. It owns parsing and serialization. A driver build error caused by absent or wrong public signatures is an interface failure. Do not patch the candidate's API to make it fit the driver.

Compile in an isolated environment with no credentials, network, dataset mount, unrelated repository, or hidden results. The same isolation applies to any participant compilation window. Static initialization and compile-time evaluation must not gain access to information that is forbidden at runtime.

### O5.2 Admission must check capabilities, not keywords alone

Inspect parsed source and dependency/import closure. A text search for a few package names is insufficient. Resolve aliases, fully qualified paths, inline modules, macro use, and any route to forbidden capabilities. Require safe code throughout the participant crate.

The admissible code consists of the original submitted file plus compiler-provided standard facilities specifically permitted by the prompt. Reject dependency installation, foreign linkage, source inclusion, arbitrary macros that import code, and OS interactions.

Numerical correctness does not excuse an unauthorized implementation dependency. Conversely, similarity to a known mathematical method is not evidence that source was copied. Review code provenance and actual dependencies; do not claim the absence of copied code can be proven from a manifesto alone.

### O5.3 Runtime isolation

Use a fresh isolated process for each entrant/task/fold/seed job. Supply the assigned complete training table with its target in the final column. After fitting succeeds, supply the held-out features with that column removed. Keep held-out target bytes in a separate scorer process or service that the participant process cannot access. Never place the complete held-out table in the participant process and rely only on an API convention to hide its labels.

Enforce single-core CPU access, no network, no writable shared location, no available credentials, no subprocesses, no dynamic libraries beyond the approved runtime, and the common process-tree memory/time limits. The host controls file loading and output transport.

Catching a panic inside the driver is not the complete isolation mechanism. Timeouts, OOM events, process aborts, and malformed responses require supervisor-level handling and durable receipts.

Do not feed dataset names or IDs through error strings, working-directory names, command-line arguments, environment variables, or filenames visible to the submission. The submission API receives arrays and seed only.

### O5.4 Fitting and prediction accounting

Start the fitting clock before `Regressor::new` and include constructor work, participant-side validation, allocation, and `fit`. All calls needed to predict the complete held-out partition, including their input validation, count toward one aggregate prediction limit. Splitting predictions into batches does not reset the allowance. Cleanup is bounded separately and cannot be used to perform unmetered learning work.

Input buffers count toward the common memory limit. Record peak memory consistently, including the fitted state and subprocess tree. Compilation time is recorded separately and does not increase an algorithm's fitting allowance.

Use the same row order, feature order, input representation, and seeds for all submissions on a job. The target is always the final complete-table column; any feature permutation must leave that rule intact. Construct outer splits before learning any preprocessing statistics. Any arena preprocessing must be frozen, trained on permitted rows, applied identically, and disclosed in the common input contract. Do not add a helpful preprocessing step only for a failing entrant. Ensure input transport itself fits the resource envelope; the input contract must not demand an impossible allocation before any contestant code runs.

The input contract permits NaN features but requires finite targets and excludes infinite features. Tasks or adapters violating that contract are harness-input failures, not contestant losses. Correct the harness uniformly and rerun every affected entrant with unchanged source; keep the original receipts.

## O6. Organizer-only correctness checks

These checks execute only after source freeze. They are not performance feedback for the author. Use tiny invented arrays to test the API and state contract, with no scoring advantage for a particular statistical method.

| Check | Required result |
|---|---|
| Valid finite arrays | Fit completes or produces an explicit algorithm failure under the common policy; successful prediction has the right length and finite values |
| Wrong flattened matrix length | Explicit error; no silent truncation |
| Dimension multiplication overflow | Explicit error before allocation or indexing |
| Final training column | Used only as the target, with feature width exactly one less than total width |
| Prediction matrix | Contains all and only the feature columns; there is no extra target placeholder |
| One-column training table | Supported as a zero-feature regression task |
| Empty fitting rows or zero total training columns | Explicit error |
| Infinite feature value | Explicit error |
| Nonfinite target | Explicit error |
| NaN feature entries | Supported as missing values; valid-input algorithm failures count under the common failure policy |
| Constant target | Defined finite behavior; no numerical exception treated as success |
| Constant or repeated columns | Defined finite behavior, subject to a recorded algorithm failure if unsuccessful |
| Repeated rows | No interface break or assumption that rows are unique |
| Prediction before fitting | Explicit error |
| Prediction after a failed refit | Explicit error; no stale earlier model |
| Prediction with different width | Explicit error |
| Zero-row prediction after successful fit | Empty vector |
| Repeated fitting with fresh equivalent state and same seed | Reproducible outputs on the pinned platform |
| Repeated prediction | Same outputs and no model mutation |
| Full prediction versus several batches | Same per-row outputs within the declared conformance tolerance |
| Permuted prediction rows | Correspondingly permuted outputs within the declared conformance tolerance |
| One instance refitted on a second task | Same outcome as a fresh instance fitted on that second task with the same seed |

Do not require a particular learned formula or use a synthetic accuracy threshold as an admission gate. That would reward an undisclosed inductive bias. The correctness checks concern interface behavior, state isolation, numerical outputs, and declared error handling.

Distinguish invalid source/interface from a numerical failure of an otherwise admissible algorithm. A runtime failure is recorded; it is not grounds for silently dropping hard benchmark cases. The arena's existing treatment of failed entrants must be applied consistently.

Use the prompt's fixed conformance tolerance for numerically equivalent output checks. Also inspect the code's information flow: passing a numerical tolerance check does not authorize reading the target as a feature, using another test row to adapt a prediction, or retaining cross-task state. Across different machines or compilers, use a separately published reproducibility protocol; do not change tolerances after seeing which entrant fails.

## O7. Hidden benchmark construction and rating integrity

Use thousands of unknown datasets under your existing arena's governance. Freeze dataset eligibility and task sampling before examining entrants' performance.

The main protections are:

- **No outcome-based selection:** do not choose datasets because a preferred algorithm wins on them.
- **Same test opportunities:** all entrants receive the same scheduled tasks; failures stay visible.
- **No related-data leakage:** related versions, duplicate tables, repeated entities, and other shared origins must follow the arena's predefined grouping/splitting policy.
- **No unrecorded retries:** distinguish an infrastructure retry from a new learning seed or source revision.
- **No best-of-seed advantage:** use the frozen aggregation rule for every declared seed and split.
- **No opportunistic Elo update order:** if the arena is order-sensitive, freeze and preserve the update sequence; if batch-fitted, pin the actual fitting implementation.
- **No opponent-pool drift within a comparison:** retain the exact opponent identities used to calculate each rating.
- **No contamination feedback:** release no candidate-specific quality diagnostics while other primary submissions are still being authored.

Thousands of tasks are not automatically thousands of independent statistical units. Use the arena's dataset-family grouping for uncertainty analysis and disclose the number of independent groups as well as the task count.

The headline table should report Elo with completion/failure coverage and uncertainty, not Elo alone. Keep raw task-level error and resource measurements available for audit. Ratings must be traceable to the exact source, task, prediction, and scorer identities.

## O8. Required receipts

### O8.1 Authorship receipt

The organizer, not the participant, records:

```json
{
  "schema_version": "regressor-authorship/v2",
  "entrant_id": "Assigned by the organizer",
  "model_id": "Exact served model identifier",
  "model_revision": "Record when available; otherwise null",
  "prompt_sha256": "Computed from the actual participant packet",
  "resource_card_sha256": "Computed by the organizer",
  "scoring_card_sha256": "Computed by the organizer",
  "generation_settings": {},
  "attempt_index": 1,
  "authoring_mode": "reasoning_only",
  "compilation_requests": 0,
  "compilation_only_feedback_used": false,
  "transcript_ref": "Organizer-controlled transcript reference",
  "source_sha256": "Computed from the frozen submission.rs bytes",
  "frozen_at_utc": "Actual freeze timestamp"
}
```

This is a schema illustration. Replace descriptions with actual values; never publish them as completed evidence. Use JSON null for unavailable revision metadata rather than guessing.

### O8.2 Task receipt

Every scheduled job requires the following facts:

| Field group | Required contents |
|---|---|
| Entrant identity | Entrant ID, frozen source hash, compiled artifact hash |
| Runtime identity | Compiler version, target, flags, container/runtime digest, driver version |
| Protocol identity | Campaign, input contract, resource card, scoring card hashes |
| Task identity | Private dataset content reference, split identity, fold, seed; not disclosed to candidate |
| Dimensions | Training rows, total training columns `C`, feature count `P = C - 1`, prediction rows; confirmation that prediction inputs exclude the target |
| Outcome | Success or specific failure state; no ambiguous empty score |
| Measurements | Fit/predict elapsed and CPU time; peak memory; prediction count |
| Quality | Arena task metric when defined; never fabricated on failure |
| Evidence | Prediction hash and private storage reference; bounded stderr/error record |
| Rating inputs | Exact match outcomes or their derivation reference |
| Retry ancestry | Original attempt, retry reason, same or changed infrastructure identity |

Use explicit failure states such as `compile_error`, `interface_error`, `forbidden_capability`, `fit_error`, `predict_error`, `timeout_fit`, `timeout_predict`, `memory_limit`, `nonfinite_prediction`, `shape_error`, and `infrastructure_error`.

A rating pipeline should refuse to finalize when scheduled outcomes are missing. Missing is neither success nor a zero error. Convert recorded algorithm failures to pairwise outcomes only through the frozen arena policy.

## O9. Acceptance criteria for releasing the challenge

The challenge is ready when:

1. The participant packet contains only the neutral prompt, permitted fundamentals, interface, and common cards.
2. No existing algorithm, recommended configuration, performance clue, or benchmark data has entered that packet.
3. A complete source file can be compiled and invoked through the exact Rust trait interface without installing a participant dependency; training receives the target as its final column and prediction receives no target column.
4. Source admission rejects unavailable capabilities and external implementations.
5. Post-freeze contract checks have deterministic expected behavior and do not prescribe an algorithm.
6. Runtime isolation prevents access to hidden labels, other tasks, unrelated files, and external services.
7. The resource limits are enforced over the complete fitting/prediction work, including internal adaptation.
8. The existing Elo scorer's exact settings, opponent pool, task panel, and aggregation are pinned.
9. Every scheduled job produces a receipt, including failures and infrastructure retries.
10. No post-score repair, selective omission, best-seed selection, or hidden extra submission is possible in the primary round.
11. The final report links ratings to frozen source and clearly states evaluation coverage and uncertainty.

The result is a clean test of what each model can produce from mathematical fundamentals and reasoning: **one implemented regressor, one frozen submission, thousands of unknown datasets, and the same Elo arena for everyone.**
