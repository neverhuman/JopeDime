**Algorithm kernel audit — 8 September 2026**

**Recommendation: retain the existing S-expression syntax and add a versioned, typed algorithm representation underneath it. Make learner state, training updates, tensor operations, and bounded control flow evolvable. Use the same representation for mutation, validation, interpretation, and export.**

The current system can discover useful compositions and symbolic features, but it cannot yet express arbitrary learning procedures. Its largest limitation is that most learning mathematics lives inside fixed Rust heads. Adding more named heads or scalar functions alone will not meet the intended goal.

There are also concrete problems that should be addressed before expanding search: accepted expressions can panic, canonicalization can change behavior, fitted C++ can access memory outside a row, and some valid learner families cannot produce useful offspring through the mutator used by generalized GP.

This is an audit and proposed design, not an implementation. Production source, services, campaigns, and model artifacts were not changed. Reproducers and observations are preserved in [the evidence directory](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08). The working tree already contained substantial changes; findings describe the inspected working tree, not just commit `991bedcb3d5bc0f7705fa3f9d34966e1ee2feae0`. [Source SHA-256 values](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/source-sha256.json) identify the audited files.

The parallel workspace changed during the audit. Its current teacher mutation calls were rechecked before delivery and still use the legacy APIs described here; the final digest is preserved separately in `source-final-recheck.json`.

**What the current kernel actually provides**

| Layer | Implemented capability | Expressivity or integration limit |
|---|---|---|
| Legacy genome | Pipelines, scalar symbolic models, explicit portfolios | Fixed pipeline stages and learner implementations; no general programmable training state or update body |
| Scalar mathematics | 19 named functions, five binary operators, element/column/row-summary variables | Row-local scalar expressions; 64 nodes, depth 16; protected floating-point semantics |
| Legacy transformations | 36 planned/fitted operation variants | No general tensor program, reusable local functions, or programmable dataset reduction |
| Learners | 12 planned-head variants versus 13 fitted-head variants | `hurdle2` fits but is absent from the legacy emitter planner; additional v3 heads use separate paths |
| GPR kernel expressions | Sums/products of RBF, rational quadratic, periodic, and linear primitives | At most three primitives; a useful specialized algebra, not a general mathematical kernel language |
| Advanced-v3 | Schemas, feature roles, missingness, categorical encoding, ordered rows, additional heads | Its strict operation registry is disconnected from public fitting and evolution; the public fitter has a different validator |
| Search | Several legacy, FullRust, advanced-v3, generalized, and parallel search paths | Different operation vocabularies, validation, mutation, task coverage, and fitness policies |
| Data representation | `Vec<Vec<f64>>`, plus v3 masks and lineage | Primarily two-dimensional tables with scalar targets; sparse-column metadata is not a general sparse tensor representation |

Evidence: [genome forms](/home/ubuntu/jope-dime/vendor/feat-math/src/genome.rs), [model fitting](/home/ubuntu/jope-dime/vendor/feat-math/src/model.rs:15), [scalar AST and limits](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:3), [GPR algebra](/home/ubuntu/jope-dime/vendor/feat-math/src/gpr.rs:321), [feature storage](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/feature_block.rs:75).

There is a useful foundation to preserve: deterministic RNG/folds, explicit fitted state, training-scoped preprocessing, feature lineage, bounded feature expansion, stage-prefix caching, and substantial parity/freeze tests. The new design should extend these mechanisms.

The parallel workspace uses its own vendored dependency at `parallel/assets/source/jope-dime/vendor/feat-math`, as specified in [its Cargo manifest](/home/ubuntu/jope-dime/parallel/Cargo.toml:37). The key syntax, math, planner, and mutation files checked there are byte-identical to their root counterparts. Its teacher loop nevertheless selects a narrower legacy search API. Experimental expanded mutation must not be described as active in every campaign.

**Highest-priority findings, with reproductions**

**1. Canonical text is not a lossless program identity.**

Two examples change actual behavior or validity:

```text
(pipe X (step t=0.0000000000001) (ridge))
  canonicalizes to (pipe X (step t=0.0) (ridge))
  on input [[5e-14]], the transform changes from [[0.0]] to [[1.0]].

(pipe X (map expr="0.5") (ridge))
  canonicalizes to (pipe X (map expr=0.5) (ridge))
  a valid string expression becomes a numeric parameter that fails fitting.
```

Float formatting rounds values within `1e-12` of integers; string formatting omits quotes from strings that can subsequently parse as numbers or booleans. This is especially consequential when canonical text drives hashing, deduplication, caching, or neural training. See [formatting](/home/ubuntu/jope-dime/vendor/feat-math/src/dsl.rs:82) and [reproduction log](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/syntax-probe.log).

Add a new lossless typed serialization and identity version. Preserve old canonical text and historical hashes as legacy identities. Do not silently recompute old identities using a corrected printer. Values already erased by old serialization cannot be recovered without the original source or artifact.

**2. Validly parsed expressions are not guaranteed to execute safely.**

`clip(1,(-1)^0.5,(-1)^0.5)` parses successfully but panics: both bounds become NaN, and Rust's `clamp` rejects them. The outer finite-value guard executes too late. See [expression evaluation](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:517), especially line 558, and the syntax probe log above.

Define every operation's behavior for invalid inputs and intermediate values. Return a structured evaluation failure, or apply an explicitly specified protected result. Treat timeout, invalid shape, nonfinite computation, and resource exhaustion as distinct evaluation outcomes. An admitted program must not crash the search worker. A typed program can still diverge numerically; type validity is not a promise of good fitness.

**3. Fitted C++ symbolic code can read outside its input buffer.**

`(sym raw "c9 + 1")` fitted on two-column data produces `[0.0]` through the Rust bound model. Fitted C++ emission succeeds, but running the generated program under AddressSanitizer reports a heap-buffer-overflow. [Expression lowering](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:135) emits unchecked `row[index]`; [symbolic fitted emission](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/fitted_cpp.rs:289) supplies no guard. Map/derive use the same unchecked lowering.

Preserve the reference interpreter's behavior: if a required column is unavailable, the **whole expression** becomes zero. Replacing only the missing leaf with zero would make this example return one and would still be wrong. Add dimension checking for new programs and retain the legacy whole-expression guard when lowering old programs. [ASan evidence](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/codegen-asan.log).

**4. Mutation and crossover do not preserve valid target-stage placement.**

Legacy mutation treats a leading target transform as an ordinary feature stage. For the valid parent `(pipe X (target kind=log1p) (standardize) (ridge))`, a deterministic sweep of seeds `0..9999` produced:

| Operator | Children rejected by emitter planning |
|---|---:|
| Legacy v1 mutation | 2,248 / 10,000 |
| Legacy v2 mutation | 1,947 / 10,000 |
| Legacy self-crossover | 2,271 / 10,000 |

Mutation seed 3 moves `target` after `standardize`; self-crossover seed 8 duplicates `target`. Fitting independently confirms the placement error. These are controlled probe rates, not measured campaign-wide failure rates. See [legacy stage splitting](/home/ubuntu/jope-dime/vendor/feat-math/src/evolve.rs:1298), [runtime placement rule](/home/ubuntu/jope-dime/vendor/feat-math/src/model.rs:225), and [probe results](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/evolution-probe.log).

Give target preprocessing, input preprocessing, feature transformations, and the learner separate typed slots. FullRust already has a useful target-slot separation; reuse that design in an explicitly versioned mutator.

**5. Generalized GP accepts modern heads that its mutator does not recognize.**

Generalized GP accepts `logit_ridge`, `tab_boost`, `extra_trees`, `gam_boost`, and `pls`. The inherited mutator's `HEAD_OPS` omits them, so it treats the existing learner as a transform. Generalized repair replaces only the last head, leaving an illegal internal head.

For a simple `standardize → logit_ridge` parent, 10,000 mutation seeds followed by the equivalent head-repair/placement check produced 9,293 structurally invalid children and 707 unchanged parents: **zero unique structurally valid children**. The other four omitted families had identical counts. A recognized-head `logistic` control produced 658 unique structurally valid children. This probe isolates mutation and repair; it is not a full production campaign simulation.

The generalized loop retries 512 times before using an RFF-specific refill. This makes a vocabulary mismatch a direct search-quality and throughput problem. See [accepted heads](/home/ubuntu/jope-dime/src/jope_generalized_gp/contracts.rs:601), [mutator heads](/home/ubuntu/jope-dime/vendor/feat-math/src/evolve.rs:41), [repair](/home/ubuntu/jope-dime/src/jope_generalized_gp/evolution.rs:309), and [retry loop](/home/ubuntu/jope-dime/src/jope_generalized_gp/evolution.rs:423).

One registry should drive head recognition, parameter domains, task support, mutation, and lowering. A new mutator should operate on an already classified learner slot.

**6. Advanced-v3's strict compiler does not establish the validity of public fitted models or evolutionary children.**

The strict `compile_v3_genome` has a substantial descriptor/shape/admission implementation, but its callers in the inspected kernel are its own grammar tests. Public `fit_table` calls the different `validate_v3_genome` implementation. Public fitting accepts `map`, unknown ridge parameters, and a string-valued `lambda_frac` that silently defaults. Consequently, the strict registry's narrower vocabulary must not be mistaken for the fitter's actual language. See [strict compiler](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/grammar.rs:1596), [public fit path](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/pipeline.rs:314), and [admission probe](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/admission-probe.log).

Advanced-v3 evolutionary canonicalization checks parsing. Its test named `one_hundred_thousand_typed_children_compile_without_retry` checks parse/canonical roundtrips, not schema compilation or execution. A 10,000-seed probe over its default population produced 141 invalid mutations and 346 invalid crossovers. A growth-biased probe also reached an accepted 101-stage descendant; mutation lacks the cap that crossover applies. See [mutation](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/evolution.rs:293) and [the existing test](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/evolution.rs:629).

Connect one typed admission path to new fitting and evolutionary APIs. Preserve existing permissive acceptance in an explicit compatibility path. Parser success, executable validity, and exportability should be separate, accurately named checks.

**7. Much of the existing mathematical language is unreachable through ordinary mutation.**

Generic evolution generates 18 transformations plus five opt-in additions; `map` and `derive` are explicitly excluded from generation. String-valued expression and kernel parameters are copied unchanged by parameter jitter. For `(pipe X (map expr="sin(x)+0.1") (ridge))`, that mutation arm changed zero of 1,000 trials. FullRust has an expanded expression generator, but it is a separate experimental path. [Generation vocabulary](/home/ubuntu/jope-dime/vendor/feat-math/src/evolve.rs:57), [expanded FullRust path](/home/ubuntu/jope-dime/vendor/feat-math/src/full_rust.rs:960), [parallel teacher calls](/home/ubuntu/jope-dime/parallel/src/gp.rs:766).

The immediate expressive improvement is to expose the existing expression AST to typed subtree mutation, crossover, constant fitting, and parameterized reusable fragments. This is substantially smaller than implementing a complete new algorithm language and makes mathematics already present in the system discoverable.

**The proposed language architecture**

Keep S-expressions. Surface punctuation is not the bottleneck. Introduce a new top-level `algorithm` form with a versioned typed core, and allow legacy pipelines to participate through wrappers. Avoid replacing the existing trees with an unconstrained global graph mutation system.

| Layer | Responsibility | Evolutionary representation |
|---|---|---|
| Surface language | Existing `pipe`/`sym`/`portfolio`, plus new `algorithm` blocks | Small typed trees and structured blocks |
| Typed algorithm IR | Values, dimensions, state, phases, effects, numeric semantics, budgets | Mutation and crossover occur here |
| Execution IR | Shared expressions, explicit operation order, buffers, loops, intrinsic calls | Derived and cached; not a second independently evolving program |
| Backend lowering | Rust/Python/C++17 first; JavaScript and CUDA incrementally | Code generated from the execution IR |

SSA—a representation where each intermediate is assigned once—is useful internally for reuse and compilation. It need not become the user-facing syntax. Local `let` bindings and typed block boundaries retain understandable genomes and manageable crossover. Pure shared subexpressions may be reused; random or stateful operations require their effect dependencies to be preserved.

A registry entry should specify stable operation ID/version, argument and result types, shape rule, parameter domains/defaults, phase/effect requirements, numeric behavior, resource estimate, backend capabilities, and optional derivative rule. Separate this semantic registry from a run's generation weights. Otherwise adding one primitive can unexpectedly change all historical random choices.

This is an application of strongly typed GP to shape and execution constraints; the original work explicitly uses typed functions and generic types to constrain program construction. [Montana's paper](https://davidmontana.net/papers/stgp.pdf).

**First-class learner state and learning procedures are the essential extension.**

Support initialization, general batch fitting, prediction, and optional online update. A complete algorithm must be able to evolve its parameters, optimizer state, memory, update schedule, and equations. Prediction should normally be pure; sequence prediction can explicitly return updated inference state. Training targets are available only in authorized training/update blocks.

Proposed illustrative syntax, **not currently accepted**:

```lisp
(algorithm version=1
  (state (w (zeros D)) (m (zeros D)) (b 0.0))

  (predict (s x)
    (sigmoid (+ (dot s.w x) s.b)))

  (update (s x y)
    (let p (predict s x)
      (let g (* (- p y) x)
        (let m (+ (* 0.9 s.m) (* 0.1 g))
          (record
            w (- s.w (* eta m))
            m m
            b (- s.b (* eta (- p y))))))))

  (fit (s X Y)
    (repeat epochs s
      (fold_rows X Y s update))))
```

Here `D` is the bound input width; `eta` and `epochs` are declared search parameters, and `repeat` carries state between bounded iterations. The example is an online logistic learner with momentum on its weights. Evolution can alter the momentum expression, normalization, state, or update rule. A general `fit` block is also needed for batch solvers, tree construction, clustering, and other algorithms that cannot be reduced conveniently to an online update.

This direction has direct research precedent: AutoML-Zero searches initialization, prediction, and learning procedures built from elementary operations; symbolic optimizer discovery demonstrates the value of evolving update rules. Those results support the design direction, not a guarantee about this implementation's search efficiency. [AutoML-Zero](https://proceedings.mlr.press/v119/real20a.html), [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675).

**A compact mathematical foundation**

| Addition | What it makes expressible | How to contain its cost |
|---|---|---|
| Scalars, booleans, integers/indices, shaped tensors, tuples/records | Numerical, discrete, structured and multi-output algorithms | Infer types; restrict dimensions and specializations per run |
| `let`, calls to typed reusable functions, explicit shared parameters | Reusable submodels, tied weights, compact formulas | Bound call graph and expansion size; control inlining |
| Elementwise arithmetic, comparisons, lazy `if`, explicit broadcasting | Gates, nonlinearities, decision logic, novel elementwise updates | Fixed-arity instructions; type-compatible mutation |
| Axis reductions, dot/matmul, transpose/reshape/slice, gather/scatter | Neural layers, attention, kernels, batch statistics, indexing algorithms | Shape and memory checks before execution; deterministic reduction/scatter order |
| `map`, `fold`, `scan`, bounded `for`, fuel-bounded `while` | Recurrence, optimizers, iterative inference and training | Explicit loop carry, iteration limits, buffer and nesting budgets |
| Fitted/trainable state, explicitly keyed randomness | New learners, stochastic optimization, sampling, memory | Fixed state schema per candidate; stable serialized random-stream IDs |
| Optional differentiation of pure blocks | Efficient fitting of constants and parameters; evolving losses/optimizers | Expand AD once; charge resulting work and activation memory to the budget |
| Typed array-based structures and sparse operations, added when needed | Trees, graphs, sparse learners, structured inference | Bounded storage and index validity; task-specific generation profiles |

Do not make every primitive differentiable. Hard thresholds, sorting, discrete structures, and custom non-gradient learning remain legitimate. Derivative rules for guarded arithmetic, discontinuities, and branch boundaries must be explicit; a derivative of the unprotected textbook expression is not automatically a derivative of the implemented operation.

Keep useful named learners as intrinsics/macros initially. Add inspectable typed implementations over time, and permit mutation inside selected macro bodies. This provides short paths to good algorithms while allowing their mathematics to change. The same strategy applies to `softmax`, convolutions, linear solves, and kernel constructors. Prefer stable solves over explicit matrix inversion in standard library implementations, while leaving low-level arithmetic available for experimental alternatives.

For positive-semidefinite kernels, offer a constrained library using feature inner products, sums/products of admitted kernels, and nonnegative scaling. Also allow unconstrained similarities in learners that do not require PSD. Do not silently treat every evolved similarity as a valid covariance matrix.

MLIR's structured linear algebra and StableHLO's typed operation specifications are useful compiler design references. They are possible future lowering targets, not prerequisites for shipping the initial kernel, and neither alone specifies this project's fitted state, evolutionary budgets, or historical protected math. [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/), [StableHLO specification](https://openxla.org/stablehlo/spec).

**Task semantics need to become part of the program interface.**

The current `TaskKind` has regression, binary, timeseries, and multiclass. It has no multilabel variant. The top-level directory name `multi_label` maps to `Task::Multiclass` with contiguous single-class indices. Generalized GP currently accepts regression/binary; top-level multiclass/timeseries readiness is marked shadow-only. [Task mapping](/home/ubuntu/jope-dime/src/task.rs:186), [kernel task enum](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/types.rs:8), [generalized task contract](/home/ubuntu/jope-dime/src/jope_generalized_gp/contracts.rs:44).

| Task | Required target/output contract | Specific additions |
|---|---|---|
| Regression | `Y[N,K]`, including `K=1` | Multi-target values, optional quantiles/distribution parameters, masks and explicit output meaning |
| Binary | One binary label and one probability/logit per example | Preserve probability-versus-logit distinction, calibration and threshold semantics |
| Multiclass | One class index; class scores/probabilities `[N,C]` | Matrix output ABI and normalized categorical predictions |
| True multilabel | Labels/mask `[N,L]`; independent probabilities or a declared structured distribution | Per-label loss, micro/macro metrics, classifier chains/shared representations, training-only threshold selection |
| Timeseries | Ordered grouped sequences, observed masks, horizon/output axes | Stateful scan, forecast horizon, availability of covariates/targets, rolling-origin evaluation, explicit state reset/continuation |

Keep training loss separate from external fitness. Evolution may change a learner's training loss; it must not change the held-out evaluator used to compare candidates. The data interface should express what is known at prediction time and when delayed labels become available.

Timeseries has an especially important compatibility trap: legacy `lag` shifts **columns within a row**, and legacy `roll` circularly shifts columns. Advanced-v3 uses group-aware row history, with `roll` computing a rolling mean including the current row. These names are not interchangeable. Preserve versioned meaning and use explicit axis/time operations in the new IR. The v3 code already checks prediction rows against fitted time cutoffs; extend this foundation to forecast horizons and streaming state. [Legacy behavior](/home/ubuntu/jope-dime/vendor/feat-math/src/ops.rs:1113), [v3 behavior](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/transforms/mod.rs:1112).

**How to increase expressivity without overwhelming evolution**

1. **Separate what the language can express from what one search is allowed to generate.** Keep existing fast pipeline search as one profile. Add expression, update-rule, recurrent, and general-program profiles with explicit budgets. Use a measured allocation of compute across profiles rather than flattening all operations into one uniformly sampled vocabulary.
2. **Generate valid replacements at the mutation site.** Match result type, dimensions, allowed effects, accessible variables, phase, and remaining budget. Check dependent consumers after dimension changes. Crossover exchanges closed blocks or rebinds only compatible inputs; it must not introduce dangling references, cycles, or invalid state schemas.
3. **Prefer local changes.** Tune constants, replace one compatible operator, modify a small expression, add a state component together with its initialization/use, or expand one macro. Add exact no-op structural mutations only where their floating-point semantics are actually preserved. Keep novel larger structures in separate niches so they have time to improve.
4. **Maintain useful representations at several scales.** Evolve pipelines and library calls frequently, formulas and update blocks less frequently, and larger new algorithms under a dedicated allowance. Learn reusable fragments from successful programs; do not require every learner to be rediscovered from scalar arithmetic.
5. **Preserve opportunities for learners the surrogate cannot recognize.** The default generic surrogate replaces every head with ridge. LiteRace eventually evaluates actual heads but begins with ridge screening. New update rules need early evaluation using their actual learner, plus exploration promotions and occasional checks of rejected candidates. Measure false rejection and ranking quality by learner family. [Current surrogate](/home/ubuntu/jope-dime/vendor/feat-math/src/evolve.rs:1652).
6. **Measure effective search, not just proposals.** Track valid unique executable children per second, unchanged children, repair/refill frequency, semantic duplicates, allocation/timeouts, and best fitness versus wall-clock and evaluation count. A larger grammar that mostly produces duplicates is not more useful.
7. **Budget computation and memory, not only syntax length.** Estimate tensor sizes, peak live bytes, pairwise expansion, loop work, state size, fitting cost, and inference cost. Keep a hard resource envelope and a quality/runtime/model-size frontier. A stage count cannot distinguish scaling from a cubic solver.
8. **Preserve exact cache boundaries.** Cache pure blocks and fitted preprocessing only with keys binding semantics, data/splits, targets where used, schema, RNG, and numeric policy. Reuse the existing stage-prefix cache where applicable. Behavioral fingerprints may guide diversity; agreement on a few rows is not proof that two algorithms can share a fitness cache entry.

No fixed budget partition or speedup is claimed here. The appropriate mix should be selected by paired, repeated experiments. In particular, expanding search and proving it does not reduce quality at a fixed budget is an empirical obligation.

**Speed improvements that need little additional search complexity**

- Compile expressions once into a small register/instruction representation. `MathExpr` currently allocates a vector for every function call and rescans rows for summaries. `map expr="x-row_mean"` consequently performs order `N×D²` work. Hoisting the identical row mean once per row reduces this to order `N×D`, retaining its reduction order. [Call allocation](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:540), [row summaries](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:626), [per-element map](/home/ubuntu/jope-dime/vendor/feat-math/src/ops.rs:1152).
- Hoist immutable fitted vectors/matrices out of per-row emitted C++ functions and reuse planned scratch buffers. Current emission reconstructs fitted constants inside transformation and learner functions. This is avoidable allocation/copying, not required model computation. [Example emission](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/fitted_cpp.rs:898).
- Separate admission from artifact generation. `full_rust::canonicalize` emits both Python and Rust, lints them, hashes them, and counts tokens. Cache that work and generate full artifacts only when needed. If exact emitted token cost is a configured objective, retain exact accounting at the required decision point instead of silently substituting an estimate. [Current path](/home/ubuntu/jope-dime/vendor/feat-math/src/full_rust.rs:692).
- Infer widths before allocating expansions. Legacy append computes features and then truncates to 512 columns, so an append at full width can do expensive work whose result is entirely discarded. New profiles should select a valid replacement/removal or explicitly allocate capacity. Preserve old truncation semantics for old programs. [Legacy finish](/home/ubuntu/jope-dime/vendor/feat-math/src/ops.rs:1274).
- Use contiguous buffers and fused elementwise blocks for the new execution IR. Preserve sanitization boundaries, aliasing, operation order, and RNG effects. Compilation must not reassociate reductions or remove computations with observable legacy failure behavior under the guise of ordinary algebra.

These are source-supported optimization opportunities. Actual speed improvements still require measurement.

**Code generation: capability must include the execution phase.**

| Current API/backend | Artifact | Actual coverage |
|---|---|---|
| Legacy `transpile::emit` | Python/Rust training and prediction recipe | Broad legacy planner coverage; explicit nested portfolios and `hurdle2` are outside this planner |
| `emit_cpp17` | C++17 training and prediction recipe | 22 transforms, ridge/elasticnet; no target stage or symbolic program |
| `emit_javascript` | ES2020 training and prediction recipe | Same narrow 22-transform/two-head slice |
| `emit_bound` | Fitted Python/Rust prediction | V3 scalar heads and legacy ridge, with a limited transform subset; rejects multiclass probability matrices |
| `emit_fitted_cpp17` | Fitted C++17 prediction | Broad legacy/v3 scalar support, symbolic/portfolio and ordered operations; rejects multiclass heads; bounds defect described above |
| CUDA backend | Batched ridge screening | Specialized screening implementation, not a general GP-to-CUDA compiler |

Sources: [legacy emitter](/home/ubuntu/jope-dime/vendor/feat-math/src/transpile.rs:171), [C++ recipe](/home/ubuntu/jope-dime/vendor/feat-math/src/emit_cpp.rs:20), [JavaScript recipe](/home/ubuntu/jope-dime/vendor/feat-math/src/emit_js.rs:8), [bound emitters](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/emit/mod.rs:111), [fitted C++](/home/ubuntu/jope-dime/vendor/feat-math/src/advanced_v3/fitted_cpp.rs:56), [GPU scope](/home/ubuntu/jope-dime/vendor/battle-gpu/src/lib.rs:1).

Define support as `(backend, phase, task/output type, operation set, numeric policy)`. Require support for the selected export targets at admission for export-constrained searches. Other searches can admit broader programs but must retain explicit capability metadata. A fitted predictor does not establish that the learning algorithm itself can be regenerated and trained in that language.

Implement the new primitive core in Rust, Python, and C++17 together. Add backend-neutral tensor/state ABI and a fitted-state codec before adding multi-output heads to every emitter. Standard-library intrinsics may have optimized implementations and portable reference fallbacks. Source generation should be deterministic and dependency requirements explicit. Python can remain a correctness/export backend while compiled backends supply peak throughput.

CUDA needs device memory planning, legal parallel regions, synchronization/reduction policy, supported dtypes and intrinsics, and a host driver. A general stateful program may require sequential or host execution; source-language neutrality does not imply every program is efficiently GPU-parallelizable. JavaScript needs explicit integer/index and RNG representations rather than relying on floating-point `Number` for all values.

**Numerical policy and backward compatibility**

The legacy mathematics deliberately differs from textbook real arithmetic. Division by a denominator whose absolute value is smaller than `1e-12` returns zero; powers clamp exponents to `[-8,8]`; exponentials clamp arguments; final expressions clamp to `±1e12`; symbolic predictions have a further bound. For example `2^100` evaluates to `256`, and `-2^2` evaluates to `4` under the current parser. These are existing language semantics. Changing them globally would invalidate old algorithms. [Protected arithmetic](/home/ubuntu/jope-dime/vendor/feat-math/src/math_expr.rs:517).

Use explicit profiles:

| Profile | Contract |
|---|---|
| Existing legacy versions | Preserve old parsing, defaults, protected math, axis meanings, RNG, width rules, and selected artifact behavior |
| New reference profile | Document IEEE/domain/error behavior, dtypes, reductions, rounding and RNG; no implicit global magnitude clamp unless the program requests it |
| Portable deterministic profile | Qualify specified operations across selected backends, with fixed reductions and explicit FMA/libm rules |
| Fast profile | Permit listed transformations/precision choices with a distinct identity and tolerance-based validation |

Current portable scalar helpers are useful, but Python/Rust recipe emission rejects `Portable`; C++/JavaScript support it only for their narrower admitted subset. Existing platform-libm agreement does not imply universal bit identity across machines, JS engines, or CUDA. [Current math policy](/home/ubuntu/jope-dime/vendor/feat-math/src/transpile.rs:44), [Python/Rust admission](/home/ubuntu/jope-dime/vendor/feat-math/src/transpile.rs:162).

The migration contract should be:

1. Old trees retain their source/version and load through the old parser and semantic adapter. New features and strict validation apply through new entry points. Preserve legacy acceptance of previously valid forms.
2. Store both the legacy identity and the new typed program identity. New hashes include semantic/opset version, typed constants, task/schema, numeric policy and relevant state; evaluation keys additionally bind data, splits, budget, implementation and RNG.
3. Keep old mutation/generation profiles immutable, including operation ordering and probabilities. New search profiles opt into changes. The existing code explicitly notes that changing vocabulary length alters recorded RNG streams.
4. Represent an old learner as a typed `legacy` intrinsic at first. It can be composed with new blocks where types permit. Do not claim old algorithms are fully evolvable internally until their bodies have been expressed in the new IR.
5. Preserve old fitted-model codecs and frozen emissions. New emitters and serializers receive distinct identities. Memory safety repairs should match the defined reference semantics; prior undefined C++ results are not a meaningful compatibility target.
6. Test old predictions/fitness under the same data, split, evaluator, RNG and numeric conditions. An old tree remaining valid does not imply its historical score remains valid under a different dataset or evaluator. Keep old scores attached to their original evaluation identity.

**What “any known or unknown algorithm” can reasonably mean**

With arithmetic, discrete indices, bounded storage, conditionals, reusable functions and programmable iteration, the kernel can aim to represent an extremely broad class of finite computations within an execution budget. It should have no fixed whitelist of acceptable learning equations. A fixed finite resource cap still excludes computations exceeding that cap; neither Turing completeness nor more operators guarantees useful evolutionary discovery.

Use concrete representability examples as a release requirement:

| Algorithm family | Required construction |
|---|---|
| Linear/logistic learners, momentum/Adam-like or novel optimizers | Tensor parameters, update state, reductions, optional AD |
| MLPs, residual/shared-weight networks | Matmul, nonlinearities, local functions, shared parameters, differentiation or custom updates |
| Attention and recurrent/state-space models | Axis-aware tensor operations, masking, normalization, scan |
| Trees, forests, boosting | Comparisons, data partitioning/indexing, bounded worklists, fitted structures, residual/loss updates |
| Kernel/nearest-neighbor and Gaussian-process methods | Pairwise computation, indexed memory, stable solvers, declared kernel assumptions |
| Clustering, EM, variational or sampling-based methods | Dataset reductions, iterative state, probability operations, explicit randomness |
| Multi-label and forecasting algorithms | Structured targets/outputs, masks, per-label or temporal state and correct evaluation contracts |

These examples should include algorithms reconstructed from the primitive core, not only calls to built-in implementations. Exact arithmetic, rationals, complex values, symbolic algebra, and proof certificates can be optional future modules if a discovery task needs them. Do not impose those costs on ordinary floating-point ML search.

Predictive improvement is evidence about an evaluated algorithm. A mathematical breakthrough needs a separate correctness/generalization claim and an appropriate verifier. AlphaTensor illustrates successful discovery in a carefully specified, verifiable algorithm space; it does not establish that arbitrary mathematical discovery follows from a universal syntax. [AlphaTensor](https://www.nature.com/articles/s41586-022-05172-4).

**Implementation order and acceptance criteria**

| Order | Deliverable | Acceptance evidence |
|---|---|---|
| 1 | Fix expression failure/bounds handling; add lossless identity; unify stage/head classification in a new admission/mutation path | Reproducers resolve with specified semantics; old freeze behavior retained; admitted children fit and satisfy selected backend capability |
| 2 | Share the expression AST/operation specification; add typed expression mutation and instruction execution | Reach currently unavailable formulas; lossless roundtrips; no invalid column references; measured unique valid offspring throughput |
| 3 | Add typed tensors, records, state, `let`, calls, branches and bounded iteration; initial Rust/Python/C++17 core | Hand-constructed SGD/momentum, MLP, recurrent and non-gradient examples run and export; state/loop/shape checks pass |
| 4 | Add optional AD, macro expansion and programmable loss/update search | Rediscovery/variation of known updates; AD checked numerically on smooth domains; budget includes derivative work |
| 5 | Add structured target/output ABI, true multilabel, forecast/state contracts | Matrix predictions and fitted state roundtrip; causal split/state tests; per-task metrics and backend parity |
| 6 | Qualify optimized execution, broader JavaScript and CUDA lowering | Matched-budget quality/throughput experiments, backend-specific parity and realistic hardware measurements |

For each step, keep a frozen control and compare both fixed evaluation budgets and fixed wall-clock budgets across repeated seeds. Include binary, regression, genuine multilabel and rolling-origin timeseries tasks, with independent final data. Measure valid unique children/second, compile/check overhead, fit/inference time, peak memory, error/no-op rates, and quality versus cost. Define the allowable non-regression margin before running the comparison; do not promise zero search degradation without that experiment.

The first two steps provide the best immediate return: they repair lost search capability and make existing mathematical syntax evolvable. The third step makes this a programmable learning-algorithm kernel. The remaining steps extend its practical reach and qualify performance.

**Validation performed during this audit**

| Check | Result |
|---|---|
| `cargo build -p feat-math` | Passed |
| Scalar math unit-test filter | 4 passed |
| Strict advanced-v3 grammar unit-test filter | 13 passed; this does not establish public-path integration |
| Existing target-stage placement unit test | Passed |
| Portable scalar parity fixture | Passed across Rust, C++17, Python and JavaScript |
| JavaScript/C++ portable claimed corpus | Passed, 44 combinations: 22 transforms × two heads |
| Public advanced-v3 Python/Rust/runtime bound parity | Passed for two fixture models |
| Fitted C++ deterministic numeric-pipeline test | Passed |
| Legacy freeze suite | 2 passed, 1 failed, 1 ignored |
| Added audit probes | Reproduced canonicalization changes, parsed-expression panic, public/strict admission differences, invalid/unchanged mutations and C++ bounds violation |

The legacy freeze failure is the pinned **source-file hash** for `tests/golden.rs`, which was already modified in the supplied working tree. The genome/prediction/emission freeze and parser/RNG/folds freeze passed. This is not evidence of a newly introduced prediction regression. [Saved freeze output](/home/ubuntu/jope-dime/docs/audits/algorithm-kernel-2026-09-08/legacy-freeze-test.log).

No full dataset campaign, throughput benchmark, or new kernel implementation was run. The measured probe counts establish the listed cases; the architecture and performance improvements remain recommendations to validate through the acceptance criteria above.
