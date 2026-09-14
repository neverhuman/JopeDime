# Faster program discovery through representation, reuse and adaptive precision

**Yes: smarter representations and selective precision are promising here. The best candidate design is an execution system that spends precision and computation according to the decision being made.** Preserve an exact authoritative evaluation path; make exploration cheaper only where experiments show it retains valuable candidates. This is an engineering and scientific proposal, not a measured breakthrough.

This design extends the [full red-team plan](REPORT.md). It incorporates independent [representation/source findings](precision-layout.md) and [hardware/precision findings](precision-addendum.md). No new fits or GPU benchmarks were run for it. All proposed numerical work remains separately reserved and governed.

## 1. The most useful “smarter data type” is not one floating-point format

Use a typed data block with dimensions, layout, missingness, frozen transformation state and numerical contract. Its storage can differ by column or operator while its logical meaning stays explicit. A conceptual interface is `DenseView<f64>`, `DenseView<f32>`, `CategoricalCodes<u8/u16>`, `IndicatorBits`, `BinnedFeatures`, and `SparseBlock`. This is a proposed representation family, not an instruction to add a large abstraction framework before profiling.

The current code already contains important pieces:

| Existing fact | Concrete opportunity | What must be demonstrated |
|---|---|---|
| Sealed evaluator input is contiguous little-endian f64; workers rebuild nested numeric rows | Borrow checked views of immutable mapped bytes, avoiding redundant full buffers and per-row allocations | Correct alignment/endian handling, bounds, lifetime and fewer measured copies; unchanged final bits |
| Many feature transforms allocate new nested rows; append can clone again | Plan widths once; use reusable workspaces and borrowed unchanged columns | Same column order, sanitize points, clipping and 512-width behavior; bounded peak allocation |
| Histogram bins already use flat u16; their accumulators are f64 | Reuse compatible training bin assignments; test u8 only if bin values and sentinels provably fit | Same bins, thresholds, ties and accumulation order; conversion cost included |
| GPR/KRR numerical matrices already use flat f64 | Reuse scratch and compatible expensive intermediate work; batch sufficient matching work | Actual head-level speedup; input flattening cannot be assumed to fix an already-flat solve |
| A stage-prefix cache already exists but the parallel fit route does not use it | Integrate verified immutable prefix reuse through authority | Complete cache identities, reference counting, concurrency and true peak-memory bounds |
| Program IDs already use 16-bit storage; vocabulary has 401 tokens | Keep compact IDs; avoid repeated string parse/canonicalization | No uint8 truncation; canonical identities survive lowering |

These facts are source-bound in [precision-layout.md](precision-layout.md). Root and embedded copies of eight inspected vendor files match; that is not a complete release closure.

Lossless integer coding is particularly attractive for categories, Boolean features, missing masks and existing histogram bins. It removes unnecessary floating-point storage without approximating the underlying category. It is not valid to quantize arbitrary continuous features and call that lossless. Preserve sorted dictionaries, unknown and missing sentinels, row order and feature roles.

Sparse/bitset execution is worthwhile only when supported operations stay sparse and conversion costs amortize. Centering a sparse column can densify it. Skipping zeros can change floating-point exceptional behavior and reduction order. Duplicate rows cannot generally be replaced by one weighted row unless the exact head, sampling and objective semantics support that replacement. Measure shape, density, cardinality and peak materialization before choosing a representation.

## 2. Precision should be assigned by operation

The inspected machine has eight A100-SXM4-40GB GPUs and exposed EPYC 7542 CPU flags including AVX2/FMA, without AVX-512/BF16 CPU acceleration. A100 has hardware support for BF16, TF32 and FP64 matrix operations; usable speed depends on compatible kernels and sufficiently large/batched work. Tensor-core peak ratios are not end-to-end speedups. TF32 is a computation mode for FP32 operands, not a compact storage type.[^hardware]

| Type/path | Best initial role | Why not use it everywhere? |
|---|---|---|
| FP64 | Existing authoritative fits, sensitive solves, exact comparison evidence | Required by the current fit contract; generic arithmetic can be slower, but tiny tasks may be dominated elsewhere |
| FP32 | First approximate screening experiment, using the same rows as its f64 reference | Half the element storage and twice the AVX2 lane capacity; no guaranteed 2× speed, and near ties/ill conditioning can change winners |
| BF16 with FP32 accumulation | Existing neural forward path; suitable large neural matrix work | Already used in training; coarse significand is risky for distances, covariance, thresholds and residual cancellation |
| Explicit TF32 | Measured FP32 GEMM bottlenecks in a separate inference/screen contract | No storage saving; input rounding can change ranking and numerical solves |
| INT8 | Exact category/bin codes, or separately calibrated neural inference | Quantized continuous values and neural scores need scales/calibration; not a drop-in GP solver or optimizer type |
| FP16 | Only a measured special case | No higher advertised dense A100 tensor peak than BF16 and narrower exponent range |

Keep FP32 master parameters, AdamW moments, gradient reduction and sensitive loss arithmetic in the existing neural training contract. Record actual deployed library versions and reduction/TF32 flags; do not assume current library defaults describe a frozen binary. Lower-precision inference must preserve candidate ordering and relevant score tails, not merely average logit error.[^hardware]

For regression heads, risks differ. Covariance factorization, weakly regularized ridge, PLS/SVD and cancellation-prone distance calculations need special treatment. Tree thresholds, nearest-neighbor ties and near-equal fitness can also flip under rounding. Scaling and preconditioning can help numerical conditioning, but changing them changes the model semantics unless they are already in the frozen preprocessing contract.

One advanced candidate is **mixed-precision iterative refinement**: use a lower-precision factorization, compute residuals with the original higher-precision matrix, and solve corrections, falling back when conditioning or convergence is unsuitable. Published numerical work establishes why this can recover high accuracy under suitable conditions. It does not promise identical historical solution bits, and retaining multiple precisions can increase memory. For Jope it belongs in a separate screened solver experiment, not a silent substitution for exact evidence.[^refinement]

## 3. Reuse expensive mathematical work before approximating it

The largest possible win may be eliminating repeated work shared by related programs. A mutation often preserves most of a transformation prefix. Retain immutable fitted prefix state and resulting training/scoring features when all dependencies match. A robust key includes canonical prefix, raw/prepared data, ordered training rows, target-dependent inputs, preprocessing, missing semantics, source/runtime, precision, and any stochastic state. Hash names and caller-supplied tags alone are not proof of equivalence.

Head-specific reuse may include histogram bin maps, unchanged kernel distances, or linear sufficient statistics. Each requires a dependency audit: changing a kernel length scale does not preserve its covariance matrix; changing rows invalidates statistics; changing a target-aware prefix changes more than X; and different histogram or reduction orders can change fitted results. Memory ownership and peak old-plus-new allocations matter under 104 workers. The existing stage-cache estimate and deferred sweep are not a hard concurrent memory bound.

Lower canonical programs to a typed operation schedule with shape and buffer-lifetime annotations. This enables specialized precompiled kernels and compatible batching. Reuse parsed programs and unchanged subexpressions, preserving evaluation order and intermediate clipping. Compile model-specific C++ or Rust only when expected future executions amortize compilation. The current C++ fitted-model route still invokes a model-specific compiler; moving that cost into the search loop could make small programs slower.[^layout]

Do not describe a Rust-to-C++17 rewrite as an optimization result. Rust can express contiguous storage, vectorized loops and native-library calls. A small C++/CUDA extension is justified by a specific measured kernel or integration need. Maintain separate contracts for the interpreter, emitted Rust, fitted C++ and CUDA.

Two source-confirmed head hotspots make this more concrete. Despite its name, `histgbm2` uses exact split search, with a once-per-fit presort and repeated sampled/child index lists. The separate `histgbm` head already provides histogram binning. Comparing that existing binned algorithm as a cheap screen is more direct than inventing a universal datatype. It is an algorithm change with possible quality loss. `histgbm2` defaults to 500 rounds but allows 1,000 and can stop early; default rounds are not measured rounds executed.

The composite GPR gradient path constructs three scratch vectors per kernel term, off-diagonal pair and optimization step. For one term, 256 optimization rows and 40 steps, the source loop implies 3,916,800 vector constructions. This is not a measured allocation count or timing share, and plain RBF uses another path. Reusable scratch and identical-expression reuse are strong candidates to test separately. Both head sources were independently matched to the retained native-v5 build in [head-hotspot-source.json](authority-gp/head-hotspot-source.json). No 1.7× or larger speedup is established by counting constructors.

Execution custody itself needs a performance budget. Current authority code performs runtime identity hashing, locking, durable journal/database work and witness acknowledgment. Measure lock wait, bytes hashed, fsync latency, witness delay and full accepted-task throughput before release. Qualified immutable-runtime caching, shorter critical sections or group commits may amortize work, but every affected token must wait for its independently retained durable acknowledgment. Cached identity must invalidate on process-birth, executable or mapping changes; PID-only caching is insufficient. Preserve this correctness boundary while making it fast.

## 4. Explore cheaply, verify survivors accurately

A proposed staged evaluator would work as follows:

1. **Static rejection/reuse.** Canonicalize, check types/shapes/cost limits, reject duplicate canonical programs, and reuse compatible committed original results. Do no numerical fitting to rediscover an excluded identity.
2. **Cheap screening.** Initially evaluate eligible candidates in FP32 on the same rows, or use a calibrated retained surrogate. Record precision, uncertainty, cost and all rejected outcomes. Never label approximate scores as full-f64 observations.
3. **Selective fidelity escalation.** Escalate promising, uncertain, rare-operator and near-tied candidates. Reserve an exploration fraction for candidates the screen would otherwise discard; retain selection probabilities to diagnose screen-induced bias.
4. **Authoritative final evaluation.** Use the unchanged full-data f64 evaluator for surviving new eligible computations; reuse compatible originals and keep ambiguous originals unavailable. Only appropriate verified full-fidelity outputs can enter the strict corpus.

A screen’s main accuracy question is **“did we retain the programs that matter?”**, not “is its average RMSE close?” High rank correlation can coexist with losing the single best candidate. Measure top-k recall, champion retention, missed-winner regret, per-head/per-family failures and resulting full-quality improvement per total reservation. A final f64 evaluator cannot recover a winner discarded before that stage.

A simple cost model makes the requirement concrete. For N candidates, screen time `Ts` per candidate, K finalists and full-evaluation time `Tf`, staged cost is approximately `N*Ts + K*Tf + overhead`. Evaluating all candidates costs `N*Tf`. If, hypothetically, `Ts=0.05*Tf` and `K=0.10*N`, the arithmetic allows at most about 6.7× improvement before overhead. This is an illustration, **not a measured Jope speedup**; winner loss or poor batching can make the design worse.

Adaptive resource allocation is supported by methods such as Hyperband, but candidate program fidelity is not automatically a monotone learning curve. Pruning thresholds must be calibrated for Jope’s heads and tasks. Lower optimizer steps, fewer trees, different bins, approximate kernels and smaller row sets are different algorithmic conditions, not merely different dtypes.[^hyperband]

For GPR specifically, source already sets default optimization steps to 40 and an optimization basis cap of 256. A universal 1,000-row screen is therefore not a direct solution to that inner optimization cost. A shorter optimization schedule or smaller basis is a separately identified approximation whose winner retention must be measured. Histogram methods already compress continuous data into bins; changing bin count can change the hypothesis class. Keep each fidelity axis separate before combining them.[^gpr]

Exact early rejection can occasionally avoid prediction work: accumulated nonnegative squared error gives a lower bound on the final full-denominator error. Only discard when a conservative bound proves a candidate cannot beat a threshold under the frozen arithmetic and scoring protocol. This does not save a completed fit, is not a full observation, and is useful only where prediction/scoring is material. Do not repurpose protected outcomes as an adaptive screening oracle.

Every numerical screen still needs authority, funding, a deadline, original identity and retained acceptance/completion. Current authority accepts f64 only; adding an FP32 screening namespace is a separately qualified extension. “It never writes training data” does not exempt it from compute accounting or historical exclusions. A changed executable or precision is not a device for rerunning a forbidden original computation.

## 5. Row subsampling must follow the actual workload

The active-index census has median 275 training rows, p90 800, p99 819 and maximum 4,751. Only 10/8,894 contexts exceed 1,000 rows. A 1,000-row cap affects 0.112% of those contexts. It may matter in a different future cohort, but it is not a broad accelerator for this one.[^data]

If a row-fidelity experiment is later warranted, consider a nested 128 → 256 → full training schedule where sizes permit. Use deterministic groups/strata selected from training data only; regression target quantiles may be used within that authorized training set, while temporal data must preserve valid order and horizons. Keep scoring roles and frozen protected panels unchanged. Refit each subset’s preprocessing from its own training rows under an explicit fidelity lineage. A nested subset does not generally make two fitted models exactly resumable.

Pilot precision on unchanged rows **first**, then row fidelity, then their combination only if both help. Otherwise a speed/quality change cannot be attributed to the datatype. Use genuine independent family coverage to test generalization; repeated cuts do not provide independent confidence in rare screening errors.

## 6. A bounded experiment that would justify deployment

Select a training-only fixture manifest **before opening payload files**, covering actual rows, transformed widths, dominant heads, cache states and numerical stress cases. Reuse retained original reference outputs wherever allowed. Include constant columns, large offsets, near-singular systems, tiny regularization, near ties, missing values and extreme target scales. No protected panel is required for this engineering stage.

Freeze source, binary, compiler/ISA, numerical flags, precise row/candidate identities, thread count, reservation, absolute deadline and output contract. Run sequentially qualified exact-layout and same-row FP32-screen conditions, with no more than the existing 16-slot canary concurrency and no new unbudgeted work. Current canonical budget availability is unresolved; these are proposed conditions.

| Decision | Proposed engineering gate to freeze before execution |
|---|---|
| Exact-layout replacement | All required exact model/prediction fixtures agree; no new failures; at least 10% median full-task improvement with no more than 5% p95 regression on the declared fixture population |
| FP32 screening eligibility per head | At least 99% reference top-10 recall in the fixed shortlist, no missed reference champion under the predefined tie rule, unchanged dimensions and explicit failure/fallback accounting |
| Useful cost improvement | At least 1.25× full screen-plus-finalist completion improvement, including copies, startup, compilation, verification, failures and reconciliation; warm and cold reported separately |
| Generalization of screening | Family/head-stratified miss and regret report with uncertainty; no universal reliability claim from a small zero-miss pilot |
| Production use | Qualified authority namespace and source closure; unchanged authoritative f64 evidence and promotion gates; rollback path and monitored distribution shift |

These thresholds are new proposed engineering criteria, **not replacements for the existing scientific promotion policy**. A small pilot cannot certify 99% population recall. If the pilot passes, qualify prospective monitoring and a conservative escape path for unfamiliar heads, extreme conditioning and shifted data. Stop screening a stratum when its retained-winner performance fails the frozen rule.

The existing CUDA GPR prototype should remain excluded: its retained fixture is about 3.23× slower cold end-to-end and fails its own looser qualification, before the required exact contract. A later resident batched kernel study can answer a different question; it must include initialization, transfers, queueing and finalists. Do not select on kernel-only timing.[^cuda]

The first investment is therefore: **remove demonstrated redundant work; measure head-specific cost; qualify exact representation/reuse improvements; then buy fewer expensive full fits through a screen that demonstrably preserves winners.** That combination has a credible route to substantially faster discovery. Its size of benefit remains an experimental result to obtain.

## Sources

[^layout]: [Authority precision/layout audit](precision-layout.md) and [source hashes](authority-gp/precision-layout-findings.json). These establish current implementations and proposed opportunities, not speed.
[^hardware]: [Native precision and hardware audit](precision-addendum.md), [hardware capture](learning-precision-hardware.json), NVIDIA [Ampere architecture](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/) and [GPU performance background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html). Hardware features and bottleneck concepts are separate from measured project throughput.
[^refinement]: Baboulin et al., [Accelerating scientific computations with mixed precision algorithms](https://www.netlib.org/utk/people/JackDongarra/PAPERS/202_2009_Accelerating-Scientific-Computations-with-Mixed-Precision-Algorithms.pdf), 2009. Used for lower-precision factorization with high-precision residual/correction and conditioning/fallback limitations.
[^hyperband]: Li et al., [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://www.jmlr.org/papers/v18/16-558.html), 2018. Adaptive resource allocation motivates this proposed program-screening design; the paper does not establish its Jope performance.
[^gpr]: [Current GPR source](../../../vendor/feat-math/src/gpr.rs), constants `DEFAULT_STEPS=40`, `OPT_CAP=256`, and the source identities in [representation findings](authority-gp/precision-layout-findings.json).
[^data]: [Active context census](data-folds-evidence/metadata-census.json) and [data/fold audit](data-folds.md). Older 765-cut geometry is a different denominator.
[^cuda]: [Retained CUDA prototype report capture](evidence/retained-results.json), original report SHA-256 `04cb88b43d47ae3c757fc052686f311b779df2e4553884d9e6be62c8885f7fe5`.
