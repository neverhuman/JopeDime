# Certified fast ridge scoring (B11)

B11 makes the surrogate search's ridge scoring faster without moving a single score bit, and adds
certified ridge tools that later search work can build on. It is engine-internal: no operator,
wire, program or model ID changes, and the final fit head and output codec are unchanged.

## Exact fast fit

`RidgeModel::fit` keeps the legacy result bit for bit and computes it faster. Every output scalar
keeps the legacy sequence of IEEE-754 binary64 operations on the same operand bits, so loop order,
blocking, packing and caching cannot change it:

- **Column moments.** One row-major pass of per-column accumulators replaces one strided pass per
  column. Each column still sums its rows in ascending order from `0.0`.
- **Normal equations.** Every Gram and right-hand-side entry is a running sum that starts at `0.0`
  and adds `z_ri z_rj` (and `z_ri yc_r`) in ascending row order, as the legacy one-row-at-a-time
  rank-one update did. Rows run in panels of 128, which are packed column block by column block.
  A 4 x 4 tile of entries accumulates in registers over a panel, continuing from the sums earlier
  panels left. Lanes past the last column are zero and are never stored.
- **Solve.** The factorization and the triangular solves are `cholesky_solve` itself.

Rust neither fuses multiply-adds nor reassociates by default, and the kernel relies on that. A
verbatim copy of the previous loop is the test oracle. The oracle's bits are compared on designs
covering constant, heavy-tailed (the ±20 clamp is active), near-duplicate, integer-valued and
signed-zero columns, at widths that are and are not multiples of the tile, at the penalty floor
and with degenerate pivots.

## Certified scoring in the surrogate search

`EvolveConfig::ridge_scoring` is `RidgeScoring::Legacy` by default, which leaves the search
unchanged. `RidgeScoring::Certified` returns the same fold scores, and with lexicase the same
cases, bit for bit, with less work:

- **Surrogate memo.** The score depends only on the ridge surrogate the search fits
  (`search_surrogate`), which drops the head and caps the knobs. Candidates whose surrogates print
  the same text therefore reuse one score. Head-only mutations are the common case. The memo holds
  the fold scores, plus the residuals on the run's fixed lexicase case rows.
- **Stage cache.** Each fold fit goes through `model::fit_with_stage_cache`, keyed by the fold, the
  fold count and the chain of canonical stage texts. A hit returns the output of the identical
  computation, so shared transform prefixes are fitted once per fold.
- **Order and threads.** Distinct surrogates are scored once each, in candidate order. The result
  is the same on one thread and on four. `Observer::on_scoring` reports `candidates` and
  `surrogate_reuses`, which follow the candidate stream alone. It also reports the stage-cache
  `stage_hits` and `stage_misses`, which are instrumentation and depend on thread scheduling.
- **Refusals.** It applies to the surrogate search only, under `FaultPolicy::Legacy`.
  `RidgeScoring::validate` refuses the race and the strict policies.

## Certified tools (`kernel/ridge_fast.rs`, profile `feat-math.ridge-fast.v1`)

- **Prefix factor.** `Factor` is `G = ZᵀZ + λI = L Lᵀ`, built in `cholesky_solve`'s operation
  order, together with the forward vector `f = L⁻¹Zᵀyc`.
  - The factor of a design's leading columns is exactly the leading block of its full factor.
  - `Factor::extend` resumes from a prefix bit for bit.
  - Its coefficients equal the legacy solve's.
- **Schur gain.** For an appended column block, `schur_gain(p_a) = sum_{k >= p_a} f_k²` is the
  decrease of the ridge objective from the first `p_a` columns to all of them. It is tested
  against independent refits to 1e-10 relative.
- **PRESS / leave-one-out at a frozen penalty.** `e_(i) = e_i / (1 - h_ii)`, with
  `h_ii = ‖L⁻¹z_i‖²`. The rows, standardization, label mean and absolute penalty stay frozen. It is
  tested against `n` brute-force refits to 1e-10 relative.
- **Refusals.**
  - Any pivot below `λ/2` refuses (exact arithmetic gives at least `λ`, and this also covers the
    legacy `1e-12` branch), and the caller refits with the legacy authority.
  - So does any leave-one-out denominator below `1e-4`.
  - So do non-finite or misshapen designs.

## Evaluator identity

`heads.rs` and `evolve.rs` are ADR-0004 identity inputs, so B11 mints an evaluator identity at
publication, in a transition chained after B10's. No emitted byte, serialized model or search
stream changes.

## Evidence

`/home/ubuntu/jope-evidence/kernel2/bt/B11_RIDGE_FAST/`: kernel benchmarks (`kernel-benchmark-*.log`)
and the pinned 5 000-row end-to-end throughput runs (`e2e-*/`).
