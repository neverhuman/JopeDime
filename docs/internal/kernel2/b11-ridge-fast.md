# Certified fast ridge scoring (B11)

B11 makes the surrogate search's ridge scoring faster without moving a single score bit, and adds
certified ridge tools that later search work can build on. It is engine-internal: no operator,
wire, program or model ID changes, and the final fit head and output codec are unchanged.

## Exact fast fit

`RidgeModel::fit` keeps the legacy result bit for bit and computes it faster. Every output scalar
keeps the legacy sequence of IEEE-754 binary64 operations on the same operand bits, so loop order,
blocking, packing and caching cannot change it. The one exception is a NaN result's sign and
payload, which Rust leaves unspecified in either build; an optimized build was seen to give the
two loops' NaNs opposite signs. NaN positions, infinities, signed zeros and every finite bit are
kept:

- **Column moments.** One row-major pass of per-column accumulators replaces one strided pass per
  column. Each column still sums its rows in ascending order from `0.0`.
- **Normal equations.** Every Gram and right-hand-side entry is a running sum that starts at `0.0`
  and adds `z_ri z_rj` (and `z_ri yc_r`) in ascending row order, as the legacy one-row-at-a-time
  rank-one update did. Rows run in panels of 128, which are packed column block by column block.
  A 4 x 4 tile of entries accumulates in registers over a panel, continuing from the sums earlier
  panels left. Lanes past the last column are zero and are never stored.
- **Solve.** The factorization and the triangular solves are `cholesky_solve` itself.
- **Memory.** The fit keeps the whole standardized design for its panels, where the legacy loop
  kept one standardized row, so a fit's peak memory grows by `8·n·p` bytes.

Rust neither fuses multiply-adds nor reassociates by default, and the kernel relies on that. A
verbatim copy of the previous loop is the test oracle. The oracle's bits are compared on designs
covering constant, heavy-tailed (the ±20 clamp is active), near-duplicate, integer-valued and
signed-zero columns, at widths that are and are not multiples of the tile, at the penalty floor,
with zero-variance and huge constant columns, and on inputs that really overflow or are not
finite: a column of `f64::MAX`, alternating `±f64::MAX`, a single `+inf` or NaN cell, and a `-inf`
or NaN label. Each of those cases is checked to reach a non-finite value, and NaNs compare by
position.

## Certified scoring in the surrogate search

`EvolveConfig::ridge_scoring` is `RidgeScoring::Refit` by default, which leaves the search
unchanged. `RidgeScoring::Certified` returns the same fold scores, and with lexicase the same
cases, bit for bit, with less work:

- **Surrogate memo.** The score depends only on the ridge surrogate the search fits
  (`search_surrogate`), which drops the head and caps the knobs. Candidates with exactly the same
  surrogate therefore reuse one score. Head-only mutations are the common case. The memo holds
  the fold scores, plus the residuals on the run's fixed lexicase case rows.
- **Stage cache.** Each fold fit goes through `model::fit_with_stage_cache`, keyed by the fold, the
  fold count and the chain of exact stage encodings. A hit returns the output of the identical
  computation, so shared transform prefixes are fitted once per fold. The cache's counted bytes
  never pass its 1 GiB budget, not even within a generation. They are the f64 payload and the
  serialized fit, not allocation overhead. A full cache refuses new keys until a
  generation-boundary sweep drops the entries a whole generation passed without using, and a
  refused key is recomputed on its next miss, with identical bits.
- **Exact keys.** The memo and the stage cache key on each stage's exact encoding: its op and every
  parameter in key order, length-prefixed and type-tagged, with every float as its IEEE-754 bits.
  They never key on canonical text, which prints a float within 1e-12 of an integer as that
  integer, so two different computations can print alike. Surrogates that print alike but differ
  in any bit are therefore scored apart, as `Refit` scores them. Regression tests pin this for
  `gamma` = 1.0 against 1.0 + 5e-13, and for `lag k` = 2.0 against 2.0 - 5e-13, which `lag`
  truncates to 2 and 1: two different programs that print alike. Further tests pin floats within
  1e-12 of zero, two NaNs and strings that read as numbers or bools apart in the stage cache, a
  cached fit of `symlog c` = 1.0 + 5e-13 equal to its plain fit, and, through the whole search on
  one thread and on four, the independent review's two `lag` seeds that print alike scored as
  `Refit` scores them. The memo keys on the encodings themselves; a stage-cache key is the
  SHA-256 of its length-prefixed fields, so two different chains share one only through a
  SHA-256 collision. Neither key holds the data: one scorer serves one search run, whose rows,
  labels, folds and case rows are fixed. The search's own genome memo, which merges genomes whose
  whole canonical text matches, is unchanged and acts identically in both modes, so it never
  makes them differ.
- **Changes to the shared `StageCache`.** Two. Its other users are the campaigns' `exact_cv`
  command, which uses it unless `JOPE_STAGE_CACHE=0`, and the train-only study.
  - Its keys are exact. Before B11 two stages that printed alike shared one entry, so the second
    could receive the first's block instead of its own; now each gets its own computation. The
    key's domain tag moved from `jope-stage-cache.v1` to `v2`. Keys live only in memory. The
    other users evaluate only genomes parsed from canonical text that prints back to itself, so
    exact keys group their stages exactly as text did: nothing they compute or count changes.
  - Its budget holds at insert. Before B11 it admitted any entry no larger than the whole budget,
    could pass the budget within a generation, and evicted oldest-first at the next sweep. Its
    other users still get the same values; only which entries stay resident, and so the hit
    counts, can differ.
- **Order and threads.** Distinct surrogates are scored once each, in candidate order. The result
  is the same on one thread and on four. `Observer::on_scoring` reports `candidates` and
  `surrogate_reuses`, which follow the candidate stream alone. It also reports the stage-cache
  `stage_hits` and `stage_misses`, which are instrumentation and depend on thread scheduling.
- **Refusals.** It applies to the surrogate search only, under `FaultPolicy::Legacy`.
  `RidgeScoring::validate` refuses the race and the strict policies, and the search panics with
  that reason when it is given either; tests run both panics.

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
- **Refusals.** On any refusal the caller refits with the legacy authority.
  - A pivot that is not finite or is below `λ/2` refuses. Exact arithmetic gives at least `λ`,
    and since every design's penalty is at least `2e-12`, this also covers the legacy `1e-12`
    replacement branch. A pivot that roundoff raises instead is accepted.
  - So does a non-finite factor entry or forward value, which a finite design whose Gram or
    right-hand side overflows produces, and any leave-one-out denominator below `1e-4`.
  - So do non-finite columns or labels, labels that overflow when centered, misshapen designs
    (ragged rows included) and a factor of another width or penalty.
  - Pairing a factor with its own design is the caller's guarantee, for `Factor::extend` and
    `leave_one_out` alike; only another width or penalty can be detected.

## Evaluator identity

`heads.rs`, `evolve.rs` and `stage_cache.rs` are ADR-0004 identity inputs, so B11 mints an
evaluator identity at publication, in a transition chained after B10's. No emitted byte,
serialized model or search stream changes.

## Evidence and benches

This note quotes no speedup. Timing evidence is under
`/home/ubuntu/jope-evidence/kernel2/bt/B11_RIDGE_FAST/`. Every session ran on xbabe2 (128 CPUs,
shared with other lanes). The later sessions (`bench-final/session-*/`) ran through
`bench-final/bench-final.sh`, which builds the head's unit tests and the parent's harness
(`bench-final/parent_throughput.rs`) in release mode and runs the same tests with
`--ignored --exact`. Each session's `provenance.txt`, like the earlier end-to-end sessions'
(`prep/e2e-*/provenance.txt`), records the head and parent it measured, the binaries' hashes and
the host load. Sessions started after `bench-final.sh` gained its host line also record the host
at run time; the first `bench-final` session carries it only as an annotation added afterwards,
and the `prep/` sessions not at all. Sessions before exact keys measured earlier heads of this
card; the pull request lists every session with the head it measured. The benches assert nothing
about speed. There are three `#[ignore]`d timing benches, all unit tests in
`kernel/ridge_fast.rs`:

- the fast fit against the verbatim legacy loop at four widths (`B11_KERNEL` lines):
  `cargo test --release -p feat-math --lib kernel::ridge_fast::tests::kernel_benchmark -- --ignored --nocapture`;
- the ridge fit's share of surrogate fold fits (`B11_SHARE`), and end-to-end search throughput on
  the pinned 5 000-row fixture (`B11_E2E`), both in `kernel::ridge_fast::bench`:
  `cargo test --release -p feat-math --lib kernel::ridge_fast::bench -- --ignored --nocapture`.
