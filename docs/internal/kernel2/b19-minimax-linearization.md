# Bounded minimax linearization after each search round

The kernel2 profile route can run, opt-in, a deterministic pass at the end of each search round
that offers a bounded minimax linearization of one-variable subtrees. A subtree of a `derive`
stage's expression that reads a single variable is sampled on that stage's training input, fitted
by its degree-1 discrete minimax line, and replaced by the line only when the line meets the fit's
minimax bound on those rows and the profile admits the result. Each offered program is measured
as a new program, and its measured score decides where it ranks.

It is degree 1 only. Under the kernel2 caps (15 nodes, depth 4) a polynomial of degree `n` lowered
in Horner form over its variable has depth `2n + 1`, so no higher degree fits an admissible
expression. This is a linearization, not a general polynomial or rational compression; the
higher-degree case is held (see the last section but one).

Its stable id is `feat-math.minimax-linearize.v1`, built on feat-math's `MathExpr::linearize_v2`.
The pass draws from no random stream, and with it off nothing calls it, so receipts and every
published file keep their bytes.

## Switch

- CLI: `jope pipeline run --search-profile <kernel2 profile> --refine-top-k <k>
  --linearize-subtrees`. The legacy profile refuses the flag, as it refuses the `--refine-*`
  flags. The pass needs `--refine-top-k` of at least 1, which sets how many programs each round's
  pass reads. It combines with `--rewrite-simplify` (the rewrite pass runs first),
  `--refine-method lm` and `--pareto-archive`.
- API: `EngineSelection { linearize: true, .. }` through `with_engine` on `Kernel2Evaluator`, the
  coordinator's `search::Options` or `run::Options`.

## What the pass does

After a round's measurements, its constant refinement and the rewrite pass (when selected), the
pass ranks the round's measured programs as the search ranks them, keeps the first occurrence of
each program, keeps those with a `derive` stage that has a subtree it can consider, and takes the
first `top_k`. For each of them, in rank order, and each `derive` stage in stage order:

1. The stage's training input is built from the cut's training rows and labels alone: the earlier
   stages are fitted and applied on them, as the constant refinement does. No held-out row or
   label is read. If an earlier stage faults on them, every considered subtree of the stage is
   recorded as refused, unsampled.
2. The subtrees considered are those, in preorder, that read exactly one variable, have more than
   three nodes (the smallest line, `x * a`, has three), and sit where that line fits the depth
   limit. Under depth 4 these are the root and its children, and a grandchild only when it is a
   call of three arguments. `map` stages are not considered: their expressions read each cell, not
   each row.
3. `linearize_v2` admits the stage's expression under the profile, then for each subtree samples
   `(x, f(x))` on every training-input row, fits the discrete minimax line (the exchange algorithm
   in feat-math's `simplify.rs`, degree 1), lowers it to ordinary nodes, `x * a1 + a0`, applies the
   refusal rule of the next section, and splices the line in place of the subtree.
4. When the line is offered, the program is reassembled and checked by the pipeline admission
   every measured program passes. A refused program is not measured, and its reason is recorded.
5. A program already measured is not measured again. Otherwise the allowance and the start guard
   of refinement (cancellation and the seconds bound) are checked, and the program is measured like
   any other: one evaluation from the cut's shared allowance, with origin `linearize`.

Each offered line is its own child program, so one parent can yield several. Proposal work (the
training input, sampling, fitting and lowering) is not charged; each measured program costs one
evaluation and is one more measurement, so the receipt's measurements and objective trials still
add up to its completed evaluations. When the allowance is spent or the start guard refuses, the
parent's remaining offers are recorded unmeasured and the pass ends.

## The refusal rule

A line is offered only when every check holds, in this order; the first that fails is the
recorded refusal:

1. Sampled: on every training-input row the subtree and its variable evaluate with no fault and a
   finite value. Otherwise `a training row faults or is not finite in the subtree` (or, before any
   sampling, `the stage's training input faults`).
2. Fitted: the fit accepts its samples. It refuses fewer than three distinct values of the
   variable, a value above the evaluator's range (1e12), or a singular first reference. Otherwise
   `the fit refused its samples: <the fit's reason>`.
3. Closed: the fit's stop is `exact_fit`, `converged` or `noise_limited`, the stops whose error
   bracket `[levelled, max_error]` closed to their own tolerance (a few ulps of `max|f|`,
   `1e-12 max_error`, and the rounding level `64 ε max|f|` with the exchange spent). A `stalled`,
   `degenerate` or `iteration_limit` fit is only the best fit seen, and its `max_error` need not be
   the minimax error. Otherwise `the fit's bracket did not close: <stop>`.
4. Lowered: both coefficients are finite (`a coefficient of the line is not finite`); the line has
   fewer nodes than the subtree (`the line is not smaller than the subtree`); it fits the node
   limit the rest of the expression leaves it, 15 minus the expression's nodes outside the subtree
   (`the line is over the expression's node budget`); and it fits the 64-node, depth-16 expression
   limits on its own (`the line exceeds the expression limits`).
5. Within the minimax bound: the line's own largest error over the training samples, `lowered
   error`, evaluated in the line's own operation order with the evaluator's arithmetic, stays in
   the evaluator's range (`the line leaves the evaluator's range on a sample`) and is at most the
   bound `max_error (1 + 1e-6) + 64 ε max|f|` (`the line misses the minimax bound on the samples`).
   `max_error` is the fit's largest error over the samples in its Chebyshev basis and `max|f|` the
   largest absolute value of the subtree over them. The monomial line can lose the fit far from
   0: on a training range near 1e6 its rounding is about 1e-10, far over the bound when the fit's
   own error is small.
6. Admitted: the profile admits the whole expression with the line in place (the pool, 15 nodes,
   depth 4, and the constants-only negation rule). A negative slope lowers to a minus over a
   literal, one node and one level more, so one level under the root it takes the expression to
   depth 5. Otherwise `the profile refused the linearized expression: <the profile's reason>`.

The contract is narrow. `max_error`, `levelled` and the lowered error are floating-point
residuals at the training samples, not certified bounds; the bound says nothing between the
samples' values in other rows, and nothing off the training range, where the line can be far from
the subtree. The pass never accepts a program on its line's error: it only offers it, and the
whole program's native protected measurement on the cut decides where it ranks.

## Publication and verification

- `search/cut-0/moves.json`: the linearization pass adds one record per considered subtree, in the
  order the pass made them, with the method id, the round, the parent's sha256, the reassembled
  program (when the line was offered), the outcome, the pipeline admission's reason (when it
  refused), the measurement's index in `observations.json` (when measured), and the subtree's
  record: the stage, its path, the subtree as printed, its nodes, the degree (always 1), the fit
  (stop, `[a0, a1]`, the training range, `levelled`, `max_error`, `max|f|`, the exchange step and
  steps run), the bound, the line's nodes, the lowered error, the admitted expression, and the
  refusal.
- Outcomes: `refused` (the line, or the program, was refused), `duplicate` (already measured),
  `not_started` (the start guard refused the measurement; the run then stops on its seconds
  bound), `no_allowance` (the allowance was spent; the run then stops on its evaluations bound)
  and `measured`.
- `search.engine_selection.moves`, in the receipt and in `search/cut-0/search.json`:
  `{"ids": ["feat-math.minimax-linearize.v1"], "sha256": "<sha256 of moves.json>"}`, or
  `["feat-math.rewrite-saturate.v1", "feat-math.minimax-linearize.v1"]` with both passes, in the
  order each round runs them.
- The v4 receipt schema is unchanged: the selection record already holds an id list, and the move
  records are a sidecar.

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives the records:

- the selection names known ids once, in their order, and is canonical;
- records are in round order, the rewrite pass's before the linearization pass's within a round,
  with at most `top_k` parents per pass and round, each parent read once per round;
- each parent's records re-derive from its published program and the fits they record: the
  considered subtrees and their order, every stage, path, subtree and node count, the degree, the
  bound, the bracket rule, the line and its nodes from the recorded coefficients, every refusal
  after the fit, the admitted expression, the program and the pipeline admission's reason are
  recomputed and compared;
- each outcome re-derives from the published measurements, as the rewrite pass's do, a
  `no_allowance` record needs an evaluations stop, nothing is measured after a pass ended, and
  every `linearize` measurement is accounted for by exactly one record.

Not re-derived: the fit itself and the line's measured error. The training rows are not published
evidence, so verification takes them as recorded and checks only that they agree with the fit's
stop (finite values, a bracket that does not invert, at most 32 steps, and the stop's own closing
test). An edit to those numbers that keeps them consistent and leaves the bound, the line and the
decisions unchanged is not detected; a fit replay needs the training rows.

## Default behaviour is byte-identical

Without the flag the pass is not called, nothing is written and no record is added (the field is
skipped when absent), so the receipt and every published file keep their bytes. The existing
pinned default sidecars and the frozen-clock replays are unchanged, and a test for this flag
repeats the seeded pins. With the flag, every measurement before the first linearized program is
identical to the run without it.

## Evaluator identity

No evaluator identity changes. The pass lives in jope-pipeline's kernel2 route and in feat-math's
`math_expr/` and `kernel/` subdirectories, which the ADR-0004 implementation hash
(`compute_implementation_sha256` in `crates/jope-campaigns/src/evaluation_v2.rs`) does not cover.
No dependency is added.

## Not accuracy evidence

A line in place of a curve usually loses accuracy unless the program's later stages gain from the
simpler feature, and each measurement the pass spends is one the search does not spend elsewhere.
The tests show a line that ranks above its parent and a line that is worse off the training
range; neither is evidence of lift. An equal-budget ablation (on against off, paired wins, ties
and losses overall and per dataset, beside both medians) waits for kernel2 arms in the ablation
harness (`ablation-protocol.md`).

## Held follow-up: degree above 1

Held, not built. Degree 2 lowers to depth 5 in Horner form and degree `n` to `2n + 1`, over the
kernel2 depth limit of 4, and the plan forbids raising the caps. A higher-degree move would need
one of: a profile with deeper caps, a lowering that is not Horner over ordinary nodes (a power node
has unspecified precision), or a rational form `P(x)/Q(x)`, which also needs the pole-free
certificate (`certify` in `simplify.rs`, compiled for its tests only). It would be a separate
method id with its own refusal rule, records, tests and ablation, after an owner decision. The fit
itself already handles degrees up to 12, and its tests keep them.

## Tests

feat-math lib, the `linearization_tests` module of `simplify.rs`:

- a near-linear subtree becomes its line within the bound, spliced at the root and under a root
  that reads two variables, with the native line within its measured error on every training row;
- a line that misses the minimax bound on the training range is refused by name, on a range near
  1e6, while the same expression on a centred range is offered;
- a fit whose bracket did not close (`stalled`, `degenerate`, `iteration_limit`) is refused before
  any line is built, and `closed` holds for exactly the three closing stops;
- the recorded path re-derives the live result exactly, across every refusal; recorded numbers
  drive the bound and must agree with their stop;
- each check of the lowering is named in order; only one-variable subtrees where a line fits are
  considered.

jope-pipeline lib, `kernel2::search::moves::linearize_tests`:

- refusal: a line over its bound is refused with its reason, and nothing is measured or charged;
- acceptance: lines are measured once each, with origin `linearize`, scored natively, and ranked
  by that score, not by their lowered error;
- holdout counterexample: a line offered on the training range is worse off it, and its native
  score ranks it under its parent;
- proposals are identical with a poisoned holdout, on fixed parents;
- the allowance, the start guard, `top_k` and deduplication bound the pass, and the budget
  identity holds;
- frozen-clock replay of the receipt and every file with the flag alone, with the rewrite pass,
  and with the damped least-squares refinement, the front and the rewrite pass, each verified;
- tamper refusals, the selection record, the flag, and the shipped-text lint (of this note too);
- with the flag off: the seeded pins, no file and no record.

Not done: the three-language emitted-winner check of linearized programs. They are ordinary
pipelines over pool elements, so emission is unchanged.
