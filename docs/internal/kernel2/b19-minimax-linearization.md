# Bounded minimax linearization after each search round

The kernel2 profile route can run, opt-in, a deterministic pass at the end of each search round
that offers a bounded minimax linearization of one-variable subtrees. A subtree of a `derive`
stage's expression that reads a single variable is sampled on that stage's training input, fitted
by its degree-1 discrete minimax line, and replaced by the line only when the line meets the fit's
minimax bound on those rows and the profile admits the result. Each offered program is measured
as a new program, and its measured score decides where it ranks.

It is degree 1 only, by the owner's ruling. Under the kernel2 caps (15 nodes, depth 4) a dense
polynomial of degree `n` (every coefficient nonzero) lowered in Horner form over its variable has
depth `2n + 1`, so no dense higher degree fits an admissible expression; a sparse one can be
shallower (the lowering skips zero coefficients, so `2 x²` lowers to `c0*(c0*2)`, depth 3), but it
is not built either. This is a linearization, not a general polynomial or rational compression;
the higher-degree case is held (see the last section but one).

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

What the bound measures: it is the fit's own largest training error, with room for rounding. It
checks that lowering the fit to ordinary nodes, `x * a1 + a0` evaluated in that order, lost no
precision; it does not say the line is a good stand-in for the subtree. A poor fit has a large
`max_error` and a bound to match, and its line is still offered; the native score is the only
judge of quality. (Open question for the owner, with the D7 wording "bounded minimax
linearization": whether the pass should also refuse lines whose fit error is large relative to
the subtree's range, which would be a separate, versioned rule.)

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
- The linearization pass adds no receipt field: the selection record already holds an id list, and
  the move records are a sidecar. (With the rewrite pass also selected, the record carries that
  pass's `refused_by_pool` and `not_run` counts, the optional fields the v4 schema gained for it.)

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives the records:

- the selection names known ids once, in their order, and is canonical;
- the set and order of records: for each round, the measurements made before the pass (this
  round's rewritten programs included) are ranked as the pass ranked them, and the records must
  name, in order, the first `top_k` distinct ranked programs with a subtree the pass considers,
  each with one record per considered subtree. Only the run's last round may stop short of that
  list, on the allowance or the start guard, in a run that stopped on its evaluations or seconds
  bound, and a pass that ended must end with that parent's records. Under an evaluations stop the
  last round is pinned exactly: a parent's records may start only while the allowance lasts (some
  measurement of the run is still to come), and a pass that stops short must stop where it ran
  out (every measurement made). Under a seconds stop it is not: where the start guard refused is
  not in the evidence, so in the last round the trailing records that measure nothing can be
  deleted, and records that measure nothing appended for the next ranked parents, undetected. A
  parent's whole group of records deleted, a group for a program the pass did not read, or groups
  swapped are refused;
- each parent's records re-derive from its published program and the fits they record: the
  considered subtrees and their order, every stage, path, subtree and node count, the degree, the
  bound, the bracket rule, the line and its nodes from the recorded coefficients, every refusal
  after the fit, the admitted expression, the program and the pipeline admission's reason are
  recomputed and compared. The published parents are untrusted, so each is parsed within the B17
  trace check's bounds (`parse_untrusted_pipeline`: the pipeline form only, and an expression
  source holding more than 256 of the bytes `( + - * / ^` refused before the expression parser
  sees it), and the public linearization entries check a tree's structure before they copy it;
- the recorded evidence agrees with itself, checked per stage expression: each fit is consistent
  with its stop and within the evaluator's range (`|lo|`, `|hi|` and `max|f|` at most 1e12, the
  only range the sampling produces), over a range that holds three distinct abscissae (a line's
  fit needs three), with its bracket's ends and `max|f|` of positive sign; a closing stop
  records the fit of its own step (the iteration cap, the last step), and only `exact_fit` is
  within the exact-fit test, which the fit applies before any other stop; each line's measured
  error, its largest absolute residual, is finite and not negative (negative zero included), at
  most `1e12 + max|f|`, and, for a line that can be lowered, at least the line's residual at
  either sampled end, `lo` or `hi` (`|p| - max|f|`), where the line must be in the evaluator's
  range, at most `M + max|f|`, and at least `max|f| - M`, with `M` the line's larger size at the
  two ends (its computed values are monotone between them, and some sample attains `max|f|`).
  A line whose coefficients are not finite is refused before any measurement, so its ends are
  not read. Those are the bounds the live measurement keeps, and rounding is monotone, so they
  hold exactly with no allowance; a faulting training input refuses every
  subtree of its stage or none; a subtree that occurs twice in one expression records the same
  samples both times, compared bit for bit (a coefficient that is not finite is published as
  absent and read back as NaN, which compares equal to itself only that way); the
  fits of one variable in one stage share its training range; and a fit and a refusal of that
  variable's values alone (too few distinct values) do not both occur. The range bound stops an
  inflated `max|f|` only past 1e12: within it, an `exact_fit` claim still closes for a `max_error`
  up to `4 ε 1e12` (about 8.9e-4), and a `noise_limited` one for a bracket up to `64 ε 1e12`
  (about 0.014), with the bound widened to match;
- each outcome re-derives from the published measurements, as the rewrite pass's do, a
  `no_allowance` record needs an evaluations stop, nothing is measured after a pass ended, and
  every `linearize` measurement is accounted for by exactly one record.

What verification cannot prove. Without the training rows, `verify_run` checks each record's
internal consistency and re-derives the line, the program and the outcome from it; it does not
re-measure the fit. The fit, its training range, `max|f|` and the line's measured error are taken
as recorded. So an edit that keeps those numbers consistent, meeting every check above, passes:
for example, a closed fit's
`max_error` and `levelled` scaled up together, with the bound recomputed to match, re-derives the
same decisions and is accepted (a test pins this limit). So does a refused subtree's fit replaced
by the refusal "a training row faults or is not finite in the subtree", which only that subtree's
own rows could contradict, unless the same subtree occurs twice. Neither edit can change which
programs were measured or how they scored: an offered line's record is bound to its measured
program, and the measurement is the native score. A fit replay needs the training rows. The
line's measured error is checked only against what its recorded line and samples allow (above):
within that, an offered line's error edited up to its bound re-derives the same decision and is
accepted (a test pins this limit). The bracket's lower end is no floor for it, since the lowered
line's own rounding grows with its coefficients' conditioning.

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
and losses overall and per dataset, beside both medians) has not been run: the ablation harness
has the `kernel2-baseline` and `kernel2-experimental` profiles (PR-0, #221;
`ablation-protocol.md`) but no arm that selects this pass.

## Held follow-up: degree above 1

Held, not built. A dense degree 2 lowers to depth 5 in Horner form and a dense degree `n` to
`2n + 1`, over the kernel2 depth limit of 4, and the plan forbids raising the caps; only sparse
higher-degree polynomials (zero coefficients skipped) can be shallower. A higher-degree move would need
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
  drive the bound and must agree with their stop and the evaluator's range, with their own step
  and positive zeros, an exact fit under its own stop only, and a range of three distinct
  abscissae; a recorded line error that is negative, negative zero or not finite is refused, and
  so is one its line and samples cannot give (past `1e12 + max|f|`, smaller than the line's
  residual at a sampled end, for a line outside the range there, past `M + max|f|`, or under
  `max|f| - M`), each accepted at its exact bound; records of one expression must
  agree with each other (a repeated subtree, bit for bit, with NaN coefficients equal to
  themselves and other numbers still refused; one variable's training range; a fit beside a
  refusal of the same values);
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
  and with the damped least-squares refinement, the front and the rewrite pass, four parents per
  round in each, each verified;
- parents over the parsing bound offer nothing and are refused unparsed, and parents at it offer
  what their plain form offers; the public linearization entries refuse a decoded tree outside
  the structure limits before they copy it (feat-math);
- tamper refusals, the selection record, the flag, and the shipped-text lint (of this note too);
  among them a fit inflated past the evaluator's range, a fit hidden behind a faulting-input
  refusal beside sibling fits, a parent's group deleted, swapped with the next one, or repeated
  after the next one, each refused by the check named for it; and the three documented limits,
  accepted: a consistently scaled fit, a refused line's fit hidden as an unsampled subtree, and
  an offered line's error edited up to its bound
  (a group for a parent the pass did not read is tested for the rewrite pass, whose records share
  the same parent-order check); a measured line's error edited to `-1`, `-0` or `1e100`, or past
  its line's size, or kept while its line moves off its samples or under a largest target far
  past the line, is refused;
- a run on a one-feature dataset whose values are subnormal (`-1e-310`, `0`, `1e-310`): the two
  equal children of `(sin(c0)+c0)*(sin(c0)+c0)` are refused for a coefficient past the finite
  range, published without their coefficients, and the untouched run verifies; a finite pair
  given to one of the two is refused as a repeated subtree recording other samples;
- with the flag off: the seeded pins, no file and no record.

Not done: the three-language emitted-winner check of linearized programs. They are ordinary
pipelines over pool elements, so emission is unchanged.
