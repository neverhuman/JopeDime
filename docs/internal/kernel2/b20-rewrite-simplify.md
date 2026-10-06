# Exact rewrite pass after each search round

The kernel2 profile route can run, opt-in, a deterministic rewrite pass at the end of each search
round. It simplifies the expressions of the round's best measured programs with e-graph saturation
under rewrite rules that keep every value bit for bit, and measures each smaller program the
profile admits as a new program. Its stable id is `feat-math.rewrite-saturate.v1`, built on
feat-math's `MathExpr::simplify_v2`. The pass draws from no random stream, and with it off nothing
calls it, so receipts and every published file keep their bytes.

## Switch

- CLI: `jope pipeline run --search-profile <kernel2 profile> --refine-top-k <k> --rewrite-simplify`.
  The legacy profile refuses the flag, as it refuses the `--refine-*` flags. The pass needs
  `--refine-top-k` of at least 1, which sets how many programs each round's pass reads. It
  combines with `--refine-method lm` and `--pareto-archive`.
- API: `EngineSelection { simplify: true, .. }` through `with_engine` on `Kernel2Evaluator`, the
  coordinator's `search::Options` or `run::Options`.

## What the pass does

After a round's measurements and its constant refinement, the pass ranks the round's measured
programs as the search ranks them, keeps the first occurrence of each program, keeps those with a
`map` or `derive` stage, and takes the first `top_k`. For each of them, in rank order:

1. Each `map` or `derive` stage's expression is simplified on its own, from its text alone. No
   row, label or score is read; the pipeline admission of step 4 reads only the input's column
   count.
2. `simplify_v2` admits the stage's expression under the profile (the expression pool, the
   15-node and depth-4 caps, and the constants-only negation rule), saturates it under the fixed
   rule set (at most 8 rounds and 8 e-nodes per input node, clamped to 16..=512), and extracts the
   smallest tree of its class (fewest nodes, then least depth). The tree is kept only when it is
   strictly smaller, or as small and shallower, and the profile admits it under the same check as
   every search edit. A stage whose own expression cannot be rewritten (it does not parse, the
   profile refuses it, or the rewrite fails) keeps its expression, and its entry records the stop
   `not_run` and the reason; the pass goes on, and the run does not end. A measured parent passed
   the same admission, so this is a guard, not an expected path.
3. A smaller tree the profile refuses is discarded with the profile's reason. For example,
   `c0*(-1.0)` reduces to the negation of `c0`, which the constants-only rule refuses, so the
   stage keeps `c0*(-1.0)`. The other stages' admitted rewrites still apply. The extraction is not
   changed to prefer admissible trees; that would be a different method id. Such drops are never
   silent: each is an entry with the profile's refusal, and the moves record counts them
   (`refused_by_pool`, under "Publication and verification").
4. When a stage changed, the program is reassembled and checked by the pipeline admission every
   measured program passes. A refused program is not measured, and its reason is recorded.
5. A program already measured is not measured again. Otherwise the start guard of refinement
   (cancellation and the seconds bound) is checked, and the program is measured like any other:
   one evaluation from the cut's shared allowance, with origin `rewrite`. (Not `simplify`: the
   search's own simplification edit already has that name in `arm_counts`.)

Proposal work is not charged; each measured program costs one evaluation and is one more
measurement, so the receipt's measurements and objective trials still add up to its completed
evaluations. The pass stops when the allowance is spent or the start guard refuses.

Every rule is exact under the engine's protected semantics on every row whose cells and literals
are not signaling NaNs, and text parsing never yields a signaling NaN. A rewritten program
therefore scores exactly as its original: the same training and held-out R², RMSE, selection score
and fault counts, bit for bit. Ranking has no cost term, so a rewritten program ties its original
and ranks by its text, as any tie does.

## Publication and verification

- `search/cut-0/moves.json`: one record per proposal, in the order the pass made them, with the
  method id, the round, the parent's sha256, the reassembled program (when a stage changed), the
  outcome, the pipeline admission's reason (when it refused), the measurement's index in
  `observations.json` (when measured), and one entry per expression stage: the stage index, the
  saturation's stop (`saturated`, `iteration_limit`, `node_limit` or `malformed`, or `not_run`
  when it never ran), its rounds, e-nodes, rule applications and peak e-nodes, and the admitted
  expression or the refusal.
- Outcomes: `unchanged` (nothing smaller), `refused` (no stage change survived the profile, or the
  pipeline admission refused the program), `duplicate` (already measured), `not_started` (the
  start guard refused the measurement; the run then stops on its seconds bound) and `measured`.
- `search.engine_selection.moves`, in the receipt and in `search/cut-0/search.json`:
  `{"ids": ["feat-math.rewrite-saturate.v1"], "sha256": "<sha256 of moves.json>",
  "refused_by_pool": <count>, "not_run": <count>}`. The selection's own record has neither the
  digest nor the counts; publication adds them, and receipt validation refuses a published record
  without a well-formed digest. The `ids` list leaves room for a later move, run in its order.
- `refused_by_pool` counts the times the pass dropped a smaller expression because the profile's
  admission (`Search::finish`: the expression pool, the caps and the negation rule) refused it:
  the stage entries with a refusal whose saturation ran. A unary minus over a variable under the
  constants-only negation rule is the case the rules produce (owner decision D9b: such outputs
  stay dropped, never silently). What it counts (the owner's D9b ruling):
  refusal events per round a parent is read; a parent ranked top for N rounds adds N. It is not
  a count of distinct expressions, and no second field counts those. Each refused stage entry is
  one event, so a parent with two such stages adds two in each round it is read. It is present
  exactly when the rewrite pass is selected, 0 included.
- `not_run` counts the stage entries with the stop `not_run`: a stage whose own expression could
  not be rewritten (it did not parse, the profile refused it, or the rewrite failed; a panic's
  own message is kept in the entry's reason when it carries one). A measured parent passed the
  same admission, so a genuine run publishes 0, and the replay tests assert it; a nonzero count
  makes such a fault visible in the receipt instead of hidden in `moves.json`. It is present
  exactly when the rewrite pass is selected.
- The v4 receipt schema gains the record as an optional field; its id is unchanged.

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives it:

- the record names the known id once, and is canonical;
- `moves.json` exists exactly when the record does, and matches the record's digest; without the
  record no measurement has the `rewrite` origin;
- the set and order of records: for each round, the measurements made before the pass are ranked
  as the pass ranked them, and the records must name, in order, the first `top_k` distinct ranked
  programs with an expression stage, one record each. Only the run's last round may stop short of
  that list, on the allowance or the start guard, in a run that stopped on its evaluations or
  seconds bound, and an unstarted record must be the pass's last. Under an evaluations stop the
  last round is pinned exactly: a parent's record may start only while the allowance lasts (a
  measurement of the run is still to come), and a pass that stops short must stop where it ran
  out (every measurement made). A deleted, inserted, swapped or reordered record, one for a
  program the pass did not read, or one for a round past the run's last (every record must be
  read), is refused, with one inherent exception: under a seconds stop, where the start guard
  refused is not in the evidence, so in the last round the trailing records that measure nothing
  can be deleted, and records that measure nothing appended for the next ranked parents,
  undetected;
- every record re-derives from its parent's published program alone: the stage entries, the
  program and the reason are recomputed and compared. The published parents are untrusted, so
  each is parsed within the B17 trace check's bounds (`parse_untrusted_pipeline`): the pipeline
  form only, and an expression source holding more than 256 of the bytes `( + - * / ^` is
  refused before the expression parser sees it;
- each outcome re-derives from the published measurements: a measured record's index is the next
  measurement, a `rewrite` measurement of its program in its round that no earlier measurement
  has; a duplicate names an earlier measurement; an unstarted record needs a seconds stop; and
  every `rewrite` measurement is accounted for by exactly one record;
- a measured rewritten program scores bit for bit as its parent: the same status, training and
  held-out R², RMSE, selection score and fault counts;
- `refused_by_pool` and `not_run` equal their counts in `moves.json`.

What it cannot prove: the records re-derive from published text and published scores, so they
show that each proposal follows from its parent and that each measured child scored exactly as its
parent did on this cut. They do not show that the measurements themselves are the cut's: no fit is
replayed.

Tests rehash `search.json` and the receipt after editing a record's program, stage expression,
stop, counters, refusal, outcome, measurement index, round, parent or id, removing, inserting,
swapping or reordering records (an unchanged, refused or duplicate record removed, a record for a
program the pass did not read, an unstarted record not last, a record appended after the
allowance was spent, the last measured record of the last round and everything after it removed,
the tail of an earlier round removed so that only the last-round rule sees it, a record appended
for a round past the run's last), editing a rewritten child's scores or fault counts, or the pool
refusal or not-run count, and after editing the selection record's id list or digest or dropping
it, and `verify_run` refuses every edit, as it refuses an edited or missing `moves.json`.

## Default behaviour is byte-identical

Without the flag the pass is not called, nothing is written and no record is added (the field is
skipped when absent), so the receipt and every published file keep their bytes. The existing
pinned default sidecars (`default_runs_keep_the_pre_wiring_search_sidecars`) and the frozen-clock
replay are unchanged, and a test for this flag repeats the seeded pins. With the flag, every
measurement before the first rewritten program is identical to the run without it.

## Evaluator identity

No evaluator identity changes. The pass lives in jope-pipeline's kernel2 route and in feat-math's
`math_expr/` and `kernel/` subdirectories, which the ADR-0004 implementation hash
(`compute_implementation_sha256` in `crates/jope-campaigns/src/evaluation_v2.rs`) does not cover.
No dependency is added.

## Not accuracy evidence

A rewritten program scores exactly as its original, and ranking has no cost term, so the pass
cannot raise the selected program's score; each measurement it spends is one the search does not
spend elsewhere. An equal-budget ablation is expected to show ties or losses on M1. Its value is
smaller programs, which shows only with a cost-aware ranking or the loss × cost front. The
ablation harness has no kernel2 profile yet (`ablation-protocol.md`).

## Tests

jope-pipeline lib, `kernel2::search::moves::tests`:

- rewritten programs score bit for bit as their parents (two fault policies, a `map` and a
  `derive` carrier, a fold), cost one evaluation each, keep the budget identity, and are measured
  once (a second spelling of the same program is a duplicate);
- the same bit for bit, fault counts included, with parents that raise DIV_NEAR_ZERO,
  LOG1P_DOMAIN and SQRT_PROJECTED cells, on a `-0.0` cell, and with a `map` stage that reads a
  row summary (the compiled evaluator);
- a refused output is discarded with the profile's reason, and a refused stage beside an admitted
  one keeps its expression; the pool refusal count of fixed parents is pinned;
- a parent whose only rewrite output the profile refuses, ranked top for N rounds of one run (five
  in the test), publishes a pool refusal count of N, which `verify_run` accepts and re-derives;
  N - 1 or N + 1, with `search.json` and the receipt rehashed, is refused;
- a parent whose expression `simplify_v2` refuses (outside the pool, or over the 15-node cap) is
  recorded as refused with the stop `not_run`, and the pass goes on to the next parent;
- the allowance, the start guard and `top_k` bound the pass;
- proposals are identical with a poisoned holdout, on fixed parents;
- frozen-clock replay of the receipt and every file with the flag, alone and with the damped
  least-squares refinement and the front, four parents per round in both, with `verify_run`
  accepting each run and every measured rewrite scoring as its parent;
- parents over the parsing bound are not readable and are refused unparsed; parents at it are
  read and rewritten as their plain form;
- tamper refusals, each by the check named for it where it is new (set and order, exactness, the
  pool refusal count), the record, the flag, and the shipped-text lint (of this note too);
- with the flag off: the seeded pins, no file and no record.

Not done: the three-language emitted-winner check of the rewritten programs. They are ordinary
pipelines over pool elements, so emission is unchanged.
