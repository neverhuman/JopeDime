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
   every search edit.
3. A smaller tree the profile refuses is discarded with the profile's reason. For example,
   `c0*(-1.0)` reduces to the negation of `c0`, which the constants-only rule refuses, so the
   stage keeps `c0*(-1.0)`. The other stages' admitted rewrites still apply. The extraction is not
   changed to prefer admissible trees; that would be a different method id.
4. When a stage changed, the program is reassembled and checked by the pipeline admission every
   measured program passes. A refused program is not measured, and its reason is recorded.
5. A program already measured is not measured again. Otherwise the start guard of refinement
   (cancellation and the seconds bound) is checked, and the program is measured like any other:
   one evaluation from the cut's shared allowance, with origin `simplify`.

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
  saturation's stop (`saturated`, `iteration_limit`, `node_limit` or `malformed`), its rounds,
  e-nodes, rule applications and peak e-nodes, and the admitted expression or the profile's
  refusal.
- Outcomes: `unchanged` (nothing smaller), `refused` (no stage change survived the profile, or the
  pipeline admission refused the program), `duplicate` (already measured), `not_started` (the
  start guard refused the measurement; the run then stops on its seconds bound) and `measured`.
- `search.engine_selection.moves`, in the receipt and in `search/cut-0/search.json`:
  `{"ids": ["feat-math.rewrite-saturate.v1"], "sha256": "<sha256 of moves.json>"}`. The selection's
  own record has no digest; publication adds it, and receipt validation refuses a published record
  without a well-formed one. The `ids` list leaves room for a later move, run in its order.
- The v4 receipt schema gains the record as an optional field; its id is unchanged.

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives it:

- the record names the known id once, and is canonical;
- `moves.json` exists exactly when the record does, and matches the record's digest; without the
  record no measurement has the `simplify` origin;
- every record re-derives from its parent's published program alone: the stage entries, the
  program and the reason are recomputed and compared;
- each outcome re-derives from the published measurements: a measured record's index holds a
  `simplify` measurement of its program in its round, after its parent and before any other
  measurement of that program; a duplicate names an earlier measurement; an unstarted record needs
  a seconds stop; and every `simplify` measurement is accounted for by exactly one record;
- rounds are in order, within the search's rounds, with at most `top_k` records each.

Tests rehash `search.json` and the receipt after editing a record's program, stage expression,
stop, counters, refusal, outcome, measurement index, round, parent or id, or removing a record, and
after editing the selection record's id list or digest or dropping it, and `verify_run` refuses
every edit, as it refuses an edited or missing `moves.json`.

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
- a refused output is discarded with the profile's reason, and a refused stage beside an admitted
  one keeps its expression;
- the allowance, the start guard and `top_k` bound the pass;
- proposals are identical with a poisoned holdout, on fixed parents;
- frozen-clock replay of the receipt and every file with the flag, alone (four parents per
  round) and with the damped least-squares refinement (one parent per round) and the front, with
  `verify_run` accepting each run and every measured rewrite scoring as its parent;
- tamper refusals, the record, the flag, and the shipped-text lint (of this note too);
- with the flag off: the seeded pins, no file and no record.

Not done: the three-language emitted-winner check of the rewritten programs. They are ordinary
pipelines over pool elements, so emission is unchanged.
