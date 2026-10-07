# Loss and cost front of measured programs

The kernel2 profile route can publish, opt-in, the loss × cost front of every program it measured:
the programs that no other measured program dominates, that is, none is at least as good on both
the validation loss and a recorded program cost and strictly better on one. A program of the same
cost with a lower loss removes a member, as does a simpler one with a loss no higher. Of programs
with equal loss and equal cost one is kept: the one with fewer faulted cells, then the lower key
(see "What is archived"). Its stable id is `feat-math.front-loss-cost.v1`, built on feat-math's
`kernel::pareto` archive. It is evidence only: ranking, parents and the selected program never
read it, so with the archive selected every measurement, every refinement and the emitted program
are unchanged.

## Switch

- CLI: `jope pipeline run --search-profile <kernel2 profile> --pareto-archive`. The legacy profile
  refuses the flag, as it refuses the `--refine-*` flags. It combines with `--refine-method lm`.
- API: `EngineSelection { archive: Some(ArchivePolicy::ArchiveOnly), .. }` through `with_engine`
  on `Kernel2Evaluator`, the coordinator's `search::Options` or `run::Options`. Only publication
  builds the front (`run::run`, through `publication::publish`). `search::run_cut` with the
  selection records it in its result's `engine` and builds and writes nothing; its search output
  is the same with or without it (a test pins this). A caller that runs cuts itself and wants the
  front builds it with `archive::build` from the result's observations.
- `ArchiveOnly` is the only policy. A mode in which the front chooses the next parents is a
  separate, later opt-in with its own ablation.

## What is archived

The archive is built after the search from the run's observations. Its inputs are exactly the
observations ranking admits (measured, a finite selection score, no Hard fault), and each enters
as:

| Field | Value |
|---|---|
| loss | The negated selection score: the held-out R² on the cut's validation rows (a fold of the training rows) after the configured fault penalty, as ranking uses it |
| complexity | The program cost under `feat-math.cost.pipeline-nodes.v1` (see "Program cost") |
| faulted cells | The observation's domain and saturation cells |
| Hard flag | The observation's Hard flag (always false after the filter) |
| key | The sha256 of the program's exact codec bytes (`kernel::Program::import_legacy`, then `to_bytes`). The codec keeps an expression's text as written, so `1*c0` and `1 * c0` are two keys; it keeps a number parameter as its value, so `lambda_frac=1e-9` and `lambda_frac=0.000000001` share one |
| age | None |

The front keeps a program when its loss is strictly lower than that of every simpler one, so it
holds at most one program per cost. Equal (loss, cost) programs are settled by fewer faulted cells,
then the lower key. The result depends only on the set of admitted observations, never on their
order, and a program measured twice is archived once.

**Program cost** (`feat-math.cost.pipeline-nodes.v1`): the program's stages, the terminal head
included, plus every expression node (literal, variable, operator, call and negation) of each
`map` or `derive` stage's `expr`. For example `(pipe X (derive expr="sin(0.5*c0)") (ridge))` costs
2 + 4 = 6. The archive records its cost version, and the reader accepts this version and the
module's original `feat-math.cost.expression-nodes.v1` (still the default of an archive built
without a version) and refuses any other.

**The selected program.** Without the fault tie-break, ranking's winner has the lowest loss, so
the front holds the winner, or holds its loss at a simpler program, or at the same cost with fewer
faulted cells or a lower key. With `--fault-tie-break true` the winner can be a near tie with fewer
faulted cells, which a program with a loss lower by at most the tie tolerance dominates.

## Publication and verification

- `search/cut-0/archive.json`: the archive's own canonical JSON (`Archive::to_json`): schema,
  cost version, and the members in front order, with each loss as its f64 bits and each key in
  hex, so a round trip is exact.
- `search.engine_selection.archive`, in the receipt and in `search/cut-0/search.json`:
  `{"id": "feat-math.front-loss-cost.v1", "cost_version": "feat-math.cost.pipeline-nodes.v1",
  "sha256": "<sha256 of archive.json>"}`. The selection's own record has no digest; publication adds
  it, and receipt validation refuses a published record without a well-formed one.
- The v4 receipt schema gains the record as an optional field; its id is unchanged.

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives it:

- the record names the known id and cost version, and is canonical;
- `archive.json` exists exactly when the record does, and matches the record's digest;
- it reads under the archive reader's own refusals (schema, cost version, NaN or `-0.0` loss,
  front order, a dominated member) with the record's cost version;
- it is byte for byte the archive rebuilt from the re-derived observations. The published
  observations are untrusted, so the rebuild parses each program within the B17 trace check's
  bounds (`parse_untrusted_pipeline`): the pipeline form only, and an expression source holding
  more than 256 of the bytes `( + - * / ^` is refused before the expression parser sees it. A
  seed keeps its expression text as written, and unary plus and redundant parentheses never reach
  the tree, so an admitted seed can exceed that bound; with the archive selected such a seed is
  refused by the search's option check before any cut runs, not at publication.

Tests rehash `search.json` and the receipt after editing a member's loss, key or cost, the cost
version, the member set, the record's id, version and digest, and the record's or the file's
presence, and `verify_run` refuses every edit, each by the check named for it.

## Default behaviour is byte-identical

Without the flag nothing is written and no record is added (the field is skipped when absent), so
the receipt and every published file keep their bytes; the existing pinned default sidecars
(`default_runs_keep_the_pre_wiring_search_sidecars`) and the frozen-clock replay are unchanged.
With the flag, a test pins that `prepared.json`, `observations.json`, `refinements.json` and the
emitted files are byte-identical to the run without it, and that the summary and the receipt differ
only by the archive record and the summary's digest, with and without `--refine-method lm`.

## Evaluator identity

No evaluator identity changes. The archive lives in jope-pipeline's kernel2 route and in feat-math's
`kernel/` subdirectory, which the ADR-0004 implementation hash (`compute_implementation_sha256` in
`crates/jope-campaigns/src/evaluation_v2.rs`) does not cover. No dependency is added.

## Not accuracy evidence

The archive cannot change the selected program, so it cannot change M1: an equal-budget ablation
of the archive alone must show A = B. The ablation harness runs kernel2 searches
(`ablation-protocol.md`), but this change adds no archive arm and reports no such result. Any
lift claim needs a later mode in which the front chooses parents, measured with the paired
protocol of `b17-damped-ls-refinement.md`.

## Tests

- feat-math lib, `kernel::pareto`: the pipeline cost version round-trips, and unknown versions are
  refused by the constructor and the reader; the original tests are unchanged.
- jope-pipeline lib, `kernel2::archive_tests`: members non-dominated and re-derived under the
  recorded cost version, the winner's place on the front, input order and ties, the input filter
  equal to ranking's, the cost definition, bit-identity with the archive on and off (through
  `run::run`, and through `search::run_cut`, which builds nothing), frozen-clock replay, round trip
  and tamper refusals by named check, the record and the flag (with `--refine-method lm` run and
  verified, and the legacy refusal through the CLI's own dispatcher, before it writes anything),
  and the shipped-text lint (of this note too).
