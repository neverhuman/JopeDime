# WP-1.5: fault policies for the library search (kernel2 PR 08)

Base: `d1a1d21f820d5b1867bb55cb90c42e18cd604a2d`, `origin/main` after PR 06 (#132) merged. The
series was prepared on `dd5ccf3f19c3314f5ef9bf34ccf362422650cce7`, `origin/main` after PR 05
(#131), and replayed onto this base before its first push ("History"). PR 01 to PR 06 are #126 to
#128 and #130 to #132, and #129 is the islands-v2 source import. PR 07 (fitted-C++ revision 3) is
prepared elsewhere and is not on this base; nothing here depends on it or on #132. This pull
request is #133: branch `kernel2/claude-m1-admission`, in the lane `.agent/worktrees/kernel2-next`.
Measured on 2026-09-29 on xbabe2: x86_64-unknown-linux-gnu, glibc 2.39, AMD Ryzen Threadripper PRO
3995WX, rustc 1.97.1 (8bab26f4f 2026-07-14), cargo-nextest 0.9.133, jq 1.7, g++ 13.3.0, clang++
18.1.3, node v26.1.0, Python 3.12.3 (for the emitted Python program only). Environment:
`CARGO_INCREMENTAL=0`, `CARGO_PROFILE_DEV_DEBUG=line-tables-only`,
`TMPDIR=~/.cache/jope-test-tmp/kernel2-pr08` and `nice -n 19 ionice -c3`, with
`JOPE_COMPAT_FREEZE_REGENERATE` and `JOPE_COMPAT_FREEZE_SHA` unset except where a step sets them;
test runs add the rust-test lane's `JOPE_PARITY_STRICT=1` and one OpenMP, rayon, OpenBLAS and MKL
thread. This is local evidence, not hosted CI, and the wall times are for this host only. The
evidence is in two directories, each with `SHA256SUMS`:
`~/jope-evidence/kernel2/pr08/prepare-20260929T202059Z/` holds the implementation and its first
gate, and `~/jope-evidence/kernel2/pr08/fix-20260929T220014Z/` the review round, the replay and
every command under "Commands and results".

## What changed

| Commit | Change |
|---|---|
| `a8bb7242c` | `crates/feat-math/src/kernel/admission.rs` (new), registered by one line in `kernel/mod.rs`: `FaultTotals`, `Rejection`, `TIE_TOLERANCE`, `prefer_fewer_faults` and `outranks`, the arithmetic of the policies, with 8 unit tests. |
| `e1ef18c6f` | `crates/feat-math/src/evolve.rs`: `FaultPolicy`, the configuration's `fault_policy` field (`Legacy` by default), four `GenStats` counters, `Observer::on_reject`, and the policies' private twins of the scorers, the rung evaluator, the ranking and the parent selection, which the two search loops dispatch to; 4 unit tests, and the new field in the file's four full configuration literals. `kernel/scope.rs`: its module doc now says who calls `validate_scope`. |
| `63bf54bd9` | `crates/feat-math/tests/search_admission.rs` (new): 14 tests, and 1 ignored informational run. |
| `3e823f5c2` | `crates/jope-campaigns/known-evaluator-identities.json`: row 2, the identity of the base (#128's), with its receipt in `docs/internal/kernel2/receipts/evaluator-identity-3ec654cb/`. |
| `e4f515c65` | `FaultPolicy`'s documentation: what the non-legacy policies change, and how the tie-break counts (review F4 and F3). |
| `57033f41e` | Tests: the scope check sees the program as the search fits it, in `evolve.rs` and a new integration test (review F5). |
| `56cf2f301` | Tests: the values of the domain-rate counters, in `evolve.rs` and a new integration test (review F1). |
| `18d177808` | The informational run shows `Legacy`'s own columns as not measured and measures every run's invalid candidates after it (review F2 and F6). |
| `605cc5ae3` | `crates/feat-math/tests/fixtures/compat_freeze/evaluator_identity.v1.json`, regenerated once by the identity-only mode: the one fixture that changes. |
| last | These records: `deploy/ci/coverage.json`, both test inventories, `CHANGELOG.md`, `MASTER_PLAN.md` and this receipt. |

Every commit has Jepson Taylor `<130782313+jepsontaylor@users.noreply.github.com>` as author and
committer.

The public API changes by additions only. `feat_math::evolve` gains `FaultPolicy` (`Legacy`,
`RejectHard`, `Penalize { lambda_domain, lambda_saturation, tie_break }`; `Clone`, `Copy`,
`Debug`, `Default`, `PartialEq`, as the brief specifies) and `FaultPolicy::validate`; the field
`EvolveConfig::fault_policy`; the fields `GenStats::{hard_rejected_share, hard_rejected,
pop_domain_rate, elite_domain_rate}`; and `Observer::on_reject`, a default no-op, so every existing
observer compiles unchanged. `feat_math::kernel::admission` is new: `FaultTotals` and `Rejection`
are `#[non_exhaustive]`. No enum gains a variant, no public signature changes, no item is removed,
the crate root re-exports nothing new, and there is no new dependency. Outside `evolve.rs`, every
`EvolveConfig` literal ends with `..EvolveConfig::default()` (`compat_freeze.rs`,
`examples/evolve_smoke.rs` and jope-bench's `methods.rs`), nothing builds a `GenStats`, and nothing
in `deploy/lambda` names either type. Nothing that ships sets a policy other than `Legacy`.

## History

Nothing was pushed before the series took its present form, and every earlier head was local:

- **First round,** ending at `ea502766a` (five commits, the same first commit `c16ff2a62`). While
  its gate ran, a re-read of the race loop found that the elites' rung-2 re-check can reject every
  candidate still ranked, after which the loop indexed an empty ranking (decision 10). The later
  commits were redone on `c16ff2a62`, the policy commit with the check and its test. The first
  round's identity (`7189f71e…`) is on no commit of the branch and in no table.
- **Second round,** ending at `0d2983b6e` (the policy commit, the first version of the test file and
  the known-table row). The first eight perturbations below ran there, and one was not caught: no
  test failed when the held-out passes were not counted, because a first-stage expression faults on
  the same rows in the training and held-out passes of two folds, so the rates agree. The test and
  row commits were redone: the formula test now uses a program whose expression reads a fitted
  stage, so its two passes fault at different rates, and asserts that they do. That perturbation
  ran again at `1e1d3817d` and is caught.
- **The reviewed head,** `97b6a17303d6a351e66367591e687f164ee2033d` on `dd5ccf3f1`: `c16ff2a62`,
  `32977f7f4`, `c76f27c1f`, `1e1d3817d`, `a29bc13d8` (the identity fixture, minting `1ca042e9…`)
  and `97b6a1730` (the records; `f04cd1fc8` amended once to take the literal placeholder token out
  of descriptive text). Three reviews read it, for correctness, for regressions and for the rules;
  "Review round" lists their findings and what became of each.
- **The replay.** #132 merged at 21:46:27Z while the reviews ran, so the series was replayed onto
  its merge, `d1a1d21f8`, before its first push, as #131 was replayed onto #130 and #132 onto #131.
  On the branch, `git reset --hard 1e1d3817d` set aside the identity-fixture and records commits,
  which the series regenerates at its end, and `git rebase --onto d1a1d21f8 dd5ccf3f1` picked the
  other four without a conflict: #132's `pub mod width;` and this series' `pub mod admission;` sit
  at different lines of `kernel/mod.rs`. The row commit was then amended, so that its `builds`
  range and its message end at `d1a1d21f8`, and the policy commit's message was reworded by a
  `git rebase -i` with a scripted todo and message edit, because its last paragraph placed the
  identity "two commits on". `git range-diff dd5ccf3f1..1e1d3817d d1a1d21f8..3e823f5c2` shows
  `c16ff2a62` → `a8bb7242c` and `c76f27c1f` → `63bf54bd9` unchanged (`=`), `32977f7f4` →
  `e1ef18c6f` with that paragraph only (`!`), and `1e1d3817d` → `3e823f5c2` with the `builds` value
  and the message (`!`). At `3e823f5c2` the series' source and test files are byte-identical to the
  reviewed head's, and every other difference under `crates/` is #132's. Every replayed commit
  keeps its author and author date.
- **Where the earlier heads are.** None of `ea502766a`, `0d2983b6e`, `f04cd1fc8` and the reviewed
  `97b6a1730` is on a ref, and no ref was made for them. The lane's reflogs still reach them, but
  reflog entries expire and are deleted with the lane, and git may then prune the commits. What
  they held is kept in the evidence instead, and this receipt cites that: each series as patches
  (`00-pre-replay/patches/`: `round1-ea502766a`, `round2-0d2983b6e`, `reviewed-97b6a1730` and
  `records-f04cd1fc8`), which, applied to their bases in a scratch index, rebuild each head's tree
  exactly (`00-pre-replay/patch-verify.txt`); their commit and tree ids (`00-pre-replay/state.txt`,
  and the implementation's `30-pre-rewrite.txt`); the branch reflog before the replay
  (`00-pre-replay/branch-reflog.txt`); and what the second round ran (the implementation's
  `40-perturb/`: each perturbation's diff, the original file and the log).

## Review round

The reviews read the reviewed head `97b6a1730`. Their findings, and what was done:

| Finding | Severity | Disposition |
|---|---|---|
| F1: no test checked the values of the domain-rate counters, so an elite rate stuck at 0, the wrong elite slice or the wrong rung's totals passed | should-fix | Fixed in `56cf2f301`; perturbations F1-a to F1-c below are caught |
| F2: the informational table gave `Legacy`'s rejections and rejected offspring as 0, as if measured, and scoped no conclusion to the rejecting policies | should-fix | Fixed in `18d177808`: `Legacy`'s own columns are `–`, a new column measures every run's invalid candidates alike, and "The informational run" below uses it |
| F3: the tie-break compares raw counts from evaluations of different sizes | nit | Kept as the brief words it ("the lower fault count") and documented: `e4f515c65` on `FaultPolicy::Penalize`, decision 9, and "Still open" |
| F4: `FaultPolicy`'s documentation said the other policies draw the same random numbers | nit | Fixed in `e4f515c65` |
| F5: no test showed that the program as fitted, not as generated, is checked | nit | Fixed in `57033f41e`; perturbation F5 below is caught |
| F6: the receipt called `datasets/test` "the D5 fixtures" | nit | Fixed: the informational test's documentation (`18d177808`) and "Data" below name the real-data half of D5 |
| NR-1: three existing function bodies change, where the lens asks for none | nit, no change needed | Recorded in decision 2 |
| R1: PR 06 was open as #132, and the records said it was prepared elsewhere | should-fix | #132 merged during the review; the series was replayed onto its merge before its first push ("History"), and these records are regenerated on it |
| R2: the receipt cited earlier heads as kept by the lane's reflog | nit | Fixed: "History" cites the patches and evidence files |

## Decisions taken while implementing

1. **Where the code lives.** The policy is a field of the search configuration, so the enum, the
   field, the counters, the observer hook and the policies' twins of the scorers, the rung
   evaluator, the ranking and the parent selection are in `evolve.rs`, whose private items they
   need. The arithmetic that needs none of them (the totals, the rates, the penalty, the tie orders
   and the rejection reason) is in the new `kernel/admission.rs`, beside `kernel/scope.rs` and
   `kernel/telemetry.rs`, where its unit tests can build stage counts by hand (`StageFaultCounts`
   is `#[non_exhaustive]`). Nothing is re-exported at the crate root: `lib.rs` is an identity
   input. `model.rs`, `ops.rs`, `transpile.rs`, `grammar.rs`, `fit_boundary.rs` and
   `kernel/telemetry.rs` are untouched.
2. **Legacy runs today's code.** The six functions the brief names (`score_genome`, `score_rung`,
   `eval_rung`, `prep_rung0_candidate`, `rank_race`, `tournament`) keep their bodies byte for
   byte, and so do `promote`, `evaluate_rung_chunk`, the GPU wave and the exact reuse. Three
   existing function bodies change (review NR-1). The two search loops, `evolve_surrogate` and
   `evolve_lite_race_impl`, dispatch at their call sites: the scoring call of the surrogate loop,
   the four rung evaluations (rungs 0, 1 and 2 and the elites' rung-2 re-check) and two rankings of
   the race, and the parent draws of both; the race loop also gains the guarded reseed of decision
   10. `evolve_lite_race_with_exact_initial_population` first returns the policy's validation
   error, which `Legacy` never has. Under `Legacy` each dispatch makes the original call with the
   original arguments, the tie post-pass and the counters are skipped, and the four new `GenStats`
   fields are zero. The one new branch that draws random numbers, the reseed after the elites'
   rung-2 re-check, is unreachable under `Legacy`, whose ranking cannot shrink there: `eval_rung`
   writes only the rung it evaluates and `RaceEntry::best` keeps a valid lower-rung score when rung
   2 fails, so the set `rank_race` returns can only grow during the re-check, and it was not empty
   before it. compat_freeze's frozen searches and generator streams compare equal in compare mode,
   and so does `v1_regression_bit_identical_champion`.
3. **What is checked, and when.** The program that is actually fitted is checked, as the brief
   asks: `search_surrogate`'s ridge program in the surrogate search and at rung 0, and
   `lite_budget`'s program at rungs 1 and 2. Both clamp `degree` and `n`, and so can narrow
   widths: `(rff_cos n=128)` then `(derive expr="col(100)")` is in scope as generated (128
   columns) and out of scope as fitted (64), and tests pin that this program is rejected with the
   fitted program's reason and that nothing is fitted (review F5). The width is the first input
   row's (`x.first()`: an input without rows is left to the folds, which fail on it as the legacy
   scorers do, instead of panicking on `x[0]`). Every failure of `validate_scope` rejects,
   including the ones without a class (an expression that does not parse, a stage whose width only
   the table-v3 path computes): such a program cannot be admitted either way, and the reason names
   it. The check runs before the first fold, so a rejected program is never fitted.
4. **Both passes of every fold are traced.** Each fold is fitted with `fit_traced`, and its
   held-out rows are predicted with `predict_traced`, whose bits equal `predict`'s. The held-out
   pass is what is scored, so a fault there is scored too; tracing only the training pass would
   score a held-out Hard fault as a valid 0.0. The totals pool both passes of every fold.
5. **Rates and the per-severity count.** The brief defines the rates as faulted cells over
   evaluated cells. `StageFaultCounts` now records `domain_cells` and `saturation_cells` once per
   generated cell with a class of that severity. Its original per-class counts remain available.
   A cell raising two Domain classes, such as `log(c0) / c1` with `c0 <= 0` and `c1` near zero,
   contributes one Domain cell. A cell raising two Saturation classes also contributes one
   Saturation cell. Only `map` and `derive` cells count: the input sanitize and stage epilogues are
   left out, so a program without an expression stage has no cells, rates of 0 and legacy scores.
6. **A constant stage.** A stage pass whose output is constant counts every one of its cells as
   Domain-faulted, as the brief says, but only from two cells: one cell is trivially constant, and
   a one-row held-out fold would otherwise penalize a fault-free program.
7. **Penalize rejects Hard faults too.** The workplan's goal is "search rejects Hard faults and
   penalizes Domain faults", and plan decision OQ-4 moves kernel2 profiles from `RejectHard` to
   `Penalize` without re-admitting them: a Hard fault is a structurally invalid program.
8. **The penalty.** It applies to every fold score, `s - lambda_domain * domain_rate -
   lambda_saturation * saturation_rate`, evaluated in that order, with the rates pooled over the
   candidate's evaluation at that rung. Every score derived from the folds then carries it: the
   mean, the rank-robust score (whose fold spread is unchanged up to rounding) and the rung means
   that break `rank_race`'s ties. A term whose rate is 0 is left out, so a fault-free candidate
   keeps its bits for every lambda. The lambdas must be finite and non-negative:
   `FaultPolicy::validate` returns the reason, the search panics with it before any work, and
   `evolve_lite_race_with_exact_initial_population` returns it. A penalized score that is not
   finite fails the evaluation (`None`, not counted as a rejection), as a non-finite metric does.
   The reported fold scores and `cv_mean` are the penalized ones: they are what the search
   optimized.
9. **The tie-break.** `sort_by` needs a total order, and "within 1e-12" is not transitive. The
   ranking is sorted by the unchanged legacy comparator, then cut into runs anchored at their first
   score, and each run is sorted stably by faulted cells; equal counts keep the legacy order,
   rung-mean fallbacks included. Parent selection keeps its three draws and compares pairwise: a
   contender within 1e-12 with a different count wins exactly when its count is lower, and
   otherwise the legacy strict comparison decides. The count is a candidate's Domain and Saturation
   faulted cells, from the evaluation its score comes from (in the race, its highest scored rung).
   It is a count of cells, not a rate (review F3), and evaluations differ in size: rung 2 covers
   768 row passes (256 rows in 3 folds, each row trained on twice and held out once) against 256 at
   rungs 0 and 1 (128 rows in 2 folds), and a `map` over `w` columns generates `w` times the cells
   of a `derive`. Among near-tied candidates, one evaluated on more cells therefore counts more at
   the same rate, and on an exact tie the count can put a candidate scored only at rung 0 or 1
   above one validated at rung 2, which `rank_race`'s comparator puts first. The brief words the
   tie-break as the lower fault count, and a candidate without a fault still wins every tie against
   one with a fault, so the count is kept; `FaultPolicy::Penalize` documents it, and a rate, or the
   count of the rung every candidate has, is listed under "Still open". Promotion between rungs has
   no tie-break: the brief names the ranking, the surrogate ranking and parent selection, and
   promotion already sees the penalized scores. With the tie-break on, the ranking, and so
   `GenStats::pop_fitness`, is in descending score order only up to the tolerance.
10. **Rejections in the race.** A rejection at any rung removes the candidate from the ranking, so
    it is never carried over or selected, even though its earlier rungs scored it (`best()` would
    otherwise fall back to them). Its record in `RaceOutcome::candidates` keeps the scores of the
    rungs it passed and `None` at the rejecting rung. Promotion needs a score at the rung below, so
    a rejected candidate is never promoted, and the elites' rung-2 re-check that rejects one
    re-ranks without it. That re-check can now reject every candidate still ranked, which the
    legacy loop never meets: the ranking then empties after the check for an empty ranking has
    passed. The loop checks again after the re-check and takes the same reseed, and the generation
    ends there; without that check it indexed the empty ranking. This was found while the first
    round's gate ran, before anything was pushed (see "History").
11. **The GPU rung-0 path is skipped, not traced.** `prep_rung0_candidate` fits with `fit_op`,
    `apply_op_train` and `apply_op` directly, without the traced entry points. Under a non-legacy
    policy every rung is scored on the CPU by the traced scorer, as `eval_rung` scores when no
    backend is given (routes `CpuExact`), and a backend passed to the race is never called. No
    production caller passes a backend. A traced prepared path is left for a later change.
12. **Exact rung-0 reuse is off.** A reused score copies a representative's fold scores without
    evaluating the member, so it would carry no fault counts of its own. When a caller requests
    reuse under a non-legacy policy, the receipt says `active: false` and names the policy as the
    `activation_blocker`; its other counters stay 0.
13. **Counters.** `GenStats` gains four fields, zero under `Legacy`: `hard_rejected_share`, the
    share of the generation's distinct candidates that the policy has rejected; `hard_rejected`,
    the distinct candidates rejected since the run started; and `pop_domain_rate` and
    `elite_domain_rate`, the mean domain rate of the ranked candidates and of the elites (the first
    `(pop / 8).clamp(1, 8)` ranked), each candidate's rate taken from the evaluation its score
    comes from. Tests pin their values (review F1): in the race each rate is the one of the
    candidate's ranking rung, bit for bit, and both means follow the final ranking and the elite
    count in both loops. They are aggregated in ranking order, never by iterating a hash map.
    `GenStats` is not serialized, so no receipt schema moves. The reason of each rejection goes to
    the new `Observer::on_reject` (a default no-op), in evaluation order. A rejected candidate is
    not evaluated again in a later generation or at a later rung; only the surrogate search
    evaluates a program twice, when it appears twice in one generation, as it always has.
14. **Emission never reads telemetry.** `transpile::emit` takes a program and `EmitOptions`, and
    neither carries fault telemetry; the loops emit the winner from its canonical text as before.
15. **jope-bench keeps `Legacy`, and gets no flag here.** The lite race's one production caller is
    `jope-bench run --search` (`methods::search_program`), not `jope optimize`, which calls the
    canonical wrappers directly (`panel/optimize.rs`) and never builds the search configuration.
    A `--fault-policy` flag would add a field to `SearchConfig`, which every bench report
    serializes and `tests/committed_evidence.rs` compares as a full literal, and it would need its
    own method id, since `afm.lite-race.v1` must not mix policies. That is not small or additive,
    so it is left for PR 14 (plan E6), beside the CLI switch that the M2 decisions move to PR 17.
16. **The optional `fit_boundary.rs` receipt field is dropped.** `fit_boundary.rs` is a lineage
    source of the teaching store (`lineage_sources.v1.json`), so any edit moves a lineage, and the
    fit boundary is not on this search's path: the library search fits through `model::fit` and
    `fit_traced`.
17. **The stage cache is not involved.** The library search fits through `model::fit`, not
    `fit_with_stage_cache`, so #130's open question about a traced cache path does not arise here.
18. **What the documentation promises.** `FaultPolicy`'s documentation says what the non-legacy
    policies change (review F4): parent selection keeps its draws, but a rejection or a penalty
    changes the ranking, and so the parents and the programs generated from them, and only these
    policies can take the race's reseed after an emptied re-check. A candidate that passes the scope
    check and raises no fault is scored as under `Legacy`.

## Tests

**Unit tests, `kernel/admission.rs` (8).** Each builds stage counts by hand.

| Test | What it pins |
|---|---|
| `stages_without_expression_cells_add_nothing` | a stage that generated no cell (every op but `map` and `derive`) and the input sanitize add nothing; without a fault, `penalize` returns the score's bits for scores `0.0`, `-0.0`, `0.75`, `-3.25` and `1e-300` and lambdas `(0.05, 0)`, `(NaN, inf)` and `(-1, -0)` |
| `rates_pool_every_expression_stage_of_every_pass` | two passes, three stages: 14 cells, 3 Domain, 1 Saturation; the rates are `3/14` and `1/14` exactly |
| `each_severity_counts_a_cell_once_across_its_classes` | two Domain and two Saturation classes on each of 2 cells count 2 and 2, a Domain rate of exactly 1; below the cap, one cell with two Domain classes counts 1 |
| `a_constant_stage_pass_is_domain_faulted_from_two_cells` | constant passes of 5 and 3 cells count all 8 cells, whatever their classes; a constant single cell counts 0 |
| `hard_classes_are_collected_and_named` | a Domain fault alone is not Hard; `COLUMN_OOB` and `INVALID_CLIP_BOUNDS` are collected with their rows and displayed as `hard fault: {INVALID_CLIP_BOUNDS, COLUMN_OOB}` |
| `the_penalty_is_the_documented_formula` | `0.8 - 0.05 * (3/12) - 0.02 * (1/12)` bit for bit; a zero lambda leaves the score; only the Saturation term applies without a Domain fault; an infinite or NaN lambda on a faulted score gives `None` |
| `ties_prefer_fewer_faulted_cells_within_anchored_runs` | runs anchored at their first score (a score 1.5e-12 below the anchor starts a new run although it ties its neighbour), stable within equal counts; an empty list |
| `pairwise_ties_are_decided_by_faulted_cells` | `outranks` within and outside the tolerance, with equal and different counts |

**Unit tests, `evolve.rs` (5).**

| Test | What it pins |
|---|---|
| `scope_rejection_comes_before_any_fit` | `(pipe X (derive expr="col(12)") (ridge))` on 4 columns, under both non-legacy policies, through both scorers and all three rungs, with a fit that panics when called: every result is the scope rejection `stage 0: COLUMN_OOB: column index 12 out of bounds for width 4`, so nothing was fitted. Legacy scores the same program (its fallback zeros). `(pipe X (rff_cos n=128) (derive expr="col(100)") (ridge))`, in scope as generated, is rejected the same way with the fitted program's reason, `stage 1: COLUMN_OOB: column index 100 out of bounds for width 64` (review F5). A program in scope goes through the same fit parameter once per fold (3 and 2 calls), so the check stands on the path of every fit |
| `a_hard_fault_at_a_later_rung_removes_the_program_from_the_ranking` | `clip(col(0), col(1)*1e300*1e300*0, ...)`, whose bounds are NaN only on the rows the rung-2 sample adds (column 1 is 0 on the permutation's first 128 rows and 1 after): under `RejectHard` its rung-0 and rung-1 scores equal Legacy's, rung 2 is `None`, it is not ranked, and the reason is `hard fault: {INVALID_CLIP_BOUNDS}`; Legacy ranks it by its rung-2 score |
| `a_ranking_emptied_by_rung_2_rejections_reseeds` | two such programs in a race of two: one is promoted to rung 2 and rejected there, the other is rejected when the elites' rung-2 re-check reaches it as the only ranked candidate. The generation ends in the reseed an empty ranking takes, and the next generation ranks the reseeded candidates. Without the new check after the re-check, the loop indexed an empty ranking (perturbation 1) |
| `the_domain_rate_counters_read_the_rung_of_the_ranking_score` | on the same data, `log(col(0)*col(0) + 0.5 - col(1))` reaches `log`'s domain guard only on rows that rung 2 adds: its rung-0 evaluation has no faulted cell and its rung-2 rate is between 0 and 1 (0.2890625 here: 74 of the 256 rows). Raced alone under both non-legacy policies it is ranked by its rung-2 score, and both counters equal the rate of the rung-2 evaluation the race makes, bit for bit (review F1) |
| `tie_breaking_selection_draws_the_legacy_numbers` | over 300 seeds, the tie-breaking parent selection leaves the generator in the legacy state and picks the legacy parent when counts are equal; with counts it differs only inside the near-tied group, toward fewer faulted cells, and it does differ |

**Integration tests, `tests/search_admission.rs` (17 run, 1 ignored after the review correction).**

| Test | What it proves |
|---|---|
| `one_cell_with_two_classes_of_a_severity_is_penalized_once` | `log(col(0))/col(1)` on `[0,0]` and `[2,1]` raises two Domain classes in one of two cells; `exp(col(0))` on `[31]` and `[0]` raises two Saturation classes in one of two cells. Each exact rate is 1/2 and its weight is applied once |
| `legacy_reproduces_the_frozen_search_runs` | `Legacy`, set explicitly, reproduces the three frozen searches of `search_runs.v1.json` line for line (ranked, raced and summary lines, in `compat_freeze`'s formats) and reports no rejection and no rate |
| `reject_hard_reproduces_the_frozen_search_runs`, `penalize_reproduces_the_frozen_search_runs`, `other_penalty_settings_reproduce_the_frozen_search_runs` | the same for `RejectHard`, `Penalize{0.05, 0, tie-break}`, `Penalize{0, 0, tie-break}` and `Penalize{0.05, 0.01}` without tie-break. None of those searches has an expression stage, so this is the zero-fault identity end to end: the traced fits, the CPU route, the tie post-pass and the tie-breaking parent draws change no score, rank, route, slot or random draw |
| `an_out_of_scope_candidate_is_rejected_and_never_ranked` | the `col(12)` program at width 4, seeded into the surrogate search (3 generations) and into the race's exact generation 0: rejected once, with its reason, counted (`hard_rejected` 1, share 1/8 and 1/4 in generation 0), never ranked, its race record `[None, None, None]`; Legacy ranks it |
| `the_scope_check_sees_the_program_as_the_search_fits_it` | the `rff_cos n=128` program: in scope as generated at width 4; its rung-0 program (`rung0_trace_identity`) fails with `stage 1: COLUMN_OOB: column index 100 out of bounds for width 64`, and a traced fit of it raises `COLUMN_OOB`, the Hard fault a check of the generated program would have let the search meet. Seeded into the surrogate search and the race's generation 0 under both non-legacy policies, it is rejected once with the scope reason, never ranked, its race record `[None, None, None]` (review F5) |
| `penalize_ranks_the_fault_free_program_first_on_equal_loss` | column 0 in (-2, -1] and a constant target, so R² is 0 for every program and fold: `log(col(0))` (a constant derived column, domain rate 1) and `log1p(abs(col(0)))` tie on loss exactly. In the race and in the surrogate search the fault-free program ranks first under `Penalize{0.05, 0}` with and without tie-break (by the penalty) and under `Penalize{0, 0, tie-break}` (by the tie-break alone), and keeps the legacy order under `Legacy` and `Penalize{0, 0}` without tie-break. The faulting program's rung-0 folds are `0 - 0.05 * 1` exactly |
| `the_domain_rate_counters_follow_the_ranking_and_the_elites` | on the same data, where every program scores 0 and only `log(col(0))` faults (rate 1). The race (two candidates, one elite): the ranked mean is 0.5, and the elite mean 1 when `log(col(0))` ranks first (`RejectHard`, `Penalize{0, 0}` without tie-break) and 0 when the fault-free program does (`Penalize{0.05, 0, tie-break}`, `Penalize{0, 0, tie-break}`); `Legacy` reports 0. The surrogate search (sixteen candidates, two elites): the ranked mean is one over the ranked count, and the elite mean 0.5 while `log(col(0))` is among the first two and 0 once the penalty or the tie-break moves it down (review F1) |
| `fault_free_programs_score_as_under_legacy` | three programs with fault-free `map` and `derive` stages score bit for bit as under `Legacy`, in the race and in the surrogate search, under `RejectHard` and two `Penalize` settings |
| `the_penalty_is_the_documented_formula_on_the_published_rates` | `(standardize)` then `log(col(0)) + exp(col(1)*20)` on `toy_data` (both rates between 0 and 1): rung 0 replayed through the public `fit_traced`, `predict_traced` and `FaultTotals` on the published rung-0 rows and folds gives Legacy's rung-0 scores bit for bit, and the penalized run's rung-0 folds equal `s - 0.05 * dr - 0.02 * sr` on the replayed rates. The expression reads a fitted stage, so the training passes alone give other rates, and the test asserts they do: it fails when the held-out passes are not counted (perturbation 3) |
| `non_legacy_policies_never_call_a_gpu_backend` | a backend that panics when called, with an eligibility policy that would route every fold to it: both non-legacy races finish, equal the race without a backend, and record only `CpuExact` routes and 0 GPU folds |
| `exact_reuse_stays_off_under_a_non_legacy_policy` | the same reuse request turns reuse on under `Legacy`; under both non-legacy policies the receipt is `requested`, not `active`, names the policy as its blocker, counts nothing, and the race equals the race without the request |
| `penalty_weights_must_be_finite_and_non_negative` | six invalid weight pairs are refused with their reason, which the exact-initial-population entry point returns; three valid pairs, `Legacy` and `RejectHard` pass; the default is `Legacy` |
| `the_surrogate_search_refuses_an_invalid_penalty`, `the_race_refuses_an_invalid_penalty` | the search panics with the reason before any work |
| `outcomes_do_not_depend_on_the_thread_count` | a race and a surrogate search seeded with faulting, out-of-scope and fault-free expression programs under `Penalize{0.05, 0.01, tie-break}`, in a one-thread and a four-thread rayon pool: the same records, rejections and per-generation counters (the ranked count included), with at least one rejection and one non-zero domain rate |
| `informational_fault_policy_rates_on_the_test_datasets` (ignored) | the informational run below |

## Perturbations caught

Each was applied to the committed sources, the tests named ran with `--no-fail-fast`, and the file
was restored with `git checkout`; the tree was clean after each. Every caught perturbation exited
100. Perturbations 1 to 8 ran at the second round's `0d2983b6e` over `evolve::tests`,
`kernel::admission` and, from the second, `search_admission` (see "History"), and perturbation 3
again at `1e1d3817d`; the lines they perturb are the same at this head. Perturbations F1-a to F5,
of the code the review's findings F1 and F5 name, ran at `605cc5ae3` over `evolve::tests`,
`kernel::admission` and `search_admission` (`33-perturb/` in the fix evidence).

| Perturbation | Failing tests |
|---|---|
| 1. The check after the elites' rung-2 re-check is removed | `a_ranking_emptied_by_rung_2_rejections_reseeds` (1) |
| 2. The scope check never rejects | `scope_rejection_comes_before_any_fit` (the fit that panics was called), `an_out_of_scope_candidate_is_rejected_and_never_ranked` (2) |
| 3. The held-out passes' counts are dropped | none at `0d2983b6e` (exit 0); at `1e1d3817d`, `the_penalty_is_the_documented_formula_on_the_published_rates` (1) |
| 4. A candidate rejected at a later rung stays in the race's ranking | `a_hard_fault_at_a_later_rung_removes_the_program_from_the_ranking`, `a_ranking_emptied_by_rung_2_rejections_reseeds` (2) |
| 5. The tie-break never applies | `penalize_ranks_the_fault_free_program_first_on_equal_loss` (1) |
| 6. The tie-breaking parent selection draws one more number | `tie_breaking_selection_draws_the_legacy_numbers`, `penalize_reproduces_the_frozen_search_runs`, `other_penalty_settings_reproduce_the_frozen_search_runs` (3) |
| 7. A zero-rate penalty term is applied anyway | `stages_without_expression_cells_add_nothing`, `the_penalty_is_the_documented_formula` (2) |
| 8. A Hard fault is scored (its fallback zeros) instead of rejected | `a_hard_fault_at_a_later_rung_removes_the_program_from_the_ranking`, `a_ranking_emptied_by_rung_2_rejections_reseeds` (2) |
| F1-a. The elites' domain-rate counter is always 0 | `the_domain_rate_counters_read_the_rung_of_the_ranking_score`, `the_domain_rate_counters_follow_the_ranking_and_the_elites` (2) |
| F1-b. The race's counters read each candidate's rung-0 totals, not those of the rung its score comes from | `the_domain_rate_counters_read_the_rung_of_the_ranking_score` (1) |
| F1-c. The elites' mean covers the first ranked candidate only | `the_domain_rate_counters_follow_the_ranking_and_the_elites` (1) |
| F5. The scope check sees the program as generated, not as the search fits it | `scope_rejection_comes_before_any_fit` (the fit that panics was called), `the_scope_check_sees_the_program_as_the_search_fits_it` (2) |

## Evaluator identity (ADR-0004)

`crates/feat-math/src/evolve.rs` is one of the 38 inputs of the ADR-0004 evaluator identity
(`crates/jope-campaigns/src/evaluation_v2.rs`, `compute_implementation_sha256`), so this pull
request mints an identity, the first since #128. None of the other 37 inputs changes, and neither
does any of the 14 lineage sources or the frozen pins; #132 changed none of them either.
`kernel/admission.rs`, `kernel/scope.rs` and `kernel/telemetry.rs`, through which the non-legacy
policies also score, are outside the hash list (see "Still open").

| | Base `d1a1d21f8` | This pull request |
|---|---|---|
| `evolve.rs` sha256 | `2cddde10580bb38f…` | `292b0731f279a74f…` |
| `implementation_sha256` | `3ec654cb6a44f26d5770bcab589273879605831570025cb42475fa1aad1e1f63` | `640eda2ac2d3b30ffb0d091938b8158e42de38b86e947bb405ab81da1f265a84` |
| identity digest | `891dae89b1b56ef4b95c83ef1a1c5e4501f1da8c9c03c8aea1eaef3bb01efd6b` | `a29697ebd410a6914862e8a4e7c229273d68c13b773e1d42188961a252f7db3b` |
| `cargo_lock_sha256` | `8ecdd1fc461f5700…` | unchanged |
| `dsl_abi_sha256` | `c8b4babfa7dc5b65…` | unchanged |
| known identity table | `5a2d18c0f0b8afa1…`, 1 row | `07b98359b4a15f02…`, 2 rows |

- **The known-table row** (`3e823f5c2`) is the base's identity, #128's: implementation
  `3ec654cb…`, cargo lock `8ecdd1fc…`, digest `891dae89…`, builds from `0a9bc2d2c`, the #128
  commit that last changed an identity input, through `d1a1d21f8`. `git diff --quiet 0a9bc2d2c
  d1a1d21f8 --` over the 38 inputs, `Cargo.lock` and `rust-toolchain.toml` exits 0. The runtime
  reads only the first two fields; compat_freeze checks that they are 64 lowercase hex digits.
- **Its receipt,** `docs/internal/kernel2/receipts/evaluator-identity-3ec654cb/receipt.json`
  (sha256 `438b8600a8fba7c67057b618f942bcef3c7940cb40588c0d0623e000dbc98a6e`), is a
  `JopeDimeSemanticEvaluationReceipt.v2` written by the build of `dd5ccf3f1`, with a clean tree
  and before any source change of this series (`cargo build --locked -p jope-dime-regressor`,
  63 s, then `jope-dime-regressor --registry <dir> --dataset fixture verify-evaluation-v2
  --manifest manifest.json`, exit 0). Its inputs are the `manifest.json` and `train.csv` of the
  v0.2.0 receipt, copied beside it byte for byte. Its `run_contract_sha256` commits to the
  evaluator identity digest, and its bytes equal the receipt PR 03's build wrote at its head for
  the same identity. The same build and command at `d1a1d21f8`, with a clean tree, write the same
  bytes (build 57 s, exit 0; `adr/base-d1a1d21f8/` in the fix evidence).
- **The regeneration** ran `JOPE_COMPAT_FREEZE_REGENERATE=identity
  JOPE_COMPAT_FREEZE_SHA=18d177808256e6f1f20187eb9e4900e5727466c7 cargo nextest run --locked -P pr
  -p feat-math --test compat_freeze` once, at `18d177808`, the commit carrying the series' last
  source change, with a clean tree: exit 0, 12 of 12 passed, so every other surface still compared
  equal (20.0 s). The mode also rewrote `lineage_sources.v1.json`, but only its `generated_at_sha`
  (`629e69d86` → `18d177808`); with that field removed, `jq -S` shows the two files equal, and the
  file was restored with `git checkout`. The one fixture that changes against the base is
  `evaluator_identity.v1.json`: its `generated_at_sha`, `evolve.rs`'s sha256, the implementation
  sha256, the digest, the known table's sha256 and its row count.
- **The earlier identities.** The reviewed head minted `1ca042e9…` (digest `17066787…`) and the
  first round `7189f71e…`; the review round's documentation and tests in `evolve.rs` moved the
  former to `640eda2a…`. Neither is in the known table: no build that carries one has shipped.
- **The head's own receipt.** The same build and command at `605cc5ae3`, the sources of the head
  (the records are no build input), on the same `manifest.json` and `train.csv`: build 55 s, exit
  0, sha256 `e2bc68d1e7be0da9…` (`adr/head-605cc5ae3/`). Against the base's receipt, the
  candidate, the fold, the objective, the truth and prediction bits, their validity and the fitness
  bits are equal. The seven leaves that bind the evaluator identity differ: the evaluation key
  (twice), the fit binding, the tape digest, the run contract, the score key and the semantic
  receipt digest. The reviewed head's receipt (`7a2343fc…`, the implementation's
  `adr/head-97b6a1730/`) differs from this one in the same seven leaves and no other. The head's
  identity is not in the known table; as here, the next pull request that changes an identity
  input adds it, with this receipt.

## The informational run

**Informational only: there is no quality gate, and nothing here is a qualification.** The
workplan's acceptance asks for the invalid-offspring and domain rates of `Legacy`, `RejectHard` and
`Penalize{0.05, 0, tie-break}` on the ablation set. PR 02's harness drives only `jope pipeline`'s
search (`run_cuts_with`), which does not use this configuration, so the rates come from a direct
seeded lite race: the ignored test `informational_fault_policy_rates_on_the_test_datasets`, run with
`JOPE_PARITY_STRICT=1 cargo nextest run --locked -P main -p feat-math --test search_admission
--run-ignored ignored-only --no-capture -E
'test(=informational_fault_policy_rates_on_the_test_datasets)'` (one rayon thread; the main
profile, since the run now takes longer than the pr profile's 240-second stop on this host).

- **Data.** The ten regression datasets of `datasets/test` (collection `test-10`): the real-data
  half of fixture set D5, as `docs/internal/kernel2/ablation-protocol.md` names it. The synthetic
  half (`datasets/synthetic/*`) is not in the layout the loader reads and is not used. `train.csv`
  only, each checked against the sha256 its `metadata.json` records; `test.csv` is never read.
- **Search.** `evolve_lite_race` with 24 candidates, 4 generations, seed `0x0DA7A5E7`, at most 3
  transforms, no time budget, `rank_robust` on (as `jope-bench run --search` runs it), and
  generation 0 seeded with 15 expression programs: the 11 of compat_freeze's corpus, `derive x`
  (an illegal scope anywhere), and `log(c0)`, `1/c1` and `sqrt(c0 - c1)`, which the scaled data
  drives into their guards. Without seeds the default vocabulary generates no expression stage, and
  every rate would be 0: the champion, the starters and every generated candidate have none.
- **Columns.** `raced`: distinct candidates evaluated at rung 0. `rejected`: distinct candidates the
  policy rejected, with the reasons by kind. `invalid`: measured after every run alike, the raced
  candidates that rung 0 of a rejecting policy would reject, over the raced candidates, and then
  the offspring among them (first raced after generation 0) over the offspring: the candidate's
  rung-0 program fails the scope check, or a pass of its traced rung-0 evaluation, on the published
  rung-0 rows and folds, raises a Hard class (`rung0_rejection`). `Legacy` scored those candidates
  like any other; `RejectHard` and `Penalize` rejected them. `pop` and `elite`: the last
  generation's mean domain rates (`GenStats`). `best`: the best program's score under the run's own
  policy, and the same program measured afterwards at rung 0 through the public telemetry (its
  expression stages, its scope verdict, and its domain and saturation rates over both passes),
  which is how the `Legacy` runs, which measure nothing, compare. **`Legacy` checks nothing and
  measures no rate, so its `rejected` and domain-rate cells are `–`, not 0** (review F2).

It ran twice at `605cc5ae3`, the sources of the head, under the main profile (exit 0, 1 passed each
time; 249.9 s and 257.7 s of test time; `38-ablation-main-1/` and `39-ablation-main-2/` in the fix
evidence), and its 30 lines are equal apart from the seconds. The measurement after each run
replays every raced candidate's rung 0, and with it the test runs past the pr profile's 240-second
stop on this host: the first run under that profile was stopped there before its last line
(`34-ablation-1/`; its 29 lines equal the main runs' apart from the seconds), and the second was
stopped by hand (`35-ablation-2.note`). Every column that the reviewed head's run also printed is
unchanged, run for run (`40-ablation-compare.txt`), so the searches did not move. The table keeps
every number but the program texts.

| Dataset (features) | Policy | Raced | Rejected (reasons) | Invalid when raced: all, offspring | Domain rate: ranked, elites | Best score | The best program at rung 0 |
|---|---|---|---|---|---|---|---|
| openml-23-cmc (9) | Legacy | 49 | – | 8/49, 7/25 | – | 0.3886 | 3 expression stages, in scope, domain rate 0.0007 |
| | RejectHard | 57 | 1 (illegal scope) | 1/57, 0/33 | 0.0024, 0.0005 | 0.3886 | the same program |
| | Penalize | 49 | 1 (illegal scope) | 1/49, 0/25 | 0.0028, 0.0033 | 0.3885 | 3 expression stages, in scope, rate 0 |
| openml-29-credit-approval (15) | Legacy | 61 | – | 3/61, 2/37 | – | −0.2544 | `derive log(col(0))`, in scope, domain rate 1.0 (a constant column) |
| | RejectHard | 57 | 1 (illegal scope) | 1/57, 0/33 | 0, 0 | −0.2550 | no expression stage |
| | Penalize | 58 | 1 (illegal scope) | 1/58, 0/34 | 0, 0 | −0.2542 | no expression stage |
| pmlb-229-pwlinear (10) | Legacy | 57 | – | 1/57, 0/33 | – | 0.8078 | no expression stage |
| | RejectHard | 58 | 1 (illegal scope) | 1/58, 0/34 | 0.0174, 0 | 0.8078 | the same program |
| | Penalize | 71 | 1 (illegal scope) | 1/71, 0/47 | 0.0132, 0 | 0.8247 | 2 expression stages, in scope, rate 0 |
| pmlb-695-chatfield-4 (12) | Legacy | 56 | – | 2/56, 1/32 | – | 0.8384 | `derive sqrt(col(0) - col(1))`, in scope, domain rate 0.2188 |
| | RejectHard | 47 | 1 (illegal scope) | 1/47, 0/23 | 0.0201, 0.0718 | 0.8419 | 2 expression stages, in scope, domain rate 0.0168 |
| | Penalize | 50 | 1 (illegal scope) | 1/50, 0/26 | 0.0583, 0 | 0.8377 | 1 expression stage, in scope, rate 0 |
| pmlb-first-principles-supernovae-zg (1) | Legacy | 62 | – | 11/62, 0/38 | – | 0.9995 | no expression stage |
| | RejectHard | 72 | 12 (11 column out of bounds, 1 illegal scope) | 12/72, 1/48 | 0, 0 | 0.9993 | no expression stage |
| | Penalize | 68 | 11 (10 column out of bounds, 1 illegal scope) | 11/68, 0/44 | 0, 0 | 0.9995 | no expression stage |
| rdatasets-agridat-bancroft-peanut-uniformity (3) | Legacy | 65 | – | 5/65, 0/41 | – | 0.2959 | 1 expression stage, in scope, rate 0 |
| | RejectHard | 65 | 5 (4 column out of bounds, 1 illegal scope) | 5/65, 0/41 | 0, 0 | 0.2959 | the same program |
| | Penalize | 56 | 5 (4 column out of bounds, 1 illegal scope) | 5/56, 0/32 | 0, 0 | 0.3406 | 2 expression stages, in scope, rate 0 |
| rdatasets-stevedata-chile88 (7) | Legacy | 62 | – | 1/62, 0/38 | – | 0.5025 | 1 expression stage, in scope, rate 0 |
| | RejectHard | 62 | 1 (illegal scope) | 1/62, 0/38 | 0, 0 | 0.5025 | the same program |
| | Penalize | 62 | 1 (illegal scope) | 1/62, 0/38 | 0, 0 | 0.5025 | the same program |
| rdatasets-wooldridge-barium (30) | Legacy | 52 | – | 1/52, 0/28 | – | 0.8248 | 2 expression stages, in scope, rate 0 |
| | RejectHard | 52 | 1 (illegal scope) | 1/52, 0/28 | 0.0567, 0 | 0.8248 | the same program |
| | Penalize | 52 | 1 (illegal scope) | 1/52, 0/28 | 0, 0 | 0.8248 | the same program |
| socrata-cdc-food-safety-culture (39) | Legacy | 67 | – | 1/67, 0/43 | – | −1.0248 | no expression stage |
| | RejectHard | 67 | 1 (illegal scope) | 1/67, 0/43 | 0, 0 | −1.0248 | the same program |
| | Penalize | 67 | 1 (illegal scope) | 1/67, 0/43 | 0, 0 | −1.0248 | the same program |
| uci-parkinsons-telemonitoring (21) | Legacy | 51 | – | 5/51, 4/27 | – | 0.9029 | `derive x`: out of scope (`ILLEGAL_SCOPE`), domain rate 1.0, Hard |
| | RejectHard | 53 | 1 (illegal scope) | 1/53, 0/29 | 0.0021, 0 | 0.9122 | 2 expression stages, in scope, rate 0 |
| | Penalize | 57 | 1 (illegal scope) | 1/57, 0/33 | 0.0023, 0.0002 | 0.9064 | no expression stage |

`Legacy` checks nothing and measures no rate, so its `rejected` and domain-rate cells are `–`. No
saturation rate above 0 reached a best program; the ranked means are in the run's output. The runs
took 1.1 s to 41 s each, not counting the measurements after them.

What the run shows, and only for these seeds, data and one search seed:

- **Every invalid candidate is out of scope.** Measured after the runs, no raced candidate that
  passed the scope check raised a Hard class at rung 0, under any policy. Under `RejectHard` and
  `Penalize` the invalid candidates are exactly the ones the policy rejected, on every dataset and
  among the offspring too, and every rejection is a scope rejection. Seeds account for every
  rejection but one: `derive x` on every dataset, and on the one- and three-feature datasets the
  seeds that read columns 1 to 4.
- **`Legacy` races and scores invalid programs, and breeds from them.** Under `Legacy`, 38 of the
  582 raced candidates were out of scope, and 14 of its 342 offspring: 7 of 25 on openml-23 and 4
  of 27 on uci-parkinsons. `Legacy` keeps an invalid seed in its ranking, since its fallback values
  score like any other, so the seed can be a parent and pass its stages on: its best program on
  uci-parkinsons is the `derive x` seed's stage under another head. Under `RejectHard` 1 offspring
  in 350 was out of scope (supernovae), and under `Penalize` none in 350: they reject the
  invalid seeds in generation 0, and the operators copy expression stages but never create or edit
  them, so an offspring of admitted parents is out of scope only when a stage change narrows what
  reaches its expression.
- **`Legacy` ranks invalid and faulting programs first.** On uci-parkinsons its best program derives
  `x` outside a map: every row takes the error's fallback 0.0, so the appended column is the
  constant 0.0 and gives the head nothing. On openml-29 its best derives `log(col(0))`, a constant
  column on this data (domain rate 1.0), and on chatfield-4 `sqrt(col(0) - col(1))` takes its domain
  guard on 22% of the cells. Under `RejectHard` and `Penalize` those programs are rejected or
  penalized, and the winners' rates are 0 or near it.
- **Scores move little.** The best scores of the three policies differ by at most 0.045
  (bancroft) and mostly by less than 0.01. One seed per dataset cannot rank the policies; that is
  the E8 qualification's job, with the harness and several seeds.

## Commands and results

At `605cc5ae3`, the sources of the head. The steps up to the ADR-0004 receipt ran one after another
with a clean tree before and after each (the fix evidence's `30-src-all.txt`; their logs in `31-src`
to `36-adr-head`). The informational runs under the main profile and the rerun of the consumers'
six failures ran after them (`38-src-more.txt`), the rerun while these records were being written
in the working tree; no step reads the records. The cargo and script steps ran under
`nice -n 19 ionice -c3`, the test runs with one OpenMP, rayon, OpenBLAS and MKL thread and
`JOPE_PARITY_STRICT=1` where the command shows it. Where a step ran twice, both exit codes and wall
times are given.

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| Format | `cargo fmt --all -- --check` | 0 | clean | 6.8 s |
| feat-math clippy | `cargo clippy -p feat-math --all-targets --locked -- -D warnings` | 0 | no findings (the test targets were rebuilt by the same command at `18d177808`, which has the same sources, 8.2 s) | 0.3 s |
| Workspace check | `cargo check --workspace --all-targets --locked` | 0 | warnings only in battle-gpu, jope-dime-regressor and jope-parallel, as before; none in feat-math | 37.6 s |
| feat-math PR tier | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math` | 0, 0 | 561 passed in 26 binaries (532 at the base, 29 new); 11 skipped (7 ignored, 4 by the pr filter) | 80.7 s with the build, 62.9 s |
| compat_freeze, compare mode | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze` | 0, 0 | 12 passed | 9.1 s, 9.1 s |
| The legacy searches' pins | the same, `-E 'test(=seeded_search_runs_match_the_frozen_fixture) \| test(=seeded_generator_streams_match_the_frozen_fixture)'` | 0 | 2 passed | 9.0 s |
| The surrogate search's champion | `cargo nextest run --locked -P pr -p feat-math --lib -E 'test(=evolve::tests::v1_regression_bit_identical_champion)'` | 0 | 1 passed | 4.7 s |
| The new tests, uncaptured | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test search_admission --no-capture` | 0, 0 | 16 passed, 1 ignored | 122.7 s, 123.4 s |
| Named targets | `cargo nextest run --locked -P pr -p feat-math --test compat_freeze --test mathexpr_current_semantics --test fault_trace --test ip_scrub --test legacy_v2_freeze --test portable_parity --test kernel_pipeline --test kernel_expression --test scope_validation --test stage_telemetry --test search_admission` | 0 | 109 passed; 4 skipped | 21.7 s |
| Parity with the emitted programs | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test parity_bits --no-capture` | 0 | 6 passed; rust 6, python 6 and cpp17-fitted 10 divergences, all allowlisted, 0 unexpected, 0 stale | 4.3 s |
| Legacy value digest | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p feat-math --test fault_trace --run-ignored ignored-only --no-capture -E 'test(=legacy_value_digest)'` | 0 | `b64cf538…`, 191,716 results, as PR 03 recorded | 0.8 s |
| jope-bench PR tier | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P pr -p jope-bench` | 0 | 111 passed, 1 skipped | 43.8 s |
| jope-bench PR tier, `asm` | the same with `--features asm` | 0 | 177 passed, 4 skipped | 59.6 s |
| The identity table's tests | `cargo nextest run --locked -P main -p jope-dime-regressor --bin jope-dime-regressor -E 'test(/^evaluation_v2::tests::/) \| test(/^commands::exact_cv_v2::tests::/)'` | 0 | 15 passed | 55.6 s with the build |
| Lambda workspaces | `bash deploy/ci/lambda.sh` | 0 | 6 built and tested with `--locked`, 66 tests passed (40, 2, 5, 2, 3, 14); cargo-deny skipped in all 6 (0.19.8 installed, 0.20.2 asserted), so that step is not a pass | 47.3 s |
| Emitters | `EMITTERS_OUT=<evidence> bash deploy/ci/emitters.sh` | 0 | `8944cbdf`: the Python and Rust programs agree bit for bit on 34 rows (its GBM head has no C++ or JavaScript target); `portable-4lang`: 7 programs agree (Python, Rust, g++ and clang++ at C++17 and C++20, JavaScript); every expectation digest matches | 16.9 s |
| Consumers, main profile | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P main -p jope-parallel -p jope-dime-regressor` | 100 | 1,766 of 1,772 passed, 25 skipped; the 6 failures are not this series': five `bounded_smartcluster` fixtures refuse to run without the workspace's own `scqd`, which this lane had never built (the fixture says to build it first; the full-default CI lane builds it with the workspace), and `failed_signaled_timed_out_and_replaced_custody_are_terminal`, whose fit service has a two-second deadline, failed with "owned child fit/deadline changed" | 239.6 s |
| The same, with `scqd` | `cargo build --locked -p jain-smartcluster-daemon --bin scqd`, then the six failed tests twice with the same command and `-E 'binary(bounded_smartcluster) \| test(=eval_registry::service::tests::failed_signaled_timed_out_and_replaced_custody_are_terminal)'` (`37-heavy-rerun`) | 0; 0, 0 | `scqd` built; the six passed both times, the deadline test in 25.6 s and 11.1 s; the records were in the working tree, and no test of these two packages reads them | 11.6 s; 67.0 s, 11.9 s |
| Perturbations F1-a to F5 | see "Perturbations caught" (`33-perturb`) | 100 each | each caught by the tests written for it; the tree clean after each | about 1 min each |
| The informational run | `JOPE_PARITY_STRICT=1 cargo nextest run --locked -P main -p feat-math --test search_admission --run-ignored ignored-only --no-capture -E 'test(=informational_fault_policy_rates_on_the_test_datasets)'` | 0, 0 | 1 passed; 30 lines, equal apart from the seconds; under `-P pr` it stops at 240 s (`34-ablation-1`) | 250.5 s, 258.3 s |
| ADR-0004 receipt of the head | see "Evaluator identity" (`36-adr-head.out`, `36-adr-compare.txt`) | 0 | sha256 `e2bc68d1…`; 7 identity-bound leaves differ from the base's | 55 s build |

Before these, on the way to this head: the base receipt at `d1a1d21f8` (0; the known row's receipt,
byte for byte; 57 s build), the identity-only regeneration at `18d177808` (0; 12 passed; 20.0 s),
the exported patches of the earlier heads, each rebuilding its head's tree exactly
(`00-pre-replay/patch-verify.txt`), and during the review round's commits, the new tests alone
(`10-dev-f5`, `11-dev-f1`: 3 passed each), feat-math clippy with `-D warnings` and `cargo fmt`
(`12-dev-f2*`).

With these records in the working tree (the sources as at `605cc5ae3`; this receipt and the plan's
record as committed but for their lists of these results; `40-records-apply`, `41-records` and
`42-records-extra` in the fix evidence):

| Step | Command | Exit | Result | Wall time |
|---|---|---|---|---|
| The records re-applied | mapper B's `records-recipes.sh` (`coverage`, `findings`, and `relist feat-math` on `cargo nextest list --locked -P main -p feat-math --message-format json`) on the new main's files | 0 | the coverage item and the findings role equal the reviewed head's; the parity inventory's feat-math section re-listed, and with that section and the Rust counts removed the file equals main's | not timed |
| Coverage mapping | `cargo run -q --locked -p jope-records -- coverage validate` | 0 | 291 targets, 0 unassigned or invalid | 0.8 s |
| Regression test inventory | `cargo run -q --locked -p jope-records -- verify-test-inventory` | 0 | 327 items, 198 of them regression | 1.7 s |
| Coverage unit tests | `cargo test -q --locked -p jope-records --lib -- coverage::` | 0 | passed | 1.3 s |
| jope-records PR tier | `cargo nextest run --locked -P pr -p jope-records` | 0 | 166 passed, 1 skipped; its `records` tests read the live coverage map and inventory | 14.4 s |
| Parity test inventory | `bash deploy/ci/test-inventory.sh --no-rust --static-python`, then with `--update` | 0, 0 | consistent, the Rust half kept at 3,935 tests; `--update` changed no byte | 0.1 s, 0.1 s |
| The re-listed sections | `cargo nextest list --locked -P main -p <package> --message-format json`, written into that package's section with `jq`, for feat-math and for jope-bench | 0, 0 | each reproduces the file byte for byte | 0.6 s, 17.7 s |
| Prose scrub | `bash deploy/ci/ip-scrub.sh` | 0 | 47 files clean | 1.5 s |
| Hygiene | `bash deploy/ci/check.sh` | 0 | every step ok: coverage 291 targets; gitleaks found no leaks (738 commits scanned); the repository layout, 0 findings | 20.7 s |
| Whitespace | `git diff --check d1a1d21f8` | 0 | clean | 0.0 s |
| Number placeholders | `git grep -nI` for the number placeholder in `CHANGELOG.md`, `MASTER_PLAN.md` and `docs/internal/kernel2/receipts` | 0 | the CHANGELOG line, and the plan's record and next action; this receipt, untracked until committed, has one more | 0.0 s |

The feat-math part of the parity inventory goes from 542 to 572 tests (6 to 7 ignored) and the Rust
total from 3,905 to 3,935 (65 to 66 ignored). The library binary goes from 296 to 309 tests, the 13
new unit tests, and loses none; `search_admission` is new with 17 (1 ignored); every other feat-math
binary is listed as before. With these records committed, the records steps, the identity checks
(`git diff` from `d1a1d21f8` over the identity inputs, the lineage sources, the pins, the fixtures
and the Cargo files; the commits' subjects, identities and trailers; the paths added),
`compat_freeze`, the identity table's tests and the head's ADR-0004 receipt ran again at the head
(`60-final-summary.txt`, `61-records` to `66-adr-final.out`).

## Still open

- **Hosted CI.** Every lane runs for this change: `crates/feat-math/**`, `crates/jope-campaigns/**`
  and `deploy/ci/coverage.json` select all nine jobs. The first hosted run is also the cross-host
  check of the searches this pull request compares, which are always two runs of one build, apart
  from the frozen fixture that compat_freeze already compares.
- **Post-review correction.** An independent review found that class incidences can count one
  cell twice within a severity. The failing regression was recorded before the fix (2 observed,
  1 expected); the appended telemetry fields now count each severity once per cell, and admission
  uses those fields. This changed no ADR-0004 hash input. The current hash list omits admission and
  telemetry, so the evaluator identity is unchanged; the identity coverage gap remains for its
  planned ADR amendment.
- **The tie-break's count** (decision 9, review F3). A rate of faulted cells over evaluated cells,
  falling back to the count when the rates are equal, or the count of rung 0, which every raced
  candidate has, would compare evaluations of different sizes evenly. Either changes which
  candidate wins some near-ties, so it needs its own change and review, before the kernel2 profiles
  move to `Penalize` (plan OQ-4).
- **The ADR-0004 amendment** (plan WP-8.3) should list `kernel/admission.rs`, beside
  `kernel/telemetry.rs`, `kernel/scope.rs`, `math_expr/fault.rs` and `math_expr/compiled.rs`: the
  scores of the non-legacy policies depend on all of them, and none is in the hash list. It needs
  Codex's agreement.
- **jope-bench.** A `--fault-policy` flag for `jope-bench run --search` (decision 15), with its own
  method id and a `SearchConfig` field that old reports still read, is left for PR 14.
- **The GPU rung-0 path.** A traced twin of the prepared path would let a non-legacy race use a GPU
  backend (decision 11), and fault totals keyed by the evaluation signature would let it reuse
  exact rung-0 scores (decision 12). Neither has a production caller today.
- **Promotion.** Promotion between rungs has no tie-break (decision 9).
- **The earlier heads.** `ea502766a`, `0d2983b6e`, `f04cd1fc8` and the reviewed `97b6a1730` are on
  no ref ("History"), and their patches are in the evidence. If the commits themselves should stay,
  the orchestrator can pin them under `refs/preserve/kernel2/`, as #131's and #132's pre-replay
  heads were.
- **Local gaps.** cargo-deny was skipped by `lambda.sh` (0.19.8 is installed, 0.20.2 is asserted);
  the hosted hygiene lane runs it. Clippy `-D warnings` on `jope-dime-regressor`, which fails at
  #128's base in files this series does not touch, was not rerun: the series changes no Rust file
  of that package. The informational test now needs the main profile on this host (it passes the
  pr profile's 240-second stop); it is ignored by default and no lane runs it.
- **The pull request number** replaces the placeholder in `CHANGELOG.md`, `MASTER_PLAN.md` and
  this receipt once the pull request is opened.
- **The feat-math merge announcement** in `AGENT_CHAT.md` belongs to the orchestrator.
