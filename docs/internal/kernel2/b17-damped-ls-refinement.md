# Damped least-squares constant refinement

The kernel2 profile route refines the literals of its best programs' expression stages with the
coordinate search (`--refine-top-k`, `--refine-trials`). This adds a second method, opt-in:
damped least squares (Levenberg-Marquardt with Marquardt scaling) on a `derive` stage's literals,
kept only when the whole pipeline's training loss strictly falls. Its stable id is
`feat-math.constant-refinement.damped-ls.v1`. It is engine-internal: no operator, wire, program or
model id changes, no emitter changes, and a refined program is an ordinary pipeline that every
existing runtime and emitter executes.

## Switch

- CLI: `jope pipeline run --search-profile <kernel2 profile> --refine-method lm`. `coordinate` is
  the default and the only other value. Any other value is refused before anything is written.
- `--refine-method` is accepted only with a kernel2 `--search-profile`. The legacy profile refuses
  it, as it refuses the other `--refine-*` flags.
- `lm` needs `--refine-top-k` of at least 1 and `--refine-trials` of at least 3
  (`DAMPED_LS_MIN_TRIALS`). That is the selection's floor, not the method's minimum: a stage is
  fitted only when its remaining allowance reaches its surrogate minimum (see Budget units).
- API: `Kernel2Evaluator::with_engine(EngineSelection { refine: RefineMethod::Lm(options) })`, or
  `with_engine` on the coordinator's `search::Options` and on `run::Options`, which re-run their
  checks. `LmOptions::default()` is τ = 1e-3 (the initial damping relative to the largest scaled
  curvature) with the `sites` Jacobian charge, and it is the only selection the CLI makes. The
  feat-math entry is `MathExpr::optimize_constants_lm_guarded`.

## How a stage is refined

The dual-number Jacobian exists for a stage's expression, not for the whole pipeline. The method
therefore fits a surrogate and lets the native pipeline objective decide:

1. **Start pass.** The coordinate search's own first evaluation of the stage: training prefix,
   admission, guard question, whole-pipeline training objective and charge. A start that fails
   ends the stage exactly as it does on the coordinate route.
2. **Surrogate.** Damped least squares on the stage's expression, with the coordinate search's
   linear-scaled residuals, over the training-prefix rows (the stage's inputs as the fitted prefix
   produces them) and the training targets. It asks the same guard before every pass and stays
   within the stage's allowance less the two pipeline passes.
3. **Native rescore.** When the surrogate changed the expression, the winner's whole-pipeline
   training pass runs, and the winner replaces the start only on a strictly lower loss. A tie, a
   higher loss, a refused pass or a stop keeps the start. The surrogate's own re-score never
   decides.

The refinement receives the training rows and targets and nothing else. A test poisons the
held-out features with ±1e100 and the held-out labels with NaN and infinities, and the refinement,
its counts and its trace stay identical.

## Budget units

Every pass is one objective trial against the refinement allowance (`--refine-trials` per program,
within the run's evaluation budget):

| Pass | Charge |
|---|---|
| The start's and the winner's whole-pipeline passes | 1 each |
| The surrogate's initial value | 1 |
| Each surrogate trial step | 1 |
| Each Jacobian | the literal sites (`sites`, the default), or 1 (`one`, API only) |
| The surrogate's re-score of its winner | 1 |

A stage's surrogate minimum is the fewest evaluations with which it can change the stage: the two
pipeline passes, the initial value, one Jacobian, one trial and the re-score. Under `sites` that is
the stage's literal sites plus five. A stage spends at most its allowance, so the coordinator's
`budget.used() == observations + objective_trials` identity and `evaluations <= trials_per_program`
both hold with the method selected.

**Equal evaluations are not equal compute.** A surrogate pass evaluates one expression on the
training-prefix rows; a pipeline pass refits and scores the whole pipeline. Any comparison at an
equal evaluation budget must report wall seconds next to the evaluation counts, together with the
ablation harness's seconds-bound refusal. This note quotes no timing.

## Fallbacks

These stages use the coordinate route's stage refinement unchanged, bit for bit, and the trace
records a `coordinate` stage with the reason and the coordinate counts:

| Stage | Recorded reason |
|---|---|
| A `map` (elementwise) stage | `elementwise stage` |
| A `derive` stage outside 1 to 14 literal sites (`LM_MAX_SITES`: the solver's 16 parameters less the two linear-scale ones) | `unsupported expression shape` |
| A `derive` stage whose remaining allowance is under its surrogate minimum | `allowance under the surrogate minimum` |

With `--refine-trials` under 6, no stage is fitted by the method, and a CLI test pins that such a
run equals the coordinate run except for the trace and the selection record. In a program with
both kinds of stage, the `map` stage falls back and the `derive` stage uses the surrogate.

Two more records end a stage early: `failed` (the start's pass, or its training prefix, was
refused: one refused evaluation, as the coordinate route charges) and `stopped` (the guard refused
the start's pass, and nothing was spent).

## Receipt, trace and verification

- `search.engine_selection`, in the receipt and in `search/cut-0/search.json`:
  `{"refine": {"id": "feat-math.constant-refinement.damped-ls.v1", "jacobian": "sites", "tau": 0.001}}`.
  It is absent for the default selection.
- Each refinement in `search/cut-0/refinements.json` that spent a trial carries
  `refinement.trace`: the method id and one record per refined stage, in stage order, of kind
  `coordinate`, `failed`, `stopped` or `surrogate`. A `surrogate` record holds:
  - `surrogate`: the status, whether the expression changed, the literal sites and Jacobian cost,
    the counts of initial values, trials, Jacobians, re-scores, refusals, failed solves and
    accepted steps, the winner's re-score, the rows where the expression was non-smooth or
    saturated, and each iteration's damping, objective and trial value with its accepted and
    solved flags;
  - `pipeline`: the passes (1, or 2 when the winner's ran), the refusals, the start's and the
    winner's training losses, and whether the winner was accepted.

  Every floating-point value of a trace is stored as its f64 bits, so a replay compares them
  exactly.
- The v4 receipt schema (`schemas/jope.pipeline-receipt.v4.schema.json`) gains these as optional
  fields; its id is unchanged.

`jope pipeline verify-receipt --evidence` (`publication::verify_run`) re-derives all of it from the
published evidence:

- the selection names a known method and charge with a finite positive τ, is canonical (the
  selection it names writes the same record back), and the refinement allowance meets the
  selection's floor;
- a refinement carries a trace exactly when the method was selected and the refinement spent a
  trial;
- each stage's charge is re-derived from its counts, and the counts from the iterations (accepted
  steps against Jacobians, refusals, failed solves and the re-score); the pipeline decision is
  re-derived from the two loss bit patterns, accepted exactly when the winner's loss is strictly
  lower;
- the records are bound to the parent program. They are its constant-bearing expression stages, in
  order. A `map` stage appears only as an elementwise fallback. An unsupported-shape fallback lies
  outside the site range, and a small-allowance fallback inside it and under its minimum. A
  surrogate, failed or stopped record is a supported `derive` stage whose allowance could hold the
  minimum. A surrogate record reports the stage's literal sites and spends at most its allowance
  less the pipeline passes. A record that ends the refinement is the last;
- the stages' charges sum to the refinement's evaluations and refused trials.

Tests rehash `search.json` and the receipt after editing each of these counts, decisions and
records, and `verify_run` refuses every edit.

## Default behaviour is byte-identical

Without the flag, or with `--refine-method coordinate`, the selection is the default: no selection
record and no trace is written (both are skipped when absent). The coordinate route's only edits
are the absent `trace` field and the visibility of its stage refinement, which the method reuses.
The dispatch to the method is the first statement of the guarded refinement and is not taken by
default. Tests pin:

- the receipt and all seven published files byte-identical with `--refine-method coordinate` and
  without the flag;
- `prepared.json`, `search.json`, `observations.json` and `refinements.json` of two default runs
  equal, by sha256, to the files measured on the parent commit of the change;
- the default route's refinement results (40 rows of program, counts and loss bits) and its
  refined children with their measurements, equal to the values measured on that parent commit;
- the existing coordinate-search tests (`kernel2_refinement_control`, jope-pipeline's kernel2
  tests and the frozen-clock sidecar replay), unchanged and passing.

## Evaluator identity

No evaluator identity changes. The method lives in jope-pipeline's kernel2 route and in feat-math's
`kernel/` and `math_expr/` subdirectories. The ADR-0004 implementation hash
(`compute_implementation_sha256` in `crates/jope-campaigns/src/evaluation_v2.rs`) covers its own
sources, named top-level feat-math source files, the feat-math `Cargo.toml` and the workspace
`Cargo.lock`. This change touches none of them and adds no dependency, so there is no identity
transition and no rebaseline.

## Not accuracy evidence

This is wiring and verification. It makes no accuracy, R², held-out or quality claim, and none may
be made for it until the equal-budget paired ablation has run: kernel2 with and without the method
on the same datasets, folds, cuts, seeds and evaluation budget, reported as paired W/T/L (wins, ties
and losses) on M1, overall and per dataset, beside both medians, with wall seconds. The ablation
harness has the `kernel2-baseline` and `kernel2-experimental` profiles (PR-0, #221;
`ablation-protocol.md`) but no arm that selects this method, and the paired run has not been made.
The tests fit known curves to show that the method runs, is accounted for and replays, not that it
helps. Nor may anything cite the B10 parent-selection change as an accuracy improvement.

## Tests and evidence

- feat-math lib: the public entry against the core fitter, its refusals, charges and replay.
- jope-pipeline lib: `kernel2::refinement_lm_tests` (default bit-identity, strict-decrease
  acceptance, fallbacks, held-out poison, budget accounting, trace binding and tampering) and
  `kernel2::cli_tests` (frozen-clock replay with the method, flag checks, rehashed-evidence
  refusals, default sidecars). `the_lm_refined_fixture_program_is_pinned` pins the refined program
  of the sine fixture.
- feat-math `kernel2_refinement_control`, module `emitted_winners`: that pinned program, bound to
  the jope-pipeline pin, through the `parity_bits.rs` lanes (runtime against emitted Rust, Python
  and fitted C++17) and the recipe C++17 driver of `mathexpr_boundary_parity.rs`, every prediction
  bit for bit, with `JOPE_PARITY_STRICT=1`.
- jope-cli `pipeline_cli`: `kernel2_lm_refinement_records_its_selection_and_verifies_evidence`, a
  seeded `--refine-method lm` run with a surrogate stage, then `verify-receipt --evidence`.
- Evidence: `~/jope-evidence/kernel2/bt/B17_CONSTANTS_LM/`.
