//! The opt-in damped least-squares refinement on the guarded route (laneD:B17-lm-refinement).
use super::tests::{data, evaluator, measured};
use super::*;
use feat_math::kernel::expression_search::{LmJacobianCharge, LmOptions};
use std::cell::Cell;

const SINE: &str =
    "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (ridge lambda_frac=0.000000001))";
const MAP: &str =
    "(pipe X (standardize append=false) (map expr=\"sin(0.5*x)\") (ridge lambda_frac=0.000000001))";
const MIXED: &str =
    "(pipe X (map expr=\"sin(0.5*x)\") (derive expr=\"sin(0.7*c0)\" append=false) (ridge))";
const FAULTING: &str = "(pipe X (derive expr=\"sqrt(c0-0.5)\" append=false) (ridge))";

/// The refinement-control grid with a one-frequency target (as `refinement_tests.rs`).
fn sine_data() -> Cut {
    let mut c = data();
    c.train_x = (0..32).map(|i| vec![-1.55 + i as f64 / 10.0]).collect();
    c.train_y = c
        .train_x
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    c.hold_y = c
        .hold_x
        .iter()
        .map(|r| 3.0 * (1.7 * r[0]).sin() + 0.5)
        .collect();
    c
}

fn lm_selection() -> EngineSelection {
    EngineSelection {
        refine: RefineMethod::Lm(LmOptions::default()),
        archive: None,
        simplify: false,
        linearize: false,
    }
}

fn lm() -> Kernel2Evaluator {
    evaluator().with_engine(lm_selection()).unwrap()
}

/// Guarded refinement with a counting, always-admitting guard: (result, budget used, questions).
fn refine(
    e: &Kernel2Evaluator,
    program: &str,
    c: &Cut,
    trials: usize,
    limit: usize,
) -> (GuardedTrainingRefinement, usize, usize) {
    let asked = Cell::new(0usize);
    let mut budget = EvaluationBudget::new(limit);
    let result = e
        .refine_training_guarded(
            program,
            &c.train_x,
            &c.train_y,
            trials,
            &mut budget,
            &mut || {
                asked.set(asked.get() + 1);
                true
            },
        )
        .unwrap();
    (result, budget.used(), asked.get())
}

/// The trace accounts for every charged trial and refusal, under the default `sites` charge.
fn assert_accounted(refinement: &TrainingRefinement) {
    match &refinement.trace {
        Some(trace) => assert_eq!(
            trace.charged("sites").unwrap(),
            (refinement.evaluations, refinement.rejected_trials),
            "{refinement:?}"
        ),
        None => assert_eq!(refinement.evaluations, 0, "{refinement:?}"),
    }
}

/// The surrogate's charge: initial value, trials, Jacobians at their cost, re-score.
fn surrogate_evaluations(s: &SurrogateTrace) -> usize {
    s.initial + s.trials + s.jacobian_cost * s.jacobians + s.rescore
}

fn surrogate_stages(trace: &RefinementTrace) -> Vec<(&SurrogateTrace, Option<&PipelineCheck>)> {
    trace
        .stages
        .iter()
        .filter_map(|s| match s {
            StageTrace::Surrogate {
                surrogate,
                pipeline,
                ..
            } => Some((surrogate, pipeline.as_ref())),
            _ => None,
        })
        .collect()
}

/// The LM-refined program of the sine fixture, pinned so the emitted-winner parity lanes can
/// replay exactly this text.
const PINNED_LM_PROGRAM: &str = "(pipe X (derive append=false expr=\"sin((1.6999999999997006*c0))\") (ridge lambda_frac=0.000000001))";

/// Every refusal point: a refused guard is never asked again, nothing is charged after it, and the
/// trace still accounts for what was.
/// The index of the one expression stage of a fixture program.
fn program_stage(program: &str) -> usize {
    let GenomeIr::Pipeline(pipe) = GenomeIr::parse(program).unwrap() else {
        unreachable!("pipeline fixture")
    };
    pipe.stages
        .iter()
        .position(|s| matches!(s.op.as_str(), "map" | "derive"))
        .unwrap()
}

/// A hand-built consistent trace: a coordinate stage (5 trials, 1 refused) and a surrogate stage
/// with two sites (initial 1 + trials 2 + 2 Jacobians at 2 + re-score 1 = 8, one refused trial)
/// whose winner the pipeline check accepted (2 passes).
fn synthetic_trace() -> RefinementTrace {
    let bits = f64::to_bits;
    RefinementTrace {
        method: DAMPED_LS.into(),
        stages: vec![
            StageTrace::Coordinate {
                stage: 0,
                reason: super::refinement::ELEMENTWISE.into(),
                evaluations: 5,
                rejected_trials: 1,
            },
            StageTrace::Surrogate {
                stage: 1,
                surrogate: SurrogateTrace {
                    status: "gradient".into(),
                    changed: true,
                    sites: 2,
                    jacobian_cost: 2,
                    initial: 1,
                    trials: 2,
                    jacobians: 2,
                    rescore: 1,
                    rejected_trials: 1,
                    solve_failures: 1,
                    accepted: 1,
                    rescored: Some(bits(0.5)),
                    nonsmooth_rows: 0,
                    saturated_rows: 0,
                    iterations: vec![
                        IterationBits {
                            mu: bits(1e-3),
                            objective: bits(2.0),
                            trial: None,
                            accepted: false,
                            solved: true,
                        },
                        IterationBits {
                            mu: bits(2e-3),
                            objective: bits(2.0),
                            trial: None,
                            accepted: false,
                            solved: false,
                        },
                        IterationBits {
                            mu: bits(8e-3),
                            objective: bits(2.0),
                            trial: Some(bits(1.0)),
                            accepted: true,
                            solved: true,
                        },
                    ],
                },
                pipeline: Some(PipelineCheck {
                    evaluations: 2,
                    rejected_trials: 0,
                    start: Some(bits(3.0)),
                    winner: Some(bits(2.0)),
                    accepted: true,
                }),
            },
        ],
    }
}

/// The method id, fallback reasons, statuses, a serialized trace, failure text and the CLI help
/// all pass the shipped-text lint.
// fix-01: the surrogate minimum, the start's pipeline pass first, strict decrease, the binding of a
// trace to its parent program, and the default route against the stack before the wiring.

/// A derive stage with two literal sites (the text the pre-wiring probe measured).
const TWO: &str = "(pipe X (derive expr=\"sin(0.5*c0)+0.3*c0\" append=false) (ridge))";
/// Two derive stages with one literal site each; the second reads the first's appended column.
const TWO_STAGES: &str = "(pipe X (derive append=true expr=\"sin((0.5*c0))\") (derive append=false expr=\"cos((0.7*c1))\") (ridge))";
/// On `cos_data`. The last stage reads only the original column, so the whole pipeline's loss
/// ignores the first stage: any surrogate winner ties the start bit for bit. Written in printed
/// form, so a kept start is no change at all.
const TIE: &str = "(pipe X (derive append=true expr=\"cos((0.5*c0))\") (derive append=false expr=\"sin(c0)\") (ridge))";
/// On `cos_data`. `abs` folds the negative lobe of the frequency the surrogate recovers, so its
/// winner fits the whole pipeline worse than the start.
const RISE: &str =
    "(pipe X (derive append=false expr=\"cos((0.5*c0))\") (map expr=\"abs(x)\") (ridge))";
/// On `cos_data`. The start is positive on every row; the winner's negative lobe makes `sqrt`
/// fault, so the winner's pipeline pass is refused.
const WINNER_FAULT: &str =
    "(pipe X (derive append=false expr=\"cos((0.5*c0))\") (map expr=\"sqrt(x)\") (ridge))";
/// On `sine_data`. The start is clean on the training prefix, but `sqrt` of its negative half
/// faults the whole pipeline: the start's own pipeline pass fails.
const START_FAULT: &str =
    "(pipe X (derive expr=\"sin(0.5*c0)\" append=false) (map expr=\"sqrt(x)\") (ridge))";

/// The grid of `sine_data` with the target `2 cos(1.3 x) + 0.5`.
fn cos_data() -> Cut {
    let mut c = sine_data();
    c.train_y = c
        .train_x
        .iter()
        .map(|r| 2.0 * (1.3 * r[0]).cos() + 0.5)
        .collect();
    c.hold_y = c
        .hold_x
        .iter()
        .map(|r| 2.0 * (1.3 * r[0]).cos() + 0.5)
        .collect();
    c
}

fn lm_one() -> Kernel2Evaluator {
    evaluator()
        .with_engine(EngineSelection {
            refine: RefineMethod::Lm(LmOptions {
                jacobian: LmJacobianCharge::One,
                ..LmOptions::default()
            }),
            archive: None,
            simplify: false,
            linearize: false,
        })
        .unwrap()
}

/// The trace accounts for every charge and is bound to its parent under `trials` per program.
fn assert_bound(refinement: &TrainingRefinement, parent: &str, jacobian: &str, trials: usize) {
    match &refinement.trace {
        Some(trace) => assert_eq!(
            trace.check(parent, jacobian, trials).unwrap(),
            (refinement.evaluations, refinement.rejected_trials),
            "{refinement:?}"
        ),
        None => assert_eq!(refinement.evaluations, 0, "{refinement:?}"),
    }
}

/// The one stage of a single-stage trace, which must be a surrogate stage with its passes.
fn only_surrogate(refinement: &TrainingRefinement) -> (&SurrogateTrace, &PipelineCheck) {
    match &refinement.trace.as_ref().unwrap().stages[..] {
        [StageTrace::Surrogate {
            surrogate,
            pipeline: Some(pipeline),
            ..
        }] => (surrogate, pipeline),
        other => panic!("one surrogate stage expected: {other:?}"),
    }
}

/// Under its surrogate minimum (literal sites plus five, or six under the `one` charge) a derive
/// stage is the coordinate search bit for bit, with the reason recorded, where the method used to
/// spend surrogate passes that could change nothing. At the minimum the surrogate runs.
/// The surrogate improves the stage, but the whole pipeline's loss ties bit for bit: the start is
/// kept, nothing changes, and the tie is not accepted.
/// The surrogate improves the stage, but the whole pipeline's loss rises: the start is kept.
/// The surrogate improves the stage, but the winner's whole-pipeline pass faults: a refused
/// trial, not a failure, and the start is kept.
/// The start fails the whole-pipeline objective though it is clean on the training prefix: the
/// stage ends exactly as in the coordinate route (failure text, one refused evaluation, program,
/// budget and guard questions) before the surrogate spends anything.
/// A sum of `count` one-literal terms in a balanced tree.
fn literal_sum(count: usize) -> String {
    fn balanced(terms: &[String]) -> String {
        match terms {
            [one] => one.clone(),
            _ => {
                let (a, b) = terms.split_at(terms.len() / 2);
                format!("({}+{})", balanced(a), balanced(b))
            }
        }
    }
    let terms: Vec<String> = (1..=count).map(|i| format!("({i}.0*c0)")).collect();
    balanced(&terms)
}

/// Every legitimate trace passes the parent-bound check with the counts it was charged, however
/// the stage ended; records whose own counts re-derive, but that do not fit the parent's stages
/// or the allowance, are refused.
/// The validation refusal and the new trace records pass the shipped-text lint too.
/// The default (coordinate) route against the stack before the method was wired: the digests and
/// values below were measured at a8bd39855, the commit before the wiring, by these same loops
/// (the fix-01 probe ran them in this lane on that commit's sources), so this compares the
/// default route with the old code, not with itself.
/// Review 01 of the wiring PR (B17-R01-COUNTER-OVERFLOW): a sidecar's surrogate counts are
/// untrusted, so extreme `accepted` and `jacobians` values are refused as a disagreement, never by
/// an overflow panic, whatever the build's overflow checks.
/// Review 02 of the wiring PR (B17-R02-UNBOUNDED-PARENT-PARSE): a refinement parent read back from
/// evidence is untrusted, so the trace check parses only the pipeline form and counts each
/// expression source's nesting bytes before the expression parser sees it. Unary plus and
/// redundant parentheses never reach the parsed tree, so a source at the limit binds exactly as
/// its plain form does; one byte more is refused unparsed. A closed-form parent is refused by the
/// pre-parse check's own message (review 03, B17-R03-PIPELINE-GUARD-COVERAGE): the variant check
/// after the parse refuses it too, but without that message, so the order is observed.
mod refinement_lm_tests;
