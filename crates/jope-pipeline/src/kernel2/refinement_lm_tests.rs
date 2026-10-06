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

#[test]
fn the_default_engine_keeps_every_result_and_charge_bit_for_bit() {
    let c = sine_data();
    let plain = evaluator();
    let explicit = evaluator().with_engine(EngineSelection::default()).unwrap();
    assert!(explicit.engine_record().is_none() && plain.engine_record().is_none());
    for program in [
        SINE,
        MAP,
        MIXED,
        FAULTING,
        "(pipe X (map expr=\"sqrt(x)\") (derive expr=\"sin(0.5*c0)\" append=false) (ridge))",
    ] {
        for (trials, limit) in [(0, 10), (1, 10), (3, 10), (7, 7), (60, 60), (60, 25)] {
            let (a, used_a, asked_a) = refine(&plain, program, &c, trials, limit);
            let (b, used_b, asked_b) = refine(&explicit, program, &c, trials, limit);
            assert_eq!(
                serde_json::to_vec(&a).unwrap(),
                serde_json::to_vec(&b).unwrap(),
                "{program} {trials} {limit}"
            );
            assert_eq!((used_a, asked_a), (used_b, asked_b));
            assert!(b.refinement.trace.is_none());
        }
    }
    let parents = vec![measured(&plain, SINE, &c), measured(&plain, MIXED, &c)];
    for limit in [0, 3, 12, 40] {
        let run = |e: &Kernel2Evaluator| {
            let mut budget = EvaluationBudget::new(limit);
            let result = e
                .finish_generation_guarded(
                    &parents,
                    &c,
                    4,
                    2,
                    10,
                    Clock::Frozen,
                    &mut budget,
                    &mut || true,
                )
                .unwrap();
            (serde_json::to_vec(&result).unwrap(), budget.used())
        };
        assert_eq!(run(&plain), run(&explicit), "{limit}");
    }
}

#[test]
fn lm_refines_a_derive_stage_only_on_a_strict_pipeline_decrease_and_charges_every_pass() {
    let c = sine_data();
    let (result, used, _) = refine(&lm(), SINE, &c, 60, 60);
    let refinement = &result.refinement;
    assert!(
        refinement.failure.is_none() && !result.stopped,
        "{refinement:?}"
    );
    assert!(refinement.changed, "{refinement:?}");
    assert_eq!(used, refinement.evaluations);
    assert!(refinement.evaluations <= 60);
    assert_accounted(refinement);
    let trace = refinement.trace.as_ref().unwrap();
    assert_eq!(trace.method, DAMPED_LS);
    let stages = surrogate_stages(trace);
    assert_eq!(stages.len(), 1, "{trace:?}");
    let (surrogate, pipeline) = stages[0];
    assert!(surrogate.changed && surrogate_evaluations(surrogate) <= 58);
    let pipeline = pipeline.unwrap();
    assert_eq!(pipeline.evaluations, 2);
    assert!(pipeline.accepted);
    assert!(f64::from_bits(pipeline.winner.unwrap()) < f64::from_bits(pipeline.start.unwrap()));
    // The accepted program is admissible and fits the training rows better than the start.
    feat_math::kernel::search::admit_child(&refinement.program, evaluator().profile(), 1).unwrap();
    let loss = |g: &str| {
        let model = feat_math::fit(&GenomeIr::parse(g).unwrap(), &c.train_x, &c.train_y).unwrap();
        let p = model.predict(&c.train_x).unwrap();
        p.iter()
            .zip(&c.train_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / p.len() as f64
    };
    assert!(loss(&refinement.program) < loss(SINE), "{refinement:?}");
}

/// The LM-refined program of the sine fixture, pinned so the emitted-winner parity lanes can
/// replay exactly this text.
#[test]
fn the_lm_refined_fixture_program_is_pinned() {
    let c = sine_data();
    let (result, _, _) = refine(&lm(), SINE, &c, 60, 60);
    assert_eq!(result.refinement.program, PINNED_LM_PROGRAM);
    let parsed = GenomeIr::parse(PINNED_LM_PROGRAM).unwrap();
    assert_eq!(parsed.canonical(), PINNED_LM_PROGRAM);
}

const PINNED_LM_PROGRAM: &str = "(pipe X (derive append=false expr=\"sin((1.6999999999997006*c0))\") (ridge lambda_frac=0.000000001))";

#[test]
fn held_out_poison_cannot_change_lm_refinement_at_either_level() {
    let c = sine_data();
    let mut poisoned = c.clone();
    poisoned.hold_x = vec![vec![-1e100], vec![1e100]];
    poisoned.hold_y = vec![f64::NAN, f64::INFINITY];
    let e = lm();
    // The training call: the held-out half has no way in.
    let clean = refine(&e, SINE, &c, 60, 60);
    let dirty = refine(&e, SINE, &poisoned, 60, 60);
    assert_eq!(clean.0.refinement, dirty.0.refinement);
    assert_eq!((clean.1, clean.2), (dirty.1, dirty.2));
    assert!(clean.0.refinement.changed && clean.0.refinement.trace.is_some());
    // The generation hook: same refinement and trace; only the child measurement differs.
    let parent = measured(&e, SINE, &c);
    let run = |c: &Cut| {
        let mut budget = EvaluationBudget::new(180);
        let result = e
            .finish_generation_guarded(
                std::slice::from_ref(&parent),
                c,
                4,
                1,
                179,
                Clock::Frozen,
                &mut budget,
                &mut || true,
            )
            .unwrap();
        (result, budget.used())
    };
    let (clean, clean_used) = run(&c);
    let (dirty, dirty_used) = run(&poisoned);
    assert_eq!(clean.children[0].refinement, dirty.children[0].refinement);
    assert_eq!(clean_used, dirty_used);
    assert!(clean.children[0].refinement.trace.is_some());
    assert!(clean.children[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
    assert!(!dirty.children[0]
        .measurement
        .as_ref()
        .unwrap()
        .observation
        .measured());
}

#[test]
fn lm_budget_accounting_holds_at_every_allowance() {
    let c = sine_data();
    let e = lm();
    for limit in 0..40 {
        let mut budget = EvaluationBudget::new(limit);
        let parent = e.measure(SINE, &c, 3, "random", Clock::Frozen, &mut budget);
        let before = budget.used();
        let result = e
            .finish_generation_guarded(
                &parent.into_iter().collect::<Vec<_>>(),
                &c,
                4,
                1,
                30,
                Clock::Frozen,
                &mut budget,
                &mut || true,
            )
            .unwrap();
        assert!(budget.used() <= limit, "{limit}");
        assert_eq!(
            budget.used() - before,
            result
                .children
                .iter()
                .map(|c| c.refinement.evaluations + usize::from(c.measurement.is_some()))
                .sum::<usize>(),
            "{limit}"
        );
        for child in &result.children {
            assert!(child.refinement.evaluations <= 30, "{limit}");
            assert_accounted(&child.refinement);
            if let Some(trace) = &child.refinement.trace {
                for (surrogate, _) in surrogate_stages(trace) {
                    // Two passes of the stage's allowance stay reserved for the pipeline check.
                    assert!(surrogate_evaluations(surrogate) + 2 <= 30, "{limit}");
                }
            }
        }
    }
    // A whole cut: every measurement and trial shares one budget, and the coordinator's own
    // accounting check (`budget.used() == observations + objective trials`) passes.
    let mut search = crate::search::SearchOptions::new(
        crate::search::Budget {
            evaluations: 60,
            seconds: 120,
        },
        7,
    );
    search.clock = Clock::Frozen;
    search.seeds = vec![GenomeIr::parse(SINE).unwrap().canonical()];
    let options = search::Options::new(
        search,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        search::Refinement {
            top_k: 2,
            trials_per_program: 20,
        },
    )
    .unwrap()
    .with_engine(lm_selection())
    .unwrap();
    let searched = search::run_cut(&c, &options).unwrap();
    assert_eq!(searched.engine, lm_selection().record());
    assert!(searched.objective_trials > 0);
    assert_eq!(
        searched.completed_evaluations,
        searched.candidate_measurements + searched.objective_trials
    );
    assert!(searched
        .refinements
        .iter()
        .any(|r| r.refinement.trace.is_some()));
    for refined in &searched.refinements {
        assert!(refined.refinement.evaluations <= 20);
        assert_accounted(&refined.refinement);
    }
}

/// Every refusal point: a refused guard is never asked again, nothing is charged after it, and the
/// trace still accounts for what was.
#[test]
fn a_refusal_at_any_question_stops_cleanly_and_stays_accounted() {
    let c = sine_data();
    let e = lm();
    let (_, _, total) = refine(&e, SINE, &c, 60, 60);
    assert!(total > 4);
    for allowed in 0..total {
        let asked = Cell::new(0usize);
        let mut budget = EvaluationBudget::new(60);
        let result = e
            .refine_training_guarded(SINE, &c.train_x, &c.train_y, 60, &mut budget, &mut || {
                asked.set(asked.get() + 1);
                asked.get() <= allowed
            })
            .unwrap();
        assert!(result.stopped, "{allowed}");
        assert_eq!(asked.get(), allowed + 1, "{allowed}");
        assert_eq!(budget.used(), result.refinement.evaluations, "{allowed}");
        assert!(result.refinement.failure.is_none(), "{allowed}");
        assert_accounted(&result.refinement);
    }
    // Malformed training is refused before any question or charge.
    for (rows, targets) in [
        (vec![vec![1.0], vec![]], vec![1.0, 2.0]),
        (Vec::new(), Vec::new()),
        (c.train_x.clone(), vec![f64::NAN; c.train_x.len()]),
    ] {
        let mut budget = EvaluationBudget::new(20);
        assert!(e
            .refine_training_guarded(SINE, &rows, &targets, 20, &mut budget, &mut || panic!(
                "no question before validation"
            ))
            .is_err());
        assert_eq!(budget.used(), 0);
    }
}

#[test]
fn elementwise_and_small_allowance_stages_use_the_coordinate_search_bit_for_bit() {
    let c = sine_data();
    let (coordinate, method) = (evaluator(), lm());
    let cases = [
        (
            MAP,
            vec![1usize, 2, 3, 20, 60],
            super::refinement::ELEMENTWISE,
        ),
        (
            "(pipe X (map expr=\"sin(0.5*x)\") (ridge))",
            vec![1, 5, 40],
            super::refinement::ELEMENTWISE,
        ),
        (SINE, vec![1, 2], super::refinement::SMALL_ALLOWANCE),
    ];
    for (program, trials, reason) in cases {
        for trials in trials {
            let (old, old_used, old_asked) = refine(&coordinate, program, &c, trials, 100);
            let (new, new_used, new_asked) = refine(&method, program, &c, trials, 100);
            let mut stripped = new.refinement.clone();
            let trace = stripped.trace.take();
            assert_eq!(stripped, old.refinement, "{program} {trials}");
            assert_eq!(new.stopped, old.stopped);
            assert_eq!(
                (new_used, new_asked),
                (old_used, old_asked),
                "{program} {trials}"
            );
            let expected = (old.refinement.evaluations > 0).then(|| RefinementTrace {
                method: DAMPED_LS.into(),
                stages: vec![StageTrace::Coordinate {
                    stage: program_stage(program),
                    reason: reason.into(),
                    evaluations: old.refinement.evaluations,
                    rejected_trials: old.refinement.rejected_trials,
                }],
            });
            assert_eq!(trace, expected, "{program} {trials}");
            assert_accounted(&new.refinement);
        }
    }
    // In a mixed program the map stage falls back and the derive stage is fitted by the method;
    // the allowance leaves the derive stage room after the coordinate search's halvings.
    let (mixed, used, _) = refine(&method, MIXED, &c, 400, 400);
    assert_accounted(&mixed.refinement);
    assert_eq!(used, mixed.refinement.evaluations);
    let trace = mixed.refinement.trace.unwrap();
    assert!(matches!(
        &trace.stages[0],
        StageTrace::Coordinate { stage: 0, reason, .. } if reason == super::refinement::ELEMENTWISE
    ));
    assert!(trace
        .stages
        .iter()
        .any(|s| matches!(s, StageTrace::Surrogate { stage: 1, .. })));
}

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

#[test]
fn the_trace_check_refuses_every_count_that_does_not_re_derive() {
    let good = synthetic_trace();
    assert_eq!(good.charged("sites").unwrap(), (15, 2));
    assert_eq!(
        RefinementTrace {
            method: DAMPED_LS.into(),
            stages: vec![StageTrace::Failed { stage: 3 }],
        }
        .charged("sites")
        .unwrap(),
        (1, 1)
    );
    // Two sites: the `one` charge disagrees with the recorded Jacobian cost.
    assert!(good.charged("one").is_err());
    assert!(good.charged("two").is_err());
    type Edit = fn(&mut RefinementTrace);
    fn surrogate(t: &mut RefinementTrace) -> &mut SurrogateTrace {
        match &mut t.stages[1] {
            StageTrace::Surrogate { surrogate, .. } => surrogate,
            _ => unreachable!(),
        }
    }
    fn pipeline(t: &mut RefinementTrace) -> &mut Option<PipelineCheck> {
        match &mut t.stages[1] {
            StageTrace::Surrogate { pipeline, .. } => pipeline,
            _ => unreachable!(),
        }
    }
    let edits: [(&str, Edit); 18] = [
        ("method", |t| t.method = "other".into()),
        ("order", |t| t.stages.swap(0, 1)),
        ("reason", |t| {
            if let StageTrace::Coordinate { reason, .. } = &mut t.stages[0] {
                *reason = "faster".into();
            }
        }),
        ("coordinate refusals", |t| {
            if let StageTrace::Coordinate {
                rejected_trials, ..
            } = &mut t.stages[0]
            {
                *rejected_trials = 6;
            }
        }),
        ("status", |t| surrogate(t).status = "unsupported".into()),
        ("unknown status", |t| surrogate(t).status = "done".into()),
        ("jacobian cost", |t| surrogate(t).jacobian_cost = 1),
        ("trials", |t| surrogate(t).trials += 1),
        ("failed solves", |t| surrogate(t).solve_failures += 1),
        ("accepted", |t| surrogate(t).accepted = 0),
        ("surrogate refusals", |t| surrogate(t).rejected_trials = 2),
        ("initial", |t| surrogate(t).initial = 2),
        ("re-score without value", |t| surrogate(t).rescored = None),
        ("missing check", |t| *pipeline(t) = None),
        ("check decision", |t| {
            pipeline(t).as_mut().unwrap().accepted = false
        }),
        ("check refusals", |t| {
            pipeline(t).as_mut().unwrap().rejected_trials = 1;
        }),
        ("check passes", |t| {
            pipeline(t).as_mut().unwrap().evaluations = 1
        }),
        ("unchanged with a check", |t| surrogate(t).changed = false),
    ];
    for (label, edit) in edits {
        let mut trace = synthetic_trace();
        edit(&mut trace);
        let charged = trace.charged("sites");
        assert!(charged.is_err(), "{label}: {charged:?}");
    }
}

#[test]
fn a_refused_start_is_one_charged_failure_with_its_trace() {
    let c = data();
    for limit in [3, 20] {
        let (result, used, _) = refine(&lm(), FAULTING, &c, limit, limit);
        let refinement = &result.refinement;
        assert!(refinement.failure.is_some(), "{refinement:?}");
        assert_eq!((refinement.evaluations, refinement.rejected_trials), (1, 1));
        assert_eq!(used, 1);
        assert!(!refinement.changed);
        assert_accounted(refinement);
    }
}

#[test]
fn the_selection_needs_an_allowance_it_can_run_and_records_canonically() {
    let options = |top_k, trials| {
        let mut search = crate::search::SearchOptions::new(
            crate::search::Budget {
                evaluations: 40,
                seconds: 120,
            },
            7,
        );
        search.clock = Clock::Frozen;
        search::Options::new(
            search,
            "kernel2-baseline",
            FaultPolicy::RejectHard,
            search::Refinement {
                top_k,
                trials_per_program: trials,
            },
        )
        .unwrap()
        .with_engine(lm_selection())
    };
    assert!(options(0, 10).is_err());
    assert!(options(1, 2).is_err());
    assert!(options(1, DAMPED_LS_MIN_TRIALS).is_ok());
    assert!(evaluator()
        .with_engine(EngineSelection {
            refine: RefineMethod::Lm(LmOptions {
                tau: f64::NAN,
                ..LmOptions::default()
            }),
            archive: None,
        })
        .is_err());
    // Round trip, and refusal of anything the selection would not write back.
    for jacobian in [LmJacobianCharge::Sites, LmJacobianCharge::One] {
        let selection = EngineSelection {
            refine: RefineMethod::Lm(LmOptions {
                jacobian,
                ..LmOptions::default()
            }),
            archive: None,
        };
        let record = selection.record().unwrap();
        assert_eq!(record.selection().unwrap(), selection);
        let bytes = serde_json::to_vec(&record).unwrap();
        let back: EngineRecord = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, record);
    }
    let record = lm_selection().record().unwrap();
    assert_eq!(
        serde_json::to_value(&record).unwrap(),
        serde_json::json!({"refine": {"id": DAMPED_LS, "jacobian": "sites", "tau": 1e-3}})
    );
    for bad in [
        serde_json::json!({}),
        serde_json::json!({"refine": {"id": "other", "jacobian": "sites", "tau": 1e-3}}),
        serde_json::json!({"refine": {"id": DAMPED_LS, "jacobian": "two", "tau": 1e-3}}),
        serde_json::json!({"refine": {"id": DAMPED_LS, "jacobian": "sites", "tau": 0.0}}),
    ] {
        let parsed: EngineRecord = serde_json::from_value(bad.clone()).unwrap();
        assert!(parsed.selection().is_err(), "{bad}");
    }
    assert!(serde_json::from_value::<EngineRecord>(
        serde_json::json!({"refine": {"id": DAMPED_LS, "jacobian": "sites", "tau": 1e-3, "x": 1}})
    )
    .is_err());
}

/// The method id, fallback reasons, statuses, a serialized trace, failure text and the CLI help
/// all pass the shipped-text lint.
#[test]
fn shipped_text_of_the_method_passes_the_lint() {
    use clap::CommandFactory;
    let lint = |text: &str| {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    };
    lint(DAMPED_LS);
    for reason in super::refinement::FALLBACK_REASONS {
        lint(reason);
    }
    for status in feat_math::kernel::expression_search::LmStatus::ALL {
        lint(status.as_str());
    }
    let c = sine_data();
    for (program, trials) in [(MIXED, 400), (SINE, 60)] {
        let (result, _, _) = refine(&lm(), program, &c, trials, trials);
        let trace = result.refinement.trace.unwrap();
        lint(&serde_json::to_string(&trace).unwrap());
    }
    lint(&serde_json::to_string(&lm_selection().record()).unwrap());
    let (refused, _, _) = refine(&lm(), FAULTING, &data(), 20, 20);
    lint(refused.refinement.failure.as_deref().unwrap());
    let mut command = crate::cli::Cli::command();
    command.build();
    let run = command.find_subcommand_mut("run").unwrap();
    let flag = run
        .get_arguments()
        .find(|a| a.get_id() == "refine_method")
        .unwrap();
    let help = flag.get_help().unwrap().to_string();
    assert!(help.contains("coordinate") && help.contains("lm"), "{help}");
    lint(&help);
    if let Some(long) = flag.get_long_help() {
        lint(&long.to_string());
    }
}

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
#[test]
fn a_stage_under_its_surrogate_minimum_uses_the_coordinate_search_bit_for_bit() {
    use super::refinement::{stage_minimum, SMALL_ALLOWANCE};
    let c = sine_data();
    let coordinate = evaluator();
    // The start's and the winner's pipeline passes, the surrogate's initial value, one Jacobian
    // at its charge, one trial and the re-score.
    assert_eq!((stage_minimum(1), stage_minimum(2)), (6, 7));
    for (e, jacobian, program, minimum) in [
        (lm(), "sites", SINE, 6),
        (lm(), "sites", TWO, 7),
        (lm_one(), "one", TWO, 6),
    ] {
        for trials in DAMPED_LS_MIN_TRIALS..minimum {
            let (old, old_used, old_asked) = refine(&coordinate, program, &c, trials, 100);
            let (new, new_used, new_asked) = refine(&e, program, &c, trials, 100);
            let mut stripped = new.refinement.clone();
            let trace = stripped.trace.take();
            assert_eq!(stripped, old.refinement, "{program} {trials}");
            assert_eq!(new.stopped, old.stopped, "{program} {trials}");
            assert_eq!(
                (new_used, new_asked),
                (old_used, old_asked),
                "{program} {trials}"
            );
            let expected = RefinementTrace {
                method: DAMPED_LS.into(),
                stages: vec![StageTrace::Coordinate {
                    stage: 0,
                    reason: SMALL_ALLOWANCE.into(),
                    evaluations: old.refinement.evaluations,
                    rejected_trials: old.refinement.rejected_trials,
                }],
            };
            assert_eq!(trace, Some(expected), "{program} {trials}");
            assert_bound(&new.refinement, program, jacobian, trials);
        }
        // At the minimum the surrogate runs, within the allowance.
        let (at, used, _) = refine(&e, program, &c, minimum, 100);
        assert_eq!(used, at.refinement.evaluations);
        assert!(at.refinement.evaluations <= minimum);
        assert!(
            matches!(
                &at.refinement.trace.as_ref().unwrap().stages[..],
                [StageTrace::Surrogate { .. }]
            ),
            "{program} {at:?}"
        );
        assert_bound(&at.refinement, program, jacobian, minimum);
    }
}

/// The surrogate improves the stage, but the whole pipeline's loss ties bit for bit: the start is
/// kept, nothing changes, and the tie is not accepted.
#[test]
fn a_tied_pipeline_loss_keeps_the_start() {
    let c = cos_data();
    assert_eq!(GenomeIr::parse(TIE).unwrap().canonical(), TIE);
    let (result, used, _) = refine(&lm(), TIE, &c, 60, 60);
    let refinement = &result.refinement;
    let (surrogate, pipeline) = only_surrogate(refinement);
    // The surrogate's winner re-scored strictly better on the training prefix.
    assert!(
        surrogate.changed && surrogate.rescored.is_some(),
        "{refinement:?}"
    );
    assert_eq!((pipeline.evaluations, pipeline.rejected_trials), (2, 0));
    assert!(pipeline.start.is_some());
    assert_eq!(pipeline.winner, pipeline.start, "a bit-for-bit tie");
    assert!(!pipeline.accepted);
    assert_eq!(refinement.program, TIE);
    assert!(!refinement.changed && refinement.failure.is_none() && !result.stopped);
    assert_eq!(used, refinement.evaluations);
    assert_eq!(refinement.evaluations, surrogate_evaluations(surrogate) + 2);
    assert_bound(refinement, TIE, "sites", 60);
}

/// The surrogate improves the stage, but the whole pipeline's loss rises: the start is kept.
#[test]
fn a_rising_pipeline_loss_keeps_the_start() {
    let c = cos_data();
    assert_eq!(GenomeIr::parse(RISE).unwrap().canonical(), RISE);
    let (result, used, _) = refine(&lm(), RISE, &c, 60, 60);
    let refinement = &result.refinement;
    let (surrogate, pipeline) = only_surrogate(refinement);
    assert!(
        surrogate.changed && surrogate.rescored.is_some(),
        "{refinement:?}"
    );
    assert_eq!((pipeline.evaluations, pipeline.rejected_trials), (2, 0));
    let (start, winner) = (pipeline.start.unwrap(), pipeline.winner.unwrap());
    assert!(
        f64::from_bits(winner) > f64::from_bits(start),
        "{pipeline:?}"
    );
    assert!(!pipeline.accepted);
    assert_eq!(refinement.program, RISE);
    assert!(!refinement.changed && refinement.failure.is_none() && !result.stopped);
    assert_eq!(used, refinement.evaluations);
    assert_eq!(refinement.evaluations, surrogate_evaluations(surrogate) + 2);
    assert_bound(refinement, RISE, "sites", 60);
}

/// The surrogate improves the stage, but the winner's whole-pipeline pass faults: a refused
/// trial, not a failure, and the start is kept.
#[test]
fn a_faulting_winner_pass_keeps_the_start() {
    let c = cos_data();
    assert_eq!(
        GenomeIr::parse(WINNER_FAULT).unwrap().canonical(),
        WINNER_FAULT
    );
    let (result, used, _) = refine(&lm(), WINNER_FAULT, &c, 60, 60);
    let refinement = &result.refinement;
    let (surrogate, pipeline) = only_surrogate(refinement);
    assert!(
        surrogate.changed && surrogate.rescored.is_some(),
        "{refinement:?}"
    );
    assert_eq!((pipeline.evaluations, pipeline.rejected_trials), (2, 1));
    assert!(pipeline.start.is_some() && pipeline.winner.is_none() && !pipeline.accepted);
    assert_eq!(refinement.program, WINNER_FAULT);
    assert!(!refinement.changed && refinement.failure.is_none() && !result.stopped);
    assert_eq!(used, refinement.evaluations);
    assert_eq!(
        (refinement.evaluations, refinement.rejected_trials),
        (
            surrogate_evaluations(surrogate) + 2,
            surrogate.rejected_trials + 1
        )
    );
    assert_bound(refinement, WINNER_FAULT, "sites", 60);
}

/// The start fails the whole-pipeline objective though it is clean on the training prefix: the
/// stage ends exactly as in the coordinate route (failure text, one refused evaluation, program,
/// budget and guard questions) before the surrogate spends anything.
#[test]
fn a_faulting_start_pass_matches_the_coordinate_route() {
    let c = sine_data();
    let (coordinate, method) = (evaluator(), lm());
    for trials in [6, 7, 12, 60] {
        let (old, old_used, old_asked) = refine(&coordinate, START_FAULT, &c, trials, trials);
        assert_eq!(
            (old.refinement.evaluations, old.refinement.rejected_trials),
            (1, 1)
        );
        assert!(old.refinement.failure.is_some());
        let (new, new_used, new_asked) = refine(&method, START_FAULT, &c, trials, trials);
        let mut stripped = new.refinement.clone();
        let trace = stripped.trace.take().unwrap();
        assert_eq!(stripped, old.refinement, "{trials}");
        assert_eq!(new.stopped, old.stopped);
        assert_eq!((new_used, new_asked), (old_used, old_asked), "{trials}");
        assert_eq!(trace.stages, vec![StageTrace::Failed { stage: 0 }]);
        assert_bound(&new.refinement, START_FAULT, "sites", trials);
    }
}

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
#[test]
fn the_trace_check_binds_every_stage_to_the_parent_program() {
    use super::refinement::{ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED};
    let (sine, cos, line) = (sine_data(), cos_data(), data());
    let (sites, one) = (lm(), lm_one());
    let mut kinds = std::collections::BTreeSet::new();
    let mut note = |refinement: &TrainingRefinement| {
        if let Some(trace) = &refinement.trace {
            for stage in &trace.stages {
                kinds.insert(match stage {
                    StageTrace::Coordinate { reason, .. } => reason.clone(),
                    StageTrace::Failed { .. } => "failed start".into(),
                    StageTrace::Stopped { .. } => "stopped start".into(),
                    StageTrace::Surrogate {
                        surrogate,
                        pipeline,
                        ..
                    } => format!(
                        "{} {} {}",
                        surrogate.status,
                        surrogate.changed,
                        pipeline.as_ref().unwrap().evaluations
                    ),
                });
            }
        }
    };
    let runs: [(&Kernel2Evaluator, &str, &str, &Cut, &[usize]); 13] = [
        (&sites, "sites", SINE, &sine, &[3, 4, 6, 7, 12, 60]),
        (&sites, "sites", MAP, &sine, &[3, 20]),
        (&sites, "sites", MIXED, &sine, &[400]),
        (&sites, "sites", FAULTING, &line, &[3, 20]),
        (&sites, "sites", TWO, &sine, &[6, 7, 12, 60]),
        (&sites, "sites", TWO, &line, &[60]),
        (&one, "one", TWO, &sine, &[6, 12, 60]),
        (&sites, "sites", TWO_STAGES, &sine, &[60]),
        (&one, "one", TWO_STAGES, &sine, &[40]),
        (&sites, "sites", TIE, &cos, &[6, 60]),
        (&sites, "sites", RISE, &cos, &[6, 60]),
        (&sites, "sites", WINNER_FAULT, &cos, &[6, 60]),
        (&sites, "sites", START_FAULT, &sine, &[6, 60]),
    ];
    for (e, jacobian, program, c, allowances) in runs {
        for &trials in allowances {
            let (result, _, total) = refine(e, program, c, trials, trials);
            assert_bound(&result.refinement, program, jacobian, trials);
            note(&result.refinement);
            // A refusal at every guard question: zero-evaluation stages, a stopped start, a
            // stopped surrogate and a changed winner whose pass did not run.
            for allowed in (0..total).filter(|_| total <= 80) {
                let asked = Cell::new(0usize);
                let mut budget = EvaluationBudget::new(trials);
                let stopped = e
                    .refine_training_guarded(
                        program,
                        &c.train_x,
                        &c.train_y,
                        trials,
                        &mut budget,
                        &mut || {
                            asked.set(asked.get() + 1);
                            asked.get() <= allowed
                        },
                    )
                    .unwrap();
                assert!(stopped.stopped, "{program} {trials} {allowed}");
                assert_bound(&stopped.refinement, program, jacobian, trials);
                note(&stopped.refinement);
            }
        }
    }
    for kind in [
        ELEMENTWISE,
        SMALL_ALLOWANCE,
        "failed start",
        "stopped start",
        "gradient true 2",
        "gradient true 1",
        "budget_exhausted false 1",
        "stopped false 1",
    ] {
        assert!(kinds.contains(kind), "{kind}: {kinds:?}");
    }

    // Surrogate outcomes the fixtures do not reach, on the one-site parent: a refused start, a
    // refused re-score, failed solves, an exact fit at the start and after a step, a non-finite
    // start, and a refused trial.
    let bits = f64::to_bits;
    let step = |accepted: bool, trial: Option<f64>| IterationBits {
        mu: bits(1e-3),
        objective: bits(2.0),
        trial: trial.map(bits),
        accepted,
        solved: true,
    };
    let unsolved = IterationBits {
        mu: bits(1e-3),
        objective: bits(2.0),
        trial: None,
        accepted: false,
        solved: false,
    };
    let base = SurrogateTrace {
        status: String::new(),
        changed: false,
        sites: 1,
        jacobian_cost: 1,
        initial: 1,
        trials: 0,
        jacobians: 0,
        rescore: 0,
        rejected_trials: 0,
        solve_failures: 0,
        accepted: 0,
        rescored: None,
        nonsmooth_rows: 0,
        saturated_rows: 0,
        iterations: Vec::new(),
    };
    let start_only = PipelineCheck {
        evaluations: 1,
        rejected_trials: 0,
        start: Some(bits(3.0)),
        winner: None,
        accepted: false,
    };
    let exact_after_step = SurrogateTrace {
        status: "exact_fit".into(),
        changed: true,
        trials: 1,
        jacobians: 1,
        rescore: 1,
        accepted: 1,
        rescored: Some(bits(0.0)),
        iterations: vec![step(true, Some(0.0))],
        ..base.clone()
    };
    let one_stage = |surrogate: SurrogateTrace, pipeline: PipelineCheck| RefinementTrace {
        method: DAMPED_LS.into(),
        stages: vec![StageTrace::Surrogate {
            stage: 0,
            surrogate,
            pipeline: Some(pipeline),
        }],
    };
    let synthetic = [
        (
            "initial_refused",
            SurrogateTrace {
                status: "initial_refused".into(),
                rejected_trials: 1,
                ..base.clone()
            },
            start_only.clone(),
            (2, 1),
        ),
        (
            "refused re-score",
            SurrogateTrace {
                status: "gradient".into(),
                trials: 1,
                jacobians: 2,
                rescore: 1,
                rejected_trials: 1,
                accepted: 1,
                iterations: vec![step(true, Some(1.0))],
                ..base.clone()
            },
            start_only.clone(),
            (6, 1),
        ),
        (
            "failed solves",
            SurrogateTrace {
                status: "damping_limit".into(),
                trials: 1,
                jacobians: 1,
                solve_failures: 2,
                iterations: vec![unsolved, unsolved, step(false, Some(4.0))],
                ..base.clone()
            },
            start_only.clone(),
            (4, 0),
        ),
        (
            "exact fit at the start",
            SurrogateTrace {
                status: "exact_fit".into(),
                ..base.clone()
            },
            start_only.clone(),
            (2, 0),
        ),
        (
            "exact fit after a step",
            exact_after_step.clone(),
            PipelineCheck {
                evaluations: 2,
                winner: Some(bits(1.0)),
                accepted: true,
                ..start_only.clone()
            },
            (6, 0),
        ),
        (
            "non-finite start",
            SurrogateTrace {
                status: "non_finite_jacobian".into(),
                ..base.clone()
            },
            start_only.clone(),
            (2, 0),
        ),
        (
            "refused trial",
            SurrogateTrace {
                status: "budget_exhausted".into(),
                trials: 2,
                jacobians: 1,
                rejected_trials: 1,
                iterations: vec![step(false, None), step(false, Some(4.0))],
                ..base.clone()
            },
            start_only.clone(),
            (5, 1),
        ),
    ];
    for (label, surrogate, pipeline, charged) in synthetic {
        let trace = one_stage(surrogate, pipeline);
        assert_eq!(trace.charged("sites").unwrap(), charged, "{label}");
        assert_eq!(trace.check(SINE, "sites", 60).unwrap(), charged, "{label}");
    }
    // A shape outside the surrogate's site range is a legitimate unsupported fallback, and only
    // that.
    let many = format!(
        "(pipe X (derive expr=\"{}\" append=false) (ridge))",
        literal_sum(15)
    );
    let fallback = |reason: &str| RefinementTrace {
        method: DAMPED_LS.into(),
        stages: vec![StageTrace::Coordinate {
            stage: 0,
            reason: reason.into(),
            evaluations: 3,
            rejected_trials: 0,
        }],
    };
    assert_eq!(
        fallback(UNSUPPORTED).check(&many, "sites", 60).unwrap(),
        (3, 0)
    );
    assert!(fallback(SMALL_ALLOWANCE).check(&many, "sites", 60).is_err());
    assert!(fallback(ELEMENTWISE).check(&many, "sites", 60).is_err());

    // The review's first trigger: a surrogate stage replaced by a fallback with equal counts
    // re-derives its counts, but no fallback reason fits this supported derive stage.
    let (real, _, _) = refine(&lm(), SINE, &sine, 60, 60);
    let trace = real.refinement.trace.unwrap();
    let (evaluations, rejected) = trace.charged("sites").unwrap();
    assert!(evaluations >= 6);
    for reason in [ELEMENTWISE, SMALL_ALLOWANCE, UNSUPPORTED] {
        let mut faked = trace.clone();
        faked.stages[0] = StageTrace::Coordinate {
            stage: 0,
            reason: reason.into(),
            evaluations,
            rejected_trials: rejected,
        };
        assert_ne!(faked, trace);
        assert_eq!(faked.charged("sites").unwrap(), (evaluations, rejected));
        assert!(faked.check(SINE, "sites", 60).is_err(), "{reason}");
    }
    // The review's second trigger with one site: two more Jacobians and two fewer trials keep
    // the charge, the iteration count and every refusal; the Jacobian bound refuses it.
    let mut shifted = trace.clone();
    let StageTrace::Surrogate { surrogate, .. } = &mut shifted.stages[0] else {
        unreachable!("one surrogate stage")
    };
    assert_eq!(surrogate.sites, 1);
    let before = surrogate_evaluations(surrogate);
    surrogate.jacobians += 2;
    surrogate.trials -= 2;
    for _ in 0..2 {
        let unaccepted = surrogate
            .iterations
            .iter()
            .position(|i| i.solved && !i.accepted && i.trial.is_some())
            .expect("two scored trials that were not accepted");
        surrogate.iterations.remove(unaccepted);
    }
    assert_eq!(surrogate_evaluations(surrogate), before);
    assert_eq!(
        surrogate.iterations.len(),
        surrogate.trials + surrogate.solve_failures
    );
    assert_ne!(shifted, trace);
    let error = shifted.charged("sites").unwrap_err();
    assert!(format!("{error:#}").contains("accepted steps"), "{error:#}");

    // Records that re-derive their own counts but do not fit the parent or the allowance.
    let mut next_stage = trace.clone();
    if let StageTrace::Surrogate { stage, .. } = &mut next_stage.stages[0] {
        *stage = 1;
    }
    let stages = |stages: Vec<StageTrace>| RefinementTrace {
        method: DAMPED_LS.into(),
        stages,
    };
    let small = |stage: usize, evaluations: usize| StageTrace::Coordinate {
        stage,
        reason: SMALL_ALLOWANCE.into(),
        evaluations,
        rejected_trials: 0,
    };
    let refused: [(&str, RefinementTrace, &str, usize); 9] = [
        ("not the next stage", next_stage, SINE, 60),
        ("other site count", trace.clone(), TWO, 60),
        (
            "map stage",
            trace.clone(),
            "(pipe X (map expr=\"sin((0.5*x))\") (ridge))",
            60,
        ),
        (
            "surrogate over its allowance",
            trace.clone(),
            SINE,
            evaluations - 1,
        ),
        (
            "failed start under the minimum",
            stages(vec![StageTrace::Failed { stage: 0 }]),
            SINE,
            5,
        ),
        (
            "stopped start before another stage",
            stages(vec![StageTrace::Stopped { stage: 0 }, small(1, 1)]),
            TWO_STAGES,
            60,
        ),
        (
            "small allowance at the minimum",
            stages(vec![small(0, 6)]),
            SINE,
            60,
        ),
        (
            "map stage as a small allowance",
            stages(vec![small(1, 2)]),
            MAP,
            60,
        ),
        (
            "unrun winner pass before another stage",
            stages(vec![
                StageTrace::Surrogate {
                    stage: 0,
                    surrogate: exact_after_step.clone(),
                    pipeline: Some(start_only.clone()),
                },
                small(1, 1),
            ]),
            TWO_STAGES,
            60,
        ),
    ];
    for (label, trace, parent, trials) in refused {
        trace.charged("sites").unwrap();
        assert!(trace.check(parent, "sites", trials).is_err(), "{label}");
    }
    // The same kinds of record fit where they are legitimate.
    assert_eq!(
        stages(vec![StageTrace::Failed { stage: 0 }])
            .check(SINE, "sites", 6)
            .unwrap(),
        (1, 1)
    );
    assert_eq!(
        stages(vec![small(0, 1), StageTrace::Stopped { stage: 1 }])
            .check(TWO_STAGES, "sites", 60)
            .unwrap(),
        (1, 0)
    );
    assert_eq!(
        stages(vec![small(0, 5)]).check(SINE, "sites", 60).unwrap(),
        (5, 0)
    );
    assert_eq!(
        trace.check(SINE, "sites", evaluations).unwrap(),
        (evaluations, rejected)
    );
}

/// The validation refusal and the new trace records pass the shipped-text lint too.
#[test]
fn the_validation_text_and_new_records_pass_the_lint() {
    let lint = |text: &str| {
        let hits = feat_math::transpile::lint_shipped_text(text);
        assert!(hits.is_empty(), "{text}: {hits:?}");
    };
    let refusal = lm_selection()
        .validate(search::Refinement {
            top_k: 1,
            trials_per_program: 2,
        })
        .unwrap_err();
    lint(&format!("{refusal:#}"));
    let records = vec![
        StageTrace::Failed { stage: 0 },
        StageTrace::Stopped { stage: 1 },
    ];
    lint(&serde_json::to_string(&records).unwrap());
    for (program, c, trials) in [(TIE, cos_data(), 60), (START_FAULT, sine_data(), 60)] {
        let (result, _, _) = refine(&lm(), program, &c, trials, trials);
        lint(&serde_json::to_string(&result.refinement).unwrap());
    }
}

/// The default (coordinate) route against the stack before the method was wired: the digests and
/// values below were measured at a8bd39855, the commit before the wiring, by these same loops
/// (the fix-01 probe ran them in this lane on that commit's sources), so this compares the
/// default route with the old code, not with itself.
#[test]
fn the_default_route_keeps_the_pre_wiring_results() {
    let e = evaluator();
    let train_loss = |program: &str, c: &Cut| -> Option<u64> {
        let model = feat_math::fit(&GenomeIr::parse(program).ok()?, &c.train_x, &c.train_y).ok()?;
        let p = model.predict(&c.train_x).ok()?;
        let mse = p
            .iter()
            .zip(&c.train_y)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / p.len() as f64;
        Some(mse.to_bits())
    };
    let mut lines = String::new();
    let mut rows = Vec::new();
    for (name, c) in [("sine", sine_data()), ("line", data())] {
        for program in [SINE, MAP, MIXED, FAULTING, TWO] {
            for (trials, limit) in [(3usize, 10usize), (7, 7), (20, 20), (60, 60)] {
                let (result, used, asked) = refine(&e, program, &c, trials, limit);
                let loss = train_loss(&result.refinement.program, &c);
                lines.push_str(&format!(
                    "UNIT\t{name}\t{program}\t{trials}\t{limit}\t{}\t{used}\t{asked}\t{loss:?}\n",
                    serde_json::to_string(&result).unwrap()
                ));
                rows.push(((name, program, trials), (result.refinement, loss)));
            }
        }
    }
    assert_eq!(
        crate::sha256_hex(lines.as_bytes()),
        "027b30765712af8811e5ffe237d837970e8e698ba97f6734cd38065f84b6aae3"
    );
    // Four of those rows in full: program text, evaluations, rejected trials, failure and the
    // training loss bits of the result.
    let row = |name: &str, program: &str, trials: usize| {
        rows.iter()
            .find(|((n, p, t), _)| *n == name && *p == program && *t == trials)
            .map(|(_, r)| r.clone())
            .unwrap()
    };
    for ((name, program, trials), text, counts, failure, bits) in [
        (
            ("sine", SINE, 60),
            "(pipe X (derive append=false expr=\"sin((1.7000007629394531*c0))\") (ridge lambda_frac=0.000000001))",
            (60, 0),
            None,
            4482624987248756777u64,
        ),
        (
            ("sine", MIXED, 20),
            "(pipe X (map expr=\"sin((1.71875*x))\") (derive append=false expr=\"sin(0.7*c0)\") (ridge))",
            (20, 0),
            None,
            4571347593955826666,
        ),
        (
            ("sine", FAULTING, 60),
            "(pipe X (derive append=false expr=\"sqrt(c0-0.5)\") (ridge))",
            (1, 1),
            Some("faulting constant trial"),
            4612781733306906755,
        ),
        (
            ("line", TWO, 60),
            "(pipe X (derive append=false expr=\"(sin((0.5*c0))+(14.3*c0))\") (ridge))",
            (60, 15),
            None,
            4568963289968063070,
        ),
    ] {
        let (refinement, loss) = row(name, program, trials);
        assert_eq!(refinement.program, text);
        assert_eq!(
            (refinement.evaluations, refinement.rejected_trials),
            counts,
            "{text}"
        );
        assert_eq!(refinement.failure.as_deref(), failure, "{text}");
        assert!(refinement.trace.is_none());
        assert_eq!(loss, Some(bits), "{text}");
    }
    // The generation hook, children and their measurements included.
    let c = sine_data();
    let parents = vec![
        measured(&e, SINE, &c),
        measured(&e, MIXED, &c),
        measured(&e, TWO, &c),
    ];
    for (limit, digest, used) in [
        (
            12,
            "cfff0c3a86da51ced4e20eb771c70c68c07a778a9292d1ee1981c71f74159b01",
            11,
        ),
        (
            40,
            "15ace454ac4038af8cfe954e2c719cfa12fa1fcf65e554e7317f9a26ee956700",
            33,
        ),
        (
            200,
            "15ace454ac4038af8cfe954e2c719cfa12fa1fcf65e554e7317f9a26ee956700",
            33,
        ),
    ] {
        let mut budget = EvaluationBudget::new(limit);
        let result = e
            .finish_generation_guarded(
                &parents,
                &c,
                4,
                3,
                10,
                Clock::Frozen,
                &mut budget,
                &mut || true,
            )
            .unwrap();
        let bytes = serde_json::to_vec(&result).unwrap();
        assert_eq!(
            (crate::sha256_hex(&bytes), budget.used()),
            (digest.to_string(), used),
            "{limit}"
        );
    }
}

/// Review 01 of the wiring PR (B17-R01-COUNTER-OVERFLOW): a sidecar's surrogate counts are
/// untrusted, so extreme `accepted` and `jacobians` values are refused as a disagreement, never by
/// an overflow panic, whatever the build's overflow checks.
#[test]
fn extreme_surrogate_counts_are_refused_without_overflow() {
    fn surrogate(t: &mut RefinementTrace) -> &mut SurrogateTrace {
        match &mut t.stages[1] {
            StageTrace::Surrogate { surrogate, .. } => surrogate,
            _ => unreachable!(),
        }
    }
    for (accepted, jacobians) in [
        (usize::MAX, usize::MAX),
        (usize::MAX - 1, usize::MAX),
        (usize::MAX, usize::MAX - 1),
        (0, usize::MAX),
        (usize::MAX, 0),
    ] {
        let mut trace = synthetic_trace();
        let s = surrogate(&mut trace);
        s.accepted = accepted;
        s.jacobians = jacobians;
        let charged = trace.charged("sites");
        assert!(
            charged.is_err(),
            "accepted {accepted}, jacobians {jacobians}: {charged:?}"
        );
    }
}

/// Review 02 of the wiring PR (B17-R02-UNBOUNDED-PARENT-PARSE): a refinement parent read back from
/// evidence is untrusted, so the trace check parses only the pipeline form and counts each
/// expression source's nesting bytes before the expression parser sees it. Unary plus and
/// redundant parentheses never reach the parsed tree, so a source at the limit binds exactly as
/// its plain form does; one byte more is refused unparsed.
#[test]
fn an_untrusted_parent_expression_is_bounded_before_it_is_parsed() {
    use super::refinement::{MAX_PARENT_EXPRESSION_NESTING, SMALL_ALLOWANCE};
    let trace = RefinementTrace {
        method: DAMPED_LS.into(),
        stages: vec![StageTrace::Coordinate {
            stage: 0,
            reason: SMALL_ALLOWANCE.into(),
            evaluations: 5,
            rejected_trials: 0,
        }],
    };
    let parent = |source: &str| {
        format!("(pipe X (derive expr=\"{source}\" append=false) (ridge lambda_frac=0.000000001))")
    };
    let body = "sin(0.5*c0)";
    assert_eq!(parent(body), SINE);
    assert_eq!(trace.check(SINE, "sites", 60).unwrap(), (5, 0));
    // `(` and `*`.
    let room = MAX_PARENT_EXPRESSION_NESTING - 2;
    let plus = |n: usize| format!("{}{body}", "+".repeat(n));
    let grouped = |n: usize| format!("{}{body}{}", "(".repeat(n), ")".repeat(n));
    let product = |n: usize| format!("{body}{}", "*1".repeat(n));
    for source in [plus(room), grouped(room)] {
        assert_eq!(
            trace.check(&parent(&source), "sites", 60).unwrap(),
            (5, 0),
            "{source}"
        );
    }
    for source in [plus(room + 1), grouped(room + 1), product(room + 1)] {
        let error = trace.check(&parent(&source), "sites", 60).unwrap_err();
        assert!(
            format!("{error:#}").contains("nests too deeply"),
            "{source}: {error:#}"
        );
    }
    let closed_form = format!("(sym raw \"{}\")", plus(room + 1));
    let error = trace.check(&closed_form, "sites", 60).unwrap_err();
    assert!(format!("{error:#}").contains("not a pipeline"), "{error:#}");
}
