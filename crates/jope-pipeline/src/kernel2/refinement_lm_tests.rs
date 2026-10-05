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
            })
        })
        .is_err());
    // Round trip, and refusal of anything the selection would not write back.
    for jacobian in [LmJacobianCharge::Sites, LmJacobianCharge::One] {
        let selection = EngineSelection {
            refine: RefineMethod::Lm(LmOptions {
                jacobian,
                ..LmOptions::default()
            }),
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
