use super::*;

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
