use super::*;

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
            simplify: false,
            linearize: false,
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
            simplify: false,
            linearize: false,
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
