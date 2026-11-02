use super::*;

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

#[test]
fn the_lm_refined_fixture_program_is_pinned() {
    let c = sine_data();
    let (result, _, _) = refine(&lm(), SINE, &c, 60, 60);
    assert_eq!(result.refinement.program, PINNED_LM_PROGRAM);
    let parsed = GenomeIr::parse(PINNED_LM_PROGRAM).unwrap();
    assert_eq!(parsed.canonical(), PINNED_LM_PROGRAM);
}

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
