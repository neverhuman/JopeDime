use super::*;

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
