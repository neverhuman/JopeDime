use super::*;

#[test]
fn legacy_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(FaultPolicy::Legacy), frozen_runs());
}

#[test]
fn reject_hard_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(FaultPolicy::RejectHard), frozen_runs());
}

#[test]
fn penalize_reproduces_the_frozen_search_runs() {
    assert_eq!(seeded_runs(PENALIZE), frozen_runs());
}

#[test]
fn other_penalty_settings_reproduce_the_frozen_search_runs() {
    let frozen = frozen_runs();
    for policy in [penalize(0.0, 0.0, true), penalize(0.05, 0.01, false)] {
        assert_eq!(seeded_runs(policy), frozen, "{policy:?}");
    }
}

#[test]
fn an_out_of_scope_candidate_is_rejected_and_never_ranked() {
    let (x, y) = four_columns(120, 3);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let key = canonical(OUT_OF_SCOPE);
    let reason = "scope: stage 0: COLUMN_OOB: column index 12 out of bounds for width 4";

    // The surrogate search: generation 0 holds the champion, the seed and six starters.
    let config = |gens: usize, policy: FaultPolicy| EvolveConfig {
        pop: 8,
        gens,
        folds: 3,
        seed: 7,
        max_transforms: 3,
        seed_genomes: vec![OUT_OF_SCOPE.to_string()],
        fault_policy: policy,
        ..EvolveConfig::default()
    };
    let legacy = evolve(
        &x,
        &y,
        &metric,
        &config(1, FaultPolicy::Legacy),
        &Log::default(),
    );
    assert!(
        legacy.iter().any(|scored| scored.dsl == key),
        "Legacy scores the fallback zeros"
    );
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let log = Log::default();
        let ranked = evolve(&x, &y, &metric, &config(3, policy), &log);
        assert!(!ranked.is_empty());
        assert!(ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), vec![(key.clone(), reason.to_string())]);
        let generations = log.generations();
        assert_eq!(generations.len(), 3);
        assert!(generations
            .iter()
            .all(|counters| counters.hard_rejected == 1));
        assert_eq!(
            generations[0].hard_rejected_share,
            (1.0_f64 / 8.0).to_bits()
        );
    }

    // The lite race, from an exact generation 0.
    let initial = [
        OUT_OF_SCOPE,
        "(pipe X (standardize) (ridge))",
        "(pipe X (rank) (ridge))",
        "(pipe X (derive expr=\"col(3)\") (ridge))",
    ]
    .map(str::to_string);
    let race = |policy: FaultPolicy, log: &Log| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(4, 1, 11, policy),
            log,
            &initial,
        )
        .unwrap()
    };
    let legacy = race(FaultPolicy::Legacy, &Log::default());
    assert!(scores_of(&legacy, &key)[0].is_some());
    assert_eq!(legacy.ranked.len(), 4);
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let log = Log::default();
        let outcome = race(policy, &log);
        assert_eq!(scores_of(&outcome, &key), [None, None, None]);
        assert_eq!(outcome.ranked.len(), 3);
        assert!(outcome.ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), vec![(key.clone(), reason.to_string())]);
        assert_eq!(log.generations()[0].hard_rejected, 1);
        assert_eq!(
            log.generations()[0].hard_rejected_share,
            (1.0_f64 / 4.0).to_bits()
        );
    }
}

/// The scope check sees the program as the search fits it, not as it was generated: the search
/// caps `rff_cos n=128` at 64 columns, so `col(100)` is in scope for the generated program and out
/// of scope for the fitted one. Both searches reject the candidate with the fitted program's
/// reason, which names width 64. A check of the generated program would have passed it, and the
/// fit would have met `col(100)` as a Hard fault instead.
#[test]
fn the_scope_check_sees_the_program_as_the_search_fits_it() {
    let (x, y) = four_columns(120, 3);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let key = canonical(NARROWED);
    let reason = "stage 1: COLUMN_OOB: column index 100 out of bounds for width 64";
    assert!(validate_scope(&GenomeIr::parse(NARROWED).unwrap(), 4).is_ok());
    // The rung-0 program, as the search fits it: out of scope, and a traced fit of it meets the
    // Hard fault that the check spares the search.
    let (surrogate, _) = rung0_trace_identity(NARROWED).unwrap();
    let fitted = GenomeIr::parse(&surrogate).unwrap();
    assert_eq!(validate_scope(&fitted, 4).unwrap_err().to_string(), reason);
    let (_, training) = fit_traced(&fitted, &x, &y).unwrap();
    let mut totals = FaultTotals::default();
    totals.add_pass(&training);
    assert_eq!(
        Rejection::HardFault(totals.hard_classes).to_string(),
        "hard fault: {COLUMN_OOB}"
    );

    let rejection = vec![(key.clone(), format!("scope: {reason}"))];
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        // The surrogate search: generation 0 holds the champion, the program and six starters.
        let log = Log::default();
        let config = EvolveConfig {
            pop: 8,
            gens: 1,
            folds: 3,
            seed: 7,
            max_transforms: 3,
            seed_genomes: vec![NARROWED.to_string()],
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let ranked = evolve(&x, &y, &metric, &config, &log);
        assert!(ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), rejection, "{policy:?}");

        // The lite race, from an exact generation 0.
        let log = Log::default();
        let initial = [NARROWED, "(pipe X (standardize) (ridge))"].map(str::to_string);
        let outcome = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 11, policy),
            &log,
            &initial,
        )
        .unwrap();
        assert_eq!(scores_of(&outcome, &key), [None, None, None]);
        assert!(outcome.ranked.iter().all(|scored| scored.dsl != key));
        assert_eq!(log.rejections(), rejection, "{policy:?}");
    }
}

#[test]
fn penalize_ranks_the_fault_free_program_first_on_equal_loss() {
    let (x, y) = equal_loss_data(160);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let (log_key, log1p_key) = (canonical(LOG_NEGATIVE), canonical(LOG1P_ABS));
    let initial = [LOG_NEGATIVE.to_string(), LOG1P_ABS.to_string()];
    let race = |policy: FaultPolicy| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 17, policy),
            &Log::default(),
            &initial,
        )
        .unwrap()
    };
    let order = |policy: FaultPolicy| -> Vec<String> {
        race(policy)
            .ranked
            .into_iter()
            .map(|scored| scored.dsl)
            .collect()
    };
    let legacy_order = vec![log_key.clone(), log1p_key.clone()];
    let fault_free_first = vec![log1p_key.clone(), log_key.clone()];
    // Legacy keeps the tie in generation order.
    assert_eq!(order(FaultPolicy::Legacy), legacy_order);
    // The penalty alone decides: the faulting program loses 0.05 times its domain rate of 1.
    assert_eq!(order(PENALIZE), fault_free_first);
    assert_eq!(order(penalize(0.05, 0.0, false)), fault_free_first);
    // At zero weights the scores stay tied exactly, and only the tie-break reorders them.
    assert_eq!(order(penalize(0.0, 0.0, true)), fault_free_first);
    assert_eq!(order(penalize(0.0, 0.0, false)), legacy_order);

    // The scores themselves: every rung-0 fold of the faulting program is 0 - 0.05 * 1.
    let outcome = race(PENALIZE);
    let folds = RACE_RUNG_FOLDS[0];
    let penalized: f64 = 0.0 - 0.05 * 1.0;
    assert_eq!(
        scores_of(&outcome, &log_key)[0].as_deref().map(bits),
        Some(vec![penalized.to_bits(); folds])
    );
    assert_eq!(
        scores_of(&outcome, &log1p_key)[0].as_deref().map(bits),
        Some(vec![0.0_f64.to_bits(); folds])
    );

    // The surrogate search: its champion scores 0 here as well.
    let positions = |policy: FaultPolicy| {
        let config = EvolveConfig {
            pop: 3,
            gens: 1,
            folds: 3,
            seed: 17,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let ranked = evolve(&x, &y, &metric, &config, &Log::default());
        let position = |key: &str| {
            ranked
                .iter()
                .position(|scored| scored.dsl == key)
                .unwrap_or_else(|| panic!("{key} is not ranked"))
        };
        (position(&log_key), position(&log1p_key))
    };
    let (log_at, log1p_at) = positions(FaultPolicy::Legacy);
    assert!(log_at < log1p_at);
    for policy in [PENALIZE, penalize(0.0, 0.0, true)] {
        let (log_at, log1p_at) = positions(policy);
        assert!(log1p_at < log_at, "{policy:?}");
    }
    let (log_at, log1p_at) = positions(penalize(0.0, 0.0, false));
    assert!(log_at < log1p_at);
}

/// The domain-rate counters' values in runs whose ranking is known. On `equal_loss_data` every
/// program scores 0 (the R² of a constant target), `LOG_NEGATIVE`'s domain rate is 1 at every
/// rung, and no other program has an expression fault: the mean over the ranked candidates is one
/// over their number, and the mean over the elites counts `LOG_NEGATIVE` only when it is an elite.
#[test]
fn the_domain_rate_counters_follow_the_ranking_and_the_elites() {
    let (x, y) = equal_loss_data(160);
    let metric = |truth: &[f64], predicted: &[f64]| feat_math::r2(truth, predicted);
    let (log_key, log1p_key) = (canonical(LOG_NEGATIVE), canonical(LOG1P_ABS));
    let initial = [LOG_NEGATIVE.to_string(), LOG1P_ABS.to_string()];
    let rates = |counters: &Counters| (counters.pop_domain_rate, counters.elite_domain_rate);

    // The race: two candidates and one elite, the first ranked.
    let race = |policy: FaultPolicy, log: &Log| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &metric,
            &race_config(2, 1, 17, policy),
            log,
            &initial,
        )
        .unwrap()
    };
    let log = Log::default();
    race(FaultPolicy::Legacy, &log);
    log.assert_nothing_reported();
    for (policy, first, elite_rate) in [
        (FaultPolicy::RejectHard, &log_key, 1.0),
        (penalize(0.0, 0.0, false), &log_key, 1.0),
        (PENALIZE, &log1p_key, 0.0),
        (penalize(0.0, 0.0, true), &log1p_key, 0.0),
    ] {
        let log = Log::default();
        let outcome = race(policy, &log);
        let order: Vec<&str> = outcome
            .ranked
            .iter()
            .map(|scored| scored.dsl.as_str())
            .collect();
        assert_eq!(order.len(), 2, "{policy:?}");
        assert_eq!(order[0], first.as_str(), "{policy:?}");
        let generations = log.generations();
        let last = generations.last().expect("a ranked generation");
        assert_eq!(
            rates(last),
            (0.5_f64.to_bits(), f64::to_bits(elite_rate)),
            "{policy:?}"
        );
        assert_eq!(
            (last.hard_rejected, last.hard_rejected_share),
            (0, 0.0_f64.to_bits())
        );
    }

    // The surrogate search: sixteen candidates and two elites. Generation 0 is the champion, the
    // two programs, the eight starters and random programs, and the ranking keeps that order among
    // equal scores, so the elites are the champion and `LOG_NEGATIVE` unless the penalty or the
    // tie-break moves `LOG_NEGATIVE` down.
    let surrogate = |policy: FaultPolicy| {
        let config = EvolveConfig {
            pop: 16,
            gens: 1,
            folds: 3,
            seed: 17,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let ranked = evolve(&x, &y, &metric, &config, &log);
        (ranked, log.generations())
    };
    for (policy, elite_rate) in [
        (FaultPolicy::RejectHard, 0.5),
        (PENALIZE, 0.0),
        (penalize(0.0, 0.0, true), 0.0),
    ] {
        let (ranked, generations) = surrogate(policy);
        let log_at = ranked
            .iter()
            .position(|scored| scored.dsl == log_key)
            .expect("LOG_NEGATIVE is ranked");
        assert_eq!(log_at < 2, elite_rate > 0.0, "{policy:?}");
        assert_eq!(generations.len(), 1);
        assert_eq!(generations[0].ranked, ranked.len());
        assert_eq!(
            rates(&generations[0]),
            (
                (1.0 / ranked.len() as f64).to_bits(),
                f64::to_bits(elite_rate)
            ),
            "{policy:?}"
        );
    }
}

#[test]
fn fault_free_programs_score_as_under_legacy() {
    let (x, y) = toy_data(160, 5, 11);
    let initial = FAULT_FREE.map(str::to_string);
    // Generation 0 only: later generations may hold offspring that fail the scope check.
    let race = |policy: FaultPolicy| {
        let log = Log::default();
        let outcome = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(3, 1, 19, policy),
            &log,
            &initial,
        )
        .unwrap();
        log.assert_nothing_reported();
        race_run(&outcome)
    };
    let surrogate = |policy: FaultPolicy| {
        // Generation 0 is the champion and the three programs.
        let config = EvolveConfig {
            pop: 4,
            gens: 1,
            folds: 3,
            seed: 19,
            seed_genomes: initial.to_vec(),
            fault_policy: policy,
            ..EvolveConfig::default()
        };
        let log = Log::default();
        let ranked = evolve(&x, &y, &feat_math::r2, &config, &log);
        log.assert_nothing_reported();
        ranked.iter().map(ranked_line).collect::<Vec<_>>()
    };
    let (legacy_race, legacy_surrogate) =
        (race(FaultPolicy::Legacy), surrogate(FaultPolicy::Legacy));
    assert_eq!(legacy_race.ranked.len(), 3);
    for policy in [
        FaultPolicy::RejectHard,
        PENALIZE,
        penalize(0.05, 0.01, true),
    ] {
        assert_eq!(race(policy), legacy_race, "{policy:?}");
        assert_eq!(surrogate(policy), legacy_surrogate, "{policy:?}");
    }
}

#[test]
fn the_penalty_is_the_documented_formula_on_the_published_rates() {
    let (x, y) = toy_data(160, 5, 13);
    let key = canonical(PARTIAL_FAULTS);
    let initial = [PARTIAL_FAULTS.to_string()];
    let race = |policy: FaultPolicy| {
        evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(1, 1, 23, policy),
            &Log::default(),
            &initial,
        )
        .unwrap()
    };
    let legacy = race(FaultPolicy::Legacy);
    let penalized = race(penalize(0.05, 0.02, false));

    // Replay rung 0 through the public telemetry: its program, rows and folds.
    let replay = replay_rung0(&legacy, &x, &y, &key).unwrap();
    let totals = replay.all;
    assert!(!totals.is_hard());
    assert!(
        totals.domain_cells > 0 && totals.saturation_cells > 0,
        "{totals:?}"
    );
    assert!(totals.domain_rate() < 1.0 && totals.saturation_rate() < 1.0);
    // The held-out passes count: without them the rates would differ.
    assert_ne!(
        (
            replay.training.domain_rate().to_bits(),
            replay.training.saturation_rate().to_bits()
        ),
        (
            totals.domain_rate().to_bits(),
            totals.saturation_rate().to_bits()
        ),
        "the training passes alone give the same rates"
    );

    let legacy_folds = scores_of(&legacy, &key)[0].clone().unwrap();
    assert_eq!(
        bits(&legacy_folds),
        bits(&replay.scores),
        "the replay is the rung-0 evaluation"
    );
    let by_hand: Vec<f64> = legacy_folds
        .iter()
        .map(|score| score - 0.05 * totals.domain_rate() - 0.02 * totals.saturation_rate())
        .collect();
    let formula: Vec<f64> = legacy_folds
        .iter()
        .map(|&score| totals.penalize(score, 0.05, 0.02).unwrap())
        .collect();
    assert_eq!(bits(&formula), bits(&by_hand));
    assert_eq!(
        scores_of(&penalized, &key)[0].as_deref().map(bits),
        Some(bits(&by_hand))
    );
}
