use super::*;

#[test]
fn non_legacy_policies_never_call_a_gpu_backend() {
    let (x, y) = toy_data(160, 5, 42);
    let backend = |_: Vec<GpuRidgeScoreProblem>| -> Result<GpuRidgeScoreBatch, String> {
        panic!("a non-legacy fault policy called the GPU backend")
    };
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let config = race_config(16, 2, 41, policy);
        let cpu = evolve_lite_race(&x, &y, &feat_math::r2, &config, &Log::default());
        let routed = evolve_lite_race_with_gpu_policy(
            &x,
            &y,
            &feat_math::r2,
            &config,
            &Log::default(),
            Some(&backend),
            GpuRidgeScorePolicy::new(1, 1),
        );
        assert_eq!(race_run(&routed), race_run(&cpu));
        assert_eq!(routed.gpu_screen_timings.gpu_folds, 0);
        assert!(routed
            .candidates
            .iter()
            .filter_map(|candidate| candidate.r0_fold_routes.as_ref())
            .flatten()
            .all(|receipt| receipt.route == Rung0FoldRoute::CpuExact));
    }
}

#[test]
fn exact_reuse_stays_off_under_a_non_legacy_policy() {
    let (x, y) = toy_data(160, 5, 42);
    let reuse = Rung0ExactSurrogateDedupPolicy::from_authority_receipt(
        RUNG0_EXACT_SURROGATE_DEDUP_AUTHORITY_SHA256,
        &"ab".repeat(32),
    )
    .unwrap();
    let requested = |policy: FaultPolicy| {
        evolve_lite_race_with_gpu_policy_and_exact_surrogate_dedup(
            &x,
            &y,
            &feat_math::r2,
            &race_config(16, 2, 41, policy),
            &Log::default(),
            None,
            None,
            Some(reuse),
        )
    };
    assert!(
        requested(FaultPolicy::Legacy)
            .rung0_exact_surrogate_dedup
            .active,
        "under Legacy the request turns reuse on"
    );
    for policy in [FaultPolicy::RejectHard, PENALIZE] {
        let outcome = requested(policy);
        let receipt = &outcome.rung0_exact_surrogate_dedup;
        assert!(receipt.requested && !receipt.active);
        assert!(receipt
            .activation_blocker
            .as_deref()
            .is_some_and(|blocker| blocker.contains("non-legacy fault policy")));
        assert_eq!(
            (
                receipt.logical_candidates,
                receipt.physical_candidate_evaluations,
                receipt.reused_candidates
            ),
            (0, 0, 0)
        );
        let plain = evolve_lite_race(
            &x,
            &y,
            &feat_math::r2,
            &race_config(16, 2, 41, policy),
            &Log::default(),
        );
        assert_eq!(race_run(&outcome), race_run(&plain));
    }
}

#[test]
fn penalty_weights_must_be_finite_and_non_negative() {
    let (x, y) = toy_data(32, 5, 1);
    for (lambda_domain, lambda_saturation) in [
        (f64::NAN, 0.0),
        (0.0, f64::NAN),
        (-0.5, 0.0),
        (0.0, -1e-9),
        (f64::INFINITY, 0.0),
        (0.0, f64::NEG_INFINITY),
    ] {
        let policy = penalize(lambda_domain, lambda_saturation, true);
        let error = policy.validate().unwrap_err();
        assert!(error.contains("must be finite and non-negative"), "{error}");
        let refused = evolve_lite_race_with_exact_initial_population(
            &x,
            &y,
            &feat_math::r2,
            &race_config(1, 1, 3, policy),
            &Log::default(),
            &["(pipe X (ridge))".to_string()],
        )
        .unwrap_err();
        assert_eq!(refused, error);
    }
    for (lambda_domain, lambda_saturation) in [(0.0, 0.0), (0.05, 0.0), (1e3, 2.5)] {
        assert_eq!(
            penalize(lambda_domain, lambda_saturation, false).validate(),
            Ok(())
        );
    }
    assert_eq!(FaultPolicy::Legacy.validate(), Ok(()));
    assert_eq!(FaultPolicy::RejectHard.validate(), Ok(()));
    assert_eq!(FaultPolicy::default(), FaultPolicy::Legacy);
    assert_eq!(EvolveConfig::default().fault_policy, FaultPolicy::Legacy);
}

#[test]
#[should_panic(expected = "lambda_domain must be finite and non-negative")]
fn the_surrogate_search_refuses_an_invalid_penalty() {
    let (x, y) = toy_data(32, 5, 1);
    let config = EvolveConfig {
        pop: 2,
        gens: 1,
        fault_policy: penalize(f64::NAN, 0.0, false),
        ..EvolveConfig::default()
    };
    evolve(&x, &y, &feat_math::r2, &config, &Log::default());
}

#[test]
#[should_panic(expected = "lambda_saturation must be finite and non-negative")]
fn the_race_refuses_an_invalid_penalty() {
    let (x, y) = toy_data(32, 5, 1);
    let config = race_config(2, 1, 3, penalize(0.05, -1.0, true));
    evolve_lite_race(&x, &y, &feat_math::r2, &config, &Log::default());
}

/// Everything the policy decides is a function of the data and the seed: the same searches in a
/// one-thread and a four-thread pool give the same candidates, scores, rejections and counters.
#[test]
fn outcomes_do_not_depend_on_the_thread_count() {
    let (x, y) = toy_data(160, 5, 31);
    let seeds: Vec<String> = [
        PARTIAL_FAULTS,
        "(pipe X (derive expr=\"col(12)\") (ridge))",
        "(pipe X (derive expr=\"sqrt(col(2)) + 1/col(3)\" append=true) (ridge))",
        FAULT_FREE[0],
        FAULT_FREE[1],
    ]
    .map(str::to_string)
    .to_vec();
    let policy = penalize(0.05, 0.01, true);
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let race_log = Log::default();
            let race = evolve_lite_race(
                &x,
                &y,
                &feat_math::r2,
                &EvolveConfig {
                    seed_genomes: seeds.clone(),
                    ..race_config(12, 3, 5, policy)
                },
                &race_log,
            );
            let surrogate_log = Log::default();
            let surrogate = evolve(
                &x,
                &y,
                &feat_math::r2,
                &EvolveConfig {
                    pop: 12,
                    gens: 3,
                    folds: 3,
                    seed: 5,
                    max_transforms: 3,
                    seed_genomes: seeds.clone(),
                    fault_policy: policy,
                    ..EvolveConfig::default()
                },
                &surrogate_log,
            );
            (
                race_run(&race),
                race_log.rejections(),
                race_log.generations(),
                surrogate.iter().map(ranked_line).collect::<Vec<_>>(),
                surrogate_log.rejections(),
                surrogate_log.generations(),
            )
        })
    };
    let one = run(1);
    assert!(
        !one.1.is_empty() && !one.4.is_empty(),
        "the searches must reject something"
    );
    assert!(
        one.2
            .iter()
            .any(|counters| counters.pop_domain_rate != 0.0_f64.to_bits()),
        "the searches must rank a faulting candidate"
    );
    assert_eq!(run(4), one);
}
