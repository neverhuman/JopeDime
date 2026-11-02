use super::*;

#[test]
fn queue_policy_is_fixed_while_p95_remains_observable() {
    assert_eq!(
        dynamic_queue_targets(&[], 364).unwrap(),
        QueueTargets {
            target: 2_048,
            hard_cap: 4_096,
            p95_jobs_per_minute: 0.0,
        }
    );
    assert_eq!(
        dynamic_queue_targets(&[1_000.0], 1).unwrap(),
        QueueTargets {
            target: 2_048,
            hard_cap: 4_096,
            p95_jobs_per_minute: 1_000.0,
        }
    );
    assert_eq!(dynamic_queue_targets(&[], 2_000).unwrap().target, 2_500);
    assert_eq!(dynamic_queue_targets(&[], 4_000).unwrap().target, 4_096);
    let values = (1..=20).map(f64::from).collect::<Vec<_>>();
    assert_eq!(
        dynamic_queue_targets(&values, 0)
            .unwrap()
            .p95_jobs_per_minute,
        19.0
    );
    assert!(dynamic_queue_targets(&[f64::NAN], 1).is_err());
    assert!(dynamic_queue_targets(&[], -1).is_err());
}

#[test]
fn refill_stops_at_target_and_rejects_inverted_cap() {
    assert_eq!(queue_refill_needed(200, 256, 512).unwrap(), 56);
    assert_eq!(queue_refill_needed(256, 256, 512).unwrap(), 0);
    assert!(queue_refill_needed(0, 513, 512).is_err());
}

#[test]
fn family_split_is_deterministic_and_never_splits_a_family() {
    let families = BTreeMap::from([
        ("a".to_owned(), "shared".to_owned()),
        ("b".to_owned(), "shared".to_owned()),
        ("c".to_owned(), "other-c".to_owned()),
        ("d".to_owned(), "other-d".to_owned()),
    ]);
    let first = family_disjoint_split(&families, 0.25, 20260726).unwrap();
    let second = family_disjoint_split(&families, 0.25, 20260726).unwrap();
    assert_eq!(first, second);
    assert_eq!(first["a"], first["b"]);
    assert!(first.values().any(|split| split == "train"));
    assert!(first.values().any(|split| split == "validation"));
}

#[test]
fn family_split_rejects_impossible_topology() {
    let one = BTreeMap::from([("a".to_owned(), "only".to_owned())]);
    assert!(family_disjoint_split(&one, 0.1, 1).is_err());
    let same = BTreeMap::from([
        ("a".to_owned(), "only".to_owned()),
        ("b".to_owned(), "only".to_owned()),
    ]);
    assert!(family_disjoint_split(&same, 0.1, 1).is_err());
}

#[test]
fn csv_parser_accepts_quotes_and_last_target() {
    let dataset = parse_last_target_csv("\"1\",2,0\r\n3,4,1\n").unwrap();
    assert_eq!(dataset.features, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    assert_eq!(dataset.targets, vec![0.0, 1.0]);
    assert!(parse_last_target_csv("x,1\n").is_err());
    assert!(parse_last_target_csv("1,NaN\n").is_err());
}

struct MeanEstimator {
    fitted: bool,
    mean: f64,
}

impl AdaptiveEstimator for MeanEstimator {
    fn fit(&mut self, _: &[Vec<f64>], targets: &[f64]) -> PolicyResult<()> {
        self.fitted = true;
        self.mean = targets.iter().sum::<f64>() / targets.len() as f64;
        Ok(())
    }

    fn predict(&self, features: &[Vec<f64>]) -> PolicyResult<Vec<f64>> {
        assert!(self.fitted);
        Ok(vec![self.mean; features.len()])
    }
}

#[test]
fn adaptive_evaluation_fits_train_and_scores_test_without_cv() {
    let mut estimator = MeanEstimator {
        fitted: false,
        mean: 0.0,
    };
    let fitness =
        evaluate_adaptive("regression", "0,1\n1,3\n", "2,2\n3,4\n", &mut estimator).unwrap();
    assert!(estimator.fitted);
    assert_eq!(fitness.fit_split, "train.csv");
    assert_eq!(fitness.fitness_split, "test.csv");
    assert_eq!(fitness.metric, "r2");
    assert_eq!(fitness.secondary_metric, "rmse");
    assert!(!fitness.cross_validation_fitness);
    assert!(fitness.test_adaptive);
    assert!(!fitness.benchmark_valid);
}

#[test]
fn binary_metrics_handle_ties_and_clamp_log_loss() {
    assert_eq!(
        auc_score(&[0.0, 1.0, 0.0, 1.0], &[0.1, 0.9, 0.5, 0.5]).unwrap(),
        0.875
    );
    assert!(log_loss(&[0.0, 1.0], &[-10.0, 10.0]).unwrap().is_finite());
    assert!(auc_score(&[1.0, 1.0], &[0.1, 0.2]).unwrap().is_nan());
}

#[test]
fn ranking_uses_score_binary_log_loss_and_semantic_order() {
    let rows = vec![
        RankingObservation {
            task: "binary".into(),
            dataset_hash: "d".into(),
            semantic_sha256: "b".into(),
            score: Some(0.8),
            secondary_score: Some(0.4),
            valid: true,
        },
        RankingObservation {
            task: "binary".into(),
            dataset_hash: "d".into(),
            semantic_sha256: "a".into(),
            score: Some(0.8),
            secondary_score: Some(0.3),
            valid: true,
        },
        RankingObservation {
            task: "binary".into(),
            dataset_hash: "d".into(),
            semantic_sha256: "invalid".into(),
            score: None,
            secondary_score: None,
            valid: false,
        },
    ];
    let incumbents = BTreeMap::from([(("binary".into(), "d".into()), 0.95)]);
    let ranked = attach_ranking_labels(&rows, Some(&incumbents));
    assert_eq!(ranked[0].observation.semantic_sha256, "a");
    assert_eq!(ranked[0].within_dataset_rank, Some(1));
    assert_eq!(ranked[0].within_dataset_percentile, Some(1.0));
    assert!((ranked[0].regret_to_incumbent.unwrap() - 0.15).abs() < f64::EPSILON);
    assert_eq!(ranked[2].within_dataset_rank, None);
}

#[test]
fn compile_cache_key_is_canonical_and_dataset_specific() {
    let key = compiled_cache_key(
        "regression",
        &"1".repeat(64),
        &"2".repeat(64),
        &"3".repeat(64),
        &"4".repeat(64),
        "x86_64-unknown-linux-gnu",
    )
    .unwrap();
    let changed = compiled_cache_key(
        "regression",
        &"5".repeat(64),
        &"2".repeat(64),
        &"3".repeat(64),
        &"4".repeat(64),
        "x86_64-unknown-linux-gnu",
    )
    .unwrap();
    assert_eq!(key.len(), 64);
    assert_ne!(key, changed);
    assert!(compiled_cache_key(
        "regression",
        "BAD",
        &"2".repeat(64),
        &"3".repeat(64),
        &"4".repeat(64),
        "target"
    )
    .is_err());
}

#[test]
fn family_panel_takes_one_per_family_before_remainders() {
    let rows = (0..32)
        .flat_map(|family| {
            [0, 1].map(move |member| PanelDataset {
                dataset_hash: format!("d-{family:02}-{member}"),
                source_family: format!("f-{family:02}"),
            })
        })
        .collect::<Vec<_>>();
    let panel = family_diverse_panel(&rows, 32, 20260726).unwrap();
    assert_eq!(
        panel
            .iter()
            .map(|row| &row.source_family)
            .collect::<BTreeSet<_>>()
            .len(),
        32
    );
    assert!(family_diverse_panel(&rows[..3], 32, 1).is_err());
}

#[test]
fn compile_finalists_are_bounded_at_two_per_dataset() {
    let rows = (0..4)
        .map(|index| CompileCandidate {
            task: "regression".into(),
            dataset_hash: "d".into(),
            semantic_sha256: format!("{index}"),
            score: index as f64 / 10.0,
            valid: true,
        })
        .collect::<Vec<_>>();
    let incumbents = BTreeMap::from([(("regression".into(), "d".into()), 0.15)]);
    let finalists = compiled_finalists(&rows, &incumbents);
    assert_eq!(finalists.len(), 2);
    assert_eq!(finalists[0].score, 0.3);
    assert_eq!(finalists[1].score, 0.2);
}

#[test]
fn parity_is_bit_exact_including_signed_zero() {
    assert!(exact_parity(&[1.0], &[1.0], 0.5, 0.5).qualified);
    let mismatch = exact_parity(&[-0.0], &[0.0], 0.5, 0.5);
    assert!(!mismatch.prediction_bit_parity);
    assert!(!mismatch.qualified);
}

#[test]
fn slot_targets_floor_multiplier_and_capacity_is_explicit() {
    assert_eq!(node_slot_target(20, 2.0).unwrap(), 40);
    assert_eq!(node_slot_target(3, 1.2).unwrap(), 3);
    let capacity = node_capacity("xbabe4", 24, 20, 13, 2.0).unwrap();
    assert_eq!(capacity.slot_target, 40);
    assert_eq!(capacity.physical_cores, 24);
    assert!(node_capacity("xbabe4", 19, 20, 0, 2.0).is_err());
}

#[test]
fn pressure_thresholds_are_inclusive_except_failure_delta() {
    let baseline = PressureBaseline {
        failure_rate: 0.1,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    };
    let sample = PressureSample {
        cpu_psi: 0.25,
        memory_psi: 0.05,
        io_psi: 0.10,
        available_memory_fraction: Some(0.149),
        failure_rate: 0.1026,
        p95_runtime_seconds: 15.1,
        throughput_per_slot: 4.0,
    };
    assert_eq!(
        unsafe_sample(&sample, &baseline).unwrap(),
        vec![
            "cpu_psi",
            "memory_psi",
            "io_psi",
            "available_memory",
            "failure_rate_regression",
            "runtime_and_throughput_regression",
        ]
    );
}

#[test]
fn ramp_advances_after_ten_minutes_and_rolls_back_after_three_samples() {
    let baseline = PressureBaseline {
        failure_rate: 0.01,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    };
    let state = RampState::baseline(20, baseline).unwrap();
    let state = ramp_decision(&state, &[], 1).unwrap();
    assert_eq!(state.stage, RampStage::Ramp30);
    assert_eq!(state.slot_target, 30);
    let state = ramp_decision(&state, &[], 600_001).unwrap();
    assert_eq!(state.stage, RampStage::Target40);
    assert_eq!(state.slot_target, 40);
    let unsafe_samples = vec![
        PressureSample {
            cpu_psi: 0.25,
            ..PressureSample::default()
        };
        3
    ];
    let state = ramp_decision(&state, &unsafe_samples, 700_000).unwrap();
    assert_eq!(state.stage, RampStage::RolledBack);
    assert_eq!(state.slot_target, 20);
    assert_eq!(state.rollback_reasons, vec!["cpu_psi"]);
}

#[test]
fn xbabe2_gate_requires_fifteen_safe_underutilized_samples_and_fresh_work() {
    let samples = vec![
        BurnSample {
            cluster_cpu_fraction: 0.74,
            ..BurnSample::default()
        };
        15
    ];
    assert!(xbabe2_burn_eligible(&samples, 1).unwrap());
    assert!(!xbabe2_burn_eligible(&samples[..14], 1).unwrap());
    assert!(!xbabe2_burn_eligible(&samples, 0).unwrap());
    let busy = vec![
        BurnSample {
            cluster_cpu_fraction: 0.75,
            ..BurnSample::default()
        };
        15
    ];
    assert!(!xbabe2_burn_eligible(&busy, 1).unwrap());
}
