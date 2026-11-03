use super::*;

#[test]
fn numeric_csv_accepts_one_header_and_rejects_later_malformed_rows() {
    let data = read_numeric_csv("feature,target\n1.0,2.0\n3.0,4.0\n", "fixture").unwrap();
    assert_eq!(data.x, vec![vec![1.0], vec![3.0]]);
    assert_eq!(data.y, vec![2.0, 4.0]);
    assert!(read_numeric_csv("feature,target\n1.0,nope\n", "fixture").is_err());
}

#[test]
fn metric_maps_agree_with_task_authority() {
    use jope_dime_regressor::task::TaskKind;
    for kind in [
        TaskKind::Regression,
        TaskKind::Binary,
        TaskKind::Timeseries,
        TaskKind::Multiclass,
    ] {
        assert_eq!(task_metric(kind.as_str()), Some(kind.primary_metric()));
    }
    assert_eq!(task_metric("multi_label"), None);
    assert_eq!(secondary_metric("multiclass"), "accuracy");
    assert_eq!(secondary_metric("timeseries"), "mase");
    assert_eq!(secondary_metric("binary"), "logloss");
    assert_eq!(secondary_metric("regression"), "rmse");
}

#[cfg(unix)]
#[test]
fn repeated_timeout_cleanup_leaves_no_process_group() {
    for _ in 0..50 {
        let mut command = Command::new("sh");
        command.args(["-c", "sleep 30 & wait"]);
        // SAFETY: the closure only calls setsid and reads errno via last_os_error; both are
        // async-signal-safe and neither allocates nor locks in the forked child.
        unsafe {
            command.pre_exec(|| {
                if libc::setsid() < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let mut child = command.spawn().expect("spawn timeout fixture");
        let pid = child.id() as libc::pid_t;
        terminate_process_group(&mut child, Duration::from_millis(50))
            .expect("terminate and reap timeout fixture");
        assert!(!process_group_exists(pid).expect("probe timeout fixture group"));
    }
}

#[test]
fn multiclass_decoding_preserves_declared_missing_classes_and_noise() {
    let meta = serde_json::json!({
        "target_scaling": {"classes": ["a", "b", "c", "d"]}
    });
    let train = [0.0 + 5e-7, 0.0, 2.0 / 3.0, 2.0 / 3.0 - 5e-7];
    let test = [1.0 / 3.0, 1.0];
    let decoded = decode_multiclass_splits(&train, &test, &meta).unwrap();
    assert_eq!(decoded.train, vec![0.0, 0.0, 1.0, 1.0]);
    assert_eq!(decoded.test, vec![1.0, 3.0]);
    assert_eq!(decoded.context.observed_declared, vec![0, 2]);
    assert_eq!(decoded.context.declared_classes, 4);
}

#[test]
fn multiclass_decoding_rejects_ambiguous_nonfinite_and_out_of_range_codes() {
    let ambiguous = serde_json::json!({
        "class_mapping": [
            {"code": 0, "scaled_code": 0.0},
            {"code": 1, "scaled_code": 0.0000015},
            {"code": 2, "scaled_code": 1.0}
        ]
    });
    assert!(
        decode_multiclass_splits(&[0.00000075, 1.0], &[0.0], &ambiguous)
            .unwrap_err()
            .contains("ambiguous")
    );
    let ordinary = serde_json::json!({"target_scaling": {"classes": [0, 1, 2]}});
    assert!(decode_multiclass_splits(&[0.0, f64::NAN], &[1.0], &ordinary).is_err());
    assert!(decode_multiclass_splits(&[0.0, 0.5], &[0.51], &ordinary).is_err());
}

#[test]
fn multiclass_decoding_supports_one_hundred_declared_classes() {
    let classes: Vec<usize> = (0..100).collect();
    let meta = serde_json::json!({"target_scaling": {"classes": classes}});
    let decoded = decode_multiclass_splits(&[0.0, 1.0], &[50.0 / 99.0], &meta).unwrap();
    assert_eq!(decoded.context.declared_classes, 100);
    assert_eq!(decoded.test, vec![50.0]);
}

#[test]
fn declared_macro_f1_assigns_zero_to_undefined_classes() {
    assert_eq!(declared_macro_f1(&[0.0, 0.0], &[0.0, 0.0], 4), 0.25);
}

#[test]
fn multiclass_evaluation_returns_aligned_probability_evidence() {
    let canonical = GenomeIr::parse(
        "(pipe X (input numeric=median) \
             (multiclass_ridge class_weight=balanced l2=0.02))",
    )
    .unwrap()
    .canonical();
    let candidate = ShardCandidate {
        semantic_sha256: sha256_hex(canonical.as_bytes()),
        canonical_genome: canonical,
    };
    let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0];
    let rows: Rows = labels
        .iter()
        .enumerate()
        .map(|(row, label)| {
            vec![
                f64::from(*label == 0.0) + row as f64 * 1e-3,
                f64::from(*label == 1.0),
                f64::from(*label == 2.0),
            ]
        })
        .collect();
    let table = raw_table(3, &rows).unwrap();
    let context = MulticlassContext {
        observed_declared: vec![0, 1, 2],
        declared_classes: 3,
    };
    let outcome = evaluate_multiclass(&candidate, &table, &labels, &table, &labels, 3, &context);
    let Outcome::Score {
        score,
        secondary,
        predictions,
        multiclass: Some(evidence),
        ..
    } = outcome
    else {
        panic!("multiclass fixture did not produce a score");
    };
    assert!(score.is_finite() && secondary.is_finite());
    assert_eq!(predictions.len(), labels.len());
    assert_eq!(evidence.classes, vec![0, 1, 2]);
    assert_eq!(evidence.probabilities.len(), labels.len());
    assert!(evidence
        .probabilities
        .iter()
        .all(|row| row.len() == 3 && (row.iter().sum::<f64>() - 1.0).abs() < 1e-12));
}

#[test]
fn multiclass_evaluation_accepts_singleton_and_pads_unseen_declared_class() {
    let canonical = GenomeIr::parse(
        "(pipe X (input numeric=median) \
             (multiclass_ridge class_weight=balanced l2=0.02))",
    )
    .unwrap()
    .canonical();
    let candidate = ShardCandidate {
        semantic_sha256: sha256_hex(canonical.as_bytes()),
        canonical_genome: canonical,
    };
    let train_labels = vec![0.0, 0.0, 1.0, 1.0, 2.0];
    let train_rows: Rows = train_labels
        .iter()
        .enumerate()
        .map(|(row, label)| vec![*label, row as f64 * 1e-3])
        .collect();
    let test_labels = vec![0.0, 1.0, 2.0, 3.0];
    let test_rows: Rows = test_labels
        .iter()
        .enumerate()
        .map(|(row, label)| vec![*label, row as f64 * 1e-3])
        .collect();
    let context = MulticlassContext {
        observed_declared: vec![0, 1, 2],
        declared_classes: 4,
    };
    let outcome = evaluate_multiclass(
        &candidate,
        &raw_table(2, &train_rows).unwrap(),
        &train_labels,
        &raw_table(2, &test_rows).unwrap(),
        &test_labels,
        2,
        &context,
    );
    let Outcome::Score {
        multiclass: Some(evidence),
        ..
    } = outcome
    else {
        panic!("singleton-class multiclass fixture did not produce a score");
    };
    assert!(evidence.probabilities.iter().all(|row| {
        row.len() == 4 && row[3] == 0.0 && (row.iter().sum::<f64>() - 1.0).abs() < 1e-12
    }));
}

fn shard(task: &str, authority: Option<LabelAuthority>) -> Shard {
    Shard {
        schema: SHARD_SCHEMA.to_string(),
        task: task.to_string(),
        dataset_hash: "dataset".to_string(),
        train_csv: "train.csv".to_string(),
        test_csv: "test.csv".to_string(),
        train_sha256: "0".repeat(64),
        test_sha256: "1".repeat(64),
        checkpoint_epoch: None,
        checkpoint_sha256: None,
        catalog_sha256: None,
        protocol_sha256: None,
        generated_at_ms: None,
        label_authority: authority,
        meta_json: None,
        meta_sha256: None,
        horizon: None,
        seasonality: None,
        frequency: None,
        target_is_final_column: None,
        known_future_covariates: Vec::new(),
        past_only_covariates: Vec::new(),
        reduction: None,
        candidates: Vec::new(),
    }
}

fn corrected(metric: &str) -> LabelAuthority {
    LabelAuthority {
        score_tier: "test_exact".to_string(),
        metric: metric.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: Some("test-exact-reused".to_string()),
        generation: "strict-wave-000001".to_string(),
        frozen_seed_sha256: "2".repeat(64),
        model_checkpoint_sha256: "3".repeat(64),
        hall_of_fame_sha256: "4".repeat(64),
    }
}

fn legacy(metric: &str) -> LabelAuthority {
    LabelAuthority {
        score_tier: "test_exact".to_string(),
        metric: metric.to_string(),
        test_adaptive: false,
        benchmark_valid: true,
        selection_bias: None,
        generation: "zero".to_string(),
        frozen_seed_sha256: "2".repeat(64),
        model_checkpoint_sha256: "3".repeat(64),
        hall_of_fame_sha256: "4".repeat(64),
    }
}

fn multiclass_authority() -> LabelAuthority {
    LabelAuthority {
        score_tier: "test".to_string(),
        metric: "macro_f1".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: Some("test-selected".to_string()),
        generation: "multiclass-generation-001".to_string(),
        frozen_seed_sha256: "2".repeat(64),
        model_checkpoint_sha256: "3".repeat(64),
        hall_of_fame_sha256: "4".repeat(64),
    }
}

fn timeseries_test_selected_authority() -> LabelAuthority {
    LabelAuthority {
        score_tier: "test_exact".to_string(),
        metric: "negative_mase".to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        selection_bias: Some("test-selected".to_string()),
        generation: "strict-wave-000001".to_string(),
        frozen_seed_sha256: "2".repeat(64),
        model_checkpoint_sha256: "3".repeat(64),
        hall_of_fame_sha256: "4".repeat(64),
    }
}

#[test]
fn legacy_shards_remain_adaptive() {
    let resolved = resolve_authority(&shard("regression", None)).unwrap();
    assert_eq!(resolved.tier, "test_exact");
    assert_eq!(resolved.metric, "r2");
    assert!(resolved.test_adaptive);
    assert!(!resolved.benchmark_valid);
    assert_eq!(resolved.generation, "adaptive");
}

#[test]
fn corrected_wave_discloses_test_adaptation() {
    let resolved = resolve_authority(&shard("binary", Some(corrected("auc")))).unwrap();
    assert_eq!(resolved.metric, "auc");
    assert!(resolved.test_adaptive);
    assert!(!resolved.benchmark_valid);
    assert_eq!(
        resolved.selection_bias.as_deref(),
        Some("test-exact-reused")
    );
    assert_eq!(resolved.generation, "strict-wave-000001");
    assert!(resolved.authority_sha256.is_some());
}

#[test]
fn multiclass_authority_is_explicitly_test_selected_and_fail_closed() {
    let resolved = resolve_authority(&shard("multiclass", Some(multiclass_authority()))).unwrap();
    assert_eq!(resolved.tier, "test");
    assert_eq!(resolved.metric, "macro_f1");
    assert_eq!(resolved.selection_bias.as_deref(), Some("test-selected"));
    assert_eq!(resolved.generation, "multiclass-generation-001");
    assert!(resolved.test_adaptive);
    assert!(!resolved.benchmark_valid);

    let mut wrong = multiclass_authority();
    wrong.selection_bias = Some("test-exact-reused".to_string());
    assert!(resolve_authority(&shard("multiclass", Some(wrong))).is_err());
}

#[test]
fn timeseries_authority_is_explicitly_test_selected_and_fail_closed() {
    let resolved = resolve_authority(&shard(
        "timeseries",
        Some(timeseries_test_selected_authority()),
    ))
    .unwrap();
    assert_eq!(resolved.tier, "test_exact");
    assert_eq!(resolved.metric, "negative_mase");
    assert_eq!(resolved.selection_bias.as_deref(), Some("test-selected"));
    assert_eq!(resolved.generation, "strict-wave-000001");
    assert!(resolved.test_adaptive);
    assert!(!resolved.benchmark_valid);

    let mut wrong = timeseries_test_selected_authority();
    wrong.metric = "mase".to_string();
    assert!(resolve_authority(&shard("timeseries", Some(wrong))).is_err());
}

#[test]
fn superseded_generation_zero_authority_remains_replayable() {
    let resolved = resolve_authority(&shard("regression", Some(legacy("r2")))).unwrap();
    assert!(!resolved.test_adaptive);
    assert!(resolved.benchmark_valid);
    assert_eq!(resolved.generation, "zero");
}

#[test]
fn missing_disclosure_or_wrong_metric_is_rejected() {
    let mut undisclosed = corrected("r2");
    undisclosed.selection_bias = None;
    assert!(resolve_authority(&shard("regression", Some(undisclosed))).is_err());
    assert!(resolve_authority(&shard("binary", Some(corrected("r2")))).is_err());
    assert!(resolve_authority(&shard("timeseries", Some(corrected("r2")))).is_err());
}

#[test]
fn strict_authority_rejects_unpinned_provenance() {
    let mut authority = corrected("r2");
    authority.frozen_seed_sha256 = "not-a-hash".to_string();
    assert!(resolve_authority(&shard("regression", Some(authority))).is_err());
}

#[test]
fn reduction_resolution_fails_closed_across_schema_versions() {
    use jope_dime_regressor::timeseries::{TS_REDUCTION_V1, TS_REDUCTION_V2};
    // v1 schema: no pin allowed, resolves to V1.
    let plain = shard("timeseries", None);
    assert_eq!(resolve_reduction(&plain).unwrap(), ReductionVersion::V1);
    let mut pinned_on_v1 = shard("timeseries", None);
    pinned_on_v1.reduction = Some(ReductionSpec {
        version: TS_REDUCTION_V2.to_string(),
    });
    assert!(resolve_reduction(&pinned_on_v1).is_err());
    // v2 schema: timeseries must pin; explicit v1 and v2 pins both resolve.
    let mut unpinned_v2 = shard("timeseries", None);
    unpinned_v2.schema = SHARD_SCHEMA_V2.to_string();
    assert!(resolve_reduction(&unpinned_v2).is_err());
    let mut pinned_v2 = unpinned_v2;
    pinned_v2.reduction = Some(ReductionSpec {
        version: TS_REDUCTION_V2.to_string(),
    });
    assert_eq!(resolve_reduction(&pinned_v2).unwrap(), ReductionVersion::V2);
    pinned_v2.reduction = Some(ReductionSpec {
        version: TS_REDUCTION_V1.to_string(),
    });
    assert_eq!(resolve_reduction(&pinned_v2).unwrap(), ReductionVersion::V1);
    pinned_v2.reduction = Some(ReductionSpec {
        version: "ts-reduction-v9".to_string(),
    });
    assert!(resolve_reduction(&pinned_v2).is_err());
    // non-timeseries tasks never carry a pin, on either schema.
    let mut regression_v2 = shard("regression", None);
    regression_v2.schema = SHARD_SCHEMA_V2.to_string();
    assert_eq!(
        resolve_reduction(&regression_v2).unwrap(),
        ReductionVersion::V1
    );
    regression_v2.reduction = Some(ReductionSpec {
        version: TS_REDUCTION_V2.to_string(),
    });
    assert!(resolve_reduction(&regression_v2).is_err());
    // unknown schema strings are rejected outright.
    let mut unknown = shard("timeseries", None);
    unknown.schema = "JopeFlywheelCorpusEvalShard.v3".to_string();
    assert!(resolve_reduction(&unknown).is_err());
}

#[test]
fn prediction_encoding_and_balanced_accuracy_are_deterministic() {
    let predictions = [0.1, 0.7, 0.8, 0.2];
    let first = prediction_bytes(&predictions);
    let second = prediction_bytes(&predictions);
    assert_eq!(first, second);
    assert_eq!(first.len(), predictions.len() * 8);
    assert_eq!(sha256_hex(&first), sha256_hex(&second));
    assert_eq!(balanced_accuracy(&[0.0, 0.0, 1.0, 1.0], &predictions), 0.5);
}

#[test]
fn binary_task_admission_rejects_regression_heads() {
    let task = TaskSpec::binary(Objective::RocAuc).unwrap();
    let regression = GenomeIr::parse("(pipe X (ridge))").unwrap();
    let binary = GenomeIr::parse("(pipe X (logit_ridge l2=0.1))").unwrap();
    assert!(validate_v3_genome(&regression, &task, &binary_schema(8)).is_err());
    assert!(validate_v3_genome(&binary, &task, &binary_schema(8)).is_ok());
}

#[test]
fn timeseries_authority_uses_negative_mase_and_stays_adaptive() {
    let resolved = resolve_authority(&shard("timeseries", None)).unwrap();
    assert_eq!(resolved.metric, "negative_mase");
    assert!(resolved.test_adaptive);
    assert!(!resolved.benchmark_valid);

    let corrected =
        resolve_authority(&shard("timeseries", Some(corrected("negative_mase")))).unwrap();
    assert_eq!(corrected.metric, "negative_mase");
    assert!(corrected.test_adaptive);
    assert!(!corrected.benchmark_valid);

    assert!(resolve_authority(&shard("timeseries", Some(legacy("negative_mase")))).is_err());
}

#[test]
fn timeseries_rejects_binary_classification_heads() {
    let binary = GenomeIr::parse("(pipe X (logit_ridge l2=0.1))").unwrap();
    let logistic = GenomeIr::parse("(pipe X (logistic))").unwrap();
    let regression = GenomeIr::parse("(pipe X (ridge))").unwrap();
    assert!(matches!(terminal_head(&binary), Some("logit_ridge")));
    assert!(matches!(terminal_head(&logistic), Some("logistic")));
    assert!(matches!(terminal_head(&regression), Some("ridge")));
}
