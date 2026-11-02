use super::*;

#[test]
fn sealed_config_covers_requested_contract() {
    let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
    config.validate().unwrap();
    assert!(config.development_only);
    assert!(!config.confirmatory_valid);
    assert_eq!(config.dataset_ids.len(), 10);
    assert_eq!(config.smart_v3.mutation_mix.values().sum::<f64>(), 1.0);
    assert_eq!(
        config.authoritative_baselines.prior_train_only.method_id,
        "jope-dime-regressor"
    );
    assert_eq!(
        config.authoritative_baselines.oracle_aspfm.method_id,
        "jope-dime"
    );
    let smoke = TrainOnlyRedteamConfig::sealed(
        "golden42-train-only-redteam-20260730-v1-smoke",
        DEFAULT_PARENT_RUN,
        true,
    );
    smoke.validate().unwrap();
    assert_eq!(smoke.dataset_ids.len(), 10);
    assert_eq!(smoke.smart_v3.minimum_generations, 2);
    assert_eq!(smoke.smart_v3.maximum_generations, 2);
    let smart_cpu = TrainOnlyRedteamConfig::sealed(SMART_V3_CPU_RUN_ID, DEFAULT_PARENT_RUN, false);
    smart_cpu.validate().unwrap();
    assert_eq!(smart_cpu.scq.evaluation_cpus, 4);
    assert_eq!(smart_cpu.scq.evaluation_ram_gb, 16);
    assert!(smart_cpu.scq.gpu_proposal_nodes.is_empty());
    assert_eq!(
        smart_cpu.scq.evaluation_ownership["55f06d147bcc2a32"],
        "xbabe1"
    );
    assert_eq!(
        smart_cpu.scq.evaluation_ownership["ee56a1015e344c11"],
        "xbabe3"
    );
}

#[test]
fn fold_repeats_cover_every_row_once_and_are_separated() {
    let targets = (0..103)
        .map(|row| ((row * 37) % 101) as f64 / 101.0)
        .collect::<Vec<_>>();
    let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
    let bank = build_fold_bank(
        DEFAULT_RUN_ID,
        DATASETS[0],
        &"a".repeat(64),
        &targets,
        &config,
    )
    .unwrap();
    validate_fold_bank(&bank, &config).unwrap();
    let hashes = bank
        .search
        .iter()
        .chain(&bank.selector)
        .chain(&bank.audit)
        .map(|repeat| &repeat.repeat_sha256)
        .collect::<HashSet<_>>();
    assert_eq!(hashes.len(), 5);
}

#[test]
fn robust_loss_uses_pooled_repeat_rmse() {
    let score = robust_cv_score(
        &[25.0, 36.0],
        &[100, 100],
        vec![vec![0.5; 5], vec![0.6; 5]],
        1.0,
    )
    .unwrap();
    assert_eq!(score.repeat_nrmse, vec![0.5, 0.6]);
    assert!((score.robust_loss - 0.5625).abs() < 1e-15);
    assert_eq!(score.worst_fold_nrmse, 0.6);
}

#[test]
fn one_se_prefers_simpler_eligible_model() {
    let candidates = vec![
        OneSeCandidate {
            semantic_sha256: "b".repeat(64),
            complexity: 8,
            robust_loss: 0.100,
            repeat_nrmse: vec![0.09, 0.11],
            worst_fold_nrmse: 0.12,
        },
        OneSeCandidate {
            semantic_sha256: "a".repeat(64),
            complexity: 3,
            robust_loss: 0.105,
            repeat_nrmse: vec![0.10, 0.11],
            worst_fold_nrmse: 0.12,
        },
    ];
    assert_eq!(one_standard_error_select(&candidates).unwrap(), 1);
}

#[test]
fn bradley_terry_uses_official_relative_tie_rule_and_center() {
    let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
    let scores = BTreeMap::from([
        (
            "d1".to_string(),
            BTreeMap::from([
                ("a".to_string(), 1.0),
                ("b".to_string(), 1.0 + 5e-13),
                ("c".to_string(), 2.0),
            ]),
        ),
        (
            "d2".to_string(),
            BTreeMap::from([
                ("a".to_string(), 1.0),
                ("b".to_string(), 0.9),
                ("c".to_string(), 2.0),
            ]),
        ),
    ]);
    let ratings = bradley_terry_elo(&scores, &config.scoring).unwrap();
    let mean = ratings.iter().map(|row| row.rating).sum::<f64>() / ratings.len() as f64;
    assert!((mean - 1500.0).abs() < 1e-10);
    assert_eq!(ratings[0].method_id, "b");
    assert_eq!(ratings[2].method_id, "c");
}

#[test]
fn bradley_terry_matches_canonical_arena_reference() {
    let config = TrainOnlyRedteamConfig::sealed(DEFAULT_RUN_ID, DEFAULT_PARENT_RUN, false);
    let scores = BTreeMap::from([
        (
            "d1".to_string(),
            BTreeMap::from([
                ("a".to_string(), 1.0),
                ("b".to_string(), 1.0 + 5e-13),
                ("c".to_string(), 2.0),
                ("d".to_string(), 0.8),
            ]),
        ),
        (
            "d2".to_string(),
            BTreeMap::from([
                ("a".to_string(), 1.2),
                ("b".to_string(), 0.9),
                ("c".to_string(), 1.4),
                ("d".to_string(), 1.1),
            ]),
        ),
        (
            "d3".to_string(),
            BTreeMap::from([
                ("a".to_string(), 0.5),
                ("b".to_string(), 0.6),
                ("c".to_string(), 0.7),
                ("d".to_string(), 0.4),
            ]),
        ),
    ]);
    let rust = bradley_terry_elo(&scores, &config.scoring)
        .unwrap()
        .into_iter()
        .map(|rating| (rating.method_id, rating.rating))
        .collect::<BTreeMap<_, _>>();
    // The maintained port of sotarena/elo.py is qualified separately by
    // jope-arena/tests/elo_regression.rs against retained ladder fixtures.
    // This is cross-implementation Rust parity, not a Python execution.
    let reference = jope_arena::elo::fit_elo(&jope_arena::elo::matches_from_scores(&scores, 1e-12));
    assert_eq!(
        rust.keys().collect::<Vec<_>>(),
        reference.keys().collect::<Vec<_>>()
    );
    for (method, rating) in rust {
        assert!(
            (rating - reference[&method]).abs() < 1e-9,
            "{method}: campaign={rating}, arena reference={}",
            reference[&method]
        );
    }
}

#[test]
fn immutable_receipts_fail_closed_after_tampering() {
    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("receipt.json");
    write_receipt(
        &path,
        &json!({
            "schema": "TrainOnlyRedteamTamperFixture.v1",
            "development_only": true,
            "confirmatory_valid": false,
        }),
    )
    .unwrap();
    let mut value: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    value["development_only"] = Value::Bool(false);
    fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(read_receipt::<Value>(&path).is_err());
    assert!(write_receipt(
        &path,
        &json!({
            "schema": "TrainOnlyRedteamTamperFixture.v1",
            "development_only": true,
            "confirmatory_valid": false,
        }),
    )
    .is_err());
}
