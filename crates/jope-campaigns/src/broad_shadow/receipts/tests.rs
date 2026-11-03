use super::super::canonical::{python_float, sha256};
use super::*;

fn digest(value: &str) -> String {
    sha256(value.as_bytes())
}

fn valid_receipt(train: &str) -> Value {
    let genome = "(pipe X (ridge))";
    build_receipt(ReceiptInput {
        dataset_hash: "0123456789abcdef".to_owned(),
        train_sha256: digest(train),
        semantic_sha256: digest(genome),
        canonical_genome: genome.to_owned(),
        evaluator_release_sha256: digest("evaluator"),
        fold_plan_sha256: digest("fold-plan"),
        vocabulary_sha256: digest("vocabulary"),
        fidelity: "screening".to_owned(),
        valid: true,
        fold_r2: Some(vec![0.8, 0.9, 0.7]),
        oof_r2: Some(0.85),
        runtime_seconds: Some(1.25),
        failure: None,
        selection_channel: "broad-shadow".to_owned(),
        recorded_at_ms: Some(123),
    })
    .unwrap()
}

#[test]
fn build_receipt_binds_fields_and_selects_the_screening_protocol() {
    let receipt = valid_receipt("train");
    assert_eq!(receipt["schema"], BROAD_RECEIPT_SCHEMA);
    assert_eq!(receipt["protocol"]["folds"], 3);
    assert_eq!(receipt["protocol"]["row_cap"], 768);
    assert_eq!(receipt["recorded_at_ms"], 123);
    validate_broad_receipt(&receipt).unwrap();

    let mut unhashed = receipt.clone();
    let expected = unhashed["receipt_sha256"].as_str().unwrap().to_owned();
    unhashed.as_object_mut().unwrap().remove("receipt_sha256");
    assert_eq!(content_sha256(&unhashed).unwrap(), expected);
}

#[test]
fn store_append_is_idempotent_and_verify_counts_receipts() {
    let temp = tempfile::tempdir().unwrap();
    let store = ReceiptStore::new(temp.path()).unwrap();
    let receipt = valid_receipt("train");
    let (path, created) = store.append(&receipt).unwrap();
    assert!(created);
    assert!(path.is_file());
    assert_eq!(store.append(&receipt).unwrap(), (path, false));
    assert_eq!(
        store
            .import_all([receipt.clone(), receipt.clone()])
            .unwrap(),
        json!({"inserted":0,"duplicates":2})
    );
    assert_eq!(store.iter().unwrap(), vec![receipt]);
    assert_eq!(
        store.verify().unwrap(),
        json!({
            "schema": STORE_VERIFICATION_SCHEMA,
            "receipts": 1,
            "valid": 1,
            "failures": 0,
            "store_sha256": content_sha256(&json!([
                store.iter().unwrap()[0]["receipt_sha256"]
            ]))
            .unwrap(),
            "duplicate_keys": 0
        })
    );
}

#[test]
fn same_scientific_key_cannot_publish_different_evidence() {
    let temp = tempfile::tempdir().unwrap();
    let store = ReceiptStore::new(temp.path()).unwrap();
    store.append(&valid_receipt("train-a")).unwrap();
    let error = store.append(&valid_receipt("train-b")).unwrap_err();
    assert!(error
        .to_string()
        .contains("receipt dedup key has conflicting immutable evidence"));
}

#[test]
fn receipt_validation_refuses_tampered_hashes_and_protocols() {
    let receipt = valid_receipt("train");
    let mut changed = receipt.clone();
    changed["selection_channel"] = json!("changed");
    assert!(validate_broad_receipt(&changed).is_err());

    let mut wrong_protocol = receipt;
    wrong_protocol["protocol"]["folds"] = json!(4);
    wrong_protocol["receipt_sha256"] = json!(content_sha256(&{
        let mut value = wrong_protocol.clone();
        value.as_object_mut().unwrap().remove("receipt_sha256");
        value
    })
    .unwrap());
    assert!(validate_broad_receipt(&wrong_protocol).is_err());
}

#[test]
fn invalid_receipt_keeps_failure_labels_null() {
    let genome = "(pipe X (ridge))";
    let failure = Map::from_iter([
        ("class".to_owned(), json!("TimeoutError")),
        ("message".to_owned(), json!("timed out")),
    ]);
    let receipt = build_receipt(ReceiptInput {
        dataset_hash: "0123456789abcdef".to_owned(),
        train_sha256: digest("train"),
        semantic_sha256: digest(genome),
        canonical_genome: genome.to_owned(),
        evaluator_release_sha256: digest("evaluator"),
        fold_plan_sha256: digest("fold-plan"),
        vocabulary_sha256: digest("vocabulary"),
        fidelity: "exact".to_owned(),
        valid: false,
        fold_r2: None,
        oof_r2: None,
        runtime_seconds: None,
        failure: Some(failure),
        selection_channel: "broad-shadow".to_owned(),
        recorded_at_ms: Some(123),
    });
    let receipt = receipt.unwrap();
    assert_eq!(receipt["protocol"]["folds"], 5);
    assert_eq!(receipt["fold_r2"], json!([]));
    assert!(receipt["oof_r2"].is_null());
    assert!(receipt["robust_score"].is_null());
    validate_broad_receipt(&receipt).unwrap();
}

#[test]
fn python_float_canonicalization_uses_json_dump_exponent_thresholds() {
    assert_eq!(python_float(1e-5), "1e-05");
    assert_eq!(python_float(1e-4), "0.0001");
    assert_eq!(python_float(1e16), "1e+16");
    assert_eq!(python_float(-0.0), "-0.0");
}
