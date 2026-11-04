use super::*;

#[test]
fn v2_cohort_contexts_resolve_only_from_the_frozen_catalog_with_receipts() {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let data_path = root.join("contexts.bin");
    let context_a = [1_u8, 2, 3, 4];
    let context_b = [5_u8, 6, 7, 8];
    let context_c = [9_u8, 10, 11, 12];
    fs::write(
        &data_path,
        [
            context_a.as_slice(),
            context_b.as_slice(),
            context_c.as_slice(),
        ]
        .concat(),
    )
    .unwrap();
    let data_sha256 = sha256_file(&data_path).unwrap();
    let catalog_path = root.join("catalog.json");
    atomic_json(
        &catalog_path,
        &json!({
            "schema": "JopeExpandedCatalog.v2",
            "data_sha256": data_sha256,
            "record_stride_bytes": 4,
            "entries": [
                {
                    "task": "binary",
                    "dataset_hash": "valid",
                    "catalog_index": 0,
                    "context_sha256": sha256_bytes(&context_a),
                },
                {
                    "task": "regression",
                    "dataset_hash": "ambiguous",
                    "catalog_index": 1,
                    "context_sha256": sha256_bytes(&context_b),
                },
                {
                    "task": "regression",
                    "dataset_hash": "ambiguous",
                    "catalog_index": 2,
                    "context_sha256": sha256_bytes(&context_c),
                }
            ]
        }),
    )
    .unwrap();
    let catalog_sha256 = sha256_file(&catalog_path).unwrap();
    let pointer_path = root.join("frozen-pointer.json");
    let pointer = json!({
        "catalog": catalog_path,
        "catalog_sha256": catalog_sha256,
        "data": data_path,
        "data_sha256": data_sha256,
    });
    atomic_json(&pointer_path, &pointer).unwrap();
    let cohort_path = root.join("cohort.json");
    atomic_json(
        &cohort_path,
        &json!({
            "schema": "JopeMicroCohort.v2",
            "entries": [
                {"task": "binary", "dataset_hash": "valid", "micro_lane": "a"},
                {"task": "regression", "dataset_hash": "ambiguous", "micro_lane": "b"},
                {"task": "regression", "dataset_hash": "absent", "micro_lane": "c"}
            ]
        }),
    )
    .unwrap();
    let cohort_sha256 = sha256_file(&cohort_path).unwrap();
    let state = PersistedSupplyState {
        schema: SUPPLY_STATE_V2_SCHEMA.to_owned(),
        wave: "strict-wave-000223".to_owned(),
        checkpoint_sha256: "c".repeat(64),
        cohort: cohort_path,
        cohort_sha256: cohort_sha256.clone(),
        catalog_sha256: catalog_sha256.clone(),
        decoder_configuration_sha256: "e".repeat(64),
        seed: COHORT_PROPOSAL_SEED,
        proposal_reservoir_per_context: 64,
        stage: PersistedSupplyStage::FreezeCohort,
        revision: 0,
        recovery_accounting_only: false,
        proposal_attempt: 0,
        proposal_job_id: Some("proposal".to_owned()),
        cohort_proposal_pointer: None,
        cohort_proposal_pointer_sha256: None,
        sanitized_proposal_pointer: None,
        sanitized_proposal_pointer_sha256: None,
        runtime_pinned_attempts: Vec::new(),
    };
    let args = audit_args(root);
    let request_path =
        context_proposal_request(&args, &state, &pointer_path, &pointer, &"d".repeat(64)).unwrap();
    let request: Value = serde_json::from_slice(&fs::read(&request_path).unwrap()).unwrap();
    assert_eq!(
        request.get("schema").and_then(Value::as_str),
        Some("ContextProposalRequest.v4")
    );
    assert_eq!(
        request
            .get("requested_contexts")
            .and_then(Value::as_array)
            .unwrap()
            .len(),
        1
    );
    assert_eq!(
        request
            .pointer("/requested_contexts/0/context_sha256")
            .and_then(Value::as_str),
        Some(sha256_bytes(&context_a).as_str())
    );
    let receipt_path = root.join("proposals/cohort-inputs").join(format!(
        "{cohort_sha256}.{catalog_sha256}.{}.context-migration.v1.json",
        sha256_file(&request_path).unwrap()
    ));
    let receipt: Value = serde_json::from_slice(&fs::read(receipt_path).unwrap()).unwrap();
    assert_eq!(
        receipt.get("resolved_entries").and_then(Value::as_u64),
        Some(1)
    );
    assert_eq!(
        receipt.get("excluded_entries").and_then(Value::as_u64),
        Some(2)
    );

    fs::write(&data_path, b"catalog drift").unwrap();
    assert!(
        context_proposal_request(&args, &state, &pointer_path, &pointer, &"d".repeat(64),).is_err()
    );
}
