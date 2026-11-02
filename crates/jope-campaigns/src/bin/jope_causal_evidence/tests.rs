use super::*;
use jope_dime_regressor::selfplay_shadow::{
    build_causal_evidence_ledger_v2, causal_pair_id, DlAncestryKind, PairRepairRequestV1,
    Sha256Digest, Task,
};
use std::fs;

#[test]
fn wave_189_is_excluded_and_receipt_era_starts_at_191() {
    assert_eq!(parse_wave("strict-wave-000189"), Some(189));
    assert_eq!(parse_wave("strict-wave-000191"), Some(191));
    assert_eq!(parse_wave("shadow-wave-000191"), None);
}

#[test]
fn executor_permanent_failure_is_reflected_in_the_authoritative_queue() {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir_all(temporary.path().join("ops")).unwrap();
    let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
    let pair_id = causal_pair_id(&parent, 191, Task::Binary, "context", 0);
    let direct = CausalEvaluationReceiptV2 {
        schema: CAUSAL_EVALUATION_RECEIPT_SCHEMA.to_owned(),
        receipt_sha256: Sha256Digest::of_bytes(b"receipt"),
        manifest_sha256: Sha256Digest::of_bytes(b"manifest"),
        candidate_sha256: Sha256Digest::of_bytes(b"candidate"),
        task: Task::Binary,
        context_identity: "context".to_owned(),
        accepted_parent_checkpoint_sha256: parent.clone(),
        wave: 191,
        lane: CausalLane::NativeDlDirect,
        pair_index: 0,
        pair_id: pair_id.clone(),
        score: Some(1.0),
        success: true,
        proposal_valid: true,
        runtime_ms: 1,
        ancestry: DlAncestryKind::AcceptedDl,
        cross_context_backfill: false,
        completed_causal_wave: true,
        valid_lineage: true,
        canary: false,
    };
    let mut ledger = build_causal_evidence_ledger_v2(
        &BTreeMap::from([(Task::Binary, parent)]),
        &BTreeMap::from([(Task::Binary, 8)]),
        vec![direct],
        1,
    )
    .unwrap();
    assert_eq!(
        ledger.pair_repairs,
        vec![PairRepairRequestV1 {
            pair_id: pair_id.clone(),
            task: Task::Binary,
            missing_lane: CausalLane::MatchedGpComparator,
            original_manifest_sha256: Sha256Digest::of_bytes(b"manifest"),
            permanent_failure: false,
        }]
    );
    atomic_json(
        &temporary
            .path()
            .join("ops/causal-pair-repair-state.v1.json"),
        &json!({
            "schema": "CausalPairRepairExecutorState.v1",
            "updated_at_ms": 1,
            "entries": {
                format!("{pair_id}:matched_gp_comparator"): {
                    "pair_id": pair_id,
                    "missing_lane": "matched_gp_comparator",
                    "status": "permanent_failure"
                }
            }
        }),
    )
    .unwrap();
    apply_permanent_repair_failures(temporary.path(), &mut ledger).unwrap();
    assert!(ledger.pair_repairs[0].permanent_failure);
}

#[test]
fn missing_evaluator_is_classified_unreplayable_without_imputation() {
    let temporary = tempfile::tempdir().unwrap();
    fs::create_dir_all(temporary.path().join("ops")).unwrap();
    atomic_json(
        &temporary
            .path()
            .join("ops/causal-pair-repair-state.v1.json"),
        &json!({
            "schema": "CausalPairRepairExecutorState.v1",
            "entries": {
                "pair": {
                    "pair_id": "pair",
                    "status": "permanent_failure"
                },
                "replayable": {
                    "pair_id": "replayable",
                    "status": "permanent_failure",
                    "original_evaluator_sha256": "a".repeat(64)
                }
            }
        }),
    )
    .unwrap();
    let disposition = unreplayable_pair_disposition(temporary.path(), 10).unwrap();
    assert_eq!(disposition["schema"], "JopeUnreplayablePairDisposition.v2");
    assert_eq!(disposition["count"], 2);
    assert_eq!(disposition["unique_pair_count"], 2);
    assert_eq!(disposition["promotion_eligible"], false);
    assert_eq!(disposition["evaluator_substitution_allowed"], false);
    assert_eq!(disposition["outcome_imputation_allowed"], false);
    let object = PathBuf::from(disposition["pair_identities_path"].as_str().unwrap());
    assert!(object.is_file());
    assert_eq!(
        sha256_file(&object).unwrap(),
        disposition["pair_identities_sha256"]
    );
    assert!(disposition.get("pair_ids").is_none());
}
