use super::*;

use super::*;
use jope_dime_regressor::selfplay_shadow::{
    EvidenceEpochContextV1, EvidencePreflightCohortV1, CAUSAL_EVIDENCE_EPOCH_SCHEMA,
    EVIDENCE_SLOTS_PER_CONTEXT, PREFLIGHT_COHORTS_REQUIRED, PREFLIGHT_CONTEXTS_PER_COHORT,
};
use tempfile::tempdir;

fn digest(value: &str) -> Sha256Digest {
    Sha256Digest::of_bytes(value.as_bytes())
}

#[test]
fn sealed_candidate_producer_binds_authorities_and_rejects_prior_use() {
    let temporary = tempdir().unwrap();
    let root = temporary.path();
    let facts_directory = root.join("facts");
    fs::create_dir_all(&facts_directory).unwrap();
    let base_seal = root.join("base-seal.json");
    let catalog = root.join("catalog.json");
    atomic_json(&base_seal, &json!({"schema":"BaseSeal.v1"})).unwrap();
    atomic_json(&catalog, &json!({"schema":"Catalog.v1"})).unwrap();
    let catalog_sha256 = Sha256Digest::parse(sha256_file(&catalog).unwrap()).unwrap();
    let context_sha256 = digest("context");
    atomic_json(
        &facts_directory.join("context.v1.json"),
        &json!({
            "schema": "SealedPanelContextFacts.v1",
            "task": "binary",
            "context_identity": "context",
            "context_sha256": context_sha256,
            "catalog_sha256": catalog_sha256,
            "train_sha256": digest("train"),
            "test_sha256": digest("test"),
            "source_family_sha256": digest("family"),
            "near_duplicate_cluster_sha256": digest("cluster"),
            "origin": "real",
            "rows": 10,
            "columns": 2,
            "missing_fraction": "0",
            "size_slice": "small",
            "missingness_slice": "none",
            "imbalance_slice": "balanced",
            "target_shape_slice": "balanced",
            "created_at_ms": 1
        }),
    )
    .unwrap();
    let prior = root.join("prior.json");
    atomic_json(
        &prior,
        &json!({
            "schema": "SealedPanelPriorUseIndex.v1",
            "watermark": 7,
            "used_context_sha256": []
        }),
    )
    .unwrap();
    let output = root.join("candidates.json");
    produce_sealed_panel_candidates(
        root,
        &base_seal,
        &catalog,
        &facts_directory,
        &prior,
        7,
        Some(output.clone()),
    )
    .unwrap();
    let candidates: SealedPanelCandidateSetV2 =
        serde_json::from_value(read_json(&output).unwrap()).unwrap();
    candidates.validate().unwrap();
    assert_eq!(candidates.entries.len(), 1);
    assert_eq!(candidates.entries[0].admission_watermark, 7);

    atomic_json(
        &prior,
        &json!({
            "schema": "SealedPanelPriorUseIndex.v1",
            "watermark": 7,
            "used_context_sha256": [context_sha256]
        }),
    )
    .unwrap();
    assert!(produce_sealed_panel_candidates(
        root,
        &base_seal,
        &catalog,
        &facts_directory,
        &prior,
        7,
        Some(root.join("rejected.json")),
    )
    .is_err());
}

#[test]
fn opening_an_epoch_is_idempotent_but_not_resettable() {
    let temporary = tempdir().unwrap();
    let root = temporary.path();
    let parent = digest("parent");
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "tasks": {
                "binary": {"checkpoint_sha256": parent},
                "regression": {"checkpoint_sha256": digest("regression")},
            }
        }),
    )
    .unwrap();
    let epoch = CausalEvidenceEpochV1 {
        schema: CAUSAL_EVIDENCE_EPOCH_SCHEMA.to_owned(),
        task: Task::Binary,
        accepted_parent_checkpoint_sha256: parent.clone(),
        start_wave: 300,
        runtime_policy_sha256: digest("runtime"),
        proposer_policy_sha256: digest("proposer"),
        evaluator_policy_sha256: digest("evaluator"),
        catalog_eligibility_sha256: digest("catalog"),
        requested_contexts: vec![EvidenceEpochContextV1 {
            context_identity: "context".to_owned(),
            context_sha256: digest("context"),
            requested_wave: 300,
        }],
        slots_per_context: EVIDENCE_SLOTS_PER_CONTEXT,
        preflight_cohorts: (0..PREFLIGHT_COHORTS_REQUIRED)
            .map(|sequence| EvidencePreflightCohortV1 {
                sequence,
                task: Task::Binary,
                accepted_parent_checkpoint_sha256: parent.clone(),
                contexts: PREFLIGHT_CONTEXTS_PER_COHORT,
                requested_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                valid_slots: PREFLIGHT_CONTEXTS_PER_COHORT * EVIDENCE_SLOTS_PER_CONTEXT,
                completed: true,
            })
            .collect(),
        opened_at_ms: 1,
    };
    let source = root.join("epoch.json");
    atomic_json(&source, &serde_json::to_value(&epoch).unwrap()).unwrap();
    assert_eq!(
        open_evidence_epoch(root, &source).unwrap()["status"],
        "opened"
    );
    assert_eq!(
        open_evidence_epoch(root, &source).unwrap()["status"],
        "already_open_identical"
    );
    let mut changed = epoch;
    changed.start_wave += 1;
    atomic_json(&source, &serde_json::to_value(&changed).unwrap()).unwrap();
    assert!(open_evidence_epoch(root, &source).is_err());
}

#[test]
fn external_evaluation_publishes_only_after_consumption_and_family_checks() {
    use jope_dime_regressor::selfplay_shadow::{
        DatasetOriginV2, PanelPoolKindV2, SealedPanelEntryV2, SEALED_PANEL_CONSUMPTION_V1_SCHEMA,
        SEALED_PANEL_POOL_V2_SCHEMA,
    };

    let temporary = tempdir().unwrap();
    let root = temporary.path();
    let parent = digest("parent");
    let candidate = digest("candidate");
    atomic_json(
        &root.join("training/accepted-by-task.json"),
        &json!({
            "tasks": {
                "binary": {"checkpoint_sha256": parent},
                "regression": {"checkpoint_sha256": digest("regression")},
            }
        }),
    )
    .unwrap();
    let pool = SealedPanelPoolV2 {
        schema: SEALED_PANEL_POOL_V2_SCHEMA.to_owned(),
        entries: (0..1_000)
            .map(|index| {
                let kind = match index {
                    0..=699 => PanelPoolKindV2::Training,
                    700..=799 => PanelPoolKindV2::Development,
                    800..=949 => PanelPoolKindV2::Promotion,
                    _ => PanelPoolKindV2::Audit,
                };
                SealedPanelEntryV2 {
                    task: Task::Binary,
                    context_identity: format!("context-{index}"),
                    context_sha256: digest(&format!("context-{index}")),
                    source_family_sha256: digest(&format!("family-{index}")),
                    near_duplicate_cluster_sha256: digest(&format!("cluster-{index}")),
                    origin: DatasetOriginV2::Real,
                    pool: kind,
                    promotion_panel: (kind == PanelPoolKindV2::Promotion)
                        .then(|| "panel-a".to_owned()),
                }
            })
            .collect(),
        created_at_ms: 1,
    };
    let pool_path = root.join("pool.json");
    atomic_json(&pool_path, &serde_json::to_value(&pool).unwrap()).unwrap();
    let panel = pool.promotion_panel(Task::Binary, "panel-a");
    let contexts = panel
        .iter()
        .take(128)
        .map(|entry| entry.context_sha256.clone())
        .collect::<Vec<_>>();
    let consumption = SealedPanelConsumptionV1 {
        schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
        task: Task::Binary,
        panel: "panel-a".to_owned(),
        candidate_checkpoint_sha256: candidate,
        run_index: 0,
        seed: 9,
        context_sha256: contexts.clone(),
        consumed_at_ms: 1,
    };
    let pending = root.join("consumption.json");
    atomic_json(&pending, &serde_json::to_value(&consumption).unwrap()).unwrap();
    let consumption_sha = sha256_file(&pending).unwrap();
    let consumption_path = root
        .join("training/sealed-panels/consumed")
        .join(format!("{consumption_sha}.v1.json"));
    fs::create_dir_all(consumption_path.parent().unwrap()).unwrap();
    fs::rename(pending, &consumption_path).unwrap();
    let observations = contexts
        .into_iter()
        .enumerate()
        .map(|(index, context_sha256)| PromotionObservationV3 {
            context_sha256,
            source_family_sha256: panel[index].source_family_sha256.clone(),
            candidate_score: 1.02,
            accepted_score: 1.0,
            gp_score: 1.0,
            candidate_conditional_advantage: 0.02,
            accepted_conditional_advantage: 0.01,
            valid: true,
            replay_verified: true,
            candidate_failed: false,
            accepted_failed: false,
            candidate_latency_ms: 100,
            accepted_latency_ms: 100,
            strict_win: index == 0,
            equal_budget_win: index == 1,
            slices: std::collections::BTreeMap::from([
                ("origin".to_owned(), "real".to_owned()),
                ("size".to_owned(), "small".to_owned()),
                ("missingness".to_owned(), "none".to_owned()),
                ("imbalance".to_owned(), "balanced".to_owned()),
                ("target_shape".to_owned(), "balanced".to_owned()),
            ]),
        })
        .collect::<Vec<_>>();
    let observation_path = root.join("observations.json");
    atomic_json(
        &observation_path,
        &json!({
            "schema": "TaskPromotionObservationSet.v3",
            "panel_consumption_sha256": consumption_sha,
            "observations": observations,
        }),
    )
    .unwrap();
    let result = evaluate_task_promotion(
        root,
        &pool_path,
        &consumption_path,
        &observation_path,
        parent.as_str(),
        7,
    )
    .unwrap();
    assert_eq!(result["accepted"], true);
    assert!(Path::new(result["evaluation"].as_str().unwrap()).is_file());
}
