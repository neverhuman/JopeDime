use super::*;
struct Resolver;
impl GenomeEngineResolver for Resolver {
    fn resolve(&mut self, _: &str, hint: Option<&str>) -> Option<String> {
        Some(hint.unwrap_or("featmath").into())
    }
}
struct Embed;
impl DatasetEmbedder for Embed {
    fn embed(&mut self, _: &str) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
        (Some(vec![1.0; 512]), Some(vec![2.0; 512]))
    }
}

#[test]
fn aggregate_row_maps_score_and_split_metadata() {
    let input = serde_json::json!({"schema":"jope-prime-realization-aggregate-v1","dataset_hash":"d1","canonical_genome":"x","reward":0.75,"oof_r2":0.7,"task":"regression"});
    let (row, reason) = label_to_training_row(
        &input,
        &mut Embed,
        &mut Resolver,
        "validation",
        DEFAULT_PROVENANCE,
    )
    .unwrap();
    let row = row.unwrap();
    assert_eq!(reason, "ok");
    assert_eq!(row.score, Some(0.75));
    assert_eq!(row.engine, "featmath");
    assert_eq!(row.metadata["split"], "validation");
    assert_eq!(row.metadata["score_robust"], 0.75);
    assert_eq!(row.metadata["oof_r2_raw"], 0.7);
    assert_eq!(row.embedding.len(), 512);
}

#[test]
fn report_only_and_unresolvable_labels_are_skipped() {
    let report =
        serde_json::json!({"split":"arena","dataset_hash":"d","canonical_genome":"x","reward":1.0});
    assert_eq!(
        label_to_training_row(
            &report,
            &mut Embed,
            &mut Resolver,
            "train",
            DEFAULT_PROVENANCE
        )
        .unwrap()
        .1,
        "report_only_evidence"
    );
    struct Missing;
    impl GenomeEngineResolver for Missing {
        fn resolve(&mut self, _: &str, _: Option<&str>) -> Option<String> {
            None
        }
    }
    let row = serde_json::json!({"dataset_hash":"d","canonical_genome":"x","reward":1.0});
    assert_eq!(
        label_to_training_row(&row, &mut Embed, &mut Missing, "train", DEFAULT_PROVENANCE)
            .unwrap()
            .1,
        "unencodable_genome"
    );
}

#[test]
fn failed_label_is_kept_as_invalid_negative_without_score() {
    let row = serde_json::json!({"dataset_hash":"d","canonical_genome":"x","failure":"timeout"});
    let (result, reason) =
        label_to_training_row(&row, &mut Embed, &mut Resolver, "train", DEFAULT_PROVENANCE)
            .unwrap();
    assert_eq!(reason, "ok");
    let result = result.unwrap();
    assert!(!result.valid);
    assert_eq!(result.score, None);
    assert_eq!(result.metadata["eval_error"], true);
}

#[test]
fn league_labels_keep_comparison_authority_and_runtime() {
    let row = serde_json::json!({
        "schema": LEAGUE_LABEL_SCHEMA, "split": "train", "dataset": "data-7",
        "canonical_genome": "program", "robust_score": 0.82, "task": "regression",
        "folds": 2, "fold_plan": "plan-v1", "evidence_tier": "heldout",
        "evaluator_version": "eval-v3", "runtime_seconds": 1.5, "failure": null,
        "provenance": {"gp_task_id":"task-4", "oof_r2":0.8, "fold_r2":[0.7,0.9], "robust_formula":"min"}
    });
    let (result, reason) =
        label_to_training_row(&row, &mut Embed, &mut Resolver, "train", DEFAULT_PROVENANCE)
            .unwrap();
    let result = result.unwrap();
    assert_eq!(reason, "ok");
    assert_eq!(result.score, Some(0.82));
    assert_eq!(result.cost_ms, Some(1500.0));
    assert_eq!(result.metadata["gp_task_id"], "task-4");
    assert!(result.metadata["reward_pair_identity"]
        .as_str()
        .unwrap()
        .starts_with("reward-pair-v1:"));
}

#[test]
fn missing_embedding_uses_stable_placeholders() {
    struct NoEmbeddings;
    impl DatasetEmbedder for NoEmbeddings {
        fn embed(&mut self, _: &str) -> (Option<Vec<f64>>, Option<Vec<f64>>) {
            (None, None)
        }
    }
    let row =
        serde_json::json!({"dataset_hash":"data-9", "canonical_genome":"program", "reward":0.5});
    let (first, _) = label_to_training_row(
        &row,
        &mut NoEmbeddings,
        &mut Resolver,
        "train",
        DEFAULT_PROVENANCE,
    )
    .unwrap();
    let (second, _) = label_to_training_row(
        &row,
        &mut NoEmbeddings,
        &mut Resolver,
        "train",
        DEFAULT_PROVENANCE,
    )
    .unwrap();
    let first = first.unwrap();
    let second = second.unwrap();
    assert_eq!(first.embedding.len(), 1536);
    assert_eq!(first.target_embedding.as_ref().unwrap().len(), 512);
    assert_eq!(first.embedding, second.embedding);
    assert_eq!(first.metadata["embedding_synthetic"], true);
}
