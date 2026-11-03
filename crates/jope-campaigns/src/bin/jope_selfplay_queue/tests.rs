#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint() -> String {
        "a".repeat(64)
    }

    fn manifest_value(wave: &str, epoch: &str) -> Value {
        let checkpoint = checkpoint();
        let genome = "(pipe X (ridge))";
        let train_sha256 = "d".repeat(64);
        let test_sha256 = "e".repeat(64);
        let protocol_sha256 = jope_dime_regressor::selfplay_runtime::evaluation_protocol_sha256(
            "regression",
            "r2",
            "test_exact",
            &train_sha256,
            &test_sha256,
        )
        .unwrap();
        json!({
            "schema": MANIFEST_SCHEMA,
            "strict_wave": wave,
            "task": "regression",
            "dataset_hash": "dataset",
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": checkpoint,
            "catalog_sha256": "b".repeat(64),
            "protocol_sha256": protocol_sha256,
            "generated_at_ms": 10,
            "micro_lane": "least_observed",
            "train_csv": "/train.csv",
            "test_csv": "/test.csv",
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": "r2",
                "test_adaptive": true,
                "benchmark_valid": false,
                "generation": wave,
                "model_checkpoint_sha256": checkpoint,
            },
            "candidates": [{
                "canonical_genome": genome,
                "semantic_sha256": sha256_bytes(genome.as_bytes()),
                "model_checkpoint_sha256": checkpoint,
            }]
        })
    }

    fn generated_registry() -> GeneratedOperatorRegistry {
        GeneratedOperatorRegistry::from_bytes(
            br#"{
              "registry_version":"generated-v1",
              "registry_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "maximum_output_columns":512,
              "operators":[
                {"id":0,"name":"ridge","role":"head","regression":true,"estimated_cost":2,
                 "lifecycle":{"parseable":true,"quarantined":false,"deprecated":false}}
              ]
            }"#,
        )
        .unwrap()
    }

    #[path = "manifest.rs"]
    mod manifest;
    #[path = "queue.rs"]
    mod queue;
    #[path = "submission.rs"]
    mod submission;
}
