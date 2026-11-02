use super::*;
use crate::elo::EloConfig;

/// The reporter's `test_config`.
pub fn test_config(direction: Direction) -> Config {
    Config {
        schema: "sotarena.config".to_owned(),
        tasks: vec![TaskConfig {
            id: "binary".to_owned(),
            metric: "roc_auc".to_owned(),
            direction,
            weight: 1.0,
            analysis_blacklist: "data/binary/analysis-blacklist.json".into(),
        }],
        elo: EloConfig::default(),
        families: vec![FamilyConfig {
            id: "family".to_owned(),
            display_name: "Family".to_owned(),
            patterns: vec!["family-*".to_owned()],
        }],
    }
}

pub fn data_file(url: Option<&str>) -> DataFile {
    DataFile {
        url: url.map(str::to_owned),
        sha256: "a".repeat(64),
        bytes: 1,
        rows: 200,
    }
}

/// Write a reporter data dir with one binary dataset and the given result records.
pub fn write_reporter_root(root: &Path, records: &[(&str, &str, f64, bool)]) {
    fs::create_dir_all(root.join("data/binary/0123456789abcdef/results")).unwrap();
    fs::write(
            root.join("config.json"),
            serde_json::json!({
                "schema": "sotarena.config",
                "tasks": [{"id": "binary", "metric": "roc_auc", "direction": "maximize", "weight": 1.0, "analysis_blacklist": "data/binary/analysis-blacklist.json"}],
                "elo": {"initial_rating": 1500.0, "k_factor": 40.0, "tie_tolerance": 1e-12, "iterations": 250, "regularization": 1e-4, "convergence_tolerance": 1e-8},
                "families": [{"id": "family", "display_name": "Family", "patterns": ["family-*"]}]
            })
            .to_string(),
        )
        .unwrap();
    fs::write(
        root.join("data/binary/analysis-blacklist.json"),
        r#"{"schema":"sotarena.analysis-blacklist","task":"binary","datasets":[]}"#,
    )
    .unwrap();
    fs::write(
            root.join("data/binary/0123456789abcdef/dataset.json"),
            serde_json::json!({
                "schema": "sotarena.dataset", "dataset_hash": "0123456789abcdef", "dataset_id": "0123456789abcdef",
                "task": "binary", "metric": "roc_auc",
                "train": {"url": "https://example.test/train.csv", "sha256": "a".repeat(64), "bytes": 1, "rows": 200},
                "test": {"url": "https://example.test/test.csv", "sha256": "b".repeat(64), "bytes": 1, "rows": 200}
            })
            .to_string(),
        )
        .unwrap();
    for (index, (method, dataset, value, valid)) in records.iter().enumerate() {
        fs::write(
                root.join(format!("data/binary/0123456789abcdef/results/{index}.json")),
                serde_json::json!({
                    "schema": "sotarena.result.v1", "task": "binary", "metric": "roc_auc",
                    "method_id": method, "display_name": method, "claim_scope": "test",
                    "dataset_id": dataset, "dataset_hash": dataset, "value": value,
                    "test_sha256": "b".repeat(64), "predictions_sha256": null, "run_id": format!("run-{index}"),
                    "benchmark_valid": valid
                })
                .to_string(),
            )
            .unwrap();
    }
}
