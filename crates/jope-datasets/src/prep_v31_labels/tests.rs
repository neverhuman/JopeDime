use super::{prep_v31_labels, PrepV31Report};
use serde_json::{json, Value};
use std::fs;

fn write_rows(path: &std::path::Path, rows: &[Value]) {
    let body = rows
        .iter()
        .map(serde_json::to_string)
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
        .join("\n");
    fs::write(path, format!("{body}\n")).unwrap();
}

#[test]
fn average_rank_ties_singletons_and_missing_scores_match_python_labels() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corpus.jsonl");
    write_rows(
        &path,
        &[
            json!({"dataset_id":"a", "score":1.0}),
            json!({"dataset_id":"a", "score":1.0}),
            json!({"dataset_id":"a", "score":3.0}),
            json!({"dataset_id":"b", "metadata":{"score_robust":7.0}}),
            json!({"dataset_id":"c", "score":null}),
        ],
    );

    let report = prep_v31_labels(&path, false).unwrap();
    assert_eq!(
        report,
        PrepV31Report {
            rows: 5,
            datasets: 2,
            rows_ranked: 4,
            rows_censored_runtime: 0,
        }
    );
    let rows = fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows[0]["metadata"]["rank_pct"], json!(0.166667));
    assert_eq!(rows[1]["metadata"]["rank_pct"], json!(0.166667));
    assert_eq!(rows[2]["metadata"]["rank_pct"], json!(1.0));
    assert_eq!(rows[3]["metadata"]["rank_pct"], json!(0.5));
    assert_eq!(rows[4]["metadata"]["rank_pct"], Value::Null);
}

#[test]
fn dry_run_does_not_write_or_create_backup() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corpus.jsonl");
    let original = "{\"dataset_id\":\"d\",\"score\":1}\n";
    fs::write(&path, original).unwrap();

    prep_v31_labels(&path, false).unwrap();

    assert_eq!(fs::read_to_string(&path).unwrap(), original);
    assert!(!directory.path().join("corpus.v3.bak.jsonl").exists());
}

#[test]
fn write_creates_backup_once_and_censors_runtime_at_the_cap() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("corpus.jsonl");
    write_rows(
        &path,
        &[
            json!({"dataset_id":"d", "score":1.0, "cost_ms":600000.0}),
            json!({"dataset_id":"d", "score":2.0, "cost_ms":12.5, "metadata":{"task_type":"classification"}}),
        ],
    );
    let original = fs::read(&path).unwrap();

    let report = prep_v31_labels(&path, true).unwrap();
    let backup = directory.path().join("corpus.v3.bak.jsonl");
    assert_eq!(fs::read(&backup).unwrap(), original);
    assert_eq!(report.rows_censored_runtime, 1);

    let rows = fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(rows[0]["metadata"]["runtime_censored"], json!(true));
    assert_eq!(rows[0]["metadata"]["task_type"], json!("regression"));
    assert_eq!(rows[1]["metadata"]["runtime_censored"], json!(false));
    assert_eq!(rows[1]["metadata"]["task_type"], json!("classification"));
    let weight_total = rows[0]["metadata"]["rwr_weight_global"].as_f64().unwrap()
        + rows[1]["metadata"]["rwr_weight_global"].as_f64().unwrap();
    assert!((weight_total - 2.0).abs() < 1e-6);

    fs::write(&path, &original).unwrap();
    prep_v31_labels(&path, true).unwrap();
    assert_eq!(fs::read(backup).unwrap(), original);
}
