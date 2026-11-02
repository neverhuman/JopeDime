use super::*;
use serde_json::json;
use tempfile::tempdir;

fn read_rows(path: &Path) -> Vec<Value> {
    read_jsonl(path).unwrap()
}

#[test]
fn operators_and_sha_match_python_wire_rules() {
    assert_eq!(
        ops_of("(pipe X (ridge) (xgb2) (ridge)"),
        vec!["ridge".to_string(), "xgb2".to_string(), "ridge".to_string()]
    );
    assert_eq!(
        sha("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(py_num(Some(&json!(true))), Some(1.0));
    assert_eq!(py_num(Some(&json!("NaN"))), None);
    assert_eq!(py_num(Some(&json!("Infinity"))), Some(f64::INFINITY));
    let temp = tempdir().unwrap();
    let path = temp.path().join("python-nonfinite.jsonl");
    fs::write(&path, "{\"score\":Infinity,\"note\":\"NaN stays text\"}\n").unwrap();
    let row = read_rows(&path).pop().unwrap();
    assert_eq!(py_num(row.get("score")), Some(f64::INFINITY));
    assert_eq!(row["note"], "NaN stays text");
}

#[test]
fn export_filters_deduplicates_and_writes_gp_jsonl() {
    let temp = tempdir().unwrap();
    let input = temp.path().join("rows.jsonl");
    fs::write(
        &input,
        concat!(
            "{\"engine\":\"newmath\",\"canonical_genome\":\"(pipe X (ridge))\",\"oof_r2\":0.7,\"metadata\":{\"score_robust\":0.4}}\n",
            "{\"engine\":\"newmath\",\"genome\":\"(pipe X (ridge))\",\"oof_r2\":0.9}\n",
            "{\"engine\":\"featmath\",\"genome\":\"(pipe X (bad))\",\"oof_r2\":1.0}\n",
            "{\"engine\":\"newmath\",\"genome\":\"(pipe X (known))\",\"oof_r2\":0.8}\n"
        ),
    )
    .unwrap();
    let state = temp.path().join("state");
    fs::create_dir_all(&state).unwrap();
    fs::write(
        state.join("novelty_archive.jsonl"),
        "{\"canonical_expr\":\"(pipe X (known))\"}\n",
    )
    .unwrap();
    let output = temp.path().join("export");
    let result = gp_export(&[input], &output, Some(&state), ExportOptions::default()).unwrap();
    assert_eq!(result["exported"], 1);
    assert_eq!(result["considered"], 1);
    let concepts = read_rows(&output.join("concepts.jsonl"));
    let candidates = read_rows(&output.join("candidates.jsonl"));
    assert_eq!(concepts[0]["canonical_expr"], "(pipe X (ridge))");
    assert_eq!(concepts[0]["judge_scores"]["jope_oof_r2"], 0.9);
    assert_eq!(candidates[0]["predicted_score"], 0.9);
    assert_eq!(candidates[0]["acquisition_score"], 14.5);
    assert_eq!(
        candidates[0]["parent_concepts"][0],
        concepts[0]["concept_id"]
    );
}

#[test]
fn harvest_sorts_winners_deduplicates_and_emits_training_rows() {
    let temp = tempdir().unwrap();
    let winners = temp.path().join("winners");
    fs::create_dir_all(&winners).unwrap();
    fs::write(
        winners.join("winner-b.json"),
        r#"{"dsl_expr":"(pipe X (ridge))","benchmark_receipt":{"aggregate_elo":120.5,"aggregate_delta_lcb":0.4},"promotion_reason":"promoted"}"#,
    )
    .unwrap();
    fs::write(
        winners.join("winner-a.json"),
        r#"{"concept":{"canonical_expr":"(pipe X (ridge))","concept_id":"first"},"benchmark_receipt":{"aggregate_elo":100,"aggregate_delta_lcb":0.2}}"#,
    )
    .unwrap();
    fs::write(winners.join("winner-invalid.json"), "not-json").unwrap();
    fs::write(winners.join("other.json"), "{}").unwrap();
    let out = temp.path().join("rows.jsonl");
    let result = gp_harvest(Some(&winners), &out).unwrap();
    assert_eq!(result["harvested"], 1);
    let rows = read_rows(&out);
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["dataset_id"], "gp_winner");
    assert_eq!(rows[0]["engine"], "newmath");
    assert_eq!(rows[0]["embedding"].as_array().unwrap().len(), 512);
    assert_eq!(rows[0]["genome"], "(pipe X (ridge))");
    assert_eq!(rows[0]["elo"], 100.0);
    assert_eq!(rows[0]["score"], 0.2);
    assert_eq!(rows[0]["metadata"]["concept_id"], "first");
}
