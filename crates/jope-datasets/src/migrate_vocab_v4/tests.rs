use super::{default_tasks, default_true, migrate_vocab_v4, SOURCE_GRAMMARS};
use serde_json::{json, Value};
use std::fs;

fn legacy_vocab() -> (Value, String, String, String, usize, Vec<Value>) {
    let mut engines: Value = serde_json::from_str(SOURCE_GRAMMARS).unwrap();
    let mut selected = None::<(String, String, String, usize, Vec<Value>)>;
    for (engine, grammar) in engines.as_object_mut().unwrap().iter_mut() {
        for field in ["transforms", "heads"] {
            let operations = grammar[field].as_object_mut().unwrap();
            for (name, operation) in operations {
                let params = operation["params"].as_array_mut().unwrap();
                for (index, param) in params.iter_mut().enumerate() {
                    if selected.is_none() && param["kind"] == "expr" {
                        let original = param["bins"].as_array().unwrap().clone();
                        param["bins"] = json!(["legacy_expr_token"]);
                        selected =
                            Some((engine.clone(), field.into(), name.clone(), index, original));
                    }
                }
                operation.as_object_mut().unwrap().remove("tasks");
                operation.as_object_mut().unwrap().remove("generate");
            }
        }
    }
    let (engine, field, name, index, bins) = selected.unwrap();
    (
        json!({"syntax":"active_multi_engine_v3", "engines":engines}),
        engine,
        field,
        name,
        index,
        bins,
    )
}

#[test]
fn dry_run_replaces_expr_bins_and_repairs_legacy_task_fields() {
    let (vocab, engine, field, name, index, bins) = legacy_vocab();
    let source: Value = serde_json::from_str(SOURCE_GRAMMARS).unwrap();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("active_vocab.json");
    fs::write(&path, serde_json::to_vec(&vocab).unwrap()).unwrap();
    let migrated = migrate_vocab_v4(&path, false).unwrap();
    assert_eq!(migrated["vocab"]["schema_version"], 4);
    assert_eq!(migrated["report"]["drift_resolved"], true);
    assert_eq!(
        migrated["report"]["source_hash"],
        migrated["report"]["frozen_hash"]
    );
    assert!(!path.with_extension("v3.bak.json").exists());
    let migrated_op = &migrated["vocab"]["engines"][&engine][&field][&name];
    let patched = &migrated_op["params"][index];
    assert_eq!(patched["bins"], json!(bins));
    let source_op = &source[&engine][&field][&name];
    let expected_tasks = source_op
        .get("tasks")
        .cloned()
        .unwrap_or_else(|| json!(default_tasks()));
    let expected_generate = source_op
        .get("generate")
        .cloned()
        .unwrap_or_else(|| json!(default_true()));
    assert_eq!(migrated_op["tasks"], expected_tasks);
    assert_eq!(migrated_op["generate"], expected_generate);
}

#[test]
fn write_creates_a_non_overwritten_v3_backup() {
    let (vocab, _, _, _, _, _) = legacy_vocab();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("active_vocab.json");
    let original = serde_json::to_vec(&vocab).unwrap();
    fs::write(&path, &original).unwrap();
    migrate_vocab_v4(&path, true).unwrap();
    let backup = path.with_extension("v3.bak.json");
    assert_eq!(fs::read(&backup).unwrap(), original);
    fs::write(&backup, b"preserve-existing-backup").unwrap();
    migrate_vocab_v4(&path, true).unwrap();
    assert_eq!(fs::read(&backup).unwrap(), b"preserve-existing-backup");
}
