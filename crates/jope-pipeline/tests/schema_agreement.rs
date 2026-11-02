//! The committed `jope.pipeline-receipt/v3` schema and the schemars output of
//! [`jope_pipeline::Receipt`] must agree on every object's required set and
//! property list, so neither can drift without this failing. The walk is the one
//! `crates/jope-datasets/tests/schema_agreement.rs` uses. v1 and v2 receipts keep
//! validating against their own committed schemas. The objective receipt is v3 plus the
//! objective, generated from the same type.

use jope_pipeline::receipt::{
    derived_objective_schema, derived_schema, objective_schema, schema, validate_value,
    OBJECTIVE_SCHEMA_ID, OBJECTIVE_SCHEMA_JSON, RECEIPT_SCHEMA_ID, RECEIPT_SCHEMA_JSON,
    V2_SCHEMA_ID, V2_SCHEMA_JSON,
};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::path::Path;

fn resolve<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    if let Some(Value::String(reference)) = node.get("$ref") {
        return resolve(
            root,
            root.pointer(reference.trim_start_matches('#'))
                .unwrap_or_else(|| panic!("unresolvable {reference}")),
        );
    }
    if let Some(Value::Array(alternatives)) = node.get("anyOf") {
        let non_null: Vec<&Value> = alternatives
            .iter()
            .filter(|a| a.get("type") != Some(&Value::String("null".into())))
            .collect();
        if non_null.len() == 1 {
            return resolve(root, non_null[0]);
        }
    }
    node
}

fn set(value: Option<&Value>) -> BTreeSet<String> {
    value
        .and_then(Value::as_array)
        .map(|a| a.iter().map(Value::to_string).collect())
        .unwrap_or_default()
}

fn compare(
    committed_root: &Value,
    derived_root: &Value,
    committed: &Value,
    derived: &Value,
    path: &str,
    findings: &mut Vec<String>,
) {
    let (committed, derived) = (
        resolve(committed_root, committed),
        resolve(derived_root, derived),
    );
    if committed.get("required").is_some() || derived.get("required").is_some() {
        let (c, d) = (set(committed.get("required")), set(derived.get("required")));
        if c != d {
            findings.push(format!(
                "{path}: required differs: committed {c:?} vs derived {d:?}"
            ));
        }
    }
    if let Some(enumeration) = derived.get("enum") {
        if set(committed.get("enum")) != set(Some(enumeration)) {
            findings.push(format!(
                "{path}: enum differs: committed {:?} vs derived {enumeration}",
                committed.get("enum")
            ));
        }
    }
    let (c, d) = (
        committed.get("properties").and_then(Value::as_object),
        derived.get("properties").and_then(Value::as_object),
    );
    if let (Some(c), Some(d)) = (c, d) {
        for (key, schema) in c {
            match d.get(key) {
                Some(other) => compare(
                    committed_root,
                    derived_root,
                    schema,
                    other,
                    &format!("{path}/{key}"),
                    findings,
                ),
                None => findings.push(format!(
                    "{path}: property {key} is missing from the derived schema"
                )),
            }
        }
        for key in d.keys() {
            if !c.contains_key(key) {
                findings.push(format!(
                    "{path}: property {key} is missing from the committed schema"
                ));
            }
        }
    } else if c.is_some() != d.is_some() {
        findings.push(format!(
            "{path}: only one side is an object with properties"
        ));
    }
    if let (Some(c), Some(d)) = (committed.get("items"), derived.get("items")) {
        compare(
            committed_root,
            derived_root,
            c,
            d,
            &format!("{path}/items"),
            findings,
        );
    }
}

#[test]
fn receipt_schema_agrees_with_the_rust_type() {
    let (committed, derived) = (schema(), derived_schema());
    let mut findings = Vec::new();
    compare(
        &committed,
        &derived,
        &committed,
        &derived,
        "",
        &mut findings,
    );
    assert!(findings.is_empty(), "{}", findings.join("\n"));
    assert_eq!(
        committed["$schema"],
        "https://json-schema.org/draft/2020-12/schema"
    );
    assert_eq!(committed["title"], RECEIPT_SCHEMA_ID);
    assert_eq!(
        committed["properties"]["schema"]["const"],
        RECEIPT_SCHEMA_ID
    );
}

#[test]
fn the_committed_file_is_byte_identical_to_the_embedded_schema() {
    let schemas = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    assert_eq!(
        std::fs::read_to_string(schemas.join("jope.pipeline-receipt.v3.schema.json")).unwrap(),
        RECEIPT_SCHEMA_JSON
    );
    assert_eq!(
        std::fs::read_to_string(schemas.join("jope.pipeline-receipt.v2.schema.json")).unwrap(),
        V2_SCHEMA_JSON
    );
}

/// A receipt shaped exactly as `run` writes one, with a measured champion.
fn fixture() -> Value {
    let mut value = json!({
        "schema": RECEIPT_SCHEMA_ID,
        "ok": true,
        "tool_version": jope_pipeline::TOOL_VERSION,
        "created_at": "2026-09-21T00:00:00Z",
        "policy": "skip",
        "dataset": {"id": "1cb97b1d7314d43d", "dataset_hash": "1cb97b1d7314d43d",
            "path": "datasets/test/pmlb-229-pwlinear", "task": "regression",
            "rows": 200, "columns": 11, "training_rows": 166,
            "files": [{"name": "train.csv", "sha256": "a".repeat(64), "bytes": 12}]},
        "prepared": {"sha256": "b".repeat(64), "seed": 20260907, "rows": 166, "folds": 5,
            "cuts": [0]},
        "stages": [
            {"name": "dataset", "status": "passed", "seconds": 0.1},
            {"name": "prepare", "status": "passed", "seconds": 0.1},
            {"name": "propose", "status": "skipped", "reason": "not built here", "seconds": 0.0},
            {"name": "search", "status": "passed", "seconds": 1.0},
            {"name": "select", "status": "passed", "seconds": 0.0},
            {"name": "emit", "status": "passed", "seconds": 0.2}
        ],
        "search": {"engine": "feat-math canonical-pipeline GP (in-process)",
            "rng_seed": 7, "evaluations": 8, "seconds": 120,
            "population": 12, "max_transforms": 4, "bootstrap_genome": "(pipe X (ridge))",
            "cuts": [{"cut": 0, "search_sha256": "e".repeat(64),
                "observations_sha256": "f".repeat(64), "train_rows": 133, "hold_rows": 33,
                "generations": 1, "completed_evaluations": 8,
                "successful_measurements": 6, "failed_or_unavailable_measurements": 2,
                "stopped_by": "evaluations", "best_validation_rmse": 2.5,
                "best_validation_r2": 0.42, "best_program": "(pipe X (ridge))"}],
            "selection": "min_mean_validation_rmse",
            "selected_genome": "(pipe X (ridge))",
            "selected_program_sha256": "0".repeat(64),
            "best": {"genome": "(pipe X (ridge))", "program_sha256": "0".repeat(64),
                "validation_rmse": 2.5, "validation_r2": 0.42, "train_r2": 0.55,
                "seconds": 0.8, "cuts_measured": 1}},
        "artifacts": [{"language": "rust", "path": "model.rs", "sha256": "1".repeat(64),
            "bytes": 4096}],
        "emit_exclusions": [],
        "observations_sha256": "2".repeat(64)
    });
    let sha = jope_pipeline::sha256_hex(b"(pipe X (ridge))");
    value["search"]["selected_program_sha256"] = json!(sha);
    value["search"]["best"]["program_sha256"] = json!(sha);
    value["search"]["selection"] = json!(jope_pipeline::select::MIN_MEAN_VALIDATION_RMSE);
    value["search"]["budget_policy"] = json!(jope_pipeline::search::BUDGET_POLICY);
    value["search"]["compared_cuts"] = json!([0]);
    value["search"]["coverage"] = json!((0..8)
        .map(|i| json!({
            "program_sha256": if i == 0 {sha.clone()} else {format!("{i:064x}")},
            "measured_cuts": if i < 6 {vec![0]} else {vec![]},
            "failed_cuts": if i < 6 {vec![]} else {vec![0]}, "missing_cuts": [],
        }))
        .collect::<Vec<_>>());
    value["search"]["cuts"][0]["accounting"] = json!({
        "active_seconds": 1.0, "deadline_overrun": false,
        "exploration": {"evaluation_limit": 8, "seconds_limit": 120.0,
            "completed_evaluations": 8, "successful_measurements": 6, "failed_measurements": 2,
            "seconds": 1.0, "stopped_by": "evaluations"},
        "comparison": jope_pipeline::receipt::Phase::not_requested()
    });
    value["search"]["seeding"] = json!({
        "offered": 0, "seeds": [], "parent_receipt_sha256": null, "seeds_file_sha256": null
    });
    value["search"]["islands"] = json!({"count": 1, "migration_interval": 4, "migrations": []});
    value
}

/// The same run recorded as a v2 receipt: no seeding or islands record.
fn v2_fixture() -> Value {
    let mut value = fixture();
    value["schema"] = json!(V2_SCHEMA_ID);
    value["search"].as_object_mut().unwrap().remove("seeding");
    value["search"].as_object_mut().unwrap().remove("islands");
    value
}

mod generated_schema;
mod receipts;
