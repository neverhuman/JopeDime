//! The committed `jope.pipeline-receipt/v1` schema and the schemars output of
//! [`jope_pipeline::Receipt`] must agree on every object's required set and
//! property list, so neither can drift without this failing. The walk is the one
//! `crates/jope-datasets/tests/schema_agreement.rs` uses.

use jope_pipeline::receipt::{
    derived_schema, schema, validate_value, RECEIPT_SCHEMA_ID, RECEIPT_SCHEMA_JSON,
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
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../schemas/jope.pipeline-receipt.v1.schema.json");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), RECEIPT_SCHEMA_JSON);
}

/// A receipt shaped exactly as `run` writes one, with a measured champion.
fn fixture() -> Value {
    json!({
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
    })
}

#[test]
fn a_well_formed_receipt_validates() {
    validate_value(&fixture()).unwrap();
}

/// A run that emitted nothing must not be able to describe itself as one that emitted code.
/// The CLI already fails such a run, but the receipt is what a later reader trusts, so the
/// contradiction is refused here too rather than only at the moment of the run.
#[test]
fn the_validator_refuses_an_emit_stage_that_disagrees_with_the_artifacts() {
    // `passed` with nothing emitted.
    let mut receipt = fixture();
    receipt["artifacts"] = json!([]);
    receipt["emit_exclusions"] = json!([
        {"language": "rust", "reason": "the emitter does not cover head gpr"}
    ]);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("emit passed but the receipt names no artifact")),
        "{errors:?}"
    );

    // `failed` while artifacts are listed: the mirror image, equally a lie.
    let mut receipt = fixture();
    receipt["ok"] = json!(false);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("emit failed but the receipt names 1 artifact")),
        "{errors:?}"
    );

    // `ok` may not survive a failed stage.
    let mut receipt = fixture();
    receipt["artifacts"] = json!([]);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("ok is true but a stage failed")),
        "{errors:?}"
    );

    // And the honest shape of the same run validates.
    let mut receipt = fixture();
    receipt["ok"] = json!(false);
    receipt["artifacts"] = json!([]);
    receipt["stages"][5]["status"] = json!("failed");
    receipt["stages"][5]["reason"] = json!("rust: the emitter does not cover head gpr");
    receipt["emit_exclusions"] = json!([
        {"language": "rust", "reason": "the emitter does not cover head gpr"}
    ]);
    validate_value(&receipt).unwrap();
}

#[test]
fn the_validator_rejects_what_the_schema_and_the_invariants_forbid() {
    let cases = [
        ("schema", json!("jope.pipeline-receipt/v2"), "const"),
        ("policy", json!("maybe"), "policy"),
        ("observations_sha256", json!("short"), "match"),
    ];
    for (key, value, needle) in cases {
        let mut receipt = fixture();
        receipt[key] = value;
        let errors = validate_value(&receipt).unwrap_err();
        assert!(
            errors.iter().any(|e| e.contains(needle)),
            "{key}: {errors:?} has no {needle}"
        );
    }
    // An unknown key is refused rather than ignored.
    let mut receipt = fixture();
    receipt["surprise"] = json!(1);
    assert!(validate_value(&receipt).is_err());
    // A skipped stage without a reason is refused by the invariants.
    let mut receipt = fixture();
    receipt["stages"][2]
        .as_object_mut()
        .unwrap()
        .remove("reason");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("must record a reason")),
        "{errors:?}"
    );
    // A missing stage is refused.
    let mut receipt = fixture();
    receipt["stages"] = json!([]);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.contains("stage propose is missing")),
        "{errors:?}"
    );
    // A search without a best program must say it fell back.
    let mut receipt = fixture();
    receipt["search"].as_object_mut().unwrap().remove("best");
    receipt["search"]["cuts"][0]["successful_measurements"] = json!(0);
    receipt["search"]["cuts"][0]["failed_or_unavailable_measurements"] = json!(8);
    receipt["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("best_program");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("without a best program")),
        "{errors:?}"
    );

    // Measurements plus the unmeasured fallback is a contradiction.
    let mut receipt = fixture();
    receipt["search"]["selection"] = json!("unmeasured_fallback_bootstrap_genome");
    receipt["search"].as_object_mut().unwrap().remove("best");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("unmeasured fallback")),
        "{errors:?}"
    );

    // A cut's evaluations must account for every candidate's outcome.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]["completed_evaluations"] = json!(99);
    let errors = validate_value(&receipt).unwrap_err();
    assert!(errors.iter().any(|e| e.contains("outcomes")), "{errors:?}");

    // A cut that measured something must name what it measured.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]
        .as_object_mut()
        .unwrap()
        .remove("best_program");
    let errors = validate_value(&receipt).unwrap_err();
    assert!(
        errors.iter().any(|e| e.contains("no best program")),
        "{errors:?}"
    );

    // `stopped_by` is closed: a cut cannot claim an invented bound.
    let mut receipt = fixture();
    receipt["search"]["cuts"][0]["stopped_by"] = json!("boredom");
    assert!(validate_value(&receipt).is_err());
}
