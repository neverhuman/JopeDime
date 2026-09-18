//! The committed JSON Schemas and the schemars output of the Rust types must agree on every
//! object's required set and on every enum, so neither can drift without a test failing.

use jope_datasets::catalog::{
    catalog_schema, collection_schema, derived_catalog_schema, derived_collection_schema,
    CatalogRow, Collection,
};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

fn resolve<'a>(root: &'a Value, node: &'a Value) -> &'a Value {
    if let Some(Value::String(r)) = node.get("$ref") {
        return resolve(
            root,
            root.pointer(r.trim_start_matches('#'))
                .unwrap_or_else(|| panic!("unresolvable {r}")),
        );
    }
    if let Some(Value::Array(alts)) = node.get("anyOf") {
        let non_null: Vec<&Value> = alts
            .iter()
            .filter(|a| a.get("type") != Some(&Value::String("null".into())))
            .collect();
        if non_null.len() == 1 {
            return resolve(root, non_null[0]);
        }
    }
    node
}

fn set(v: Option<&Value>) -> BTreeSet<String> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().map(|x| x.to_string()).collect())
        .unwrap_or_default()
}

fn compare(
    c_root: &Value,
    d_root: &Value,
    c: &Value,
    d: &Value,
    path: &str,
    findings: &mut Vec<String>,
) {
    let (c, d) = (resolve(c_root, c), resolve(d_root, d));
    if c.get("required").is_some() || d.get("required").is_some() {
        let (cr, dr) = (set(c.get("required")), set(d.get("required")));
        if cr != dr {
            findings.push(format!(
                "{path}: required differs: committed {cr:?} vs derived {dr:?}"
            ));
        }
    }
    if let Some(de) = d.get("enum") {
        if set(c.get("enum")) != set(Some(de)) {
            findings.push(format!(
                "{path}: enum differs: committed {:?} vs derived {de}",
                c.get("enum")
            ));
        }
    }
    let (cp, dp) = (
        c.get("properties").and_then(Value::as_object),
        d.get("properties").and_then(Value::as_object),
    );
    if let (Some(cp), Some(dp)) = (cp, dp) {
        for (k, cs) in cp {
            match dp.get(k) {
                Some(ds) => compare(c_root, d_root, cs, ds, &format!("{path}/{k}"), findings),
                None => findings.push(format!(
                    "{path}: property {k} is missing from the derived schema"
                )),
            }
        }
        for k in dp.keys() {
            if !cp.contains_key(k) {
                findings.push(format!(
                    "{path}: property {k} is missing from the committed schema"
                ));
            }
        }
    } else if cp.is_some() != dp.is_some() {
        findings.push(format!(
            "{path}: only one side is an object with properties"
        ));
    }
    if let (Some(ci), Some(di)) = (c.get("items"), d.get("items")) {
        compare(c_root, d_root, ci, di, &format!("{path}/items"), findings);
    }
}

#[test]
fn catalog_schema_agrees_with_the_rust_types() {
    let (c, d) = (catalog_schema(), derived_catalog_schema());
    let mut findings = Vec::new();
    compare(&c, &d, &c, &d, "", &mut findings);
    assert!(findings.is_empty(), "{}", findings.join("\n"));
    assert_eq!(c["$schema"], "https://json-schema.org/draft/2020-12/schema");
    assert_eq!(c["title"], "jope.dataset-catalog/v1");
    assert_eq!(c["properties"]["dataset_hash"]["pattern"], "^[0-9a-f]{16}$");
}

#[test]
fn collection_schema_agrees_with_the_rust_types() {
    let (c, d) = (collection_schema(), derived_collection_schema());
    let mut findings = Vec::new();
    compare(&c, &d, &c, &d, "", &mut findings);
    assert!(findings.is_empty(), "{}", findings.join("\n"));
    assert_eq!(c["title"], "jope.dataset-collection/v1");
}

#[test]
fn committed_files_are_byte_identical_to_the_embedded_schemas() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    assert_eq!(
        std::fs::read_to_string(root.join("jope.dataset-catalog.v1.schema.json")).unwrap(),
        jope_datasets::catalog::CATALOG_SCHEMA_JSON
    );
    assert_eq!(
        std::fs::read_to_string(root.join("jope.dataset-collection.v1.schema.json")).unwrap(),
        jope_datasets::catalog::COLLECTION_SCHEMA_JSON
    );
}

#[test]
fn committed_test_datasets_validate() {
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets");
    let coll: Value =
        serde_json::from_slice(&std::fs::read(datasets.join("manifest.json")).unwrap()).unwrap();
    let coll = Collection::validate_value(&coll).unwrap_or_else(|e| panic!("{e:?}"));
    assert_eq!((coll.name.as_str(), coll.members.len()), ("test-10", 10));
    let mut seen = 0;
    for entry in std::fs::read_dir(datasets.join("test")).unwrap() {
        let dir = entry.unwrap().path();
        if !dir.is_dir() {
            continue;
        }
        let v: Value =
            serde_json::from_slice(&std::fs::read(dir.join("metadata.json")).unwrap()).unwrap();
        let row =
            CatalogRow::validate_value(&v).unwrap_or_else(|e| panic!("{}: {e:?}", dir.display()));
        assert!(coll.members.contains(&row.id), "{} is not a member", row.id);
        assert_eq!(
            row.provenance.license.status,
            jope_datasets::catalog::LicenseStatus::Recorded
        );
        seen += 1;
    }
    assert_eq!(seen, 10);
    let rows: Vec<CatalogRow> = Vec::new();
    let (errors, warnings) = coll.license_findings(&rows.iter().collect::<Vec<_>>());
    assert!(errors.is_empty() && warnings.is_empty());
    let mut bad = serde_json::to_value(&coll).unwrap();
    bad["members_sha256"] = Value::String("0".repeat(64));
    assert!(Collection::validate_value(&bad).is_err());
}

#[test]
fn schema_rejects_bad_rows() {
    let datasets = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../datasets/test/pmlb-229-pwlinear/metadata.json");
    let good: Value = serde_json::from_slice(&std::fs::read(datasets).unwrap()).unwrap();
    assert!(CatalogRow::validate_value(&good).is_ok());
    let mut v = good.clone();
    v["origin"] = Value::String("synthetic".into());
    let errs = CatalogRow::validate_value(&v).unwrap_err();
    assert!(errs.iter().any(|e| e.contains("generator")), "{errs:?}");
    let mut v = good.clone();
    v["canonical_sha256"] = Value::String("f".repeat(64));
    assert!(CatalogRow::validate_value(&v)
        .unwrap_err()
        .iter()
        .any(|e| e.contains("dataset_hash")));
    let mut v = good.clone();
    v["provenance"]["license"]["evidence_url"] = Value::Null;
    assert!(CatalogRow::validate_value(&v)
        .unwrap_err()
        .iter()
        .any(|e| e.contains("evidence_url")));
    let mut v = good.clone();
    v["extra"] = Value::Bool(true);
    assert!(CatalogRow::validate_value(&v)
        .unwrap_err()
        .iter()
        .any(|e| e.contains("additional property")));
    let mut v = good;
    v["task"] = Value::String("clustering".into());
    assert!(CatalogRow::validate_value(&v).is_err());
}
