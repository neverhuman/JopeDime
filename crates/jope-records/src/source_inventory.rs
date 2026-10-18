//! `verify-source-inventory`: static source/test classification check ported from
//! `deploy/dl-flywheel/verify_source_inventory.py` (`PORT-RUST-03`).
//!
//! Checks the exact test-bearing key rule, roles/subsystems against `deploy/ci/coverage.json`
//! and `docs/internal/core-ip/findings/test-inventory.json`, and verifies the non-test research
//! source hash inventory in `deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json`.

use std::collections::HashSet;
use std::path::{Component, Path};

use crate::py;
use crate::pyjson::{Dict, Value};
use crate::test_inventory::{REQUIRED, ROLES, SCHEMA as TEST_INVENTORY_SCHEMA, SUBSYSTEM_ROLES};
use crate::{require, Error, Result};

pub const RESEARCH_SCHEMA: &str = "jope.flywheel-research-source-classification/v1";
pub const OUTPUT_SCHEMA: &str = "jope.flywheel-source-classification-check/v1";
pub const SCOPE: &str = "Static key/role/source-hash validation only; no tests executed. Native jope-records baseline-triage validation remains required.";

/// Summary returned by `verify-source-inventory`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub test_items: usize,
    pub research_sources: usize,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("research_sources", self.research_sources.into());
        dict.insert("schema", OUTPUT_SCHEMA.into());
        dict.insert("scope", SCOPE.into());
        dict.insert("state", "passed".into());
        dict.insert("test_items", self.test_items.into());
        Value::Dict(dict)
    }
}

fn dict_field<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a Dict> {
    value
        .get(name)
        .and_then(Value::as_dict)
        .ok_or_else(|| Error::Shape(format!("{context}: {name} is not an object")))
}

fn has_unsafe_relative_parts(relative: &Path) -> bool {
    if relative.is_absolute() {
        return true;
    }
    relative.components().any(|c| {
        matches!(
            c,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

/// Verify `deploy/ci/coverage.json`, `docs/internal/core-ip/findings/test-inventory.json`, and
/// `deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json` inside `repo`.
pub fn verify(repo: &Path) -> Result<Summary> {
    let repo = repo.canonicalize().map_err(|e| Error::io(repo, e))?;
    let coverage = py::load_json(&repo.join("deploy/ci/coverage.json"))?;
    let inventory =
        py::load_json(&repo.join("docs/internal/core-ip/findings/test-inventory.json"))?;
    let sources =
        py::load_json(&repo.join("deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json"))?;

    require!(
        inventory.get("schema") == Some(&Value::from(TEST_INVENTORY_SCHEMA)),
        "test inventory schema"
    );

    let cov_items = dict_field(&coverage, "items", "coverage")?;
    let cov_lanes = dict_field(&coverage, "lanes", "coverage")?;

    let mut expected: HashSet<String> = HashSet::new();
    for (key, value) in cov_items.iter() {
        let discovery = value
            .get("discovery")
            .ok_or_else(|| Error::Shape(format!("coverage item {key}: no discovery")))?;
        let kind = discovery.get("kind").and_then(Value::as_str);
        if kind == Some("python-family")
            || (kind == Some("target") && discovery.get("test").is_some_and(Value::truthy))
        {
            expected.insert(key.to_owned());
        }
    }

    let inv_items = dict_field(&inventory, "items", "inventory")?;
    let actual: HashSet<String> = inv_items.keys().map(str::to_owned).collect();
    if expected != actual {
        let mut unclassified: Vec<&str> =
            expected.difference(&actual).map(String::as_str).collect();
        unclassified.sort_unstable();
        let mut stale: Vec<&str> = actual.difference(&expected).map(String::as_str).collect();
        stale.sort_unstable();
        return Err(Error::Invalid(format!(
            "{{'unclassified': {}, 'stale': {}}}",
            py::str_list_repr(&unclassified),
            py::str_list_repr(&stale)
        )));
    }

    let inv_subsystems = dict_field(&inventory, "subsystems", "inventory")?;
    require!(
        REQUIRED.iter().all(|req| inv_subsystems.contains_key(req)),
        "required subsystem definitions"
    );

    let mut covered: HashSet<String> = HashSet::new();
    for (key, value) in inv_items.iter() {
        let role = value
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Invalid(format!("unknown test role: {key}")))?;
        require!(ROLES.contains(&role), "unknown test role: {key}");
        if SUBSYSTEM_ROLES.contains(&role) {
            let names = value
                .get("subsystems")
                .and_then(Value::as_list)
                .map(|s| s.as_slice())
                .unwrap_or(&[]);
            let mut unique: HashSet<&str> = HashSet::new();
            let valid_names = !names.is_empty()
                && names.iter().all(|n| {
                    n.as_str()
                        .is_some_and(|s| unique.insert(s) && inv_subsystems.contains_key(s))
                });
            require!(valid_names, "regression subsystems: {key}");

            if role == "regression" {
                let item_lanes = cov_items
                    .get(key)
                    .and_then(|item| item.get("lanes"))
                    .and_then(Value::as_list)
                    .ok_or_else(|| Error::Shape(format!("coverage item {key}: lanes")))?;
                let executes = item_lanes.iter().any(|l| {
                    l.as_str().is_some_and(|lane| {
                        lane != "rust-check"
                            && cov_lanes
                                .get(lane)
                                .and_then(|cfg| cfg.get("stage"))
                                .and_then(Value::as_str)
                                == Some("execution")
                    })
                });
                if executes {
                    for name in unique {
                        covered.insert(name.to_owned());
                    }
                }
            }
        } else {
            let has_subsystems = value.get("subsystems").is_some_and(Value::truthy);
            let has_reason = value.get("reason").is_some_and(Value::truthy);
            require!(
                !has_subsystems && has_reason,
                "non-regression reason: {key}"
            );
        }
    }

    require!(
        REQUIRED.iter().all(|req| covered.contains(*req)),
        "missing executing regression subsystem"
    );

    require!(
        sources.get("schema") == Some(&Value::from(RESEARCH_SCHEMA)),
        "research source schema"
    );

    let src_items = dict_field(&sources, "items", "sources")?;
    for (key, value) in src_items.iter() {
        let classification = dict_field(value, "classification", key)?;
        let role = classification.get("role").and_then(Value::as_str);
        require!(
            !actual.contains(key) && key.starts_with("python:") && role == Some("research"),
            "source is not a distinct non-test research item: {key}"
        );
        let rel_str = &key["python:".len()..];
        let relative = Path::new(rel_str);
        require!(
            !has_unsafe_relative_parts(relative),
            "source path confinement"
        );
        let path = repo.join(relative);
        require!(!py::is_symlink(&path), "source path confinement");
        let resolved = path
            .canonicalize()
            .map_err(|_| Error::Invalid("source path confinement".into()))?;
        require!(resolved.starts_with(&repo), "source path confinement");

        let raw = py::read_bytes(&path)?;
        let expected_bytes = value.get("bytes").and_then(Value::as_int);
        let expected_sha = value.get("sha256").and_then(Value::as_str);
        require!(
            expected_bytes == Some(raw.len() as i64)
                && expected_sha == Some(py::sha256_hex(&raw).as_str()),
            "research source hash: {key}"
        );
        require!(
            classification.get("reason").is_some_and(Value::truthy),
            "research source reason"
        );
    }

    Ok(Summary {
        test_items: actual.len(),
        research_sources: src_items.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn write_minimal_repo(root: &Path) {
        fs::create_dir_all(root.join("deploy/ci")).unwrap();
        fs::create_dir_all(root.join("docs/internal/core-ip/findings")).unwrap();
        fs::create_dir_all(root.join("deploy/dl-flywheel")).unwrap();

        let src_bytes = b"# synthetic research helper\n";
        let src_sha = py::sha256_hex(src_bytes);
        fs::write(root.join("deploy/dl-flywheel/helper.py"), src_bytes).unwrap();

        let coverage = r#"{
          "lanes": {
            "rust-test": { "stage": "execution" }
          },
          "items": {
            "target:jope-policy/test/t1": {
              "discovery": { "kind": "target", "test": true },
              "lanes": ["rust-test"]
            }
          }
        }"#;
        fs::write(root.join("deploy/ci/coverage.json"), coverage).unwrap();

        let inventory = r#"{
          "schema": "jope.regression-test-inventory/v1",
          "subsystems": {
            "preparation": {},
            "encoders": {},
            "policy": {},
            "gp": {},
            "fitting": {},
            "teaching": {},
            "training": {},
            "storage": {},
            "export": {}
          },
          "items": {
            "target:jope-policy/test/t1": {
              "role": "regression",
              "subsystems": ["preparation", "encoders", "policy", "gp", "fitting", "teaching", "training", "storage", "export"]
            }
          }
        }"#;
        fs::write(
            root.join("docs/internal/core-ip/findings/test-inventory.json"),
            inventory,
        )
        .unwrap();

        let sources = format!(
            r#"{{
          "schema": "jope.flywheel-research-source-classification/v1",
          "items": {{
            "python:deploy/dl-flywheel/helper.py": {{
              "bytes": {},
              "sha256": "{src_sha}",
              "classification": {{
                "role": "research",
                "reason": "research helper"
              }}
            }}
          }}
        }}"#,
            src_bytes.len()
        );
        fs::write(
            root.join("deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json"),
            sources,
        )
        .unwrap();
    }

    #[test]
    fn accepts_valid_minimal_repo() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_repo(dir.path());
        let summary = verify(dir.path()).unwrap();
        assert_eq!(
            summary,
            Summary {
                test_items: 1,
                research_sources: 1
            }
        );
    }

    #[test]
    fn rejects_tampered_research_source_hash() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_repo(dir.path());
        fs::write(dir.path().join("deploy/dl-flywheel/helper.py"), b"tampered").unwrap();
        let err = verify(dir.path()).unwrap_err().to_string();
        assert!(err.contains("research source hash: python:deploy/dl-flywheel/helper.py"));
    }

    #[test]
    fn rejects_parent_traversal_in_research_source_key() {
        let dir = tempfile::tempdir().unwrap();
        write_minimal_repo(dir.path());
        let bad = r#"{
          "schema": "jope.flywheel-research-source-classification/v1",
          "items": {
            "python:../outside.py": {
              "bytes": 0,
              "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "classification": { "role": "research", "reason": "x" }
            }
          }
        }"#;
        fs::write(
            dir.path()
                .join("deploy/dl-flywheel/RESEARCH_SOURCE_CLASSIFICATIONS.json"),
            bad,
        )
        .unwrap();
        let err = verify(dir.path()).unwrap_err().to_string();
        assert_eq!(err, "source path confinement");
    }
}
