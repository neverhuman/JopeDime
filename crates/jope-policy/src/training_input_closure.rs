//! Authenticate the projected data a learning worker may read, before model loading.
//!
//! Faithful Rust port of `jope.policy.training_input_closure`. Python remains in use by
//! `training_campaign` until that caller is migrated.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const COMMON: &[&str] = &[
    "corpus",
    "context-index",
    "encoder-identity",
    "vocabulary",
    "teaching-admission",
    "family-partition",
    "checkpoint",
];
const CONTEXT: &[&str] = &[
    "context-values",
    "context-producer",
    "fitting-input",
    "teaching-recipe",
];
const EXCLUDED: &[&str] = &[
    "outer-test",
    "sealed-meta-test",
    "outside-selection-evaluation",
];

fn require(ok: bool, message: &str) -> Result<(), String> {
    if ok {
        Ok(())
    } else {
        Err(message.to_owned())
    }
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("missing or invalid {key}"))
}

fn reference(value: &Value) -> Result<PathBuf, String> {
    require(
        value.is_object()
            && value.as_object().unwrap().len() == 2
            && value.get("path").and_then(Value::as_str).is_some()
            && value.get("sha256").and_then(Value::as_str).is_some(),
        "closure artifact reference required",
    )?;
    let path = PathBuf::from(string(value, "path")?);
    let canonical = fs::canonicalize(&path).map_err(|e| e.to_string())?;
    require(
        path.is_absolute() && canonical == path && path.is_file(),
        "closure path must be physical",
    )?;
    let bytes = fs::read(&path).map_err(|e| e.to_string())?;
    require(
        crate::sha256_hex(&bytes) == string(value, "sha256")?,
        "closure artifact changed",
    )?;
    Ok(path)
}

fn role_allowed(role: &str) -> bool {
    COMMON.contains(&role) || CONTEXT.contains(&role)
}

fn scan_metadata(
    value: &Value,
    inventory: &BTreeMap<String, Value>,
    pending: &mut Vec<String>,
) -> Result<(), String> {
    match value {
        Value::Object(object) if object.contains_key("path") => {
            require(
                object.len() == 2 && object.contains_key("sha256"),
                "unlisted path-bearing metadata",
            )?;
            let path = string(value, "path")?.to_owned();
            let item = inventory
                .get(&path)
                .ok_or_else(|| "unlisted transitive artifact".to_owned())?;
            require(
                item.get("artifact") == Some(value),
                "unlisted transitive artifact",
            )?;
            pending.push(path);
        }
        Value::Object(object) => {
            for child in object.values() {
                scan_metadata(child, inventory, pending)?;
            }
        }
        Value::Array(items) => {
            for child in items {
                scan_metadata(child, inventory, pending)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Verify and return the manifest that defines the worker's complete readable input closure.
/// Unlisted references are rejected before the referenced file is opened.
pub fn admit(inputs: &Value) -> Result<Value, String> {
    require(
        inputs.get("input_closure").is_some(),
        "projected input closure is required",
    )?;
    let manifest_path = reference(&inputs["input_closure"])?;
    let manifest_bytes = fs::read(&manifest_path).map_err(|e| e.to_string())?;
    let manifest: Value = serde_json::from_slice(&manifest_bytes).map_err(|e| e.to_string())?;
    require(
        manifest.get("schema").and_then(Value::as_str) == Some("jope.training-input-closure/v1"),
        "unsupported input closure",
    )?;

    let root = PathBuf::from(string(&manifest, "root")?);
    let canonical_root = fs::canonicalize(&root).map_err(|e| e.to_string())?;
    require(
        root.is_absolute()
            && canonical_root == root
            && root == manifest_path.parent().unwrap_or(Path::new("")),
        "closure root differs",
    )?;
    require(!root.starts_with("/tmp"), "closure root must be durable")?;

    let excluded: BTreeSet<&str> = manifest
        .get("excluded_roles")
        .and_then(Value::as_array)
        .ok_or_else(|| "closure exclusion domains differ".to_owned())?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    require(
        excluded == EXCLUDED.iter().copied().collect(),
        "closure exclusion domains differ",
    )?;

    let families = manifest
        .get("families")
        .and_then(Value::as_object)
        .ok_or_else(|| "closure family role is forbidden".to_owned())?;
    require(!families.is_empty(), "closure family role is forbidden")?;
    for family_role in families.values() {
        require(
            matches!(family_role.as_str(), Some("train" | "development")),
            "closure family role is forbidden",
        )?;
    }
    let contexts = manifest
        .get("contexts")
        .and_then(Value::as_array)
        .ok_or_else(|| "closure contexts absent or duplicated".to_owned())?;
    let context_set: BTreeSet<&str> = contexts.iter().filter_map(Value::as_str).collect();
    require(
        !contexts.is_empty() && context_set.len() == contexts.len(),
        "closure contexts absent or duplicated",
    )?;

    let artifacts = manifest
        .get("artifacts")
        .and_then(Value::as_array)
        .ok_or_else(|| "forbidden closure artifact role".to_owned())?;
    let mut inventory = BTreeMap::<String, Value>::new();
    for item in artifacts {
        let artifact = item
            .get("artifact")
            .ok_or_else(|| "closure artifact reference required".to_owned())?;
        let role = string(item, "role")?;
        require(role_allowed(role), "forbidden closure artifact role")?;
        let path = PathBuf::from(string(artifact, "path")?);
        require(
            path.parent() == Some(root.as_path())
                && !inventory.contains_key(path.to_string_lossy().as_ref()),
            "closure artifact escapes root or is duplicated",
        )?;
        if CONTEXT.contains(&role) {
            let split = item.get("split").and_then(Value::as_str);
            let source_family = item.get("source_family").and_then(Value::as_str);
            require(
                matches!(split, Some("train" | "development"))
                    && source_family
                        .and_then(|f| families.get(f))
                        .and_then(Value::as_str)
                        == split
                    && item
                        .get("context_id")
                        .and_then(Value::as_str)
                        .is_some_and(|c| context_set.contains(c)),
                "closure context family/split differs",
            )?;
        }
        inventory.insert(path.to_string_lossy().into_owned(), item.clone());
    }

    let input_roles = [
        ("corpus", "corpus"),
        ("contexts", "context-index"),
        ("encoder", "encoder-identity"),
        ("vocabulary", "vocabulary"),
        ("admission", "teaching-admission"),
        ("checkpoint", "checkpoint"),
    ];
    let allowed_inputs: BTreeSet<&str> = input_roles.iter().map(|(name, _)| *name).collect();
    require(
        inputs.as_object().is_some_and(|o| {
            o.keys()
                .all(|k| k == "input_closure" || allowed_inputs.contains(k.as_str()))
        }),
        "unlisted input kind",
    )?;
    for (name, role) in &input_roles {
        if let Some(reference_value) = inputs.get(*name) {
            let path = string(reference_value, "path")?;
            let item = inventory
                .get(path)
                .ok_or_else(|| "input is outside its declared closure role".to_owned())?;
            require(
                item.get("artifact") == Some(reference_value)
                    && item.get("role").and_then(Value::as_str) == Some(*role),
                "input is outside its declared closure role",
            )?;
        }
    }

    let mut expected: BTreeSet<String> = inventory.keys().cloned().collect();
    expected.insert(manifest_path.to_string_lossy().into_owned());
    expected.insert(root.join("inputs.json").to_string_lossy().into_owned());
    for entry in fs::read_dir(&root).map_err(|e| e.to_string())? {
        let path = entry
            .map_err(|e| e.to_string())?
            .path()
            .to_string_lossy()
            .into_owned();
        require(
            expected.contains(&path),
            "undeclared file in worker input root",
        )?;
    }

    let mut pending: Vec<String> = inputs
        .as_object()
        .into_iter()
        .flat_map(|o| o.iter())
        .filter(|(name, _)| name.as_str() != "input_closure")
        .filter_map(|(_, reference)| {
            reference
                .get("path")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
        .collect();
    let mut visited = BTreeSet::new();
    while let Some(name) = pending.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let item = inventory
            .get(&name)
            .ok_or_else(|| "unlisted transitive artifact".to_owned())?;
        let path = reference(&item["artifact"])?;
        if item.get("role").and_then(Value::as_str) == Some("checkpoint") {
            continue;
        }
        if item.get("role").and_then(Value::as_str) == Some("corpus") {
            let file = fs::File::open(path).map_err(|e| e.to_string())?;
            for line in BufReader::new(file).lines() {
                let line = line.map_err(|e| e.to_string())?;
                if !line.trim().is_empty() {
                    let value: Value = serde_json::from_str(&line).map_err(|e| e.to_string())?;
                    scan_metadata(&value, &inventory, &mut pending)?;
                }
            }
        } else {
            let bytes = fs::read(path).map_err(|e| e.to_string())?;
            let value: Value = serde_json::from_slice(&bytes).map_err(|e| e.to_string())?;
            scan_metadata(&value, &inventory, &mut pending)?;
        }
    }
    require(
        visited.len() == inventory.len(),
        "unused declared artifact in worker closure",
    )?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static NEXT: AtomicUsize = AtomicUsize::new(0);

    struct Fixture(PathBuf);

    impl Fixture {
        fn new() -> Self {
            let fixture_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../target/jope-policy-training-closure-tests")
                .join(format!(
                    "{}-{}-{}",
                    std::process::id(),
                    NEXT.fetch_add(1, Ordering::Relaxed),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos()
                ));
            let root = fixture_dir.join("root");
            fs::create_dir_all(&root).unwrap();
            Self(fs::canonicalize(root).unwrap())
        }

        fn write(&self, name: &str, bytes: &[u8]) -> Value {
            let path = self.0.join(name);
            fs::write(&path, bytes).unwrap();
            json!({"path": path.to_string_lossy(), "sha256": crate::sha256_hex(bytes)})
        }

        fn projected(&self) -> (Value, Value) {
            let corpus = self.write("corpus.jsonl", b"{}\n");
            let manifest = json!({
                "schema": "jope.training-input-closure/v1",
                "root": self.0,
                "excluded_roles": EXCLUDED,
                "families": {"family-1": "train"},
                "contexts": ["context-1"],
                "artifacts": [{"artifact": corpus, "role": "corpus"}],
            });
            let manifest_path = self.0.join("input-closure.json");
            let manifest_bytes = serde_json::to_vec(&manifest).unwrap();
            fs::write(&manifest_path, &manifest_bytes).unwrap();
            let closure = json!({"path": manifest_path.to_string_lossy(), "sha256": crate::sha256_hex(&manifest_bytes)});
            (
                json!({"corpus": corpus, "input_closure": closure}),
                manifest,
            )
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            if let Some(fixture_dir) = self.0.parent() {
                let _ = fs::remove_dir_all(fixture_dir);
            }
        }
    }

    #[test]
    fn projected_closure_is_invariant_to_excluded_artifact_mutation() {
        let fixture = Fixture::new();
        let (inputs, manifest) = fixture.projected();
        fs::write(
            fixture.0.parent().unwrap().join("excluded-fixture.json"),
            b"{\"fixture_label\":1}",
        )
        .unwrap();
        assert_eq!(admit(&inputs).unwrap(), manifest);
        fs::write(
            fixture.0.parent().unwrap().join("excluded-fixture.json"),
            b"{\"fixture_label\":999}",
        )
        .unwrap();
        assert_eq!(admit(&inputs).unwrap(), manifest);
    }

    #[test]
    fn unlisted_reachable_artifact_is_refused_before_open() {
        let fixture = Fixture::new();
        let (inputs, mut manifest) = fixture.projected();
        let forbidden = fixture.0.parent().unwrap().join("must-not-open.json");
        let encoder = fixture.write(
            "encoder.json",
            format!(
                "{{\"nested\":{{\"path\":{:?},\"sha256\":\"{}\"}}}}",
                forbidden.to_string_lossy(),
                "a".repeat(64)
            )
            .as_bytes(),
        );
        let item = json!({"artifact": encoder, "role": "encoder-identity"});
        manifest["artifacts"].as_array_mut().unwrap().push(item);
        let manifest_path = PathBuf::from(inputs["input_closure"]["path"].as_str().unwrap());
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(&manifest_path, &bytes).unwrap();
        let changed_inputs = json!({
            "corpus": inputs["corpus"],
            "encoder": manifest["artifacts"][1]["artifact"],
            "input_closure": {"path": manifest_path.to_string_lossy(), "sha256": crate::sha256_hex(&bytes)},
        });
        assert_eq!(
            admit(&changed_inputs).unwrap_err(),
            "unlisted transitive artifact"
        );
        assert!(!forbidden.exists());
    }

    #[test]
    fn projection_rejects_forbidden_or_changed_closure() {
        for corruption in ["role", "family", "extra-file", "outside-root", "hash"] {
            let fixture = Fixture::new();
            let (inputs, mut manifest) = fixture.projected();
            match corruption {
                "role" => manifest["artifacts"][0]["role"] = json!("sealed-meta-test"),
                "family" => {
                    manifest["families"]["family-1"] = json!("outside-selection-evaluation")
                }
                "extra-file" => fs::write(fixture.0.join("unlisted.json"), b"{}").unwrap(),
                "outside-root" => {
                    manifest["artifacts"][0]["artifact"] =
                        json!({"path":"/must-not-read","sha256":"a".repeat(64)})
                }
                "hash" => fs::write(fixture.0.join("corpus.jsonl"), b"changed\n").unwrap(),
                _ => unreachable!(),
            }
            let manifest_path = PathBuf::from(inputs["input_closure"]["path"].as_str().unwrap());
            let bytes = serde_json::to_vec(&manifest).unwrap();
            fs::write(&manifest_path, &bytes).unwrap();
            let changed_inputs = json!({"corpus": inputs["corpus"], "input_closure": {"path": manifest_path.to_string_lossy(), "sha256": crate::sha256_hex(&bytes)}});
            assert!(admit(&changed_inputs).is_err(), "corruption={corruption}");
        }
    }

    #[test]
    fn unused_declared_artifact_is_refused_without_opening() {
        let fixture = Fixture::new();
        let (inputs, mut manifest) = fixture.projected();
        let unused_path = fixture.0.join("unused.json");
        let unused =
            json!({"path": unused_path.to_string_lossy(), "sha256": crate::sha256_hex(b"unused")});
        manifest["artifacts"]
            .as_array_mut()
            .unwrap()
            .push(json!({"artifact": unused, "role": "family-partition"}));
        let manifest_path = PathBuf::from(inputs["input_closure"]["path"].as_str().unwrap());
        let bytes = serde_json::to_vec(&manifest).unwrap();
        fs::write(&manifest_path, &bytes).unwrap();
        let changed_inputs = json!({"corpus": inputs["corpus"], "input_closure": {"path": manifest_path.to_string_lossy(), "sha256": crate::sha256_hex(&bytes)}});
        assert_eq!(
            admit(&changed_inputs).unwrap_err(),
            "unused declared artifact in worker closure"
        );
        assert!(!unused_path.exists());
    }
}
