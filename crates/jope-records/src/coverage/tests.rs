use super::discovery::relative;
use super::*;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn mapping() -> Value {
    py::load_json(&repo().join(MAPPING)).unwrap()
}

/// The discovery the map itself records (no Cargo or git needed).
fn recorded(mapping: &Value) -> Dict {
    mapping
        .get("items")
        .and_then(Value::as_dict)
        .unwrap()
        .iter()
        .map(|(key, item)| (key.to_owned(), item.get("discovery").unwrap().clone()))
        .collect()
}

fn section<'a>(mapping: &'a mut Value, name: &str) -> &'a mut Dict {
    mapping
        .as_dict_mut()
        .unwrap()
        .get_mut(name)
        .unwrap()
        .as_dict_mut()
        .unwrap()
}

fn entry<'a>(dict: &'a mut Dict, key: &str) -> &'a mut Dict {
    dict.get_mut(key).unwrap().as_dict_mut().unwrap()
}

#[test]
fn current_mapping() {
    let m = mapping();
    assert_eq!(validate(&m, &recorded(&m)).unwrap(), Vec::<String>::new());
}

#[test]
fn each_unassigned_kind_fails() {
    let m = mapping();
    for kind in [
        "package",
        "feature",
        "target",
        "python-family",
        "workflow",
        "workflow-job",
        "benchmark-source",
    ] {
        let mut discovered = recorded(&m);
        discovered.insert(
            format!("new:{kind}"),
            Value::Dict(super::discovery::kind(kind)),
        );
        assert!(validate(&m, &discovered)
            .unwrap()
            .contains(&format!("unassigned addition: new:{kind}")));
    }
}

#[test]
fn changed_target_requirements_fail() {
    let m = mapping();
    let mut discovered = recorded(&m);
    let key = discovered
        .keys()
        .find(|k| k.starts_with("target:"))
        .unwrap()
        .to_owned();
    entry(&mut discovered, &key).insert(
        "required_features",
        Value::List(vec![Value::from("new-capability")]),
    );
    assert!(validate(&m, &discovered)
        .unwrap()
        .iter()
        .any(|e| e.contains("changed discovery")));
}

#[test]
fn required_feature_target_cannot_claim_default_execution() {
    let base = mapping();
    let discovered = recorded(&base);
    let key = "target:jope-policy/test/jope_dime";
    for (lane, disposition) in [
        (Some("rust-check"), None),
        (Some("rust-test"), None),
        (None, Some("full-cpu-default")),
    ] {
        let mut m = base.clone();
        if let Some(lane) = lane {
            // Exercise a default-only lane even when production CI has gained an explicit
            // feature selection for other targets.
            entry(section(&mut m, "lanes"), lane)
                .insert("feature_selection", Value::from("package-defaults"));
        }
        let item = entry(section(&mut m, "items"), key);
        item.insert(
            "lanes",
            Value::List(lane.map(|l| vec![Value::from(l)]).unwrap_or_default()),
        );
        item.insert(
            "dispositions",
            Value::List(
                disposition
                    .map(|d| vec![Value::from(d)])
                    .unwrap_or_default(),
            ),
        );
        assert!(
            validate(&m, &discovered)
                .unwrap()
                .iter()
                .any(|e| e.contains("non-default required features")),
            "{lane:?} {disposition:?}"
        );
    }
    let expected: BTreeSet<String> = ["default", "hyperion-cpu", "batch-worker"]
        .map(String::from)
        .into();
    assert_eq!(default_features(&discovered, "hyperion").unwrap(), expected);
}

/// `Path(p).relative_to(ROOT)`: a metadata path spelled with POSIX's distinct `//` root is
/// outside the resolved root; one spelled `///`, or with `.` and repeated separators, is not.
#[test]
fn metadata_paths_are_relative_to_the_root_as_pathlib_compares_them() {
    let root = Path::new("/r");
    assert_eq!(relative("/r/Cargo.toml", root, "m").unwrap(), "Cargo.toml");
    assert_eq!(
        relative("///r//src/./main.rs", root, "m").unwrap(),
        "src/main.rs"
    );
    assert_eq!(relative("/r/", root, "m").unwrap(), ".");
    let refused = relative("//r/Cargo.toml", root, "m")
        .unwrap_err()
        .to_string();
    assert!(
        refused.contains("ValueError: '//r/Cargo.toml' is not in the subpath of '/r'"),
        "{refused}"
    );
    assert!(relative("/rx/Cargo.toml", root, "m").is_err());
    assert!(relative("r/Cargo.toml", root, "m").is_err());
}

#[test]
fn workflow_comments_flow_values_and_indentation() {
    let text = "name: proof\njobs: # comment\n    build: # comment\n      runs-on: ubuntu-latest\n    \"test\": {runs-on: ubuntu-latest}\n";
    assert_eq!(workflow_jobs(text).unwrap(), ["build", "test"]);
    for text in [
        "jobs: {build: {runs-on: ubuntu-latest}}",
        "jobs:\n  <<: *other\n",
        "jobs:\n  build:\n  build:\n",
        "jobs:\n",
    ] {
        assert!(workflow_jobs(text).is_err(), "{text:?}");
    }
}

#[test]
fn unassigned_and_stale_items_fail() {
    let mut m = mapping();
    let mut discovered = recorded(&m);
    let key = discovered.keys().next().unwrap().to_owned();
    let item = entry(section(&mut m, "items"), &key);
    item.insert("lanes", Value::List(vec![]));
    item.insert("dispositions", Value::List(vec![]));
    assert!(validate(&m, &discovered)
        .unwrap()
        .iter()
        .any(|e| e.contains("unassigned item")));
    discovered.remove(&key);
    assert!(validate(&m, &discovered)
        .unwrap()
        .iter()
        .any(|e| e.contains("stale assignment")));
}

#[test]
fn incomplete_disposition_and_lane_fail() {
    let mut m = mapping();
    let discovered = recorded(&m);
    section(&mut m, "dispositions")
        .values_mut()
        .next()
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .insert("dependencies", Value::List(vec![]));
    section(&mut m, "lanes")
        .values_mut()
        .next()
        .unwrap()
        .as_dict_mut()
        .unwrap()
        .remove("requires_nonzero_tests");
    assert!(validate(&m, &discovered).unwrap().len() >= 2);
}

#[test]
fn every_lane_declares_feature_selection() {
    let mut m = mapping();
    let discovered = recorded(&m);
    entry(section(&mut m, "lanes"), "rust-check").remove("feature_selection");
    assert!(validate(&m, &discovered)
        .unwrap()
        .iter()
        .any(|e| e.contains("feature selection")));
}

/// `deploy/ci/test-inventory.sh`: a failing uv collection keeps its exit status and never
/// falls back to the static scan.
#[test]
fn collection_failure_preserves_exit_without_static_fallback() {
    for (phase, code) in [("sync", 23), ("run", 2)] {
        let scratch = tempfile::tempdir().unwrap();
        let root = scratch.path();
        std::fs::create_dir_all(root.join("deploy/ci")).unwrap();
        std::fs::create_dir(root.join("python")).unwrap();
        std::fs::write(root.join("python/uv.lock"), "").unwrap();
        std::fs::copy(
            repo().join("deploy/ci/test-inventory.sh"),
            root.join("deploy/ci/test-inventory.sh"),
        )
        .unwrap();
        let uv = root.join("uv");
        std::fs::write(
                &uv,
                format!("#!/bin/sh\nif [ \"$1\" = {phase} ]; then echo collection-fixture >&2; exit {code}; fi\n"),
            )
            .unwrap();
        std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
        let path = format!(
            "{}:{}",
            root.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        let result = std::process::Command::new("bash")
            .args(["deploy/ci/test-inventory.sh", "--no-rust"])
            .current_dir(root)
            .env("PATH", path)
            .output()
            .unwrap();
        assert_eq!(result.status.code(), Some(code), "{phase}");
        assert!(String::from_utf8_lossy(&result.stderr).contains("collection-fixture"));
        assert!(!String::from_utf8_lossy(&result.stdout).contains("static"));
        assert!(!root.join("docs/parity/test-inventory.json").exists());
    }
}
