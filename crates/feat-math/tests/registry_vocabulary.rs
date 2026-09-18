//! ADR-0002 §3: the registry (op executability authority) and the searchable vocabularies live in one
//! repository and must agree. Every op the restricted product vocabulary can generate is executable and
//! not quarantined in the registry, and the committed executable-op snapshot names exactly the registry's
//! executable set — so a registry change cannot land without regenerating the snapshot.
use feat_math::registry::{operator, operator_registry};
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::PathBuf;

fn grammar_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar")
}

fn read_json(name: &str) -> Value {
    let path = grammar_dir().join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

#[test]
fn restricted_vocabulary_ops_are_executable_in_the_registry() {
    let vocab = read_json("featmath-jain-v1.json");
    let ops = vocab["ops"].as_object().expect("ops map");
    let mut checked = 0;
    for (token, op) in ops {
        if op["engine"] != "featmath" || op["generate"] != true {
            continue;
        }
        let name = op["name"].as_str().unwrap_or_else(|| panic!("op {token} has no name"));
        let descriptor = operator(name).unwrap_or_else(|| panic!("vocabulary op {name} (token {token}) is not in the registry"));
        assert!(descriptor.lifecycle.executable, "vocabulary op {name} is not executable in the registry");
        assert!(!descriptor.lifecycle.quarantined, "vocabulary op {name} is quarantined in the registry");
        checked += 1;
    }
    assert!(checked >= 40, "expected the restricted vocabulary to carry dozens of generatable featmath ops, saw {checked}");
}

#[test]
fn executable_snapshot_matches_the_registry() {
    let snapshot = read_json("jain-math-executable-9fa594e2.json");
    let snapshot_exec: BTreeSet<String> = snapshot["executable"]
        .as_array()
        .expect("executable")
        .iter()
        .map(|e| e["name"].as_str().expect("name").to_owned())
        .collect();
    let snapshot_not: BTreeSet<String> = snapshot["not_executable"]
        .as_array()
        .expect("not_executable")
        .iter()
        .map(|e| e["name"].as_str().expect("name").to_owned())
        .collect();
    let registry_exec: BTreeSet<String> = operator_registry()
        .iter()
        .filter(|d| d.lifecycle.executable && !d.lifecycle.quarantined)
        .map(|d| d.name.to_owned())
        .collect();
    let registry_not: BTreeSet<String> = operator_registry()
        .iter()
        .filter(|d| !(d.lifecycle.executable && !d.lifecycle.quarantined))
        .map(|d| d.name.to_owned())
        .collect();
    assert_eq!(snapshot_exec, registry_exec, "executable set drifted: regenerate the snapshot with `cargo run -p feat-math --example registry_dump`");
    assert_eq!(snapshot_not, registry_not, "non-executable set drifted: regenerate the snapshot");
    assert_eq!(snapshot["provenance"]["registry_hash"], feat_math::registry::registry_hash(), "registry hash drifted");
}
