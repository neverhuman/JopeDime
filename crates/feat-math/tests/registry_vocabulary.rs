//! ADR-0002 §3: the registry (op executability authority) and the searchable vocabularies live in one
//! repository and must agree. Every op the restricted product vocabulary can generate is executable and
//! not quarantined in the registry, and the committed executable-op snapshot names exactly the registry's
//! executable set — so a registry change cannot land without regenerating the snapshot.
use feat_math::registry::{operator, operator_registry};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::PathBuf;

fn grammar_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar")
}

fn read_json(name: &str) -> Value {
    let path = grammar_dir().join(name);
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
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
        let name = op["name"]
            .as_str()
            .unwrap_or_else(|| panic!("op {token} has no name"));
        let descriptor = operator(name).unwrap_or_else(|| {
            panic!("vocabulary op {name} (token {token}) is not in the registry")
        });
        assert!(
            descriptor.lifecycle.executable,
            "vocabulary op {name} is not executable in the registry"
        );
        assert!(
            !descriptor.lifecycle.quarantined,
            "vocabulary op {name} is quarantined in the registry"
        );
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
    assert_eq!(
        snapshot_not, registry_not,
        "non-executable set drifted: regenerate the snapshot"
    );
    assert_eq!(
        snapshot["provenance"]["registry_hash"],
        feat_math::registry::registry_hash(),
        "registry hash drifted"
    );
}

#[test]
fn grammar_ops_follow_the_authority_rule() {
    use feat_math::kernel::capabilities::{
        authority, grammar_violations, Authority, JAIN_V1_GRAMMAR_SHA256, V3_GRAMMAR_SHA256,
    };
    let mut files = std::fs::read_dir(grammar_dir())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("featmath") && name.ends_with(".json"))
        })
        .collect::<Vec<_>>();
    files.sort();
    for frozen in ["featmath.json", "featmath-jain-v1.json"] {
        assert!(
            files
                .iter()
                .any(|path| path.file_name().is_some_and(|name| name == frozen)),
            "missing frozen grammar {frozen}"
        );
    }
    for path in &files {
        let bytes = std::fs::read(path).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let grammar: Value = serde_json::from_slice(&bytes).unwrap();
        match path.file_name().and_then(|name| name.to_str()) {
            Some("featmath.json") => assert_eq!(hash, V3_GRAMMAR_SHA256),
            Some("featmath-jain-v1.json") => assert_eq!(hash, JAIN_V1_GRAMMAR_SHA256),
            _ => assert_eq!(authority(&hash), Authority::Manifest),
        }
        assert!(
            grammar_violations(&hash, &grammar).is_empty(),
            "{}: {:?}",
            path.display(),
            grammar_violations(&hash, &grammar)
        );
    }
    assert_eq!(authority(V3_GRAMMAR_SHA256), Authority::Registry);
    assert_eq!(authority(JAIN_V1_GRAMMAR_SHA256), Authority::Registry);
    let synthetic_hash = "0".repeat(64);
    assert_eq!(authority(&synthetic_hash), Authority::Manifest);
    for (name, passes) in [("map", true), ("count_sketch", false), ("norm_row", false)] {
        let grammar = serde_json::json!({"ops":{"999":{"name":name,"generate":true}}});
        let violations = grammar_violations(&synthetic_hash, &grammar);
        assert_eq!(violations.is_empty(), passes, "{name}: {violations:?}");
        if !passes {
            assert!(violations.iter().any(|v| v.element == name));
        }
    }
    let mut frozen_clone = read_json("featmath.json");
    frozen_clone["ops"]["999"] = serde_json::json!({
        "name":"lag", "role":"transform", "generate":true, "engine":"featmath"
    });
    let violations = grammar_violations(V3_GRAMMAR_SHA256, &frozen_clone);
    assert!(
        violations.iter().any(|v| v.element == "lag"),
        "{violations:?}"
    );
}
