//! W1-T04: fitting inputs, scoring labels and result release are separate.
//!
//! Upstream producers (audit, preparation, views) may read the sealed final
//! test's *features* where their contract says so, never its labels. A scorer
//! reads only its declared scoring labels, for an authenticated frozen subject,
//! and only under a grant from the separate outer release authority, which also
//! decides whether a computed score becomes visible.

#[path = "support/legacy_dataset.rs"]
mod legacy;

use legacy::{Fixture, Spec};
use serde_json::{json, Value};
use std::fs;

fn audit_json(dir: &std::path::Path) -> Value {
    serde_json::from_slice(&fs::read(dir.join("audit.json")).unwrap()).unwrap()
}

/// R7 (consumer map): `audit.json` published the sealed test label value set
/// and label-derived statistics. The audit may read the sealed split's feature
/// columns (it reports original-split feature overlap); its labels are never
/// parsed, so no audit output can depend on them.
#[test]
fn audit_never_parses_or_publishes_sealed_test_labels() {
    let mut f = Fixture::new(Spec::default());
    let baseline = f.audit("baseline").unwrap();
    let report = audit_json(&baseline);
    let test = &report["datasets"][0]["test"];
    assert_eq!(
        test["target_values"],
        json!([]),
        "audit.json publishes the sealed test label value set: {test}"
    );
    for label in f.spec.test.iter().map(|r| r.scaled()[3].to_string()) {
        assert!(
            !fs::read_to_string(baseline.join("audit.json"))
                .unwrap()
                .contains(&format!("\"{label}\"")),
            "sealed label {label} appears in audit.json"
        );
    }
    // Dimensions the preparation contract consumes are still reported.
    assert_eq!(test["rows"], 10);
    assert_eq!(test["columns"], 4);
    assert_eq!(
        report["sealed_test_access"]["labels_parsed"], false,
        "the audit does not declare how it read the sealed split"
    );

    // Forbidden-only perturbation: every sealed label changes and the rows are
    // reordered. The audit report is byte-identical; only the transfer
    // manifest's content hash of the sealed file (an identifier of the changed
    // input) differs.
    for (i, row) in f.spec.test.iter_mut().enumerate() {
        row.y = 219.0 - i as f64 * 1.5;
    }
    f.spec.test.reverse();
    f.write();
    let changed = f.audit("sealed-labels-changed").unwrap();
    assert_eq!(
        fs::read(baseline.join("audit.json")).unwrap(),
        fs::read(changed.join("audit.json")).unwrap(),
        "an audit output depends on sealed test labels"
    );
    assert_ne!(
        fs::read(baseline.join("transfer.json")).unwrap(),
        fs::read(changed.join("transfer.json")).unwrap()
    );

    // Authorized control: the audit does read sealed *features*. A test row
    // that repeats a training row's features is reported as overlap.
    let copied = f.spec.source[f.spec.train[3]].clone();
    f.spec.test[0] = legacy::Row {
        y: f.spec.test[0].y,
        ..copied
    };
    f.write();
    let overlap = audit_json(&f.audit("sealed-feature-overlap").unwrap());
    assert_eq!(overlap["datasets"][0]["split_feature_overlap"], 1);
    assert_eq!(report["datasets"][0]["split_feature_overlap"], 0);
}
