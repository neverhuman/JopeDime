//! MR12 capability join audit (partial). Every registry_v2 entry must carry a known
//! dialect, admission, and target classification, and each enabled entry must support
//! every target it declares. The join to OperationProfile, native/challenger policy
//! masks, evaluators and emitters is NOT covered here: the source has no table that
//! maps registry entries to those surfaces yet. That gap is reported, not hidden.

use feat_math::kernel::registry_v2::{Entry, OPERATORS};
use std::collections::BTreeSet;

const DIALECTS: &[&str] = &["advanced_v3", "legacy_recipe"];
const ADMISSIONS: &[&str] = &["compatibility", "disabled", "enabled", "quarantined", "receipt"];
const TARGETS: &[&str] = &[
    "cpp17_recipe",
    "javascript",
    "python_bound",
    "python_recipe",
    "runtime",
    "rust_bound",
    "rust_recipe",
];

fn unclassified(entry: &Entry) -> Vec<String> {
    let id = format!("{}/{}", entry.dialect, entry.name);
    let mut problems = Vec::new();
    if !DIALECTS.contains(&entry.dialect) {
        problems.push(format!("{id}: unknown dialect {}", entry.dialect));
    }
    if !ADMISSIONS.contains(&entry.admission) {
        problems.push(format!("{id}: unknown admission {}", entry.admission));
    }
    for target in entry.targets {
        if !TARGETS.contains(target) {
            problems.push(format!("{id}: unknown target {target}"));
        }
    }
    if entry.admission == "enabled" {
        if entry.targets.is_empty() {
            problems.push(format!("{id}: enabled with no targets"));
        }
        for target in entry.targets {
            if entry.supports(target).is_err() {
                problems.push(format!("{id}: enabled but unsupported target {target}"));
            }
        }
    }
    problems
}

#[test]
fn zero_unclassified_primitives_across_all_surfaces() {
    let mut seen = BTreeSet::new();
    let mut problems = Vec::new();
    for entry in OPERATORS {
        let key = (entry.dialect, entry.name);
        if !seen.insert(key) {
            problems.push(format!("duplicate entry {}/{}", entry.dialect, entry.name));
        }
        // facts() must succeed for every entry, including non-enabled ones.
        let _ = entry.facts();
        problems.extend(unclassified(entry));
    }
    assert_eq!(OPERATORS.len(), 90, "registry snapshot size changed");
    assert!(
        problems.is_empty(),
        "unclassified primitives ({}):\n{}",
        problems.len(),
        problems.join("\n")
    );
}
