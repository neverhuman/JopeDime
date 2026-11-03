use super::*;
use jope_parallel::family_growth::{self, GrowthError};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

/// The engine-side growth fixture: a `GrowthFixture` over the frozen registry at `parent_path`
/// whose new component is renamed to the first `development-<i>` id the component bucket rule
/// sends to `development_validation`, then assigned by the policy and written to
/// `growth-assignment.json` beside it. References are absolute, as the Python fixture wrote them.
fn growth_fixture(f: &Fixture, parent_path: &Path) -> (PathBuf, PathBuf, String) {
    let root = f.dir.path().canonicalize().unwrap();
    let parent: Value = serde_json::from_slice(&fs::read(parent_path).unwrap()).unwrap();
    let mut g = GrowthFixture::new(&root, Some(parent), true);
    let cid = (0..1000)
        .map(|i| format!("development-{i}"))
        .find(|c| family_growth::stable_assignment_v2(c).2 == "development_validation")
        .unwrap();
    let last = g.graph["components"]
        .as_array_mut()
        .unwrap()
        .last_mut()
        .unwrap();
    last["component_id"] = json!(cid);
    last["aliases"] = json!([cid, "new-alias"]);
    g.graph_ref = g.w.put("graph.json", &g.graph.clone());
    g.decision["component_id"] = json!(cid);
    g.decision["graph_sha256"] = g.graph_ref["sha256"].clone();
    g.sign();
    let assignment = assign(&root, &g.inputs(&[cid.as_str(), "new-alias"])).unwrap();
    g.w.put("growth-assignment.json", &assignment);
    (
        root.join("parent.json"),
        root.join("growth-assignment.json"),
        cid,
    )
}

#[test]
fn authenticated_component_growth_preserves_panels_and_is_membership_only() {
    let f = Fixture::new();
    let (original, before) = f.freeze("growth-base", &[], &[], None);
    let (parent, assignment, component) = growth_fixture(&f, &original);
    let output = f.dir.path().join("grown.json");
    let grown = families::freeze_growth(&parent, &assignment, &output).unwrap();
    for (family, prior) in &before.families {
        assert_eq!(
            serde_json::to_value(&grown.families[family]).unwrap(),
            serde_json::to_value(prior).unwrap()
        );
    }
    let new = &grown.families[&component];
    assert_eq!(new.partition, "development_validation");
    assert_eq!(new.assignment_method, "stable_sha256_component_v2");
    assert!(new.datasets.is_empty());
    assert_eq!(
        grown.quota_report()["remaining_families_by_partition"],
        before.quota_report()["remaining_families_by_partition"]
    );
    assert!(families::binding(&output, "new", "new-alias", &component).is_err());
    let mut forged = grown.clone();
    forged.families.get_mut(&component).unwrap().partition = "meta_test".into();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    forged.families.get_mut("family-c").unwrap().partition = "meta_train".into();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    forged.growth.clear();
    assert!(forged.validate().is_err());
    let mut forged = grown.clone();
    let dataset = before.families["family-a"].datasets[0].clone();
    forged
        .families
        .get_mut(&component)
        .unwrap()
        .datasets
        .push(dataset);
    assert!(forged.validate().is_err());
}

#[test]
fn component_growth_refuses_forged_summaries_changed_helper_and_stale_parent() {
    let f = Fixture::new();
    let (original, _) = f.freeze("growth-base", &[], &[], None);
    let (parent, assignment, _) = growth_fixture(&f, &original);
    let genuine = fs::read(&assignment).unwrap();
    let mut value: Value = read_json(&assignment).unwrap();
    value["n_test"] = json!(1);
    write_json(&assignment, &value).unwrap();
    assert!(
        families::freeze_growth(&parent, &assignment, &f.dir.path().join("forged.json")).is_err()
    );
    // The verifier is compiled into the release. A helper planted where the retired Python
    // verifier used to be looked up, printing the forged summary, is never consulted: it can
    // neither authorize the forgery nor block the genuine receipt.
    fs::create_dir_all(f.dir.path().join("tools")).unwrap();
    fs::write(
        f.dir.path().join("tools/family_growth_policy.py"),
        format!("print({:?})", serde_json::to_string(&value).unwrap()),
    )
    .unwrap();
    assert!(families::freeze_growth(
        &parent,
        &assignment,
        &f.dir.path().join("helper-changed.json")
    )
    .is_err());
    assert!(!f.dir.path().join("helper-changed.json").exists());
    fs::write(&assignment, &genuine).unwrap();
    assert!(families::freeze_growth(
        &parent,
        &assignment,
        &f.dir.path().join("helper-ignored.json")
    )
    .is_ok());
    let mut stale: Value = read_json(&parent).unwrap();
    stale["registry_id"] = json!("changed");
    let stale_path = f.dir.path().join("stale-parent.json");
    write_json(&stale_path, &stale).unwrap();
    assert!(
        families::freeze_growth(&stale_path, &assignment, &f.dir.path().join("stale.json"))
            .is_err()
    );
}

// ---- the grown-family policy against the retired Python implementation ----------------------
//
// `family_growth` is the native port of `crates/jope-engine/policy/family_growth_policy.py`
// (archived at 8a55c4fef; restore with `git checkout 8a55c4fef -- <path>`). Before that file
// was deleted, `capture_growth_policy_outputs_from_python` ran it (`python3 -I -B <policy>
// --verify <request>`, exactly as `families::freeze_growth` used to) on every scenario below and
// recorded its outcome in `tests/fixtures/family-growth/python-verify-outputs.jsonl`. The
// scenarios are the cases of the Python module's own test file (`existing/*`) and a fixed-seed
// random set (`random/*`). `growth_policy_matches_the_captured_python_outputs` replays them
// against the native verifier, so the parity outlives the Python.

/// Differential proof, replayed: the native verifier reproduces the outcome the Python policy
/// produced on every captured scenario, byte for byte on success and with the same refusal
/// message otherwise.
#[test]
fn growth_policy_matches_the_captured_python_outputs() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(PARITY_FIXTURE);
    let records: Vec<Value> = fs::read_to_string(&fixture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let names = scenario_names();
    assert_eq!(
        records
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        names,
        "the captured scenario list differs from this test's"
    );
    let (mut accepted, mut refused) = (0, 0);
    for (name, record) in names.iter().zip(&records) {
        let dir = tempfile::tempdir().unwrap();
        let (root, request) = build_scenario(name, dir.path(), false);
        assert_eq!(
            scenario_digest(dir.path(), &root, &request),
            record["inputs_sha256"].as_str().unwrap(),
            "{name}: the scenario inputs differ from the captured ones"
        );
        let outcome = family_growth::verify_assignment(&request);
        if let Err(why) = agrees(&outcome, &root, record) {
            panic!("{name}: {why}");
        }
        match outcome {
            Ok(_) => accepted += 1,
            Err(GrowthError::Refused(_)) => refused += 1,
            Err(_) => {}
        }
    }
    // The set is not vacuous: it grows families and refuses forgeries.
    assert!(
        accepted >= 30 && refused >= 30,
        "accepted {accepted}, refused {refused}"
    );
}
