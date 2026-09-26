//! Port of `test_verify_test_inventory.py`.

use jope_records::layout;
use jope_records::pyjson::Value;
use jope_records::test_inventory::verify;

use crate::{assert_refused, at, dump, load, nth, pop, repo, scratch, set, strs};

#[test]
fn committed_inventory() {
    let repo = repo();
    let result =
        verify(&layout::test_inventory(&repo), &repo).expect("committed inventory verifies");
    assert_eq!(result.baseline_failures, 33);
    for subsystem in [
        "preparation",
        "encoders",
        "policy",
        "gp",
        "fitting",
        "teaching",
        "training",
        "storage",
        "export",
    ] {
        let count = result
            .executing_regression_subsystems
            .get(subsystem)
            .copied()
            .unwrap_or(0);
        assert!(count > 0, "{subsystem} has no executing regression item");
    }
}

fn first_key(document: &Value, role: &str) -> String {
    document
        .get("items")
        .and_then(Value::as_dict)
        .and_then(|items| {
            items
                .iter()
                .find(|(_, v)| v.get("role") == Some(&Value::from(role)))
                .map(|(k, _)| k.to_owned())
        })
        .unwrap_or_else(|| panic!("no {role} item"))
}

fn first_triage(document: &Value, field: &str, value: &str) -> usize {
    document
        .get("baseline_triage")
        .and_then(Value::as_list)
        .and_then(|records| {
            records
                .iter()
                .position(|r| r.get(field) == Some(&Value::from(value)))
        })
        .unwrap_or_else(|| panic!("no triage record with {field} {value}"))
}

/// Remove every regression claim for one required subsystem.
fn only_subsystem(document: &mut Value, name: &str) {
    let target = Value::from(name);
    for item in at(document, "items")
        .as_dict_mut()
        .expect("items")
        .values_mut()
    {
        if item.get("role") != Some(&Value::from("regression")) {
            continue;
        }
        let subsystems = at(item, "subsystems")
            .as_list_mut()
            .expect("regression subsystems");
        if let Some(position) = subsystems.iter().position(|s| s == &target) {
            subsystems.remove(position);
            if subsystems.is_empty() {
                subsystems.push("orchestration".into());
            }
        }
    }
}

fn items(document: &mut Value) -> &mut Value {
    at(document, "items")
}

fn triage(document: &mut Value, index: usize) -> &mut Value {
    nth(at(document, "baseline_triage"), index)
}

#[test]
fn mutations_are_rejected() {
    let repo = repo();
    let document = load(&layout::test_inventory(&repo));
    let regression = first_key(&document, "regression");
    let historical = first_key(&document, "historical");
    let fence = first_triage(&document, "triage", "fit-authority-fence");
    let fixed = first_triage(&document, "status", "fixed-locally");

    type Mutate<'a> = Box<dyn Fn(&mut Value) + 'a>;
    let mutations: Vec<(&str, Mutate<'_>, &str)> = vec![
        (
            "unclassified item",
            Box::new(|d| drop(pop(items(d), &regression))),
            "unclassified",
        ),
        (
            "stale item",
            Box::new(|d| {
                set(
                    items(d),
                    "target:ghost/test/none",
                    crate::json(r#"{"role": "research", "reason": "x"}"#),
                );
            }),
            "stale",
        ),
        (
            "unknown role",
            Box::new(|d| set(at(items(d), &regression), "role", "passing".into())),
            ": role",
        ),
        (
            "unknown subsystem",
            Box::new(|d| set(at(items(d), &regression), "subsystems", strs(&["magic"]))),
            ": subsystems",
        ),
        (
            "regression without subsystem",
            Box::new(|d| set(at(items(d), &regression), "subsystems", Value::List(vec![]))),
            ": subsystems",
        ),
        (
            "non-regression without reason",
            Box::new(|d| drop(pop(at(items(d), &historical), "reason"))),
            ": reason",
        ),
        (
            "uncovered subsystem",
            Box::new(|d| only_subsystem(d, "teaching")),
            r"no executing regression item for \['teaching'\]",
        ),
        (
            "untriaged failure",
            Box::new(|d| {
                at(d, "baseline_triage")
                    .as_list_mut()
                    .expect("triage")
                    .remove(0);
            }),
            "untriaged baseline",
        ),
        (
            "duplicate triage",
            Box::new(|d| {
                let copy = triage(d, 0).clone();
                at(d, "baseline_triage")
                    .as_list_mut()
                    .expect("triage")
                    .push(copy);
            }),
            "duplicate triage",
        ),
        (
            "invented failure",
            Box::new(|d| set(triage(d, 0), "test", "imaginary".into())),
            "unknown baseline failure",
        ),
        (
            "fence marked fixed",
            Box::new(move |d| {
                set(triage(d, fence), "status", "fixed-locally".into());
                set(triage(d, fence), "evidence", strs(&["x"]));
            }),
            "cannot be marked fixed",
        ),
        (
            "fixed without evidence",
            Box::new(move |d| drop(pop(triage(d, fixed), "evidence"))),
            "without evidence",
        ),
        (
            "unknown status",
            Box::new(|d| set(triage(d, 0), "status", "passed".into())),
            ": status",
        ),
        (
            "subsystem without package",
            Box::new(|d| {
                set(
                    at(at(d, "subsystems"), "gp"),
                    "work_packages",
                    Value::List(vec![]),
                )
            }),
            "subsystem gp definition",
        ),
    ];
    for (label, mutate, reason) in mutations {
        let directory = scratch();
        let mut changed = document.clone();
        mutate(&mut changed);
        let path = directory.path().join("inventory.json");
        dump(&path, &changed);
        let result = verify(&path, &repo);
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn messages_name_the_python_identities() {
    // Addition: failure messages print Python reprs, so logs read the same as before the port.
    let repo = repo();
    let mut document = load(&layout::test_inventory(&repo));
    let triage = nth(at(&mut document, "baseline_triage"), 0);
    let lane = triage.get("lane").expect("lane").py_str();
    let class = triage.get("class").expect("class").py_str();
    set(triage, "test", "imaginary".into());
    let directory = scratch();
    let path = directory.path().join("inventory.json");
    dump(&path, &document);
    assert_refused(
        verify(&path, &repo),
        &format!(
            "^triage for unknown baseline failure {}$",
            regex::escape(&format!("('{lane}', '{class}', 'imaginary')"))
        ),
    );
}
