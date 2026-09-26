//! Port of `test_verify_findings.py`.

use jope_records::findings::verify;
use jope_records::layout;
use jope_records::pyjson::Value;

use crate::{assert_refused, at, copy_tree, dump, load, nth, repo, scratch, set};

#[test]
fn original_inventory() {
    let result = verify(&layout::findings(&repo())).expect("committed register verifies");
    assert_eq!((result.gap, result.improvement), (97, 98));
    assert_eq!(result.entries, 195);
}

#[test]
fn closure_and_dependency_clauses_are_preserved() {
    let record = load(&layout::findings(&repo()).join("register.json"));
    let entries = record
        .get("entries")
        .and_then(Value::as_list)
        .expect("entries");
    // These independently inventoried source labels must not disappear when the extraction
    // vocabulary changes. Do not derive expectations from its regex.
    for (shard, prefix, count, label) in [
        (5, 'G', 8, "**Close with:**"),
        (5, 'I', 8, "**Completion evidence:**"),
        (10, 'K', 10, "**Stop / dependency:**"),
    ] {
        for number in 1..=count {
            let id = format!("S{shard:02}-{prefix}{number}");
            let entry = entries
                .iter()
                .find(|e| e.get("id") == Some(&Value::from(id.as_str())))
                .unwrap_or_else(|| panic!("{id} missing"));
            let clauses = entry
                .get("closure_route")
                .and_then(|r| r.get("source_predicates"))
                .and_then(Value::as_list)
                .expect("source predicates");
            let matching = clauses
                .iter()
                .filter(|c| c.as_str().is_some_and(|c| c.starts_with(label)))
                .count();
            assert_eq!(matching, 1, "shard {shard} {prefix}{number} {label}");
        }
    }
}

fn remove_clause(record: &mut Value, entry_id: &str, label: &str) {
    let entries = at(record, "entries").as_list_mut().expect("entries");
    let entry = entries
        .iter_mut()
        .find(|e| e.get("id") == Some(&Value::from(entry_id)))
        .expect("entry");
    let clauses = at(at(entry, "closure_route"), "source_predicates")
        .as_list_mut()
        .expect("clauses");
    clauses.retain(|c| !c.as_str().is_some_and(|c| c.starts_with(label)));
}

type Mutation = fn(&mut Value);

#[test]
fn mutations_are_rejected() {
    // No builds, clones or worktrees here; small disposable fixture copies only. Python asserted
    // only the exception class; the message patterns are an addition.
    let mutations: [(&str, Mutation, &str); 12] = [
        (
            "omission",
            |r| {
                at(r, "entries").as_list_mut().expect("entries").pop();
            },
            "^entry count$",
        ),
        (
            "duplicate",
            |r| {
                let second = nth(at(r, "entries"), 1).clone();
                *nth(at(r, "entries"), 0) = second;
            },
            "^entry IDs$",
        ),
        (
            "renumber",
            |r| set(nth(at(r, "entries"), 0), "original_id", "G01".into()),
            ": original_id$",
        ),
        (
            "source substitution",
            |r| {
                set(nth(at(r, "sources"), 0), "sha256", "0".repeat(64).into());
            },
            "^source digest$",
        ),
        (
            "section substitution",
            |r| {
                set(
                    nth(at(r, "entries"), 0),
                    "section_sha256",
                    "0".repeat(64).into(),
                );
            },
            ": section_sha256$",
        ),
        (
            "acceptance missing",
            |r| {
                set(
                    at(nth(at(r, "entries"), 0), "closure_route"),
                    "acceptance",
                    "".into(),
                );
            },
            ": acceptance$",
        ),
        (
            "evidence traversal",
            |r| {
                set(
                    at(nth(at(r, "entries"), 0), "closure_route"),
                    "evidence_location",
                    "../pass.json".into(),
                );
            },
            ": evidence path$",
        ),
        (
            "closed without evidence",
            |r| {
                set(nth(at(r, "entries"), 0), "disposition", "closed".into());
            },
            ": disposition$",
        ),
        (
            "synthetic promotion",
            |r| {
                set(
                    at(nth(at(r, "entries"), 0), "attestations"),
                    "promotion",
                    crate::strs(&["PASS"]),
                );
            },
            ": unsupported attestation$",
        ),
        (
            "lost closure clause",
            |r| remove_clause(r, "S05-G1", "**Close with:**"),
            "^S05-G1: source_predicates$",
        ),
        (
            "lost completion evidence",
            |r| {
                remove_clause(r, "S05-I1", "**Completion evidence:**");
            },
            "^S05-I1: source_predicates$",
        ),
        (
            "lost stop dependency",
            |r| {
                remove_clause(r, "S10-K1", "**Stop / dependency:**");
            },
            "^S10-K1: source_predicates$",
        ),
    ];
    let findings = layout::findings(&repo());
    let committed = load(&findings.join("register.json"));
    for (label, mutate, reason) in mutations {
        let directory = scratch();
        let root = directory.path();
        copy_tree(&findings.join("sources"), &root.join("sources"));
        let mut record = committed.clone();
        mutate(&mut record);
        dump(&root.join("register.json"), &record);
        let result = verify(root);
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn unmodified_copy_verifies() {
    // Addition: the mutation harness itself (copy + re-serialisation) is not what fails.
    let findings = layout::findings(&repo());
    let directory = scratch();
    let root = directory.path();
    copy_tree(&findings.join("sources"), &root.join("sources"));
    dump(
        &root.join("register.json"),
        &load(&findings.join("register.json")),
    );
    assert_eq!(verify(root).expect("copy verifies").entries, 195);
}
