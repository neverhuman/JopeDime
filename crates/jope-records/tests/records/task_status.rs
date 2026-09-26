//! Port of `test_task_status.py`.

use std::path::Path;

use jope_records::layout;
use jope_records::pyjson::{Dict, Value};
use jope_records::task_status::{check_or_write, render, verify, Record};

use crate::{assert_refused, at, dump, json, load, repo, scratch, set, strs};

/// Copy the work-order task files (and optionally the record) into `root`.
fn copy_work_orders(root: &Path, with_record: bool) {
    let source = layout::work_orders(&repo());
    for entry in std::fs::read_dir(&source).expect("work orders") {
        let entry = entry.expect("entry");
        let name = entry.file_name().to_string_lossy().into_owned();
        if (name.starts_with('W') && name.ends_with(".tasks.md"))
            || (with_record && name == "task-status.json")
        {
            std::fs::copy(entry.path(), root.join(&name)).expect("copy");
        }
    }
}

#[test]
fn committed_status_is_complete_and_rendered() {
    let root = layout::work_orders(&repo());
    let record = verify(&root).expect("committed task status verifies");
    assert_eq!(record.tasks.len(), 92);
    let committed = std::fs::read_to_string(root.join("STATUS.md")).expect("STATUS.md");
    assert_eq!(committed, render(&record).expect("render"));
    check_or_write(&root, &render(&record).expect("render"), false)
        .expect("committed view is current");
}

type Mutate = fn(&mut Value, &str);

#[test]
fn mutations_are_rejected() {
    let record = load(&layout::work_orders(&repo()).join("task-status.json"));
    let mut ids: Vec<&str> = record
        .get("tasks")
        .and_then(Value::as_dict)
        .expect("tasks")
        .keys()
        .collect();
    ids.sort_unstable();
    let first = ids[0].to_owned();
    let mutations: [(&str, Mutate, &str); 10] = [
        (
            "missing task",
            |r, first| drop(crate::pop(at(r, "tasks"), first)),
            "missing",
        ),
        (
            "invented task",
            |r, _| {
                set(
                    at(r, "tasks"),
                    "W99-T01",
                    json(r#"{"states": ["planned"]}"#),
                )
            },
            "extra",
        ),
        (
            "unknown state",
            |r, first| set(at(at(r, "tasks"), first), "states", strs(&["done"])),
            "states",
        ),
        (
            "planned with progress",
            |r, first| {
                set(
                    at(at(r, "tasks"), first),
                    "states",
                    strs(&["planned", "implemented"]),
                )
            },
            "planned stands alone",
        ),
        (
            "progress without evidence",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["implemented"]));
                set(task, "commits", Value::List(vec![]));
                set(task, "evidence", Value::List(vec![]));
            },
            "needs commits or evidence",
        ),
        (
            "abbreviated commit",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["implemented"]));
                set(task, "commits", strs(&["abc123"]));
            },
            "full SHAs",
        ),
        (
            "fixture without tests",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["fixture-proven"]));
                set(task, "commits", strs(&[&"a".repeat(40)]));
                set(task, "tests", Value::List(vec![]));
            },
            "proving tests",
        ),
        (
            "CI without run",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["CI-verified"]));
                set(task, "commits", strs(&[&"a".repeat(40)]));
            },
            "run binding",
        ),
        (
            "live without evidence",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["live-qualified"]));
                set(task, "commits", strs(&[&"a".repeat(40)]));
            },
            "live evidence",
        ),
        (
            "blocked without blocker",
            |r, first| {
                let task = at(at(r, "tasks"), first);
                set(task, "states", strs(&["blocked"]));
                set(task, "blocker", "".into());
            },
            "blocker",
        ),
    ];
    for (label, mutate, reason) in mutations {
        let directory = scratch();
        let root = directory.path();
        copy_work_orders(root, false);
        let mut changed = record.clone();
        mutate(&mut changed, &first);
        dump(&root.join("task-status.json"), &changed);
        let result = verify(root);
        assert!(result.is_err(), "{label}: accepted");
        assert_refused(result, reason);
    }
}

#[test]
fn rendering_is_deterministic_and_stale_views_fail() {
    let record = verify(&layout::work_orders(&repo())).expect("committed task status");
    let text = render(&record).expect("render");
    assert_eq!(text, render(&record.clone()).expect("render again"));
    // Addition: record key order is presentation only.
    let reordered: Dict = {
        let mut pairs: Vec<(&str, &Value)> = record.tasks.iter().collect();
        pairs.reverse();
        pairs.into_iter().map(|(k, v)| (k, v.clone())).collect()
    };
    let reordered = Record {
        headings: record.headings.clone(),
        tasks: reordered,
    };
    assert_eq!(text, render(&reordered).expect("render reordered"));

    let directory = scratch();
    let root = directory.path();
    copy_work_orders(root, true);
    std::fs::write(root.join("STATUS.md"), "hand-edited view\n").expect("hand edit");
    let copied = verify(root).expect("copied record verifies");
    let rendered = render(&copied).expect("render copy");
    assert_ne!(
        std::fs::read_to_string(root.join("STATUS.md")).expect("STATUS.md"),
        rendered
    );
    // Addition: the check refuses the stale view, and --write regenerates it.
    assert_refused(
        check_or_write(root, &rendered, false),
        "^STATUS.md is stale; regenerate with --write$",
    );
    check_or_write(root, &rendered, true).expect("write");
    check_or_write(root, &rendered, false).expect("regenerated view is current");
}
