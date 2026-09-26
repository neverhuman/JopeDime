//! Port of `test_progress_view.py`.

use std::path::PathBuf;

use jope_records::layout;
use jope_records::progress_view::{check_or_write, collect, render};
use jope_records::pyjson::Value;

use crate::{assert_refused, at, copy_tree, dump_with, load, repo, set, strs};

/// The view without its input-digest table, which intentionally changes with the bytes.
fn body(text: &str) -> String {
    let (head, rest) = text.split_once("## Inputs").unwrap_or((text, ""));
    let tail = rest
        .split_once("## Findings and tasks")
        .map_or("", |(_, t)| t);
    format!("{head}{tail}")
}

/// A throwaway repository holding a copy of the committed core-IP records.
struct Copy {
    _directory: tempfile::TempDir,
    core: PathBuf,
}

impl Copy {
    fn new() -> Self {
        let directory = crate::scratch();
        let core = directory.path().join(layout::CORE_IP);
        copy_tree(&layout::core_ip(&repo()), &core);
        Copy {
            _directory: directory,
            core,
        }
    }

    fn edit(
        &self,
        name: &str,
        change: impl FnOnce(&mut Value),
        sort_keys: bool,
        indent: Option<usize>,
    ) {
        let path = self.core.join(name);
        let mut document = load(&path);
        change(&mut document);
        dump_with(&path, &document, sort_keys, indent);
    }

    fn render(&self) -> jope_records::Result<String> {
        render(&collect(&self.core)?)
    }
}

#[test]
fn committed_view_is_current() {
    let core = layout::core_ip(&repo());
    let text = render(&collect(&core).expect("committed records verify")).expect("render");
    let committed =
        std::fs::read_to_string(core.join("work-orders/PROGRESS.md")).expect("PROGRESS.md");
    assert_eq!(committed, text);
    assert!(text.contains("| **Total** |"));
    check_or_write(&core, &text, false).expect("committed view is current");
}

#[test]
fn regeneration_is_deterministic_and_blind_to_presentation() {
    let copy = Copy::new();
    let first = copy.render().expect("render");
    assert_eq!(first, copy.render().expect("render again"));
    let restyle = |document: &mut Value| set(document, "note", "renamed presentation note".into());
    // Key order, indentation and free-text notes are presentation only.
    copy.edit("work-orders/task-status.json", restyle, true, Some(1));
    copy.edit("claims/claims.json", restyle, true, Some(3));
    let second = copy.render().expect("render restyled");
    assert_ne!(first, second, "the digest table binds the new bytes");
    assert_eq!(body(&first), body(&second));
}

#[test]
fn task_states_cannot_close_a_finding() {
    let copy = Copy::new();
    copy.edit(
        "work-orders/task-status.json",
        |document| {
            for (task, item) in at(document, "tasks")
                .as_dict_mut()
                .expect("tasks")
                .iter_mut()
            {
                if task.starts_with("W3-") {
                    set(
                        item,
                        "states",
                        strs(&["reproduced", "implemented", "fixture-proven"]),
                    );
                    set(item, "commits", strs(&[&"a".repeat(40)]));
                    set(item, "tests", strs(&["fixture"]));
                }
            }
        },
        false,
        Some(2),
    );
    let text = copy.render().expect("render");
    let w3 = text
        .lines()
        .find(|line| line.starts_with("| W3 |"))
        .expect("W3 row");
    let cells: Vec<&str> = w3.trim_matches('|').split('|').map(str::trim).collect();
    let numbers: Vec<usize> = cells[1..5]
        .iter()
        .map(|c| c.parse().expect("count"))
        .collect();
    let (gaps, improvements, open, closed) = (numbers[0], numbers[1], numbers[2], numbers[3]);
    assert_eq!((open, closed), (gaps + improvements, 0));
    assert!(text.contains("- Findings closed: 0 of"));
}

#[test]
fn invalid_records_are_refused_before_rendering() {
    let copy = Copy::new();
    copy.edit(
        "work-orders/task-status.json",
        |document| {
            // A commit alone cannot stand in for the live evidence a live state needs.
            let task = at(at(document, "tasks"), "W7-T07");
            set(task, "states", strs(&["live-qualified"]));
            set(task, "commits", strs(&[&"a".repeat(40)]));
        },
        false,
        Some(2),
    );
    assert_refused(copy.render(), "live/measurement states need live evidence");
}

#[test]
fn stale_view_is_detected() {
    let copy = Copy::new();
    let target = copy.core.join("work-orders/PROGRESS.md");
    let text = std::fs::read_to_string(&target).expect("PROGRESS.md");
    std::fs::write(
        &target,
        text.replace("Findings closed: 0", "Findings closed: 1"),
    )
    .expect("edit");
    let rendered = copy.render().expect("render");
    assert_ne!(
        std::fs::read_to_string(&target).expect("PROGRESS.md"),
        rendered
    );
    // Addition: the check refuses it, and --write regenerates it.
    assert_refused(
        check_or_write(&copy.core, &rendered, false),
        "^PROGRESS.md is stale; regenerate with --write$",
    );
    check_or_write(&copy.core, &rendered, true).expect("write");
    check_or_write(&copy.core, &rendered, false).expect("regenerated view is current");
}
