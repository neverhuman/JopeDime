//! `task-status`: validate the W1-W12 task-status record and render `STATUS.md`
//! deterministically (W12-T03).
//!
//! Every numbered task heading in the work orders has exactly one record. States use the
//! execution contract's vocabulary and are scoped observations, not a ladder: `planned` stands
//! alone; any other state needs commits or evidence; `fixture-proven` names its proving tests;
//! `CI-verified` binds a run, attempt and head; `live-qualified` and `measurement-complete` bind
//! live evidence; `blocked` states its blocker. The rendered view is regenerated from these
//! records only, so presentation cannot strengthen a task's evidence.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::py::{self, Key};
use crate::pyjson::{Dict, Value};
use crate::{require, Error, Result};

pub const SCHEMA: &str = "jope.regression-task-status/v1";
pub const STATES: [&str; 8] = [
    "planned",
    "reproduced",
    "implemented",
    "fixture-proven",
    "CI-verified",
    "live-qualified",
    "measurement-complete",
    "blocked",
];
/// The generator named in the rendered header.
pub const GENERATOR: &str = "jope-records task-status";

static TASK_HEADING: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^### (W\d+-T\d+) — (.+)$").expect("regex"));

/// A task heading of a `W*.tasks.md` work order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Heading {
    pub title: String,
    /// The work-order file prefix, e.g. `W01`.
    pub order: String,
}

/// Task headings in work-order file order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Headings {
    ids: Vec<String>,
    by_id: HashMap<String, Heading>,
}

impl Headings {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn get(&self, id: &str) -> Option<&Heading> {
        self.by_id.get(id)
    }

    pub fn contains(&self, id: &str) -> bool {
        self.by_id.contains_key(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &Heading)> {
        self.ids.iter().map(|id| (id.as_str(), &self.by_id[id]))
    }
}

/// The validated record: headings and the `tasks` object of `task-status.json`.
#[derive(Clone, Debug)]
pub struct Record {
    pub headings: Headings,
    pub tasks: Dict,
}

/// `Path.glob("W*.tasks.md")`, sorted.
fn work_order_files(root: &Path) -> Result<Vec<std::path::PathBuf>> {
    let mut files = Vec::new();
    for entry in std::fs::read_dir(root).map_err(|e| Error::io(root, e))? {
        let entry = entry.map_err(|e| Error::io(root, e))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with('W') && name.ends_with(".tasks.md") && name.len() >= "W.tasks.md".len()
        {
            files.push(entry.path());
        }
    }
    files.sort();
    Ok(files)
}

/// Every `### W<n>-T<m> — title` heading of the work orders under `root`.
pub fn task_headings(root: &Path) -> Result<Headings> {
    let mut headings = Headings::default();
    for path in work_order_files(root)? {
        let text = py::read_text(&path)?;
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default();
        let order = name.split('-').next().unwrap_or(name).to_owned();
        for captures in TASK_HEADING.captures_iter(&text) {
            let id = captures[1].to_owned();
            require!(!headings.contains(&id), "duplicate task heading {id}");
            headings.by_id.insert(
                id.clone(),
                Heading {
                    title: py::strip(&captures[2]).to_owned(),
                    order: order.clone(),
                },
            );
            headings.ids.push(id);
        }
    }
    Ok(headings)
}

fn state_list(item: &Value) -> Option<Vec<&str>> {
    item.get("states")?
        .as_list()?
        .iter()
        .map(Value::as_str)
        .collect()
}

/// Validate `task-status.json` under `root` (the work-orders directory).
pub fn verify(root: &Path) -> Result<Record> {
    let headings = task_headings(root)?;
    let record = py::load_json(&root.join("task-status.json"))?;
    require!(
        record.get("schema") == Some(&Value::from(SCHEMA)),
        "task status schema"
    );
    let tasks = match record.get("tasks") {
        None => Dict::new(),
        Some(Value::Dict(tasks)) => tasks.clone(),
        Some(_) => return Err(Error::Shape("task-status: tasks is not an object".into())),
    };
    let mut missing: Vec<&str> = headings
        .iter()
        .map(|(id, _)| id)
        .filter(|id| !tasks.contains_key(id))
        .collect();
    missing.sort_unstable();
    let mut extra: Vec<&str> = tasks.keys().filter(|id| !headings.contains(id)).collect();
    extra.sort_unstable();
    require!(
        missing.is_empty() && extra.is_empty(),
        "task records differ from headings: missing {}, extra {}",
        py::str_list_repr(&missing[..missing.len().min(4)]),
        py::str_list_repr(&extra[..extra.len().min(4)])
    );
    for (task, item) in tasks.iter() {
        let states = item.get("states").and_then(Value::as_list);
        let states_ok = states.is_some_and(|states| {
            let unique: HashSet<Option<Key>> = states.iter().map(Key::of).collect();
            !states.is_empty()
                && unique.len() == states.len()
                && states
                    .iter()
                    .all(|s| s.as_str().is_some_and(|s| STATES.contains(&s)))
        });
        require!(states_ok, "{task}: states");
        let states = state_list(item).expect("checked above");
        require!(
            !states.contains(&"planned") || states == ["planned"],
            "{task}: planned stands alone"
        );
        let commits = match item.get("commits") {
            None => Vec::new(),
            Some(value) => py::iterate(value)
                .ok_or_else(|| Error::Shape(format!("{task}: commits is not iterable")))?,
        };
        require!(
            commits.iter().all(|c| py::is_hex(Some(c), 40)),
            "{task}: commits must be full SHAs"
        );
        if states != ["planned"] && states != ["blocked"] {
            require!(
                !commits.is_empty() || item.get("evidence").is_some_and(Value::truthy),
                "{task}: progress needs commits or evidence"
            );
        }
        if states.contains(&"fixture-proven") {
            require!(
                item.get("tests").is_some_and(Value::truthy),
                "{task}: fixture-proven needs proving tests"
            );
        }
        if states.contains(&"CI-verified") {
            let empty = Value::Dict(Dict::new());
            let ci = item.get("ci").unwrap_or(&empty);
            require!(
                ci.get("run_id").and_then(Value::as_int).is_some()
                    && ci.get("attempt").and_then(Value::as_int).is_some()
                    && py::is_hex(ci.get("head_sha"), 40),
                "{task}: CI-verified needs run binding"
            );
        }
        if states.contains(&"live-qualified") || states.contains(&"measurement-complete") {
            require!(
                item.get("live_evidence").is_some_and(Value::truthy),
                "{task}: live/measurement states need live evidence"
            );
        }
        if states.contains(&"blocked") {
            require!(
                item.get("blocker").is_some_and(Value::truthy),
                "{task}: blocked needs its blocker"
            );
        }
        require!(
            item.get("remaining").is_none_or(|r| r.as_str().is_some()),
            "{task}: remaining"
        );
    }
    Ok(Record { headings, tasks })
}

/// How many tasks hold each state.
pub fn state_counts(tasks: &Dict) -> BTreeMap<&'static str, usize> {
    STATES
        .iter()
        .map(|state| {
            let count = tasks
                .values()
                .filter(|item| state_list(item).is_some_and(|s| s.contains(state)))
                .count();
            (*state, count)
        })
        .collect()
}

/// `int(text)` for a sort key, as the Python `sorted(..., key=int(...))` required; more than 4300
/// digits refuse in CPython's words.
pub(crate) fn int_key(text: &str, what: &str) -> Result<i64> {
    py::int(text).map_err(|e| match e {
        py::IntError::Limit(message) => Error::Shape(format!("{what}: {message}")),
        py::IntError::Invalid => Error::Shape(format!("invalid literal for int(): {what}")),
    })
}

fn string_items(value: &Value, task: &str, field: &str) -> Result<Vec<String>> {
    value
        .as_list()
        .and_then(|list| list.iter().map(|v| v.as_str().map(str::to_owned)).collect())
        .ok_or_else(|| Error::Shape(format!("{task}: {field} must be a list of strings")))
}

/// Render `STATUS.md` from a validated record.
pub fn render(record: &Record) -> Result<String> {
    let Record { headings, tasks } = record;
    let mut lines: Vec<String> = vec![
        "# Regression remediation task status".into(),
        String::new(),
        format!("Generated by `{GENERATOR}` from `task-status.json`; do not edit."),
        "States are scoped observations (execution contract), not an automatic ladder. Fixture proof is".into(),
        "not CI verification, and neither is live qualification. No finding closes through this view.".into(),
        String::new(),
    ];
    let counts = state_counts(tasks);
    lines.push("| State | Tasks |".into());
    lines.push("|---|---:|".into());
    for state in STATES {
        lines.push(format!("| {state} | {} |", counts[state]));
    }
    lines.push(String::new());
    let mut orders: Vec<(i64, &str)> = Vec::new();
    for (_, heading) in headings.iter() {
        if !orders.iter().any(|(_, o)| *o == heading.order) {
            orders.push((
                int_key(&heading.order[1..], &heading.order)?,
                &heading.order,
            ));
        }
    }
    orders.sort();
    for (_, order) in orders {
        lines.push(format!("## {order}"));
        lines.push(String::new());
        lines.push("| Task | States | Evidence | Remaining or blocker |".into());
        lines.push("|---|---|---|---|".into());
        let mut owned: Vec<(i64, &str)> = Vec::new();
        for (id, heading) in headings.iter() {
            if heading.order == order {
                let number = id.split("-T").nth(1).unwrap_or_default();
                owned.push((int_key(number, id)?, id));
            }
        }
        owned.sort_by_key(|(number, _)| *number);
        for (_, task) in owned {
            let item = tasks.get(task).expect("verified: records equal headings");
            let empty = Value::List(Vec::new());
            let mut evidence: Vec<String> =
                string_items(item.get("commits").unwrap_or(&empty), task, "commits")?
                    .into_iter()
                    .map(|c| c.chars().take(12).collect())
                    .collect();
            let tests = string_items(item.get("tests").unwrap_or(&empty), task, "tests")?;
            evidence.extend(tests.iter().take(3).cloned());
            if tests.len() > 3 {
                evidence.push(format!("+{} tests", tests.len() - 3));
            }
            let blank = Value::from("");
            let note = match item.get("blocker") {
                Some(blocker) if blocker.truthy() => blocker,
                _ => item.get("remaining").unwrap_or(&blank),
            };
            let states = state_list(item).expect("verified");
            let evidence = if evidence.is_empty() {
                "—".to_owned()
            } else {
                evidence.join("; ")
            };
            let note = if note.truthy() {
                note.py_str()
            } else {
                "—".to_owned()
            };
            lines.push(format!(
                "| {task} {} | {} | {evidence} | {note} |",
                headings.get(task).expect("heading").title,
                states.join(", ")
            ));
        }
        lines.push(String::new());
    }
    Ok(lines.join("\n"))
}

/// The summary `task-status` prints.
pub fn summary(record: &Record) -> Value {
    let mut states = Dict::new();
    for (state, count) in state_counts(&record.tasks) {
        states.insert(state, count.into());
    }
    let mut dict = Dict::new();
    dict.insert("tasks", record.tasks.len().into());
    dict.insert("states", Value::Dict(states));
    Value::Dict(dict)
}

/// Compare the committed view with a fresh render, or rewrite it (`--write`).
pub fn check_or_write(root: &Path, text: &str, write: bool) -> Result<()> {
    let target = root.join("STATUS.md");
    if write {
        return std::fs::write(&target, text).map_err(|e| Error::io(&target, e));
    }
    require!(
        py::read_text(&target)? == text,
        "STATUS.md is stale; regenerate with --write"
    );
    Ok(())
}
