//! `verify-findings`: source custody and coverage of the findings register.
//!
//! This checks that `register.json` preserves every finding of the ten audit shards exactly
//! (IDs, titles, digests, line ranges and acceptance clauses) and that v1 admits only open
//! inventory states. It does not qualify product behaviour.

use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::LazyLock;

use regex::Regex;

use crate::py::{self, Key};
use crate::pyjson::{Dict, Value};
use crate::{require, Error, Result};

pub const SCHEMA: &str = "jope.regression-findings/v1";
/// Finding count of each shard, and the prefix of its improvement series.
const COUNTS: [usize; 10] = [8, 9, 10, 10, 8, 12, 10, 10, 10, 10];
const PREFIXES: [char; 10] = ['K', 'I', 'K', 'I', 'I', 'K', 'K', 'K', 'I', 'K'];
/// Attestation slots of every v1 entry; each must stay empty.
pub const STAGES: [&str; 5] = [
    "source_implementation",
    "fixture_proof",
    "hosted_ci",
    "live_qualification",
    "promotion",
];
pub const SCOPE: &str = "inventory integrity only; no product finding closed";

static HEADING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^#{1,3} .*$").expect("regex"));
static FINDING_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^### ([GIK]\d+)[\s\x1c-\x1f]").expect("regex"));
static FIELD: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^(?:- )?\*\*([^*]+)\*\*").expect("regex"));
static PREDICATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        "(?i)detection|acceptance|closure|close with|completion evidence|\
         stop condition|stop / dependency|harden",
    )
    .expect("regex")
});

/// The summary `verify-findings` prints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub entries: usize,
    pub gap: usize,
    pub improvement: usize,
}

impl Summary {
    pub fn to_json(&self) -> Value {
        let mut dict = Dict::new();
        dict.insert("entries", self.entries.into());
        if self.gap > 0 {
            dict.insert("gap", self.gap.into());
        }
        if self.improvement > 0 {
            dict.insert("improvement", self.improvement.into());
        }
        dict.insert("scope", SCOPE.into());
        Value::Dict(dict)
    }
}

/// One finding as its source snapshot defines it.
struct SourceEntry {
    shard: usize,
    first_line: usize,
    /// The recorded fields, in the order Python compared them.
    fields: Vec<(&'static str, Value)>,
}

impl SourceEntry {
    fn new(
        original_id: &str,
        shard: usize,
        title: &str,
        source_sha256: &str,
        section: &str,
        lines: [usize; 2],
        predicates: Vec<String>,
    ) -> Self {
        let kind = if original_id.starts_with('G') {
            "gap"
        } else {
            "improvement"
        };
        let fields = vec![
            ("original_id", original_id.into()),
            ("shard", shard.into()),
            ("title", title.into()),
            ("source_sha256", source_sha256.into()),
            ("section_sha256", py::sha256_hex(section.as_bytes()).into()),
            (
                "source_lines",
                Value::List(vec![lines[0].into(), lines[1].into()]),
            ),
            ("kind", kind.into()),
            (
                "source_predicates",
                Value::List(predicates.into_iter().map(Value::from).collect()),
            ),
        ];
        SourceEntry {
            shard,
            first_line: lines[0],
            fields,
        }
    }
}

fn field<'a>(value: &'a Value, name: &str, context: &str) -> Result<&'a Value> {
    value
        .get(name)
        .ok_or_else(|| Error::Shape(format!("{context}: missing {name:?}")))
}

/// Verify the register under `root` (the findings directory).
pub fn verify(root: &Path) -> Result<Summary> {
    let record = py::load_json(&root.join("register.json"))?;
    require!(record.get("schema") == Some(&Value::from(SCHEMA)), "schema");
    require!(
        py::is_hex(record.get("baseline_source"), 40),
        "baseline SHA"
    );
    let packages = field(&record, "packages", "register")?;
    let expected_packages: HashSet<Key> = (0..13).map(|i| Key::Str(format!("W{i}"))).collect();
    require!(
        packages.as_dict().is_some() && py::key_set(packages) == Some(expected_packages),
        "packages"
    );
    let packages = packages.as_dict().expect("checked above");
    let sources = field(&record, "sources", "register")?
        .as_list()
        .ok_or_else(|| Error::Shape("register: sources is not a list".into()))?;
    require!(sources.len() == 10, "source count");

    let mut source_entries: HashMap<String, SourceEntry> = HashMap::new();
    let mut source_order: Vec<String> = Vec::new();
    for (index, source) in sources.iter().enumerate() {
        let shard = index + 1;
        require!(
            source.get("shard") == Some(&Value::from(shard)),
            "source order/duplicate"
        );
        let expected_path = format!("sources/shard-{shard:02}.md");
        require!(
            source.get("path") == Some(&Value::from(expected_path.as_str())),
            "source path"
        );
        let path = root.join(&expected_path);
        require!(!py::is_symlink(&path), "source symlink");
        let raw = py::read_bytes(&path)?;
        let digest = py::sha256_hex(&raw);
        require!(
            source.get("sha256") == Some(&Value::from(digest.as_str())),
            "source digest"
        );
        let text = String::from_utf8(raw)
            .map_err(|e| Error::parse(&path, format!("'utf-8' codec can't decode: {e}")))?;
        let headings: Vec<regex::Match<'_>> = HEADING.find_iter(&text).collect();
        let mut ids: HashSet<(char, i64)> = HashSet::new();
        for (n, heading) in headings.iter().enumerate() {
            let Some(captures) = FINDING_ID.captures(heading.as_str()) else {
                continue;
            };
            let original_id = &captures[1];
            let key = format!("S{shard:02}-{original_id}");
            require!(!source_entries.contains_key(&key), "duplicate source ID");
            let prefix = original_id.chars().next().expect("regex requires a prefix");
            let number = py::int(&original_id[1..]).map_err(|e| match e {
                py::IntError::Limit(message) => Error::Shape(format!("{key}: {message}")),
                py::IntError::Invalid => {
                    Error::Shape(format!("{key}: non-ASCII finding number is not supported"))
                }
            })?;
            ids.insert((prefix, number));
            let end = headings.get(n + 1).map_or(text.len(), |next| next.start());
            let section = &text[heading.start()..end];
            let fields: Vec<regex::Captures<'_>> = FIELD.captures_iter(section).collect();
            let mut predicates = Vec::new();
            for (m, captures) in fields.iter().enumerate() {
                if PREDICATE.is_match(&captures[1]) {
                    let start = captures.get(0).expect("whole match").start();
                    let stop = fields.get(m + 1).map_or(section.len(), |next| {
                        next.get(0).expect("whole match").start()
                    });
                    predicates.push(py::strip(&section[start..stop]).to_owned());
                }
            }
            let first_line = text[..heading.start()].matches('\n').count() + 1;
            let last_line = text[..end].matches('\n').count();
            source_entries.insert(
                key.clone(),
                SourceEntry::new(
                    original_id,
                    shard,
                    &heading.as_str()[4..],
                    &digest,
                    section,
                    [first_line, last_line],
                    predicates,
                ),
            );
            source_order.push(key);
        }
        let count = COUNTS[index] as i64;
        let improvements = if shard == 2 { count + 1 } else { count };
        let mut expected: HashSet<(char, i64)> = (1..=count).map(|i| ('G', i)).collect();
        expected.extend((1..=improvements).map(|i| (PREFIXES[index], i)));
        require!(ids == expected, "shard {shard} coverage");
    }

    let entries = field(&record, "entries", "register")?
        .as_list()
        .ok_or_else(|| Error::Shape("register: entries is not a list".into()))?;
    require!(
        entries.len() == source_entries.len() && source_entries.len() == 195,
        "entry count"
    );
    let entry_ids: Option<HashSet<Key>> = entries
        .iter()
        .map(|entry| entry.get("id").and_then(Key::of))
        .collect();
    let source_ids: HashSet<Key> = source_order.iter().map(|id| Key::str(id)).collect();
    require!(entry_ids == Some(source_ids), "entry IDs");

    let mut gap = 0;
    let mut improvement = 0;
    for entry in entries {
        let key = entry
            .get("id")
            .and_then(Value::as_str)
            .expect("entry IDs equal the source keys");
        let route = field(entry, "closure_route", key)?;
        let source = &source_entries[key];
        for (name, value) in &source.fields {
            let actual = if *name == "source_predicates" {
                route.get(name)
            } else {
                entry.get(name)
            };
            require!(actual == Some(value), "{key}: {name}");
        }
        let package_name = field(entry, "work_package", key)?;
        let package = package_name
            .as_str()
            .and_then(|name| packages.get(name))
            .ok_or_else(|| {
                Error::Shape(format!(
                    "{key}: unknown work package {}",
                    package_name.repr()
                ))
            })?;
        for name in ["subsystem", "receiving_boundary", "paths"] {
            require!(field(package, name, key)?.truthy(), "{key}: {name}");
        }
        let dependencies = py::iterate(field(package, "dependencies", key)?)
            .ok_or_else(|| Error::Shape(format!("{key}: dependencies is not iterable")))?;
        require!(
            dependencies
                .iter()
                .all(|dep| dep.as_str().is_some_and(|d| packages.contains_key(d))),
            "dependency"
        );
        require!(field(entry, "scope", key)?.truthy(), "{key}: scope");
        // `entry["shard"]` and `entry["source_lines"][0]` equal these (checked above).
        let (shard, first_line) = (source.shard, source.first_line);
        require!(
            route.get("specification")
                == Some(&Value::from(format!(
                    "sources/shard-{shard:02}.md#L{first_line}"
                ))),
            "{key}: specification"
        );
        require!(
            route.get("evidence_location") == Some(&Value::from(format!("evidence/{key}.json"))),
            "{key}: evidence path"
        );
        require!(
            route.get("acceptance").is_some_and(Value::truthy),
            "{key}: acceptance"
        );
        // V1 is an open-work inventory. Qualifying evidence needs a reviewed schema extension,
        // not free-form strings that silently upgrade this inventory.
        let is_gap = entry.get("kind") == Some(&Value::from("gap"));
        let disposition = if is_gap {
            "unverified-source-finding"
        } else {
            "proposed-improvement"
        };
        require!(
            entry.get("disposition") == Some(&Value::from(disposition)),
            "{key}: disposition"
        );
        require!(
            route
                .get("reproduction_command")
                .is_some_and(Value::is_null)
                && route.get("reproduction_status")
                    == Some(&Value::from("not-yet-established-through-receiving-code")),
            "{key}: reproduction"
        );
        require!(
            route.get("evidence_status") == Some(&Value::from("not-produced")),
            "{key}: evidence status"
        );
        let slots = entry.get("attestations").and_then(Value::as_dict);
        require!(
            slots.is_some_and(|slots| {
                slots.len() == STAGES.len()
                    && STAGES.iter().all(|stage| slots.contains_key(stage))
                    && slots.values().all(|v| v == &Value::List(Vec::new()))
            }),
            "{key}: unsupported attestation"
        );
        if is_gap {
            gap += 1;
        } else {
            improvement += 1;
        }
    }
    Ok(Summary {
        entries: entries.len(),
        gap,
        improvement,
    })
}
