//! Bucket lifecycle configurations (as returned by `get-bucket-lifecycle-configuration`).
//!
//! The brain does not change lifecycle rules. It reads them to say which groups an applied rule
//! already transitions, and to raise hazards: a rule that expires current versions under a T0 or
//! training-value group, or a cold transition under a group the plan keeps hot.

use anyhow::{Context, Result};
use serde::Serialize;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, PartialEq)]
/// `Rule` is a record for the rule contract in this module: Bucket lifecycle configurations (as returned by get-bucket-lifecycle-configuration).
pub struct Rule {
    /// Stable identifier for this value.
    pub id: String,
    /// Enabled recorded for this rule.
    pub enabled: bool,
    /// Prefix recorded for this rule.
    pub prefix: String,
    /// Tag-filtered rules apply only to tagged objects; untagged groups are not covered.
    pub tagged: bool,
    /// Lower bound on size.
    pub min_size: Option<u64>,
    /// Collection of transitions values carried by this rule.
    pub transitions: Vec<(u64, String)>,
    /// Optional expiration days value; `None` means it is not available.
    pub expiration_days: Option<u64>,
    /// Optional noncurrent days value; `None` means it is not available.
    pub noncurrent_days: Option<u64>,
}

fn rule_from(v: &Value) -> Rule {
    let filter = v.get("Filter");
    let and = filter.and_then(|f| f.get("And"));
    let prefix = and
        .and_then(|a| a.get("Prefix"))
        .or_else(|| filter.and_then(|f| f.get("Prefix")))
        .or_else(|| v.get("Prefix"))
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let tagged = filter.and_then(|f| f.get("Tag")).is_some()
        || and
            .and_then(|a| a.get("Tags"))
            .and_then(Value::as_array)
            .is_some_and(|t| !t.is_empty());
    let min_size = and
        .and_then(|a| a.get("ObjectSizeGreaterThan"))
        .or_else(|| filter.and_then(|f| f.get("ObjectSizeGreaterThan")))
        .and_then(Value::as_u64);
    let transitions = v
        .get("Transitions")
        .and_then(Value::as_array)
        .map(|ts| {
            ts.iter()
                .map(|t| {
                    (
                        t.get("Days").and_then(Value::as_u64).unwrap_or(0),
                        t.get("StorageClass")
                            .and_then(Value::as_str)
                            .unwrap_or("?")
                            .to_string(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Rule {
        id: v
            .get("ID")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string(),
        enabled: v.get("Status").and_then(Value::as_str) == Some("Enabled"),
        prefix,
        tagged,
        min_size,
        transitions,
        expiration_days: v
            .get("Expiration")
            .and_then(|e| e.get("Days"))
            .and_then(Value::as_u64),
        noncurrent_days: v
            .get("NoncurrentVersionExpiration")
            .and_then(|e| e.get("NoncurrentDays"))
            .and_then(Value::as_u64),
    }
}

/// Bucket -> rules, plus the file hashes for the inputs lock.
#[derive(Debug, Default)]
pub struct Lifecycles {
    /// Mapping of rules values carried by this lifecycles.
    pub rules: BTreeMap<String, Vec<Rule>>,
    /// Collection of files values carried by this lifecycles.
    pub files: Vec<(String, String, u64)>,
}

impl Lifecycles {
    /// Reads every `<bucket>.json` in `dir`. A file that does not parse is fatal.
    pub fn load_dir(dir: &Path) -> Result<Lifecycles> {
        let mut out = Lifecycles::default();
        let mut entries: Vec<_> = fs::read_dir(dir)
            .with_context(|| format!("cannot read lifecycle dir {}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .collect();
        entries.sort();
        for path in entries {
            let bytes =
                fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
            let v: Value = serde_json::from_slice(&bytes)
                .with_context(|| format!("{} does not parse", path.display()))?;
            let bucket = path
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let rules = v
                .get("Rules")
                .and_then(Value::as_array)
                .map(|r| r.iter().map(rule_from).collect());
            out.rules.insert(bucket, rules.unwrap_or_default());
            out.files.push((
                path.display().to_string(),
                crate::canon::sha256_hex(&bytes),
                bytes.len() as u64,
            ));
        }
        Ok(out)
    }
}

/// How the applied rules touch one group.
#[derive(Clone, Debug, Default, Serialize, PartialEq)]
pub struct Coverage {
    /// `false` when no lifecycle file was supplied for the bucket.
    pub known: bool,
    /// Untagged rules that cover every object of the group: `id: what`.
    pub full: Vec<String>,
    /// Untagged rules that cover part of the group (a longer prefix inside it).
    pub partial: Vec<String>,
    /// Tag-filtered rules over the group (inactive for untagged objects).
    pub tagged: Vec<String>,
    /// Coldest untagged storage class a full-coverage transition moves the group to.
    pub cold_class: Option<String>,
    /// Untagged full or partial rules that expire current versions.
    pub expires_current: Vec<String>,
    /// Optional noncurrent days value; `None` means it is not available.
    pub noncurrent_days: Option<u64>,
}

fn describe(r: &Rule) -> String {
    let mut parts = Vec::new();
    for (d, c) in &r.transitions {
        parts.push(format!("{c}@{d}d"));
    }
    if let Some(d) = r.expiration_days {
        parts.push(format!("EXPIRE-CURRENT@{d}d"));
    }
    if let Some(d) = r.noncurrent_days {
        parts.push(format!("noncurrent-expire@{d}d"));
    }
    if let Some(m) = r.min_size {
        parts.push(format!("size>{m}"));
    }
    if parts.is_empty() {
        parts.push("abort-mpu/other".into());
    }
    format!("{}: {}", r.id, parts.join(","))
}

fn coldness(class: &str) -> u8 {
    match class {
        "DEEP_ARCHIVE" => 4,
        "GLACIER" => 3,
        "GLACIER_IR" => 2,
        "STANDARD_IA" | "ONEZONE_IA" | "INTELLIGENT_TIERING" => 1,
        _ => 0,
    }
}

/// Performs the coverage operation.
pub fn coverage(l: &Lifecycles, bucket: &str, prefix: &str, direct: bool) -> Coverage {
    let Some(rules) = l.rules.get(bucket) else {
        return Coverage::default();
    };
    let mut c = Coverage {
        known: true,
        ..Default::default()
    };
    for r in rules.iter().filter(|r| r.enabled) {
        let full = prefix.starts_with(r.prefix.as_str());
        // A longer rule prefix inside a leaf group covers part of it; a direct group's objects sit
        // right under its prefix, so a longer prefix can still match some of their keys.
        let partial = !full
            && r.prefix.starts_with(prefix)
            && (!direct || !r.prefix[prefix.len()..].contains('/'));
        if !full && !partial {
            continue;
        }
        if r.tagged {
            c.tagged.push(describe(r));
            continue;
        }
        if r.expiration_days.is_some() {
            c.expires_current.push(describe(r));
        }
        if full {
            if let Some(nd) = r.noncurrent_days {
                c.noncurrent_days = Some(c.noncurrent_days.map_or(nd, |o: u64| o.min(nd)));
            }
            if r.min_size.is_none() {
                for (_, class) in &r.transitions {
                    if c.cold_class
                        .as_deref()
                        .is_none_or(|o| coldness(class) > coldness(o))
                    {
                        c.cold_class = Some(class.clone());
                    }
                }
            }
            c.full.push(describe(r));
        } else {
            c.partial.push(describe(r));
        }
    }
    c
}
