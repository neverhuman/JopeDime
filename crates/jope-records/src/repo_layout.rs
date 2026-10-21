//! `repo-layout`: the repository layout guard of the hygiene lane.
//!
//! Seven rules over the tracked tree (`git ls-files` at the checkout root). Each finding is one
//! line that names the rule, the path and the fix:
//!
//! | Rule | Finds |
//! |---|---|
//! | `root-md` | a tracked Markdown file at the repository root that is not allowlisted |
//! | `log` | a tracked `.log`, `.err` or `.stderr` file that is not allowlisted |
//! | `dir-package` | a workspace member whose directory name differs from its `[package] name` |
//! | `dup-package` | two tracked `Cargo.toml` files declaring the same package name, except exact reviewed two-version gemm vendor pairs |
//! | `temp-dir` | calls of the process-wide temp directory (`std::env::temp_dir`) in test code under `crates/`, above the allowlisted count |
//! | `bad-manifest` | a tracked `Cargo.toml` that is not readable as TOML 1.0: not UTF-8, not TOML 1.0, or a `[package]` table without a non-empty string `name` (no allowlist) |
//! | `bad-path` | a tracked path that is not UTF-8 (no allowlist) |
//!
//! Cargo manifests are read with a TOML 1.0 parser (the `toml` 0.8 crate), never by line: every
//! TOML 1.0 spelling of a package name counts (basic, literal and multi-line strings, escapes,
//! dotted keys, inline tables, comments, CRLF), and so does every spelling of the root manifest's
//! `[workspace] members`. Cargo itself reads TOML 1.1, so a manifest that uses 1.1-only syntax
//! (the `\xHH` and `\e` escapes, a newline or trailing comma in an inline table) is one Cargo
//! reads and this guard reports as `bad-manifest`, saying to rewrite it in TOML 1.0. A manifest
//! the parser refuses is a `bad-manifest` finding, not a manifest the other rules skip. The rule
//! therefore requires every tracked manifest to be TOML 1.0. `bad-manifest` has no allowlist,
//! like `dup-package`: the package-name rules cannot see into a manifest they cannot read, so an
//! exception would reopen the gap they close, and a deliberately malformed test input can use
//! another file name.
//!
//! Tracked paths are the bytes `git ls-files -z` prints, never decoded lossily. Every rule
//! matches paths as text, so a path that is not UTF-8 is one no other rule can check: `bad-path`
//! reports it, each byte that is not UTF-8 written `\xHH`, and has no allowlist because an
//! exception would hide the path from every rule. A tracked file is skipped unread only when
//! `git ls-files --deleted` reports it deleted in the working tree; any other tracked file the
//! guard needs and cannot read (a dangling symlink, say) makes the command refuse.
//!
//! Test code, for `temp-dir`, is any file under a `tests/` directory, a file named `tests.rs` or
//! `*_tests.rs`, and otherwise the lines of a `#[cfg(test)]` block whose next non-attribute line
//! is an inline `mod` declaration. A block ends at the first later line that is a closing brace at
//! the `mod` line's indentation (the tree is rustfmt-formatted; without one, the block runs to
//! the end of the file). Production callers are out of scope.
//!
//! The exceptions are in [`ALLOWLIST`], one per line: `<kind> <path> [<value>] -- <reason>`, where
//! the value is the package name for `dir-package` and the permitted count for `temp-dir`. The
//! list is shrink-only: an entry for an untracked path, a count above the actual count, or an
//! entry that matches nothing fails with "lower or remove this entry". New entries are a reviewed
//! diff. [`check`] is pure over an [`Input`]; [`gather`] reads it from a checkout.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::process::Command;

use regex::Regex;
use sha2::Digest;

use crate::pyjson::Value;
use crate::{Error, Result};

/// The reviewed exceptions, relative to the repository root.
pub const ALLOWLIST: &str = "deploy/ci/repo-layout-allow.txt";

// These are dependency identities, not product-package exceptions. Every manifest must match
// its exact PROVENANCE.json candidate pin and versioned directory; full source custody remains
// enforced by deploy/ci/verify-vendor-security.py. An upgrade requires a reviewed pin change.
const REVIEWED_GEMM_VERSIONS: [&str; 2] = ["0.18.2", "0.19.0"];
const REVIEWED_GEMM_MANIFESTS: &[(&str, [&str; 2])] = &[
    (
        "gemm",
        [
            "6510d43872444c9f5dbab1a774573791a98a891c990ba2811dd12bab37c14db8",
            "475d12ec9c311f5970ad6574c4fbc9b9e63377e78df4e395ba67f797646a4168",
        ],
    ),
    (
        "gemm-c32",
        [
            "53296fadbbfe2ee5849469da5b441265b58622e930a95ef23fb475693da617f4",
            "9bf484b69fd49611b49a7de1d161cde81297043d64795cd21eb378c755bdb74e",
        ],
    ),
    (
        "gemm-c64",
        [
            "bb81831a93de7f5a72fd936945d96d2c4d0610a0500bba744bfd932dd0c10544",
            "eb69b31aa4dcc59b83af044318c92f7c720f07136c561c015facf069595a5e04",
        ],
    ),
    (
        "gemm-common",
        [
            "29044e6cd42521526654860c5dfdab84c7714e485eda85c34a78faaadb4a609f",
            "1ce14ff38846ec6f9f7e9071db15e00f93d2076778f795943a40662b4748efe4",
        ],
    ),
    (
        "gemm-f16",
        [
            "be2e69beffee45fd2dda2ee96da0ed8019f716725488c3b6566120b4a4b55a7a",
            "36e228585e982ba3ccdd52689e0837d7d4f35c7fdcafa8bd77adfa312949cbd6",
        ],
    ),
    (
        "gemm-f32",
        [
            "d77a7dc6f854039d027dcd5271b71ff1bde1972c7d59c04f35351c8ddd33788f",
            "9d6e298631c47522ffa71c1bd805d4f14804d07ac407daf5694e9164512a6227",
        ],
    ),
    (
        "gemm-f64",
        [
            "602c533b7ad5d05b54cab5d573cac884fb26a320c33b47958d798c5623aff457",
            "57745976614c4d3143f440562f8c20143d1dc8ce544dc8ce32971c3bf096cb17",
        ],
    ),
];

fn reviewed_vendor_pair(input: &Input, name: &str, paths: &[&str]) -> bool {
    let Some((_, pins)) = REVIEWED_GEMM_MANIFESTS
        .iter()
        .find(|(package, _)| *package == name)
    else {
        return false;
    };
    if paths.len() != REVIEWED_GEMM_VERSIONS.len() {
        return false;
    }
    REVIEWED_GEMM_VERSIONS
        .iter()
        .zip(pins)
        .all(|(version, pin)| {
            let path = format!("vendor/security/{name}-{version}/Cargo.toml");
            if !paths.contains(&path.as_str()) {
                return false;
            }
            let Some(bytes) = input.manifests.get(&path) else {
                return false;
            };
            let Ok(text) = std::str::from_utf8(bytes) else {
                return false;
            };
            let Ok(table) = parse_toml(text) else {
                return false;
            };
            let Some(package) = table.get("package").and_then(toml::Value::as_table) else {
                return false;
            };
            package.get("name").and_then(toml::Value::as_str) == Some(name)
                && package.get("version").and_then(toml::Value::as_str) == Some(*version)
                && format!("{:x}", sha2::Sha256::digest(bytes)) == *pin
        })
}

/// What the rules read: the tracked paths that are UTF-8, the tracked paths that are not (each
/// byte that is not UTF-8 written `\xHH`), the workspace members, the bytes of the tracked
/// `Cargo.toml` files (decoded by [`package_name`], so a manifest that is not UTF-8 is reported),
/// the text of the tracked `crates/**/*.rs` files that mention the temp directory, and the
/// allowlist text.
#[derive(Debug, Default, Clone)]
pub struct Input {
    pub tracked: Vec<String>,
    pub undecodable: Vec<String>,
    pub members: Vec<String>,
    pub manifests: BTreeMap<String, Vec<u8>>,
    pub sources: BTreeMap<String, String>,
    pub allow: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    RootMd,
    Log,
    DirPackage,
    TempDir,
}

impl Kind {
    fn parse(word: &str) -> Option<Kind> {
        match word {
            "root-md" => Some(Kind::RootMd),
            "log" => Some(Kind::Log),
            "dir-package" => Some(Kind::DirPackage),
            "temp-dir" => Some(Kind::TempDir),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::RootMd => "root-md",
            Kind::Log => "log",
            Kind::DirPackage => "dir-package",
            Kind::TempDir => "temp-dir",
        }
    }
}

#[derive(Debug)]
struct Entry {
    line: usize,
    kind: Kind,
    path: String,
    value: Option<String>,
    used: bool,
}

/// The allowlist entries, and one finding per malformed line.
fn parse_allow(text: &str, findings: &mut Vec<String>) -> Vec<Entry> {
    let mut entries: Vec<Entry> = vec![];
    for (index, raw) in text.lines().enumerate() {
        let line = index + 1;
        let trimmed = raw.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let at = format!("{ALLOWLIST}:{line}");
        let Some((head, reason)) = trimmed.split_once(" -- ") else {
            findings.push(format!(
                "allowlist {at}: entry has no \" -- <reason>\"; state why the exception is needed"
            ));
            continue;
        };
        if reason.trim().is_empty() {
            findings.push(format!(
                "allowlist {at}: entry has an empty reason; state why the exception is needed"
            ));
            continue;
        }
        let words: Vec<&str> = head.split_whitespace().collect();
        let Some(kind) = words.first().and_then(|w| Kind::parse(w)) else {
            findings.push(format!(
                "allowlist {at}: unknown kind {:?}; use root-md, log, dir-package or temp-dir",
                words.first().copied().unwrap_or("")
            ));
            continue;
        };
        let wants_value = matches!(kind, Kind::DirPackage | Kind::TempDir);
        let expected = if wants_value { 3 } else { 2 };
        if words.len() != expected {
            findings.push(format!(
                "allowlist {at}: {} takes {}; write `{}`",
                kind.name(),
                if wants_value {
                    "a path and a value"
                } else {
                    "exactly one path"
                },
                match kind {
                    Kind::DirPackage => "dir-package <dir> <package> -- <reason>",
                    Kind::TempDir => "temp-dir <file> <count> -- <reason>",
                    _ => "<kind> <path> -- <reason>",
                }
            ));
            continue;
        }
        let value = words.get(2).map(|v| (*v).to_owned());
        if kind == Kind::TempDir
            && !value
                .as_deref()
                .and_then(|v| v.parse::<usize>().ok())
                .is_some_and(|n| n > 0)
        {
            findings.push(format!(
                "allowlist {at}: temp-dir count must be a positive integer; lower or remove this entry"
            ));
            continue;
        }
        let path = words[1].to_owned();
        if let Some(first) = entries.iter().find(|e| e.kind == kind && e.path == path) {
            findings.push(format!(
                "allowlist {at}: duplicate {} entry for {path} (first at line {}); remove this entry",
                kind.name(),
                first.line
            ));
            continue;
        }
        entries.push(Entry {
            line,
            kind,
            path,
            value,
            used: false,
        });
    }
    entries
}

fn allowed<'a>(entries: &'a mut [Entry], kind: Kind, path: &str) -> Option<&'a mut Entry> {
    entries
        .iter_mut()
        .find(|e| e.kind == kind && e.path == path)
}

/// Whether a tracked path is a Cargo manifest.
fn is_manifest(path: &str) -> bool {
    path == "Cargo.toml" || path.ends_with("/Cargo.toml")
}

/// The end of every "not TOML 1.0" reason: Cargo reads TOML 1.1, this guard's parser TOML 1.0.
const TOML_1_0: &str = "this guard reads TOML 1.0, so rewrite any TOML 1.1-only syntax";

/// `text` as a TOML 1.0 document, or why not on one line: the parser's message with the line and
/// column where it stopped, then [`TOML_1_0`].
fn parse_toml(text: &str) -> std::result::Result<toml::Table, String> {
    text.parse::<toml::Table>().map_err(|e| {
        let message = e
            .message()
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join("; ");
        match e.span() {
            Some(span) => {
                let before = text.get(..span.start).unwrap_or(text);
                let line = before.matches('\n').count() + 1;
                let column = before.rsplit('\n').next().unwrap_or("").chars().count() + 1;
                format!("not TOML 1.0 (line {line}, column {column}: {message}); {TOML_1_0}")
            }
            None => format!("not TOML 1.0 ({message}); {TOML_1_0}"),
        }
    })
}

/// "a string", "an integer", ...: the TOML type of `value` for a message.
fn a_type(value: &toml::Value) -> String {
    let name = value.type_str();
    let article = if name.starts_with(['a', 'e', 'i', 'o', 'u']) {
        "an"
    } else {
        "a"
    };
    format!("{article} {name}")
}

/// The package a `Cargo.toml` declares, read as TOML 1.0 (every TOML 1.0 spelling):
/// `Ok(Some(name))` for a `[package]` table, or Cargo's legacy `[project]` alias, with a
/// non-empty string `name`; `Ok(None)` for a manifest without one (a virtual workspace root);
/// `Err(reason)` for bytes this guard cannot read as a manifest (not UTF-8, not TOML 1.0, or a
/// package table without a non-empty string `name`).
pub fn package_name(manifest: &[u8]) -> std::result::Result<Option<String>, String> {
    let text = std::str::from_utf8(manifest).map_err(|e| {
        format!(
            "not UTF-8 (first invalid byte at offset {})",
            e.valid_up_to()
        )
    })?;
    let table = parse_toml(text)?;
    let Some((key, package)) = ["package", "project"]
        .into_iter()
        .find_map(|key| table.get(key).map(|value| (key, value)))
    else {
        return Ok(None);
    };
    let Some(package) = package.as_table() else {
        return Err(format!("`{key}` is {}, not a table", a_type(package)));
    };
    match package.get("name") {
        None => Err(format!("[{key}] has no `name`")),
        Some(toml::Value::String(name)) if name.is_empty() => {
            Err(format!("[{key}] `name` is empty"))
        }
        Some(toml::Value::String(name)) => Ok(Some(name.clone())),
        Some(other) => Err(format!("[{key}] `name` is {}, not a string", a_type(other))),
    }
}

/// The `[workspace] members` list of the root manifest (not `default-members`), read as TOML
/// 1.0. A manifest that is not TOML 1.0, or whose `members` is missing or not a list of strings,
/// refuses.
pub fn workspace_members(manifest: &str) -> Result<Vec<String>> {
    let table =
        parse_toml(manifest).map_err(|reason| Error::Shape(format!("Cargo.toml: {reason}")))?;
    let members = table
        .get("workspace")
        .and_then(|workspace| workspace.get("members"))
        .ok_or_else(|| Error::Shape("Cargo.toml: no `[workspace] members` list".into()))?;
    let Some(list) = members.as_array() else {
        return Err(Error::Shape(format!(
            "Cargo.toml: `[workspace] members` is {}, not a list",
            a_type(members)
        )));
    };
    list.iter()
        .map(|member| {
            member.as_str().map(str::to_owned).ok_or_else(|| {
                Error::Shape(format!(
                    "Cargo.toml: a `[workspace] members` entry is {}, not a string",
                    a_type(member)
                ))
            })
        })
        .collect()
}

/// Whether the whole file is test code: under a `tests/` directory, `tests.rs` or `*_tests.rs`.
fn is_test_file(path: &str) -> bool {
    let file = path.rsplit('/').next().unwrap_or(path);
    let mut dirs = path.split('/').collect::<Vec<_>>();
    dirs.pop();
    dirs.contains(&"tests") || file == "tests.rs" || file.ends_with("_tests.rs")
}

/// Per line of `text`: whether it lies in a `#[cfg(test)]` block whose next non-attribute line
/// is an inline `mod` declaration.
fn test_block_lines(text: &str) -> Vec<bool> {
    let module = Regex::new(r"^(pub(\([^)]*\))?\s+)?mod\s+[A-Za-z_][A-Za-z0-9_]*\s*\{")
        .expect("module pattern");
    let lines: Vec<&str> = text.lines().collect();
    let mut in_test = vec![false; lines.len()];
    let mut i = 0;
    while i < lines.len() {
        let trimmed = lines[i].trim_start();
        let Some(rest) = trimmed.strip_prefix("#[cfg(test)]") else {
            i += 1;
            continue;
        };
        // The declaration may share the attribute's line or follow further attributes.
        let (decl_index, decl) = if rest.trim().is_empty() {
            let mut j = i + 1;
            while j < lines.len() && lines[j].trim_start().starts_with("#[") {
                j += 1;
            }
            match lines.get(j) {
                Some(line) => (j, *line),
                None => break,
            }
        } else {
            (i, rest)
        };
        if !module.is_match(decl.trim_start()) {
            i += 1;
            continue;
        }
        let indent =
            &lines[decl_index][..lines[decl_index].len() - lines[decl_index].trim_start().len()];
        let closing = format!("{indent}}}");
        let end = (decl_index + 1..lines.len())
            .find(|&k| lines[k].trim_end() == closing)
            .unwrap_or(lines.len() - 1);
        for flag in &mut in_test[i..=end] {
            *flag = true;
        }
        i = end + 1;
    }
    in_test
}

fn temp_dir_pattern() -> Regex {
    Regex::new(r"\benv::temp_dir\(\)").expect("temp-dir pattern")
}

/// How often test code in `path` calls the process-wide temp directory.
pub fn test_temp_dir_calls(path: &str, text: &str) -> usize {
    let pattern = temp_dir_pattern();
    if is_test_file(path) {
        return pattern.find_iter(text).count();
    }
    let in_test = test_block_lines(text);
    text.lines()
        .zip(in_test)
        .filter(|(_, test)| *test)
        .map(|(line, _)| pattern.find_iter(line).count())
        .sum()
}

/// The `crates/**/*.rs` files whose test code calls the temp directory, with the count.
pub fn temp_dir_counts(input: &Input) -> Vec<(String, usize)> {
    let tracked: std::collections::BTreeSet<&str> =
        input.tracked.iter().map(String::as_str).collect();
    input
        .sources
        .iter()
        .filter(|(path, _)| {
            tracked.contains(path.as_str()) && path.starts_with("crates/") && path.ends_with(".rs")
        })
        .map(|(path, text)| (path.clone(), test_temp_dir_calls(path, text)))
        .filter(|(_, n)| *n > 0)
        .collect()
}

/// Every finding for `input`, in a fixed order: allowlist syntax, then the rules R1..R7, then the
/// entries nothing used.
pub fn check(input: &Input) -> Vec<String> {
    let mut findings = vec![];
    let mut entries = parse_allow(&input.allow, &mut findings);

    // Every tracked Cargo.toml read once, by path; R3, R4 and R6 share the reading.
    let readings: BTreeMap<&str, std::result::Result<Option<String>, String>> = input
        .tracked
        .iter()
        .filter(|path| is_manifest(path))
        .filter_map(|path| {
            input
                .manifests
                .get(path)
                .map(|bytes| (path.as_str(), package_name(bytes)))
        })
        .collect();

    // R1 root-md and R2 log.
    let root_md = Regex::new(r"^[^/]+\.md$").expect("root-md pattern");
    let log = Regex::new(r"\.(log|err|stderr)$").expect("log pattern");
    for path in &input.tracked {
        if root_md.is_match(path) {
            match allowed(&mut entries, Kind::RootMd, path) {
                Some(entry) => entry.used = true,
                None => findings.push(format!(
                    "root-md {path}: Markdown file at the repository root; move it under docs/ \
                     (or add a reviewed `root-md {path} -- <reason>` entry to {ALLOWLIST})"
                )),
            }
        }
        if log.is_match(path) {
            match allowed(&mut entries, Kind::Log, path) {
                Some(entry) => entry.used = true,
                None => findings.push(format!(
                    "log {path}: tracked log/stderr capture; delete it or keep it as custody \
                     outside git (or add a reviewed `log {path} -- <reason>` entry to {ALLOWLIST})"
                )),
            }
        }
    }

    // R3 dir-package.
    for member in &input.members {
        let manifest_path = format!("{member}/Cargo.toml");
        let name = match readings.get(manifest_path.as_str()) {
            Some(Ok(Some(name))) => name,
            // Reported once, by R6 bad-manifest.
            Some(Err(_)) => continue,
            Some(Ok(None)) | None => {
                findings.push(format!(
                    "dir-package {member}: workspace member without a tracked {manifest_path} \
                     [package] name; fix the members list in Cargo.toml"
                ));
                continue;
            }
        };
        let directory = member.rsplit('/').next().unwrap_or(member);
        if directory == name {
            continue;
        }
        match allowed(&mut entries, Kind::DirPackage, member) {
            Some(entry) if entry.value.as_deref() == Some(name.as_str()) => entry.used = true,
            _ => findings.push(format!(
                "dir-package {member}: directory `{directory}` holds package `{name}`; rename one \
                 to match (or add a reviewed `dir-package {member} {name} -- <reason>` entry to \
                 {ALLOWLIST})"
            )),
        }
    }

    // R4 dup-package: only the exact reviewed, distinct-version vendor pairs are identities.
    // Product duplicates, extra copies, path lookalikes and manifest changes still refuse.
    let mut declared: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for (path, reading) in &readings {
        if let Ok(Some(name)) = reading {
            declared.entry(name.as_str()).or_default().push(*path);
        }
    }
    for (name, paths) in &declared {
        if let [first, others @ ..] = paths.as_slice() {
            if !others.is_empty() && !reviewed_vendor_pair(input, name, paths) {
                findings.push(format!(
                    "dup-package {first}: package `{name}` is also declared by {}; rename one \
                     package or remove the copy (this rule has no allowlist)",
                    others.join(", ")
                ));
            }
        }
    }

    // R5 temp-dir in test code.
    for (path, count) in temp_dir_counts(input) {
        match allowed(&mut entries, Kind::TempDir, &path) {
            None => findings.push(format!(
                "temp-dir {path}: {count} call(s) of std::env::temp_dir in test code; use \
                 tempfile::Builder::new().prefix(..).tempdir() so the directory is removed on drop \
                 (the allowlist is shrink-only)"
            )),
            Some(entry) => {
                entry.used = true;
                let permitted: usize = entry
                    .value
                    .as_deref()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                if count > permitted {
                    findings.push(format!(
                        "temp-dir {path}: {count} call(s) of std::env::temp_dir in test code, \
                         {ALLOWLIST}:{} permits {permitted}; use tempfile for the new call (the \
                         allowlist is shrink-only)",
                        entry.line
                    ));
                } else if count < permitted {
                    findings.push(format!(
                        "temp-dir {path}: {ALLOWLIST}:{} permits {permitted} but test code has \
                         {count}; lower or remove this entry",
                        entry.line
                    ));
                }
            }
        }
    }

    // R6 bad-manifest (no allowlist): R3 and R4 cannot see into a manifest this guard cannot read.
    for (path, reading) in &readings {
        if let Err(reason) = reading {
            findings.push(format!(
                "bad-manifest {path}: {reason}; fix it so this guard can read its package name, or \
                 give a deliberately malformed test input another file name (this rule has no \
                 allowlist)"
            ));
        }
    }

    // R7 bad-path (no allowlist): every other rule matches tracked paths as text, so none of them
    // can check a path that is not UTF-8.
    for path in &input.undecodable {
        findings.push(format!(
            "bad-path {path}: tracked path is not UTF-8 (\\xHH is a byte that is not), so no \
             other rule can check it; rename it to UTF-8 with `git mv` (this rule has no \
             allowlist)"
        ));
    }

    // Shrink-only: an entry nothing used is stale.
    for entry in entries.iter().filter(|e| !e.used) {
        findings.push(format!(
            "{} {}: {ALLOWLIST}:{} matches nothing (untracked path, or no longer a violation); \
             lower or remove this entry",
            entry.kind.name(),
            entry.path,
            entry.line
        ));
    }
    findings
}

/// The report the hygiene step prints, and whether the tree passes.
pub fn render(input: &Input, findings: &[String]) -> (String, bool) {
    if findings.is_empty() {
        let entries = input
            .allow
            .lines()
            .filter(|l| {
                let t = l.trim();
                !t.is_empty() && !t.starts_with('#')
            })
            .count();
        return (
            format!(
                "ok: repository layout: {} tracked paths, {} workspace members, {entries} \
                 allowlist entries, 0 findings\n",
                input.tracked.len(),
                input.members.len()
            ),
            true,
        );
    }
    let mut out = String::new();
    for finding in findings {
        out.push_str(finding);
        out.push('\n');
    }
    out.push_str(&format!(
        "repo-layout: {} finding(s); reviewed exceptions live in {ALLOWLIST}\n",
        findings.len()
    ));
    (out, false)
}

/// The bytes of the tracked file `path`, or `None` when `git ls-files --deleted` reported it
/// deleted in the working tree (`deleted`). Any other tracked file that cannot be read (a dangling
/// symlink, say) refuses: a file is never skipped because this guard failed to open it.
fn read_tracked(root: &Path, path: &str, deleted: &BTreeSet<Vec<u8>>) -> Result<Option<Vec<u8>>> {
    if deleted.contains(path.as_bytes()) {
        return Ok(None);
    }
    let file = root.join(path);
    std::fs::read(&file)
        .map(Some)
        .map_err(|e| Error::io(&file, e))
}

/// `git ls-files -z <args>` at `root`: each path as the bytes git prints, never decoded.
fn ls_files(root: &Path, args: &[&str]) -> Result<Vec<Vec<u8>>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .args(args)
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "git ls-files -z{} at {} exited {}: {}",
            args.iter().map(|a| format!(" {a}")).collect::<String>(),
            root.display(),
            crate::py::returncode(output.status),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output
        .stdout
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(<[u8]>::to_vec)
        .collect())
}

/// `path` for a finding: its UTF-8 runs as they are, each byte that is not UTF-8 as `\xHH`.
fn escape_path(path: &[u8]) -> String {
    let mut out = String::new();
    for chunk in path.utf8_chunks() {
        out.push_str(chunk.valid());
        for byte in chunk.invalid() {
            out.push_str(&format!("\\x{byte:02x}"));
        }
    }
    out
}

/// Read the [`Input`] of the checkout at `root`. A missing root manifest or allowlist refuses, and
/// so does a tracked file the rules need that git does not report deleted but cannot be read.
pub fn gather(root: &Path) -> Result<Input> {
    let mut tracked = vec![];
    let mut undecodable = vec![];
    for path in ls_files(root, &[])? {
        match String::from_utf8(path) {
            Ok(path) => tracked.push(path),
            Err(e) => undecodable.push(escape_path(e.as_bytes())),
        }
    }
    let deleted: BTreeSet<Vec<u8>> = ls_files(root, &["--deleted"])?.into_iter().collect();
    let manifest = crate::py::read_text(&root.join("Cargo.toml"))?;
    let members = workspace_members(&manifest)?;
    let allow = crate::py::read_text(&root.join(ALLOWLIST))?;
    let mut manifests = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for path in &tracked {
        if is_manifest(path) {
            if let Some(bytes) = read_tracked(root, path, &deleted)? {
                manifests.insert(path.clone(), bytes);
            }
        } else if path.starts_with("crates/") && path.ends_with(".rs") {
            if let Some(bytes) = read_tracked(root, path, &deleted)? {
                let text = String::from_utf8_lossy(&bytes).into_owned();
                if text.contains("temp_dir()") {
                    sources.insert(path.clone(), text);
                }
            }
        }
    }
    Ok(Input {
        tracked,
        undecodable,
        members,
        manifests,
        sources,
        allow,
    })
}

/// The `repo-layout` command: the report and exit status, or with `print_temp_dir` the current
/// `temp-dir <path> <count>` lines (the generator of the allowlist's temp-dir entries).
pub fn run(root: &Path, print_temp_dir: bool) -> Result<(String, bool)> {
    let input = gather(root)?;
    if print_temp_dir {
        let mut out = String::new();
        for (path, count) in temp_dir_counts(&input) {
            out.push_str(&format!("temp-dir {path} {count}\n"));
        }
        return Ok((out, true));
    }
    let findings = check(&input);
    Ok(render(&input, &findings))
}

/// A fixture document: `{"tracked": [...], "members": [...], "manifests": {path: text},
/// "sources": {path: text}, "allow": text, "expect": [lines]}`.
pub fn fixture(value: &Value) -> Result<(Input, Vec<String>)> {
    let strings = |key: &str| -> Result<Vec<String>> {
        value
            .get(key)
            .and_then(Value::as_list)
            .ok_or_else(|| Error::Shape(format!("fixture: {key} is not a list")))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| Error::Shape(format!("fixture: {key} holds a non-string")))
            })
            .collect()
    };
    let texts = |key: &str| -> Result<BTreeMap<String, String>> {
        let Some(dict) = value.get(key).and_then(Value::as_dict) else {
            return Err(Error::Shape(format!("fixture: {key} is not an object")));
        };
        dict.iter()
            .map(|(k, v)| {
                v.as_str()
                    .map(|t| (k.to_owned(), t.to_owned()))
                    .ok_or_else(|| Error::Shape(format!("fixture: {key}[{k}] is not a string")))
            })
            .collect()
    };
    let input = Input {
        tracked: strings("tracked")?,
        // A path that is not UTF-8 cannot be a JSON string; the CLI tests build one in git.
        undecodable: vec![],
        members: strings("members")?,
        manifests: texts("manifests")?
            .into_iter()
            .map(|(path, text)| (path, text.into_bytes()))
            .collect(),
        sources: texts("sources")?,
        allow: value
            .get("allow")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Shape("fixture: allow is not a string".into()))?
            .to_owned(),
    };
    Ok((input, strings("expect")?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn reviewed_vendor_input() -> Input {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mut input = Input::default();
        for (name, _) in REVIEWED_GEMM_MANIFESTS {
            for version in REVIEWED_GEMM_VERSIONS {
                let path = format!("vendor/security/{name}-{version}/Cargo.toml");
                input
                    .manifests
                    .insert(path.clone(), std::fs::read(root.join(&path)).unwrap());
                input.tracked.push(path);
            }
        }
        input
    }

    #[test]
    fn reviewed_vendor_pairs_match_exact_custody_identities() {
        let input = reviewed_vendor_input();
        assert_eq!(input.manifests.len(), 14);
        assert!(check(&input).is_empty());
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let provenance =
            crate::py::load_json(&root.join("vendor/security/PROVENANCE.json")).unwrap();
        let crates = provenance.get("crates").and_then(Value::as_list).unwrap();
        for (name, pins) in REVIEWED_GEMM_MANIFESTS {
            for (version, pin) in REVIEWED_GEMM_VERSIONS.iter().zip(pins) {
                let matching: Vec<_> = crates
                    .iter()
                    .filter(|record| {
                        record.get("name").and_then(Value::as_str) == Some(*name)
                            && record.get("version").and_then(Value::as_str) == Some(*version)
                    })
                    .collect();
                assert_eq!(matching.len(), 1, "{name}-{version}");
                let record = matching[0];
                assert_eq!(
                    record.get("directory").and_then(Value::as_str),
                    Some(format!("{name}-{version}").as_str())
                );
                assert_eq!(
                    record
                        .get("manifest_candidate_sha256")
                        .and_then(Value::as_str),
                    Some(*pin)
                );
            }
        }
    }

    #[test]
    fn reviewed_vendor_pairs_do_not_hide_copies_path_spoofs_or_manifest_tampering() {
        let original = reviewed_vendor_input();
        let old = "vendor/security/gemm-0.18.2/Cargo.toml";
        let new = "vendor/security/gemm-0.19.0/Cargo.toml";
        let rejects = |input: Input, label: &str| {
            assert!(
                check(&input)
                    .iter()
                    .any(|finding| finding.starts_with("dup-package ")),
                "{label}"
            );
        };
        // A product package colliding with a legitimate imported identity is still a duplicate.
        let mut input = original.clone();
        let copy = "crates/gemm/Cargo.toml".to_owned();
        input.tracked.push(copy.clone());
        input.manifests.insert(copy, input.manifests[old].clone());
        rejects(input, "product collision / third copy");
        // The path must be exact, even when bytes and parsed identity remain authentic.
        for spoof in [
            "vendor/security-copy/gemm-0.19.0/Cargo.toml",
            "vendor/security/gemm-0.19.0-copy/Cargo.toml",
            "vendor/security/nested/gemm-0.19.0/Cargo.toml",
            "vendor/security/gemm-0.20.0/Cargo.toml",
        ] {
            let mut input = original.clone();
            let bytes = input.manifests.remove(new).unwrap();
            input.tracked.retain(|path| path != new);
            input.tracked.push(spoof.to_owned());
            input.manifests.insert(spoof.to_owned(), bytes);
            rejects(input, spoof);
        }
        let mut input = original.clone();
        input
            .manifests
            .insert(new.to_owned(), input.manifests[old].clone());
        rejects(input, "same version at another reviewed directory");
        let mut input = original.clone();
        input
            .manifests
            .get_mut(new)
            .unwrap()
            .extend_from_slice(b"\n# unreviewed manifest edit\n");
        rejects(input, "even identity-preserving manifest tamper");
        // Distinct versions are not a blanket exception for product or unreviewed vendor names.
        for prefix in ["crates", "vendor/security"] {
            let mut input = Input::default();
            for version in ["1.0.0", "2.0.0"] {
                let path = format!("{prefix}/other-{version}/Cargo.toml");
                input.tracked.push(path.clone());
                input.manifests.insert(
                    path,
                    format!("[package]\nname = \"other\"\nversion = \"{version}\"\n").into_bytes(),
                );
            }
            rejects(input, prefix);
        }
    }

    fn fixtures() -> Vec<(PathBuf, Value)> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/repo-layout");
        let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("fixture directory")
            .map(|e| e.expect("fixture entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "json"))
            .collect();
        paths.sort();
        paths
            .into_iter()
            .map(|p| {
                let value = crate::py::load_json(&p).expect("fixture JSON");
                (p, value)
            })
            .collect()
    }

    /// Every fixture's findings are exactly its `expect` lines, and the named cases all exist.
    #[test]
    fn fixtures_report_exactly_their_expected_findings() {
        let all = fixtures();
        let names: Vec<String> = all
            .iter()
            .map(|(p, _)| p.file_stem().unwrap().to_string_lossy().into_owned())
            .collect();
        for required in [
            "clean",
            "nested-md-ok",
            "production-temp-dir-ignored",
            "root-md",
            "tracked-log",
            "dir-mismatch",
            "dup-package",
            "temp-dir-new",
            "temp-dir-over",
            "stale-allow-path",
            "stale-allow-count",
            // Review 5344298377: every TOML spelling of a package name, and manifests this
            // guard cannot read.
            "dup-package-literal",
            "dup-package-dotted",
            "dup-package-escaped",
            "dir-mismatch-literal",
            "bad-manifest",
        ] {
            assert!(
                names.iter().any(|n| n == required),
                "missing fixture {required}"
            );
        }
        for (path, value) in &all {
            let (input, expect) = fixture(value).expect("fixture shape");
            let findings = check(&input);
            assert_eq!(findings, expect, "{}", path.display());
            let (_, ok) = render(&input, &findings);
            assert_eq!(ok, expect.is_empty(), "{}", path.display());
        }
    }

    #[test]
    fn members_come_from_members_not_default_members() {
        let manifest = "[workspace]\nresolver = \"2\"\nmembers = [\n  \"crates/a\", # first\n  \"crates/b\",\n]\ndefault-members = [\"crates/a\"]\n";
        assert_eq!(
            workspace_members(manifest).unwrap(),
            ["crates/a", "crates/b"]
        );
        let default_first =
            "[workspace]\ndefault-members = [\"crates/x\"]\nmembers = [\"crates/y\"]\n";
        assert_eq!(workspace_members(default_first).unwrap(), ["crates/y"]);
        assert!(workspace_members("[package]\nname = \"x\"\n").is_err());
        // Review 5344298377: every TOML spelling of the list, read as TOML. A `members` key
        // outside `[workspace]` is not the list, and a `]` inside a string does not end it.
        let spellings = [
            "[workspace]\nmembers = ['crates/a', \"crates/b\"]\n",
            "workspace.members = [\n  'crates/a', # literal\n  \"crates/\\u0062\",\n]\n",
            "workspace = { members = ['''crates/a''', \"\"\"crates/b\"\"\"] }\n",
            "[workspace]\r\nmembers = [\r\n  'crates/a',\r\n  'crates/b',\r\n]\r\n",
            "[package]\nmembers = ['crates/z']\n\n[workspace]\nmembers = ['crates/a', 'crates/b']\n",
        ];
        for manifest in spellings {
            assert_eq!(
                workspace_members(manifest).unwrap(),
                ["crates/a", "crates/b"],
                "{manifest:?}"
            );
        }
        assert_eq!(
            workspace_members("[workspace]\nmembers = ['odd]dir']\n").unwrap(),
            ["odd]dir"]
        );
        // Refusals are errors, never an empty or partial list.
        for bad in [
            "[workspace]\nmembers = ['crates/a'\n",
            "[workspace]\nmembers = 'crates/a'\n",
            "[workspace]\nmembers = ['crates/a', 1]\n",
            "[package]\nmembers = ['crates/a']\n",
        ] {
            assert!(workspace_members(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn package_name_reads_only_the_package_table() {
        let manifest = "[package]\nversion = \"1\"\nname = \"real\"\n\n[[bin]]\nname = \"other\"\n";
        assert_eq!(package_name(manifest.as_bytes()), Ok(Some("real".into())));
        assert_eq!(package_name(b"[[bin]]\nname = \"bin-only\"\n"), Ok(None));
        assert_eq!(package_name(b"[workspace]\nmembers = []\n"), Ok(None));
        // A `name` in a sub-table of [package] is not the package name.
        assert_eq!(
            package_name(b"[package]\nversion = '1'\n\n[package.metadata]\nname = 'not-this'\n"),
            Err("[package] has no `name`".to_owned())
        );
    }

    /// Review 5344298377: each spelling Cargo accepts declares package `x`.
    #[test]
    fn package_name_reads_every_toml_spelling() {
        let spellings = [
            "[package]\nname = \"x\"\n",
            "[package]\nname = 'x'\n",
            "[package]\nname = \"\"\"x\"\"\"\n",
            "[package]\nname = '''\nx'''\n",
            "[package]\nname = \"\\u0078\"\n",
            "[package]\nname = \"\\U00000078\"\n",
            "package.name = 'x'\n",
            "package = { name = 'x', version = '0.1.0' }\n",
            "[ package ] # the package\n\"name\" = 'x' # comment\n",
            "[package]\r\nversion = \"0.1.0\"\r\nname = 'x'\r\n",
            "\u{feff}[package]\nname = \"x\"\n",
            "[project]\nname = 'x'\n",
        ];
        for manifest in spellings {
            assert_eq!(
                package_name(manifest.as_bytes()),
                Ok(Some("x".to_owned())),
                "{manifest:?}"
            );
        }
    }

    /// Review 5344298377: what this guard cannot read is a reason, never a silent `None`.
    #[test]
    fn package_name_refuses_what_cargo_cannot_read() {
        let cases: [(&[u8], &str); 8] = [
            (
                b"[package]\nname = \"x\xff\"\n",
                "not UTF-8 (first invalid byte at offset 19)",
            ),
            (b"[package]\nname = 'x\n", "not TOML 1.0 (line 2, column "),
            (
                b"[package]\nname = \"x\"\nname = \"y\"\n",
                "not TOML 1.0 (line 3, column ",
            ),
            (b"package = 'x'\n", "`package` is a string, not a table"),
            (b"[package]\nversion = '0.1.0'\n", "[package] has no `name`"),
            (
                b"[package]\nname = 1\n",
                "[package] `name` is an integer, not a string",
            ),
            (
                b"[package]\nname = { workspace = true }\n",
                "[package] `name` is a table, not a string",
            ),
            (b"[package]\nname = ''\n", "[package] `name` is empty"),
        ];
        for (manifest, reason) in cases {
            let got = package_name(manifest).expect_err(&String::from_utf8_lossy(manifest));
            assert!(
                got.starts_with(reason) && !got.contains('\n'),
                "{got:?} does not start with {reason:?} on one line"
            );
        }
    }

    /// Verifier finding on 7982c479f: this guard parses TOML 1.0 (the `toml` 0.8 crate) while
    /// Cargo reads TOML 1.1, so a manifest that uses 1.1-only syntax is refused with a reason that
    /// names TOML 1.0 and says what to rewrite, never one that claims Cargo could not read it.
    #[test]
    fn package_name_says_toml_1_0_for_toml_1_1_only_syntax() {
        for manifest in [
            // The `\e` escape.
            "[package]\nname = 'x'\ndescription = \"\\e[1m\"\n",
            // The `\xHH` escape (Cargo reads package `x`).
            "[package]\nname = \"\\x78\"\n",
            // A newline and a trailing comma in an inline table.
            "package = {\n  name = 'x',\n}\n",
        ] {
            let got = package_name(manifest.as_bytes()).expect_err(manifest);
            assert!(
                got.starts_with("not TOML 1.0 (line ")
                    && got.ends_with(
                        "; this guard reads TOML 1.0, so rewrite any TOML 1.1-only syntax"
                    )
                    && !got.contains('\n'),
                "{got:?} for {manifest:?}"
            );
        }
    }

    /// The call pattern is assembled at run time so this test module never contains it.
    fn call() -> String {
        format!("std::env::{}()", "temp_dir")
    }

    #[test]
    fn test_code_is_whole_test_files_and_cfg_test_modules() {
        let c = call();
        let production = format!("fn f() {{ {c}; }}\n#[cfg(test)]\nmod tests {{\n    fn g() {{ {c}; {c}; }}\n}}\nfn h() {{ {c}; }}\n");
        assert_eq!(test_temp_dir_calls("crates/x/src/lib.rs", &production), 2);
        assert_eq!(test_temp_dir_calls("crates/x/tests/t.rs", &production), 4);
        assert_eq!(
            test_temp_dir_calls("crates/x/src/a/tests.rs", &production),
            4
        );
        assert_eq!(
            test_temp_dir_calls("crates/x/src/queue_tests.rs", &production),
            4
        );
        // An out-of-line `mod tests;` and a cfg(test) function are not inline test modules.
        let out_of_line =
            format!("#[cfg(test)]\nmod tests;\n#[cfg(test)]\nfn helper() {{ {c}; }}\n");
        assert_eq!(test_temp_dir_calls("crates/x/src/lib.rs", &out_of_line), 0);
        // Attributes between cfg(test) and the module, and nested indentation.
        let nested = format!("pub mod outer {{\n    #[cfg(test)]\n    #[allow(dead_code)]\n    pub(crate) mod tests {{\n        fn g() {{ {c}; }}\n    }}\n    fn after() {{ {c}; }}\n}}\n");
        assert_eq!(test_temp_dir_calls("crates/x/src/lib.rs", &nested), 1);
    }
}
