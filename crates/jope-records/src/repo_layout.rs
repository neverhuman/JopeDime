//! `repo-layout`: the repository layout guard of the hygiene lane.
//!
//! Five rules over the tracked tree (`git ls-files` at the checkout root). Each finding is one
//! line that names the rule, the path and the fix:
//!
//! | Rule | Finds |
//! |---|---|
//! | `root-md` | a tracked Markdown file at the repository root that is not allowlisted |
//! | `log` | a tracked `.log`, `.err` or `.stderr` file that is not allowlisted |
//! | `dir-package` | a workspace member whose directory name differs from its `[package] name` |
//! | `dup-package` | two tracked `Cargo.toml` files declaring the same package name (no allowlist) |
//! | `temp-dir` | calls of the process-wide temp directory (`std::env::temp_dir`) in test code under `crates/`, above the allowlisted count |
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

use std::collections::BTreeMap;
use std::path::Path;
use std::process::Command;

use regex::Regex;

use crate::pyjson::Value;
use crate::{Error, Result};

/// The reviewed exceptions, relative to the repository root.
pub const ALLOWLIST: &str = "deploy/ci/repo-layout-allow.txt";

/// What the rules read: the tracked paths, the workspace members, the text of the tracked
/// `Cargo.toml` files, the text of the tracked `crates/**/*.rs` files that mention the temp
/// directory, and the allowlist text.
#[derive(Debug, Default, Clone)]
pub struct Input {
    pub tracked: Vec<String>,
    pub members: Vec<String>,
    pub manifests: BTreeMap<String, String>,
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

/// The `[package] name` of a manifest, read line by line: the first `name = "..."` between a
/// `[package]` header and the next table header.
pub fn package_name(manifest: &str) -> Option<String> {
    let name = Regex::new(r#"^\s*name\s*=\s*"([^"]+)""#).expect("name pattern");
    let mut in_package = false;
    for line in manifest.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            let header = trimmed.split('#').next().unwrap_or("").trim();
            in_package = header == "[package]";
            continue;
        }
        if in_package {
            if let Some(c) = name.captures(line) {
                return Some(c[1].to_owned());
            }
        }
    }
    None
}

/// The `members = [ ... ]` list of the root manifest (not `default-members`), comments removed.
pub fn workspace_members(manifest: &str) -> Result<Vec<String>> {
    let list = Regex::new(r"(?ms)^\s*members\s*=\s*\[(.*?)\]").expect("members pattern");
    let quoted = Regex::new(r#""([^"]*)""#).expect("quoted pattern");
    let inner = list
        .captures(manifest)
        .ok_or_else(|| Error::Shape("Cargo.toml: no workspace `members = [ ... ]` list".into()))?;
    let mut members = vec![];
    for line in inner[1].lines() {
        let code = line.split('#').next().unwrap_or("");
        members.extend(quoted.captures_iter(code).map(|c| c[1].to_owned()));
    }
    Ok(members)
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

/// Every finding for `input`, in a fixed order: allowlist syntax, then the rules R1..R5, then the
/// entries nothing used.
pub fn check(input: &Input) -> Vec<String> {
    let mut findings = vec![];
    let mut entries = parse_allow(&input.allow, &mut findings);

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
        let Some(name) = input
            .manifests
            .get(&manifest_path)
            .and_then(|m| package_name(m))
        else {
            findings.push(format!(
                "dir-package {member}: workspace member without a tracked {manifest_path} [package] \
                 name; fix the members list in Cargo.toml"
            ));
            continue;
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

    // R4 dup-package (no allowlist).
    let mut declared: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for path in &input.tracked {
        if path != "Cargo.toml" && !path.ends_with("/Cargo.toml") {
            continue;
        }
        if let Some(name) = input.manifests.get(path).and_then(|m| package_name(m)) {
            declared.entry(name).or_default().push(path);
        }
    }
    for (name, paths) in &declared {
        if let [first, others @ ..] = paths.as_slice() {
            if !others.is_empty() {
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

fn read_lossy(path: &Path) -> Result<Option<String>> {
    match std::fs::read(path) {
        Ok(bytes) => Ok(Some(String::from_utf8_lossy(&bytes).into_owned())),
        // A tracked path deleted in the working tree has nothing to check.
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(Error::io(path, e)),
    }
}

/// `git ls-files -z` at `root`.
fn tracked_paths(root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z"])
        .output()
        .map_err(|e| Error::io(Path::new("git"), e))?;
    if !output.status.success() {
        return Err(Error::Command(format!(
            "git ls-files at {} exited {}: {}",
            root.display(),
            crate::py::returncode(output.status),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output
        .stdout
        .split(|b| *b == 0)
        .filter(|p| !p.is_empty())
        .map(|p| String::from_utf8_lossy(p).into_owned())
        .collect())
}

/// Read the [`Input`] of the checkout at `root`. A missing root manifest or allowlist refuses.
pub fn gather(root: &Path) -> Result<Input> {
    let tracked = tracked_paths(root)?;
    let manifest = crate::py::read_text(&root.join("Cargo.toml"))?;
    let members = workspace_members(&manifest)?;
    let allow = crate::py::read_text(&root.join(ALLOWLIST))?;
    let mut manifests = BTreeMap::new();
    let mut sources = BTreeMap::new();
    for path in &tracked {
        if path == "Cargo.toml" || path.ends_with("/Cargo.toml") {
            if let Some(text) = read_lossy(&root.join(path))? {
                manifests.insert(path.clone(), text);
            }
        } else if path.starts_with("crates/") && path.ends_with(".rs") {
            if let Some(text) = read_lossy(&root.join(path))? {
                if text.contains("temp_dir()") {
                    sources.insert(path.clone(), text);
                }
            }
        }
    }
    Ok(Input {
        tracked,
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
        members: strings("members")?,
        manifests: texts("manifests")?,
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
    }

    #[test]
    fn package_name_reads_only_the_package_table() {
        let manifest = "[package]\nversion = \"1\"\nname = \"real\"\n\n[[bin]]\nname = \"other\"\n";
        assert_eq!(package_name(manifest).as_deref(), Some("real"));
        assert_eq!(package_name("[[bin]]\nname = \"bin-only\"\n"), None);
        assert_eq!(package_name("[workspace]\nmembers = []\n"), None);
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
