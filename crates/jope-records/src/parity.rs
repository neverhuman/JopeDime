//! `parity-check`: the capability-parity guard behind `deploy/ci/parity-matrix.sh`.
//!
//! Rust port of the `check` mode of `deploy/ci/parity-census.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`), with the same rules, report and problem lines:
//!
//! 1. every row has a target or a dropped/lost/pending status (dropped rows name a reason and the
//!    archive bundle);
//! 2. every binary, clap subcommand (source enum plus, when given, the parsed `--help`), example,
//!    Python tool (`python/src/jope/{tools,research}/*.{py,sh}`), `jope-py` command and systemd
//!    unit present in this tree has a row;
//! 3. `pending-WP<N>` has N in 4..12;
//! 4. a `moved`/`new` row's path-like target exists;
//! 5. the committed `## Counts` table and its covering-test tally agree with the rows recomputed
//!    from `## Rows`. Two lanes that each add a row of the same kind both bump the same cell to
//!    the same number, so the text is byte-identical, there is no conflict, and the merged file
//!    silently under-counts by one; the gate recomputes the table instead of trusting it.
//!
//! The `census` mode (which regenerated the matrix from a read-only pre-port checkout) was a
//! one-off generator with no consumer and is archived with the Python file, not ported.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::process::Command;

use regex::Regex;

use crate::py;
use crate::pyjson::{self, Value};
use crate::{Error, Result};

pub const ARCHIVE: &str = "s3://veox-jopedime-archive-use2/git/20260918-jope-dime-all-refs.bundle";
pub const MATRIX: &str = "docs/parity/capability-matrix.md";
const COUNT_CLASSES: [&str; 5] = ["moved", "new", "pending", "dropped", "lost"];

/// Why the `## Counts` table is gated at all. Required by the check, so that regenerating the
/// matrix cannot silently drop the rule and hand-editing cannot either.
pub const COUNTS_PREAMBLE: [&str; 4] = [
    "Recomputed from the rows below and checked by `deploy/ci/parity-matrix.sh`: a tally that disagrees",
    "with them fails the gate. Two lanes adding a row of the same kind each bump the same cell to the",
    "same number, so git reports no conflict — update this table from the rows, not from the last value",
    "you saw.",
];

/// One `## Rows` line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub kind: String,
    pub old: String,
    pub new: String,
    pub test: String,
    pub status: String,
    pub notes: String,
}

/// clap's default subcommand naming (heck `to_kebab_case`) for a CamelCase variant.
pub fn kebab(name: &str) -> String {
    let chars: Vec<char> = name.chars().collect();
    let mut out = String::new();
    for (i, ch) in chars.iter().enumerate() {
        if i > 0 && ch.is_uppercase() {
            let prev = chars[i - 1];
            let next = chars.get(i + 1).copied();
            if prev.is_lowercase()
                || prev.is_numeric()
                || (prev.is_uppercase() && next.is_some_and(char::is_lowercase))
            {
                out.push('-');
            }
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// A top-level variant of `enum <name> { ... }`.
#[derive(Clone, Debug)]
pub struct Variant {
    pub variant: String,
    pub name: String,
    pub neural: bool,
    pub hidden: bool,
}

/// Top-level variants of `enum <enum_name> { ... }` with their outer attributes.
pub fn parse_clap_enum(path: &Path, enum_name: &str) -> Result<Vec<Variant>> {
    let src = String::from_utf8_lossy(&py::read_bytes(path)?).into_owned();
    let block = Regex::new(&format!(
        r"(?s)enum {} \{{\n(.*?)\n\}}\n",
        regex::escape(enum_name)
    ))
    .expect("escaped pattern");
    let body = block
        .captures(&src)
        .ok_or_else(|| Error::Shape(format!("{}: enum {enum_name} not found", path.display())))?[1]
        .to_owned();
    let attr = Regex::new(r"^    #\[").expect("static pattern");
    let variant = Regex::new(r"^    ([A-Z][A-Za-z0-9]*)\s*(\{|,|\()").expect("static pattern");
    let named = Regex::new(r#"name\s*=\s*"([^"]+)""#).expect("static pattern");
    let mut rows = vec![];
    let mut attrs: Vec<String> = vec![];
    for line in py::splitlines(&body) {
        if attr.is_match(line) {
            attrs.push(py::strip(line).to_owned());
            continue;
        }
        if let Some(m) = variant.captures(line) {
            let variant = m[1].to_owned();
            let mut name = kebab(&variant);
            for a in &attrs {
                if let Some(n) = named.captures(a) {
                    if a.starts_with("#[command") {
                        name = n[1].to_owned();
                    }
                }
            }
            rows.push(Variant {
                variant,
                name,
                neural: attrs.iter().any(|a| a.contains(r#"feature = "neural""#)),
                hidden: attrs.iter().any(|a| a.contains("hide = true")),
            });
            attrs.clear();
        }
    }
    Ok(rows)
}

/// Subcommand names from clap `--help` output (the `Commands:` block).
pub fn parse_help_commands(text: &str) -> Vec<String> {
    let name = Regex::new(r"^  ([a-z0-9][a-z0-9-]*)(\s|$)").expect("static pattern");
    let mut names = vec![];
    let mut inside = false;
    for line in py::splitlines(text) {
        if py::strip(line) == "Commands:" {
            inside = true;
            continue;
        }
        if inside {
            if py::strip(line).is_empty() || line.starts_with("Options") {
                if !names.is_empty() {
                    break;
                }
                continue;
            }
            if let Some(m) = name.captures(line) {
                if &m[1] != "help" {
                    names.push(m[1].to_owned());
                }
            }
        }
    }
    names
}

/// Command paths (`data build`, `train`, ...) from an argparse CLI built with `add_parser`.
pub fn parse_argparse_commands(path: &Path) -> Result<Vec<String>> {
    let src = String::from_utf8_lossy(&py::read_bytes(path)?).into_owned();
    let subparsers = Regex::new(r"(\w+)\s*=\s*(\w+)\.add_subparsers\(").expect("static pattern");
    let parsers =
        Regex::new(r#"(\w+)\s*=\s*(\w+)\.add_parser\(\s*"([^"]+)""#).expect("static pattern");
    let owner: BTreeMap<String, String> = subparsers
        .captures_iter(&src)
        .map(|m| (m[1].to_owned(), m[2].to_owned()))
        .collect();
    let mut command_of: BTreeMap<String, String> = BTreeMap::new();
    let mut out = vec![];
    for m in parsers.captures_iter(&src) {
        let (var, subs, name) = (&m[1], &m[2], &m[3]);
        let prefix = owner
            .get(subs)
            .and_then(|parent| command_of.get(parent))
            .cloned()
            .unwrap_or_default();
        let full = py::strip(&format!("{prefix} {name}")).to_owned();
        command_of.insert(var.to_owned(), full.clone());
        out.push(full);
    }
    Ok(out)
}

/// `os.path.relpath(path, root)` for absolute paths.
fn relpath(path: &Path, root: &Path) -> String {
    let path: Vec<_> = path.components().collect();
    let base: Vec<_> = root.components().collect();
    let common = path.iter().zip(&base).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..base.len() {
        out.push("..");
    }
    for c in &path[common..] {
        out.push(c.as_os_str());
    }
    if out.as_os_str().is_empty() {
        ".".into()
    } else {
        out.to_string_lossy().into_owned()
    }
}

/// `(package, kind, name, src path)` for every bin/example target, via cargo metadata.
pub fn cargo_bins(root: &Path) -> Result<Vec<(String, String, String, String)>> {
    let output = Command::new("cargo")
        .args(["metadata", "--locked", "--no-deps", "--format-version", "1"])
        .current_dir(root)
        .output();
    let meta = match output {
        Ok(out) if out.status.success() => pyjson::parse_bytes(&out.stdout)
            .map_err(|m| Error::parse(Path::new("cargo metadata"), m))?,
        failed => {
            let why = match failed {
                Ok(out) => format!(
                    "Command '['cargo', 'metadata', '--locked', '--no-deps', '--format-version', '1']' returned non-zero exit status {}.",
                    out.status.code().unwrap_or(-1)
                ),
                Err(e) => e.to_string(),
            };
            eprintln!(
                "warning: cargo metadata unavailable ({why}); falling back to src/bin listings"
            );
            return fallback_bins(root);
        }
    };
    let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
    let mut out = vec![];
    for package in meta
        .get("packages")
        .and_then(Value::as_list)
        .into_iter()
        .flatten()
    {
        let name = package
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default();
        for target in package
            .get("targets")
            .and_then(Value::as_list)
            .into_iter()
            .flatten()
        {
            let kind = target
                .get("kind")
                .and_then(Value::as_list)
                .and_then(|k| k.first())
                .and_then(Value::as_str)
                .unwrap_or_default();
            if kind == "bin" || kind == "example" {
                let src = target
                    .get("src_path")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let target_name = target
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                out.push((
                    name.to_owned(),
                    kind.to_owned(),
                    target_name.to_owned(),
                    relpath(Path::new(src), &root),
                ));
            }
        }
    }
    out.sort();
    Ok(out)
}

fn walk_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() && !py::is_symlink(&path) {
            walk_files(&path, out);
        } else if path.is_file() {
            out.push(path);
        }
    }
}

fn rel(path: &Path, root: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

/// The src/bin listing used when cargo metadata is unavailable.
pub fn fallback_bins(root: &Path) -> Result<Vec<(String, String, String, String)>> {
    let package_name = Regex::new(r#"(?m)^\[package\]\s*\nname\s*=\s*"([^"]+)""#).expect("static");
    let bin_table = Regex::new(r#"\[\[bin\]\]\s*\nname\s*=\s*"([^"]+)"\s*\npath\s*=\s*"([^"]+)""#)
        .expect("static");
    let mut manifests = vec![];
    walk_files(&root.join("crates"), &mut manifests);
    manifests.retain(|p| p.file_name().is_some_and(|n| n == "Cargo.toml"));
    let mut out = BTreeSet::new();
    for cargo in manifests {
        let krate = cargo.parent().expect("manifest dir");
        let text = String::from_utf8_lossy(&py::read_bytes(&cargo)?).into_owned();
        let Some(m) = package_name.captures(&text) else {
            continue;
        };
        let pkg = m[1].to_owned();
        if krate.join("src/main.rs").exists() {
            out.insert((
                pkg.clone(),
                "bin".to_owned(),
                pkg.clone(),
                rel(&krate.join("src/main.rs"), root),
            ));
        }
        for (dir, kind) in [("src/bin", "bin"), ("examples", "example")] {
            let Ok(entries) = std::fs::read_dir(krate.join(dir)) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "rs") && path.is_file() {
                    let stem = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    out.insert((pkg.clone(), kind.to_owned(), stem, rel(&path, root)));
                }
            }
        }
        for m in bin_table.captures_iter(&text) {
            out.insert((
                pkg.clone(),
                "bin".to_owned(),
                m[1].to_owned(),
                rel(&krate.join(&m[2]), root),
            ));
        }
    }
    Ok(out.into_iter().collect())
}

/// Every `.py`/`.sh` under `python/src/jope/{tools,research}` except `__init__.py`.
pub fn python_tools(root: &Path) -> Vec<String> {
    let mut out = vec![];
    for sub in ["python/src/jope/tools", "python/src/jope/research"] {
        let mut files = vec![];
        walk_files(&root.join(sub), &mut files);
        for path in files {
            // `Path.suffix` and `Path.extension()` agree: a bare dotfile such as `.py` has none.
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            let ext = path.extension().map(|e| e.to_string_lossy().into_owned());
            if matches!(ext.as_deref(), Some("py" | "sh")) && name != "__init__.py" {
                out.push(rel(&path, root));
            }
        }
    }
    out
}

/// Every `.service`/`.timer` file under `deploy/systemd` (templates end in `.in` and are not).
pub fn systemd_units(root: &Path) -> Vec<String> {
    let mut files = vec![];
    walk_files(&root.join("deploy/systemd"), &mut files);
    let mut out: Vec<String> = files
        .iter()
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "service" || e == "timer")
        })
        .map(|p| rel(p, root))
        .collect();
    out.sort();
    out
}

/// A table line split into cells, `\|` read as a literal bar.
pub fn split_row(line: &str) -> Vec<String> {
    let mut parts = vec![];
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\\' && chars.peek() == Some(&'|') {
            current.push('|');
            chars.next();
        } else if ch == '|' {
            parts.push(std::mem::take(&mut current));
        } else {
            current.push(ch);
        }
    }
    parts.push(current);
    if parts.len() < 2 {
        return vec![];
    }
    parts[1..parts.len() - 1]
        .iter()
        .map(|p| py::strip(p).to_owned())
        .collect()
}

pub fn read_matrix(path: &Path) -> Result<Vec<Row>> {
    Ok(parse_matrix(&py::read_text(path)?))
}

/// The six-cell table rows of a matrix text.
pub fn parse_matrix(text: &str) -> Vec<Row> {
    let mut rows = vec![];
    for line in py::splitlines(text) {
        if !line.starts_with("| ") || line.starts_with("| kind ") || line.starts_with("|---") {
            continue;
        }
        let cells = split_row(line);
        if cells.len() != 6 {
            continue;
        }
        let mut cells = cells.into_iter();
        let mut next = || cells.next().expect("six cells");
        rows.push(Row {
            kind: next(),
            old: next(),
            new: next(),
            test: next(),
            status: next(),
            notes: next(),
        });
    }
    rows
}

fn pending_number(status: &str) -> Option<u64> {
    let digits = status.strip_prefix("pending-WP")?;
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| digits.parse().ok())
        .flatten()
}

pub fn status_class(status: &str) -> &'static str {
    match status {
        "moved" => "moved",
        "new" => "new",
        "lost" => "lost",
        s if s
            .strip_prefix("pending-WP")
            .is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())) =>
        {
            "pending"
        }
        s if s.starts_with("dropped:") => "dropped",
        _ => "invalid",
    }
}

/// The body of the `## Counts` table and the covering-test tally, recomputed from `rows`.
pub fn counts_lines(rows: &[Row]) -> (Vec<String>, usize) {
    let mut by_kind: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut statuses: BTreeMap<&str, usize> = BTreeMap::new();
    let mut no_test = 0;
    for r in rows {
        let class = status_class(&r.status);
        *by_kind
            .entry(&r.kind)
            .or_default()
            .entry(class)
            .or_default() += 1;
        *statuses.entry(class).or_default() += 1;
        if r.test == "none" && (class == "moved" || class == "new") {
            no_test += 1;
        }
    }
    let mut lines = vec![];
    for (kind, counts) in &by_kind {
        let total: usize = counts.values().sum();
        let cells: Vec<String> = COUNT_CLASSES
            .iter()
            .map(|c| counts.get(c).copied().unwrap_or(0).to_string())
            .collect();
        lines.push(format!("| {kind} | {total} | {} |", cells.join(" | ")));
    }
    let cells: Vec<String> = COUNT_CLASSES
        .iter()
        .map(|c| format!("**{}**", statuses.get(c).copied().unwrap_or(0)))
        .collect();
    lines.push(format!(
        "| **total** | **{}** | {} |",
        rows.len(),
        cells.join(" | ")
    ));
    (lines, no_test)
}

fn counts_row_kind(line: &str) -> String {
    split_row(line)
        .into_iter()
        .next()
        .unwrap_or_else(|| line.to_owned())
}

/// The `## Counts` section as committed: table lines, covering-test tally, every other line.
fn committed_counts(text: &str) -> (Vec<String>, Option<usize>, Vec<String>) {
    let lines = py::splitlines(text);
    let Some(start) = lines.iter().position(|l| *l == "## Counts") else {
        return (vec![], None, vec![]);
    };
    let tally =
        Regex::new(r"^`moved`/`new` rows without a covering test: (\d+)\.$").expect("static");
    let (mut table, mut prose, mut no_test) = (vec![], vec![], None);
    for line in &lines[start + 1..] {
        if line.starts_with("## ") {
            break;
        }
        let stripped = line.trim_end_matches(py::is_space);
        if stripped.starts_with("| ") && !stripped.starts_with("| kind ") {
            table.push(stripped.to_owned());
            continue;
        }
        if let Some(m) = tally.captures(stripped) {
            no_test = m[1].parse().ok();
            continue;
        }
        if !stripped.is_empty() {
            prose.push(stripped.to_owned());
        }
    }
    (table, no_test, prose)
}

fn head_of(new: &str) -> &str {
    py::strip(new.split(" (").next().unwrap_or(""))
}

/// What the check reads from the tree besides the matrix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tree {
    /// `jope-parallel` subcommands: the `Cmd` variants of `crates/jope-engine/src/main.rs`.
    pub clap: Vec<String>,
    /// `(package, kind, name, src path)` of every bin/example target.
    pub bins: Vec<(String, String, String, String)>,
    /// Python tools under `python/src/jope/{tools,research}`.
    pub tools: Vec<String>,
    /// `jope-py` commands, when `python/src/jope/cli.py` exists.
    pub jope_py: Option<Vec<String>>,
    /// systemd units under `deploy/systemd`.
    pub units: Vec<String>,
}

/// Read the tree's entry points (`root` canonical).
pub fn scan_tree(root: &Path) -> Result<Tree> {
    let main_rs = root.join("crates/jope-engine/src/main.rs");
    let clap = if main_rs.exists() {
        parse_clap_enum(&main_rs, "Cmd")?
            .into_iter()
            .map(|c| c.name)
            .collect()
    } else {
        vec![]
    };
    let bins = cargo_bins(root)?;
    let tools = python_tools(root);
    let cli = root.join("python/src/jope/cli.py");
    let jope_py = if cli.exists() {
        Some(parse_argparse_commands(&cli)?)
    } else {
        None
    };
    let units = systemd_units(root);
    Ok(Tree {
        clap,
        bins,
        tools,
        jope_py,
        units,
    })
}

/// Every row target that names a path (`evaluate` asks whether those exist).
pub fn target_heads(matrix_text: &str) -> BTreeSet<String> {
    parse_matrix(matrix_text)
        .iter()
        .map(|r| head_of(&r.new).to_owned())
        .filter(|h| h.contains('/'))
        .collect()
}

/// The check report and whether the matrix passes.
pub fn check(matrix: &Path, root: &Path, help_text: Option<&Path>) -> Result<(String, bool)> {
    let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
    let text = py::read_text(matrix)?;
    let help = help_text.map(py::read_text).transpose()?;
    let tree = scan_tree(&root)?;
    Ok(evaluate(&text, &tree, help.as_deref(), &|head: &str| {
        root.join(head).exists()
    }))
}

/// The check report for a matrix text against a tree, and whether the matrix passes. `exists`
/// answers whether a row target path exists. `check` gathers the inputs; the Python-parity
/// fixtures replay recorded ones.
pub fn evaluate(
    matrix_text: &str,
    tree: &Tree,
    help_text: Option<&str>,
    exists: &dyn Fn(&str) -> bool,
) -> (String, bool) {
    let rows = parse_matrix(matrix_text);
    let mut problems: Vec<String> = vec![];
    let mut targets: BTreeSet<String> = BTreeSet::new();
    let mut parens: BTreeSet<String> = BTreeSet::new();
    let paren = Regex::new(r"\(([^)]*)\)").expect("static pattern");
    let archive_location = ARCHIVE.split("/git/").next().unwrap_or(ARCHIVE);
    for (i, r) in rows.iter().enumerate() {
        let place = format!("row {} [{} | {}]", i + 1, r.kind, r.old);
        let class = status_class(&r.status);
        if class == "invalid" {
            problems.push(format!("{place}: invalid status `{}`", r.status));
            continue;
        }
        let head = head_of(&r.new);
        for m in paren.captures_iter(&r.new) {
            parens.insert(py::strip(&m[1]).to_owned());
        }
        if !head.is_empty() {
            targets.insert(head.to_owned());
        }
        match class {
            "pending" => {
                let n = pending_number(&r.status).unwrap_or(0);
                if !(4..=12).contains(&n) {
                    problems.push(format!("{place}: pending-WP{n} is outside WP4..WP12"));
                }
            }
            "dropped" => {
                if !r.status.contains(archive_location) {
                    problems.push(format!("{place}: dropped without the archive location"));
                }
                if r.status.chars().count() < "dropped: x (archive: )".len() + 20 {
                    problems.push(format!("{place}: dropped without a reason"));
                }
                if !matches!(r.new.as_str(), "dropped" | "-" | "") {
                    problems.push(format!(
                        "{place}: dropped row still names a target `{}`",
                        r.new
                    ));
                }
            }
            "lost" => {
                if r.new != "lost" {
                    problems.push(format!("{place}: lost row must have new home `lost`"));
                }
            }
            _ => {
                if head.is_empty() || matches!(head, "dropped" | "lost" | "pending" | "-" | "none")
                {
                    problems.push(format!("{place}: status `{}` but no target", r.status));
                } else if head.contains('/') && !exists(head) {
                    problems.push(format!("{place}: target `{head}` does not exist"));
                }
            }
        }
    }
    let mut missing: Vec<String> = vec![];
    for name in &tree.clap {
        if !parens.contains(&format!("jope-parallel {name}")) {
            missing.push(format!("clap subcommand: jope-parallel {name}"));
        }
    }
    if let Some(help) = help_text {
        for name in parse_help_commands(help) {
            if !parens.contains(&format!("jope-parallel {name}")) {
                missing.push(format!(
                    "clap subcommand (from --help): jope-parallel {name}"
                ));
            }
        }
    }
    for (pkg, kind, name, src) in &tree.bins {
        if !targets.contains(src) {
            missing.push(format!("{kind} {pkg}/{name}: {src}"));
        }
    }
    for tool in &tree.tools {
        if !targets.contains(tool) {
            missing.push(format!("python tool: {tool}"));
        }
    }
    for command in tree.jope_py.iter().flatten() {
        if !parens.contains(&format!("jope-py {command}")) {
            missing.push(format!("jope-py command: {command}"));
        }
    }
    for unit in &tree.units {
        if !targets.contains(unit) {
            missing.push(format!("systemd unit: {unit}"));
        }
    }
    for m in missing {
        problems.push(format!("no row for {m}"));
    }
    let (want_table, want_no_test) = counts_lines(&rows);
    let (got_table, got_no_test, got_prose) = committed_counts(matrix_text);
    for line in COUNTS_PREAMBLE {
        if !got_prose.iter().any(|p| p == line) {
            problems.push(format!(
                "## Counts is missing the line documenting why it is gated: `{line}` \
                 (restore it from COUNTS_PREAMBLE in crates/jope-records/src/parity.rs)"
            ));
        }
    }
    if got_table.is_empty() {
        problems.push("the matrix has no `## Counts` table".to_owned());
    } else if got_table != want_table {
        let wanted: BTreeMap<String, &String> =
            want_table.iter().map(|l| (counts_row_kind(l), l)).collect();
        let gotten: BTreeMap<String, &String> =
            got_table.iter().map(|l| (counts_row_kind(l), l)).collect();
        let kinds: BTreeSet<&String> = wanted.keys().chain(gotten.keys()).collect();
        for kind in kinds {
            if wanted.get(kind) != gotten.get(kind) {
                problems.push(format!(
                    "## Counts row `{kind}` disagrees with the rows below it: committed `{}`, recomputed `{}`",
                    gotten.get(kind).map_or("(no such row)", |l| l.as_str()),
                    wanted.get(kind).map_or("(no such kind)", |l| l.as_str()),
                ));
            }
        }
    }
    match got_no_test {
        None => problems
            .push("## Counts has no `moved`/`new` rows without a covering test: <n>. line".into()),
        Some(n) if n != want_no_test => problems.push(format!(
            "## Counts says {n} `moved`/`new` rows without a covering test; the rows say {want_no_test}"
        )),
        Some(_) => {}
    }
    let mut by_status: BTreeMap<&str, usize> = BTreeMap::new();
    let mut by_kind: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rows {
        *by_status.entry(status_class(&r.status)).or_default() += 1;
        *by_kind.entry(&r.kind).or_default() += 1;
    }
    let listing = |m: &BTreeMap<&str, usize>| {
        m.iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = format!("capability matrix: {} rows\n", rows.len());
    out.push_str(&format!("by status: {}\n", listing(&by_status)));
    out.push_str(&format!("by kind: {}\n", listing(&by_kind)));
    out.push_str(&format!(
        "this tree: {} binaries, {} examples, {} clap subcommands, {} python tools, {} systemd units\n",
        tree.bins.iter().filter(|b| b.1 == "bin").count(),
        tree.bins.iter().filter(|b| b.1 == "example").count(),
        tree.clap.len(),
        tree.tools.len(),
        tree.units.len()
    ));
    if !problems.is_empty() {
        out.push_str(&format!("\n{} problem(s):\n", problems.len()));
        for p in &problems {
            out.push_str(&format!("  - {p}\n"));
        }
        return (out, false);
    }
    out.push_str("ok: every row resolved and every entry point of this tree has a row\n");
    (out, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(kind: &str, new: &str, test: &str, status: &str) -> Row {
        Row {
            kind: kind.into(),
            old: "old".into(),
            new: new.into(),
            test: test.into(),
            status: status.into(),
            notes: String::new(),
        }
    }

    #[test]
    fn kebab_matches_clap_naming() {
        assert_eq!(kebab("CompareSeeds"), "compare-seeds");
        assert_eq!(kebab("GPUWorker"), "gpu-worker");
        assert_eq!(kebab("Run2Go"), "run2-go");
        assert_eq!(kebab("Enqueue"), "enqueue");
    }

    #[test]
    fn rows_read_escaped_bars() {
        assert_eq!(split_row("| a | b \\| c | d |"), ["a", "b | c", "d"]);
        assert!(split_row("no bars").is_empty());
    }

    #[test]
    fn status_classes_and_pending_range() {
        assert_eq!(status_class("moved"), "moved");
        assert_eq!(status_class("pending-WP5"), "pending");
        assert_eq!(status_class("pending-WP"), "invalid");
        assert_eq!(status_class("dropped: why (archive: x)"), "dropped");
        assert_eq!(status_class("gone"), "invalid");
        assert_eq!(pending_number("pending-WP13"), Some(13));
    }

    /// The tally is recomputed from the rows: a moved row without a test counts once, a dropped
    /// row never.
    #[test]
    fn counts_are_recomputed_from_rows() {
        let rows = [
            row("tool-py", "a/b.py", "none", "moved"),
            row(
                "tool-py",
                "dropped",
                "none",
                "dropped: archived (archive: s3://x)",
            ),
            row("new-bin", "c/main.rs (c)", "c::bin (1 tests)", "new"),
        ];
        let (lines, no_test) = counts_lines(&rows);
        assert_eq!(
            lines,
            [
                "| new-bin | 1 | 0 | 1 | 0 | 0 | 0 |",
                "| tool-py | 2 | 1 | 0 | 0 | 1 | 0 |",
                "| **total** | **3** | **1** | **1** | **0** | **1** | **0** |",
            ]
        );
        assert_eq!(no_test, 1);
    }
}
