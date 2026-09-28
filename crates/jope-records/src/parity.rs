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

/// Top-level variants of `enum <enum_name> { ... }` with their outer attributes. The source is read
/// as `read_text(errors="replace")` read it (universal newlines, so a CRLF file parses), and the
/// patterns use Python's classes ([`py::regex`]).
pub fn parse_clap_enum(path: &Path, enum_name: &str) -> Result<Vec<Variant>> {
    let src = py::read_text_replace(path)?;
    let block = py::regex(&format!(
        r"(?s)enum {} \{{\n(.*?)\n\}}\n",
        regex::escape(enum_name)
    ));
    let body = block
        .captures(&src)
        .ok_or_else(|| Error::Shape(format!("{}: enum {enum_name} not found", path.display())))?[1]
        .to_owned();
    let attr = py::regex(r"^    #\[");
    let variant = py::regex(r"^    ([A-Z][A-Za-z0-9]*)\s*(\{|,|\()");
    let named = py::regex(r#"name\s*=\s*"([^"]+)""#);
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
    let name = py::regex(r"^  ([a-z0-9][a-z0-9-]*)(\s|$)");
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
    Ok(argparse_commands(&py::read_text_replace(path)?))
}

/// [`parse_argparse_commands`] for the source text.
pub fn argparse_commands(src: &str) -> Vec<String> {
    let subparsers = py::regex(r"(\w+)\s*=\s*(\w+)\.add_subparsers\(");
    let parsers = py::regex(r#"(\w+)\s*=\s*(\w+)\.add_parser\(\s*"([^"]+)""#);
    let owner: BTreeMap<String, String> = subparsers
        .captures_iter(src)
        .map(|m| (m[1].to_owned(), m[2].to_owned()))
        .collect();
    let mut command_of: BTreeMap<String, String> = BTreeMap::new();
    let mut out = vec![];
    for m in parsers.captures_iter(src) {
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
    out
}

/// `os.path.relpath(path, root)` for absolute paths. It splits both on separators and drops the
/// empty parts, so unlike pathlib it has no distinct `//` root: `//r/x` relative to `/r` is `x`.
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
///
/// As `subprocess.run(..., check=True, capture_output=True, text=True)` ran it: both streams are
/// decoded (strict UTF-8, universal newlines) before the exit status is looked at, and a stream
/// that does not decode refuses (the Python raised `UnicodeDecodeError`, which its fallback did
/// not catch). A Cargo that cannot be started (`OSError`) or exits non-zero
/// (`CalledProcessError`) falls back to the `src/bin` listings with the Python's warning. The
/// stdout is then read as a `str`, so a UTF-8 BOM refuses.
pub fn cargo_bins(root: &Path) -> Result<Vec<(String, String, String, String)>> {
    const COMMAND: [&str; 5] = ["metadata", "--locked", "--no-deps", "--format-version", "1"];
    let output = Command::new("cargo")
        .args(COMMAND)
        .current_dir(root)
        .output();
    let why = match output {
        Err(e) => py::os_error(&e, "cargo"),
        Ok(out) => {
            let (stdout, _) = py::text_streams(&out)
                .map_err(|m| Error::Command(format!("cargo metadata: {m}")))?;
            if out.status.success() {
                let meta = pyjson::parse_str(&stdout)
                    .map_err(|m| Error::parse(Path::new("cargo metadata"), m))?;
                let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
                return bins_from_metadata(&meta, &root);
            }
            let argv: Vec<&str> = std::iter::once("cargo").chain(COMMAND).collect();
            py::called_process_error(&py::str_list_repr(&argv), py::returncode(out.status))
        }
    };
    eprintln!("warning: cargo metadata unavailable ({why}); falling back to src/bin listings");
    fallback_bins(root)
}

/// `os.path.abspath`: absolute (against the working directory) and lexically normalised. A
/// relative path when the working directory cannot be read (it was deleted) refuses, as
/// `os.getcwd()` raised, instead of being resolved against `/`.
fn abspath(text: &str) -> std::io::Result<PathBuf> {
    let path = Path::new(text);
    let joined = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut out = PathBuf::from("/");
    for component in joined.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::Normal(part) => out.push(part),
            _ => {}
        }
    }
    Ok(out)
}

/// `(package, kind, name, src path)` for every bin/example target of a `cargo metadata` document
/// (`root` canonical), as the Python read it: `pkg["targets"]`, `t["kind"][0]`, and for a bin or
/// example `t["src_path"]`, `pkg["name"]` and `t["name"]`. Where the Python crashed (a missing
/// key, an empty `kind`, a value it could not iterate or index, a falsy or non-string
/// `src_path`) this refuses too, naming the field and the offending type (see
/// [`py::exception`]; the text is not CPython's), instead of skipping the package or target. A
/// package or target name that is not a string is refused as well, which is stricter than the
/// Python: it sorted such a name with the string ones (crashing only when two names of different
/// types met) and printed it.
pub fn bins_from_metadata(
    meta: &Value,
    root: &Path,
) -> Result<Vec<(String, String, String, String)>> {
    let raised = |context: &str, exception: String| Error::Shape(format!("{context}: {exception}"));
    let text = |value: &Value, context: &str| -> Result<String> {
        value.as_str().map(str::to_owned).ok_or_else(|| {
            Error::Shape(format!(
                "{context}: expected a string, found {}",
                value.repr()
            ))
        })
    };
    let packages = py::subscript(meta, "packages").map_err(|e| raised("cargo metadata", e))?;
    let packages = py::iterate(packages).ok_or_else(|| {
        raised(
            "cargo metadata packages",
            py::exception::not_iterable(packages),
        )
    })?;
    let mut out = vec![];
    for (p, package) in packages.iter().enumerate() {
        let at = format!("cargo metadata packages[{p}]");
        let targets = py::subscript(package, "targets").map_err(|e| raised(&at, e))?;
        let targets = py::iterate(targets).ok_or_else(|| {
            raised(
                &format!("{at} targets"),
                py::exception::not_iterable(targets),
            )
        })?;
        for (t, target) in targets.iter().enumerate() {
            let at = format!("{at} targets[{t}]");
            let kinds = py::subscript(target, "kind").map_err(|e| raised(&at, e))?;
            let kind = py::first_item(kinds).map_err(|e| raised(&format!("{at} kind"), e))?;
            let kind = match kind.as_str() {
                Some(kind @ ("bin" | "example")) => kind.to_owned(),
                _ => continue,
            };
            // `os.path.relpath(t["src_path"], root)` refuses a falsy value of any type (`None`,
            // `""`, `0`, `[]`, ...) before it looks at the type, then anything but a string.
            let src = py::subscript(target, "src_path").map_err(|e| raised(&at, e))?;
            if !src.truthy() {
                return Err(raised(
                    &format!("{at} src_path"),
                    format!(
                        "ValueError: no path specified ({} {})",
                        py::type_name(src),
                        src.repr()
                    ),
                ));
            }
            let src = src
                .as_str()
                .ok_or_else(|| raised(&format!("{at} src_path"), py::exception::fspath(src)))?;
            // `os.path.relpath` runs before the names are read.
            let absolute = abspath(src)
                .map_err(|e| raised(&format!("{at} src_path"), py::os_error_bare(&e)))?;
            let name = py::subscript(package, "name").map_err(|e| raised(&at, e))?;
            let target_name = py::subscript(target, "name").map_err(|e| raised(&at, e))?;
            out.push((
                text(name, &format!("{at} package name"))?,
                kind,
                text(target_name, &format!("{at} name"))?,
                relpath(&absolute, root),
            ));
        }
    }
    out.sort();
    Ok(out)
}

/// An `OSError` the Python raised inspecting `path`, as a refusal naming what was being read.
fn os_refusal(what: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::Shape(format!(
        "{what}: {}",
        py::os_error(error, &path.to_string_lossy())
    ))
}

/// `path.exists()` ([`py::path_exists`]): a missing path (or a loop) is absent, and any other
/// error, such as a directory that may not be searched, refuses as the Python raised it.
fn exists(path: &Path, what: &str) -> Result<bool> {
    py::path_exists(path).map_err(|e| os_refusal(what, path, &e))
}

/// `path.is_dir()` ([`py::path_is_dir`]), refusing where the Python raised.
fn is_dir(path: &Path, what: &str) -> Result<bool> {
    py::path_is_dir(path).map_err(|e| os_refusal(what, path, &e))
}

/// `path.is_file()` ([`py::path_is_file`]), refusing where the Python raised.
fn is_file(path: &Path, what: &str) -> Result<bool> {
    py::path_is_file(path).map_err(|e| os_refusal(what, path, &e))
}

/// The entries of `dir` (any kind: files, directories, links, dangling or not), sorted. Where
/// pathlib's globs skipped a directory they could not list, this refuses (stricter), except a
/// directory that is gone (`ENOENT`).
fn dir_entries(dir: &Path, what: &str) -> Result<Vec<(PathBuf, std::fs::FileType)>> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(os_refusal(what, dir, &e)),
    };
    let mut out = vec![];
    for entry in entries {
        let entry = entry.map_err(|e| os_refusal(what, dir, &e))?;
        // The entry's own type (`d_type`, else `lstat`), as `Path.walk(follow_symlinks=False)` read
        // it to decide what to descend into; nothing is followed through a link here.
        let kind = entry
            .file_type()
            .map_err(|e| os_refusal(what, &entry.path(), &e))?;
        out.push((entry.path(), kind));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// `dir.rglob("*")` (Python 3.12), sorted as pathlib sorts paths (part by part): every entry below
/// `dir` of any kind, descending into real directories only (a link to a directory is listed, not
/// entered, as `Path.walk(follow_symlinks=False)` walked). A directory that may be listed but not
/// searched is listed like any other; what the caller then asks of its entries (`is_file`) refuses
/// as the Python raised. A directory that cannot be listed refuses (stricter: pathlib skipped it).
fn entries_below(dir: &Path, what: &str) -> Result<Vec<PathBuf>> {
    fn walk(dir: &Path, what: &str, out: &mut Vec<PathBuf>) -> Result<()> {
        for (path, kind) in dir_entries(dir, what)? {
            out.push(path.clone());
            if kind.is_dir() {
                walk(&path, what, out)?;
            }
        }
        Ok(())
    }
    let mut out = vec![];
    walk(dir, what, &mut out)?;
    out.sort();
    Ok(out)
}

/// `Path.name` of a listed entry matches `*<suffix>` (pathlib's `*` matches a leading dot and any
/// bytes, so a name that is not UTF-8 matches too).
fn name_ends_with(path: &Path, suffix: &str) -> bool {
    use std::os::unix::ffi::OsStrExt;
    path.file_name()
        .is_some_and(|n| n.as_bytes().ends_with(suffix.as_bytes()))
}

/// `str(path.relative_to(root))` for an entry point the check lists (and must find a row for).
///
/// pathlib compares the two as spelled after parsing: `.` and repeated separators drop out of the
/// relative path (a `[[bin]]` `path = "./src/x.rs"` is `crates/<c>/src/x.rs`), and a path that is
/// not under the root (an absolute `[[bin]]` path, either root) refuses, as the Python crashed
/// with `ValueError` there.
///
/// A name that is not UTF-8 refuses: Python kept it surrogate-escaped, so no row (read as UTF-8
/// text) could ever name it, and it failed the check reporting the entry point as missing (its
/// raw bytes printed under a `C.UTF-8` locale, a `UnicodeEncodeError` under other UTF-8 locales).
/// This report is text, so the port refuses instead (exit status 1 either way). Two such names
/// never collapse into one, as a lossy conversion made them.
fn rel(path: &Path, root: &Path, what: &str) -> Result<String> {
    let Some(text) = path.to_str() else {
        let relative = path.strip_prefix(root).unwrap_or(path);
        return Err(Error::Shape(format!(
            "{what} {}: the path is not UTF-8, so no row can name it (the Python failed the check reporting it as missing)",
            relative.to_string_lossy()
        )));
    };
    py::PosixPath::parse(text)
        .relative_to(&py::PosixPath::parse(&root.to_string_lossy()))
        .map(|relative| relative.to_string())
        .map_err(|e| Error::Shape(format!("{what}: ValueError: {e}")))
}

/// The src/bin listing used when cargo metadata is unavailable, read as the Python's globs read
/// the tree (Python 3.12 pathlib):
///
/// - `root.glob("crates/**/Cargo.toml")`: every entry named `Cargo.toml`, of any kind, in `crates`
///   and every real directory below it (links to directories are not entered), each then read as
///   text, so one that is a directory or a dangling link refuses as `read_text` raised;
/// - `(crate / "src" / "main.rs").exists()`, which raised on an error other than a missing path;
/// - `(crate / "src" / "bin").glob("*.rs")` and `examples`: nothing unless the directory
///   `is_dir()` (which raised the same way), then every entry whose name ends in `.rs`, of any
///   kind (a directory or a dangling link named `x.rs` is listed, and so is a bare `.rs`, its stem
///   `.rs`), without inspecting it.
///
/// A directory the globs could not list was skipped by pathlib; here it refuses (stricter).
pub fn fallback_bins(root: &Path) -> Result<Vec<(String, String, String, String)>> {
    let package_name = py::regex(r#"(?m)^\[package\]\s*\nname\s*=\s*"([^"]+)""#);
    let bin_table = py::regex(r#"\[\[bin\]\]\s*\nname\s*=\s*"([^"]+)"\s*\npath\s*=\s*"([^"]+)""#);
    let crates = root.join("crates");
    let mut manifests = vec![];
    if is_dir(&crates, "crates")? {
        manifests = entries_below(&crates, "crates")?;
        manifests.retain(|p| p.file_name().is_some_and(|n| n == "Cargo.toml"));
    }
    let mut out = BTreeSet::new();
    for cargo in manifests {
        let krate = cargo.parent().expect("manifest dir");
        let text = py::read_text_replace(&cargo)?;
        let Some(m) = package_name.captures(&text) else {
            continue;
        };
        let pkg = m[1].to_owned();
        if exists(&krate.join("src/main.rs"), "bin")? {
            out.insert((
                pkg.clone(),
                "bin".to_owned(),
                pkg.clone(),
                rel(&krate.join("src/main.rs"), root, "bin")?,
            ));
        }
        for (dir, kind) in [("src/bin", "bin"), ("examples", "example")] {
            let dir = krate.join(dir);
            if !is_dir(&dir, kind)? {
                continue;
            }
            for (path, _) in dir_entries(&dir, kind)? {
                if name_ends_with(&path, ".rs") {
                    let source = rel(&path, root, kind)?;
                    // `Path.stem` and `Path::file_stem` agree on every name ending in `.rs`
                    // (`x.rs` is `x`, `.rs` is `.rs`, `..rs` is `.`); a name that is not UTF-8 was
                    // refused by `rel` above.
                    let stem = path
                        .file_stem()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .into_owned();
                    out.insert((pkg.clone(), kind.to_owned(), stem, source));
                }
            }
        }
        for m in bin_table.captures_iter(&text) {
            out.insert((
                pkg.clone(),
                "bin".to_owned(),
                m[1].to_owned(),
                rel(&krate.join(&m[2]), root, "bin")?,
            ));
        }
    }
    Ok(out.into_iter().collect())
}

/// Every `.py`/`.sh` under `python/src/jope/{tools,research}` except `__init__.py`: for each
/// directory that `is_dir()`, every entry of `rglob("*")` that `is_file()` (both refuse where the
/// Python raised: a directory that may be listed but not searched holds entries that cannot be
/// inspected), asked of every entry before its name is looked at, as the Python asked it.
pub fn python_tools(root: &Path) -> Result<Vec<String>> {
    let mut out = vec![];
    for sub in ["python/src/jope/tools", "python/src/jope/research"] {
        let dir = root.join(sub);
        if !is_dir(&dir, "python tool")? {
            continue;
        }
        for path in entries_below(&dir, "python tool")? {
            if !is_file(&path, "python tool")? {
                continue;
            }
            // `Path.suffix` and `Path.extension()` agree: a bare dotfile such as `.py` has none.
            let listed = path.extension().is_some_and(|e| e == "py" || e == "sh")
                && path.file_name().is_some_and(|n| n != "__init__.py");
            if listed {
                out.push(rel(&path, root, "python tool")?);
            }
        }
    }
    Ok(out)
}

/// Every `.service`/`.timer` file under `deploy/systemd` (templates end in `.in` and are not):
/// the entries of `rglob("*")` that `is_file()`, read as [`python_tools`] reads its directories.
pub fn systemd_units(root: &Path) -> Result<Vec<String>> {
    let dir = root.join("deploy/systemd");
    if !is_dir(&dir, "systemd unit")? {
        return Ok(vec![]);
    }
    let mut out = vec![];
    for path in entries_below(&dir, "systemd unit")? {
        if is_file(&path, "systemd unit")?
            && path
                .extension()
                .is_some_and(|e| e == "service" || e == "timer")
        {
            out.push(rel(&path, root, "systemd unit")?);
        }
    }
    out.sort();
    Ok(out)
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

/// `PENDING_RE.match(status)` (`^pending-WP(\d+)$`, Python's Unicode `\d`): classifying a status
/// reads no number and never refused.
fn is_pending(status: &str) -> bool {
    status
        .strip_prefix("pending-WP")
        .is_some_and(py::is_decimal_run)
}

/// `int(PENDING_RE.match(status).group(1))` of a pending status through the strict number grammar
/// ([`py::decimal_int`]): the work-package number as ASCII decimal, or the refusal. The check
/// crashed where Python 3.12 refused more than 4300 digits; a number written in other Unicode
/// digits, which Python read, is refused as well (stricter).
fn pending_int(status: &str, place: &str) -> Result<String> {
    let digits = status.strip_prefix("pending-WP").expect("a pending status");
    py::decimal_int(digits)
        .map_err(|e| Error::Shape(format!("{place}: pending-WP<n>: ValueError: {e}")))
}

pub fn status_class(status: &str) -> &'static str {
    match status {
        "moved" => "moved",
        "new" => "new",
        "lost" => "lost",
        s if is_pending(s) => "pending",
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

/// The `## Counts` section as committed: table lines, covering-test tally (`int()` of what `\d+`
/// captured through the strict number grammar, [`py::decimal_int`]: ASCII decimal, more than 4300
/// digits refused as Python 3.12's `int()` raised, other Unicode digits refused though Python read
/// them), every other line.
fn committed_counts(text: &str) -> Result<(Vec<String>, Option<String>, Vec<String>)> {
    let lines = py::splitlines(text);
    let Some(start) = lines.iter().position(|l| *l == "## Counts") else {
        return Ok((vec![], None, vec![]));
    };
    let tally = py::regex(r"^`moved`/`new` rows without a covering test: (\d+)\.$");
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
            no_test = Some(py::decimal_int(&m[1]).map_err(|e| {
                Error::Shape(format!(
                    "## Counts `moved`/`new` rows without a covering test: ValueError: {e}"
                ))
            })?);
            continue;
        }
        if !stripped.is_empty() {
            prose.push(stripped.to_owned());
        }
    }
    Ok((table, no_test, prose))
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
    let clap = clap_names(root)?;
    scan_rest(root, clap)
}

/// The `jope-parallel` subcommands of `crates/jope-engine/src/main.rs`, when it exists
/// (`main_rs.exists()`, which raised on an error other than a missing path: that refuses).
fn clap_names(root: &Path) -> Result<Vec<String>> {
    let main_rs = root.join("crates/jope-engine/src/main.rs");
    Ok(if exists(&main_rs, "clap subcommands")? {
        parse_clap_enum(&main_rs, "Cmd")?
            .into_iter()
            .map(|c| c.name)
            .collect()
    } else {
        vec![]
    })
}

/// The entry points read after the clap subcommands (and, in `check`, after the `--help` text):
/// binaries and examples, Python tools, `jope-py` commands, systemd units, in the Python's order.
fn scan_rest(root: &Path, clap: Vec<String>) -> Result<Tree> {
    let bins = cargo_bins(root)?;
    let tools = python_tools(root)?;
    let cli = root.join("python/src/jope/cli.py");
    let jope_py = if exists(&cli, "jope-py commands")? {
        Some(parse_argparse_commands(&cli)?)
    } else {
        None
    };
    let units = systemd_units(root)?;
    Ok(Tree {
        clap,
        bins,
        tools,
        jope_py,
        units,
    })
}

/// The row targets whose existence `evaluate` asks about, in the order the Python's row loop
/// asked (`moved`/`new` rows whose head names a path).
pub fn checked_heads(matrix_text: &str) -> Vec<String> {
    let mut heads = vec![];
    for r in parse_matrix(matrix_text) {
        if !matches!(status_class(&r.status), "moved" | "new") {
            continue;
        }
        let head = head_of(&r.new);
        if !head.is_empty()
            && !matches!(head, "dropped" | "lost" | "pending" | "-" | "none")
            && head.contains('/')
        {
            heads.push(head.to_owned());
        }
    }
    heads
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
///
/// The inputs are read in `cmd_check`'s order, so the first failure is the one the Python
/// stopped at: the matrix, the row targets' existence (`Path.exists`, which raised on an error
/// other than a missing path), the clap enum, the `--help` text, then the rest of the tree.
pub fn check(matrix: &Path, root: &Path, help_text: Option<&Path>) -> Result<(String, bool)> {
    let root = root.canonicalize().map_err(|e| Error::io(root, e))?;
    let text = py::read_text(matrix)?;
    let mut exists = BTreeMap::new();
    for head in checked_heads(&text) {
        let present = py::path_exists(&root.join(&head)).map_err(|e| {
            Error::Shape(format!(
                "row target `{head}`: {}",
                py::os_error(&e, &root.join(&head).to_string_lossy())
            ))
        })?;
        exists.insert(head, present);
    }
    let clap = clap_names(&root)?;
    let help = help_text.map(py::read_text).transpose()?;
    let tree = scan_rest(&root, clap)?;
    evaluate(&text, &tree, help.as_deref(), &|head: &str| exists[head])
}

/// The check report for a matrix text against a tree, and whether the matrix passes. `exists`
/// answers whether a row target path exists. `check` gathers the inputs; the Python-parity
/// fixtures replay recorded ones. Refuses where the Python's `int()` raised (a `pending-WP<n>`
/// status or a covering-test tally of more than 4300 digits) and where the strict number grammar
/// does not read what it read (such a number in other Unicode digits).
pub fn evaluate(
    matrix_text: &str,
    tree: &Tree,
    help_text: Option<&str>,
    exists: &dyn Fn(&str) -> bool,
) -> Result<(String, bool)> {
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
                let n = pending_int(&r.status, &place)?;
                if !n.parse::<u64>().is_ok_and(|n| (4..=12).contains(&n)) {
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
    let (got_table, got_no_test, got_prose) = committed_counts(matrix_text)?;
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
        Some(n) if n != want_no_test.to_string() => problems.push(format!(
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
        return Ok((out, false));
    }
    out.push_str("ok: every row resolved and every entry point of this tree has a row\n");
    Ok((out, true))
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
        assert_eq!(pending_int("pending-WP13", "row 1").unwrap(), "13");
        assert_eq!(
            pending_int("pending-WP0099999999999999999999", "row 1").unwrap(),
            "99999999999999999999"
        );
        // Python's `\d` classifies any Unicode decimal digits as pending, as before; reading the
        // number is the strict grammar's, which refuses them (Python read 5; stricter).
        assert_eq!(status_class("pending-WP\u{665}"), "pending");
        let refused = pending_int("pending-WP\u{665}", "row 1")
            .unwrap_err()
            .to_string();
        assert!(refused.contains("is not ASCII decimal digits"), "{refused}");
        assert_eq!(status_class("pending-WP\u{b2}"), "invalid");
        let refused = committed_counts(
            "## Counts\n\n`moved`/`new` rows without a covering test: \u{ff11}.\n",
        )
        .unwrap_err()
        .to_string();
        assert!(refused.contains("is not ASCII decimal digits"), "{refused}");
    }

    /// `int()` of the pending number and of the covering-test tally: 4300 digits (leading zeros
    /// count) are read and 4301 refuse, as Python 3.12 refused them; classifying a status converts
    /// nothing.
    #[test]
    fn integers_past_4300_digits_refuse_where_the_check_converts_them() {
        let wp = |n: usize| format!("pending-WP{}5", "0".repeat(n - 1));
        assert_eq!(status_class(&wp(4301)), "pending");
        assert_eq!(pending_int(&wp(4300), "row 1").unwrap(), "5");
        let refused = pending_int(&wp(4301), "row 1").unwrap_err().to_string();
        assert!(refused.contains("value has 4301 digits"), "{refused}");
        let counts = |zeros: usize| {
            format!(
                "## Counts\n\n`moved`/`new` rows without a covering test: {}1.\n",
                "0".repeat(zeros)
            )
        };
        assert_eq!(
            committed_counts(&counts(4299)).unwrap().1,
            Some("1".to_owned())
        );
        let refused = committed_counts(&counts(4300)).unwrap_err().to_string();
        assert!(refused.contains("value has 4301 digits"), "{refused}");
    }

    /// `str(p.relative_to(root))` of a listed entry point normalises as pathlib does and refuses a
    /// path outside the root, POSIX's distinct `//` root included; `os.path.relpath` of a Cargo
    /// `src_path` has no such root.
    #[test]
    fn listed_paths_are_relative_to_the_root_as_python_compared_them() {
        let root = Path::new("/r");
        assert_eq!(
            rel(Path::new("/r/crates/c/./src//x.rs"), root, "bin").unwrap(),
            "crates/c/src/x.rs"
        );
        let refused = rel(Path::new("/elsewhere/x.rs"), root, "bin")
            .unwrap_err()
            .to_string();
        assert!(
            refused.contains("ValueError: '/elsewhere/x.rs' is not in the subpath of '/r'"),
            "{refused}"
        );
        assert!(rel(Path::new("//r/x.rs"), root, "bin").is_err());
        assert_eq!(
            relpath(&abspath("//r/src/main.rs").unwrap(), root),
            "src/main.rs"
        );
        assert_eq!(
            relpath(&abspath("/r/a/../src/main.rs").unwrap(), root),
            "src/main.rs"
        );
    }

    /// The tree walks list what pathlib's globs listed and refuse where its predicates raised:
    /// `rglob("*")` lists entries of every kind and enters real directories only; `glob("*.rs")`
    /// lists a directory, a dangling link and a bare `.rs` by name; `is_file()` of an entry in a
    /// directory that may be listed but not searched refuses; a directory that cannot be listed
    /// refuses (stricter than pathlib, which skipped it).
    #[test]
    fn walks_list_as_pathlib_listed_and_refuse_where_it_raised() {
        use std::os::unix::fs::PermissionsExt;
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        let tools = root.join("python/src/jope/tools");
        std::fs::create_dir_all(tools.join("sub")).unwrap();
        std::fs::create_dir_all(root.join("elsewhere")).unwrap();
        std::fs::write(tools.join("x.py"), "").unwrap();
        std::fs::write(tools.join("sub/y.sh"), "").unwrap();
        std::fs::write(root.join("elsewhere/z.py"), "").unwrap();
        std::os::unix::fs::symlink("nowhere", tools.join("dangling.py")).unwrap();
        std::os::unix::fs::symlink(root.join("elsewhere"), tools.join("dirlink")).unwrap();
        assert_eq!(
            python_tools(&root).unwrap(),
            [
                "python/src/jope/tools/sub/y.sh",
                "python/src/jope/tools/x.py"
            ]
        );
        let krate = root.join("crates/c");
        std::fs::create_dir_all(krate.join("src/bin/dir.rs")).unwrap();
        std::fs::write(krate.join("Cargo.toml"), "[package]\nname = \"c\"\n").unwrap();
        std::fs::write(krate.join("src/bin/.rs"), "").unwrap();
        std::fs::write(krate.join("src/bin/notes.txt"), "").unwrap();
        std::os::unix::fs::symlink("nowhere", krate.join("src/bin/gone.rs")).unwrap();
        let names: Vec<(String, String)> = fallback_bins(&root)
            .unwrap()
            .into_iter()
            .map(|(_, _, name, source)| (name, source))
            .collect();
        assert_eq!(
            names,
            [
                (".rs".to_owned(), "crates/c/src/bin/.rs".to_owned()),
                ("dir".to_owned(), "crates/c/src/bin/dir.rs".to_owned()),
                ("gone".to_owned(), "crates/c/src/bin/gone.rs".to_owned()),
            ]
        );
        // A manifest that is not a readable file refuses, as `read_text` raised.
        std::os::unix::fs::symlink("nowhere", root.join("crates/Cargo.toml")).unwrap();
        assert!(fallback_bins(&root).is_err());
        std::fs::remove_file(root.join("crates/Cargo.toml")).unwrap();
        let mode = |path: &Path, mode: u32| {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap()
        };
        // Refused only where this user is refused (a privileged user may search anything).
        mode(&tools.join("sub"), 0o644);
        if std::fs::metadata(tools.join("sub/y.sh")).is_err() {
            let refused = python_tools(&root).unwrap_err().to_string();
            assert!(
                refused.contains("[Errno 13] Permission denied"),
                "{refused}"
            );
        }
        mode(&tools.join("sub"), 0o000);
        if std::fs::read_dir(tools.join("sub")).is_err() {
            assert!(python_tools(&root).is_err());
        }
        mode(&tools.join("sub"), 0o755);
        mode(&krate.join("src"), 0o644);
        if std::fs::metadata(krate.join("src/main.rs")).is_err() {
            let refused = fallback_bins(&root).unwrap_err().to_string();
            assert!(
                refused.contains("[Errno 13] Permission denied"),
                "{refused}"
            );
        }
        mode(&krate.join("src"), 0o755);
    }

    /// Python's `\s` and `\w` in the clap, `--help` and argparse readers.
    #[test]
    fn readers_use_python_classes() {
        assert_eq!(
            parse_help_commands("Commands:\n  foo\u{1f}bar  x\n  baz\u{301} y\n"),
            ["foo"]
        );
        assert_eq!(
            argparse_commands(
                "subs = p.add_subparsers()\nt\u{b2} = subs.add_parser(\"train\")\nx\u{2040} = subs.add_parser(\"no\")\nv = subs.add_parser(\u{1f}\"sep\")\n"
            ),
            ["train", "sep"]
        );
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
