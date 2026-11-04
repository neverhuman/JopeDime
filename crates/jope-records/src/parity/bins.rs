use super::*;

/// `os.path.relpath(path, root)` for absolute paths. It splits both on separators and drops the
/// empty parts, so unlike pathlib it has no distinct `//` root: `//r/x` relative to `/r` is `x`.
/// The components compare as the bytes they are. A result that is not UTF-8 (a relative `src_path`
/// under a working directory named with such a byte) is returned as the bytes it is, for the
/// caller to refuse: the Python's surrogate-escaped `str` matched no row.
pub(super) fn relpath(path: &Path, root: &Path) -> std::result::Result<String, std::ffi::OsString> {
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
        Ok(".".into())
    } else {
        out.into_os_string().into_string()
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
pub(super) fn abspath(text: &str) -> std::io::Result<PathBuf> {
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
            let relative = relpath(&absolute, root).map_err(|name| {
                Error::Shape(format!(
                    "{at} src_path: {} relative to the root is not UTF-8, so no row can name it (the Python failed the check reporting it as missing)",
                    py::os_repr(&name)
                ))
            })?;
            let name = py::subscript(package, "name").map_err(|e| raised(&at, e))?;
            let target_name = py::subscript(target, "name").map_err(|e| raised(&at, e))?;
            out.push((
                text(name, &format!("{at} package name"))?,
                kind,
                text(target_name, &format!("{at} name"))?,
                relative,
            ));
        }
    }
    out.sort();
    Ok(out)
}

/// An `OSError` the Python raised inspecting `path`, as a refusal naming what was being read.
fn os_refusal(what: &str, path: &Path, error: &std::io::Error) -> Error {
    Error::Shape(format!("{what}: {}", py::os_error_path(error, path)))
}

/// `path.exists()` ([`py::path_exists`]): a missing path (or a loop) is absent, and any other
/// error, such as a directory that may not be searched, refuses as the Python raised it.
pub(super) fn exists(path: &Path, what: &str) -> Result<bool> {
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
///
/// A root that is not UTF-8 refuses too (review round six; `check` refuses it first). It was
/// compared lossily (U+FFFD for the byte), so a `[[bin]]` path spelled with U+FFFD in its place was
/// found under the root where the Python's `relative_to` raised.
pub(super) fn rel(path: &Path, root: &Path, what: &str) -> Result<String> {
    let Some(root_text) = root.to_str() else {
        return Err(Error::Shape(format!(
            "{what}: the root {} is not UTF-8, so no entry point under it can be named as text",
            py::os_repr(root.as_os_str())
        )));
    };
    let Some(text) = path.to_str() else {
        let relative = path.strip_prefix(root).unwrap_or(path);
        return Err(Error::Shape(format!(
            "{what} {}: the path is not UTF-8, so no row can name it (the Python failed the check reporting it as missing)",
            py::os_display(relative.as_os_str())
        )));
    };
    py::PosixPath::parse(text)
        .relative_to(&py::PosixPath::parse(root_text))
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
                        .to_str()
                        .expect("rel() refused a name that is not UTF-8")
                        .to_owned();
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
