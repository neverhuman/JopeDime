//! `jope-ci cargo-artifact`: resolve Cargo executables and reports without assuming the target
//! directory layout.
//!
//! Rust port of `deploy/ci/cargo-artifact.py` (archived at `8a55c4fef`; see
//! `docs/archive/python-archive-manifest.json`) with the same subcommands, flags, output and
//! refusal messages. `build` writes diagnostics to stderr and exactly one executable path to
//! stdout; `select` consumes the messages of a successful (possibly containerised) build,
//! translating a mounted root. Cargo's compiler-artifact messages include cached
//! (`fresh: true`) executables too:
//! <https://doc.rust-lang.org/cargo/reference/external-tools.html#artifact-messages>

use crate::py;
use jope_records::py::{self as python, exception};
use jope_records::pyjson::{self, Value};
use nix::unistd::{access, AccessFlags};
use std::fs::{self, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};

/// Why a subcommand did not print its result.
#[derive(Debug, PartialEq, Eq)]
pub enum Failure {
    /// A refusal: `cargo-artifact: <message>` on stderr and exit status 1.
    Refused(String),
    /// A Cargo subprocess failed; its exit status is propagated.
    Exit(i32),
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Failure::Refused(message)
    }
}

/// The exit status a shell reports for `status`: the code, or 128 + the terminating signal.
pub fn status_code(status: ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

/// The environment every Cargo call gets: incremental compilation off and line-table debug
/// info unless the caller already chose, and `CARGO_TARGET_DIR` when one is named.
pub fn cargo_command(target_dir: Option<&str>) -> Command {
    let mut command = Command::new("cargo");
    if std::env::var_os("CARGO_INCREMENTAL").is_none() {
        command.env("CARGO_INCREMENTAL", "0");
    }
    if std::env::var_os("CARGO_PROFILE_DEV_DEBUG").is_none() {
        command.env("CARGO_PROFILE_DEV_DEBUG", "line-tables-only");
    }
    if let Some(target) = target_dir {
        command.env("CARGO_TARGET_DIR", target);
    }
    command
}

/// `cargo metadata --locked --no-deps --format-version 1 --manifest-path <manifest>`.
pub fn metadata(manifest: &str, target_dir: Option<&str>) -> Result<Value, Failure> {
    let output = cargo_command(target_dir)
        .args(["metadata", "--locked", "--no-deps", "--format-version", "1"])
        .args(["--manifest-path", manifest])
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| python::os_error(&e, "cargo"))?;
    if !output.status.success() {
        // `except CalledProcessError as exc: return exc.returncode`, then `sys.exit(...)`: a
        // Cargo killed by a signal gives the negative return code, which exits as its low byte.
        return Err(Failure::Exit(
            i32::try_from(python::returncode(output.status)).unwrap_or(1),
        ));
    }
    // `json.loads(check_output(...))`: bytes, so the encoding is detected; Python's JSON reading
    // (NaN and Infinity literals, a repeated key keeps its last value, unbounded integers) and
    // its error messages.
    pyjson::parse_bytes_big(&output.stdout).map_err(Failure::Refused)
}

/// `Path(path).read_text()`: strict UTF-8 with universal newlines. A file that cannot be read
/// refuses as the Python printed the `OSError` it caught (`[Errno 2] No such file or directory:
/// '<path>'`, the path as `str(Path(path))`), and bytes that are not UTF-8 as its
/// `UnicodeDecodeError` (`'utf-8' codec can't decode byte 0xff in position 0: invalid start
/// byte`).
pub fn read_text(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| os_error_at(&e, path))?;
    pyjson::decode_utf8(&bytes).map(python::universal_newlines)
}

/// The `OSError` the Python printed for `path`, named as `str(Path(path))` spells it (a message
/// only). A name that is not UTF-8, which only a library caller can pass (the command line refuses
/// one), is named as `repr` printed it, never converted lossily.
fn os_error_at(error: &std::io::Error, path: &Path) -> String {
    match path.to_str() {
        Some(text) => python::os_error(error, &python::PosixPath::parse(text).to_string()),
        None => python::os_error_path(error, path),
    }
}

// How the Python read `metadata` and the message stream. JSON is read with Python's `json`
// semantics (integers of any size, every `NaN` one object) and compared with Python's `==`
// (`1 == 1.0 == True`, `[1] == [1.0]`, `NaN in [NaN]` but `NaN != NaN`), so an id matches exactly
// when it matched in Python. A missing key raised `KeyError`, which the Python caught and
// printed as `cargo-artifact: '<key>'` (exit status 1); the port prints the same. A value of the
// wrong type raised a `TypeError` or `AttributeError` it did not catch (a traceback, exit status
// 1); the port refuses too (`cargo-artifact: <field>: <what failed and the offending type>`, exit
// status 1; the text is not CPython's). No value is skipped or defaulted.

/// `value[key]`.
fn index<'a>(value: &'a Value, key: &str, context: &str) -> Result<&'a Value, String> {
    match value {
        Value::Dict(map) => map.get(key).ok_or_else(|| py::repr(key)),
        other => Err(format!(
            "{context}: {}",
            exception::not_subscriptable(other)
        )),
    }
}

/// `for x in value`: a list's items, an object's keys, a string's characters.
fn iterate(value: &Value, context: &str) -> Result<Vec<Value>, String> {
    python::iterate(value).ok_or_else(|| format!("{context}: {}", exception::not_iterable(value)))
}

/// `item in container`, with Python's `==` and hashing.
fn contains(container: &Value, item: &Value, context: &str) -> Result<bool, String> {
    python::contains_or_raise(container, item).map_err(|e| format!("{context}: {e}"))
}

/// `message.get(key, {})` used as an object (`.get` on anything else raised `AttributeError`).
fn object<'a>(
    message: &'a Value,
    key: &str,
    context: &str,
) -> Result<Option<&'a pyjson::Dict>, String> {
    match message.get(key) {
        None => Ok(None),
        Some(Value::Dict(map)) => Ok(Some(map)),
        Some(other) => Err(format!(
            "{context} {key}: {}",
            exception::no_attribute(other, "get")
        )),
    }
}

fn truthy(value: Option<&Value>) -> bool {
    value.is_some_and(Value::truthy)
}

/// The one executable a successful build produced for `package`/`kind`/`name`, optionally
/// translated from a container mount (`path_map` = container root, host root), resolved, as the
/// text the command prints.
///
/// The Python printed `path.resolve()`, a `str`: a resolved name that is not UTF-8 (a link to a
/// directory named with the byte `0xff`, say) was surrogate-escaped, so under a strict UTF-8
/// stdout (`en_US.UTF-8` and the other UTF-8 locales but C and POSIX) the print raised
/// `UnicodeEncodeError`, which it caught and printed (`cargo-artifact: 'utf-8' codec can't encode
/// character '\udcff' in position 57: surrogates not allowed`, exit status 1), and under C or
/// `C.UTF-8` (a `surrogateescape` stdout) it printed the raw bytes. The port refuses such a name
/// with that message in every locale (stricter under C and `C.UTF-8`); it never prints a lossy
/// conversion (U+FFFD), which names a path that does not exist.
pub fn executable(
    messages: &str,
    meta: &Value,
    package: &str,
    kind: &str,
    name: &str,
    path_map: Option<(&Path, &Path)>,
) -> Result<String, String> {
    // `[p["id"] for p in meta["packages"]
    //  if p["name"] == package and p["id"] in meta["workspace_members"]]`
    let mut packages = vec![];
    for (i, p) in iterate(
        index(meta, "packages", "cargo metadata")?,
        "cargo metadata packages",
    )?
    .iter()
    .enumerate()
    {
        let at = format!("cargo metadata packages[{i}]");
        if index(p, "name", &at)?.as_str() != Some(package) {
            continue;
        }
        let id = index(p, "id", &at)?;
        let members = index(meta, "workspace_members", "cargo metadata")?;
        if contains(members, id, "cargo metadata workspace_members")? {
            packages.push(id.clone());
        }
    }
    if packages.len() != 1 {
        return Err(format!(
            "expected one workspace package {}, found {}",
            py::repr(package),
            packages.len()
        ));
    }
    let mut matches = vec![];
    let mut finished = false;
    for (number, line) in py::splitlines(messages).into_iter().enumerate() {
        if !line.starts_with('{') {
            continue; // Cargo documents possible non-JSON output from other tools.
        }
        // A line that starts with `{` is a JSON object or refuses with Python's decoder message.
        let message = pyjson::parse_str_big(line)?;
        let at = format!("Cargo message on line {}", number + 1);
        let reason = message.get("reason").and_then(Value::as_str);
        if reason == Some("build-finished") {
            // `message.get("success") is True`: only JSON `true` (Python's `==` would take 1).
            finished = matches!(message.get("success"), Some(Value::Bool(true)));
        }
        // The artifact test, clause by clause in the Python's order; each clause is read only
        // when the ones before it held, and a malformed value it reads refuses.
        if reason != Some("compiler-artifact")
            || message.get("package_id").unwrap_or(&Value::Null) != &packages[0]
        {
            continue;
        }
        let target = object(&message, "target", &at)?;
        if target.and_then(|t| t.get("name")).and_then(Value::as_str) != Some(name) {
            continue;
        }
        let no_kinds = Value::List(vec![]);
        let kinds = target.and_then(|t| t.get("kind")).unwrap_or(&no_kinds);
        if !contains(kinds, &Value::from(kind), &format!("{at} target kind"))? {
            continue;
        }
        let profile = object(&message, "profile", &at)?;
        if truthy(profile.and_then(|p| p.get("test"))) || !truthy(message.get("executable")) {
            continue;
        }
        // `Path(message["executable"])`
        let executable = message.get("executable").unwrap_or(&Value::Null);
        let path = executable
            .as_str()
            .ok_or_else(|| format!("{at} executable: {}", exception::path_argument(executable)))?;
        matches.push(python::PosixPath::parse(path));
    }
    if !finished {
        return Err("Cargo did not report a successful completed build".into());
    }
    if matches.len() != 1 {
        return Err(format!(
            "expected one executable for {package}/{kind}/{name}, found {}",
            matches.len()
        ));
    }
    let mut path = matches.remove(0);
    if let Some((container, host)) = path_map {
        // `Path(host) / path.relative_to(container)`: pathlib compares the two as spelled after
        // parsing, so a path under POSIX's distinct `//` root is not under `/`, nor the reverse
        // (`///` is `/`); the refusal names both as `str()` prints them. The container itself
        // maps to the host itself. Both roots are compared and joined as text, so one that is not
        // UTF-8 refuses (the command line refuses it first; see `main`), never converted lossily:
        // a lossy container root matched a message path spelled with U+FFFD, and a lossy host
        // root named another directory.
        fn root<'a>(root: &'a Path, which: &str) -> Result<&'a str, String> {
            root.to_str().ok_or_else(|| {
                format!(
                    "--path-map {which} {} is not UTF-8",
                    python::os_repr(root.as_os_str())
                )
            })
        }
        let container = python::PosixPath::parse(root(container, "CONTAINER_ROOT")?);
        let relative = path.relative_to(&container)?;
        path = python::PosixPath::parse(root(host, "HOST_ROOT")?).joinpath(&relative);
    }
    // `path.is_file()`, `os.access(path, os.X_OK)` and `path.resolve()` of `str(path)`, which the
    // refusal prints. `is_file()` raised on an error other than a missing path (a directory that
    // may not be searched), which the Python caught and printed as the `OSError`; so does this.
    // `os.access` answered `False` on any error, which refuses.
    let file = PathBuf::from(path.to_string());
    let is_file =
        python::path_is_file(&file).map_err(|e| python::os_error(&e, &path.to_string()))?;
    if !is_file || access(&file, AccessFlags::X_OK).is_err() {
        return Err(format!(
            "Cargo executable is missing or not executable: {path}"
        ));
    }
    // `path.resolve()`: `realpath(3)` refuses a resolved name longer than `PATH_MAX` (`[Errno 36]
    // File name too long`, the executable reached through a short link to a deep directory),
    // which Python's `resolve()` built and printed (stricter), and any other error (a race).
    let resolved = file
        .canonicalize()
        .map_err(|e| python::os_error(&e, &path.to_string()))?;
    // `print(path.resolve())`: text, or the `UnicodeEncodeError` a strict UTF-8 stdout raised.
    resolved.into_os_string().into_string().map_err(|name| {
        python::utf8_encode_error(&name).expect("a name that is not UTF-8 has an undecodable byte")
    })
}

/// The TOML value types the port reads in the nextest config: those the committed
/// `.config/nextest.toml` uses (tables, inline tables, strings, integers, booleans). Any other
/// refuses before anything is written, naming the key (stricter than the Python, which read them):
///
/// - a date, time or datetime of any form: `tomllib` and the `toml` crate disagree on which are
///   valid (the crate reads a seconds field of 60 and the year 0000, which `tomllib` refused), so
///   the port reads none rather than one `tomllib` refused;
/// - a float or an array (an array of tables included): no committed configuration holds one, so
///   their readings were never compared with `tomllib`'s.
fn committed_types_only(table: &toml::Table, at: &str) -> Result<(), String> {
    for (key, value) in table {
        let at = if at.is_empty() {
            py::json_str(key)
        } else {
            format!("{at}.{}", py::json_str(key))
        };
        let refused = match value {
            toml::Value::String(_) | toml::Value::Integer(_) | toml::Value::Boolean(_) => None,
            toml::Value::Table(inner) => {
                committed_types_only(inner, &at)?;
                None
            }
            toml::Value::Datetime(_) => Some("a date, time or datetime"),
            toml::Value::Float(_) => Some("a float"),
            toml::Value::Array(_) => Some("an array"),
        };
        if let Some(kind) = refused {
            return Err(format!(
                "nextest config {at}: {kind} is not read by this port (the committed configuration uses only tables, strings, integers and booleans)"
            ));
        }
    }
    Ok(())
}

/// Write `output` (which must not exist): `store = { dir = "<target>/nextest" }` followed by
/// every byte of the nextest config at `source`. Nextest's default store is workspace-relative,
/// independently of Cargo; TOML basic strings accept JSON's escaped string form, including
/// spaces in an absolute target directory.
///
/// The steps run in the Python's order, so the first refusal is the one it printed: read the
/// source, parse it (a leading U+FEFF, which `read_text` keeps, is `tomllib`'s `Invalid statement
/// (at line 1, column 1)`; any other TOML error is reported in the `toml` crate's words, not
/// `tomllib`'s; the crate also refuses a few documents `tomllib` read, such as a dotted key that
/// extends a table a header created implicitly, `[a.b.c]` then `[a]` with `b.d = 1` (`duplicate
/// key`), which refuse before any output exists: stricter), refuse a `store`, create the output
/// (`open("x")`), and only then read `target`
/// as a path. A `target` that is not a string crashed the Python there (`TypeError`), after it had
/// created the output; the port refuses at the same point and leaves the same empty file. After
/// the `store` check, a value of a type the committed configuration does not use refuses
/// (stricter; see `committed_types_only`), before any output exists.
pub fn nextest_config(source: &Path, output: &Path, target: &Value) -> Result<(), String> {
    let text = read_text(source)?;
    if text.starts_with('\u{feff}') {
        return Err("Invalid statement (at line 1, column 1)".into());
    }
    let table: toml::Table = text.parse().map_err(|e: toml::de::Error| e.to_string())?;
    if table.contains_key("store") {
        return Err(
            "nextest already configures a store; reconcile it before relocating reports".into(),
        );
    }
    committed_types_only(&table, "")?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| os_error_at(&e, output))?;
    // `str(Path(target) / "nextest")`
    let target = target.as_str().ok_or_else(|| {
        format!(
            "cargo metadata target_directory: {}",
            exception::path_argument(target)
        )
    })?;
    let store = python::PosixPath::parse(target).join("nextest").to_string();
    let header = format!("store = {{ dir = {} }}\n", py::json_str(&store));
    file.write_all(header.as_bytes())
        .and_then(|()| file.write_all(text.as_bytes()))
        .map_err(|e| os_error_at(&e, output))
}

/// `build`: compile one bin/example and return its executable (never one from a failed build).
pub struct Build<'a> {
    /// Cargo manifest path passed to `cargo metadata` and `cargo build`.
    pub manifest_path: &'a str,
    /// Optional explicit Cargo target directory.
    pub target_dir: Option<&'a str>,
    /// Package name to build.
    pub package: &'a str,
    /// Cargo target kind, either `bin` or `example`.
    pub kind: &'a str,
    /// Name of the selected binary or example target.
    pub name: &'a str,
    /// Cargo profile used for the build.
    pub profile: &'a str,
    /// Feature names enabled for the build.
    pub features: &'a [String],
    /// Whether to disable the package's default features.
    pub no_default_features: bool,
}

/// Build the selected binary or example and return its resolved executable path.
pub fn build(request: &Build<'_>) -> Result<String, Failure> {
    let meta = metadata(request.manifest_path, request.target_dir)?;
    let mut command = cargo_command(request.target_dir);
    command
        .args([
            "build",
            "--locked",
            "--message-format=json-render-diagnostics",
        ])
        .args(["--manifest-path", request.manifest_path])
        .args(["--package", request.package])
        .args(["--profile", request.profile])
        .arg(format!("--{}", request.kind))
        .arg(request.name);
    for features in request.features {
        command.args(["--features", features]);
    }
    if request.no_default_features {
        command.arg("--no-default-features");
    }
    let output = command
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .output()
        .map_err(|e| python::os_error(&e, "cargo"))?;
    // `subprocess.run(..., stdout=PIPE, text=True)` decoded the stream before it returned, so
    // bytes that are not UTF-8 refuse (with the `UnicodeDecodeError` the Python printed) even
    // when the build failed.
    let messages = python::text_output(&output.stdout)?;
    // Never return an artifact from a build that later failed.
    if !output.status.success() {
        return Err(Failure::Exit(status_code(output.status)));
    }
    Ok(executable(
        &messages,
        &meta,
        request.package,
        request.kind,
        request.name,
        None,
    )?)
}

/// `select`: the executable of a successful build recorded in `metadata` + `messages` files.
pub fn select(
    metadata_path: &Path,
    messages_path: &Path,
    package: &str,
    kind: &str,
    name: &str,
    path_map: Option<(&Path, &Path)>,
) -> Result<String, Failure> {
    // `json.loads(Path(...).read_text())`: a `str` (a UTF-8 BOM refuses), positions counted after
    // universal-newline translation, unbounded integers; then the messages.
    let meta = pyjson::parse_str_big(&read_text(metadata_path)?).map_err(Failure::Refused)?;
    let messages = read_text(messages_path)?;
    Ok(executable(&messages, &meta, package, kind, name, path_map)?)
}

/// `meta["target_directory"]` of Cargo's metadata for the manifest; a missing key refuses as the
/// Python's caught `KeyError` did (`cargo-artifact: 'target_directory'`).
pub fn target_value(manifest: &str, target_dir: Option<&str>) -> Result<Value, Failure> {
    let meta = metadata(manifest, target_dir)?;
    Ok(index(&meta, "target_directory", "cargo metadata")?.clone())
}

/// `target-dir`: Cargo's effective target directory for the manifest. A value that is not a
/// string refuses, which is stricter than the Python (it printed `str(value)`, such as `None`).
pub fn target_directory(manifest: &str, target_dir: Option<&str>) -> Result<String, Failure> {
    let value = target_value(manifest, target_dir)?;
    value.as_str().map(str::to_owned).ok_or_else(|| {
        Failure::Refused(format!(
            "cargo metadata target_directory: {}",
            exception::path_argument(&value)
        ))
    })
}
