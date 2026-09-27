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
use nix::unistd::{access, AccessFlags};
use serde_json::Value;
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
        .map_err(|e| format!("cargo: {e}"))?;
    if !output.status.success() {
        return Err(Failure::Exit(status_code(output.status)));
    }
    serde_json::from_slice(&output.stdout).map_err(|e| Failure::Refused(e.to_string()))
}

fn list<'a>(value: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| py::repr(key))
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|f| f != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    }
}

/// The one executable a successful build produced for `package`/`kind`/`name`, optionally
/// translated from a container mount (`path_map` = container root, host root), resolved.
pub fn executable(
    messages: &str,
    meta: &Value,
    package: &str,
    kind: &str,
    name: &str,
    path_map: Option<(&Path, &Path)>,
) -> Result<PathBuf, String> {
    let members = list(meta, "workspace_members")?;
    let mut packages = vec![];
    for p in list(meta, "packages")? {
        if p.get("name").and_then(Value::as_str) == Some(package)
            && p.get("id").is_some_and(|id| members.contains(id))
        {
            packages.push(p["id"].clone());
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
    for line in py::splitlines(messages) {
        if !line.starts_with('{') {
            continue; // Cargo documents possible non-JSON output from other tools.
        }
        let message: Value = serde_json::from_str(line).map_err(|e| e.to_string())?;
        let reason = message.get("reason").and_then(Value::as_str);
        if reason == Some("build-finished") {
            finished = message.get("success") == Some(&Value::Bool(true));
        }
        let target = message.get("target");
        let kinds = target.and_then(|t| t.get("kind"));
        let kind_matches = match kinds {
            Some(Value::Array(items)) => items.iter().any(|k| k == kind),
            Some(Value::String(s)) => s.contains(kind),
            _ => false,
        };
        if reason == Some("compiler-artifact")
            && message.get("package_id") == Some(&packages[0])
            && target.and_then(|t| t.get("name")).and_then(Value::as_str) == Some(name)
            && kind_matches
            && !truthy(message.get("profile").and_then(|p| p.get("test")))
            && truthy(message.get("executable"))
        {
            let path = message["executable"]
                .as_str()
                .ok_or("executable is not a path string")?;
            matches.push(py::pure_path(path));
        }
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
        let relative = path
            .strip_prefix(py::pure_path(&container.to_string_lossy()))
            .map_err(|_| {
                format!(
                    "{} is not in the subpath of {}",
                    py::repr(&path.to_string_lossy()),
                    py::repr(&container.to_string_lossy())
                )
            })?;
        path = py::pure_path(&host.to_string_lossy()).join(relative);
    }
    if !path.is_file() || access(&path, AccessFlags::X_OK).is_err() {
        return Err(format!(
            "Cargo executable is missing or not executable: {}",
            path.display()
        ));
    }
    path.canonicalize().map_err(|e| e.to_string())
}

/// Write `output` (which must not exist): `store = { dir = "<target>/nextest" }` followed by
/// every byte of the nextest config at `source`. Nextest's default store is workspace-relative,
/// independently of Cargo; TOML basic strings accept JSON's escaped string form, including
/// spaces in an absolute target directory.
pub fn nextest_config(source: &Path, output: &Path, target: &Path) -> Result<(), String> {
    let raw = fs::read_to_string(source).map_err(|e| format!("{}: {e}", source.display()))?;
    // Python's read_text() translates newlines universally.
    let text = raw.replace("\r\n", "\n").replace('\r', "\n");
    let table: toml::Table = text.parse().map_err(|e: toml::de::Error| e.to_string())?;
    if table.contains_key("store") {
        return Err(
            "nextest already configures a store; reconcile it before relocating reports".into(),
        );
    }
    let store = py::pure_path(&target.to_string_lossy()).join("nextest");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .map_err(|e| format!("{}: {e}", output.display()))?;
    let header = format!(
        "store = {{ dir = {} }}\n",
        py::json_str(&store.to_string_lossy())
    );
    file.write_all(header.as_bytes())
        .and_then(|()| file.write_all(text.as_bytes()))
        .map_err(|e| format!("{}: {e}", output.display()))
}

/// `build`: compile one bin/example and return its executable (never one from a failed build).
pub struct Build<'a> {
    pub manifest_path: &'a str,
    pub target_dir: Option<&'a str>,
    pub package: &'a str,
    pub kind: &'a str,
    pub name: &'a str,
    pub profile: &'a str,
    pub features: &'a [String],
    pub no_default_features: bool,
}

pub fn build(request: &Build<'_>) -> Result<PathBuf, Failure> {
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
        .map_err(|e| format!("cargo: {e}"))?;
    // Never return an artifact from a build that later failed.
    if !output.status.success() {
        return Err(Failure::Exit(status_code(output.status)));
    }
    let messages = String::from_utf8_lossy(&output.stdout);
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
) -> Result<PathBuf, Failure> {
    let read = |path: &Path| {
        fs::read_to_string(path).map_err(|e| Failure::Refused(format!("{}: {e}", path.display())))
    };
    let meta: Value =
        serde_json::from_str(&read(metadata_path)?).map_err(|e| Failure::Refused(e.to_string()))?;
    let messages = read(messages_path)?;
    Ok(executable(&messages, &meta, package, kind, name, path_map)?)
}

/// `target-dir`: Cargo's effective target directory for the manifest.
pub fn target_directory(manifest: &str, target_dir: Option<&str>) -> Result<String, Failure> {
    let meta = metadata(manifest, target_dir)?;
    meta.get("target_directory")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| Failure::Refused(py::repr("target_directory")))
}
