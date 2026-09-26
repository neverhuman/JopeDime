//! The prediction-API latency stage of the GPU timing study (W11), run by the CUDA controller
//! after the full-GP phase. The stage itself needs no GPU.
//!
//! For every dataset's warm export and each backend whose export completed and whose bundle still
//! hashes to its receipt, the bundle is materialized into a fresh
//! `prediction-<dataset>-<backend>-<rows>` directory for 1, 32 and 1024 rows (the first fitting
//! row repeated; no scoring labels). The emitted latency harness is compiled if the backend needs
//! it and run once for ten timed public-API calls. The receipt says either completed, with
//! exact-bit parity against the qualification plan's first expected value, or failed, with the
//! cause. An existing directory is an interrupted attempt and is never replaced, and the stage
//! stops at the cutoff. Receipts and generated files keep the bytes of the retired Python driver,
//! including its failure texts for spawn errors, timeouts, exits and unreadable samples.

use super::reporting::{
    at, dumps, exists, float_repr, get_or, index, is, py_iter, py_repr, py_str, sha256_hex,
    str_repr, Json, Object, Style, EMPTY_ARRAY, EMPTY_OBJECT,
};
use anyhow::{bail, ensure, Context, Result};
use std::{
    fs,
    io::Read,
    path::{Component, Path, PathBuf},
    process::{Command, ExitStatus, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const BACKENDS: [&str; 3] = ["python", "rust", "cpp17"];
const SIZES: [usize; 3] = [1, 32, 1024];
const NOTE: &str = "Public API timing includes return serialization; process residual also includes initialization, parsing, printing and teardown. Repeated first fitting-feature row; no scoring labels.";

/// A failure recorded in the receipt, spelled as Python's `str(exception)`.
type Attempt<T> = std::result::Result<T, String>;

fn now_ms() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64() * 1000.0)
}

fn load(path: &Path) -> Result<Json> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))
}

fn text<'a>(value: &'a Json, what: &str) -> Result<&'a str> {
    match value {
        Json::Str(s) => Ok(s),
        other => bail!("TypeError: {what} is a JSON {}", py_type(other)),
    }
}

/// `value[0]`: a failure stops the stage, as the uncaught exception did.
fn first(value: &Json) -> Result<Json> {
    match value {
        Json::Array(items) => items
            .first()
            .cloned()
            .context("IndexError: list index out of range"),
        Json::Str(s) => s
            .chars()
            .next()
            .map(|c| Json::Str(c.to_string()))
            .context("IndexError: string index out of range"),
        Json::Object(_) => bail!("KeyError: 0"),
        other => bail!(
            "TypeError: '{}' object is not subscriptable",
            py_type(other)
        ),
    }
}

/// Python's type name of a JSON value.
fn py_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Float(_) => "float",
        Json::Str(_) => "str",
        Json::Array(_) => "list",
        Json::Object(_) => "dict",
    }
}

/// `str(pathlib.Path(path))`: empty and `.` components dropped.
fn python_path(path: &Path) -> String {
    let mut out = String::new();
    for component in path.components() {
        let part = match component {
            Component::RootDir => {
                out.push('/');
                continue;
            }
            Component::CurDir | Component::Prefix(_) => continue,
            Component::ParentDir => "..".into(),
            Component::Normal(part) => part.to_string_lossy(),
        };
        if !out.is_empty() && !out.ends_with('/') {
            out.push('/');
        }
        out.push_str(&part);
    }
    if out.is_empty() {
        ".".into()
    } else {
        out
    }
}

/// `str(OSError)`: `[Errno N] strerror: 'filename'`.
fn os_error(error: &std::io::Error, filename: &str) -> String {
    match error.raw_os_error() {
        Some(code) => {
            let message = std::io::Error::from_raw_os_error(code).to_string();
            let strerror = message
                .strip_suffix(&format!(" (os error {code})"))
                .unwrap_or(&message);
            format!("[Errno {code}] {strerror}: {}", str_repr(filename))
        }
        None => error.to_string(),
    }
}

fn write_file(path: &Path, bytes: &[u8]) -> Attempt<()> {
    fs::write(path, bytes).map_err(|error| os_error(&error, &python_path(path)))
}

/// Python's `repr(bytes)`.
fn bytes_repr(bytes: &[u8]) -> String {
    let delimiter = if bytes.contains(&b'\'') && !bytes.contains(&b'"') {
        b'"'
    } else {
        b'\''
    };
    let mut out = String::from("b");
    out.push(char::from(delimiter));
    for &byte in bytes {
        match byte {
            b'\\' => out.push_str("\\\\"),
            b'\t' => out.push_str("\\t"),
            b'\n' => out.push_str("\\n"),
            b'\r' => out.push_str("\\r"),
            _ if byte == delimiter => {
                out.push('\\');
                out.push(char::from(byte));
            }
            0x20..=0x7e => out.push(char::from(byte)),
            _ => out.push_str(&format!("\\x{byte:02x}")),
        }
    }
    out.push(char::from(delimiter));
    out
}

/// `bytes.splitlines()`: `\n`, `\r` and `\r\n` end a line; no empty final line.
fn splitlines(bytes: &[u8]) -> Vec<&[u8]> {
    let (mut lines, mut start, mut i) = (Vec::new(), 0, 0);
    while i < bytes.len() {
        match bytes[i] {
            b'\n' => {
                lines.push(&bytes[start..i]);
                i += 1;
                start = i;
            }
            b'\r' => {
                lines.push(&bytes[start..i]);
                i += if bytes.get(i + 1) == Some(&b'\n') {
                    2
                } else {
                    1
                };
                start = i;
            }
            _ => i += 1,
        }
    }
    if start < bytes.len() {
        lines.push(&bytes[start..]);
    }
    lines
}

/// `float(line)` for a bytes line: ASCII whitespace trimmed and underscores allowed between
/// digits; the error text is Python's.
fn python_float(line: &[u8]) -> Attempt<f64> {
    let failure = || format!("could not convert string to float: {}", bytes_repr(line));
    let space = |b: &u8| matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let start = line.iter().position(|b| !space(b)).unwrap_or(line.len());
    let end = line
        .iter()
        .rposition(|b| !space(b))
        .map_or(start, |p| p + 1);
    let trimmed = std::str::from_utf8(&line[start..end]).map_err(|_| failure())?;
    let digits = trimmed.as_bytes();
    for (i, byte) in digits.iter().enumerate() {
        let digit = |j: usize| digits.get(j).is_some_and(u8::is_ascii_digit);
        if *byte == b'_' && !(i > 0 && digit(i - 1) && digit(i + 1)) {
            return Err(failure());
        }
    }
    trimmed.replace('_', "").parse().map_err(|_| failure())
}

/// Python 3.12's built-in `sum` of floats: compensated (Neumaier) summation.
fn python_sum(values: &[f64]) -> f64 {
    let Some((head, rest)) = values.split_first() else {
        return 0.0;
    };
    let (mut total, mut compensation) = (0.0 + head, 0.0_f64);
    for &x in rest {
        let next = total + x;
        compensation += if total.abs() >= x.abs() {
            (total - next) + x
        } else {
            (x - next) + total
        };
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// `min(30, max(.001, seconds_to_cutoff))`, with its Python spelling.
fn timeout(cutoff_ms: u64) -> (Duration, String) {
    let remaining = (cutoff_ms as f64 - now_ms()) / 1000.0;
    let bounded = if remaining > 0.001 { remaining } else { 0.001 };
    if bounded < 30.0 {
        (Duration::from_secs_f64(bounded), float_repr(bounded))
    } else {
        (Duration::from_secs(30), "30".into())
    }
}

struct Finished {
    code: i32,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn exit_code(status: ExitStatus) -> i32 {
    use std::os::unix::process::ExitStatusExt;
    status
        .code()
        .unwrap_or_else(|| -status.signal().unwrap_or(0))
}

/// `subprocess.run(args, cwd=dir, capture_output=True, timeout=...)`, bounded by the cutoff.
fn run_process(args: &[String], dir: &Path, cutoff_ms: u64) -> Attempt<Finished> {
    let (limit, spelling) = timeout(cutoff_ms);
    let timed_out = || {
        let command = Json::Array(args.iter().map(|a| Json::Str(a.clone())).collect());
        format!(
            "Command '{}' timed out after {spelling} seconds",
            py_repr(&command)
        )
    };
    // A path-like program resolves against the child's working directory, as with Python.
    let program = if args[0].contains('/') {
        dir.join(&args[0])
    } else {
        PathBuf::from(&args[0])
    };
    let mut child = Command::new(program)
        .args(&args[1..])
        .current_dir(dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| os_error(&error, &args[0]))?;
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        thread::spawn(move || {
            let mut bytes = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut bytes);
            }
            bytes
        })
    };
    let stdout = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let stderr = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn Read + Send>),
    );
    let deadline = Instant::now() + limit;
    // Output is complete when the child has exited and both pipes are closed.
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) if stdout.is_finished() && stderr.is_finished() => break status,
            Ok(_) if Instant::now() >= deadline => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(timed_out());
            }
            Ok(_) => thread::sleep(Duration::from_millis(1)),
            Err(error) => return Err(error.to_string()),
        }
    };
    Ok(Finished {
        code: exit_code(status),
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

/// `str(UnicodeDecodeError)` for UTF-8 text.
fn utf8_error(bytes: &[u8], error: &std::str::Utf8Error) -> String {
    let start = error.valid_up_to();
    match error.error_len() {
        Some(_) => format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: {}",
            bytes[start],
            if (0xc2..=0xf4).contains(&bytes[start]) {
                "invalid continuation byte"
            } else {
                "invalid start byte"
            }
        ),
        None if bytes.len() - start == 1 => format!(
            "'utf-8' codec can't decode byte 0x{:02x} in position {start}: unexpected end of data",
            bytes[start]
        ),
        None => format!(
            "'utf-8' codec can't decode bytes in position {start}-{}: unexpected end of data",
            bytes.len() - 1
        ),
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

/// The Rust literal of the repeated input rows: `None`, or the value's UTF-8 bytes.
fn rust_literal(selected: &[Json]) -> Attempt<String> {
    let mut rows = Vec::with_capacity(selected.len());
    for row in selected {
        let cells = match row {
            Json::Array(items) => items.clone(),
            Json::Object(o) => o.0.iter().map(|(k, _)| Json::Str(k.clone())).collect(),
            Json::Str(s) => s.chars().map(|c| Json::Str(c.to_string())).collect(),
            other => return Err(format!("'{}' object is not iterable", py_type(other))),
        };
        let mut rendered = Vec::with_capacity(cells.len());
        for cell in &cells {
            rendered.push(match cell {
                Json::Null => "None".to_owned(),
                Json::Str(s) => format!(
                    "Some(String::from_utf8(vec![{}]).unwrap())",
                    s.bytes()
                        .map(|b| b.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                ),
                other => {
                    return Err(format!(
                        "'{}' object has no attribute 'encode'",
                        py_type(other)
                    ))
                }
            });
        }
        rows.push(format!("vec![{}]", rendered.join(",")));
    }
    Ok(format!("vec![{}]", rows.join(",")))
}

/// `actual.get('kind') == 'values'`, one value per row and every value's bits equal to
/// `expected`'s; Python's exception texts for malformed predictions.
fn parity(actual: &Json, rows: usize, expected: &Json) -> Attempt<()> {
    let refused = || Err("prediction parity failed".to_owned());
    let Json::Object(members) = actual else {
        return Err(format!(
            "'{}' object has no attribute 'get'",
            py_type(actual)
        ));
    };
    if !members.get("kind").is_some_and(|kind| is(kind, "values")) {
        return refused();
    }
    let data = members.get("data").ok_or_else(|| "'data'".to_owned())?;
    let values: Vec<Json> = match data {
        Json::Array(items) => items.clone(),
        Json::Object(o) => o.0.iter().map(|(k, _)| Json::Str(k.clone())).collect(),
        Json::Str(s) => s.chars().map(|c| Json::Str(c.to_string())).collect(),
        other => return Err(format!("object of type '{}' has no len()", py_type(other))),
    };
    if values.len() != rows {
        return refused();
    }
    let bits = |value: &Json| match value {
        Json::Float(f) => Ok(f.to_bits()),
        Json::Int(i) => Ok((*i as f64).to_bits()),
        Json::Bool(b) => Ok(f64::from(u8::from(*b)).to_bits()),
        _ => Err("required argument is not a float".to_owned()),
    };
    for value in &values {
        if bits(value)? != bits(expected)? {
            return refused();
        }
    }
    Ok(())
}

/// One timed attempt: the completed receipt members, or the failure text.
fn attempt(
    backend: &str,
    out: &Path,
    selected: &[Json],
    expected: &Json,
    cutoff_ms: u64,
) -> Attempt<Vec<(&'static str, Json)>> {
    let (commands, launch) = if backend == "rust" {
        let literal = rust_literal(selected)?;
        let harness = String::from("mod model;fn main(){let rows:Vec<Vec<Option<String>>>=")
            + &literal
            + ";for _ in 0..10 {let t=std::time::Instant::now();let y=model::predict_raw(&rows).unwrap();let s=t.elapsed().as_secs_f64();println!(\"{}\",s);std::fs::write(\"predictions.json\",y).unwrap();}}";
        write_file(&out.join("latency.rs"), harness.as_bytes())?;
        (
            vec![strings(&[
                "rustc",
                "--edition=2021",
                "-O",
                "latency.rs",
                "-o",
                "latency",
            ])],
            strings(&["./latency"]),
        )
    } else {
        let commands = if backend == "cpp17" {
            vec![strings(&[
                "g++",
                "-std=c++17",
                "-O2",
                "-ffp-contract=off",
                "-fno-fast-math",
                "-shared",
                "-fPIC",
                "model.cpp",
                "-o",
                "model.so",
            ])]
        } else {
            Vec::new()
        };
        let rows = dumps(&Json::Array(selected.to_vec()), Style::DEFAULT)
            .map_err(|error| format!("{error:#}"))?;
        write_file(&out.join("rows.json"), rows.as_bytes())?;
        let module = if backend == "python" {
            "model"
        } else {
            "boundary"
        };
        // Generated harness text, byte for byte what the controller has always emitted.
        let harness = format!(
            "import {module},json,time\nrows=json.load(open(\"rows.json\"))\nfor _ in range(10):\n t=time.perf_counter();y={module}.predict_raw(rows);s=time.perf_counter()-t;print(s)\njson.dump(y,open(\"predictions.json\",\"w\"),allow_nan=False)\n"
        );
        write_file(&out.join("latency.py"), harness.as_bytes())?;
        (commands, strings(&["python3", "latency.py"]))
    };
    for (i, args) in commands.iter().enumerate() {
        let finished = run_process(args, out, cutoff_ms)?;
        write_file(&out.join(format!("compile-{i}.stderr")), &finished.stderr)?;
        if finished.code != 0 {
            return Err(format!("compiler exit {}", finished.code));
        }
    }
    let started = Instant::now();
    let finished = run_process(&launch, out, cutoff_ms)?;
    let wall = started.elapsed().as_secs_f64();
    write_file(&out.join("stdout"), &finished.stdout)?;
    write_file(&out.join("stderr"), &finished.stderr)?;
    if finished.code != 0 {
        return Err(format!("predict exit {}", finished.code));
    }
    let samples = splitlines(&finished.stdout)
        .into_iter()
        .map(python_float)
        .collect::<Attempt<Vec<f64>>>()?;
    if samples.len() != 10 || samples.iter().any(|x| !x.is_finite() || *x < 0.0) {
        return Err("missing timing samples".into());
    }
    let predictions = out.join("predictions.json");
    let location = python_path(&predictions);
    let bytes = fs::read(&predictions).map_err(|error| os_error(&error, &location))?;
    let decoded = std::str::from_utf8(&bytes).map_err(|error| utf8_error(&bytes, &error))?;
    let actual: Json = serde_json::from_str(decoded).map_err(|error| error.to_string())?;
    parity(&actual, selected.len(), expected)?;
    let bytes = fs::read(&predictions).map_err(|error| os_error(&error, &location))?;
    let mut commands: Vec<Json> = commands
        .iter()
        .map(|args| Json::Array(args.iter().map(|a| Json::Str(a.clone())).collect()))
        .collect();
    commands.push(Json::Array(
        launch.iter().map(|a| Json::Str(a.clone())).collect(),
    ));
    Ok(vec![
        ("status", "completed".into()),
        ("parity", "exact_bits".into()),
        (
            "prediction",
            Json::Object(Object(vec![
                ("path".into(), location.into()),
                ("sha256".into(), sha256_hex(&bytes).into()),
            ])),
        ),
        (
            "samples_seconds",
            Json::Array(samples.iter().map(|s| Json::Float(*s)).collect()),
        ),
        ("process_wall_seconds", Json::Float(wall)),
        (
            "initialization_and_io_residual_seconds",
            Json::Float(wall - python_sum(&samples)),
        ),
        ("commands", Json::Array(commands)),
        ("note", NOTE.into()),
    ])
}

/// Run the stage for a plan and journal until `cutoff_ms` (Unix milliseconds). An error stops
/// the whole stage, as an uncaught exception stopped the retired driver; receipts already
/// written stay.
pub(super) fn run(plan: &Path, journal: &Path, cutoff_ms: u64) -> Result<()> {
    stage(plan, journal, cutoff_ms, &SIZES)
}

/// [`run`] over the given batch sizes; unit tests skip the 1024-row compile.
fn stage(plan: &Path, journal: &Path, cutoff_ms: u64, sizes: &[usize]) -> Result<()> {
    let plan = load(plan)?;
    for dataset in py_iter(index(&plan, "inputs")?)? {
        let raw = index(&dataset, "raw")?;
        let rows = py_iter(index(
            &load(Path::new(text(index(raw, "path")?, "raw path")?))?,
            "rows",
        )?)?
        .iter()
        .map(|row| index(row, "values").cloned())
        .collect::<Result<Vec<_>>>()?;
        let id = index(&dataset, "id")?;
        let warm = journal.join(format!("warm-{}-0", py_str(id)));
        for backend in BACKENDS {
            let source = warm.join(backend);
            let bundle_path = source.join("bundle.json");
            if !exists(&bundle_path)? {
                continue;
            }
            let direct = load(&warm.join("receipt.json"))?;
            let mut entry = None;
            for b in py_iter(get_or(
                get_or(&direct, "export", &EMPTY_OBJECT)?,
                "backends",
                &EMPTY_ARRAY,
            )?)? {
                if is(at(&b, "backend")?, backend) && is(at(&b, "status")?, "completed") {
                    entry = Some(b);
                    break;
                }
            }
            let Some(entry) = entry else {
                continue;
            };
            let bundle = load(&bundle_path)?;
            let identity = Json::Str(sha256_hex(&fs::read(&bundle_path)?));
            ensure!(
                super::reporting::py_eq(&identity, index(index(&entry, "bundle")?, "sha256")?),
                "prediction bundle changed"
            );
            let expected = first(index(
                index(&load(&source.join("qualification-plan.json"))?, "expected")?,
                "data",
            )?)?;
            for &size in sizes {
                if now_ms() >= cutoff_ms as f64 {
                    return Ok(());
                }
                let out = journal.join(format!("prediction-{}-{backend}-{size}", py_str(id)));
                if exists(&out)? {
                    continue; // An interrupted attempt is not silently replaced.
                }
                fs::create_dir(&out).with_context(|| format!("create {}", out.display()))?;
                let row = rows
                    .first()
                    .context("IndexError: list index out of range")?;
                let selected = vec![row.clone(); size];
                let Json::Object(files) = index(&bundle, "files")? else {
                    bail!("AttributeError: bundle files have no items()");
                };
                for (name, content) in &files.0 {
                    fs::write(out.join(name), text(content, "bundle file")?)
                        .with_context(|| format!("write {name}"))?;
                }
                let experiment = index(&plan, "experiment");
                let mut receipt = Object(vec![
                    ("schema".into(), index(&plan, "schema")?.clone()),
                    ("dataset".into(), id.clone()),
                    ("backend".into(), backend.into()),
                    ("batch_rows".into(), size.into()),
                    ("status".into(), "failed".into()),
                    (
                        "plan_identity".into(),
                        index(experiment?, "methods_sha256")?.clone(),
                    ),
                    (
                        "source_sha".into(),
                        index(index(&plan, "experiment")?, "source_sha")?.clone(),
                    ),
                    (
                        "binary_sha256".into(),
                        index(index(&plan, "binary")?, "sha256")?.clone(),
                    ),
                    ("input_sha256".into(), index(raw, "sha256")?.clone()),
                    (
                        "source_qualification".into(),
                        index(&entry, "qualification_plan")?.clone(),
                    ),
                    (
                        "bundle_sha256".into(),
                        sha256_hex(&fs::read(&bundle_path)?).into(),
                    ),
                ]);
                match attempt(backend, &out, &selected, &expected, cutoff_ms) {
                    Ok(completed) => {
                        for (key, value) in completed {
                            receipt.set(key, value);
                        }
                    }
                    Err(failure) => receipt.set("failure", failure.into()),
                }
                fs::write(
                    out.join("receipt.json"),
                    dumps(&Json::Object(receipt), Style::REPORT)? + "\n",
                )?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Model sources for the Rust backend: exact, one bit away, and slow.
    const EXACT: &str = r#"pub fn predict_raw(rows: &[Vec<Option<String>>]) -> Result<String, String> {
    Ok(format!("{{\"kind\":\"values\",\"data\":[{}]}}", vec!["1.5"; rows.len()].join(",")))
}
"#;
    const ONE_BIT_AWAY: &str = r#"pub fn predict_raw(rows: &[Vec<Option<String>>]) -> Result<String, String> {
    Ok(format!("{{\"kind\":\"values\",\"data\":[{}]}}", vec!["1.5000000000000002"; rows.len()].join(",")))
}
"#;

    /// Harnesses are executed, so fixtures live beside the other executed test fixtures.
    fn scratch() -> tempfile::TempDir {
        let scratch =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
        fs::create_dir_all(&scratch).unwrap();
        tempfile::tempdir_in(scratch).unwrap()
    }

    fn pin(path: &Path) -> serde_json::Value {
        json!({"path": path, "sha256": sha256_hex(&fs::read(path).unwrap())})
    }

    /// A plan and a warm journal whose datasets export the given backends and bundle files.
    /// One export: backend, export status and bundle files.
    type Export<'a> = (&'a str, &'a str, &'a [(&'a str, &'a str)]);

    fn fixture(root: &Path, datasets: &[(&str, &[Export])]) -> PathBuf {
        let journal = root.join("journal");
        let mut inputs = Vec::new();
        for (id, exports) in datasets {
            let raw = root.join(format!("raw-{id}.json"));
            fs::write(
                &raw,
                r#"{"rows": [{"values": ["3.25", null, "é"]}, {"values": ["1", "2", "3"]}]}"#,
            )
            .unwrap();
            inputs.push(json!({"id": id, "raw": {"path": raw, "sha256": "r"}}));
            let warm = journal.join(format!("warm-{id}-0"));
            let mut backends = Vec::new();
            for (backend, status, files) in *exports {
                let source = warm.join(backend);
                fs::create_dir_all(&source).unwrap();
                let files: serde_json::Map<_, _> = files
                    .iter()
                    .map(|(k, v)| ((*k).to_owned(), json!(v)))
                    .collect();
                let bundle = source.join("bundle.json");
                fs::write(
                    &bundle,
                    serde_json::to_vec(&json!({"files": files})).unwrap(),
                )
                .unwrap();
                let plan = source.join("qualification-plan.json");
                fs::write(&plan, br#"{"expected": {"kind": "values", "data": [1.5]}}"#).unwrap();
                backends.push(json!({"backend": backend, "status": status,
                    "bundle": pin(&bundle), "qualification_plan": pin(&plan)}));
            }
            fs::write(
                warm.join("receipt.json"),
                serde_json::to_vec(&json!({"export": {"backends": backends}})).unwrap(),
            )
            .unwrap();
        }
        let plan = root.join("plan.json");
        fs::write(
            &plan,
            serde_json::to_vec(&json!({"schema": "jope.inference-benchmark/v3",
                "experiment": {"methods_sha256": "m", "source_sha": "s"},
                "binary": {"sha256": "b"}, "inputs": inputs}))
            .unwrap(),
        )
        .unwrap();
        plan
    }

    fn receipt(journal: &Path, name: &str) -> Json {
        load(&journal.join(name).join("receipt.json")).unwrap()
    }

    fn far() -> u64 {
        now_ms() as u64 + 600_000
    }

    #[test]
    fn exact_and_one_bit_away_exports_leave_python_spelled_receipts() {
        let root = scratch();
        let root = root.path();
        let plan = fixture(
            root,
            &[
                ("ok", &[("rust", "completed", &[("model.rs", EXACT)])]),
                (
                    "bits",
                    &[("rust", "completed", &[("model.rs", ONE_BIT_AWAY)])],
                ),
            ],
        );
        let journal = root.join("journal");
        // 1024 rows cost a long optimized compile; the differential covers them.
        let sizes = [1, 32];
        stage(&plan, &journal, far(), &sizes).unwrap();
        assert!(!journal.join("prediction-ok-rust-1024").exists());
        for size in sizes {
            let dir = journal.join(format!("prediction-ok-rust-{size}"));
            let r = receipt(&journal, &format!("prediction-ok-rust-{size}"));
            let Json::Object(members) = &r else { panic!() };
            let keys: Vec<&str> = members.0.iter().map(|(k, _)| k.as_str()).collect();
            assert!(
                members.get("failure").is_none(),
                "{}",
                dumps(&r, Style::REPORT).unwrap()
            );
            assert_eq!(
                keys,
                [
                    "schema",
                    "dataset",
                    "backend",
                    "batch_rows",
                    "status",
                    "plan_identity",
                    "source_sha",
                    "binary_sha256",
                    "input_sha256",
                    "source_qualification",
                    "bundle_sha256",
                    "parity",
                    "prediction",
                    "samples_seconds",
                    "process_wall_seconds",
                    "initialization_and_io_residual_seconds",
                    "commands",
                    "note"
                ]
            );
            assert!(is(index(&r, "status").unwrap(), "completed"));
            assert!(super::super::reporting::py_eq(
                index(&r, "batch_rows").unwrap(),
                &Json::from(size)
            ));
            let Json::Array(samples) = index(&r, "samples_seconds").unwrap() else {
                panic!()
            };
            let samples: Vec<f64> = samples
                .iter()
                .map(|s| match s {
                    Json::Float(f) => *f,
                    other => panic!("{other:?}"),
                })
                .collect();
            assert_eq!(samples.len(), 10);
            let Json::Float(wall) = index(&r, "process_wall_seconds").unwrap() else {
                panic!()
            };
            assert_eq!(
                index(&r, "initialization_and_io_residual_seconds").unwrap(),
                &Json::Float(wall - python_sum(&samples))
            );
            let prediction = index(&r, "prediction").unwrap();
            let predictions = dir.join("predictions.json");
            assert!(is(
                index(prediction, "path").unwrap(),
                &python_path(&predictions)
            ));
            assert!(is(
                index(prediction, "sha256").unwrap(),
                &sha256_hex(&fs::read(&predictions).unwrap())
            ));
            assert_eq!(
                dumps(index(&r, "commands").unwrap(), Style::IDENTITY).unwrap(),
                r#"[["rustc","--edition=2021","-O","latency.rs","-o","latency"],["./latency"]]"#
            );
            let harness = fs::read_to_string(dir.join("latency.rs")).unwrap();
            let row = "vec![Some(String::from_utf8(vec![51,46,50,53]).unwrap()),None,Some(String::from_utf8(vec![195,169]).unwrap())]";
            assert_eq!(
                harness,
                format!("mod model;fn main(){{let rows:Vec<Vec<Option<String>>>=vec![{}];for _ in 0..10 {{let t=std::time::Instant::now();let y=model::predict_raw(&rows).unwrap();let s=t.elapsed().as_secs_f64();println!(\"{{}}\",s);std::fs::write(\"predictions.json\",y).unwrap();}}}}", vec![row; size].join(","))
            );
            let bytes = fs::read(dir.join("receipt.json")).unwrap();
            assert_eq!(
                bytes,
                (dumps(&r, Style::REPORT).unwrap() + "\n").into_bytes()
            );

            let refused = receipt(&journal, &format!("prediction-bits-rust-{size}"));
            assert!(is(index(&refused, "status").unwrap(), "failed"));
            assert!(is(
                index(&refused, "failure").unwrap(),
                "prediction parity failed"
            ));
            assert!(lookup_key(&refused, "samples_seconds").is_none());
        }
    }

    fn lookup_key<'a>(value: &'a Json, key: &str) -> Option<&'a Json> {
        match value {
            Json::Object(o) => o.get(key),
            _ => None,
        }
    }

    #[test]
    fn a_failing_compiler_is_recorded_with_the_generated_inputs() {
        let root = scratch();
        let root = root.path();
        let plan = fixture(
            root,
            &[("c", &[("cpp17", "completed", &[("model.cpp", "not c++")])])],
        );
        let journal = root.join("journal");
        stage(&plan, &journal, far(), &[1, 32]).unwrap();
        let dir = journal.join("prediction-c-cpp17-1");
        let r = receipt(&journal, "prediction-c-cpp17-1");
        // `g++` refuses the source; a host without `g++` records the spawn failure instead.
        let failure = py_str(index(&r, "failure").unwrap());
        let compiled = failure == "compiler exit 1";
        assert!(
            compiled || failure == "[Errno 2] No such file or directory: 'g++'",
            "{failure}"
        );
        // `json.dumps` escapes non-ASCII text.
        let row = format!(r#"["3.25", null, "{}u00e9"]"#, '\\');
        assert_eq!(
            fs::read_to_string(dir.join("rows.json")).unwrap(),
            format!("[{row}]")
        );
        assert_eq!(
            fs::read_to_string(dir.join("latency.py")).unwrap(),
            "import boundary,json,time\nrows=json.load(open(\"rows.json\"))\nfor _ in range(10):\n t=time.perf_counter();y=boundary.predict_raw(rows);s=time.perf_counter()-t;print(s)\njson.dump(y,open(\"predictions.json\",\"w\"),allow_nan=False)\n"
        );
        assert_eq!(dir.join("compile-0.stderr").exists(), compiled);
        assert!(!dir.join("stdout").exists());
        assert_eq!(
            fs::read_to_string(journal.join("prediction-c-cpp17-32/rows.json")).unwrap(),
            format!("[{}]", vec![row; 32].join(", "))
        );
    }

    #[test]
    fn deadline_interruptions_and_changed_bundles_stop_without_replacing() {
        let root = scratch();
        let root = root.path();
        let plan = fixture(
            root,
            &[
                ("d", &[("rust", "completed", &[("model.rs", EXACT)])]),
                ("f", &[("rust", "failed", &[("model.rs", EXACT)])]),
            ],
        );
        let journal = root.join("journal");
        let predictions = || {
            let mut names: Vec<String> = fs::read_dir(&journal)
                .unwrap()
                .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
                .filter(|n| n.starts_with("prediction-"))
                .collect();
            names.sort();
            names
        };
        // A cutoff already passed stops before any attempt.
        run(&plan, &journal, 0).unwrap();
        assert!(predictions().is_empty());
        // Interrupted attempts are never replaced; a failed export is never timed.
        for size in SIZES {
            let dir = journal.join(format!("prediction-d-rust-{size}"));
            fs::create_dir(&dir).unwrap();
            fs::write(dir.join("partial"), b"kept").unwrap();
        }
        run(&plan, &journal, far()).unwrap();
        assert_eq!(predictions().len(), 3);
        for size in SIZES {
            let dir = journal.join(format!("prediction-d-rust-{size}"));
            assert_eq!(fs::read(dir.join("partial")).unwrap(), b"kept");
            assert!(!dir.join("receipt.json").exists());
        }
        // A bundle that no longer matches its export receipt stops the stage.
        fs::write(
            journal.join("warm-d-0/rust/bundle.json"),
            b"{\"files\": {}}",
        )
        .unwrap();
        let error = run(&plan, &journal, far()).unwrap_err();
        assert!(format!("{error:#}").contains("prediction bundle changed"));
    }

    #[test]
    fn process_failures_keep_the_python_text() {
        let dir = scratch();
        let dir = dir.path();
        let missing = run_process(&strings(&["jope-no-such-compiler", "-v"]), dir, far());
        assert_eq!(
            missing.err().unwrap(),
            "[Errno 2] No such file or directory: 'jope-no-such-compiler'"
        );
        // A cutoff already passed leaves the 1 ms floor.
        let late = run_process(&strings(&["sleep", "5"]), dir, 0);
        assert_eq!(
            late.err().unwrap(),
            "Command '['sleep', '5']' timed out after 0.001 seconds"
        );
        let started = Instant::now();
        let slow = run_process(&strings(&["sleep", "5"]), dir, now_ms() as u64 + 300)
            .err()
            .unwrap();
        assert!(started.elapsed() < Duration::from_secs(3));
        assert!(
            slow.starts_with("Command '['sleep', '5']' timed out after 0.")
                && slow.ends_with(" seconds"),
            "{slow}"
        );
        let exit = run_process(
            &strings(&["sh", "-c", "printf out; printf err >&2; exit 3"]),
            dir,
            far(),
        )
        .ok()
        .unwrap();
        assert_eq!(
            (exit.code, exit.stdout.as_slice(), exit.stderr.as_slice()),
            (3, &b"out"[..], &b"err"[..])
        );
        let killed = run_process(&strings(&["sh", "-c", "kill -9 $$"]), dir, far())
            .ok()
            .unwrap();
        assert_eq!(killed.code, -9);
        assert_eq!(timeout(far()).1, "30");
    }

    #[test]
    fn samples_sums_and_parity_follow_python() {
        assert_eq!(python_float(b"0.25"), Ok(0.25));
        assert_eq!(python_float(b" 1_000.5\x0b"), Ok(1000.5));
        assert_eq!(python_float(b"1e-05"), Ok(1e-5));
        assert!(python_float(b"inf").unwrap().is_infinite());
        assert_eq!(
            python_float(b"1__0"),
            Err("could not convert string to float: b'1__0'".into())
        );
        assert_eq!(
            python_float(b"it's"),
            Err("could not convert string to float: b\"it's\"".into())
        );
        assert_eq!(
            python_float(b"\xff\t\\"),
            Err("could not convert string to float: b'\\xff\\t\\\\'".into())
        );
        assert_eq!(
            splitlines(b"a\r\nb\rc\n\nd"),
            [&b"a"[..], b"b", b"c", b"", b"d"]
        );
        assert!(splitlines(b"").is_empty());
        assert_eq!(python_sum(&[0.1; 10]), 1.0);
        assert_ne!([0.1; 10].iter().sum::<f64>(), 1.0);

        let values = |text: &str| -> Json { serde_json::from_str(text).unwrap() };
        let one = Json::Float(1.0);
        assert_eq!(
            parity(&values(r#"{"kind": "values", "data": [1.0, 1]}"#), 2, &one),
            Ok(())
        );
        let refused = Err("prediction parity failed".to_owned());
        assert_eq!(
            parity(&values(r#"{"kind": "values", "data": [1.0]}"#), 2, &one),
            refused
        );
        assert_eq!(
            parity(&values(r#"{"kind": "rows", "data": [1.0]}"#), 1, &one),
            refused
        );
        assert_eq!(
            parity(
                &values(r#"{"kind": "values", "data": [-0.0]}"#),
                1,
                &Json::Float(0.0)
            ),
            refused
        );
        assert_eq!(
            parity(&values("[1.0]"), 1, &one),
            Err("'list' object has no attribute 'get'".into())
        );
        assert_eq!(
            parity(&values(r#"{"kind": "values"}"#), 1, &one),
            Err("'data'".into())
        );
        assert_eq!(
            parity(&values(r#"{"kind": "values", "data": 3}"#), 1, &one),
            Err("object of type 'int' has no len()".into())
        );
        assert_eq!(
            parity(&values(r#"{"kind": "values", "data": ["x"]}"#), 1, &one),
            Err("required argument is not a float".into())
        );
        assert_eq!(
            utf8_error(
                &[0x80],
                &String::from_utf8(vec![0x80]).unwrap_err().utf8_error()
            ),
            "'utf-8' codec can't decode byte 0x80 in position 0: invalid start byte"
        );
        assert_eq!(python_path(Path::new("./a//b/./c/")), "a/b/c");
    }

    /// CPU differential harness against the retired driver. Ignored unless run explicitly with
    /// `JOPE_PREDICTION_DIFFERENTIAL=<plan>|<journal>|<cutoff unix ms>`.
    #[test]
    #[ignore = "differential harness; needs JOPE_PREDICTION_DIFFERENTIAL"]
    fn differential_stage() {
        let Ok(spec) = std::env::var("JOPE_PREDICTION_DIFFERENTIAL") else {
            return;
        };
        let parts: Vec<&str> = spec.splitn(3, '|').collect();
        let outcome = run(
            Path::new(parts[0]),
            Path::new(parts[1]),
            parts[2].parse().unwrap(),
        );
        match outcome {
            Ok(()) => println!("STAGE ok"),
            Err(error) => println!("STAGE error: {error:#}"),
        }
    }
}
