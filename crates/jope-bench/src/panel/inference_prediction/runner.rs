use super::*;

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
pub(crate) fn run_process(args: &[String], dir: &Path, cutoff_ms: u64) -> Attempt<Finished> {
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

pub(crate) fn strings(items: &[&str]) -> Vec<String> {
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
pub(crate) fn attempt(
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
pub(crate) fn run(plan: &Path, journal: &Path, cutoff_ms: u64) -> Result<()> {
    stage(plan, journal, cutoff_ms, &SIZES)
}

/// [`run`] over the given batch sizes; unit tests skip the 1024-row compile.
pub(crate) fn stage(plan: &Path, journal: &Path, cutoff_ms: u64, sizes: &[usize]) -> Result<()> {
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
                super::super::reporting::py_eq(
                    &identity,
                    index(index(&entry, "bundle")?, "sha256")?
                ),
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
