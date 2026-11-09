use super::runner::{parity, utf8_error};
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
    let scratch = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/fitted-inference-tests");
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
