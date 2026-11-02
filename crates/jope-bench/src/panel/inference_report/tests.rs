use super::*;

fn parse(text: &str) -> Json {
    serde_json::from_str(text).unwrap()
}
fn member<'a>(value: &'a Json, key: &str) -> &'a Json {
    index(value, key).unwrap()
}
fn items(value: &Json) -> &Vec<Json> {
    match value {
        Json::Array(items) => items,
        other => panic!("not an array: {other:?}"),
    }
}
fn members_mut(value: &mut Json) -> &mut Object {
    match value {
        Json::Object(members) => members,
        other => panic!("not an object: {other:?}"),
    }
}
fn write_file(path: &Path, text: &str) {
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, text).unwrap();
}
/// Write `text` under `root` and pin it.
fn pin(root: &Path, name: &str, text: &str) -> Json {
    let path = root.join(name);
    write_file(&path, text);
    object([
        ("path", path.to_str().unwrap().into()),
        ("sha256", sha256_hex(text.as_bytes()).into()),
    ])
}
/// The v2 study's ten datasets with placeholder identities.
fn plan() -> Json {
    let inputs: Vec<String> = (0..10)
        .map(|i| format!(r#"{{"id": "{i}", "raw": {{"sha256": "r"}}}}"#))
        .collect();
    parse(&format!(
        r#"{{"schema": "jope.inference-benchmark/v2", "experiment": {{"methods_sha256": "m", "source_sha": "s"}}, "binary": {{"sha256": "b"}}, "inputs": [{}]}}"#,
        inputs.join(", ")
    ))
}
fn summary(plan: &Json, journal: &Path) -> Json {
    summarize(plan, journal).unwrap().0
}

#[test]
fn missing_never_claims_speed() {
    let root = tempfile::tempdir().unwrap();
    let s = summary(&plan(), root.path());
    assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
    let warm = items(member(&s, "missing"))
        .iter()
        .filter(|m| is(member(m, "mode"), "warm"))
        .count();
    assert_eq!(warm, 100);
    assert!(lookup(&s, "speedup").is_none());
}

#[test]
fn controller_failure_before_qualification_is_explicit() {
    let root = tempfile::tempdir().unwrap();
    write_file(
        &root.path().join("launch.json"),
        r#"{"exit_code": 1, "reason": "bounded service failed"}"#,
    );
    let s = summary(&plan(), root.path());
    let controller = member(&s, "controller");
    assert!(is(member(controller, "status"), "failed"));
    assert!(is(
        member(controller, "qualification_status"),
        "not_executed_or_interrupted"
    ));
    assert_eq!(items(member(&s, "prediction")).len(), 90);
    assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
}

#[test]
fn interrupted_direct_and_prediction_attempts_remain_missing() {
    let root = tempfile::tempdir().unwrap();
    fs::create_dir(root.path().join("warm-0-0")).unwrap();
    fs::create_dir(root.path().join("prediction-0-rust-32")).unwrap();
    let s = summary(&plan(), root.path());
    let request = items(member(&s, "missing"))
        .iter()
        .find(|r| {
            is(member(r, "dataset"), "0")
                && is(member(r, "mode"), "warm")
                && py_eq(member(r, "seed"), &ZERO)
        })
        .unwrap();
    assert!(is(member(request, "status"), "unexecuted_or_interrupted"));
    let prediction = items(member(&s, "prediction"))
        .iter()
        .find(|r| {
            is(member(r, "dataset"), "0")
                && is(member(r, "backend"), "rust")
                && py_eq(member(r, "batch_rows"), &Json::Int(32))
        })
        .unwrap();
    assert!(is(member(prediction, "status"), "interrupted"));
    assert_eq!(member(prediction, "accepted"), &Json::Bool(false));
}

#[test]
fn failed_and_wrong_device_remain_failures() {
    let root = tempfile::tempdir().unwrap();
    write_file(
        &root.path().join("warm-0-0/receipt.json"),
        r#"{"schema": "jope.inference-benchmark/v2", "dataset": "0", "seed": 0, "mode": "warm", "success": true, "elapsed_seconds": 1, "devices": ["Cpu", "Cpu", "Cpu"]}"#,
    );
    let s = summary(&plan(), root.path());
    assert_eq!(member(&s, "completed_successfully"), &Json::Int(0));
    assert!(!items(member(&s, "failures")).is_empty());
}

#[test]
fn qualification_flag_without_evidence_refused() {
    let root = tempfile::tempdir().unwrap();
    let cases = [r#"{"pass": true}"#; 8].join(", ");
    write_file(
        &root.path().join("qualification.json"),
        &format!(r#"{{"passed": true, "cases": [{cases}]}}"#),
    );
    let s = summary(&plan(), root.path());
    assert_eq!(member(&s, "qualified"), &Json::Bool(false));
}

#[test]
fn every_planned_slot_reconciles_to_one_outcome() {
    let root = tempfile::tempdir().unwrap();
    let s = summary(&plan(), root.path());
    let census = member(&s, "reconciliation");
    assert_eq!(
        [
            member(census, "planned"),
            member(census, "missing"),
            member(census, "receipts")
        ],
        [&Json::Int(140), &Json::Int(140), &Json::Int(0)]
    );
    assert_eq!(member(census, "reconciled"), &Json::Bool(true));
    assert_eq!(member(census, "failure_or_missing_rate"), &Json::Float(1.0));
    assert!(is(member(member(&s, "census"), "source"), "default-v2"));
}

#[test]
fn unplanned_receipts_are_listed_and_block_the_headline() {
    let root = tempfile::tempdir().unwrap();
    for name in ["warm-0-15", "warm-ghost-0"] {
        write_file(
            &root.path().join(name).join("receipt.json"),
            r#"{"schema": "jope.inference-benchmark/v2", "success": true}"#,
        );
    }
    let s = summary(&plan(), root.path());
    assert_eq!(
        member(member(&s, "reconciliation"), "unplanned_receipts"),
        &parse(r#"["warm-0-15", "warm-ghost-0"]"#)
    );
    assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
}

#[test]
fn census_comes_from_the_frozen_plan() {
    let root = tempfile::tempdir().unwrap();
    let mut plan = plan();
    let first = items(member(&plan, "inputs"))[0].clone();
    members_mut(&mut plan).set("inputs", Json::Array(vec![first]));
    members_mut(&mut plan).set(
        "census",
        parse(r#"{"warm": 2, "cold": 0, "warmup": 0, "profile": 0}"#),
    );
    let s = summary(&plan, root.path());
    assert_eq!(
        (
            member(&s, "expected"),
            member(member(&s, "reconciliation"), "planned")
        ),
        (&Json::Int(2), &Json::Int(2))
    );
    assert!(is(member(member(&s, "census"), "source"), "plan"));
    members_mut(&mut plan).set("census", parse(r#"{"warm": 2}"#));
    let error = summarize(&plan, root.path()).unwrap_err();
    assert!(format!("{error:#}").contains("census"), "{error:#}");
}

#[test]
fn exact_parity_requires_matching_bundle() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    // Python `json.dumps` file spelling; the identity is the compact encoding in file order.
    let bundle = r#"{"manifest": {"backend": "python"}, "files": {"model.py": "content"}}"#;
    let identity =
        sha256_hex(br#"{"manifest":{"backend":"python"},"files":{"model.py":"content"}}"#);
    let values = r#"{"kind": "values", "data": [1.0]}"#;
    let plan =
        format!(r#"{{"expected": {values}, "backend": "python", "bundle_sha256": "{identity}"}}"#);
    let mut pins = object([
        ("bundle", pin(root, "bundle", bundle)),
        ("qualification_plan", pin(root, "qualification_plan", &plan)),
        ("prediction", pin(root, "prediction", values)),
    ]);
    assert!(parity_verified(&pins).unwrap());
    members_mut(&mut pins).set(
        "prediction",
        pin(root, "prediction", r#"{"kind": "values", "data": [2.0]}"#),
    );
    assert!(!parity_verified(&pins).unwrap());
}

#[test]
fn corruption_is_rejected() {
    let root = tempfile::tempdir().unwrap();
    let asset = pin(root.path(), "asset", "original");
    assert!(verified(&asset).unwrap());
    fs::write(root.path().join("asset"), "corrupted").unwrap();
    assert!(!verified(&asset).unwrap());
}

/// Pins whose bundle binding is valid, so only the value comparison can refuse them.
fn bound(root: &Path, expected: &str, actual: &str) -> Json {
    let bundle = r#"{"manifest": {"backend": "rust"}, "files": {"model.rs": "content"}}"#;
    let identity = sha256_hex(br#"{"manifest":{"backend":"rust"},"files":{"model.rs":"content"}}"#);
    let plan = format!(
        r#"{{"expected": {{"kind": "values", "data": {expected}}}, "backend": "rust", "bundle_sha256": "{identity}"}}"#
    );
    object([
        ("bundle", pin(root, "bound-bundle", bundle)),
        ("qualification_plan", pin(root, "bound-plan", &plan)),
        (
            "prediction",
            pin(
                root,
                "bound-prediction",
                &format!(r#"{{"kind": "values", "data": {actual}}}"#),
            ),
        ),
    ])
}

#[test]
fn parity_rejects_wrong_selected_prediction() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let pins = object([
        (
            "qualification_plan",
            pin(
                root,
                "qualification_plan",
                r#"{"expected": {"kind": "values", "data": [1.0]}}"#,
            ),
        ),
        (
            "prediction",
            pin(root, "prediction", r#"{"kind": "values", "data": [2.0]}"#),
        ),
    ]);
    assert!(!parity_verified(&pins).unwrap());
    // With a valid bundle binding, the value comparison alone decides.
    assert!(!parity_verified(&bound(root, "[1.0]", "[2.0]")).unwrap());
    assert!(parity_verified(&bound(root, "[1.0]", "[1.0]")).unwrap());
    assert!(parity_verified(&bound(root, "[1.0]", "[1]")).unwrap());
}

#[test]
fn relative_tolerance_does_not_pass_exact_bit_policy() {
    let root = tempfile::tempdir().unwrap();
    let root = root.path();
    let pins = object([
        (
            "qualification_plan",
            pin(
                root,
                "qualification_plan",
                r#"{"expected": {"kind": "values", "data": [0.0]}}"#,
            ),
        ),
        (
            "prediction",
            pin(root, "prediction", r#"{"kind": "values", "data": [-0.0]}"#),
        ),
    ]);
    assert!(!parity_verified(&pins).unwrap());
    assert!(!parity_verified(&bound(root, "[0.0]", "[-0.0]")).unwrap());
    assert!(!parity_verified(&bound(root, "[1.0]", "[1.0000000000000002]")).unwrap());
    assert!(parity_verified(&bound(root, "[-0.0]", "[-0.0]")).unwrap());
}

/// A complete, qualified single-dataset journal: every link of the acceptance chain holds.
fn qualified_journal(root: &Path) -> Json {
    let cuda = r#"["Cuda(0)", "Cuda(1)", "Cuda(2)"]"#;
    let errors = QUALIFICATION_ERRORS
        .iter()
        .map(|k| format!(r#""{k}": 0.0001"#))
        .collect::<Vec<_>>()
        .join(", ");
    let cases = (0..8)
        .map(|i| format!(r#"{{"case": {i}, "pass": true, "errors": {{{errors}}}}}"#))
        .collect::<Vec<_>>()
        .join(", ");
    write_file(
        &root.join("qualification.json"),
        &format!(
            r#"{{"schema": "jope.inference-benchmark/v2", "passed": true, "plan_identity": "m", "binary_sha256": "b", "source_sha": "s", "cases": [{cases}], "devices": {cuda}, "policy_device": "Cuda(0)"}}"#
        ),
    );
    let artifacts = root.join("artifacts");
    let dump = |value: &Json| dumps(value, Style::IDENTITY).unwrap();
    let mut backends = Vec::new();
    let mut python_pins = None;
    for (name, seconds) in [("python", 0.5), ("rust", 0.75)] {
        let bundle =
            format!(r#"{{"manifest": {{"backend": "{name}"}}, "files": {{"model": "{name}"}}}}"#);
        let bundle_pin = pin(&artifacts, &format!("{name}-bundle.json"), &bundle);
        let identity = sha256_hex(dump(&parse(&bundle)).as_bytes());
        let plan_pin = pin(
            &artifacts,
            &format!("{name}-plan.json"),
            &format!(
                r#"{{"expected": {{"kind": "values", "data": [1.5, -0.0]}}, "backend": "{name}", "bundle_sha256": "{identity}"}}"#
            ),
        );
        let prediction_pin = pin(
            &artifacts,
            &format!("{name}-prediction.json"),
            r#"{"kind": "values", "data": [1.5, -0.0]}"#,
        );
        let file_pin = pin(&artifacts, &format!("{name}-model"), name);
        if name == "python" {
            python_pins = Some((plan_pin.clone(), bundle_pin.clone()));
        }
        backends.push(format!(
                r#"{{"backend": "{name}", "status": "completed", "first_prediction_seconds": {seconds}, "qualification_plan": {}, "prediction": {}, "bundle": {}, "qualification": {{"values": 2, "maximum_absolute_error": 0}}, "files": {{"model": {}}}, "commands": [{{"exit_code": 0}}]}}"#,
                dump(&plan_pin),
                dump(&prediction_pin),
                dump(&bundle_pin),
                dump(&file_pin)
            ));
    }
    backends.push(r#"{"backend": "cpp17", "status": "unsupported"}"#.into());
    let fitted = pin(&artifacts, "fitted.json", "{}");
    write_file(
        &root.join("warm-d-0/receipt.json"),
        &format!(
            r#"{{"schema": "jope.inference-benchmark/v2", "plan_identity": "m", "binary_sha256": "b", "source_sha": "s", "dataset": "d", "seed": 0, "mode": "warm", "input_sha256": "r", "selected_origin": "neural_policy", "ranking_backend": "candle_cuda_float32", "devices": {cuda}, "policy_device": "Cuda(0)", "selected_program": "(pipe X (ridge))", "success": true, "elapsed_seconds": 2.5, "export": {{"validation_in_timer": true, "recipe": {{"original_proposal": "(pipe X (ridge))"}}, "fitted_output": {}, "backends": [{}]}}}}"#,
            dump(&fitted),
            backends.join(", ")
        ),
    );
    let (plan_pin, bundle_pin) = python_pins.unwrap();
    for (rows, data) in [(1, "[1.5]"), (32, "[1.5]")] {
        let prediction = pin(
            &artifacts,
            &format!("batch-{rows}.json"),
            &format!(r#"{{"kind": "values", "data": {data}}}"#),
        );
        write_file(
            &root.join(format!("prediction-d-python-{rows}/receipt.json")),
            &format!(
                r#"{{"dataset": "d", "backend": "python", "batch_rows": {rows}, "source_sha": "s", "plan_identity": "m", "binary_sha256": "b", "input_sha256": "r", "source_qualification": {}, "prediction": {}, "parity": "exact_bits", "bundle_sha256": "{}", "samples_seconds": [{}], "status": "completed"}}"#,
                dump(&plan_pin),
                dump(&prediction),
                py_str(member(&bundle_pin, "sha256")),
                ["0.001"; 10].join(", ")
            ),
        );
    }
    parse(
        r#"{"schema": "jope.inference-benchmark/v2", "experiment": {"methods_sha256": "m", "source_sha": "s"}, "binary": {"sha256": "b"}, "inputs": [{"id": "d", "raw": {"sha256": "r"}}], "census": {"warm": 1, "cold": 0, "warmup": 0, "profile": 0}}"#,
    )
}

#[test]
fn complete_qualified_evidence_releases_the_headline() {
    let root = tempfile::tempdir().unwrap();
    let plan = qualified_journal(root.path());
    let (s, records) = summarize(&plan, root.path()).unwrap();
    assert_eq!(member(&s, "qualified"), &Json::Bool(true));
    assert!(is(
        member(member(&s, "controller"), "qualification_status"),
        "passed"
    ));
    assert_eq!(member(&s, "publication_headline"), &Json::Bool(true));
    assert_eq!(member(&s, "completed_successfully"), &Json::Int(1));
    assert_eq!(member(&s, "failures"), &parse("[]"));
    assert_eq!(member(&records[0], "accepted"), &Json::Bool(true));
    assert_eq!(
        member(&s, "warm_cumulative_backend_milestones"),
        &parse(
            r#"[{"dataset": "d", "backend": "python", "n": 1, "median_seconds": 0.5, "p95_seconds": 0.5, "mean_seconds": 0.5},
                    {"dataset": "d", "backend": "rust", "n": 1, "median_seconds": 0.75, "p95_seconds": 0.75, "mean_seconds": 0.75}]"#
        )
    );
    assert_eq!(
        member(member(&s, "reconciliation"), "failure_or_missing_rate"),
        &Json::Float(0.0)
    );
    let prediction = items(member(&s, "prediction"));
    assert_eq!(prediction.len(), 9);
    // One exact batch of one row is accepted; 32 rows claimed with one value is not.
    assert_eq!(member(&prediction[0], "accepted"), &Json::Bool(true));
    assert!(is(member(&prediction[1], "status"), "completed"));
    assert_eq!(member(&prediction[1], "accepted"), &Json::Bool(false));
    let cpp = prediction
        .iter()
        .filter(|p| is(member(p, "backend"), "cpp17"))
        .collect::<Vec<_>>();
    assert_eq!(cpp.len(), 3);
    assert!(cpp.iter().all(|p| is(member(p, "status"), "unsupported")));

    // Any unplanned direct receipt withholds the headline.
    write_file(&root.path().join("warmup-d-9/receipt.json"), "{}");
    let s = summary(&plan, root.path());
    assert_eq!(member(&s, "publication_headline"), &Json::Bool(false));
    assert_eq!(
        member(member(&s, "reconciliation"), "unplanned_receipts"),
        &parse(r#"["warmup-d-9"]"#)
    );
}

#[test]
fn float_spelling_matches_python_repr() {
    for (value, spelling) in [
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (0.1, "0.1"),
        (1.5, "1.5"),
        (100.0, "100.0"),
        (1.0 / 3.0, "0.3333333333333333"),
        (2.0 / 3.0, "0.6666666666666666"),
        (0.1 + 0.2, "0.30000000000000004"),
        (1e-4, "0.0001"),
        (0.00012, "0.00012"),
        (1e-5, "1e-05"),
        (-2.5e-7, "-2.5e-07"),
        (1e15, "1000000000000000.0"),
        (1234567890123456.8, "1234567890123456.8"),
        (1e16, "1e+16"),
        (12345678901234567.0, "1.2345678901234568e+16"),
        (1e22, "1e+22"),
        (1e23, "1e+23"),
        (5e-324, "5e-324"),
        (f64::MAX, "1.7976931348623157e+308"),
    ] {
        assert_eq!(float_repr(value), spelling, "{value:?}");
    }
    assert_eq!(
        py_str(&parse(
            r#"[1, "a'b", null, true, 1.0, {"k": "v"}, "\u00e9\u0085"]"#
        )),
        r#"[1, "a'b", None, True, 1.0, {'k': 'v'}, 'é\x85']"#
    );
}

#[test]
fn statistics_are_exact_like_the_python_module() {
    let values = |text: &str| items(&parse(text)).clone();
    // The exact mean of the three doubles rounds to 0.2; summing in floats gives 0.2000…04.
    assert_eq!(mean(&values("[0.1, 0.2, 0.3]")).unwrap(), Json::Float(0.2));
    assert_ne!((0.1 + 0.2 + 0.3) / 3.0, 0.2);
    assert_eq!(mean(&values("[1e308, 1e308]")).unwrap(), Json::Float(1e308));
    assert_eq!(mean(&values("[1, 2]")).unwrap(), Json::Float(1.5));
    assert_eq!(mean(&values("[1, 3]")).unwrap(), Json::Int(2));
    assert_eq!(mean(&values("[1, 2.0]")).unwrap(), Json::Float(1.5));
    assert_eq!(mean(&values("[true, true]")).unwrap(), Json::Int(1));
    assert_eq!(mean(&values("[-0.0]")).unwrap(), Json::Float(0.0));
    assert_eq!(mean(&values("[5e-324, 0.0]")).unwrap(), Json::Float(0.0));
    assert_eq!(
        mean(&values("[5e-324, 1e-323]")).unwrap(),
        Json::Float(1e-323)
    );
    assert_eq!(
        mean(&values("[-5e-324, -1e-323]")).unwrap(),
        Json::Float(-1e-323)
    );
    assert_eq!(
        distribution(&values("[3, 1, 2, 4]")).unwrap(),
        parse(r#"{"n": 4, "median_seconds": 2.5, "p95_seconds": 4, "mean_seconds": 2.5}"#)
    );
    assert_eq!(
        distribution(&values("[3, 1]")).unwrap(),
        parse(r#"{"n": 2, "median_seconds": 2.0, "p95_seconds": 3, "mean_seconds": 2}"#)
    );
    assert_eq!(
        distribution(&values("[]")).unwrap(),
        parse(r#"{"n": 0, "median_seconds": null, "p95_seconds": null, "mean_seconds": null}"#)
    );
}

#[test]
fn written_report_keeps_python_spelling() {
    let root = tempfile::tempdir().unwrap();
    let journal = root.path().join("journal");
    let output = root.path().join("output");
    write_file(
        &journal.join("launch.json"),
        r#"{"exit_code": 0, "t": 1e-05, "big": 1e16, "neg": -0.0, "n": 12345678901234567890, "s": "tab\there é"}"#,
    );
    write_file(
        &journal.join("window.json"),
        r#"{"started": 1.5, "list": [], "obj": {}}"#,
    );
    write_file(
        &journal.join("warm-é1-0/receipt.json"),
        r#"{"mode": "warm", "dataset": "é1", "seed": 0, "elapsed_seconds": 2.5, "failure": "bad, \"quoted\"\nline", "success": true}"#,
    );
    let plan = r#"{"schema": "jope.inference-benchmark/v2", "experiment": {"methods_sha256": "m", "source_sha": "s"}, "binary": {"sha256": "b"}, "inputs": [{"id": "é1", "raw": {"sha256": "r"}}], "census": {"warm": 1, "cold": 0, "warmup": 0, "profile": 0}}"#;
    write(plan.as_bytes(), &journal, &output).unwrap();

    let prediction = PREDICTION_BACKENDS
            .iter()
            .flat_map(|backend| {
                BATCH_ROWS.iter().map(move |rows| {
                    format!(
                        "    {{\n      \"dataset\": \"\\u00e91\",\n      \"backend\": \"{backend}\",\n      \"batch_rows\": {rows},\n      \"status\": \"unexecuted\",\n      \"accepted\": false\n    }}"
                    )
                })
            })
            .collect::<Vec<_>>()
            .join(",\n");
    let expected = format!(
        r#"{{
  "schema": "jope.inference-benchmark/v2",
  "controller": {{
    "status": "completed",
    "launch": {{
      "exit_code": 0,
      "t": 1e-05,
      "big": 1e+16,
      "neg": -0.0,
      "n": 12345678901234567890,
      "s": "tab\there \u00e9"
    }},
    "qualification_status": "not_executed_or_interrupted"
  }},
  "qualified": false,
  "qualification": {{}},
  "publication_headline": false,
  "completed_successfully": 0,
  "expected": 1,
  "census": {{
    "modes": {{
      "warm": 1,
      "cold": 0,
      "warmup": 0,
      "profile": 0
    }},
    "source": "plan"
  }},
  "reconciliation": {{
    "planned": 1,
    "receipts": 1,
    "accepted": 0,
    "rejected": 1,
    "missing": 0,
    "unplanned_receipts": [],
    "reconciled": true,
    "failure_or_missing_rate": 1.0
  }},
  "missing": [],
  "failures": [
    {{
      "dataset": "\u00e91",
      "mode": "warm",
      "seed": 0,
      "failure": "bad, \"quoted\"\nline"
    }}
  ],
  "warm_cumulative_backend_milestones": [],
  "cold_observations": [],
  "full_gp": [],
  "prediction": [
{prediction}
  ],
  "window": {{
    "started": 1.5,
    "list": [],
    "obj": {{}}
  }},
  "warm_complete_request_distribution": {{
    "n": 0,
    "median_seconds": null,
    "p95_seconds": null,
    "mean_seconds": null
  }},
  "boundary": "{BOUNDARY}"
}}
"#
    );
    assert_eq!(
        fs::read_to_string(output.join("summary.json")).unwrap(),
        expected
    );
    assert_eq!(
            fs::read_to_string(output.join("requests.csv")).unwrap(),
            "dataset,mode,seed,elapsed_seconds,accepted,failure\r\né1,warm,0,2.5,False,\"bad, \"\"quoted\"\"\nline\"\r\n"
        );
    let methodology = fs::read_to_string(output.join("methodology.md")).unwrap();
    assert!(methodology.starts_with(&format!("# GPU timing study\n\n{BOUNDARY}\n\n")));
    assert!(methodology.contains("Qualified: False. Completed warm Python and Rust paths: 0/1. Planned direct slots: 1, accepted 0, missing 0, unplanned receipts 0. Missing attempts are unexecuted, never zero latency."));
    assert!(methodology.ends_with("containing I/O and teardown.\n"));
    // An unsupported contract is refused before any output changes.
    let v1 = plan.replace("inference-benchmark/v2", "inference-benchmark/v1");
    let error = write(v1.as_bytes(), &journal, &root.path().join("v1")).unwrap_err();
    assert!(format!("{error:#}").contains("receipt version unsupported"));
    assert!(!root.path().join("v1").exists());
}
