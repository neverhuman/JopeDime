use super::summary::summarize;
use super::*;

#[path = "tests/tests_01.rs"]
mod tests_01;
#[path = "tests/tests_02.rs"]
mod tests_02;

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
