// Extracted from inference_contract_tests.rs; test behavior is unchanged.
use super::*;

pub(crate) fn run(directory: &Path, argv: &[String]) -> std::process::Output {
    let output = Command::new(&argv[0])
        .args(&argv[1..])
        .current_dir(directory)
        .env("OMP_NUM_THREADS", "1")
        .env("OPENBLAS_NUM_THREADS", "1")
        .output()
        .unwrap_or_else(|error| panic!("required tool {} unavailable: {error}", argv[0]));
    assert!(
        output.status.success(),
        "{argv:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

pub(crate) fn rust_literal(rows: &[Vec<Option<String>>]) -> String {
    format!(
        "vec![{}]",
        rows.iter()
            .map(|row| format!(
                "vec![{}]",
                row.iter()
                    .map(|value| value.as_ref().map_or_else(
                        || "None".to_owned(),
                        |text| format!("Some({text:?}.to_owned())")
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    )
}

/// Execute every case through one emitted bundle. Returns one entry per case:
/// `Some({"ok":predictions})`, `Some({"err":type})`, or `None` if not representable.
pub(crate) fn execute(
    directory: &Path,
    artifact: &FittedRustArtifact,
    bundle: &PublicBundle,
    cases: &[Case],
) -> Vec<Option<Value>> {
    std::fs::create_dir_all(directory).unwrap();
    for (name, text) in &bundle.files {
        std::fs::write(directory.join(name), text).unwrap();
    }
    let backend = bundle.manifest.backend;
    if backend == Backend::Rust {
        let indices: Vec<usize> = (0..cases.len())
            .filter(|i| cases[*i].typed.is_some())
            .collect();
        let literal = format!(
            "vec![{}]",
            indices
                .iter()
                .map(|i| rust_literal(cases[*i].typed.as_ref().unwrap()))
                .collect::<Vec<_>>()
                .join(",")
        );
        std::fs::write(
            directory.join("driver.rs"),
            format!(
                "mod model;\nfn main() {{\n let cases: Vec<Vec<Vec<Option<String>>>> = {literal};\n let mut out = String::from(\"[\");\n for (i, case) in cases.iter().enumerate() {{\n  if i > 0 {{ out.push(','); }}\n  match model::predict_raw(case) {{\n   Ok(text) => {{ out.push_str(\"{{\\\"ok\\\":\"); out.push_str(&text); out.push('}}'); }}\n   Err(_) => out.push_str(\"{{\\\"err\\\":\\\"Err\\\"}}\"),\n  }}\n }}\n out.push(']');\n println!(\"{{out}}\");\n}}\n"
            ),
        )
        .unwrap();
        run(
            directory,
            &["rustc", "--edition=2021", "-O", "driver.rs", "-o", "driver"].map(str::to_owned),
        );
        let output = run(directory, &["./driver".to_owned()]);
        let values: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(values.len(), indices.len());
        let mut result = vec![None; cases.len()];
        for (index, value) in indices.into_iter().zip(values) {
            result[index] = Some(value);
        }
        return result;
    }
    let text = format!(
        "[{}]",
        cases
            .iter()
            .map(|case| case.text.as_str())
            .collect::<Vec<_>>()
            .join(",")
    );
    std::fs::write(directory.join("contract-cases.json"), text).unwrap();
    let argv: Vec<String> = match backend {
        Backend::JavaScript => {
            std::fs::write(directory.join("contract-driver.js"), "const model=require('./model.js'); const fs=require('fs');\nconst cases=JSON.parse(fs.readFileSync('contract-cases.json','utf8'));\nconst out=cases.map(c=>{ try { return {ok:model.predictRaw(c)}; } catch (e) { return {err:e.constructor.name}; } });\nprocess.stdout.write(JSON.stringify(out)+'\\n');\n").unwrap();
            vec!["node".into(), "contract-driver.js".into()]
        }
        _ => {
            if matches!(backend, Backend::Cpp17 | Backend::Cpp20) {
                // Reuse the exact qualification compile argv; only its prediction step differs.
                let plan =
                    qualification_plan(artifact, bundle, &[vec![Some("1".into()), None]]).unwrap();
                for argv in &plan.commands[..plan.commands.len() - 1] {
                    run(directory, argv);
                }
            }
            let module = if backend == Backend::Python {
                "model"
            } else {
                "boundary"
            };
            std::fs::write(directory.join("contract_driver.py"), format!("import json, sys\nimport {module} as target\nwith open('contract-cases.json', encoding='utf-8') as stream:\n    cases = json.load(stream)\nout = []\nfor case in cases:\n    try:\n        out.append({{'ok': target.predict_raw(case)}})\n    except Exception as error:\n        out.append({{'err': type(error).__name__}})\nsys.stdout.write(json.dumps(out, ensure_ascii=False, allow_nan=False) + '\\n')\n")).unwrap();
            vec!["python3".into(), "contract_driver.py".into()]
        }
    };
    let output = run(directory, &argv);
    let values: Vec<Value> = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(values.len(), cases.len());
    values.into_iter().map(Some).collect()
}

/// Compare one emitted result to the native one under the backend's declared policy.
pub(crate) fn compare(
    policy: ParityPolicy,
    actual: &Predictions,
    expected: &Predictions,
) -> std::result::Result<(), String> {
    let (a, e) = match (actual, expected) {
        (Predictions::Values(a), Predictions::Values(e)) => (a.clone(), e.clone()),
        (
            Predictions::Probabilities {
                classes: ac,
                values: av,
            },
            Predictions::Probabilities {
                classes: ec,
                values: ev,
            },
        ) => {
            if ac != ec || av.len() != ev.len() || av.iter().any(|r| r.len() != ac.len()) {
                return Err("class order or matrix shape differs".into());
            }
            (
                av.iter().flatten().copied().collect(),
                ev.iter().flatten().copied().collect(),
            )
        }
        _ => return Err("prediction kind differs".into()),
    };
    if a.len() != e.len() {
        return Err(format!("cardinality {} != native {}", a.len(), e.len()));
    }
    for (index, (a, e)) in a.iter().zip(&e).enumerate() {
        let ok = match policy {
            ParityPolicy::ExactBits => a.to_bits() == e.to_bits(),
            ParityPolicy::AbsoluteRelative { absolute, relative } => {
                a.is_finite() && (a - e).abs() <= absolute + relative * e.abs()
            }
        };
        if !ok {
            return Err(format!("value {index}: {a:?} != native {e:?}"));
        }
    }
    Ok(())
}

/// Per-backend outcome counts, so a pass cannot be vacuous.
pub(crate) struct Tally {
    pub backend: Backend,
    pub accepted: usize,
    pub refused: usize,
}

/// Differential check of one fitted fixture across every backend it exports to.
/// Returns human-readable divergences (an empty vector is a pass) and outcome counts.
pub(crate) fn divergences(
    label: &str,
    artifact: &FittedRustArtifact,
    cases: &[Case],
    root: &Path,
) -> (Vec<String>, Vec<Tally>) {
    let view = FittedInferenceView::registered(artifact).unwrap();
    let native: Vec<Option<Result<Predictions>>> = cases
        .iter()
        .map(|case| {
            case.typed
                .as_ref()
                .map(|rows| view.predict_raw(&columns(), rows))
        })
        .collect();
    let mut found = Vec::new();
    let mut tallies = Vec::new();
    for backend in [
        Backend::Python,
        Backend::Rust,
        Backend::JavaScript,
        Backend::Cpp17,
        Backend::Cpp20,
    ] {
        let ExportOutcome::Emitted { bundle } = export_inference(&view, backend).unwrap() else {
            continue;
        };
        let directory = root.join(format!("{label}-{backend:?}"));
        let results = execute(&directory, artifact, &bundle, cases);
        let mut tally = Tally {
            backend,
            accepted: 0,
            refused: 0,
        };
        for (index, case) in cases.iter().enumerate() {
            let Some(result) = &results[index] else {
                continue;
            };
            if result.get("ok").is_some() {
                tally.accepted += 1;
            } else {
                tally.refused += 1;
            }
            let tag = format!("{label}/{backend:?}/{}", case.name);
            match (&native[index], result.get("ok"), result.get("err")) {
                (Some(Ok(expected)), Some(ok), _) => {
                    match serde_json::from_value::<Predictions>(ok.clone()) {
                        Ok(actual) => {
                            if let Err(detail) = compare(backend.parity_policy(), &actual, expected)
                            {
                                found.push(format!("{tag}: {detail}"));
                            }
                        }
                        Err(error) => found.push(format!("{tag}: undecodable output {error}")),
                    }
                    if case.expect_refusal {
                        found.push(format!("{tag}: native accepted a case declared malformed"));
                    }
                }
                (Some(Ok(_)), None, Some(err)) => found.push(format!(
                    "{tag}: backend refused ({err}) but native predicted"
                )),
                (Some(Err(_)) | None, Some(ok), _) => {
                    found.push(format!("{tag}: backend accepted malformed input: {ok}"))
                }
                (Some(Err(error)), None, Some(_)) => {
                    if !case.expect_refusal {
                        found.push(format!(
                            "{tag}: native refused a case declared valid: {error:#}"
                        ));
                    }
                }
                (None, None, Some(_)) => {}
                _ => found.push(format!("{tag}: unrecognized backend result {result}")),
            }
        }
        for entry in std::fs::read_dir(&directory).unwrap() {
            let name = entry.unwrap().file_name();
            assert_ne!(
                name, "PWNED",
                "{label}/{backend:?}: hostile string executed"
            );
        }
        tallies.push(tally);
    }
    (found, tallies)
}

/// Every applicable case produced exactly the declared accept/refuse outcome count.
pub(crate) fn assert_tallies(cases: &[Case], tallies: &[Tally], backends: usize) {
    assert_eq!(tallies.len(), backends);
    for tally in tallies {
        let applicable = |case: &&Case| tally.backend != Backend::Rust || case.typed.is_some();
        let valid = cases
            .iter()
            .filter(applicable)
            .filter(|c| !c.expect_refusal)
            .count();
        let malformed = cases
            .iter()
            .filter(applicable)
            .filter(|c| c.expect_refusal)
            .count();
        assert!(valid > 0 && malformed > 0);
        assert_eq!(
            (tally.accepted, tally.refused),
            (valid, malformed),
            "{:?}",
            tally.backend
        );
    }
}

pub(crate) fn scratch() -> tempfile::TempDir {
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-inference-tests");
    std::fs::create_dir_all(&scratch).unwrap();
    tempfile::tempdir_in(scratch).unwrap()
}
