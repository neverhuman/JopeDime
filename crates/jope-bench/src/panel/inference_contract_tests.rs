//! W9-T03/T04 raw-input and hostile-string contract of the emitted public backends.
//!
//! Every case runs through the actual emitted source (Python, Rust, JavaScript and the
//! C++17/C++20 library behind its declared Python typed-table host) and is compared
//! with the native fitted predictor on the same raw rows. Fixtures are fitted once
//! through the registered Rust fitter and only read afterwards; no backend refits.
//! This is fixture proof of the emitted boundary, not a released-package qualification.
use super::super::data::raw_table;
use super::super::methods::{fit_rust, rust_methods, FittedRustArtifact, RustFitRequest};
use super::*;
use feat_math::advanced_v3::{DataSchema, RowSemantics};
use jope_pipeline::experiment::{Column, ColumnKind as Kind};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;

/// Present category strings that must stay data in every backend. Several imitate
/// shell, Python, JavaScript and Rust syntax, template markers or the missing sentinel.
pub(crate) const HOSTILE_CATEGORIES: [&str; 18] = [
    "NA",
    "null",
    "",
    "na",
    "line\n\"quoted\"\u{2028}",
    "__PREDICT____REGRESSION____CLASSES_JSON__",
    "\"; import os; os.system('touch PWNED') #",
    "'); require('child_process').execSync('touch PWNED'); ('",
    "\\\"}]); panic!(\"x\"); //",
    "\u{0}MISSING",
    "jope.category.v1:\"NA\"",
    "\u{1f642} emoji",
    "\u{7f}\u{1b}[31m",
    "*/ /*",
    "${HOME}",
    "$(touch PWNED)",
    "`touch PWNED`",
    "\\u0041",
];

pub(crate) fn columns() -> Vec<Column> {
    vec![
        Column {
            name: "amount".into(),
            kind: Kind::Numeric,
        },
        Column {
            name: "country".into(),
            kind: Kind::Categorical,
        },
    ]
}

/// Fit one registered recipe on explicit synthetic rows. `amount_missing_everywhere`
/// makes the numeric training column entirely null.
pub(crate) fn fit(
    task: Task,
    family: &str,
    categories: &[Option<String>],
    repeats: &dyn Fn(usize) -> usize,
    amount_missing_everywhere: bool,
) -> Result<FittedRustArtifact> {
    let method = rust_methods(task)?
        .into_iter()
        .find(|method| method.family == family)
        .context("family")?;
    fit_method(task, method, categories, repeats, amount_missing_everywhere)
}

pub(crate) fn fit_method(
    task: Task,
    method: super::super::methods::RustMethod,
    categories: &[Option<String>],
    repeats: &dyn Fn(usize) -> usize,
    amount_missing_everywhere: bool,
) -> Result<FittedRustArtifact> {
    let mut rows = Vec::new();
    let mut targets = Vec::new();
    let mut index = 0usize;
    for (position, category) in categories.iter().enumerate() {
        for _ in 0..repeats(position) {
            let amount = if amount_missing_everywhere || index % 5 == 0 {
                None
            } else {
                Some(format!("{}", index as f64 * 0.75 - 4.0))
            };
            rows.push(vec![amount, category.clone()]);
            targets.push(match task {
                Task::Regression => (index % 7) as f64 * 1.5 + position as f64 * 0.25,
                Task::Binary => ((position + index) % 2) as f64,
                _ => (index % 3) as f64,
            });
            index += 1;
        }
    }
    let classes: Vec<String> = match task {
        Task::Regression => vec![],
        Task::Binary => vec![
            "no\"; os.system('touch PWNED')".into(),
            "yes\u{2028}".into(),
        ],
        _ => vec!["zebra".into(), "`touch PWNED`".into(), "${x}".into()],
    };
    let request = RustFitRequest {
        task,
        schema: DataSchema {
            columns: vec![
                ColumnSpec::numeric("amount"),
                ColumnSpec::categorical("country"),
            ],
            row_semantics: RowSemantics::Iid,
        },
        classes,
        train: raw_table(&columns(), &rows)?,
        targets,
        predict: raw_table(&columns(), &[vec![Some("1".into()), None]])?,
        method,
    };
    Ok(fit_rust(&request)?.artifact)
}

fn hostile_categories() -> Vec<Option<String>> {
    let mut categories: Vec<Option<String>> = vec![None];
    categories.extend(
        HOSTILE_CATEGORIES
            .iter()
            .map(|value| Some((*value).to_owned())),
    );
    categories
}

pub(crate) fn hostile_fixture(task: Task, family: &str) -> FittedRustArtifact {
    fit(task, family, &hostile_categories(), &|_| 3, false).unwrap()
}

fn row(values: &[Option<&str>]) -> Value {
    Value::Array(
        values
            .iter()
            .map(|value| value.map_or(Value::Null, |text| Value::String(text.into())))
            .collect(),
    )
}

/// Raw JSON cases. `None` typed rows mean the case is not representable through the
/// typed Rust/native API (JSON type confusion); every dynamic backend must refuse it.
pub(crate) struct Case {
    pub name: String,
    pub text: String,
    pub typed: Option<Vec<Vec<Option<String>>>>,
    pub expect_refusal: bool,
}

fn typed(value: &Value) -> Option<Vec<Vec<Option<String>>>> {
    value
        .as_array()?
        .iter()
        .map(|row| {
            row.as_array()?
                .iter()
                .map(|cell| match cell {
                    Value::Null => Some(None),
                    Value::String(text) => Some(Some(text.clone())),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
        })
        .collect()
}

fn case(name: &str, rows: Value, refuse: bool) -> Case {
    Case {
        name: name.into(),
        text: serde_json::to_string(&rows).unwrap(),
        typed: typed(&rows),
        expect_refusal: refuse,
    }
}

pub(crate) fn raw_cases(categories: &[&str]) -> Vec<Case> {
    let n = |amount: &str| row(&[Some(amount), Some("NA")]);
    let mut cases = vec![
        case("all-missing-row", json!([[null, null]]), false),
        case(
            "all-missing-column-batch",
            json!([[null, "NA"], [null, null], [null, "unseen"]]),
            false,
        ),
        case(
            "unseen-category",
            json!([["5", "never-seen-category"]]),
            false,
        ),
        case(
            "present-null-like-strings-are-not-missing",
            json!([
                ["3.5", "NA"],
                ["3.5", "null"],
                ["3.5", ""],
                ["3.5", "na"],
                ["3.5", null]
            ]),
            false,
        ),
        case(
            "numeric-lexical-forms",
            Value::Array(
                [
                    " 3.5\t",
                    "\u{a0}2",
                    "+1e2",
                    ".5",
                    "5.",
                    "-0.0",
                    "1e-320",
                    "1E3",
                    "\u{2028}3\u{2029}",
                    "\u{3000}4",
                    "0000012",
                    "-.5e-3",
                    "+.5",
                    "1.e5",
                    "4.9e-324",
                    "1.7976931348623157e308",
                    "-1e308",
                    "1e-400",
                ]
                .iter()
                .map(|value| n(value))
                .collect(),
            ),
            false,
        ),
        case(
            "hostile-present-categories",
            Value::Array(
                categories
                    .iter()
                    .map(|value| row(&[Some("1.25"), Some(value)]))
                    .collect(),
            ),
            false,
        ),
        case("long-category", json!([["2", "x".repeat(100_000)]]), false),
        case(
            "cardinality-fifty-rows",
            Value::Array(
                (0..50)
                    .map(|i| {
                        row(&[
                            if i % 4 == 0 {
                                None
                            } else {
                                Some(["1", "-2.5", "7e1"][i % 3])
                            },
                            if i % 6 == 0 {
                                None
                            } else {
                                Some(categories[i % categories.len()])
                            },
                        ])
                    })
                    .collect(),
            ),
            false,
        ),
        case("empty-input", json!([]), false),
    ];
    for (name, value) in [
        ("non-number", "abc"),
        ("nan", "NaN"),
        ("lower-nan", "nan"),
        ("inf", "inf"),
        ("negative-infinity", "-Infinity"),
        ("overflow", "1e400"),
        ("empty-string-numeric", ""),
        ("blank-numeric", "   "),
        ("underscore", "1_000"),
        ("hex", "0x10"),
        ("comma-decimal", "1,5"),
        ("arabic-indic-digit", "\u{661}"),
        ("dangling-exponent", "1e"),
        ("bare-exponent", "e5"),
        ("double-sign", "+-1"),
        ("two-points", "1.2.3"),
        ("unit-separator-padding", "\u{1c}5"),
        ("zero-width-space", "\u{200b}5"),
    ] {
        cases.push(case(
            &format!("malformed-numeric-{name}"),
            json!([n(value)]),
            true,
        ));
    }
    cases.extend([
        case("narrow-row", json!([["1"]]), true),
        case("extra-column", json!([["1", "NA", "extra"]]), true),
        case("empty-row", json!([[]]), true),
        case(
            "one-malformed-row-in-batch-has-no-partial-output",
            json!([["1", "NA"], ["bad", "NA"], ["2", "NA"]]),
            true,
        ),
        case("json-number-for-numeric", json!([[5, "NA"]]), true),
        case("json-number-for-category", json!([["5", 7]]), true),
        case("boolean-numeric", json!([[true, "NA"]]), true),
        case("boolean-category", json!([[null, false]]), true),
        case("nested-cell", json!([[["5"], "NA"]]), true),
        case(
            "named-record-row",
            json!([{"amount": "5", "country": "NA"}]),
            true,
        ),
        case("string-row", json!(["5,NA"]), true),
        case("object-batch", json!({"rows": []}), true),
        case("null-batch", json!(null), true),
        case("null-row", json!([null]), true),
    ]);
    // Not representable in serde_json::Value or Rust strings: a lone UTF-16 surrogate.
    cases.push(Case {
        name: "lone-surrogate-category".into(),
        text: "[[\"5\",\"\\ud800\"]]".into(),
        typed: None,
        expect_refusal: true,
    });
    cases
}

fn run(directory: &Path, argv: &[String]) -> std::process::Output {
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

fn rust_literal(rows: &[Vec<Option<String>>]) -> String {
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
fn assert_tallies(cases: &[Case], tallies: &[Tally], backends: usize) {
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

fn scratch() -> tempfile::TempDir {
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-inference-tests");
    std::fs::create_dir_all(&scratch).unwrap();
    tempfile::tempdir_in(scratch).unwrap()
}

#[test]
fn emitted_backends_match_native_on_raw_edge_inputs_and_refuse_malformed_rows() {
    let root = scratch();
    let cases = raw_cases(&HOSTILE_CATEGORIES);
    let mut all = Vec::new();
    // ridge/logit: all five backends; multiclass: no scalar C++ ABI; branch heads: Python/Rust.
    for (task, family, backends) in [
        (Task::Regression, "ridge", 5),
        (Task::Binary, "logit_ridge", 5),
        (Task::Multiclass, "multiclass_ridge", 3),
        (Task::Binary, "tab_boost", 2),
        (Task::Regression, "extra_trees", 2),
    ] {
        let artifact = hostile_fixture(task, family);
        let before = artifact.bytes().unwrap();
        let (found, tallies) = divergences(
            &format!("{task:?}-{family}"),
            &artifact,
            &cases,
            root.path(),
        );
        all.extend(found);
        assert_tallies(&cases, &tallies, backends);
        // Execution through every backend is read-only for the fitted subject.
        assert_eq!(artifact.bytes().unwrap(), before);
    }
    assert!(
        all.is_empty(),
        "{} divergences:\n{}",
        all.len(),
        all.join("\n")
    );
}

/// Positive control for the comparator itself: one changed fitted constant in the
/// emitted Python source, or one dropped output row, is reported as a divergence.
#[test]
fn raw_input_comparator_detects_changed_constants_and_truncated_outputs() {
    let root = scratch();
    let artifact = hostile_fixture(Task::Regression, "ridge");
    let view = FittedInferenceView::registered(&artifact).unwrap();
    let ExportOutcome::Emitted { bundle } = export_inference(&view, Backend::Python).unwrap()
    else {
        panic!("ridge Python export")
    };
    let cases = vec![case(
        "valid",
        json!([["1", "NA"], ["2", null], [null, "unseen"]]),
        false,
    )];
    let expected = view
        .predict_raw(&columns(), cases[0].typed.as_ref().unwrap())
        .unwrap();
    let intact = execute(&root.path().join("intact"), &artifact, &bundle, &cases);
    let actual: Predictions =
        serde_json::from_value(intact[0].as_ref().unwrap()["ok"].clone()).unwrap();
    compare(ParityPolicy::ExactBits, &actual, &expected).unwrap();
    // Truncation and reordering cannot pass the cardinality/row-order comparison.
    let Predictions::Values(values) = &actual else {
        panic!("regression values")
    };
    let truncated = Predictions::Values(values[..2].to_vec());
    assert!(compare(ParityPolicy::ExactBits, &truncated, &expected).is_err());
    let mut reordered = values.clone();
    reordered.swap(0, 1);
    assert_ne!(values[0].to_bits(), values[1].to_bits());
    assert!(compare(
        ParityPolicy::ExactBits,
        &Predictions::Values(reordered),
        &expected
    )
    .is_err());
    // A changed learned weight inside the serialized state (the intercept).
    let mut tampered = bundle.clone();
    let source = tampered.files.get_mut("model.py").unwrap();
    let marker = "\\\"weights\\\":[";
    let start = source.find(marker).expect("serialized weights") + marker.len();
    let end = start + source[start..].find(',').unwrap();
    let weight: f64 = source[start..end].parse().unwrap();
    let changed = weight + 1e-3 * (1.0 + weight.abs());
    source.replace_range(start..end, &format!("{changed:?}"));
    let result = execute(&root.path().join("tampered"), &artifact, &tampered, &cases);
    let actual: Predictions =
        serde_json::from_value(result[0].as_ref().unwrap()["ok"].clone()).unwrap();
    assert!(compare(ParityPolicy::ExactBits, &actual, &expected).is_err());
}

/// W9-T06 source determinism, a separate contract from binary reproducibility:
/// re-emitting the same fitted capsule, directly or after an exact-bytes reload, yields
/// byte-identical public bundles and private plans for every exported backend.
#[test]
fn re_emission_of_an_identical_capsule_is_source_deterministic() {
    let rows = vec![
        vec![Some("1".to_owned()), Some("NA".to_owned())],
        vec![None, None],
    ];
    let mut compared = 0;
    for (task, family) in [
        (Task::Regression, "ridge"),
        (Task::Binary, "tab_boost"),
        (Task::Multiclass, "multiclass_ridge"),
    ] {
        let artifact = hostile_fixture(task, family);
        let bytes = artifact.bytes().unwrap();
        let reloaded = FittedRustArtifact::from_bytes(&bytes, &digest(&bytes)).unwrap();
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::JavaScript,
            Backend::Cpp17,
            Backend::Cpp20,
        ] {
            let first = export_fitted(&artifact, backend).unwrap();
            let second = export_fitted(&reloaded, backend).unwrap();
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap()
            );
            if let ExportOutcome::Emitted { bundle } = first {
                let a = qualification_plan(&artifact, &bundle, &rows).unwrap();
                let b = qualification_plan(&reloaded, &bundle, &rows).unwrap();
                assert_eq!(a.identity().unwrap(), b.identity().unwrap());
                compared += 1;
            }
        }
    }
    assert_eq!(compared, 5 + 2 + 3);
}

/// W9-T03: fitted strings are data. Qualification argv is the fixed harness for every
/// backend; no category, class or column string ever reaches a command line.
#[test]
fn hostile_fitted_strings_never_reach_command_arguments() {
    let rows: Vec<Vec<Option<String>>> = HOSTILE_CATEGORIES
        .iter()
        .map(|value| vec![Some("1".to_owned()), Some((*value).to_owned())])
        .collect();
    for (task, family) in [(Task::Regression, "ridge"), (Task::Binary, "logit_ridge")] {
        let artifact = hostile_fixture(task, family);
        let mut needles: Vec<&str> = HOSTILE_CATEGORIES
            .iter()
            .copied()
            .filter(|value| value.len() >= 3)
            .collect();
        needles.extend(artifact.classes.iter().map(String::as_str));
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::JavaScript,
            Backend::Cpp17,
            Backend::Cpp20,
        ] {
            let ExportOutcome::Emitted { bundle } = export_fitted(&artifact, backend).unwrap()
            else {
                panic!("{family} {backend:?}")
            };
            let plan = qualification_plan(&artifact, &bundle, &rows).unwrap();
            verify_plan_binding(&plan, &bundle).unwrap();
            for argument in plan.commands.iter().flatten() {
                for needle in &needles {
                    assert!(
                        !argument.contains(needle),
                        "{argument:?} carries {needle:?}"
                    );
                }
            }
        }
    }
}

/// K06: the pre-fit inventory is a tested registry, not narrative. Every registered
/// recipe is fitted once and its actual export outcome for each backend must equal the
/// inventory claim; an unsupported outcome leaves the fitted state byte-identical.
#[test]
fn registered_inference_inventory_matches_actual_export_outcomes() {
    let inventory = registered_inference_inventory().unwrap();
    let mut checked = 0;
    let mut mismatches = Vec::new();
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        for method in rust_methods(task).unwrap() {
            let id = method.id.clone();
            let claim = inventory
                .iter()
                .find(|row| row["method"] == id.as_str() && row["task"] == json!(task))
                .unwrap_or_else(|| panic!("{id} absent from inventory"));
            let artifact = fit_method(task, method, &hostile_categories(), &|_| 3, false).unwrap();
            let before = artifact.bytes().unwrap();
            for (backend, key) in [
                (Backend::Python, "standalone_python"),
                (Backend::Rust, "standalone_rust"),
                (Backend::JavaScript, "javascript"),
                (Backend::Cpp17, "cpp17_cpp20"),
                (Backend::Cpp20, "cpp17_cpp20"),
            ] {
                let emitted = match export_fitted(&artifact, backend).unwrap() {
                    ExportOutcome::Emitted { .. } => true,
                    ExportOutcome::Unsupported { .. } => false,
                };
                if emitted != (claim[key]["supported"] == true) {
                    mismatches.push(format!("{id}/{backend:?}: emitted={emitted}"));
                }
            }
            assert_eq!(artifact.bytes().unwrap(), before);
            checked += 1;
        }
    }
    assert_eq!(checked, 26);
    assert!(mismatches.is_empty(), "{mismatches:?}");
}

fn deletions(value: &Value, path: &mut Vec<String>, found: &mut Vec<Vec<String>>) {
    match value {
        Value::Object(map) => {
            for (key, child) in map {
                path.push(key.clone());
                found.push(path.clone());
                deletions(child, path, found);
                path.pop();
            }
        }
        Value::Array(items) => {
            for (index, child) in items.iter().enumerate() {
                path.push(index.to_string());
                deletions(child, path, found);
                path.pop();
            }
        }
        _ => {}
    }
}
fn delete(value: &mut Value, path: &[String]) {
    let (last, parents) = path.split_last().unwrap();
    let mut at = value;
    for key in parents {
        at = match at {
            Value::Object(map) => map.get_mut(key).unwrap(),
            Value::Array(items) => &mut items[key.parse::<usize>().unwrap()],
            _ => unreachable!(),
        };
    }
    at.as_object_mut().unwrap().remove(last);
}

/// No learned state is reconstructed with a default. Deleting any single field of a
/// fitted envelope either fails decode/validation, or (only for an absent null option)
/// re-serializes to the identical bytes; the exact-bytes loader refuses every deletion.
#[test]
fn missing_learned_state_is_refused_never_defaulted() {
    for (task, family) in [
        (Task::Regression, "ridge"),
        (Task::Binary, "tab_boost"),
        (Task::Multiclass, "multiclass_ridge"),
    ] {
        let artifact = hostile_fixture(task, family);
        let bytes = artifact.bytes().unwrap();
        let sha = digest(&bytes);
        let original: Value = serde_json::from_slice(&bytes).unwrap();
        let mut paths = Vec::new();
        deletions(&original, &mut Vec::new(), &mut paths);
        assert!(paths.len() > 20);
        let (mut refused, mut identical, mut defaulted) = (0, 0, Vec::new());
        for path in &paths {
            let mut value = original.clone();
            delete(&mut value, path);
            let changed = serde_json::to_vec(&value).unwrap();
            assert!(FittedRustArtifact::from_bytes(&changed, &sha).is_err());
            match FittedRustArtifact::from_bytes(&changed, &digest(&changed)) {
                Err(_) => refused += 1,
                Ok(decoded) if decoded.bytes().unwrap() == bytes => identical += 1,
                Ok(decoded) => {
                    // A defaulted state never inherits the original fitted identity.
                    assert_ne!(
                        FittedInferenceView::registered(&decoded)
                            .unwrap()
                            .fitted_sha256(),
                        sha
                    );
                    defaulted.push(path.join("/"));
                }
            }
        }
        assert!(refused > 0);
        assert!(
            defaulted.is_empty(),
            "{task:?}-{family}: {defaulted:?} ({identical} null options)"
        );
    }
}

#[test]
fn onehot_limit_boundary_and_all_missing_training_column_keep_native_semantics() {
    let root = scratch();
    // 300 categories; the first 256 occur twice and the rest once, so the fitted
    // max_onehot=256 limit is exercised on both sides of the boundary.
    let categories: Vec<Option<String>> = (0..300).map(|i| Some(format!("c{i:03}"))).collect();
    let artifact = fit(
        Task::Regression,
        "ridge",
        &categories,
        &|position| if position < 256 { 2 } else { 1 },
        false,
    )
    .unwrap();
    let probes: Vec<String> = [
        "c000", "c254", "c255", "c256", "c257", "c299", "c300", "C000",
    ]
    .map(str::to_owned)
    .to_vec();
    let probe_refs: Vec<&str> = probes.iter().map(String::as_str).collect();
    let cases = raw_cases(&probe_refs);
    let (found, tallies) = divergences("onehot-boundary", &artifact, &cases, root.path());
    assert!(found.is_empty(), "{}", found.join("\n"));
    assert_tallies(&cases, &tallies, 5);
    // The public state carries exactly the fitted top-256 vocabulary: the first
    // excluded category is absent from emitted source, the last included one present.
    let view = FittedInferenceView::registered(&artifact).unwrap();
    let ExportOutcome::Emitted { bundle } = export_inference(&view, Backend::Rust).unwrap() else {
        panic!("ridge Rust export")
    };
    let source = &bundle.files["model.rs"];
    assert!(source.contains("c255"));
    assert!(!source.contains("c256") && !source.contains("c299"));

    // Entirely missing numeric training column: the native fitter admits it, and every
    // exported backend reproduces that fitted state (fill and indicator) exactly.
    let mut categories: Vec<Option<String>> = vec![None];
    categories.extend(["a", "b", "c"].map(|v| Some(v.to_owned())));
    let artifact = fit(Task::Regression, "ridge", &categories, &|_| 4, true).unwrap();
    let cases = raw_cases(&["a", "b"]);
    let (found, tallies) = divergences(
        "all-missing-training-column",
        &artifact,
        &cases,
        root.path(),
    );
    assert!(found.is_empty(), "{}", found.join("\n"));
    assert_tallies(&cases, &tallies, 5);
}
