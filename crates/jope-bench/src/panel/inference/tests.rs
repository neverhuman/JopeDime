use super::super::data::raw_table;
use super::super::methods::{fit_rust, rust_methods, RustFitRequest};
use super::*;
use feat_math::advanced_v3::{DataSchema, RowSemantics};
use jope_pipeline::experiment::{Column, ColumnKind as Kind};
use std::process::Command;

use super::super::data::raw_table;
use super::super::methods::{fit_rust, rust_methods, RustFitRequest};
use super::*;
use feat_math::advanced_v3::{DataSchema, RowSemantics};
use jope_pipeline::experiment::{Column, ColumnKind as Kind};
use std::process::Command;

pub(crate) fn fixture(task: Task, family: &str) -> (FittedRustArtifact, Vec<Vec<Option<String>>>) {
    let (request, rows) = fixture_request(task, family);
    (fit_rust(&request).unwrap().artifact, rows)
}
pub(crate) fn fixture_request(
    task: Task,
    family: &str,
) -> (RustFitRequest, Vec<Vec<Option<String>>>) {
    let columns = vec![
        Column {
            name: "amount".into(),
            kind: Kind::Numeric,
        },
        Column {
            name: "country".into(),
            kind: Kind::Categorical,
        },
    ];
    let categories = [
        None,
        Some("NA"),
        Some("null"),
        Some(""),
        Some("line\n\"quoted\"\u{2028}"),
        Some("__PREDICT____REGRESSION____CLASSES_JSON__"),
    ];
    let rows = (0..18)
        .map(|i| {
            vec![
                if i % 7 == 0 {
                    None
                } else {
                    Some(i.to_string())
                },
                categories[i % categories.len()].map(str::to_owned),
            ]
        })
        .collect::<Vec<_>>();
    let prediction = vec![
        vec![None, Some("NA".into())],
        vec![Some("1000000".into()), Some("unseen".into())],
        vec![Some("-0.0".into()), None],
        vec![Some("3.25".into()), Some("line\n\"quoted\"\u{2028}".into())],
    ];
    let classes = match task {
        Task::Regression => vec![],
        Task::Binary => vec!["tree\nquoted\"".into(), "search".into()],
        _ => vec!["zebra".into(), "alpha".into(), "middle".into()],
    };
    let targets = (0..18)
        .map(|i| {
            if task == Task::Regression {
                (i as f64) * 1.25 - 3.0
            } else {
                (i % classes.len()) as f64
            }
        })
        .collect();
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
        train: raw_table(&columns, &rows).unwrap(),
        targets,
        predict: raw_table(&columns, &prediction).unwrap(),
        method: rust_methods(task)
            .unwrap()
            .into_iter()
            .find(|method| method.family == family)
            .unwrap(),
    };
    (request, prediction)
}
pub(crate) fn emitted(artifact: &FittedRustArtifact, backend: Backend) -> PublicBundle {
    match export_fitted(artifact, backend).unwrap() {
        ExportOutcome::Emitted { bundle } => *bundle,
        ExportOutcome::Unsupported { reason, .. } => panic!("{backend:?}: {reason:?}"),
    }
}

/// W9-T01: named records follow the manifest's column order whatever their key order,
/// and on every backend they yield the very plan (native reference included) of the
/// positional rows; ambiguous or malformed records are refused before anything runs.
#[test]
fn named_column_records_follow_the_manifest_and_refuse_ambiguity() {
    let (artifact, rows) = fixture(Task::Binary, "logit_ridge");
    let named = serde_json::to_vec(
        &rows
            .iter()
            .map(|row| serde_json::json!({"country": row[1], "amount": row[0]}))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let view = FittedInferenceView::registered(&artifact).unwrap();
    for backend in [
        Backend::Python,
        Backend::Rust,
        Backend::Cpp17,
        Backend::Cpp20,
        Backend::JavaScript,
    ] {
        let bundle = emitted(&artifact, backend);
        let positional = bundle.manifest.positional_rows(&named).unwrap();
        assert_eq!(positional, rows, "{backend:?}");
        assert_eq!(
            qualification_plan_for_view(&view, &bundle, &positional)
                .unwrap()
                .identity()
                .unwrap(),
            qualification_plan_for_view(&view, &bundle, &rows)
                .unwrap()
                .identity()
                .unwrap(),
            "{backend:?}"
        );
    }
    let manifest = emitted(&artifact, Backend::Python).manifest;
    assert!(manifest.positional_rows(b"[]").unwrap().is_empty());
    for (label, text, needle) in [
        (
            "missing",
            r#"[{"amount":"1"}]"#,
            r#"column "country" missing"#,
        ),
        (
            "unknown",
            r#"[{"amount":"1","country":"x","extra":"y"}]"#,
            r#"unknown column "extra""#,
        ),
        (
            "case-changed",
            r#"[{"Amount":"1","country":"x"}]"#,
            r#"unknown column "Amount""#,
        ),
        (
            "repeated",
            r#"[{"amount":"1","country":"x","amount":"2"}]"#,
            r#"column "amount" named twice"#,
        ),
        ("number", r#"[{"amount":1,"country":"x"}]"#, "invalid type"),
        ("positional", r#"[["1","x"]]"#, "named records"),
        ("object", r#"{"amount":"1","country":"x"}"#, "named records"),
    ] {
        let error = format!(
            "{:#}",
            manifest.positional_rows(text.as_bytes()).unwrap_err()
        );
        assert!(error.contains(needle), "{label}: {error}");
    }
    let mut ambiguous = manifest.clone();
    ambiguous.columns[1].name = ambiguous.columns[0].name.clone();
    let error = format!(
        "{:#}",
        ambiguous
            .positional_rows(br#"[{"amount":"1"}]"#)
            .unwrap_err()
    );
    assert!(error.contains("not distinct"), "{error}");
}

#[test]
fn public_files_are_allowlisted_and_never_contain_private_recipe_or_envelope() {
    for (task, head) in [
        (Task::Regression, "ridge"),
        (Task::Binary, "logit_ridge"),
        (Task::Multiclass, "multiclass_ridge"),
    ] {
        let (artifact, _) = fixture(task, head);
        for backend in [
            Backend::Python,
            Backend::Rust,
            Backend::Cpp17,
            Backend::Cpp20,
            Backend::JavaScript,
        ] {
            if task == Task::Multiclass && matches!(backend, Backend::Cpp17 | Backend::Cpp20) {
                continue;
            }
            let bundle = emitted(&artifact, backend);
            for text in bundle.files.values().chain(std::iter::once(
                &serde_json::to_string(&bundle.manifest).unwrap(),
            )) {
                assert!(
                    lint_fitted_public_text(&artifact.model, &artifact.classes, text).is_empty()
                );
            }
            let public = serde_json::to_string(&bundle).unwrap();
            for canary in [
                &artifact.method.dsl,
                &artifact.model.genome_sha256,
                "genome_sha256",
                "jope.panel-fitted-rust/v1",
                "jope.panel-fitted-rust/v2",
                "fitting_input_sha256",
                "discovery_provenance",
            ] {
                assert!(
                    !public.contains(canary),
                    "private canary in {backend:?}: {canary}"
                );
            }
            assert_eq!(bundle.manifest.qualification, "emitted-unqualified");
            assert_eq!(bundle.manifest.parity_policy, backend.parity_policy());
            let mut tampered = bundle.clone();
            tampered.files.insert(
                "private-recipe.json".into(),
                serde_json::to_string(&artifact).unwrap(),
            );
            assert!(tampered.validate().is_err());
            let mut tampered = bundle;
            tampered.files.values_mut().next().unwrap().push(' ');
            assert!(tampered.validate().is_err());
        }
    }
    let (artifact, _) = fixture(Task::Regression, "ridge");
    let cpp17 = emitted(&artifact, Backend::Cpp17);
    let cpp20 = emitted(&artifact, Backend::Cpp20);
    assert_eq!(
        cpp17.files, cpp20.files,
        "one C++ source and codec for both standards"
    );
}

#[test]
fn generated_winners_preserve_raw_boundary_and_private_provenance_on_export() {
    use super::super::synthesized::{fit_synthesized, SynthesizedFitRequest};
    use jope_pipeline::synthesized_recipe::SynthesizedRecipe;
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-inference-tests");
    std::fs::create_dir_all(&scratch).unwrap();
    let root = tempfile::tempdir_in(scratch).unwrap();
    let mut replays = 0;
    for (task, head) in [
        (Task::Regression, "extra_trees"),
        (Task::Binary, "portfolio"),
        (Task::Multiclass, "multiclass_diagonal_lda"),
    ] {
        let (request, rows) = fixture_request(task, head);
        // Explicit generated input need not equal the roster recipe's median.
        let proposal = format!(
            "  {}\n",
            request.method.dsl.replace("numeric=median", "numeric=mean")
        );
        let request = SynthesizedFitRequest {
            task,
            schema: request.schema,
            classes: request.classes,
            train: request.train,
            targets: request.targets,
            predict: request.predict,
            recipe: SynthesizedRecipe::new(proposal).unwrap(),
            scheduling_seed: 17,
        };
        let result = fit_synthesized(&request).unwrap();
        let artifact = result.artifact;
        let private = artifact.bytes().unwrap();
        for backend in [Backend::Python, Backend::Rust] {
            let ExportOutcome::Emitted { bundle } = export_synthesized(&artifact, backend).unwrap()
            else {
                panic!("generated fitted head unsupported")
            };
            assert_eq!(bundle.manifest.input_policy, FITTED_INPUT_POLICY);
            let public = serde_json::to_string(&bundle).unwrap();
            for canary in [
                &artifact.recipe.original_proposal,
                &artifact.recipe.executable_canonical,
                &artifact.recipe.executable_sha256,
                "original_proposal",
                "jope.panel-fitted-synthesized/v1",
                "discovery_provenance",
            ] {
                assert!(
                    !public.contains(canary),
                    "private generated provenance leaked"
                );
            }
            let plan = qualification_plan_synthesized(&artifact, &bundle, &rows).unwrap();
            assert_eq!(plan.fitted_sha256, digest(&private));
            let directory = root.path().join(format!("{task:?}-{backend:?}"));
            std::fs::create_dir(&directory).unwrap();
            for (name, text) in &plan.files {
                std::fs::write(directory.join(name), text).unwrap();
            }
            let mut output = Vec::new();
            for args in &plan.commands {
                let run = Command::new(&args[0])
                    .args(&args[1..])
                    .current_dir(&directory)
                    .env("OMP_NUM_THREADS", "1")
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .output()
                    .unwrap();
                assert!(
                    run.status.success(),
                    "{head} {backend:?}: {}",
                    String::from_utf8_lossy(&run.stderr)
                );
                output = run.stdout;
            }
            verify_qualification(&plan, &plan.identity().unwrap(), &output).unwrap();
            replays += 1;
        }
        assert_eq!(artifact.bytes().unwrap(), private);
        let mut broken = artifact.clone();
        broken.recipe.original_proposal.push(' ');
        assert!(export_synthesized(&broken, Backend::Python).is_err());
    }
    assert_eq!(replays, 6);
    let (request, _) = fixture_request(Task::Regression, "ridge");
    let recipe =
        SynthesizedRecipe::new(request.method.dsl.replace("(ridge", "(standardize) (ridge"))
            .unwrap();
    let artifact = fit_synthesized(&SynthesizedFitRequest {
        task: request.task,
        schema: request.schema,
        classes: request.classes,
        train: request.train,
        targets: request.targets,
        predict: request.predict,
        recipe,
        scheduling_seed: 17,
    })
    .unwrap()
    .artifact;
    let original = artifact.bytes().unwrap();
    assert!(matches!(
        export_synthesized(&artifact, Backend::Python).unwrap(),
        ExportOutcome::Unsupported {
            reason: UnsupportedReason::FittedState(PortableUnsupported::Operations),
            ..
        }
    ));
    assert_eq!(artifact.bytes().unwrap(), original);
}

#[test]
fn complete_registry_has_explicit_pre_fit_inference_coverage() {
    let rows = registered_inference_inventory().unwrap();
    assert_eq!(rows.len(), 26);
    assert_eq!(
        rows.iter()
            .filter(|r| r["standalone_python"]["supported"] == true)
            .count(),
        14
    );
    assert_eq!(
        rows.iter()
            .filter(|r| r["standalone_rust"]["supported"] == true)
            .count(),
        14
    );
    assert_eq!(
        rows.iter()
            .filter(|r| r["javascript"]["supported"] == true)
            .count(),
        6
    );
    assert!(rows
        .iter()
        .all(|r| r["unsupported_export_replaces_winner"] == false));
}
#[test]
fn extended_scalar_export_keeps_auxiliary_backends_explicitly_unsupported() {
    let (artifact, _) = fixture(Task::Binary, "portfolio");
    let original = artifact.bytes().unwrap();
    for backend in [Backend::Cpp17, Backend::Cpp20, Backend::JavaScript] {
        assert!(matches!(
            export_fitted(&artifact, backend).unwrap(),
            ExportOutcome::Unsupported {
                reason: UnsupportedReason::FittedState(PortableUnsupported::Head),
                ..
            }
        ));
    }
    for backend in [Backend::Python, Backend::Rust] {
        let bundle = emitted(&artifact, backend);
        let text = serde_json::to_string(&bundle).unwrap();
        for canary in [
            &artifact.method.dsl,
            &artifact.model.genome_sha256,
            "genome_sha256",
            "\"sums\"",
            "\"counts\"",
            "discovery_provenance",
        ] {
            assert!(!text.contains(canary));
        }
    }
    assert_eq!(artifact.bytes().unwrap(), original);
}

#[test]
fn unsupported_backend_keeps_the_fitted_winner_unchanged() {
    let (artifact, _) = fixture(Task::Regression, "histgbm");
    let before = artifact.bytes().unwrap();
    assert!(matches!(
        export_fitted(&artifact, Backend::JavaScript).unwrap(),
        ExportOutcome::Unsupported {
            reason: UnsupportedReason::FittedState(PortableUnsupported::Head),
            ..
        }
    ));
    assert_eq!(artifact.bytes().unwrap(), before);
    let (artifact, _) = fixture(Task::Multiclass, "multiclass_ridge");
    assert!(matches!(
        export_fitted(&artifact, Backend::Cpp17).unwrap(),
        ExportOutcome::Unsupported {
            reason: UnsupportedReason::CppMatrix,
            ..
        }
    ));
}

#[test]
fn all_matrix_heads_and_scalar_boundaries_compile_and_replay_synthetic_fitted_state() {
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/fitted-inference-tests");
    std::fs::create_dir_all(&scratch).unwrap();
    let root = tempfile::tempdir_in(scratch).unwrap();
    let cases = [
        (Task::Regression, "tab_boost"),
        (Task::Regression, "extra_trees"),
        (Task::Regression, "gam_boost"),
        (Task::Regression, "pls"),
        (Task::Binary, "tab_boost"),
        (Task::Binary, "extra_trees"),
        (Task::Binary, "gam_boost"),
        (Task::Binary, "portfolio"),
        (Task::Regression, "ridge"),
        (Task::Binary, "logit_ridge"),
        (Task::Multiclass, "multiclass_ridge"),
        (Task::Multiclass, "multiclass_ovr_logistic"),
        (Task::Multiclass, "multiclass_diagonal_lda"),
        (Task::Multiclass, "multiclass_nearest_centroid"),
    ];
    let mut executed = 0;
    for (task, head) in cases {
        let (artifact, rows) = fixture(task, head);
        let mut backends = vec![Backend::Python, Backend::Rust];
        if fitted_javascript_unsupported(&artifact.model)
            .unwrap()
            .is_none()
        {
            backends.push(Backend::JavaScript);
        }
        if fitted_public_cpp_unsupported(&artifact.model)
            .unwrap()
            .is_none()
        {
            backends.extend([Backend::Cpp17, Backend::Cpp20]);
        }
        for backend in backends {
            let bundle = emitted(&artifact, backend);
            let plan = qualification_plan(&artifact, &bundle, &rows).unwrap();
            let plan_identity = plan.identity().unwrap();
            let directory = root.path().join(format!("{task:?}-{head}-{backend:?}"));
            std::fs::create_dir(&directory).unwrap();
            for (path, text) in &plan.files {
                std::fs::write(directory.join(path), text).unwrap();
            }
            let mut output = Vec::new();
            for argv in &plan.commands {
                let result = Command::new(&argv[0])
                    .args(&argv[1..])
                    .current_dir(&directory)
                    .env("OMP_NUM_THREADS", "1")
                    .env("OPENBLAS_NUM_THREADS", "1")
                    .output()
                    .unwrap_or_else(|error| {
                        panic!("required tool {} unavailable: {error}", argv[0])
                    });
                assert!(
                    result.status.success(),
                    "{head} {backend:?} {argv:?}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                output = result.stdout;
            }
            let receipt =
                verify_qualification(&plan, &plan_identity, &output).unwrap_or_else(|error| {
                    panic!(
                        "{head} {backend:?}: {error:#}; {}",
                        String::from_utf8_lossy(&output)
                    )
                });
            assert_eq!(
                receipt.values,
                rows.len()
                    * if task == Task::Regression {
                        1
                    } else {
                        artifact.classes.len()
                    }
            );
            assert_eq!(receipt.plan_sha256, plan_identity);
            assert!(verify_qualification(&plan, &"0".repeat(64), &output).is_err());
            let mut stale = plan.clone();
            stale
                .files
                .get_mut("qualification-input.json")
                .unwrap()
                .push(' ');
            assert!(verify_qualification(&stale, &plan_identity, &output).is_err());
            if let Predictions::Probabilities {
                mut classes,
                values,
            } = serde_json::from_slice::<Predictions>(&output).unwrap()
            {
                classes.swap(0, 1);
                let changed =
                    serde_json::to_vec(&Predictions::Probabilities { classes, values }).unwrap();
                assert!(verify_qualification(&plan, &plan_identity, &changed).is_err());
            }
            executed += 1;
        }
    }
    assert_eq!(executed, 38);
}
#[test]
fn fitted_boundary_template_bytes_are_pinned_for_version_one() {
    let actual = templates(Backend::Python);
    for (name, sha) in [
        (
            "boundary.py",
            "88692ca12accbc84f741b0b632e299e2e5a8527a889de50aa10692431182cbdf",
        ),
        (
            "boundary.rs",
            "e3aec495101ca9e21aafe823e52ca05660029127eb084556a0a8ce32a2ce46fb",
        ),
        (
            "boundary.js",
            "5a13d2e63c13903691b641734692aa914e5e96d6fddcfad9e3894b0ac86a96be",
        ),
        (
            "typed-host.py",
            "1a05ba9e745daec635a3b20d9a8dba7426840be139e0cdf00b0e48f8578b6b1b",
        ),
    ] {
        assert_eq!(actual[name], sha);
    }
}
#[test]
fn invalid_fitted_coefficients_fail_instead_of_becoming_unsupported() {
    let (artifact, _) = fixture(Task::Regression, "ridge");
    let mut encoded = serde_json::to_value(&artifact).unwrap();
    fn corrupt(value: &mut serde_json::Value) -> bool {
        match value {
            serde_json::Value::Object(fields) => {
                if let Some(serde_json::Value::Array(scales)) = fields.get_mut("scales") {
                    if let Some(first) = scales.first_mut() {
                        *first = serde_json::json!(0.0);
                        return true;
                    }
                }
                fields.values_mut().any(corrupt)
            }
            serde_json::Value::Array(items) => items.iter_mut().any(corrupt),
            _ => false,
        }
    }
    assert!(corrupt(&mut encoded));
    let altered: FittedRustArtifact = serde_json::from_value(encoded).unwrap();
    for backend in [
        Backend::Python,
        Backend::Rust,
        Backend::Cpp17,
        Backend::Cpp20,
        Backend::JavaScript,
    ] {
        assert!(export_fitted(&altered, backend).is_err(), "{backend:?}");
    }
}
