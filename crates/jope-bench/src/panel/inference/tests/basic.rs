use super::*;

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
    use super::super::super::synthesized::{fit_synthesized, SynthesizedFitRequest};
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
