use super::*;

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
