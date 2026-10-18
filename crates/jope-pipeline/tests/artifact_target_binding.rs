use feat_math::{artifact_manifest::*, evolve::FaultPolicy};
use jope_pipeline::{
    clock::Clock,
    emit,
    kernel2::{publication, receipt::Receipt, run, search::Refinement},
    run::{Policy, RunOptions},
    search::Budget,
};
use std::{fs, path::Path};
fn scratch() -> tempfile::TempDir {
    let root = Path::new(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn rows() -> Vec<Vec<f64>> {
    (0..10)
        .map(|i| vec![i as f64, 2.0 * i as f64 + 1.0])
        .collect()
}
#[test]
fn writer_reader_accounts_for_all_recipe_and_fitted_components() {
    let dir = scratch();
    let mut targets = Target::RECIPES.to_vec();
    targets.push(Target::Cpp17Fitted);
    let out = emit::emit_v3("(pipe X (ridge))", &targets, dir.path(), Some(&rows())).unwrap();
    assert!(out.manifest.all_supported());
    assert_eq!(
        emit::read_identity_v3(dir.path(), &out.expected).unwrap(),
        out.manifest
    );
    let fitted = out
        .manifest
        .targets
        .iter()
        .find(|r| r.target == Target::Cpp17Fitted)
        .unwrap();
    assert!(
        matches!(&fitted.state, TargetState::Supported { files, fitted: Some(_), .. } if files.len() == 3)
    );
}
#[test]
fn fitted_codec_and_standalone_deletion_or_substitution_refused() {
    for name in ["model.cpp", "model.codec.json", "standalone.cpp"] {
        for deletion in [false, true] {
            let dir = scratch();
            let out = emit::emit_v3(
                "(pipe X (ridge))",
                &[Target::Cpp17Fitted],
                dir.path(),
                Some(&rows()),
            )
            .unwrap();
            let path = dir.path().join("cpp17-fitted").join(name);
            if deletion {
                fs::remove_file(path).unwrap();
            } else {
                fs::write(path, "substitute").unwrap();
            }
            assert!(
                emit::read_identity_v3(dir.path(), &out.expected).is_err(),
                "{name} deletion={deletion}"
            );
        }
    }
}
#[test]
fn fitted_source_cannot_be_replaced_by_recipe() {
    let dir = scratch();
    let out = emit::emit_v3(
        "(pipe X (ridge))",
        &[Target::Cpp17, Target::Cpp17Fitted],
        dir.path(),
        Some(&rows()),
    )
    .unwrap();
    fs::copy(
        dir.path().join("model.cpp"),
        dir.path().join("cpp17-fitted/model.cpp"),
    )
    .unwrap();
    assert!(emit::read_identity_v3(dir.path(), &out.expected).is_err());
}
#[test]
fn fitted_rows_are_required_and_invalid_rows_are_hard_errors() {
    let dir = scratch();
    assert!(emit::emit_v3("(pipe X (ridge))", &[Target::Cpp17Fitted], dir.path(), None).is_err());
    for bad in [
        vec![],
        vec![vec![f64::NAN, 1.0]],
        vec![vec![0.0, 1.0], vec![0.0, 2.0, 3.0]],
    ] {
        assert!(emit::emit_v3(
            "(pipe X (ridge))",
            &[Target::Cpp17Fitted],
            dir.path(),
            Some(&bad)
        )
        .is_err());
    }
    assert!(!dir.path().join("manifest.json").exists());
}
#[test]
fn unsupported_writer_result_never_claims_all_targets_supported() {
    let dir = scratch();
    let out = emit::emit_v3(
        "(pipe X (histgbm depth=5 lr=0.15 rounds=150))",
        &Target::RECIPES,
        dir.path(),
        None,
    )
    .unwrap();
    assert!(!out.manifest.all_supported());
    assert_eq!(out.manifest.targets.len(), 5);
    assert!(!dir.path().join("model.cpp").exists());
    emit::read_identity_v3(dir.path(), &out.expected).unwrap();
}
#[test]
fn writer_refuses_missing_or_extra_sources_before_publication() {
    let g = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    for extra in [false, true] {
        let dir = scratch();
        let mut e = feat_math::transpile::emit_targets_v3(&g, &Default::default(), &[Target::Rust])
            .unwrap();
        if extra {
            e.sources.insert("rogue.rs".into(), "unexpected".into());
        } else {
            e.sources.clear();
        }
        assert!(emit::publish_v3(e, dir.path()).is_err());
        assert!(!dir.path().join("manifest.json").exists());
    }
}
fn legacy_run(root: &Path) -> Receipt {
    let dataset =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../datasets/test/pmlb-229-pwlinear");
    let mut base = RunOptions::new(dataset.display().to_string(), root);
    base.budget = Budget {
        evaluations: 1,
        seconds: 30,
    };
    base.policy = Policy::Skip;
    base.languages = emit::Language::ALL.to_vec();
    base.clock = Clock::Wall;
    base.source_commit = Some("a".repeat(40));
    let opt = run::Options::new(
        base,
        "kernel2-baseline",
        FaultPolicy::RejectHard,
        Refinement {
            top_k: 0,
            trials_per_program: 0,
        },
    )
    .unwrap();
    run::run(&opt).unwrap()
}
fn write_receipt(root: &Path, r: &Receipt) {
    fs::write(root.join("receipt.json"), serde_json::to_vec(r).unwrap()).unwrap();
}
/// Replace only this test's emitted evidence, retaining the actual native run's
/// prepared/search evidence. The production verifier consumes both together.
fn install_v3(
    root: &Path,
    mut r: Receipt,
    stage: &Path,
    out: emit::EmittedV3,
) -> (Receipt, ExpectedArtifactV3) {
    r.emit_manifest_sha256 = out.expected.manifest_sha256.clone();
    r.artifacts.clear();
    r.emit_exclusions.clear();
    fs::copy(stage.join("manifest.json"), root.join("emit/manifest.json")).unwrap();
    for record in &out.manifest.targets {
        match &record.state {
            TargetState::Supported { files, .. } => {
                for f in files {
                    let destination = root.join("emit").join(&f.path);
                    fs::create_dir_all(destination.parent().unwrap()).unwrap();
                    fs::copy(stage.join(&f.path), destination).unwrap();
                    let language = if record.target == Target::Cpp17Fitted {
                        if f.path.ends_with(".codec.json") {
                            "cpp17-fitted-codec"
                        } else if f.path.ends_with("standalone.cpp") {
                            "cpp17-fitted-standalone"
                        } else {
                            "cpp17-fitted"
                        }
                    } else {
                        record.target.as_str()
                    };
                    r.artifacts.push(jope_pipeline::receipt::Artifact {
                        language: language.into(),
                        path: f.path.clone(),
                        sha256: f.sha256.clone(),
                        bytes: f.bytes,
                        build: None,
                    });
                }
            }
            TargetState::Unsupported { reason, .. } => {
                r.emit_exclusions.push(jope_pipeline::receipt::Exclusion {
                    language: record.target.as_str().into(),
                    reason: reason.clone(),
                })
            }
        }
    }
    write_receipt(root, &r);
    (r, out.expected)
}
fn v3_run(root: &Path, requested: &[Target]) -> (Receipt, ExpectedArtifactV3) {
    let r = legacy_run(root);
    let stage = root.join("v3");
    let rows = rows();
    let training = requested
        .contains(&Target::Cpp17Fitted)
        .then_some(rows.as_slice());
    let out = emit::emit_v3(
        &r.search.best.observation.genome,
        requested,
        &stage,
        training,
    )
    .unwrap();
    install_v3(root, r, &stage, out)
}

#[test]
fn native_publication_accepts_supported_pinned_v3_run() {
    let dir = scratch();
    let root = dir.path().join("run");
    let (r, expected) = v3_run(&root, &Target::RECIPES);
    assert_eq!(
        publication::verify_run_v3(&root.join("receipt.json"), &expected).unwrap(),
        r
    );
}
fn substituted_run(target: Target, name: &str, language: &str) {
    let dir = scratch();
    let root = dir.path().join("run");
    let (mut r, expected) = v3_run(&root, &[target]);
    fs::write(root.join("emit").join(name), "substitute").unwrap();
    for a in &mut r.artifacts {
        if a.language == language {
            a.sha256 = sha256(b"substitute");
            a.bytes = 10;
        }
    }
    write_receipt(&root, &r);
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}
#[test]
fn native_publication_refuses_cpp17_substitution_even_with_loose_hash_updated() {
    substituted_run(Target::Cpp17, "model.cpp", "cpp17");
}
#[test]
fn native_publication_refuses_cpp20_substitution_even_with_loose_hash_updated() {
    substituted_run(Target::Cpp20, "model.cpp", "cpp20");
}
#[test]
fn native_publication_refuses_javascript_substitution_even_with_loose_hash_updated() {
    substituted_run(Target::JavaScript, "model.js", "javascript");
}
#[test]
fn native_publication_refuses_python_substitution_even_with_loose_hash_updated() {
    substituted_run(Target::Python, "model.py", "python");
}
#[test]
fn native_publication_refuses_rust_substitution_even_with_loose_hash_updated() {
    substituted_run(Target::Rust, "model.rs", "rust");
}
#[test]
fn native_publication_refuses_omitted_request_and_deleted_file() {
    let dir = scratch();
    let root = dir.path().join("run");
    let (mut r, expected) = v3_run(&root, &Target::RECIPES);
    let original = r.clone();
    r.artifacts.retain(|a| a.language != "javascript");
    write_receipt(&root, &r);
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
    write_receipt(&root, &original);
    fs::remove_file(root.join("emit/model.cpp")).unwrap();
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}
#[test]
fn native_publication_refuses_manifest_and_receipt_rehash_under_external_pin() {
    let dir = scratch();
    let root = dir.path().join("run");
    let (mut r, expected) = v3_run(&root, &[Target::JavaScript]);
    let mut manifest = emit::read_identity_v3(&root.join("emit"), &expected).unwrap();
    fs::write(root.join("emit/model.js"), "substitute").unwrap();
    if let TargetState::Supported { files, .. } = &mut manifest.targets[0].state {
        files[0] = FileBinding::new("model.js", b"substitute");
    }
    let changed = manifest.envelope_bytes(Domain::Native).unwrap();
    fs::write(root.join("emit/manifest.json"), &changed).unwrap();
    r.emit_manifest_sha256 = sha256(&changed);
    r.artifacts[0].sha256 = sha256(b"substitute");
    r.artifacts[0].bytes = 10;
    write_receipt(&root, &r);
    // A mutable self-consistent evidence chain alone is not an external attestation.
    assert!(publication::verify_run(&root.join("receipt.json")).is_ok());
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}
#[test]
fn legacy_cpp_js_rehash_reproduces_the_documented_binding_gap() {
    for name in ["model.cpp", "model.js"] {
        let dir = scratch();
        let root = dir.path().join("run");
        let mut r = legacy_run(&root);
        fs::write(root.join("emit").join(name), "substitute").unwrap();
        for a in &mut r.artifacts {
            if a.path == name {
                a.sha256 = sha256(b"substitute");
                a.bytes = 10;
            }
        }
        write_receipt(&root, &r);
        assert!(publication::verify_run(&root.join("receipt.json")).is_ok());
        let expected = ExpectedArtifactV3 {
            manifest_sha256: r.emit_manifest_sha256.clone(),
            program_sha256: r.search.best.observation.program_sha256.clone(),
            requested: Target::RECIPES.to_vec(),
        };
        assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
    }
}

/// A tiny executable fixture, separate from the manifest's UNKNOWN build claim.
/// It consumes only sources that passed the pinned readback verifier.
#[test]
fn artifact_runtime_fixture_executes_every_supported_product() {
    use std::process::Command;
    fn execute(command: &mut Command) {
        println!("executed: {command:?}");
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let dir = scratch();
    let root = dir.path();
    let mut targets = Target::RECIPES.to_vec();
    targets.extend([Target::Cpp17RecipeV2, Target::Cpp17Fitted]);
    let program = "(pipe X (ridge alpha=1.0))";
    let out = emit::emit_v3(program, &targets, root, Some(&rows())).unwrap();
    assert!(out.manifest.all_supported());
    emit::read_identity_v3(root, &out.expected).unwrap();
    let training = root.join("train.csv");
    let probes = root.join("probes.csv");
    fs::write(
        &training,
        (0..10)
            .map(|i| format!("{i},{}\n", 2 * i + 1))
            .collect::<String>(),
    )
    .unwrap();
    fs::write(&probes, "0.5\n2.5\n").unwrap();
    let train = rows();
    let x: Vec<_> = train.iter().map(|r| vec![r[0]]).collect();
    let y: Vec<_> = train.iter().map(|r| r[1]).collect();
    let expected = feat_math::fit(&feat_math::GenomeIr::parse(program).unwrap(), &x, &y)
        .unwrap()
        .predict(&vec![vec![0.5], vec![2.5]])
        .unwrap();
    let bits = |v: &[f64]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    for target in targets {
        let predictions = root.join(format!("{}.predictions", target.as_str()));
        match target {
            Target::Python => execute(
                Command::new("python3")
                    .arg(root.join("model.py"))
                    .arg(&training)
                    .arg(&probes)
                    .arg(&predictions),
            ),
            Target::Rust => {
                let binary = root.join("model-rust");
                execute(
                    Command::new("rustc")
                        .args(["--edition=2021", "-O"])
                        .arg(root.join("model.rs"))
                        .arg("-o")
                        .arg(&binary),
                );
                execute(
                    Command::new(&binary)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::JavaScript => {
                let driver = include_str!("../../../deploy/ci/emitters/driver.js");
                let source = fs::read_to_string(root.join(target.path())).unwrap();
                let executable = root.join("program.js");
                fs::write(&executable, format!("{source}\n{driver}")).unwrap();
                execute(
                    Command::new("node")
                        .arg(&executable)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::Cpp17 | Target::Cpp20 | Target::Cpp17RecipeV2 => {
                let driver = include_str!("../../../deploy/ci/emitters/driver.cpp");
                let source = fs::read_to_string(root.join(target.path())).unwrap();
                let executable = root.join(format!("program-{}.cpp", target.as_str()));
                fs::write(&executable, format!("{source}\n{driver}")).unwrap();
                let binary = root.join(format!("model-{}", target.as_str()));
                let std = if target == Target::Cpp20 {
                    "-std=c++20"
                } else {
                    "-std=c++17"
                };
                execute(
                    Command::new(feat_math::cxx::compiler())
                        .args([std, "-O3", "-ffp-contract=off", "-fno-fast-math"])
                        .arg(&executable)
                        .arg("-o")
                        .arg(&binary),
                );
                execute(
                    Command::new(&binary)
                        .arg(&training)
                        .arg(&probes)
                        .arg(&predictions),
                );
            }
            Target::Cpp17Fitted => {
                let binary = root.join("model-fitted");
                execute(
                    Command::new(feat_math::cxx::compiler())
                        .args([
                            "-std=c++17",
                            "-O3",
                            "-ffp-contract=off",
                            "-fno-fast-math",
                            "-DJOPE_STANDALONE",
                        ])
                        .arg(root.join("cpp17-fitted/standalone.cpp"))
                        .arg("-o")
                        .arg(&binary),
                );
                let header = root.join("probes-header.csv");
                fs::write(&header, "a\n0.5\n2.5\n").unwrap();
                execute(Command::new(&binary).arg(&header).arg(&predictions));
            }
        }
        let text = fs::read_to_string(&predictions).unwrap();
        let actual: Vec<f64> = text
            .lines()
            .filter(|s| *s != "prediction")
            .map(|s| s.parse().unwrap())
            .collect();
        assert_eq!(bits(&actual), bits(&expected), "{target:?}");
        println!(
            "{}: predictions_sha256={}",
            target.as_str(),
            sha256(text.as_bytes())
        );
    }
}

#[test]
fn native_publication_verifies_fitted_closure_and_refuses_codec_deletion() {
    let dir = scratch();
    let root = dir.path().join("run");
    let (_, expected) = v3_run(&root, &[Target::Cpp17, Target::Cpp20, Target::Cpp17Fitted]);
    publication::verify_run_v3(&root.join("receipt.json"), &expected).unwrap();
    fs::remove_file(root.join("emit/cpp17-fitted/model.codec.json")).unwrap();
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}
#[test]
fn native_publication_requires_explicit_unsupported_accounting() {
    use feat_math::{
        BoundEmitted, ColumnSpec, DataSchema, GenomeIr, Objective, RawTable, RowSemantics, TaskSpec,
    };
    let dir = scratch();
    let root = dir.path().join("run");
    let r = legacy_run(&root);
    let g = GenomeIr::parse(&r.search.best.observation.genome).unwrap();
    let schema = DataSchema {
        columns: vec![ColumnSpec::numeric("a")],
        row_semantics: RowSemantics::Iid,
    };
    let table = RawTable::from_csv("a\n0\n1\n2\n3\n4\n5\n").unwrap();
    let model = feat_math::fit_table(
        &g,
        &TaskSpec::regression(Objective::Rmse).unwrap(),
        &schema,
        &table,
        &[1., 3., 5., 7., 9., 11.],
    )
    .unwrap();
    let e = BoundEmitted::emit_targets_v3(
        &model,
        &g,
        &Default::default(),
        &[Target::Rust, Target::JavaScript],
    )
    .unwrap();
    let stage = root.join("v3");
    let out = emit::publish_v3(e, &stage).unwrap();
    assert!(!out.manifest.all_supported());
    let (mut r, expected) = install_v3(&root, r, &stage, out);
    publication::verify_run_v3(&root.join("receipt.json"), &expected).unwrap();
    r.emit_exclusions.clear();
    write_receipt(&root, &r);
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}
