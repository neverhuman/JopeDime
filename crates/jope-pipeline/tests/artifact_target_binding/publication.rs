use super::*;

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
                    } else if record.target == Target::TypeScript && f.path.ends_with(".kernel.rs")
                    {
                        "typescript-kernel"
                    } else if record.target == Target::Go && f.path.ends_with(".kernel.rs") {
                        "go-kernel"
                    } else if record.target == Target::Go && f.path == "go.mod" {
                        "go-module"
                    } else if record.target == Target::Go && f.path == "go.sum" {
                        "go-checksums"
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
    let mut targets = Target::RECIPES.to_vec();
    targets.push(Target::TypeScript);
    targets.push(Target::Go);
    let (r, expected) = v3_run(&root, &targets);
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
    substituted_run(Target::TypeScript, "model.ts", "typescript");
    substituted_run(Target::TypeScript, "model.kernel.rs", "typescript-kernel");
    substituted_run(Target::Go, "model.go", "go");
    substituted_run(Target::Go, "model.go.kernel.rs", "go-kernel");
    substituted_run(Target::Go, "go.mod", "go-module");
    substituted_run(Target::Go, "go.sum", "go-checksums");
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
fn native_publication_verifies_fitted_closure_and_refuses_codec_deletion() {
    let dir = scratch();
    let root = dir.path().join("run");
    let (_, expected) = v3_run(&root, &[Target::Cpp17, Target::Cpp20, Target::Cpp17Fitted]);
    publication::verify_run_v3(&root.join("receipt.json"), &expected).unwrap();
    fs::remove_file(root.join("emit/cpp17-fitted/model.codec.json")).unwrap();
    assert!(publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err());
}

#[test]
fn native_publication_requires_every_c17_bundle_member() {
    for member in ["manifest.json", "model.h", "model_core.rs", "model_ffi.rs"] {
        let dir = scratch();
        let root = dir.path().join("run");
        let (_, expected) = v3_run(&root, &[Target::C17]);
        publication::verify_run_v3(&root.join("receipt.json"), &expected).unwrap();
        fs::remove_file(root.join("emit/c17").join(member)).unwrap();
        assert!(
            publication::verify_run_v3(&root.join("receipt.json"), &expected).is_err(),
            "{member}"
        );
    }
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
