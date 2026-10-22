use feat_math::{
    artifact_manifest::*,
    transpile::{self, EmitOptions},
    GenomeIr,
};

const PROGRAM: &str = "(pipe X (ridge alpha=1.0))";
fn emission(targets: &[Target]) -> TargetEmission {
    transpile::emit_targets_v3(
        &GenomeIr::parse(PROGRAM).unwrap(),
        &EmitOptions::default(),
        targets,
    )
    .unwrap()
}
fn scratch() -> tempfile::TempDir {
    let path = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(path).unwrap();
    tempfile::tempdir_in(path).unwrap()
}
fn publish(e: &TargetEmission) -> (tempfile::TempDir, ExpectedArtifactV3) {
    let dir = scratch();
    for (p, s) in &e.sources {
        let path = dir.path().join(p);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, s).unwrap();
    }
    let bytes = e.manifest.envelope_bytes(Domain::Native).unwrap();
    std::fs::write(dir.path().join("manifest.json"), &bytes).unwrap();
    (dir, ExpectedArtifactV3::from_emission(&bytes, &e.manifest))
}
#[test]
fn all_requested_recipe_targets_bind_exact_source_bytes() {
    let ts = emission(&[Target::TypeScript]);
    assert!(ts.manifest.all_supported());
    assert_eq!(Target::parse("ts").unwrap(), Target::TypeScript);
    assert_eq!(ts.sources.len(), 2);
    for source in ts.sources.values() {
        assert!(transpile::lint_shipped_text(source).is_empty());
    }
    let tokens = feat_math::tokens::count_tokens(
        "// typed model\nconst label: string = 'ok';",
        feat_math::tokens::Lang::TypeScript,
    );
    assert_eq!(tokens.doc_tokens, 2);
    assert!(tokens.code_tokens > 0);
    let mut weakened = ts.manifest.clone();
    if let TargetState::Supported { build, .. } = &mut weakened.targets[0].state {
        build.flags.clear();
    }
    assert!(weakened.validate().is_err());
    let again = emission(&[Target::TypeScript]);
    assert_eq!(ts.sources, again.sources);
    let (ts_dir, ts_pin) = publish(&ts);
    verify_directory(ts_dir.path(), "manifest.json", &ts_pin).unwrap();
    std::fs::write(ts_dir.path().join("model.kernel.rs"), "tampered").unwrap();
    assert!(verify_directory(ts_dir.path(), "manifest.json", &ts_pin).is_err());
    let (ts_dir, ts_pin) = publish(&ts);
    std::fs::remove_file(ts_dir.path().join("model.kernel.rs")).unwrap();
    assert!(verify_directory(ts_dir.path(), "manifest.json", &ts_pin).is_err());
    let e = emission(&Target::RECIPES);
    assert!(e.manifest.all_supported(), "{:?}", e.manifest.targets);
    assert_eq!(e.manifest.targets.len(), 5);
    let (dir, pin) = publish(&e);
    assert_eq!(
        verify_directory(dir.path(), "manifest.json", &pin).unwrap(),
        e.manifest
    );
    for r in &e.manifest.targets {
        if let TargetState::Supported { build, .. } = &r.state {
            assert_eq!(build.execution, BuildExecution::Unknown);
        }
    }
}
#[test]
fn unsupported_targets_retain_reason_and_registry_evidence() {
    let g = GenomeIr::parse("(pipe X (histgbm depth=5 lr=0.15 rounds=150))").unwrap();
    let e = transpile::emit_targets_v3(&g, &EmitOptions::default(), &Target::RECIPES).unwrap();
    assert!(!e.manifest.all_supported());
    for t in [Target::Cpp17, Target::Cpp20, Target::JavaScript] {
        assert!(e.manifest.targets.iter().any(|r| r.target == t && matches!(&r.state, TargetState::Unsupported { reason, registry_sha256 } if !reason.is_empty() && registry_sha256.len() == 64)));
    }
    let (dir, pin) = publish(&e);
    verify_directory(dir.path(), "manifest.json", &pin).unwrap();
}
fn substitute(target: Target) {
    let e = emission(&[target]);
    let (dir, pin) = publish(&e);
    std::fs::write(dir.path().join(target.path()), "substituted source").unwrap();
    let error = verify_directory(dir.path(), "manifest.json", &pin)
        .unwrap_err()
        .to_string();
    assert!(error.contains("artifact bytes disagree"), "{error}");
}
#[test]
fn python_substitution_refused() {
    substitute(Target::Python);
}
#[test]
fn rust_substitution_refused() {
    substitute(Target::Rust);
}
#[test]
fn cpp17_substitution_refused() {
    substitute(Target::Cpp17);
}
#[test]
fn cpp20_substitution_refused() {
    substitute(Target::Cpp20);
}
#[test]
fn javascript_substitution_refused() {
    substitute(Target::JavaScript);
}
#[test]
fn missing_requested_file_refused() {
    for t in Target::RECIPES {
        let e = emission(&[t]);
        let (dir, pin) = publish(&e);
        std::fs::remove_file(dir.path().join(t.path())).unwrap();
        assert!(
            verify_directory(dir.path(), "manifest.json", &pin).is_err(),
            "{t:?}"
        );
    }
}
#[test]
fn empty_duplicate_alias_and_unknown_requests_refused() {
    for value in ["", "py,python", "cpp17,cpp17", "go", "all,rs", "rust,"] {
        assert!(Target::parse_list(value).is_err(), "{value}");
    }
    assert!(transpile::emit_targets_v3(
        &GenomeIr::parse(PROGRAM).unwrap(),
        &EmitOptions::default(),
        &[]
    )
    .is_err());
    assert!(transpile::emit_targets_v3(
        &GenomeIr::parse(PROGRAM).unwrap(),
        &EmitOptions::default(),
        &[Target::Rust, Target::Rust]
    )
    .is_err());
}
#[test]
fn cpp17_and_cpp20_share_source_with_distinct_build_identities() {
    let e = emission(&[Target::Cpp17, Target::Cpp20]);
    let get = |i: usize| match &e.manifest.targets[i].state {
        TargetState::Supported { files, build, .. } => (files, build),
        _ => panic!(),
    };
    let (a, ab) = get(0);
    let (b, bb) = get(1);
    assert_eq!(a, b);
    assert_ne!(ab, bb);
    assert_eq!(ab.standard, "c++17");
    assert_eq!(bb.standard, "c++20");
}
#[test]
fn recipe_v2_is_separate_from_historical_recipe() {
    let e = emission(&[Target::Cpp17, Target::Cpp17RecipeV2]);
    assert!(e.manifest.all_supported());
    assert_eq!(e.sources.len(), 2);
    assert_ne!(e.manifest.targets[0].state, e.manifest.targets[1].state);
    let (dir, pin) = publish(&e);
    verify_directory(dir.path(), "manifest.json", &pin).unwrap();
}
#[test]
fn expected_manifest_pin_refuses_rehashed_substitution() {
    let mut e = emission(&[Target::JavaScript]);
    let (dir, pin) = publish(&e);
    let replacement = "substituted source";
    std::fs::write(dir.path().join("model.js"), replacement).unwrap();
    if let TargetState::Supported { files, .. } = &mut e.manifest.targets[0].state {
        files[0] = FileBinding::new("model.js", replacement.as_bytes());
    }
    std::fs::write(
        dir.path().join("manifest.json"),
        e.manifest.envelope_bytes(Domain::Native).unwrap(),
    )
    .unwrap();
    assert!(verify_directory(dir.path(), "manifest.json", &pin)
        .unwrap_err()
        .to_string()
        .contains("pinned manifest"));
}
#[test]
fn schema_options_and_selected_program_changes_refused() {
    let e = emission(&[Target::Python]);
    let (dir, pin) = publish(&e);
    let mut changed = e.manifest.clone();
    changed.options.title = Some("another title".into());
    let bytes = changed.envelope_bytes(Domain::Native).unwrap();
    assert!(read_manifest(&bytes, &pin).is_err());
    changed.schema = "battle-artifact-v1".into();
    assert!(changed.validate().is_err());
    let mut wrong = pin.clone();
    wrong.program_sha256 = "a".repeat(64);
    assert!(verify_directory(dir.path(), "manifest.json", &wrong).is_err());
    wrong = pin;
    wrong.requested.push(Target::Rust);
    assert!(verify_directory(dir.path(), "manifest.json", &wrong).is_err());
}
#[test]
fn traversal_and_product_confusion_refused_before_reading() {
    let mut e = emission(&[Target::Rust]);
    if let TargetState::Supported { files, .. } = &mut e.manifest.targets[0].state {
        files[0].path = "../model.rs".into();
    }
    assert!(e.manifest.validate().is_err());
    let mut e = emission(&[Target::Cpp17]);
    if let TargetState::Supported { product, .. } = &mut e.manifest.targets[0].state {
        *product = "fitted-cpp17".into();
    }
    assert!(e.manifest.validate().is_err());
}
#[test]
fn legacy_readers_cannot_be_upgraded_by_reserializing_old_data() {
    let g = GenomeIr::parse(PROGRAM).unwrap();
    let old = transpile::emit(&g, &EmitOptions::default()).unwrap();
    let e = emission(&[Target::Python]);
    let pin = ExpectedArtifactV3::from_emission(old.manifest.as_bytes(), &e.manifest);
    assert!(read_manifest(old.manifest.as_bytes(), &pin).is_err());
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&old.manifest).unwrap()["schema"],
        "battle-artifact-v1"
    );
}
#[test]
fn duplicate_keys_noncanonical_and_unknown_envelope_fields_refused() {
    let e = emission(&[Target::Rust]);
    let bytes = e.manifest.envelope_bytes(Domain::Storage).unwrap();
    let text = String::from_utf8(bytes).unwrap();
    for changed in [
        format!(" {text}"),
        text.replacen("{", "{\"extra\":true,", 1),
        text.replacen("{", "{\"schema_id\":\"jope.battle-artifact/v3\",", 1),
    ] {
        let pin = ExpectedArtifactV3::from_emission(changed.as_bytes(), &e.manifest);
        assert!(read_manifest(changed.as_bytes(), &pin).is_err());
    }
}
#[cfg(unix)]
#[test]
fn symlink_source_is_refused() {
    let e = emission(&[Target::Rust]);
    let (dir, pin) = publish(&e);
    std::fs::rename(dir.path().join("model.rs"), dir.path().join("actual.rs")).unwrap();
    std::os::unix::fs::symlink("actual.rs", dir.path().join("model.rs")).unwrap();
    assert!(verify_directory(dir.path(), "manifest.json", &pin).is_err());
}

#[test]
fn bound_adapter_preserves_v2_sources_and_binds_fitted_state() {
    use feat_math::{
        BoundEmitted, ColumnSpec, DataSchema, Objective, RawTable, RowSemantics, TaskSpec,
    };
    let g = GenomeIr::parse("(pipe X (input) (ridge))").unwrap();
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
    let opts = EmitOptions::default();
    let old = feat_math::emit_bound(&model, &opts).unwrap();
    let e = BoundEmitted::emit_targets_v3(
        &model,
        &g,
        &opts,
        &[
            Target::Python,
            Target::Rust,
            Target::JavaScript,
            Target::TypeScript,
        ],
    )
    .unwrap();
    assert_eq!(e.sources["model.py"], old.python);
    assert_eq!(e.sources["model.rs"], old.rust);
    assert!(e.sources.contains_key("model.ts"));
    assert!(e.sources.contains_key("model.kernel.rs"));
    let typescript = e
        .manifest
        .targets
        .iter()
        .find(|r| r.target == Target::TypeScript)
        .unwrap();
    assert!(matches!(
        &typescript.state,
        TargetState::Supported { fitted: Some(_), .. }
    ));
    assert!(!e.manifest.all_supported());
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&old.manifest).unwrap()["schema"],
        "battle-artifact-v2"
    );
    let (dir, pin) = publish(&e);
    verify_directory(dir.path(), "manifest.json", &pin).unwrap();
    let other = GenomeIr::parse("(pipe X (input) (ridge alpha=9))").unwrap();
    assert!(BoundEmitted::emit_targets_v3(&model, &other, &opts, &[Target::Python]).is_err());
}

#[test]
fn wire_fixture_matches_i02_domain_separated_identity_and_target_spellings() {
    use sha2::{Digest, Sha256};
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/artifact-target-v3/requests.json")).unwrap();
    let requested: Vec<Target> = serde_json::from_value(fixture["requests"].clone()).unwrap();
    assert_eq!(
        requested.iter().map(|t| t.as_str()).collect::<Vec<_>>(),
        fixture["requests"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s.as_str().unwrap())
            .collect::<Vec<_>>()
    );
    let e = emission(&[Target::Python]);
    let bytes = e.manifest.envelope_bytes(Domain::Native).unwrap();
    let envelope: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let payload = serde_json::to_vec(&envelope["payload"]).unwrap();
    let mut hasher = Sha256::new();
    hasher.update(SCHEMA.as_bytes());
    hasher.update([0]);
    hasher.update((payload.len() as u64).to_be_bytes());
    hasher.update(&payload);
    assert_eq!(
        envelope["own_id"]["sha256"],
        format!("{:x}", hasher.finalize())
    );
    assert_eq!(envelope["own_id"]["formula"], fixture["formula"]);
    assert_eq!(envelope["envelope_schema"], fixture["envelope_schema"]);
}

#[test]
fn native_inference_and_storage_envelopes_verify_without_execution_authority() {
    let e = emission(&[Target::Rust]);
    for domain in [Domain::Native, Domain::Inference, Domain::Storage] {
        let bytes = e.manifest.envelope_bytes(domain).unwrap();
        let expected = ExpectedArtifactV3::from_emission(&bytes, &e.manifest);
        assert_eq!(read_manifest(&bytes, &expected).unwrap(), e.manifest);
        assert!(!String::from_utf8(bytes)
            .unwrap()
            .contains("verified_measured"));
    }
}
#[test]
fn duplicate_records_incomplete_closures_and_build_standard_mismatch_refused() {
    let mut e = emission(&[Target::Cpp17, Target::Cpp20]);
    e.manifest.targets[1] = e.manifest.targets[0].clone();
    assert!(e.manifest.validate().is_err());
    let mut e = emission(&[Target::Cpp17]);
    if let TargetState::Supported { build, .. } = &mut e.manifest.targets[0].state {
        build.standard = "c++20".into();
    }
    assert!(e.manifest.validate().is_err());
    let mut e = emission(&[Target::Cpp17]);
    if let TargetState::Supported { files, .. } = &mut e.manifest.targets[0].state {
        files.clear();
    }
    assert!(e.manifest.validate().is_err());
}
