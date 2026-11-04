use super::*;

#[test]
fn writer_reader_accounts_for_all_recipe_and_fitted_components() {
    let dir = scratch();
    let mut targets = Target::RECIPES.to_vec();
    targets.push(Target::Cpp17Fitted);
    targets.push(Target::TypeScript);
    targets.push(Target::Go);
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
