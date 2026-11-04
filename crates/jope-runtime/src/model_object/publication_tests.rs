use super::*;

#[test]
fn invalid_platt_sections_fail_before_store_mutation() {
    let (fitted, _) = binary_fixture();
    let object = ModelObject::from_fitted(&fitted).unwrap();
    let calibration = platt_fixture();
    let mut reordered = calibration.clone();
    reordered.class_order.swap(0, 1);
    assert!(object.with_platt_calibration(reordered).is_err());
    let mut negative = calibration.clone();
    negative.sigmoid.slope = -1.0;
    assert!(object.with_platt_calibration(negative).is_err());
    let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
    for value in [
        json!({"schema":"jope.platt-calibration/v2"}),
        json!({"unknown_required_field":true}),
        json!({"sigmoid":{"intercept":0.0,"slope":-1.0,"scale":1.0,"center":0.0}}),
        json!({"cross_fit_folds":1}),
        json!({"class_order":["yes","no"]}),
        json!({"score_kind":"raw-margin/v999"}),
    ] {
        let mut encoded = serde_json::to_value(&calibration).unwrap();
        for (key, value) in value.as_object().unwrap() {
            encoded[key] = value.clone();
        }
        let bytes = serde_json::to_vec(&encoded).unwrap();
        let mut manifest = calibrated.manifest.clone();
        *manifest
            .files
            .iter_mut()
            .find(|p| p.path == calibration_platt::PART)
            .unwrap() = Part::new(calibration_platt::PART, "application/json", &bytes);
        let mut parts = calibrated.parts.clone();
        parts.insert(calibration_platt::PART.into(), bytes);
        let invalid = ModelObject::new(manifest, parts, 1 << 20).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let store = directory.path().join("store");
        assert!(invalid.save(&store, 1 << 20).is_err(), "accepted {encoded}");
        assert!(!store.exists());
    }
    let mut manifest = calibrated.manifest.clone();
    manifest.files.retain(|p| p.path != calibration_platt::PART);
    let mut parts = calibrated.parts.clone();
    parts.remove(calibration_platt::PART);
    let missing = ModelObject::new(manifest, parts, 1 << 20).unwrap();
    assert!(missing.inspect().is_err());
    let mut manifest = calibrated.manifest.clone();
    manifest.required_capabilities[0] = "org.jopedime/platt-calibration/v2".into();
    let future = ModelObject::new(manifest, calibrated.parts.clone(), 1 << 20).unwrap();
    assert!(!future.inspect().unwrap().executable);
    assert!(future.verify_executable().is_err());
    let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let x: Vec<_> = (0..16).map(|i| vec![i as f64]).collect();
    let y: Vec<_> = (0..16).map(|i| i as f64).collect();
    let reg = feat_math::advanced_v3::fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
    assert!(ModelObject::from_bound(&reg)
        .unwrap()
        .with_platt_calibration(calibration)
        .is_err());
}

#[test]
fn portable_parts_refuse_device_names_trailing_dots_and_case_collisions() {
    let object = future();
    for name in [
        "NUL",
        "con.json",
        "COM1.bin",
        "lpt9",
        "AUX",
        "PRN.txt",
        "state.",
        "MANIFEST.JSON",
    ] {
        let mut manifest = object.manifest.clone();
        manifest.files[0].path = name.into();
        manifest.primary = name.into();
        assert!(
            manifest.validate(1024).is_err(),
            "accepted nonportable part {name}"
        );
    }
    let mut manifest = object.manifest.clone();
    manifest.files[1].path = "STATE.JSON".into();
    assert!(
        manifest.validate(1024).is_err(),
        "accepted case-insensitive collision"
    );
}

#[test]
fn an_interrupted_publication_resumes_without_overwriting_retained_parts() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join(object.id());
    fs::create_dir(&path).unwrap();
    fs::write(path.join("weights.bin"), &object.parts["weights.bin"]).unwrap();
    object.save(root.path(), 1024).unwrap();
    assert_eq!(ModelObject::read(&path, 1024).unwrap().parts, object.parts);
}

#[test]
fn concurrent_identical_publication_and_conflicting_recovery_preserve_bytes() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| object.save(root.path(), 1024)))
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
    });
    assert_eq!(
        ModelObject::read(&root.path().join(object.id()), 1024)
            .unwrap()
            .parts,
        object.parts
    );
    let damaged = tempfile::tempdir().unwrap();
    let path = damaged.path().join(object.id());
    fs::create_dir(&path).unwrap();
    fs::write(path.join("weights.bin"), b"retained conflicting bytes").unwrap();
    assert!(object.save(damaged.path(), 1024).is_err());
    assert_eq!(
        fs::read(path.join("weights.bin")).unwrap(),
        b"retained conflicting bytes"
    );
    assert!(!path.join(MANIFEST).exists());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let path = root.path().join(object.id());
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o700
        );
        assert_eq!(
            fs::metadata(path.join(MANIFEST))
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o600
        );
    }
}

#[test]
fn malformed_known_state_is_refused_before_creating_a_store() {
    let mut manifest = future().manifest;
    let bytes = b"not a fitted file".to_vec();
    manifest.format = APPLIANCE.into();
    manifest.required_capabilities.clear();
    manifest.primary = "model.jope".into();
    manifest.files = vec![Part::new(
        "model.jope",
        "application/vnd.jopedime.fitted",
        &bytes,
    )];
    let object = ModelObject::new(
        manifest,
        BTreeMap::from([("model.jope".into(), bytes)]),
        1024,
    )
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let store = root.path().join("store");
    assert!(object.save(&store, 1024).is_err());
    assert!(!store.exists());
}
