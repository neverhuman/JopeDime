use super::*;

#[test]
fn math_pipeline_decode_and_javascript_export_refuse_a_missing_required_platt_part() {
    use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};

    let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        (
            PIPELINE_PART.into(),
            include_bytes!("../../tests/fixtures/math-pipeline/pipeline.json").to_vec(),
        ),
        (
            INPUT_PART.into(),
            include_bytes!("../../tests/fixtures/math-pipeline/input.json").to_vec(),
        ),
    ]);
    let mut manifest = Manifest {
        schema: SCHEMA.into(),
        minimum_reader_version: 1,
        format: MATH_PIPELINE.into(),
        producer: Producer::default(),
        primary: PIPELINE_PART.into(),
        required_capabilities: vec![],
        files: parts
            .iter()
            .map(|(path, bytes)| Part::new(path, "application/json", bytes))
            .collect(),
        metadata: Value::Null,
    };
    let legacy = ModelObject::new(manifest.clone(), parts.clone(), 1 << 20).unwrap();
    assert!(legacy.inspect().unwrap().executable);
    legacy.verify_executable().unwrap();
    legacy.math_pipeline().unwrap();
    assert!(!legacy.export_source("javascript").unwrap().is_empty());

    manifest
        .required_capabilities
        .push(calibration_platt::CAPABILITY.into());
    let invalid = ModelObject::new(manifest, parts, 1 << 20).unwrap();
    assert!(invalid.platt_calibration().is_err());
    assert!(invalid.inspect().is_err());
    assert!(invalid.verify_executable().is_err());
    let decoded = invalid.math_pipeline();
    let exported = invalid.export_source("javascript");
    let rust_exported = invalid.export_source("rust");
    assert!(
        decoded.is_err() && exported.is_err() && rust_exported.is_err(),
        "missing required Platt part accepted: math_pipeline={}, javascript_export={}, rust_export={}",
        decoded.is_ok(),
        exported.is_ok(),
        rust_exported.is_ok()
    );
    for error in [
        decoded.unwrap_err(),
        exported.unwrap_err(),
        rust_exported.unwrap_err(),
    ] {
        assert_eq!(crate::error::kind_of(&error), crate::FailureKind::Integrity);
        assert!(error.to_string().contains("must occur together"));
    }
}

#[test]
fn loaded_math_pipeline_object_exports_rust_and_legacy_version_is_refused() {
    use crate::math_pipeline::{INPUT_PART, PIPELINE_PART};
    use std::{fs, process::Command};

    fn fixture_object(input: Vec<u8>) -> ModelObject {
        let parts = BTreeMap::from([
            (
                PIPELINE_PART.to_owned(),
                include_bytes!("../../tests/fixtures/math-pipeline/pipeline.json").to_vec(),
            ),
            (INPUT_PART.to_owned(), input),
        ]);
        let manifest = Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer::default(),
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files: parts
                .iter()
                .map(|(path, bytes)| Part::new(path, "application/json", bytes))
                .collect(),
            metadata: Value::Null,
        };
        ModelObject::new(manifest, parts, 1 << 20).unwrap()
    }

    let mut input: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../tests/fixtures/math-pipeline/input.json"
    ))
    .unwrap();
    input["prediction_version"] = serde_json::json!(2);
    let object = fixture_object(serde_json::to_vec(&input).unwrap());
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1 << 20).unwrap();
    let loaded = ModelObject::read(&path, 1 << 20).unwrap();
    let source = loaded.export_source("rust").unwrap();
    let source = String::from_utf8(source).expect("Rust export is UTF-8");
    let pipeline = loaded.math_pipeline().unwrap();
    let table = crate::table::parse_csv(
        include_bytes!("../../tests/fixtures/math-pipeline/rows.csv"),
        &crate::limits::Limits::default(),
    )
    .unwrap();
    let (expected, _) = pipeline.predict_table(&table).unwrap();
    let headers = format!(
        "[{}]",
        table
            .headers
            .iter()
            .map(|header| format!("{header:?}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    let rows = format!(
        "vec![{}]",
        table
            .rows
            .iter()
            .map(|row| format!(
                "vec![{}]",
                row.iter()
                    .map(|cell| format!("{:?}", cell.as_deref()))
                    .collect::<Vec<_>>()
                    .join(",")
            ))
            .collect::<Vec<_>>()
            .join(",")
    );
    let expected = format!("vec!{:?}", expected);
    let program = format!(
        "{source}\nfn main() {{ let got = predict_table(&{headers}, &{rows}).unwrap(); let expected = {expected}; assert_eq!(got.len(), expected.len()); for (actual, expected) in got.iter().zip(expected) {{ assert!((actual - expected).abs() <= 1e-12 + 1e-10 * expected.abs()); }} }}"
    );
    let out_dir = std::env::current_dir()
        .unwrap()
        .join("target/exp-03-f01-rust-export");
    fs::create_dir_all(&out_dir).unwrap();
    let source_path = out_dir.join("fixture_export.rs");
    let binary_path = out_dir.join("fixture_export");
    fs::write(&source_path, program).unwrap();
    let compile = Command::new("rustc")
        .args(["--edition=2021", "-D", "warnings"])
        .arg(&source_path)
        .arg("-o")
        .arg(&binary_path)
        .output()
        .expect("run rustc for the exported module");
    assert!(
        compile.status.success(),
        "rustc failed: {}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let run = Command::new(&binary_path)
        .output()
        .expect("run the exported module");
    assert!(
        run.status.success(),
        "exported Rust failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let legacy =
        fixture_object(include_bytes!("../../tests/fixtures/math-pipeline/input.json").to_vec());
    let error = legacy.export_source("rust").unwrap_err();
    assert!(error.to_string().contains("prediction_version=2"));
}

fn binary_fixture() -> (FittedModel, feat_math::advanced_v3::RawTable) {
    use crate::{api, budget::Budget, fitted::CreatedBy, limits::Limits, table, task};
    let mut csv = String::from("x,y\n");
    for i in 0..40 {
        csv.push_str(&format!(
            "{},{}\n",
            i % 10,
            if i % 10 > 4 { "yes" } else { "no" }
        ));
    }
    let parsed = table::parse_csv(csv.as_bytes(), &Limits::default()).unwrap();
    let training = table::training(
        &parsed,
        "y",
        task::TaskChoice::Fixed(task::TaskKind::Binary),
        &Limits::default(),
    )
    .unwrap();
    let fitted = api::fit(
        &training,
        &api::FitOptions {
            method: "logistic".into(),
            ..Default::default()
        },
        &Budget::new(None, 1),
        &parsed.sha256,
        CreatedBy::default(),
    )
    .unwrap()
    .fitted;
    let query = table::parse_csv(b"x\n0\n3\n6\n9\n", &Limits::default()).unwrap();
    (
        fitted,
        table::prediction(&query, &training.schema, &training.target)
            .unwrap()
            .0,
    )
}

fn platt_fixture() -> calibration_platt::Calibration {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut folds = Vec::new();
    for fold in 0..3 {
        for i in 0..20 {
            scores.push(if i < 10 { 0.001 } else { 0.999 });
            labels.push(if i % 10 < (if i < 10 { 2 } else { 8 }) {
                "yes"
            } else {
                "no"
            });
            folds.push(fold);
        }
    }
    calibration_platt::cross_fit(&scores, &labels, ["no", "yes"], &folds)
        .unwrap()
        .calibration
}

#[test]
fn a_platt_part_without_its_required_capability_cannot_be_silently_stripped() {
    let (fitted, _) = binary_fixture();
    let object = ModelObject::from_fitted(&fitted).unwrap();
    let bytes = serde_json::to_vec(&platt_fixture()).unwrap();
    let mut manifest = object.manifest.clone();
    let mut parts = object.parts.clone();
    manifest.files.push(Part::new(
        calibration_platt::PART,
        "application/json",
        &bytes,
    ));
    parts.insert(calibration_platt::PART.into(), bytes);
    let object = ModelObject::new(manifest, parts, 1 << 20).unwrap();
    assert!(
        object.fitted().is_err(),
        "calibration part was discarded while exporting the bare fitted head"
    );
    assert!(object.bound().is_err());
}

#[test]
fn platt_object_round_trip_predicts_with_the_stored_map_and_retains_native_bytes() {
    use crate::methods::native::Predictions;
    let (fitted, query) = binary_fixture();
    let raw = fitted.predict(&query).unwrap();
    let Predictions::Probabilities { classes, values } = &raw else {
        panic!("binary fixture");
    };
    let calibration = platt_fixture();
    let expected = Predictions::Probabilities {
        classes: classes.clone(),
        values: values
            .iter()
            .map(|row| {
                let p = calibration.apply(row[1]).unwrap();
                vec![1.0 - p, p]
            })
            .collect(),
    };
    assert_ne!(
        serde_json::to_vec(&raw).unwrap(),
        serde_json::to_vec(&expected).unwrap()
    );
    for object in [
        ModelObject::from_fitted(&fitted).unwrap(),
        ModelObject::from_bound(&fitted.model).unwrap(),
    ] {
        let before = object.parts.clone();
        let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
        for (name, bytes) in before {
            assert_eq!(calibrated.parts[&name], bytes);
        }
        assert_eq!(object.platt_calibration().unwrap(), None);
        assert_eq!(
            calibrated.manifest.required_capabilities,
            vec![calibration_platt::CAPABILITY]
        );
        assert_eq!(
            calibrated.platt_calibration().unwrap(),
            Some(calibration.clone())
        );
        assert!(calibrated.inspect().unwrap().executable);
        assert_eq!(
            serde_json::to_vec(&calibrated.predict(&query).unwrap()).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        assert!(calibrated.fitted().is_err());
        assert!(calibrated.bound().is_err());
        for language in ["python", "rust", "javascript"] {
            assert!(calibrated.export_source(language).is_err());
        }
        assert!(calibrated
            .with_platt_calibration(calibration.clone())
            .is_err());
        let directory = tempfile::tempdir().unwrap();
        let path = calibrated.save(directory.path(), 1 << 20).unwrap();
        let loaded = ModelObject::read(&path, 1 << 20).unwrap();
        assert_eq!(loaded.manifest_bytes, calibrated.manifest_bytes);
        assert_eq!(loaded.parts, calibrated.parts);
        assert_eq!(
            serde_json::to_vec(&loaded.predict(&query).unwrap()).unwrap(),
            serde_json::to_vec(&expected).unwrap()
        );
        assert!(load_fitted(&path, 1 << 20).is_err());
    }
    let legacy = ModelObject::from_fitted(&fitted).unwrap();
    assert_eq!(
        legacy.fitted().unwrap().to_bytes().unwrap(),
        fitted.to_bytes().unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&legacy.predict(&query).unwrap()).unwrap(),
        serde_json::to_vec(&raw).unwrap()
    );
}

#[test]
fn calibrated_input_reports_the_head_contract_without_exporting_it() {
    use crate::{limits::Limits, table};
    let (fitted, query) = binary_fixture();
    let calibration = platt_fixture();
    let legacy = ModelObject::from_fitted(&fitted).unwrap();
    assert!(legacy.calibrated_input().unwrap().is_none());
    let csv = table::parse_csv(b"x\n0\n3\n6\n9\n", &Limits::default()).unwrap();
    let labelled = table::parse_csv(b"x,y\n0,no\n", &Limits::default()).unwrap();
    for (object, appliance) in [
        (ModelObject::from_fitted(&fitted).unwrap(), true),
        (ModelObject::from_bound(&fitted.model).unwrap(), false),
    ] {
        let calibrated = object.with_platt_calibration(calibration.clone()).unwrap();
        let input = calibrated.calibrated_input().unwrap().unwrap();
        assert_eq!(input.schema, fitted.model.schema);
        assert_eq!(input.genome_sha256, fitted.model.genome_sha256);
        assert_eq!(input.header.as_ref(), appliance.then_some(&fitted.header));
        let (table, ignored) = input.table(&csv).unwrap();
        assert!(ignored.is_empty());
        assert_eq!(
            serde_json::to_vec(&calibrated.predict(&table).unwrap()).unwrap(),
            serde_json::to_vec(&calibrated.predict(&query).unwrap()).unwrap()
        );
        match input.table(&labelled) {
            Ok((_, ignored)) => assert!(appliance && ignored == ["y"]),
            Err(_) => assert!(!appliance, "an appliance head ignores its target"),
        }
    }
}

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

#[test]
fn duplicate_json_metadata_is_refused_before_cross_language_disagreement() {
    let object = future();
    let text = String::from_utf8(object.manifest_bytes).unwrap();
    let duplicated = text.replace(
        "\"extension\": {",
        "\"extension\": {\"unicode\":\"other\"}, \"extension\": {",
    );
    assert!(ModelObject::decode(duplicated.into_bytes(), object.parts, 1024).is_err());
}

#[test]
fn new_gp_fields_cannot_be_silently_dropped_by_an_older_decoder() {
    let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let x = (0..16)
        .map(|i| vec![i as f64, (i % 3) as f64])
        .collect::<Vec<_>>();
    let y = (0..16).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
    let fitted = feat_math::advanced_v3::fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
    let mut value = serde_json::to_value(&fitted).unwrap();
    let head = value
        .pointer_mut("/predictor/Legacy/Pipeline/head/Ridge")
        .unwrap();
    head.as_object_mut()
        .unwrap()
        .insert("future_required_state".into(), json!({"offset": 123}));
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut manifest = ModelObject::from_bound(&fitted).unwrap().manifest;
    manifest.files = vec![Part::new("state.json", "application/json", &bytes)];
    let object = ModelObject::new(
        manifest,
        BTreeMap::from([("state.json".into(), bytes)]),
        1 << 20,
    )
    .unwrap();
    assert!(
        object.bound().is_err(),
        "older decoder discarded required fitted state"
    );
}

fn future() -> ModelObject {
    let state = b"{\"variant\":\"future\",\"operators\":[\"org.example/solver/v7\"]}".to_vec();
    let buffer = vec![0, 255, 0, 128, 1, 7];
    ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: "org.example/future-gp/v7".into(),
            producer: Producer::default(),
            primary: "state.json".into(),
            required_capabilities: vec!["org.example/solver/v7".into()],
            files: vec![
                Part::new("state.json", "application/json", &state),
                Part::new("weights.bin", "application/octet-stream", &buffer),
            ],
            metadata: json!({"extension": {"unicode": "λ 日本", "float_bits": "8000000000000000"}}),
        },
        BTreeMap::from([("state.json".into(), state), ("weights.bin".into(), buffer)]),
        1024,
    )
    .unwrap()
}

#[test]
fn future_variants_retain_all_bytes_but_require_their_decoder() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    let loaded = ModelObject::read(&path, 1024).unwrap();
    assert_eq!(object.manifest_bytes, loaded.manifest_bytes);
    assert_eq!(object.parts, loaded.parts);
    let inspection = loaded.inspect().unwrap();
    assert!(inspection.integrity_verified);
    assert!(!inspection.executable);
    assert_eq!(inspection.unsupported_capabilities.len(), 2);
    assert_eq!(
        crate::error::kind_of(&loaded.bound().unwrap_err()),
        crate::FailureKind::Backend
    );
    assert_eq!(object.save(root.path(), 1024).unwrap(), path);
}

#[test]
fn manifest_identity_uses_retained_bytes_including_foreign_whitespace() {
    let object = future();
    let mut bytes = b" \n".to_vec();
    bytes.extend_from_slice(&serde_json::to_vec(&object.manifest).unwrap());
    bytes.push(b'\n');
    let foreign = ModelObject::decode(bytes.clone(), object.parts.clone(), 1024).unwrap();
    assert_ne!(foreign.id, object.id);
    assert_eq!(foreign.id, format!("sha256-{}", sha256_hex(&bytes)));
    let root = tempfile::tempdir().unwrap();
    let path = foreign.save(root.path(), 1024).unwrap();
    assert_eq!(
        ModelObject::read(&path, 1024).unwrap().manifest_bytes,
        bytes
    );
}

#[test]
fn truncation_substitution_wrong_directory_and_uncommitted_store_are_refused() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    fs::write(path.join("weights.bin"), [0, 255]).unwrap();
    assert!(ModelObject::read(&path, 1024).is_err());
    assert!(object.save(root.path(), 1024).is_err());
    assert_eq!(fs::read(path.join("weights.bin")).unwrap(), [0, 255]);
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    let renamed = root.path().join(format!("sha256-{}", "0".repeat(64)));
    fs::rename(&path, &renamed).unwrap();
    assert!(ModelObject::read(&renamed, 1024).is_err());
    fs::remove_file(renamed.join(MANIFEST)).unwrap();
    assert!(ModelObject::read(&renamed, 1024).is_err());
}

#[test]
fn envelope_bounds_and_paths_are_checked_before_part_access() {
    let object = future();
    for bad_name in [
        "../escape",
        "/abs",
        "x/y",
        "x\\y",
        ".hidden",
        "C:state",
        MANIFEST,
    ] {
        let mut manifest = object.manifest.clone();
        manifest.files[0].path = bad_name.into();
        assert!(manifest.validate(1024).is_err(), "{bad_name}");
    }
    let mut manifest = object.manifest.clone();
    manifest.minimum_reader_version = 2;
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.format = "/invalid/".into();
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.files.push(manifest.files[0].clone());
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest
        .required_capabilities
        .push(manifest.required_capabilities[0].clone());
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.files[0].bytes = u64::MAX;
    assert!(manifest.validate(u64::MAX).is_err());
    assert!(object.manifest.validate(2).is_err());
    assert!(ModelObject::decode(
        vec![b' '; MAX_MANIFEST_BYTES as usize + 1],
        BTreeMap::new(),
        1024
    )
    .is_err());
    let mut parts = object.parts.clone();
    parts.insert("unlisted.bin".into(), vec![]);
    assert!(ModelObject::decode(object.manifest_bytes.clone(), parts, 1024).is_err());
}

#[test]
fn unknown_required_capabilities_are_never_ignored_for_a_known_format() {
    let object = future();
    let mut manifest = object.manifest.clone();
    manifest.format = APPLIANCE.into();
    let object = ModelObject::new(manifest, object.parts, 1024).unwrap();
    assert!(!object.inspect().unwrap().executable);
    assert!(object.fitted().is_err());
}

#[test]
fn atomic_create_only_exports_preserve_existing_files() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("export.bin");
    write_new(&path, b"original").unwrap();
    assert!(write_new(&path, b"replacement").is_err());
    assert_eq!(fs::read(path).unwrap(), b"original");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn symlinked_parts_and_object_directories_are_refused() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o700
    );
    let alias = root.path().join("alias");
    symlink(&path, &alias).unwrap();
    assert!(ModelObject::read(&alias, 1024).is_err());
    let payload = path.join("weights.bin");
    fs::rename(&payload, root.path().join("external.bin")).unwrap();
    symlink(root.path().join("external.bin"), &payload).unwrap();
    assert!(ModelObject::read(&path, 1024).is_err());
}
