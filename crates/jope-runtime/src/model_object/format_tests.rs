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
