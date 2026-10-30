//! `org.jopedime/math-pipeline/v1` acceptance.
//!
//! The fixture is a fitted `(pipe X (robust_scale append=true) (ridge))` regression produced
//! outside this repository, its numeric input contract, 78 generated rows (in range, out of
//! range, every missing token, booleans, infinities, unparseable text) and the producer's own
//! predictions for those rows. Decoder and exported JavaScript must both reproduce every
//! prediction as the same f32. No training rows are in the fixture.

use jope_runtime::limits::Limits;
use jope_runtime::math_pipeline::{INPUT_PART, PIPELINE_PART};
use jope_runtime::model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA};
use jope_runtime::table::parse_csv;
use std::collections::BTreeMap;
use std::io::ErrorKind;
use std::process::Command;

const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/math-pipeline");

fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{FIXTURE}/{name}")).unwrap()
}

fn object_with(parts: BTreeMap<String, Vec<u8>>) -> anyhow::Result<ModelObject> {
    let files = parts
        .iter()
        .map(|(path, bytes)| Part::new(path, "application/json", bytes))
        .collect();
    ModelObject::new(
        Manifest {
            schema: SCHEMA.into(),
            minimum_reader_version: 1,
            format: MATH_PIPELINE.into(),
            producer: Producer {
                name: "fixture".into(),
                version: "1".into(),
                language: "rust".into(),
            },
            primary: PIPELINE_PART.into(),
            required_capabilities: vec![],
            files,
            metadata: serde_json::json!({"target": "compressive_strength_mpa"}),
        },
        parts,
        Limits::default().max_model_bytes,
    )
}

fn parts(input: Vec<u8>) -> BTreeMap<String, Vec<u8>> {
    BTreeMap::from([
        (PIPELINE_PART.to_string(), fixture("pipeline.json")),
        (INPUT_PART.to_string(), input),
    ])
}

fn expected() -> Vec<f32> {
    String::from_utf8(fixture("expected.csv"))
        .unwrap()
        .lines()
        .skip(1)
        .map(|line| line.parse::<f64>().unwrap() as f32)
        .collect()
}

#[test]
fn decoder_reproduces_the_producer_for_every_row() {
    let object = object_with(parts(fixture("input.json"))).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = object
        .save(dir.path(), Limits::default().max_model_bytes)
        .unwrap();
    let object = ModelObject::read(&path, Limits::default().max_model_bytes).unwrap();
    let inspection = object.inspect().unwrap();
    assert!(inspection.executable && inspection.integrity_verified);

    let table = parse_csv(&fixture("rows.csv"), &Limits::default()).unwrap();
    let (values, ignored) = object
        .math_pipeline()
        .unwrap()
        .predict_table(&table)
        .unwrap();
    assert!(ignored.is_empty());
    let expected = expected();
    assert_eq!(values.len(), expected.len());
    for (row, (got, want)) in values.iter().zip(&expected).enumerate() {
        assert_eq!(*got as f32, *want, "row {row}");
        assert_eq!(
            f64::from(*got as f32),
            *got,
            "row {row} is not an f32 value"
        );
    }
}

#[test]
fn exported_javascript_reproduces_the_producer_for_every_row() {
    let object = object_with(parts(fixture("input.json"))).unwrap();
    let source = String::from_utf8(object.export_source("javascript").unwrap()).unwrap();
    let table = parse_csv(&fixture("rows.csv"), &Limits::default()).unwrap();
    let rows = serde_json::to_string(&table.rows).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let program = dir.path().join("program.js");
    std::fs::write(
        &program,
        format!(
            "{source}\nprocess.stdout.write(JSON.stringify(predictRows({rows})));\n\
             if(MODEL_COLUMNS.join()!=={:?})throw new Error('column order');\n",
            table.headers.join(",")
        ),
    )
    .unwrap();
    let output = match Command::new("node").arg(&program).output() {
        Ok(output) => output,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            assert!(
                std::env::var_os("JOPE_PARITY_STRICT").is_none_or(|v| v != "1"),
                "math-pipeline JavaScript: node is unavailable and JOPE_PARITY_STRICT=1"
            );
            eprintln!("math-pipeline JavaScript SKIP: node is unavailable");
            return;
        }
        Err(error) => panic!("could not start node: {error}"),
    };
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let values: Vec<f64> = serde_json::from_slice(&output.stdout).unwrap();
    let expected = expected();
    assert_eq!(values.len(), expected.len());
    for (row, (got, want)) in values.iter().zip(&expected).enumerate() {
        assert_eq!(*got, f64::from(*want), "row {row}");
    }
}

#[test]
fn decoder_refuses_what_it_cannot_reproduce() {
    let input: serde_json::Value = serde_json::from_slice(&fixture("input.json")).unwrap();
    let refused = |edit: &dyn Fn(&mut serde_json::Value), why: &str| {
        let mut changed = input.clone();
        edit(&mut changed);
        let object = object_with(parts(serde_json::to_vec(&changed).unwrap())).unwrap();
        let inspection = object.inspect();
        let error = object.math_pipeline().unwrap_err();
        assert!(format!("{error:#}").contains(why), "{why}: {error:#}");
        assert!(inspection.is_err(), "{why}: inspection accepted it");
    };
    refused(&|v| v["imp_rows"] = serde_json::json!([]), "imp_rows");
    refused(&|v| v["task"] = "binary".into(), "regression only");
    refused(
        &|v| v["selected"][0] = 99.into(),
        "selected must be distinct",
    );
    refused(
        &|v| v["selected"][1] = v["selected"][0].clone(),
        "selected must be distinct",
    );
    refused(
        &|v| {
            v["selected"].as_array_mut().unwrap().pop();
        },
        "does not read 7 selected columns",
    );
    refused(
        &|v| v["columns"][0]["max"] = v["columns"][0]["min"].clone(),
        "max <= min",
    );

    let mut extra = parts(fixture("input.json"));
    extra.insert("notes.json".into(), b"{}".to_vec());
    let error = object_with(extra).unwrap().math_pipeline().unwrap_err();
    assert!(
        format!("{error:#}").contains("exactly the parts"),
        "{error:#}"
    );

    let object = object_with(parts(fixture("input.json"))).unwrap();
    let table = parse_csv(b"cement,water\n1,2\n", &Limits::default()).unwrap();
    let error = object
        .math_pipeline()
        .unwrap()
        .predict_table(&table)
        .unwrap_err();
    assert!(
        format!("{error:#}").contains("input lacks column"),
        "{error:#}"
    );
    for language in ["python", "rust"] {
        assert!(object.export_source(language).is_err());
    }
    assert!(object.bound().is_err());
    assert!(object.fitted().is_err());
}

fn numeric_object(version: Option<u32>, min: f64, max: f64, target_scale: f64) -> ModelObject {
    let mut input = serde_json::json!({
        "task":"regression", "columns":[{"name":"x","min":min,"max":max,"constant":false}],
        "selected":[0], "target":{"name":"y","min":0.0,"scale":target_scale}
    });
    if let Some(version) = version {
        input["prediction_version"] = version.into();
    }
    let state = serde_json::json!({
        "dsl":"(pipe X (ridge))",
        "model":{"Pipeline":{"target":null,"ops":[],"head":{"Ridge":{
            "means":[0.0],"scales":[1.0],"w":[0.0,1.0]
        }}}}
    });
    object_with(BTreeMap::from([
        (INPUT_PART.into(), serde_json::to_vec(&input).unwrap()),
        (PIPELINE_PART.into(), serde_json::to_vec(&state).unwrap()),
    ]))
    .unwrap()
}

#[test]
fn legacy_precision_and_non_finite_replacement_are_retained() {
    let old = numeric_object(None, 0.0, 1.0, 1.0);
    assert_eq!(
        old.id(),
        "sha256-47aa2c52a79cead4cbdfef18e89ddbc8ab95aafabc12d83a10310f947e9e569b"
    );
    let mut model = old.math_pipeline().unwrap();
    let raw = 0.123456789012345_f64;
    let got = model
        .predict_cells(&[vec![Some("0.123456789012345")]])
        .unwrap()[0];
    assert_eq!(got, f64::from(raw as f32));
    assert!((got - raw).abs() > 1e-10);
    if let feat_math::model::Model::Pipeline { head, .. } = &mut model.state.model {
        if let feat_math::heads::FittedHead::Ridge(r) = head.as_mut() {
            r.scales[0] = 0.0;
        }
    }
    assert_eq!(model.predict_cells(&[vec![Some("0")]]).unwrap(), vec![0.0]);
}

#[test]
fn version_two_preserves_input_output_and_scale_precision() {
    for (min, max, cell, expected) in [
        (0.0, 1.0, "0.123456789012345", 0.123456789012345),
        (1.0, 1.000000000000002, "1.000000000000001", 5.0 / 9.0),
        (0.0, 1e40, "5e39", 0.5),
        (0.0, 1e-40, "5e-41", 0.5),
    ] {
        let object = numeric_object(Some(2), min, max, 1.0);
        let model = object.math_pipeline().unwrap();
        let got = model.predict_cells(&[vec![Some(cell)]]).unwrap()[0];
        assert!(
            (got - expected).abs() <= 1e-15,
            "{cell}: {got} != {expected}"
        );
    }
    let mut model = numeric_object(Some(2), 0.0, 1.0, 1.0)
        .math_pipeline()
        .unwrap();
    model.input.target.min = 1.000000000000001;
    model.input.target.scale = 2.000000000000001;
    assert_eq!(
        model.predict_cells(&[vec![Some("0.5")]]).unwrap()[0],
        1.000000000000001_f64 + 0.5 * 2.000000000000001_f64
    );
}

#[test]
fn version_two_non_finite_prediction_is_a_candidate_failure() {
    let mut model = numeric_object(Some(2), 0.0, 1.0, 1.0)
        .math_pipeline()
        .unwrap();
    if let feat_math::model::Model::Pipeline { head, .. } = &mut model.state.model {
        if let feat_math::heads::FittedHead::Ridge(r) = head.as_mut() {
            r.scales[0] = 0.0; // finite persisted parameters, but 0/0 yields NaN at prediction
        }
    }
    let error = model
        .predict_cells(&[vec![Some("0.5")], vec![Some("0")]])
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<jope_runtime::Failure>().unwrap().kind,
        jope_runtime::error::FailureKind::Numerical
    );
    assert!(format!("{error:#}").contains("FITNESS_UNDEFINED"));
    assert!(format!("{error:#}").contains("row 1"));
    let mut model = numeric_object(Some(2), 0.0, 1.0, 1e308)
        .math_pipeline()
        .unwrap();
    if let feat_math::model::Model::Pipeline { head, .. } = &mut model.state.model {
        if let feat_math::heads::FittedHead::Ridge(r) = head.as_mut() {
            r.w[1] = 60.0;
        }
    }
    assert!(
        model.predict_cells(&[vec![Some("1")]]).is_err(),
        "target inverse overflow"
    );
    assert!(numeric_object(Some(2), 0.0, 1.0, 1.0)
        .math_pipeline()
        .unwrap()
        .predict_cells(&[vec![Some("0.5")]])
        .is_ok());
}

#[test]
fn old_and_new_objects_reload_without_identity_or_prediction_migration() {
    let old = numeric_object(None, 0.0, 1.0, 1.0);
    let new = numeric_object(Some(2), 0.0, 1.0, 1.0);
    assert_ne!(old.id(), new.id());
    let dir = tempfile::tempdir().unwrap();
    for (object, expected_version) in [(&old, 1), (&new, 2)] {
        let path = object
            .save(dir.path(), Limits::default().max_model_bytes)
            .unwrap();
        let loaded = ModelObject::read(&path, Limits::default().max_model_bytes).unwrap();
        assert_eq!(loaded.id(), object.id());
        assert_eq!(loaded.manifest_bytes(), object.manifest_bytes());
        assert_eq!(loaded.parts(), object.parts());
        let model = loaded.math_pipeline().unwrap();
        assert_eq!(model.input.prediction_version, expected_version);
        let prediction = model
            .predict_cells(&[vec![Some("0.123456789012345")]])
            .unwrap()[0];
        let expected = if expected_version == 1 {
            f64::from(0.123456789012345_f64 as f32)
        } else {
            0.123456789012345
        };
        assert_eq!(prediction, expected);
    }
    assert_eq!(
        old.id(),
        "sha256-47aa2c52a79cead4cbdfef18e89ddbc8ab95aafabc12d83a10310f947e9e569b"
    );
}

#[test]
fn new_contract_constructor_records_f64_and_validates_parameters() {
    use jope_runtime::math_pipeline::{
        InputContract, NumericColumn, TargetScale, F64_PREDICTION_VERSION,
    };
    let columns = vec![NumericColumn {
        name: "x".into(),
        min: 0.0,
        max: 1e40,
        constant: false,
    }];
    let target = TargetScale {
        name: "y".into(),
        min: 0.0,
        scale: 1.0,
    };
    let input = InputContract::new(columns.clone(), vec![0], target.clone()).unwrap();
    assert_eq!(input.prediction_version, F64_PREDICTION_VERSION);
    let mut data = numeric_object(None, 0.0, 1.0, 1.0).parts().clone();
    data.insert(INPUT_PART.into(), serde_json::to_vec(&input).unwrap());
    assert_eq!(
        object_with(data)
            .unwrap()
            .math_pipeline()
            .unwrap()
            .predict_cells(&[vec![Some("5e39")]])
            .unwrap(),
        vec![0.5]
    );
    assert!(InputContract::new(columns, vec![1], target).is_err());
}

#[test]
fn new_section_versions_are_strict_and_old_f32_range_validation_is_retained() {
    let old = numeric_object(None, 0.0, 1.0, 1.0);
    for invalid in [
        serde_json::json!(0),
        serde_json::json!(3),
        serde_json::json!("2"),
        serde_json::json!(null),
        serde_json::json!(2.5),
    ] {
        let mut parts = old.parts().clone();
        let mut input: serde_json::Value = serde_json::from_slice(&parts[INPUT_PART]).unwrap();
        input["prediction_version"] = invalid;
        parts.insert(INPUT_PART.into(), serde_json::to_vec(&input).unwrap());
        assert!(object_with(parts).unwrap().math_pipeline().is_err());
    }
    assert!(numeric_object(None, 0.0, 1e40, 1.0)
        .math_pipeline()
        .is_err());
    assert!(numeric_object(Some(2), -1e308, 1e308, 1.0)
        .math_pipeline()
        .is_err());
    let legacy = numeric_object(Some(1), 0.0, 1.0, 1.0)
        .math_pipeline()
        .unwrap();
    assert_eq!(
        legacy
            .predict_cells(&[vec![Some("0.123456789012345")]])
            .unwrap()[0],
        f64::from(0.123456789012345_f64 as f32)
    );
    let mut parts = numeric_object(Some(2), 0.0, 1.0, 1.0).parts().clone();
    let mut input: serde_json::Value = serde_json::from_slice(&parts[INPUT_PART]).unwrap();
    input["future_field"] = true.into();
    parts.insert(INPUT_PART.into(), serde_json::to_vec(&input).unwrap());
    assert!(object_with(parts).unwrap().math_pipeline().is_err());
}

#[test]
fn version_two_javascript_preserves_f64_and_refuses_non_finite_outputs() {
    for (max, cells) in [
        (
            1.0,
            vec![
                Some("0.123456789012345"),
                None,
                Some("true"),
                Some("NaN"),
                Some("invalid"),
            ],
        ),
        (1e40, vec![Some("5e39")]),
        (1e-40, vec![Some("5e-41")]),
    ] {
        let model = numeric_object(Some(2), 0.0, max, 1.0)
            .math_pipeline()
            .unwrap();
        let rows: Vec<_> = cells.into_iter().map(|cell| vec![cell]).collect();
        let expected = model.predict_cells(&rows).unwrap();
        let source = String::from_utf8(model.export_javascript().unwrap()).unwrap();
        let program = format!(
            "{source}\nprocess.stdout.write(JSON.stringify(predictRows({})));",
            serde_json::to_string(&rows).unwrap()
        );
        let output = Command::new("node")
            .args(["-e", &program])
            .output()
            .expect("Node is required for version2 parity");
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let values: Vec<f64> = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(values, expected);
    }
    let mut model = numeric_object(Some(2), 0.0, 1.0, 1e308)
        .math_pipeline()
        .unwrap();
    model.input.target.min = 1e308;
    assert!(model.predict_cells(&[vec![Some("1")]]).is_err());
    let source = String::from_utf8(model.export_javascript().unwrap()).unwrap();
    let program=format!("{source}\ntry{{predictRows([[\"1\"]]);process.exit(2)}}catch(e){{if(!String(e).includes(\"prediction undefined\"))throw e;process.stdout.write(\"refused\")}}");
    let output = Command::new("node")
        .args(["-e", &program])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(output.stdout, b"refused");
}
