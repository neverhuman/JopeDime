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
