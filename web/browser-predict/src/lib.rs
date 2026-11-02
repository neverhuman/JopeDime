//! First-party, prediction-only browser ABI over the native model-object reader.
//! Models are verified data; the caller cannot supply executable JS or WASM.
use anyhow::{bail, ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use feat_math::advanced_v3::{RawTable, TaskKind};
use jope_runtime::{
    math_pipeline::MathPipeline,
    methods::native::Predictions,
    model_object::{self, ModelObject},
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::{cell::RefCell, collections::BTreeMap};

const MAX_MODEL: u64 = 32 << 20;
const MAX_MESSAGE: usize = 48 << 20;
const MAX_COLUMNS: usize = 256;

thread_local! {
    static INPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static OUTPUT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static MODEL: RefCell<Option<Loaded>> = const { RefCell::new(None) };
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    manifest: String,
    parts: BTreeMap<String, String>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    values: Vec<Option<String>>,
}

enum Predictor {
    Math(MathPipeline),
    Object(ModelObject),
}
struct Loaded {
    predictor: Predictor,
    columns: Vec<String>,
    description: Value,
}

impl Loaded {
    fn load(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_MESSAGE,
            "browser model exceeds its byte limit"
        );
        let envelope: Envelope = serde_json::from_slice(bytes)?;
        ensure!(
            envelope.manifest.len() as u64 <= model_object::MAX_MANIFEST_BYTES,
            "model manifest is too large"
        );
        ensure!(
            envelope.parts.len() <= model_object::MAX_FILES,
            "model has too many parts"
        );
        let mut total = 0usize;
        let parts = envelope
            .parts
            .into_iter()
            .map(|(name, text)| {
                let bytes = STANDARD
                    .decode(text)
                    .context("invalid model part encoding")?;
                total = total
                    .checked_add(bytes.len())
                    .context("model size overflow")?;
                ensure!(total <= MAX_MODEL as usize, "browser model exceeds 32 MiB");
                Ok((name, bytes))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let object = ModelObject::decode(envelope.manifest.into_bytes(), parts, MAX_MODEL)?;
        object.verify_executable()?;
        let id = object.id().to_owned();
        let format = object.manifest().format.clone();
        let (predictor, columns, target, task, policy) = if format == model_object::MATH_PIPELINE {
            let model = object.math_pipeline()?;
            let columns = model
                .input
                .columns
                .iter()
                .map(|c| json!({"name":c.name,"kind":"numeric","min":c.min,"max":c.max}))
                .collect::<Vec<_>>();
            let target = model.input.target.name.clone();
            (Predictor::Math(model), columns, target, "regression".to_owned(), "Values outside the training range are clamped. Blank inputs use the model's saved missing-value policy.")
        } else {
            let (schema, target, task) = if format == model_object::APPLIANCE {
                let fitted = object.fitted()?;
                (
                    fitted.model.schema.columns.clone(),
                    fitted.header.target.clone(),
                    fitted.header.task.clone(),
                )
            } else if format == model_object::BOUND {
                let bound = object.bound()?;
                ensure!(bound.task.kind == TaskKind::Regression, "This bound classification model does not record its class-label mapping. Use its native runtime.");
                (
                    bound.schema.columns.clone(),
                    "Prediction".to_owned(),
                    "regression".to_owned(),
                )
            } else {
                bail!("This model format has no browser predictor.")
            };
            let columns = schema
                .iter()
                .map(|c| json!({"name":c.name,"kind":c.kind}))
                .collect::<Vec<_>>();
            (Predictor::Object(object), columns, target, task, "Inputs use the model's saved preprocessing. Blank inputs use its saved missing-value policy.")
        };
        ensure!(
            !columns.is_empty() && columns.len() <= MAX_COLUMNS,
            "browser prediction supports 1–256 input columns"
        );
        let names = columns
            .iter()
            .map(|c| {
                c["name"]
                    .as_str()
                    .context("missing input name")
                    .map(str::to_owned)
            })
            .collect::<Result<Vec<_>>>()?;
        let description = json!({"id":id,"format":format,"columns":columns,"target":target,"task":task,"inputPolicy":policy,"engine":"JopeDime","runtimeSource":option_env!("JOPE_BROWSER_SOURCE_SHA").unwrap_or("development")});
        Ok(Self {
            predictor,
            columns: names,
            description,
        })
    }
    fn predict(&self, bytes: &[u8]) -> Result<Value> {
        ensure!(bytes.len() <= 1 << 20, "prediction row is too large");
        let row: Row = serde_json::from_slice(bytes)?;
        ensure!(
            row.values.len() == self.columns.len(),
            "prediction row width differs from model inputs"
        );
        ensure!(
            row.values.iter().flatten().all(|v| v.len() <= 8192),
            "input cell is too large"
        );
        match &self.predictor {
            Predictor::Math(model) => {
                let cells = row.values.iter().map(|s| s.as_deref()).collect::<Vec<_>>();
                let values = model.predict_cells(&[cells])?;
                ensure!(
                    values.len() == 1 && values[0].is_finite(),
                    "model returned an invalid prediction"
                );
                Ok(json!({"kind":"value","value":values[0]}))
            }
            Predictor::Object(object) => {
                let table = RawTable::new(
                    self.columns.clone(),
                    vec![row
                        .values
                        .into_iter()
                        .map(Option::unwrap_or_default)
                        .collect()],
                )?;
                match object.predict(&table)? {
                    Predictions::Values(values) => {
                        ensure!(
                            values.len() == 1 && values[0].is_finite(),
                            "model returned an invalid prediction"
                        );
                        Ok(json!({"kind":"value","value":values[0]}))
                    }
                    Predictions::Probabilities { classes, values } => {
                        ensure!(
                            values.len() == 1
                                && values[0].len() == classes.len()
                                && values[0].iter().all(|v| v.is_finite()),
                            "model returned invalid class probabilities"
                        );
                        let best = values[0]
                            .iter()
                            .enumerate()
                            .max_by(|a, b| a.1.total_cmp(b.1))
                            .context("empty class probabilities")?
                            .0;
                        Ok(
                            json!({"kind":"classification","label":classes[best],"classes":classes,"probabilities":values[0]}),
                        )
                    }
                }
            }
        }
    }
}

fn respond(result: Result<Value>) -> u32 {
    let (status, value) = match result {
        Ok(value) => (0, json!({"ok":true,"result":value})),
        Err(error) => (1, json!({"ok":false,"error":format!("{error:#}")})),
    };
    OUTPUT.with(|out| {
        *out.borrow_mut() = serde_json::to_vec(&value).expect("finite browser response")
    });
    status
}

#[no_mangle]
pub extern "C" fn browser_abi_version() -> u32 {
    1
}
#[no_mangle]
pub extern "C" fn browser_input(length: usize) -> *mut u8 {
    if length > MAX_MESSAGE {
        return std::ptr::null_mut();
    }
    INPUT.with(|input| {
        let mut input = input.borrow_mut();
        input.clear();
        input.resize(length, 0);
        input.as_mut_ptr()
    })
}
#[no_mangle]
pub extern "C" fn browser_load() -> u32 {
    feat_math::set_parallel_enabled(false);
    MODEL.with(|model| *model.borrow_mut() = None);
    respond(INPUT.with(|input| {
        let loaded = Loaded::load(&input.borrow())?;
        let description = loaded.description.clone();
        MODEL.with(|model| *model.borrow_mut() = Some(loaded));
        Ok(description)
    }))
}
#[no_mangle]
pub extern "C" fn browser_predict() -> u32 {
    respond(INPUT.with(|input| {
        MODEL.with(|model| {
            let model = model.borrow();
            model
                .as_ref()
                .context("model is not loaded")?
                .predict(&input.borrow())
        })
    }))
}
#[no_mangle]
pub extern "C" fn browser_output() -> *const u8 {
    OUTPUT.with(|out| out.borrow().as_ptr())
}
#[no_mangle]
pub extern "C" fn browser_output_len() -> usize {
    OUTPUT.with(|out| out.borrow().len())
}
#[no_mangle]
pub extern "C" fn browser_dispose() {
    MODEL.with(|model| *model.borrow_mut() = None);
    INPUT.with(|input| *input.borrow_mut() = Vec::new());
    OUTPUT.with(|out| *out.borrow_mut() = Vec::new());
}

#[cfg(test)]
mod tests {
    use super::*;
    use jope_runtime::math_pipeline::{InputContract, NumericColumn, PipelineState, TargetScale};
    use model_object::{Manifest, Part, Producer};
    fn fixture(head: &str) -> (ModelObject, MathPipeline) {
        let dsl = format!("(pipe X ({head}))");
        let genome = feat_math::GenomeIr::parse(&dsl).unwrap();
        let x = (0..16)
            .map(|i| vec![i as f64 / 15.0, (i % 3) as f64 / 2.0])
            .collect::<Vec<_>>();
        let y = x
            .iter()
            .map(|r| 0.6 * r[0] + 0.2 * r[1])
            .collect::<Vec<_>>();
        let model = feat_math::model::fit(&genome, &x, &y).unwrap();
        let input = InputContract::new(
            vec![
                NumericColumn {
                    name: "x".into(),
                    min: -5.0,
                    max: 10.0,
                    constant: false,
                },
                NumericColumn {
                    name: "z".into(),
                    min: 2.0,
                    max: 6.0,
                    constant: false,
                },
            ],
            vec![1, 0],
            TargetScale {
                name: "outcome".into(),
                min: 20.0,
                scale: 100.0,
            },
        )
        .unwrap();
        let parts: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            ("input.json".into(), serde_json::to_vec(&input).unwrap()),
            (
                "pipeline.json".into(),
                serde_json::to_vec(&PipelineState { dsl, model }).unwrap(),
            ),
        ]);
        let object = ModelObject::new(
            Manifest {
                schema: model_object::SCHEMA.into(),
                minimum_reader_version: 1,
                format: model_object::MATH_PIPELINE.into(),
                producer: Producer::default(),
                primary: "pipeline.json".into(),
                required_capabilities: vec![],
                files: parts
                    .iter()
                    .map(|(name, bytes)| Part::new(name, "application/json", bytes))
                    .collect(),
                metadata: json!({}),
            },
            parts,
            MAX_MODEL,
        )
        .unwrap();
        let native = object.math_pipeline().unwrap();
        (object, native)
    }
    fn envelope(object: &ModelObject) -> Vec<u8> {
        let parts = object
            .parts()
            .iter()
            .map(|(name, data)| (name.clone(), STANDARD.encode(data)))
            .collect::<BTreeMap<_, _>>();
        serde_json::to_vec(&json!({"manifest":String::from_utf8(object.manifest_bytes().to_vec()).unwrap(),"parts":parts})).unwrap()
    }
    #[test]
    fn saved_ridge_and_gpr_match_native_raw_cell_prediction_without_refitting() {
        feat_math::set_parallel_enabled(false);
        for head in ["ridge", "gpr kernel=\"rbf+linear\" steps=2"] {
            let (object, native) = fixture(head);
            let loaded = Loaded::load(&envelope(&object)).unwrap();
            let mut probes = Vec::new();
            for values in [
                vec![Some("0"), Some("4")],
                vec![Some("-100"), Some("100")],
                vec![None, None],
                vec![Some("NaN"), Some("yes")],
                vec![Some("1e0"), Some("2.5")],
            ] {
                let expected = native.predict_cells(std::slice::from_ref(&values)).unwrap()[0];
                probes.push(json!({"values": values, "expected": expected}));
                let result = loaded
                    .predict(&serde_json::to_vec(&json!({"values":values})).unwrap())
                    .unwrap();
                assert_eq!(
                    expected.to_bits(),
                    result["value"].as_f64().unwrap().to_bits(),
                    "{head}: {values:?}"
                );
            }
            assert!(loaded.predict(br#"{"values":["1"]}"#).is_err());
            if let Ok(directory) = std::env::var("JOPE_BROWSER_FIXTURES") {
                let directory = std::path::Path::new(&directory);
                std::fs::create_dir_all(directory).unwrap();
                let name = if head == "ridge" { "ridge" } else { "gpr" };
                let envelope: Value = serde_json::from_slice(&envelope(&object)).unwrap();
                std::fs::write(directory.join(format!("{name}.json")),serde_json::to_vec_pretty(&json!({"envelope":envelope,"probes":probes,"source":"native MathPipeline::predict_cells; deterministic synthetic fixture"})).unwrap()).unwrap();
            }
        }
    }
    #[test]
    fn native_appliance_binary_keeps_categorical_inputs_and_class_labels() {
        use feat_math::advanced_v3::{ColumnSpec, DataSchema, RowSemantics};
        use jope_runtime::{
            budget::BudgetRecord,
            fitted::{CreatedBy, FitRecord, FittedModel},
            methods::{native::fit_native, registry::rust_methods},
            task::TaskKind as NativeTask,
        };
        feat_math::set_parallel_enabled(false);
        let method = rust_methods(NativeTask::Binary)
            .unwrap()
            .into_iter()
            .find(|m| m.family == "logit_ridge")
            .unwrap();
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x"), ColumnSpec::categorical("region")],
            row_semantics: RowSemantics::Iid,
        };
        let rows = (0..24)
            .map(|i| {
                vec![
                    i.to_string(),
                    if i % 2 == 0 {
                        "north".into()
                    } else {
                        "south".into()
                    },
                ]
            })
            .collect();
        let train = RawTable::new(vec!["x".into(), "region".into()], rows).unwrap();
        let labels = vec!["No".to_owned(), "Yes".to_owned()];
        let y = (0..24).map(|i| f64::from(i >= 12)).collect::<Vec<_>>();
        let bound = fit_native(&method, NativeTask::Binary, &schema, &labels, &train, &y).unwrap();
        let budget = BudgetRecord {
            time_limit_seconds: None,
            consumed_seconds: 0.0,
            threads: 1,
            enforcement: "test",
        };
        let fitted = FittedModel::new(FitRecord {
            method: &method,
            kind: NativeTask::Binary,
            classes: &labels,
            target: "class",
            schema: &schema,
            model: bound,
            rows: 24,
            fitting_input_sha256: "a".repeat(64),
            training_file_sha256: "b".repeat(64),
            selection: None,
            budget: &budget,
            created_by: CreatedBy {
                source_sha40: "a".repeat(40),
                ..Default::default()
            },
        })
        .unwrap();
        let object = ModelObject::from_fitted(&fitted).unwrap();
        let loaded = Loaded::load(&envelope(&object)).unwrap();
        let values = vec![Some("18"), Some("north")];
        let expected = fitted
            .predict(
                &RawTable::new(
                    vec!["x".into(), "region".into()],
                    vec![vec!["18".into(), "north".into()]],
                )
                .unwrap(),
            )
            .unwrap();
        let Predictions::Probabilities {
            classes,
            values: probabilities,
        } = expected
        else {
            panic!("binary model did not return probabilities")
        };
        let actual = loaded
            .predict(&serde_json::to_vec(&json!({"values":values})).unwrap())
            .unwrap();
        assert_eq!(actual["classes"], json!(classes));
        assert_eq!(actual["probabilities"], json!(probabilities[0]));
        if let Ok(directory) = std::env::var("JOPE_BROWSER_FIXTURES") {
            let fixture = json!({"envelope":serde_json::from_slice::<Value>(&envelope(&object)).unwrap(),"probes":[{"values":values,"classes":classes,"probabilities":probabilities[0]}]});
            std::fs::write(
                std::path::Path::new(&directory).join("binary.json"),
                serde_json::to_vec_pretty(&fixture).unwrap(),
            )
            .unwrap();
        }
    }

    #[test]
    fn invalid_parts_are_refused_and_a_failed_load_clears_the_previous_model() {
        let (object, _) = fixture("ridge");
        let bytes = envelope(&object);
        INPUT.with(|input| *input.borrow_mut() = bytes.clone());
        assert_eq!(browser_load(), 0);
        let mut changed: Value = serde_json::from_slice(&bytes).unwrap();
        changed["parts"]["pipeline.json"] = json!(STANDARD.encode(b"{}"));
        let changed = serde_json::to_vec(&changed).unwrap();
        assert!(Loaded::load(&changed).is_err());
        INPUT.with(|input| *input.borrow_mut() = changed);
        assert_eq!(browser_load(), 1);
        INPUT.with(|input| *input.borrow_mut() = br#"{"values":["0","4"]}"#.to_vec());
        assert_eq!(browser_predict(), 1);
    }
}
