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
            let (schema, target, task) = if let Some(input) = object.calibrated_input()? {
                // Read only the input contract here. The complete object must keep
                // its stored calibration and class order during prediction.
                let (target, task) = match input.header {
                    Some(header) => (header.target, header.task),
                    None => ("Prediction".to_owned(), "binary".to_owned()),
                };
                (input.schema.columns, target, task)
            } else if format == model_object::APPLIANCE {
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
                        ensure!(!classes.is_empty(), "empty class probabilities");
                        // Match native api::predictions_csv: keep the first
                        // class when probabilities tie, including signed zero.
                        let best = values[0].iter().enumerate().fold(0, |best, (i, p)| {
                            if *p > values[0][best] {
                                i
                            } else {
                                best
                            }
                        });
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
mod tests;
