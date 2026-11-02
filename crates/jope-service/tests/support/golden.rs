//! Helpers for loading documented golden SafeTensors fixtures in integration tests.
//!
//! The default fixture directory is `research/flywheel_selfplay/golden` in the
//! repository. Set `JOPE_GOLDEN_DIR` to use an externally staged fixture set.

use candle_core_0_11_0::{safetensors::load_buffer, DType, Device, Tensor};
use safetensors::SafeTensors;
use std::{collections::BTreeMap, env, error::Error, fmt, fs, path::Path};

#[rustfmt::skip]
#[path = "../../src/ranker/stage1b.rs"]
mod stage1b;

const GOLDEN_DIR_ENV: &str = "JOPE_GOLDEN_DIR";
const STRICT_ENV: &str = "JOPE_PARITY_STRICT";

/// A required tensor name and its expected dimensions in a fixture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
        }
    }
}

/// A loaded fixture, retaining Candle tensors and an FP32 flattened view of each.
#[derive(Debug)]
pub struct GoldenFixture {
    pub tensors: BTreeMap<String, Tensor>,
    pub values_f32: BTreeMap<String, Vec<f32>>,
}

/// A fixture lookup, file, SafeTensors, or schema-validation failure.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenFixtureError(String);

impl fmt::Display for GoldenFixtureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for GoldenFixtureError {}

fn error(message: impl Into<String>) -> GoldenFixtureError {
    GoldenFixtureError(message.into())
}

/// Load a fixture by relative name, using `JOPE_GOLDEN_DIR` or the repository's
/// `research/flywheel_selfplay/golden` directory. Missing fixtures are skipped
/// with a diagnostic unless `JOPE_PARITY_STRICT=1`.
pub fn load_named(
    name: &str,
    schema: &[TensorSpec],
) -> Result<Option<GoldenFixture>, GoldenFixtureError> {
    let root = env::var_os(GOLDEN_DIR_ENV).map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../research/flywheel_selfplay/golden"),
        std::path::PathBuf::from,
    );
    let strict = env::var(STRICT_ENV).is_ok_and(|value| value == "1");
    load_from_dir(&root, name, schema, strict)
}

fn load_from_dir(
    root: &Path,
    name: &str,
    schema: &[TensorSpec],
    strict: bool,
) -> Result<Option<GoldenFixture>, GoldenFixtureError> {
    let relative = Path::new(name);
    if name.is_empty()
        || relative.is_absolute()
        || relative
            .components()
            .any(|component| !matches!(component, std::path::Component::Normal(_)))
    {
        return Err(error(format!("invalid golden fixture name `{name}`")));
    }

    let relative = if relative
        .extension()
        .is_some_and(|extension| extension == "safetensors")
    {
        relative.to_path_buf()
    } else {
        relative.with_extension("safetensors")
    };
    let path = root.join(relative);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            let message = format!(
                "missing golden fixture `{}`; set {GOLDEN_DIR_ENV} or stage the documented fixture",
                path.display()
            );
            if strict {
                return Err(error(message));
            }
            eprintln!("skipping golden fixture: {message}");
            return Ok(None);
        }
        Err(source) => {
            return Err(error(format!(
                "could not read golden fixture `{}`: {source}",
                path.display()
            )));
        }
    };

    validate_schema(&bytes, schema, &path)?;
    let loaded = load_buffer(&bytes, &Device::Cpu).map_err(|source| {
        error(format!(
            "could not decode golden fixture `{}` into Candle tensors: {source}",
            path.display()
        ))
    })?;
    let mut tensors = BTreeMap::new();
    let mut values_f32 = BTreeMap::new();
    for (name, tensor) in loaded {
        let values = tensor
            .to_dtype(DType::F32)
            .and_then(|tensor| tensor.flatten_all())
            .and_then(|tensor| tensor.to_vec1::<f32>())
            .map_err(|source| {
                error(format!(
                    "could not convert tensor `{name}` in `{}` to Vec<f32>: {source}",
                    path.display()
                ))
            })?;
        tensors.insert(name.clone(), tensor);
        values_f32.insert(name, values);
    }

    Ok(Some(GoldenFixture {
        tensors,
        values_f32,
    }))
}

fn validate_schema(
    bytes: &[u8],
    schema: &[TensorSpec],
    path: &Path,
) -> Result<(), GoldenFixtureError> {
    let safe_tensors = SafeTensors::deserialize(bytes).map_err(|source| {
        error(format!(
            "invalid SafeTensors fixture `{}`: {source}",
            path.display()
        ))
    })?;
    let expected = schema
        .iter()
        .map(|spec| (spec.name.as_str(), spec.shape.as_slice()))
        .collect::<BTreeMap<_, _>>();
    if expected.len() != schema.len() {
        return Err(error(format!(
            "fixture schema for `{}` contains duplicate tensor names",
            path.display()
        )));
    }

    let actual = safe_tensors
        .names()
        .into_iter()
        .map(|name| {
            safe_tensors
                .tensor(name)
                .map(|tensor| (name, tensor.shape().to_vec()))
                .map_err(|source| {
                    error(format!(
                        "invalid tensor `{name}` in `{}`: {source}",
                        path.display()
                    ))
                })
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if actual.len() != expected.len() {
        return Err(error(format!(
            "fixture `{}` has tensor names {:?}; expected {:?}",
            path.display(),
            actual.keys().collect::<Vec<_>>(),
            expected.keys().collect::<Vec<_>>()
        )));
    }
    for (name, expected_shape) in expected {
        let actual_shape = actual.get(name).ok_or_else(|| {
            error(format!(
                "fixture `{}` is missing required tensor `{name}`",
                path.display()
            ))
        })?;
        if actual_shape.as_slice() != expected_shape {
            return Err(error(format!(
                "tensor `{name}` in `{}` has shape {actual_shape:?}; expected {expected_shape:?}",
                path.display()
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
