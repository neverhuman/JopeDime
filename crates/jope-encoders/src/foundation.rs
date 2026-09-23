//! Public Starlight foundation encoder used by the Rust JOPE path.

use crate::bundle::LoadedModelBundle;
use crate::model::StarforgeModel;
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::target_encoder::NumpyPcg64;

mod io;
mod support;

pub use io::{read_dataset_csv, read_jope_train_csv};
use support::*;

pub const JOPE_FOUNDATION_SEED: u64 = 20_260_707;
pub const JOPE_FOUNDATION_CONTEXT_ROWS: usize = 768;
pub const JOPE_FOUNDATION_QUERY_ROWS: usize = 256;
pub const JOPE_FOUNDATION_HIDDEN_DIM: usize = 512;
pub const JOPE_FOUNDATION_VECTOR_DIM: usize = 2048;
pub const JOPE_EXTERNAL_MAX_ROWS: usize = 1000;
pub const JOPE_EXTERNAL_SUBSAMPLE_SEED: u64 = 20_260_710;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FoundationDatasetInputV1 {
    pub context_x: Vec<Vec<f32>>,
    pub context_y: Vec<f32>,
    pub query_x: Vec<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct FoundationEncodeOptions {
    pub max_context_rows: usize,
    pub max_query_rows: usize,
    pub seed: u64,
}

/// Typed component-level seam for Python-reference versus Rust comparisons.
///
/// Prime and Lime can compare the per-query Starlight hidden rows before
/// checking the four pooled blocks. This is not an end-to-end engine result.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct FoundationEncodingV1 {
    pub context_rows: usize,
    pub query_rows: usize,
    pub hidden_dim: usize,
    pub query_hidden: Vec<f32>,
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
    pub p10: Vec<f32>,
    pub p90: Vec<f32>,
    pub vector: Vec<f32>,
}

/// Train-fitted feature-and-label scaler from Prime's external-data path.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeExternalMinMaxScalerV1 {
    pub low: Vec<f64>,
    pub span: Vec<f64>,
}

/// Persistable result of Prime's preprocessing before foundation/target encoding.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeExternalPreprocessedV1 {
    pub schema: String,
    pub source_rows: usize,
    pub source_columns: usize,
    pub max_rows: usize,
    pub subsample_seed: u64,
    pub selected_indices: Vec<usize>,
    pub csv_float_format: String,
    pub scaler: JopeExternalMinMaxScalerV1,
    pub encoding_rows: Vec<Vec<f64>>,
}

/// Persistable foundation-specific scaling and deterministic context/query split.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeFoundationPreparedV1 {
    pub schema: String,
    pub feature_low: Vec<f64>,
    pub feature_span: Vec<f64>,
    pub context_indices: Vec<usize>,
    pub query_indices: Vec<usize>,
    pub input: FoundationDatasetInputV1,
}

impl JopeExternalMinMaxScalerV1 {
    /// Apply the train-fitted scaler and reproduce Python `format(v, ".9g")`
    /// followed by the CSV reader's f64 parse. Scientific notation is used only
    /// as an equivalent carrier for the same nine significant decimal digits.
    pub fn transform_for_encoding(&self, rows: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let width = rectangular_width(rows, "external encoding rows")?;
        if width != self.low.len()
            || width != self.span.len()
            || self.low.iter().any(|value| !value.is_finite())
            || self
                .span
                .iter()
                .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("external min/max scaler has invalid dimensions or values");
        }
        rows.iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(column, value)| {
                        let cleaned = if value.is_finite() { *value } else { 0.0 };
                        python_9g_roundtrip((cleaned - self.low[column]) / self.span[column])
                    })
                    .collect::<Result<Vec<_>>>()
            })
            .collect()
    }
}

impl JopeExternalPreprocessedV1 {
    pub fn target_values(&self) -> Result<Vec<f32>> {
        if self.source_columns < 2
            || self
                .encoding_rows
                .iter()
                .any(|row| row.len() != self.source_columns)
        {
            bail!("external encoding rows do not match their persisted width");
        }
        Ok(self
            .encoding_rows
            .iter()
            .map(|row| row[self.source_columns - 1] as f32)
            .collect())
    }
}

/// Reproduce Prime's external train preparation before either shared encoder:
/// clean non-finites to zero, fit min/max across features and label, scale,
/// take an order-preserving NumPy-PCG64 sample of at most 1,000 rows, and apply
/// the exact nine-significant-digit CSV numeric roundtrip.
pub fn preprocess_jope_external_train(rows: &[Vec<f64>]) -> Result<JopeExternalPreprocessedV1> {
    let width = rectangular_width(rows, "external train rows")?;
    let cleaned = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| if value.is_finite() { *value } else { 0.0 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut low = vec![f64::INFINITY; width];
    let mut high = vec![f64::NEG_INFINITY; width];
    for row in &cleaned {
        for (column, value) in row.iter().enumerate() {
            low[column] = low[column].min(*value);
            high[column] = high[column].max(*value);
        }
    }
    let span = low
        .iter()
        .zip(&high)
        .map(|(low, high)| {
            let delta = high - low;
            if delta > 1e-12 {
                delta
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let scaler = JopeExternalMinMaxScalerV1 { low, span };
    let scaled = scaler.transform_for_encoding(&cleaned)?;
    let selected_indices = jope_external_selected_indices(rows.len())?;
    let encoding_rows = selected_indices
        .iter()
        .map(|&index| scaled[index].clone())
        .collect();
    Ok(JopeExternalPreprocessedV1 {
        schema: "jain.jope-external-preprocessing/v1".to_owned(),
        source_rows: rows.len(),
        source_columns: width,
        max_rows: JOPE_EXTERNAL_MAX_ROWS,
        subsample_seed: JOPE_EXTERNAL_SUBSAMPLE_SEED,
        selected_indices,
        csv_float_format: ".9g".to_owned(),
        scaler,
        encoding_rows,
    })
}

/// Exact order-preserving row selection used by NumPy 1.26
/// `default_rng(20260710).choice(n, 1000, replace=False)`.
pub fn jope_external_selected_indices(row_count: usize) -> Result<Vec<usize>> {
    if row_count == 0 {
        bail!("cannot select external rows from an empty input");
    }
    let mut selected = if row_count > JOPE_EXTERNAL_MAX_ROWS {
        NumpyPcg64::new(JOPE_EXTERNAL_SUBSAMPLE_SEED).choice(
            row_count,
            JOPE_EXTERNAL_MAX_ROWS,
            false,
        )?
    } else {
        (0..row_count).collect()
    };
    selected.sort_unstable();
    Ok(selected)
}

/// Apply foundation's second, feature-only min/max fit and NumPy-PCG64 split
/// directly to the external encoding rows without producing a sidecar.
pub fn prepare_jope_foundation_rows(rows: &[Vec<f64>]) -> Result<JopeFoundationPreparedV1> {
    let width = rectangular_width(rows, "foundation train rows")?;
    if rows.len() < 8 {
        bail!("foundation train rows require at least eight rows");
    }
    let feature_count = width - 1;
    let cleaned = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|value| if value.is_finite() { *value } else { 0.0 })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut feature_low = vec![f64::INFINITY; feature_count];
    let mut feature_high = vec![f64::NEG_INFINITY; feature_count];
    for row in &cleaned {
        for (column, value) in row[..feature_count].iter().enumerate() {
            feature_low[column] = feature_low[column].min(*value);
            feature_high[column] = feature_high[column].max(*value);
        }
    }
    let feature_span = feature_low
        .iter()
        .zip(&feature_high)
        .map(|(low, high)| {
            let delta = high - low;
            if delta > 1e-12 {
                delta
            } else {
                1.0
            }
        })
        .collect::<Vec<_>>();
    let x = cleaned
        .iter()
        .map(|row| {
            row[..feature_count]
                .iter()
                .enumerate()
                .map(|(column, value)| {
                    ((value - feature_low[column]) / feature_span[column]) as f32
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let y = cleaned
        .iter()
        .map(|row| row[feature_count] as f32)
        .collect::<Vec<_>>();
    let indices = NumpyPcg64::new(JOPE_FOUNDATION_SEED).permutation(rows.len());
    let context_rows = JOPE_FOUNDATION_CONTEXT_ROWS.min(((3 * rows.len()) / 4).max(8));
    let query_rows = JOPE_FOUNDATION_QUERY_ROWS.min(rows.len() - context_rows);
    let context_indices = indices[..context_rows].to_vec();
    let query_indices = if query_rows < 4 {
        indices[rows.len().saturating_sub(8)..].to_vec()
    } else {
        indices[context_rows..context_rows + query_rows].to_vec()
    };
    Ok(JopeFoundationPreparedV1 {
        schema: "jain.jope-foundation-preprocessing/v1".to_owned(),
        feature_low,
        feature_span,
        context_indices: context_indices.clone(),
        query_indices: query_indices.clone(),
        input: FoundationDatasetInputV1 {
            context_x: context_indices
                .iter()
                .map(|&index| x[index].clone())
                .collect(),
            context_y: context_indices.iter().map(|&index| y[index]).collect(),
            query_x: query_indices
                .iter()
                .map(|&index| x[index].clone())
                .collect(),
        },
    })
}

impl Default for FoundationEncodeOptions {
    fn default() -> Self {
        Self {
            max_context_rows: 1024,
            max_query_rows: 1024,
            seed: 0,
        }
    }
}

pub struct StarlightFoundationEncoder {
    model: StarforgeModel,
}

impl StarlightFoundationEncoder {
    pub fn load(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            model: StarforgeModel::load_dtype(path, device, dtype)?,
        })
    }

    /// Load the exact descriptor-relative bytes verified by the governed bundle.
    pub fn load_bundle(bundle: &LoadedModelBundle, device: Device, dtype: DType) -> Result<Self> {
        let identity = bundle
            .manifest
            .encoders
            .get("foundation")
            .context("model bundle has no foundation encoder")?;
        let bytes = bundle
            .read_verified(&identity.artifact)
            .context("read and re-verify foundation encoder")?;
        Ok(Self {
            model: StarforgeModel::load_buffer_dtype(&bytes, device, dtype)?,
        })
    }

    pub fn encode(
        &self,
        input: &FoundationDatasetInputV1,
        options: FoundationEncodeOptions,
    ) -> Result<Vec<f32>> {
        Ok(self.encode_with_intermediates(input, options)?.vector)
    }

    /// Run raw context/query rows through Starlight and retain the query-hidden
    /// tensor plus every pooled block for cross-runtime parity proofs.
    pub fn encode_with_intermediates(
        &self,
        input: &FoundationDatasetInputV1,
        options: FoundationEncodeOptions,
    ) -> Result<FoundationEncodingV1> {
        self.encode_impl(input, options, false)
    }

    /// Native component qualification seam: consume all supplied context/query rows
    /// with context labels already normalized by the frozen FP64 recipe.
    pub fn encode_standardized(
        &self,
        input: &FoundationDatasetInputV1,
    ) -> Result<FoundationEncodingV1> {
        self.encode_impl(
            input,
            FoundationEncodeOptions {
                max_context_rows: 0,
                max_query_rows: 0,
                seed: 0,
            },
            true,
        )
    }

    fn encode_impl(
        &self,
        input: &FoundationDatasetInputV1,
        options: FoundationEncodeOptions,
        standardized: bool,
    ) -> Result<FoundationEncodingV1> {
        validate_input(input)?;
        let context_indices = take_indices(
            input.context_x.len(),
            options.max_context_rows,
            options.seed,
        );
        let context: Vec<Vec<f32>> = context_indices
            .iter()
            .map(|&index| input.context_x[index].clone())
            .collect();
        let context_y: Vec<f32> = context_indices
            .iter()
            .map(|&index| input.context_y[index])
            .collect();
        let query_indices = take_indices(
            input.query_x.len(),
            options.max_query_rows,
            options.seed ^ 0x9e37_79b9_7f4a_7c15,
        );
        let query: Vec<Vec<f32>> = query_indices
            .iter()
            .map(|&index| input.query_x[index].clone())
            .collect();
        let p = context.first().map(|row| row.len()).unwrap_or(0);
        let dump = if standardized {
            self.model.regression_fusion_features_standardized(
                &flatten(&context),
                &context_y,
                &flatten(&query),
                p,
            )?
        } else {
            self.model.regression_fusion_features_single(
                &flatten(&context),
                &context_y,
                &flatten(&query),
                p,
            )?
        };
        if dump.n_rows != query.len()
            || dump.hidden_dim != JOPE_FOUNDATION_HIDDEN_DIM
            || dump.hidden.len() != dump.n_rows * dump.hidden_dim
            || dump.hidden.iter().any(|value| !value.is_finite())
        {
            bail!(
                "foundation hidden tensor has invalid shape or non-finite values: rows={}, hidden_dim={}, values={}",
                dump.n_rows,
                dump.hidden_dim,
                dump.hidden.len()
            );
        }
        let rows: Vec<Vec<f32>> = dump
            .hidden
            .chunks_exact(dump.hidden_dim)
            .map(ToOwned::to_owned)
            .collect();
        let vector = pool_mean_std_p10_p90(&rows, standardized);
        if vector.len() != JOPE_FOUNDATION_VECTOR_DIM
            || vector.iter().any(|value| !value.is_finite())
        {
            bail!("foundation pooling produced an invalid vector");
        }
        let mean = vector[..JOPE_FOUNDATION_HIDDEN_DIM].to_vec();
        let std = vector[JOPE_FOUNDATION_HIDDEN_DIM..2 * JOPE_FOUNDATION_HIDDEN_DIM].to_vec();
        let p10 = vector[2 * JOPE_FOUNDATION_HIDDEN_DIM..3 * JOPE_FOUNDATION_HIDDEN_DIM].to_vec();
        let p90 = vector[3 * JOPE_FOUNDATION_HIDDEN_DIM..].to_vec();
        Ok(FoundationEncodingV1 {
            context_rows: context.len(),
            query_rows: dump.n_rows,
            hidden_dim: dump.hidden_dim,
            query_hidden: dump.hidden,
            mean,
            std,
            p10,
            p90,
            vector,
        })
    }
}
