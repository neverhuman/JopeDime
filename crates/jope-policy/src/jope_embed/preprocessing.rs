use super::projection::signature_512;
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub const JOPE_EMBEDDING_V1: &str = "jope-embedding/v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopePreprocessingV1 {
    pub hyperion_mean: Vec<f32>,
    pub hyperion_scale: Vec<f32>,
    pub foundation_pca_mean: Vec<f32>,
    pub foundation_components: Vec<f32>,
    pub foundation_mean: Vec<f32>,
    pub foundation_scale: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeDatasetInputV1 {
    pub dataset_id: String,
    pub input_digest: String,
    pub target_sidecar_digest: String,
    pub hyperion_rows: Vec<Vec<f32>>,
    pub foundation_raw: Vec<f32>,
    pub target_rows: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeEncodeOptions {
    pub max_hyperion_rows: usize,
    pub seed: u64,
    pub model_bundle_digest: String,
    pub policy_version: String,
    pub vocabulary_version: String,
    pub registry_version: String,
}

impl Default for JopeEncodeOptions {
    fn default() -> Self {
        Self {
            max_hyperion_rows: 0,
            seed: 0,
            model_bundle_digest: String::new(),
            policy_version: "unknown".into(),
            vocabulary_version: "unknown".into(),
            registry_version: "unknown".into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct JopeEmbeddingV1 {
    pub schema: String,
    pub dataset_id: String,
    pub input_digest: String,
    pub target_sidecar_digest: String,
    pub model_bundle_digest: String,
    pub policy_version: String,
    pub vocabulary_version: String,
    pub registry_version: String,
    pub embedding: Vec<f32>,
    pub target: Vec<f32>,
}

pub struct JopeEncoder {
    preprocessing: JopePreprocessingV1,
}

impl JopeEncoder {
    pub fn new(preprocessing: JopePreprocessingV1) -> Result<Self> {
        validate_preprocessing(&preprocessing)?;
        Ok(Self { preprocessing })
    }

    pub fn preprocessing(&self) -> &JopePreprocessingV1 {
        &self.preprocessing
    }
    pub fn from_safetensors(path: impl AsRef<Path>) -> Result<Self> {
        let preprocessing =
            JopePreprocessingV1::load_safetensors(path.as_ref(), &candle_core::Device::Cpu)?;
        Self::new(preprocessing)
    }

    pub fn encode(
        &self,
        input: &JopeDatasetInputV1,
        options: &JopeEncodeOptions,
    ) -> Result<JopeEmbeddingV1> {
        validate_input(input)?;
        if options.model_bundle_digest.trim().is_empty() {
            bail!("model bundle digest is required for JOPE encoding");
        }
        let rows = deterministic_row_cap(
            &input.hyperion_rows,
            options.max_hyperion_rows,
            options.seed,
        );
        let signature = signature_512(&rows);
        let hyperion: Vec<f32> = (0..512)
            .map(|index| {
                (signature[index] - self.preprocessing.hyperion_mean[index])
                    / self.preprocessing.hyperion_scale[index]
            })
            .collect();
        let foundation = project_foundation_v1(&input.foundation_raw, &self.preprocessing)?;
        let mut embedding = Vec::with_capacity(1536);
        embedding.extend_from_slice(&hyperion);
        embedding.extend_from_slice(&foundation);
        let target = mean_over_rows_exact(&input.target_rows, 512)?;
        let input_digest = if input.input_digest.is_empty() {
            digest_json(input)?
        } else {
            input.input_digest.clone()
        };
        Ok(JopeEmbeddingV1 {
            schema: JOPE_EMBEDDING_V1.into(),
            dataset_id: input.dataset_id.clone(),
            input_digest,
            target_sidecar_digest: input.target_sidecar_digest.clone(),
            model_bundle_digest: options.model_bundle_digest.clone(),
            policy_version: options.policy_version.clone(),
            vocabulary_version: options.vocabulary_version.clone(),
            registry_version: options.registry_version.clone(),
            embedding,
            target,
        })
    }
}
impl JopePreprocessingV1 {
    /// Load the bundle-owned preprocessing artifact.  Tensor names are part of the
    /// model-bundle/v1 contract and are intentionally not inferred from filesystem paths.
    pub fn load_safetensors(path: &Path, device: &candle_core::Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("load JOPE preprocessing artifact {}", path.display()))?;
        let vector = |name: &str| -> Result<Vec<f32>> {
            Ok(tensors
                .get(name)
                .with_context(|| format!("missing preprocessing tensor {name}"))?
                .flatten_all()?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?)
        };
        Self {
            hyperion_mean: vector("hyperion_mean")?,
            hyperion_scale: vector("hyperion_scale")?,
            foundation_pca_mean: vector("foundation_pca_mean")?,
            foundation_components: vector("foundation_components")?,
            foundation_mean: vector("foundation_mean")?,
            foundation_scale: vector("foundation_scale")?,
        }
        .into_result()
    }
}
impl JopePreprocessingV1 {
    fn into_result(self) -> Result<Self> {
        validate_preprocessing(&self)?;
        Ok(self)
    }
}

fn validate_preprocessing(p: &JopePreprocessingV1) -> Result<()> {
    if p.hyperion_mean.len() != 512
        || p.hyperion_scale.len() != 512
        || p.foundation_pca_mean.len() != 2048
        || p.foundation_components.len() != 1024 * 2048
        || p.foundation_mean.len() != 1024
        || p.foundation_scale.len() != 1024
    {
        bail!("JOPE preprocessing tensor shapes do not match v1");
    }
    if p.hyperion_scale
        .iter()
        .any(|v| !v.is_finite() || v.abs() < 1e-12)
        || p.foundation_scale
            .iter()
            .any(|v| !v.is_finite() || v.abs() < 1e-12)
        || p.hyperion_mean
            .iter()
            .chain(&p.foundation_pca_mean)
            .chain(&p.foundation_components)
            .chain(&p.foundation_mean)
            .any(|v| !v.is_finite())
    {
        bail!("JOPE preprocessing contains non-finite or zero-scale values");
    }
    Ok(())
}

fn validate_input(input: &JopeDatasetInputV1) -> Result<()> {
    if input.dataset_id.trim().is_empty()
        || input.target_sidecar_digest.trim().is_empty()
        || input.hyperion_rows.is_empty()
        || input.foundation_raw.len() != 2048
        || input.target_rows.is_empty()
    {
        bail!("JOPE input is missing a dataset identity, rows, foundation, or target sidecar");
    }
    if input
        .hyperion_rows
        .iter()
        .any(|row| row.len() != 128 || row.iter().any(|v| !v.is_finite()))
    {
        bail!("Hyperion rows must be finite 128-dimensional vectors");
    }
    if input.foundation_raw.iter().any(|v| !v.is_finite()) {
        bail!("foundation vector contains a non-finite value");
    }
    if input
        .target_rows
        .iter()
        .any(|row| row.len() != 512 || row.iter().any(|v| !v.is_finite()))
    {
        bail!("train_target.vs rows must be finite 512-dimensional vectors");
    }
    Ok(())
}

fn project_foundation_v1(raw: &[f32], p: &JopePreprocessingV1) -> Result<Vec<f32>> {
    if raw.len() != 2048 {
        bail!("foundation vector must be 2048-dimensional");
    }
    let mut out = vec![0.0f32; 1024];
    for (component, slot) in out.iter_mut().enumerate() {
        let mut value = 0.0f64;
        for (index, raw_value) in raw.iter().enumerate() {
            value += (*raw_value as f64 - p.foundation_pca_mean[index] as f64)
                * p.foundation_components[component * 2048 + index] as f64;
        }
        *slot = ((value - p.foundation_mean[component] as f64)
            / p.foundation_scale[component] as f64) as f32;
    }
    Ok(out)
}

fn mean_over_rows_exact(rows: &[Vec<f32>], dimension: usize) -> Result<Vec<f32>> {
    if rows.is_empty() || rows.iter().any(|row| row.len() != dimension) {
        bail!("target sidecar has no rows or the wrong width");
    }
    let mut out = vec![0.0f64; dimension];
    for row in rows {
        for (index, value) in row.iter().enumerate() {
            out[index] += *value as f64;
        }
    }
    Ok(out
        .into_iter()
        .map(|value| (value / rows.len() as f64) as f32)
        .collect())
}

fn deterministic_row_cap(rows: &[Vec<f32>], cap: usize, seed: u64) -> Vec<Vec<f32>> {
    if cap == 0 || rows.len() <= cap {
        return rows.to_vec();
    }
    let mut indices: Vec<usize> = (0..rows.len()).collect();
    let mut state = seed;
    for index in (1..indices.len()).rev() {
        state = splitmix64(state);
        indices.swap(index, (state as usize) % (index + 1));
    }
    indices.truncate(cap);
    indices.sort_unstable();
    indices
        .into_iter()
        .map(|index| rows[index].clone())
        .collect()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn digest_json<T: Serialize>(value: &T) -> Result<String> {
    Ok(Sha256::digest(serde_json::to_vec(value)?)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
