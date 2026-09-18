//! jope dataset-embedding assembly (P2): reproduce jope's 1536-d input vector in Rust
//! with parity (~1e-5) to the Python `embedding_from_vs` path.
//!
//! z_in(1536) = concat[ hyperion-512 (signature over the .vs rows, standardized),
//!                      foundation-1024 (PCA affine of the cached 2048-d foundation raw) ]
//! target(512) = mean over the target .vs rows.
//!
//! Deterministic, pure-Rust: reads the Hyperion `.vs` sidecars (CSV), the cached foundation
//! `<hash>.npy` (2048 f32), the `foundation_projection.npz` PCA arrays, and `standardization.json`.
//! Mirrors `jope/src/jope/propose.py:embedding_from_vs`, `data/vs.py:dataset_signature_from_vs`,
//! `encoders/projection.py:block_for_hash`, `data/target_encoding.py:target_vs_embedding`.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

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

/// Standardization + foundation config loaded from `<art_dir>/standardization.json`.
pub struct StdCfg {
    pub mean: Vec<f32>,    // 512
    pub scale: Vec<f32>,   // 512 (zeros guarded to 1.0)
    pub cache_dir: String, // foundation .npy cache dir
    pub proj_npz: String,  // foundation_projection.npz path
}

/// PCA affine for the 2048->1024 foundation block.
pub struct FoundationProj {
    pub components: Vec<f32>, // 1024 x 2048, row-major (components[j*2048 + i])
    pub pca_mean: Vec<f32>,   // 2048
    pub z_mean: Vec<f32>,     // 1024
    pub z_scale: Vec<f32>,    // 1024
}

pub fn load_std_cfg(art_dir: &Path) -> Result<StdCfg> {
    let raw = std::fs::read_to_string(art_dir.join("standardization.json"))
        .with_context(|| format!("read standardization.json in {}", art_dir.display()))?;
    let v: serde_json::Value = serde_json::from_str(&raw)?;
    let to_vec = |key: &str| -> Result<Vec<f32>> {
        v[key]
            .as_array()
            .ok_or_else(|| anyhow!("standardization.{key} not an array"))?
            .iter()
            .map(|x| {
                x.as_f64()
                    .map(|f| f as f32)
                    .ok_or_else(|| anyhow!("bad {key} entry"))
            })
            .collect()
    };
    let mut scale = to_vec("scale")?;
    for s in scale.iter_mut() {
        if s.abs() < 1e-12 {
            *s = 1.0;
        }
    }
    let f = &v["foundation"];
    Ok(StdCfg {
        mean: to_vec("mean")?,
        scale,
        cache_dir: f["cache_dir"].as_str().unwrap_or_default().to_string(),
        proj_npz: f["projection_file"]
            .as_str()
            .unwrap_or_default()
            .to_string(),
    })
}

pub fn load_projection(npz_path: &str) -> Result<FoundationProj> {
    let file = File::open(npz_path).with_context(|| format!("open {npz_path}"))?;
    let mut zip = zip::ZipArchive::new(file).context("open npz as zip")?;
    let mut get = |name: &str| -> Result<Vec<f32>> {
        // .npz members are stored as "<key>.npy"
        let mut f = zip
            .by_name(&format!("{name}.npy"))
            .with_context(|| format!("npz member {name}.npy"))?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        parse_npy_f32(&buf).map(|(v, _)| v)
    };
    Ok(FoundationProj {
        components: get("components")?,
        pca_mean: get("pca_mean")?,
        z_mean: get("z_mean")?,
        z_scale: get("z_scale")?,
    })
}

/// Assemble the 1536-d embedding + 512-d target for one dataset.
/// `dataset_dir` = `<REG_ROOT>/<hash>/` (holds `hyperion3train128dim.vs`, `train_target.vs`).
/// `row_frac` in [0.8, 1.0]: subsample rows (seeded) before pooling the Hyperion signature.
pub fn jope_embedding(
    dataset_dir: &Path,
    cfg: &StdCfg,
    proj: &FoundationProj,
    row_frac: f32,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let hash = dataset_dir
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow!("dataset_dir has no name"))?;

    // --- hyperion signature (512) over the (optionally subsampled) .vs rows ---
    let mut rows = read_vs_matrix(&dataset_dir.join("hyperion3train128dim.vs"), 128)?;
    if row_frac < 1.0 {
        rows = subsample_rows(rows, row_frac, seed);
    }
    let sig = signature_512(&rows); // f32, computed in f64
    let mut hyp = vec![0f32; 512];
    for i in 0..512 {
        hyp[i] = (sig[i] - cfg.mean[i]) / cfg.scale[i]; // standardize (f32, matches Python)
    }

    // --- foundation block (1024): cached raw-2048 -> PCA affine ---
    let raw = load_npy_f32(&Path::new(&cfg.cache_dir).join(format!("{hash}.npy")))?;
    if raw.len() != 2048 {
        bail!(
            "foundation raw for {hash}: expected 2048, got {}",
            raw.len()
        );
    }
    let block = project_foundation(&raw, proj)?; // 1024

    // --- assemble 1536 ---
    let mut emb = Vec::with_capacity(1536);
    emb.extend_from_slice(&hyp);
    emb.extend_from_slice(&block);

    // --- target (512): mean over target .vs rows ---
    let trows = read_vs_matrix_any(&dataset_dir.join("train_target.vs"))?;
    let tgt = mean_over_rows(&trows, 512);

    Ok((emb, tgt))
}

/// In-memory twin of [`jope_embedding`] for the SESSION path: assemble the 1536-d embedding
/// + 512-d target from tensors the pipeline already holds — no corpus `.vs`/`.npy` files.
///
/// Bit-identical to the file-based path by construction (same pooling/projection helpers).
///
/// - `hyperion_rows`: N×128 Hyperion row embeddings (in-memory equivalent of
///   `hyperion3train128dim.vs`; the rows whose mean is `HyperionAugment.dataset_vector`).
/// - `foundation_raw`: the 2048-d pooled foundation vector (equivalent of the per-hash `.npy`).
/// - `target_rows`: M×W target rows (equivalent of `train_target.vs`), mean-pooled to 512.
pub fn jope_embedding_from_parts(
    hyperion_rows: &[Vec<f32>],
    foundation_raw: &[f32],
    target_rows: &[Vec<f32>],
    cfg: &StdCfg,
    proj: &FoundationProj,
) -> Result<(Vec<f32>, Vec<f32>)> {
    if hyperion_rows.is_empty() {
        bail!("jope_embedding_from_parts: no hyperion rows");
    }
    let sig = signature_512(hyperion_rows);
    let mut hyp = vec![0f32; 512];
    for i in 0..512 {
        hyp[i] = (sig[i] - cfg.mean[i]) / cfg.scale[i];
    }
    if foundation_raw.len() != 2048 {
        bail!(
            "jope_embedding_from_parts: foundation_raw expected 2048, got {}",
            foundation_raw.len()
        );
    }
    let block = project_foundation(foundation_raw, proj)?;
    let mut emb = Vec::with_capacity(1536);
    emb.extend_from_slice(&hyp);
    emb.extend_from_slice(&block);
    let tgt = mean_over_rows(target_rows, 512);
    Ok((emb, tgt))
}

// ---- pooling / projection ---------------------------------------------------

fn signature_512(rows: &[Vec<f32>]) -> Vec<f32> {
    let n = rows.len();
    let d = 128;
    let mut mean = vec![0f64; d];
    for r in rows {
        for j in 0..d {
            mean[j] += r[j] as f64;
        }
    }
    for m in mean.iter_mut() {
        *m /= n as f64;
    }
    let mut var = vec![0f64; d];
    for r in rows {
        for j in 0..d {
            let dv = r[j] as f64 - mean[j];
            var[j] += dv * dv;
        }
    }
    for v in var.iter_mut() {
        *v = (*v / n as f64).sqrt(); // population std (ddof=0)
    }
    // per-column numpy-linear quantiles
    let mut cols: Vec<Vec<f64>> = (0..d).map(|_| Vec::with_capacity(n)).collect();
    for r in rows {
        for j in 0..d {
            cols[j].push(r[j] as f64);
        }
    }
    for c in cols.iter_mut() {
        c.sort_by(|a, b| a.partial_cmp(b).unwrap());
    }
    let q = |sorted: &[f64], q: f64| -> f64 {
        let n = sorted.len();
        if n == 1 {
            return sorted[0];
        }
        let pos = q * (n as f64 - 1.0);
        let lo = pos.floor() as usize;
        let frac = pos - lo as f64;
        if lo + 1 < n {
            sorted[lo] + frac * (sorted[lo + 1] - sorted[lo])
        } else {
            sorted[lo]
        }
    };
    let mut out = Vec::with_capacity(512);
    out.extend(mean.iter().take(d).map(|value| *value as f32));
    out.extend(var.iter().take(d).map(|value| *value as f32));
    out.extend(cols.iter().take(d).map(|col| q(col, 0.10) as f32));
    out.extend(cols.iter().take(d).map(|col| q(col, 0.90) as f32));
    out
}

fn project_foundation(raw: &[f32], p: &FoundationProj) -> Result<Vec<f32>> {
    // z[j] = sum_i (raw[i] - pca_mean[i]) * components[j][i]; block = (z - z_mean)/z_scale
    let centered: Vec<f64> = raw
        .iter()
        .zip(&p.pca_mean)
        .map(|(r, m)| *r as f64 - *m as f64)
        .collect();
    let mut block = vec![0f32; 1024];
    for (j, slot) in block.iter_mut().enumerate().take(1024) {
        let base = j * 2048;
        let mut acc = 0f64;
        for (i, centered_value) in centered.iter().enumerate().take(2048) {
            acc += centered_value * p.components[base + i] as f64;
        }
        *slot = ((acc - p.z_mean[j] as f64) / p.z_scale[j] as f64) as f32;
    }
    Ok(block)
}

fn mean_over_rows(rows: &[Vec<f32>], dim: usize) -> Vec<f32> {
    let w = rows.first().map(|r| r.len()).unwrap_or(0);
    let n = rows.len();
    let mut mean = vec![0f64; w];
    for r in rows {
        for j in 0..w {
            mean[j] += r[j] as f64;
        }
    }
    let mut out = vec![0f32; dim];
    for j in 0..w.min(dim) {
        out[j] = (mean[j] / n as f64) as f32;
    }
    out
}

fn subsample_rows(mut rows: Vec<Vec<f32>>, frac: f32, seed: u64) -> Vec<Vec<f32>> {
    let n = rows.len();
    let take = ((frac as f64) * n as f64).floor() as usize;
    let take = take.clamp(1, n);
    // deterministic seeded Fisher-Yates over indices (splitmix64)
    let mut idx: Vec<usize> = (0..n).collect();
    let mut s = seed;
    let mut next = || {
        s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    for i in (1..n).rev() {
        let j = (next() % (i as u64 + 1)) as usize;
        idx.swap(i, j);
    }
    idx.truncate(take);
    idx.sort_unstable(); // keep original relative order among the kept rows
    let mut out = Vec::with_capacity(take);
    for &i in &idx {
        out.push(std::mem::take(&mut rows[i]));
    }
    out
}

// ---- file readers -----------------------------------------------------------

fn read_vs_matrix(path: &Path, dim: usize) -> Result<Vec<Vec<f32>>> {
    let rows = read_vs_matrix_any(path)?;
    if let Some(r) = rows.first() {
        if r.len() != dim {
            bail!(
                "{}: expected {dim} floats/row, got {}",
                path.display(),
                r.len()
            );
        }
    }
    Ok(rows)
}

fn read_vs_matrix_any(path: &Path) -> Result<Vec<Vec<f32>>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut rows = Vec::new();
    let mut width: Option<usize> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let vals: Result<Vec<f32>> = line
            .split(',')
            .map(|p| {
                // parse as f64 then narrow to f32, matching numpy asarray(float32) rounding
                p.trim()
                    .parse::<f64>()
                    .map(|f| f as f32)
                    .map_err(|_| anyhow!("{}: non-float", path.display()))
            })
            .collect();
        let vals = vals?;
        match width {
            None => width = Some(vals.len()),
            Some(w) if w != vals.len() => bail!("{}: ragged row", path.display()),
            _ => {}
        }
        rows.push(vals);
    }
    if rows.is_empty() {
        bail!("{}: no rows", path.display());
    }
    Ok(rows)
}

/// Load a plain (non-npz) numpy `.npy` of float32 into a flat Vec (C-order).
pub fn load_npy_f32(path: &Path) -> Result<Vec<f32>> {
    let mut buf = Vec::new();
    File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_end(&mut buf)?;
    parse_npy_f32(&buf).map(|(v, _)| v)
}

/// Parse a `.npy` byte buffer: returns (flat f32 data, shape). Handles v1/v2, C-order float32.
fn parse_npy_f32(buf: &[u8]) -> Result<(Vec<f32>, Vec<usize>)> {
    if buf.len() < 10 || &buf[0..6] != b"\x93NUMPY" {
        bail!("not a .npy file");
    }
    let major = buf[6];
    let (header_len, data_start) = if major == 1 {
        (u16::from_le_bytes([buf[8], buf[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&buf[data_start..data_start + header_len])?;
    if !(header.contains("<f4") || header.contains("float32")) {
        bail!("npy dtype not float32: {header}");
    }
    let fortran = header.contains("'fortran_order': True");
    // parse shape tuple, e.g. 'shape': (1024, 2048) or (2048,)
    let shape = {
        let s = header
            .split("'shape':")
            .nth(1)
            .ok_or_else(|| anyhow!("no shape"))?;
        let inner = &s[s.find('(').ok_or_else(|| anyhow!("shape ("))? + 1
            ..s.find(')').ok_or_else(|| anyhow!("shape )"))?];
        inner
            .split(',')
            .filter_map(|t| t.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    };
    let data = &buf[data_start + header_len..];
    let n: usize = if shape.is_empty() {
        0
    } else {
        shape.iter().product()
    };
    if data.len() < n * 4 {
        bail!("npy data too short: have {}, need {}", data.len(), n * 4);
    }
    let mut out = Vec::with_capacity(n);
    for k in 0..n {
        let o = k * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    // normalize Fortran-order 2D arrays to C-order (1D order is identical)
    if fortran && shape.len() == 2 {
        let (r, c) = (shape[0], shape[1]);
        let mut t = vec![0f32; r * c];
        for i in 0..r {
            for j in 0..c {
                t[i * c + j] = out[i + j * r];
            }
        }
        out = t;
    }
    Ok((out, shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires the reviewed Starforge model-bundle artifact"]
    fn reviewed_preprocessing_bundle_loads_with_exact_v1_shapes() {
        let root = std::env::var_os("JAIN_TEST_MODEL_BUNDLE_ROOT")
            .map(std::path::PathBuf::from)
            .expect("JAIN_TEST_MODEL_BUNDLE_ROOT must identify the reviewed bundle");
        let encoder = JopeEncoder::from_safetensors(root.join("preprocessing.safetensors"))
            .expect("load reviewed preprocessing.safetensors");
        let preprocessing = encoder.preprocessing();
        assert_eq!(preprocessing.hyperion_mean.len(), 512);
        assert_eq!(preprocessing.hyperion_scale.len(), 512);
        assert_eq!(preprocessing.foundation_pca_mean.len(), 2048);
        assert_eq!(preprocessing.foundation_components.len(), 1024 * 2048);
        assert_eq!(preprocessing.foundation_mean.len(), 1024);
        assert_eq!(preprocessing.foundation_scale.len(), 1024);
    }

    #[test]
    fn versioned_encoder_rejects_missing_target_sidecar() {
        let preprocessing = JopePreprocessingV1 {
            hyperion_mean: vec![0.0; 512],
            hyperion_scale: vec![1.0; 512],
            foundation_pca_mean: vec![0.0; 2048],
            foundation_components: vec![0.0; 1024 * 2048],
            foundation_mean: vec![0.0; 1024],
            foundation_scale: vec![1.0; 1024],
        };
        let encoder = JopeEncoder::new(preprocessing).unwrap();
        let input = JopeDatasetInputV1 {
            dataset_id: "fixture".into(),
            input_digest: "input".into(),
            target_sidecar_digest: "target".into(),
            hyperion_rows: vec![vec![0.0; 128]],
            foundation_raw: vec![0.0; 2048],
            target_rows: Vec::new(),
        };
        let options = JopeEncodeOptions {
            model_bundle_digest: "bundle".into(),
            ..Default::default()
        };
        assert!(encoder.encode(&input, &options).is_err());
    }

    #[test]
    fn versioned_encoder_assembles_blocks_deterministically() {
        let mut components = vec![0.0; 1024 * 2048];
        for index in 0..1024 {
            components[index * 2048 + index] = 1.0;
        }
        let encoder = JopeEncoder::new(JopePreprocessingV1 {
            hyperion_mean: vec![0.0; 512],
            hyperion_scale: vec![1.0; 512],
            foundation_pca_mean: vec![0.0; 2048],
            foundation_components: components,
            foundation_mean: vec![0.0; 1024],
            foundation_scale: vec![1.0; 1024],
        })
        .unwrap();
        let mut first = vec![0.0; 128];
        let mut second = vec![0.0; 128];
        first[0] = 1.0;
        second[0] = 3.0;
        let mut foundation = vec![0.0; 2048];
        foundation[0] = 5.0;
        let input = JopeDatasetInputV1 {
            dataset_id: "fixture".into(),
            input_digest: String::new(),
            target_sidecar_digest: "target".into(),
            hyperion_rows: vec![first, second],
            foundation_raw: foundation,
            target_rows: vec![vec![2.0; 512], vec![4.0; 512]],
        };
        let options = JopeEncodeOptions {
            model_bundle_digest: "bundle".into(),
            ..Default::default()
        };
        let a = encoder.encode(&input, &options).unwrap();
        let b = encoder.encode(&input, &options).unwrap();
        assert_eq!(a.embedding, b.embedding);
        assert_eq!(a.target, vec![3.0; 512]);
        assert_eq!(a.embedding[0], 2.0);
        assert_eq!(a.embedding[512], 5.0);
        assert_eq!(a.embedding.len(), 1536);
        assert_eq!(a.input_digest.len(), 64);
    }

    // Parity vs the Python reference dumped for hash 0014b9dad5725a1e (full rows).
    #[test]
    #[ignore = "requires reviewed JOPE parity artifacts and reference output"]
    fn embedding_matches_python_reference() {
        let art_buf = std::env::var_os("JOPE_PARITY_ART_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new("artifacts/jope").to_path_buf());
        let ds_buf = std::env::var_os("JOPE_PARITY_DATASET_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| Path::new("target/jope-fixture").to_path_buf());
        let art = art_buf.as_path();
        let ds = ds_buf.as_path();
        let ref_path =
            std::env::var("JOPE_PARITY_REF").unwrap_or_else(|_| "target/emb_ref.json".into());
        assert!(
            ds.is_dir(),
            "JOPE parity dataset directory is missing: {}",
            ds.display()
        );
        assert!(
            std::path::Path::new(&ref_path).is_file(),
            "JOPE parity reference is missing: {ref_path}"
        );
        let cfg = load_std_cfg(art).unwrap();
        let proj = load_projection(&cfg.proj_npz).unwrap();
        let (emb, tgt) = jope_embedding(ds, &cfg, &proj, 1.0, 0).unwrap();
        let refj: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(ref_path).unwrap()).unwrap();
        let re: Vec<f32> = refj["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        let rt: Vec<f32> = refj["target"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() as f32)
            .collect();
        assert_eq!(emb.len(), 1536);
        assert_eq!(tgt.len(), 512);
        let maxd = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let de = maxd(&emb, &re);
        let dt = maxd(&tgt, &rt);
        eprintln!("max|Δ| embedding={de:.2e} target={dt:.2e}");
        assert!(de < 1e-4, "embedding parity: max diff {de}");
        assert!(dt < 1e-4, "target parity: max diff {dt}");
    }
}
