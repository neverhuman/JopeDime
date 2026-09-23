//! Explicit CPU context production, separate from held native GPU entry points.
//!
//! Produced artifacts are unqualified. Accepted-policy use additionally requires
//! numerical replay against independently pinned retained native fixtures and
//! controller-verified bounded execution evidence. No constructor calls the old
//! `contexts::apply` or changes its original-grant hold.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use hyperion::{
    data::Matrix,
    hyperion::{fit_quantile_bounds, load_settings, HyperionSweepSettings},
    hyperion_v7::{
        backend::V7Model, config::SweepPlan, grid::quantize_u8_with_bounds,
        sweep::run_composite_embedding_sweep,
    },
    progress::NullSink,
};
use serde::{Deserialize, Serialize};
use starforge::{
    bundle::LoadedModelBundle,
    foundation::{prepare_jope_foundation_rows, StarlightFoundationEncoder},
    target_encoder::{FrozenTargetWindows, FusedTargetEncoder, TargetTask},
};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
};

mod admission;
mod input;
pub mod worker;
pub use admission::*;
pub use input::*;

pub const SCHEMA: &str = "jope.cpu-context/v1";
pub const BACKEND: &str = "Rust/Candle 0.11 CPU FP32; frozen accepted transforms; native-component qualification required";
pub const TRANSFORMS_SHA256: &str =
    "1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22";
pub const ASSEMBLY_SHA256: &str =
    "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb";
const HYPERION_WEIGHTS: &str = "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be";
const HYPERION_MANIFEST: &str = "ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef";
const HYPERION_SEED: u64 = 20_260_623;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AssetFile {
    pub path: PathBuf,
    pub sha256: String,
}
impl AssetFile {
    pub fn read(&self, maximum: u64) -> Result<Vec<u8>> {
        ensure!(
            self.path.is_absolute() && self.path.canonicalize()? == self.path,
            "asset path must be absolute and physical"
        );
        let meta = std::fs::symlink_metadata(&self.path)?;
        ensure!(
            meta.is_file() && meta.len() <= maximum,
            "asset is not a bounded regular file"
        );
        let bytes = std::fs::read(&self.path)?;
        ensure!(
            bytes.len() as u64 <= maximum && sha(&bytes) == self.sha256,
            "asset bytes changed"
        );
        Ok(bytes)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderFiles {
    pub hyperion_weights: AssetFile,
    pub hyperion_manifest: AssetFile,
    pub frozen_transforms: AssetFile,
    /// Governed existing bundle, validated by LoadedModelBundle; its policy is unused.
    pub component_bundle: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EncoderIdentity {
    pub hyperion_weights: String,
    pub hyperion_manifest: String,
    pub component_bundle: String,
    pub foundation_weights: String,
    pub target_weights: String,
    pub frozen_transforms: String,
    pub assembly: String,
    pub recipe: String,
}

pub struct CpuProducer {
    hyperion: V7Model,
    settings: hyperion::hyperion::HyperionSweepSettings,
    foundation: StarlightFoundationEncoder,
    target: FusedTargetEncoder,
    transforms: FrozenTransforms,
    identity: EncoderIdentity,
}
impl CpuProducer {
    pub fn load(files: &EncoderFiles) -> Result<Self> {
        ensure!(
            files.hyperion_weights.sha256 == HYPERION_WEIGHTS
                && files.hyperion_manifest.sha256 == HYPERION_MANIFEST
                && files.frozen_transforms.sha256 == TRANSFORMS_SHA256,
            "encoder lineage differs from accepted component assets"
        );
        files.hyperion_weights.read(512 << 20)?;
        let settings = verified_hyperion_settings(&files.hyperion_manifest)?;
        let hyperion = V7Model::load_cpu(&files.hyperion_weights.path, &settings)?;
        // Loaders use paths internally: rehash after load as well as before it.
        files.hyperion_weights.read(512 << 20)?;
        ensure!(
            files.component_bundle.is_absolute()
                && files.component_bundle.canonicalize()? == files.component_bundle,
            "component bundle must be physical and absolute"
        );
        let bundle = LoadedModelBundle::load(&files.component_bundle)?;
        ensure!(bundle.target_seed() == 17, "frozen target seed changed");
        let identity=EncoderIdentity {
            hyperion_weights:HYPERION_WEIGHTS.into(), hyperion_manifest:HYPERION_MANIFEST.into(),
            component_bundle:sha(&serde_json::to_vec(&bundle.manifest)?),
            foundation_weights:bundle.manifest.encoders.get("foundation").context("foundation identity absent")?.sha256.clone(),
            target_weights:bundle.manifest.encoders.get("target").context("target identity absent")?.sha256.clone(),
            frozen_transforms:TRANSFORMS_SHA256.into(), assembly:ASSEMBLY_SHA256.into(),
            recipe:"native-hyperion-sweep3-chunks1-batch1024;foundation-pcg20260707-f64-label-normalization;target-pcg17-two-windows;frozen-512+1024+512-clip32/v1".into(),
        };
        Ok(Self {
            hyperion,
            settings,
            foundation: StarlightFoundationEncoder::load_bundle(&bundle, Device::Cpu, DType::F32)?,
            target: FusedTargetEncoder::load(&bundle)?,
            transforms: FrozenTransforms::load(&files.frozen_transforms)?,
            identity,
        })
    }
    pub fn identity(&self) -> &EncoderIdentity {
        &self.identity
    }

    /// The only ordinary producer entry point consumes original training rows.
    /// It neither receives holdout rows/labels nor opens a corpus path.
    pub fn produce(
        &self,
        input_bytes: &[u8],
        source_sha: &str,
        producer_sha: &str,
    ) -> Result<CpuContextArtifact> {
        ensure!(
            input_bytes.len() <= 128 << 20,
            "training payload exceeds 128 MiB"
        );
        validate_hex(source_sha, 40)?;
        validate_hex(producer_sha, 64)?;
        let input: TrainingInput = serde_json::from_slice(input_bytes)?;
        let prepared = input.prepare()?;
        let components = self.encode_prepared(&prepared.rows)?;
        let vector = self.transforms.assemble(&components)?;
        let artifact = CpuContextArtifact {
            schema: SCHEMA.into(),
            backend: BACKEND.into(),
            qualification: "unqualified".into(),
            source_sha256: source_sha.into(),
            producer_sha256: producer_sha.into(),
            input_sha256: sha(input_bytes),
            training: input.identity.clone(),
            selected_training_rows_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.id).collect::<Vec<_>>(),
            )?),
            original_labels_sha256: sha(&serde_json::to_vec(
                &input.rows.iter().map(|r| &r.label).collect::<Vec<_>>(),
            )?),
            preprocessing_sha256: sha(&serde_json::to_vec(&prepared.preprocessing)?),
            prepared_rows_sha256: sha(&serde_json::to_vec(&prepared.rows)?),
            preprocessing: prepared.preprocessing,
            encoder: self.identity.clone(),
            components,
            vector,
            final_test_targets_used: false,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Qualification seam only: rows are the frozen native train preprocessing
    /// result (features followed by one regression target), never a corpus root.
    /// This returns components, not an admitted context or policy binding.
    pub fn encode_prepared(&self, rows: &[Vec<f64>]) -> Result<Components> {
        let started = std::time::Instant::now();
        let width = validate_rows(rows)? - 1;
        let n = rows.len();
        let mut columns = vec![0f32; n * width];
        for (i, row) in rows.iter().enumerate() {
            for j in 0..width {
                columns[j * n + i] = row[j] as f32;
            }
        }
        ensure!(
            columns.iter().all(|v| v.is_finite()),
            "feature conversion exceeds FP32"
        );
        let matrix = Matrix::new(columns, n, width);
        let bounds = fit_quantile_bounds(&matrix, self.settings.q_low, self.settings.q_high);
        let grid = quantize_u8_with_bounds(&matrix, &bounds, self.settings.pad_u8);
        let sweep = run_composite_embedding_sweep(
            &self.hyperion,
            &grid,
            self.settings.pad_u8,
            self.settings.quant_scale,
            self.settings.max_patch,
            &SweepPlan {
                sweeps: 3,
                scales: vec![4, 7, 11, 15],
                batch: 1024,
                seed: self.settings.seed,
                target_chunks_per_sweep: 1,
                fixed_perms: None,
            },
            &NullSink,
        )?;
        ensure!(
            !sweep.cancelled && sweep.row_emb.len() == n * 128,
            "incomplete Hyperion sweep"
        );
        let hyperion = pool_hyperion(&sweep.row_emb)?;
        eprintln!(
            "context stage=hyperion elapsed_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        let mut foundation = prepare_jope_foundation_rows(rows)?;
        // Native component builder casts to f32, normalizes in f64, then casts back.
        let y = &foundation.input.context_y;
        let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
        let sd = (y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / y.len() as f64)
            .sqrt()
            + 1e-8;
        foundation.input.context_y = y.iter().map(|&v| ((v as f64 - mean) / sd) as f32).collect();
        let foundation_vector = self
            .foundation
            .encode_standardized(&foundation.input)?
            .vector;
        eprintln!(
            "context stage=foundation elapsed_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        let targets = rows.iter().map(|r| r[width]).collect::<Vec<_>>();
        let low = targets.iter().copied().fold(f64::INFINITY, f64::min);
        let high = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = if high > low { high - low } else { 1.0 };
        let targets = targets
            .iter()
            .map(|v| ((v - low) / span) as f32)
            .collect::<Vec<_>>();
        let windows = FrozenTargetWindows::native(n, TargetTask::Regression, 17)?;
        let target = self
            .target
            .encode_with_windows(&targets, TargetTask::Regression, &windows)?
            .vector;
        eprintln!(
            "context stage=target elapsed_seconds={:.6}",
            started.elapsed().as_secs_f64()
        );
        let result = Components {
            hyperion,
            foundation: foundation_vector,
            target,
        };
        result.validate()?;
        Ok(result)
    }
    pub fn assemble(&self, components: &Components) -> Result<Vec<f32>> {
        self.transforms.assemble(components)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Components {
    pub hyperion: Vec<f32>,
    pub foundation: Vec<f32>,
    pub target: Vec<f32>,
}
impl Components {
    pub fn validate(&self) -> Result<()> {
        vector(&self.hyperion, 512)?;
        vector(&self.foundation, 2048)?;
        vector(&self.target, 512)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuContextArtifact {
    pub schema: String,
    pub backend: String,
    pub qualification: String,
    pub source_sha256: String,
    pub producer_sha256: String,
    pub input_sha256: String,
    pub training: TrainingIdentity,
    pub selected_training_rows_sha256: String,
    pub original_labels_sha256: String,
    pub preprocessing_sha256: String,
    pub prepared_rows_sha256: String,
    pub preprocessing: TrainingPreprocessing,
    pub encoder: EncoderIdentity,
    pub components: Components,
    pub vector: Vec<f32>,
    pub final_test_targets_used: bool,
}
impl CpuContextArtifact {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && self.backend == BACKEND
                && self.qualification == "unqualified"
                && !self.final_test_targets_used,
            "invalid CPU context envelope"
        );
        validate_hex(&self.source_sha256, 40)?;
        for d in [
            &self.producer_sha256,
            &self.input_sha256,
            &self.selected_training_rows_sha256,
            &self.original_labels_sha256,
            &self.preprocessing_sha256,
            &self.prepared_rows_sha256,
        ] {
            validate_hex(d, 64)?;
        }
        self.training.validate()?;
        ensure!(
            self.selected_training_rows_sha256 == self.training.fitting_rows_sha256,
            "context does not match exact fitting-row assignment"
        );
        ensure!(
            self.preprocessing_sha256 == sha(&serde_json::to_vec(&self.preprocessing)?),
            "preprocessing hash changed"
        );
        ensure!(
            self.encoder.frozen_transforms == TRANSFORMS_SHA256
                && self.encoder.assembly == ASSEMBLY_SHA256
                && self.encoder.hyperion_weights == HYPERION_WEIGHTS
                && self.encoder.hyperion_manifest == HYPERION_MANIFEST,
            "encoder lineage changed"
        );
        self.components.validate()?;
        vector(&self.vector, 2048)?;
        ensure!(
            self.vector.iter().all(|v| v.abs() <= 32.0),
            "context exceeds frozen clip"
        );
        Ok(())
    }
}

struct FrozenTransforms {
    tensors: BTreeMap<String, Tensor>,
}
impl FrozenTransforms {
    fn load(file: &AssetFile) -> Result<Self> {
        ensure!(
            file.sha256 == TRANSFORMS_SHA256,
            "unaccepted frozen transforms"
        );
        let bytes = file.read(16 << 20)?;
        let tensors: BTreeMap<_, _> = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)?
            .into_iter()
            .collect();
        let shapes = [
            ("projection", vec![1024, 2048]),
            ("foundation_mean", vec![2048]),
            ("hyperion_mean", vec![512]),
            ("hyperion_scale", vec![512]),
            ("projected_mean", vec![1024]),
            ("projected_scale", vec![1024]),
            ("target_mean", vec![512]),
            ("target_scale", vec![512]),
        ];
        ensure!(
            tensors.len() == shapes.len(),
            "frozen transform inventory changed"
        );
        for (name, shape) in shapes {
            let t = tensors.get(name).context("missing frozen transform")?;
            ensure!(
                t.dtype() == DType::F32 && t.dims() == shape,
                "frozen transform shape/dtype changed: {name}"
            );
            let values = t.flatten_all()?.to_vec1::<f32>()?;
            ensure!(
                values
                    .iter()
                    .all(|v| v.is_finite() && (!name.ends_with("scale") || *v > 0.0)),
                "invalid frozen transform: {name}"
            );
        }
        Ok(Self { tensors })
    }
    fn assemble(&self, c: &Components) -> Result<Vec<f32>> {
        c.validate()?;
        let t = |v: &[f32]| Tensor::from_vec(v.to_vec(), (1, v.len()), &Device::Cpu);
        let z = t(&c.foundation)?
            .broadcast_sub(&self.tensors["foundation_mean"])?
            .matmul(&self.tensors["projection"].t()?)?;
        let h = t(&c.hyperion)?
            .broadcast_sub(&self.tensors["hyperion_mean"])?
            .broadcast_div(&self.tensors["hyperion_scale"])?;
        let z = z
            .broadcast_sub(&self.tensors["projected_mean"])?
            .broadcast_div(&self.tensors["projected_scale"])?;
        let target = t(&c.target)?
            .broadcast_sub(&self.tensors["target_mean"])?
            .broadcast_div(&self.tensors["target_scale"])?;
        let result = Tensor::cat(&[&h, &z, &target], 1)?
            .clamp(-32f32, 32f32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        vector(&result, 2048)?;
        Ok(result)
    }
}
fn pool_hyperion(flat: &[f32]) -> Result<Vec<f32>> {
    ensure!(
        !flat.is_empty() && flat.len().is_multiple_of(128) && flat.iter().all(|v| v.is_finite()),
        "invalid Hyperion row embeddings"
    );
    let n = flat.len() / 128;
    let mut out = vec![0.0; 512];
    for j in 0..128 {
        let mut values = (0..n).map(|i| flat[i * 128 + j] as f64).collect::<Vec<_>>();
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / n as f64;
        let std = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
        let q = |p: f64| {
            let x = p * (n - 1) as f64;
            values[x.floor() as usize] * (1.0 - x.fract()) + values[x.ceil() as usize] * x.fract()
        };
        out[j] = mean as f32;
        out[128 + j] = std as f32;
        out[256 + j] = q(0.1) as f32;
        out[384 + j] = q(0.9) as f32;
    }
    Ok(out)
}
pub(super) fn sha(bytes: &[u8]) -> String {
    crate::sha256_hex(bytes)
}
pub(super) fn validate_hex(s: &str, length: usize) -> Result<()> {
    ensure!(
        s.len() == length
            && s.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "invalid lowercase hex identity"
    );
    Ok(())
}
fn verified_hyperion_settings(manifest: &AssetFile) -> Result<HyperionSweepSettings> {
    ensure!(
        manifest.sha256 == HYPERION_MANIFEST,
        "Hyperion manifest pin changed"
    );
    manifest.read(1 << 20)?;
    // load_settings returns a legacy 64-bit identifier, not a SHA-256 digest.
    // Authenticate the original file bytes before and after its path-based read.
    let (settings, _) = load_settings(&manifest.path, HYPERION_SEED)?;
    manifest.read(1 << 20)?;
    ensure!(settings.emb_dim == 128, "Hyperion embedding width changed");
    Ok(settings)
}

pub(super) fn vector(v: &[f32], length: usize) -> Result<()> {
    ensure!(
        v.len() == length && v.iter().all(|v| v.is_finite()),
        "invalid finite vector of length {length}"
    );
    Ok(())
}
pub(super) fn validate_rows(rows: &[Vec<f64>]) -> Result<usize> {
    let width = rows.first().context("empty prepared rows")?.len();
    ensure!(
        rows.len() >= 8
            && width >= 2
            && rows
                .iter()
                .all(|r| r.len() == width && r.iter().all(|v| v.is_finite())),
        "prepared rows must contain at least8 finite training rows, features and target"
    );
    Ok(width)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepted_manifest_uses_sha256_bytes_not_legacy_identifier() {
        let mut manifest = AssetFile {
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/accepted-hyperion-manifest.json"),
            sha256: HYPERION_MANIFEST.into(),
        };
        let (_, legacy_id) = load_settings(&manifest.path, HYPERION_SEED).unwrap();
        assert_eq!(legacy_id.len(), 16);
        assert_ne!(legacy_id, manifest.sha256);
        let settings = verified_hyperion_settings(&manifest).unwrap();
        assert_eq!(settings.emb_dim, 128);
        assert_eq!(settings.seed, HYPERION_SEED);
        manifest.sha256 = "0".repeat(64);
        assert!(verified_hyperion_settings(&manifest).is_err());
    }
    #[test]
    fn assembly_uses_all_three_frozen_normalizers_including_target() {
        let mut tensors = BTreeMap::new();
        for (name, n, value) in [
            ("foundation_mean", 2048, 0.0f32),
            ("hyperion_mean", 512, 1.0),
            ("hyperion_scale", 512, 2.0),
            ("projected_mean", 1024, 3.0),
            ("projected_scale", 1024, 4.0),
            ("target_mean", 512, 5.0),
            ("target_scale", 512, 6.0),
        ] {
            tensors.insert(
                name.into(),
                Tensor::from_vec(vec![value; n], n, &Device::Cpu).unwrap(),
            );
        }
        tensors.insert(
            "projection".into(),
            Tensor::zeros((1024, 2048), DType::F32, &Device::Cpu).unwrap(),
        );
        let result = FrozenTransforms { tensors }
            .assemble(&Components {
                hyperion: vec![3.0; 512],
                foundation: vec![1.0; 2048],
                target: vec![17.0; 512],
            })
            .unwrap();
        assert_eq!(&result[..512], &vec![1.0; 512]);
        assert_eq!(&result[512..1536], &vec![-0.75; 1024]);
        assert_eq!(&result[1536..], &vec![2.0; 512]);
    }
    #[test]
    fn hyperion_pool_keeps_512_component_contract() {
        let mut flat = vec![1.0; 128];
        flat.extend(vec![3.0; 128]);
        let pooled = pool_hyperion(&flat).unwrap();
        assert_eq!(&pooled[..128], &vec![2.0; 128]);
        assert_eq!(&pooled[128..256], &vec![1.0; 128]);
        assert_eq!(pooled[256], 1.2);
        assert_eq!(pooled[384], 2.8);
    }
}
