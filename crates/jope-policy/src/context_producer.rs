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
    hyperion::{fit_quantile_bounds, parse_settings, HyperionSweepSettings},
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
pub mod classification;
mod input;
mod products;
pub mod worker;
pub use admission::*;
pub use input::*;
pub use products::*;

pub const SCHEMA: &str = "jope.cpu-context/v1";
pub const BACKEND: &str = "Rust/Candle 0.11 CPU FP32; frozen accepted transforms; native-component qualification required";
pub const TRANSFORMS_SHA256: &str =
    "1b779c6ff7a1a812e3001342d663353d3c7ff54eb93392a600f01813f4c66e22";
pub const ASSEMBLY_SHA256: &str =
    "6582df23f5c1ee048d11040a5f2bd1447032932fc23f6c5a3aa9cb4f2d6896cb";
const HYPERION_WEIGHTS: &str = "88b9e8345e44d3ea609a8286215618a762f14e607fcdada6d27447b1f74668be";
const HYPERION_MANIFEST: &str = "ebb745f6a0aa6398b1c3d3c9e26585e0536326b65e2083401391203fa37d21ef";
const HYPERION_SEED: u64 = 20_260_623;
/// `component_bundle` of the accepted encoder identity: sha256 of the pinned component-bundle
/// manifest ([`worker::COMPONENT_BUNDLE_MANIFEST_SHA256`], the file `LoadedModelBundle::load`
/// refuses to differ from) as re-serialized by `starforge::bundle::ModelBundleV1`. This is the
/// value every earlier producer recorded; it was derived from the pinned manifest file on
/// 2026-09-26 and `CpuProducer::load` re-derives it from the loaded bundle and refuses any
/// other value.
pub const COMPONENT_BUNDLE_IDENTITY: &str =
    "985074846ccfe92c818e920e5268ea5269a9e35f1fa2057ddae662266c6c7f9f";
/// Foundation weights of the pinned component bundle (`starlight_core_regressor.safetensors`).
pub const FOUNDATION_WEIGHTS_SHA256: &str =
    "f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128";
/// Target-encoder weights of the pinned component bundle (`target_encoder_fused_v1`).
pub const TARGET_WEIGHTS_SHA256: &str =
    "be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f";
/// The static calculation recipe of the accepted CPU encoder.
pub const ENCODER_RECIPE: &str = "native-hyperion-sweep3-chunks1-batch1024;foundation-pcg20260707-f64-label-normalization;target-pcg17-two-windows;frozen-512+1024+512-clip32/v1";

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
impl EncoderIdentity {
    /// The release decision for the accepted CPU encoder closure: all eight fields are issued by
    /// reviewed constants, never taken from a producer's output. Receivers compare every
    /// artifact, qualification and component qualification with this value.
    pub fn accepted() -> Self {
        Self {
            hyperion_weights: HYPERION_WEIGHTS.into(),
            hyperion_manifest: HYPERION_MANIFEST.into(),
            component_bundle: COMPONENT_BUNDLE_IDENTITY.into(),
            foundation_weights: FOUNDATION_WEIGHTS_SHA256.into(),
            target_weights: TARGET_WEIGHTS_SHA256.into(),
            frozen_transforms: TRANSFORMS_SHA256.into(),
            assembly: ASSEMBLY_SHA256.into(),
            recipe: ENCODER_RECIPE.into(),
        }
    }
    /// Refuse any identity other than [`EncoderIdentity::accepted`].
    pub fn require_accepted(&self) -> Result<()> {
        let accepted = Self::accepted();
        for (field, actual, expected) in [
            (
                "hyperion_weights",
                &self.hyperion_weights,
                &accepted.hyperion_weights,
            ),
            (
                "hyperion_manifest",
                &self.hyperion_manifest,
                &accepted.hyperion_manifest,
            ),
            (
                "component_bundle",
                &self.component_bundle,
                &accepted.component_bundle,
            ),
            (
                "foundation_weights",
                &self.foundation_weights,
                &accepted.foundation_weights,
            ),
            (
                "target_weights",
                &self.target_weights,
                &accepted.target_weights,
            ),
            (
                "frozen_transforms",
                &self.frozen_transforms,
                &accepted.frozen_transforms,
            ),
            ("assembly", &self.assembly, &accepted.assembly),
            ("recipe", &self.recipe, &accepted.recipe),
        ] {
            ensure!(
                actual == expected,
                "encoder lineage changed: {field} is not the accepted release identity"
            );
        }
        // W2-T02: the accepted semantic manifest is the release decision; the constants above
        // are its values (`semantic::tests::the_encoder_release_identity_is_the_manifests`).
        let release = crate::semantic::accepted()?.encoders;
        ensure!(
            self.hyperion_weights == release.hyperion_weights_sha256
                && self.hyperion_manifest == release.hyperion_manifest_sha256
                && self.component_bundle == release.component_bundle_identity
                && self.foundation_weights == release.foundation_weights_sha256
                && self.target_weights == release.target_weights_sha256
                && self.frozen_transforms == release.frozen_transforms_sha256
                && self.assembly == release.assembly_sha256
                && self.recipe == release.recipe,
            "encoder lineage changed: not the accepted semantic release {}",
            crate::semantic::ACCEPTED_SEMANTIC_MANIFEST_SHA256
        );
        Ok(())
    }
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
        Self::load_on_device(files, Device::Cpu)
    }

    fn load_on_device(files: &EncoderFiles, device: Device) -> Result<Self> {
        ensure!(
            files.hyperion_weights.sha256 == HYPERION_WEIGHTS
                && files.hyperion_manifest.sha256 == HYPERION_MANIFEST
                && files.frozen_transforms.sha256 == TRANSFORMS_SHA256,
            "encoder lineage differs from accepted component assets"
        );
        // The model is parsed from exactly the authenticated bytes; the weights path is not
        // reopened, so a replacement after authentication (A->B->A) cannot reach inference.
        let weights = files.hyperion_weights.read(512 << 20)?;
        let settings = verified_hyperion_settings(&files.hyperion_manifest)?;
        let hyperion = if device.is_cpu() {
            V7Model::load_cpu_from_bytes(&weights, &settings)?
        } else {
            #[cfg(feature = "inference-cuda")]
            {
                V7Model::load_cuda_from_bytes(&weights, &settings, device.clone())?
            }
            #[cfg(not(feature = "inference-cuda"))]
            {
                anyhow::bail!("CUDA inference feature required")
            }
        };
        drop(weights);
        ensure!(
            files.component_bundle.is_absolute()
                && files.component_bundle.canonicalize()? == files.component_bundle,
            "component bundle must be physical and absolute"
        );
        let bundle = LoadedModelBundle::load(&files.component_bundle)?;
        ensure!(bundle.target_seed() == 17, "frozen target seed changed");
        // Derived from the loaded (pinned) bundle, then held to the release decision: the
        // producer can only ever report the accepted identity.
        let identity = EncoderIdentity {
            hyperion_weights: HYPERION_WEIGHTS.into(),
            hyperion_manifest: HYPERION_MANIFEST.into(),
            component_bundle: sha(&serde_json::to_vec(&bundle.manifest)?),
            foundation_weights: bundle
                .manifest
                .encoders
                .get("foundation")
                .context("foundation identity absent")?
                .sha256
                .clone(),
            target_weights: bundle
                .manifest
                .encoders
                .get("target")
                .context("target identity absent")?
                .sha256
                .clone(),
            frozen_transforms: TRANSFORMS_SHA256.into(),
            assembly: ASSEMBLY_SHA256.into(),
            recipe: ENCODER_RECIPE.into(),
        };
        identity.require_accepted()?;
        Ok(Self {
            hyperion,
            settings,
            foundation: StarlightFoundationEncoder::load_bundle(
                &bundle,
                device.clone(),
                DType::F32,
            )?,
            target: FusedTargetEncoder::load_on_device(&bundle, device)?,
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
        let context = self.transforms.assemble(&components)?;
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
            // The components' input scope is exactly the prepared rows' JSON digest.
            prepared_rows_sha256: components.scope().into(),
            preprocessing: prepared.preprocessing,
            encoder: self.identity.clone(),
            components: components.to_vectors(),
            vector: context.values().to_vec(),
            final_test_targets_used: false,
        };
        artifact.validate()?;
        Ok(artifact)
    }

    /// Qualification seam only: rows are the frozen native train preprocessing
    /// result (features followed by one regression target), never a corpus root.
    /// This returns typed components, not an admitted context or policy binding.
    pub fn encode_prepared(&self, rows: &[Vec<f64>]) -> Result<Components> {
        self.encode_timed(rows, false).map(|v| v.0)
    }

    fn encode_timed(
        &self,
        rows: &[Vec<f64>],
        profile: bool,
    ) -> Result<(Components, BTreeMap<String, f64>, serde_json::Value)> {
        let mut times = BTreeMap::new();
        // Input scope of every product below; computed before the timed stages.
        let scope = input_scope(rows)?;
        if profile {
            self.hyperion.device().synchronize()?;
        }
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
        let surface = CellSurface::from_quantized_rows(
            quantize_u8_with_bounds(&matrix, &bounds, self.settings.pad_u8),
            n,
            &scope,
        )?;
        let sweep = run_composite_embedding_sweep(
            &self.hyperion,
            surface.grid(),
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
            !sweep.cancelled && sweep.row_emb.len() == n * RowEmbeddings::WIDTH,
            "incomplete Hyperion sweep"
        );
        let row_embeddings = RowEmbeddings::from_hyperion_sweep(sweep.row_emb, &surface)?;
        let hyperion = HyperionSignature::pool(&row_embeddings)?;
        if profile {
            self.hyperion.device().synchronize()?;
        }
        times.insert("hyperion".into(), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let mut foundation = prepare_jope_foundation_rows(rows)?;
        foundation.input.context_y = standardize_context_labels(&foundation.input.context_y);
        let foundation_vector = RawFoundation::from_foundation_encoder(
            self.foundation
                .encode_standardized(&foundation.input)?
                .vector,
            &scope,
        )?;
        if profile {
            self.foundation.device().synchronize()?;
        }
        times.insert("foundation".into(), started.elapsed().as_secs_f64());
        let started = std::time::Instant::now();
        let targets = minmax_targets(&rows.iter().map(|r| r[width]).collect::<Vec<_>>());
        let windows = FrozenTargetWindows::native(n, TargetTask::Regression, 17)?;
        let sampling = serde_json::json!({"hyperion_rows":n,"hyperion_features":width,"hyperion_sweeps":3,"hyperion_scales":[4,7,11,15],"hyperion_batch":1024,"target_chunks_per_sweep":1,
            "foundation_context_rows":foundation.context_indices.len(),"foundation_query_rows":foundation.query_indices.len(),
            "target_neural_windows":windows.neural.iter().map(Vec::len).collect::<Vec<_>>(),"target_classical_windows":windows.classical.iter().map(Vec::len).collect::<Vec<_>>()});
        let target = TargetBlock::from_target_encoder(
            self.target
                .encode_with_windows(&targets, TargetTask::Regression, &windows)?
                .vector,
            &scope,
        )?;
        if profile {
            self.target.device().synchronize()?;
        }
        times.insert("target".into(), started.elapsed().as_secs_f64());
        let result = Components {
            hyperion,
            foundation: foundation_vector,
            target,
        };
        result.validate()?;
        Ok((result, times, sampling))
    }
    /// Named transformation: the frozen assembly of one encode's typed products.
    pub fn assemble(&self, components: &Components) -> Result<FinalContext> {
        self.transforms.assemble(components)
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
    /// Serialized encoder products (claims); their layout is checked when parsed.
    pub components: ComponentVectors,
    /// Serialized final context (a claim); see [`CpuContextArtifact::replay_assembly`].
    pub vector: Vec<f32>,
    pub final_test_targets_used: bool,
}
impl CpuContextArtifact {
    /// Receiver check of the serialized products against the accepted frozen transforms: the
    /// final context must be the frozen assembly of the serialized components. Returns the
    /// typed context under this artifact's input scope (`prepared_rows_sha256`).
    pub fn replay_assembly(&self, transforms: &FrozenTransforms) -> Result<FinalContext> {
        self.validate()?;
        transforms.replay(&self.components, &self.vector, &self.prepared_rows_sha256)
    }

    /// Envelope, identities, pinned encoder release and product layout. Whether the context is
    /// the assembly of the components needs the transforms: [`Self::replay_assembly`].
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
        // All eight encoder fields are held to the release decision, not four.
        self.encoder.require_accepted()?;
        self.components.validate()?;
        vector(&self.vector, FinalContext::WIDTH).context("final context layout")?;
        ensure!(
            self.vector.iter().all(|v| v.abs() <= 32.0),
            "context exceeds frozen clip"
        );
        Ok(())
    }
}

/// Foundation context labels: the native component builder casts to f32, standardizes in f64 with
/// the population deviation plus 1e-8, then casts back.
fn standardize_context_labels(y: &[f32]) -> Vec<f32> {
    let mean = y.iter().map(|&v| v as f64).sum::<f64>() / y.len() as f64;
    let sd =
        (y.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / y.len() as f64).sqrt() + 1e-8;
    y.iter().map(|&v| ((v as f64 - mean) / sd) as f32).collect()
}

/// Target-encoder labels: min-max over the training rows, with a unit span for constant targets.
fn minmax_targets(targets: &[f64]) -> Vec<f32> {
    let low = targets.iter().copied().fold(f64::INFINITY, f64::min);
    let high = targets.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = if high > low { high - low } else { 1.0 };
    targets.iter().map(|v| ((v - low) / span) as f32).collect()
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
    // Parse exactly the authenticated bytes; the manifest path is not reopened. The legacy
    // 64-bit identifier parse_settings also returns is not a SHA-256 digest and is ignored.
    let bytes = manifest.read(1 << 20)?;
    let text = std::str::from_utf8(&bytes).context("Hyperion manifest is not UTF-8")?;
    let (settings, _) = parse_settings(text, HYPERION_SEED, "accepted Hyperion manifest")?;
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
mod tests;
#[cfg(test)]
pub(crate) use tests::test_support;

/// Benchmark-only CUDA encoder. Cannot produce a CPU admission artifact.
#[cfg(feature = "inference-cuda")]
pub struct CudaProducer(CpuProducer);
#[cfg(feature = "inference-cuda")]
impl CudaProducer {
    pub fn load(files: &EncoderFiles, device: Device) -> Result<Self> {
        ensure!(device.is_cuda(), "benchmark requires CUDA");
        let inner = CpuProducer::load_on_device(files, device)?;
        ensure!(
            inner.hyperion.device().is_cuda()
                && inner.foundation.device().is_cuda()
                && inner.target.device().is_cuda(),
            "encoder CPU fallback refused"
        );
        Ok(Self(inner))
    }
    pub fn identity(&self) -> &EncoderIdentity {
        self.0.identity()
    }
    pub fn devices(&self) -> Vec<String> {
        [
            &self.0.hyperion.device(),
            &self.0.foundation.device(),
            &self.0.target.device(),
        ]
        .iter()
        .map(|d| format!("{:?}", d.location()))
        .collect()
    }
    pub fn encode_prepared(&self, rows: &[Vec<f64>]) -> Result<Components> {
        self.0.encode_prepared(rows)
    }
    pub fn encode_request(
        &self,
        rows: &[Vec<f64>],
        profile: bool,
    ) -> Result<(Components, BTreeMap<String, f64>, serde_json::Value)> {
        self.0.encode_timed(rows, profile)
    }
    pub fn assemble(&self, components: &Components) -> Result<FinalContext> {
        self.0.assemble(components)
    }
}
