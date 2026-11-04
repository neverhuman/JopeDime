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

include!("context_producer/producer.rs");
include!("context_producer/artifact.rs");
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use tests::test_support;
include!("context_producer/cuda.rs");
