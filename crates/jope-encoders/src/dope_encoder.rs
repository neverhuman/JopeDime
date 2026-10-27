//! Dope512: the first 512 actual Dope629 coordinates, with a separate product identity.
//!
//! A prepared training table is reused across evidence updates. Neural inference stays on the
//! selected device; host transfer is explicit at serialization boundaries.
#[cfg(feature = "dope-cuda")]
use anyhow::Context;
use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};
use dope_core::mlc_v3::MlcV3Model;
use dope_core::prepare::{build_conditioning, DopeConditioning, PreparedTable};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::Path;

pub use crate::landmark_args::{LandmarkArgs, LandmarkMode};
pub use dope_core::landmark::bundle::{
    BundleScope, LandmarkBundle, LandmarkGroup, LandmarkScope, LandmarkStatus, SubsampleSpec,
};
pub use dope_core::prepare::{
    prepare_table_mixed, prepare_table_with_ids, PrepTask, PreparedTable as PreparedDopeEncoding,
};
pub use dope_core::{
    landmark::{LandmarkFrame, LandmarkOptions},
    target::Head,
};

mod session;
mod workflow;
pub use session::{EncodingSession, EncodingSnapshot, SessionOutcome};
pub use workflow::{
    collect, encode_training, start_training, start_training_with_options, write_events,
    write_snapshot, EncodingRun, SessionOptions, TrainingInput,
};

pub use candle_core::Device as DopeDevice;

pub const DOPE_WEIGHTS_SHA256: &str =
    "d0b6960328d1955386632ff78b512e3add53a9b0256db7bbf067e2e0be5f0c6c";
pub const DOPE512_RECIPE: &str = "dope-mlcv3-high-f32-descriptor256-task128-dist128/v1";
pub const ENCODING_RECEIPT_SCHEMA: &str = "jope.landmark-encoding/v1";

pub(crate) fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// The device tensor has shape [1,512]. It uses no Hyperion normalization or identity.
#[derive(Clone, Debug)]
pub struct Dope512 {
    pub tensor: Tensor,
    pub source_digest: String,
    pub conditioning_digest: String,
}

impl Dope512 {
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        ensure!(
            self.tensor.dims() == [1, 512],
            "Dope512 tensor shape differs"
        );
        Ok(self.tensor.flatten_all()?.to_vec1()?)
    }

    /// Assemble supplied foundation/target products without applying Hyperion's normalizer.
    /// Inputs are already projected/normalized by their own component recipe.
    pub fn assemble(&self, components: &FusionComponents) -> Result<DopeFusion> {
        self.assemble_resident(&components.to_device(self.tensor.device())?)
    }

    pub fn assemble_resident(&self, components: &ResidentFusionComponents) -> Result<DopeFusion> {
        ensure!(
            self.tensor.dims() == [1, 512],
            "Dope512 tensor shape differs"
        );
        ensure!(
            components.source_digest == self.source_digest,
            "fusion source scope differs"
        );
        let dataset1536 = Tensor::cat(&[&self.tensor, &components.foundation1024], 1)?;
        let policy2048 = Tensor::cat(&[&dataset1536, &components.target512], 1)?;
        Ok(DopeFusion {
            recipe: "dope512-foundation1024-target512/v1".into(),
            component_recipe: components.recipe.clone(),
            source_digest: self.source_digest.clone(),
            conditioning_digest: self.conditioning_digest.clone(),
            dataset1536,
            policy2048,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FusionComponents {
    pub source_digest: String,
    pub recipe: String,
    pub foundation1024: Vec<f32>,
    pub target512: Vec<f32>,
}

impl FusionComponents {
    pub fn to_device(&self, device: &Device) -> Result<ResidentFusionComponents> {
        ensure!(
            self.foundation1024.len() == 1024 && self.target512.len() == 512,
            "invalid fusion widths"
        );
        ensure!(
            !self.recipe.is_empty()
                && self
                    .foundation1024
                    .iter()
                    .chain(&self.target512)
                    .all(|v| v.is_finite()),
            "invalid fusion recipe or nonfinite component"
        );
        Ok(ResidentFusionComponents {
            source_digest: self.source_digest.clone(),
            recipe: self.recipe.clone(),
            foundation1024: Tensor::from_slice(&self.foundation1024, (1, 1024), device)?,
            target512: Tensor::from_slice(&self.target512, (1, 512), device)?,
        })
    }
}

#[derive(Clone, Debug)]
pub struct ResidentFusionComponents {
    pub source_digest: String,
    pub recipe: String,
    foundation1024: Tensor,
    target512: Tensor,
}

#[derive(Clone, Debug)]
pub struct DopeFusion {
    pub recipe: String,
    pub component_recipe: String,
    pub source_digest: String,
    pub conditioning_digest: String,
    pub dataset1536: Tensor,
    pub policy2048: Tensor,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopeFusionExport {
    pub recipe: String,
    pub component_recipe: String,
    pub source_digest: String,
    pub conditioning_digest: String,
    pub dataset1536: Vec<f32>,
    pub policy2048: Vec<f32>,
}

impl DopeFusion {
    /// Explicit host serialization boundary; the session product stays on its original device.
    pub fn to_export(&self) -> Result<DopeFusionExport> {
        Ok(DopeFusionExport {
            recipe: self.recipe.clone(),
            component_recipe: self.component_recipe.clone(),
            source_digest: self.source_digest.clone(),
            conditioning_digest: self.conditioning_digest.clone(),
            dataset1536: self.dataset1536.flatten_all()?.to_vec1()?,
            policy2048: self.policy2048.flatten_all()?.to_vec1()?,
        })
    }
}

/// Companion to frozen pipeline receipts. Status and mask always come from the sanitized bundle.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LandmarkEncodingReceipt {
    pub schema: String,
    pub status: LandmarkStatus,
    pub mask: [bool; 23],
    pub subsample_n: Option<usize>,
    pub source_digest: String,
    pub conditioning_digest: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options_digest: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub work_budget: Option<u64>,
    pub weights_sha256: String,
    pub embedding_identity: String,
    pub device: String,
    pub version: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome: Option<SessionOutcome>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
}

#[derive(Clone)]
pub struct DopeEncoder {
    model: std::sync::Arc<MlcV3Model>,
    cpu_pool: Option<std::sync::Arc<rayon::ThreadPool>>,
}

impl DopeEncoder {
    pub fn load(path: &Path, device: &Device) -> Result<Self> {
        let meta = std::fs::symlink_metadata(path)?;
        ensure!(
            meta.is_file() && !meta.file_type().is_symlink(),
            "Dope weights must be a regular file"
        );
        ensure!(meta.len() <= 32 << 20, "Dope weights exceed size bound");
        Self::from_bytes(&std::fs::read(path)?, device)
    }

    pub fn from_bytes(bytes: &[u8], device: &Device) -> Result<Self> {
        ensure!(
            sha256(bytes) == DOPE_WEIGHTS_SHA256,
            "Dope weight checksum mismatch"
        );
        Ok(Self {
            model: std::sync::Arc::new(MlcV3Model::from_safetensors_bytes_on_device(
                bytes, device,
            )?),
            cpu_pool: if device.is_cpu() {
                Some(std::sync::Arc::new(
                    rayon::ThreadPoolBuilder::new().num_threads(1).build()?,
                ))
            } else {
                None
            },
        })
    }

    pub fn device(&self) -> &Device {
        self.model.device()
    }

    /// Select CUDA explicitly; an unavailable build/device is an error, never a CPU fallback.
    pub fn cuda(index: usize) -> Result<Device> {
        #[cfg(feature = "dope-cuda")]
        {
            Device::new_cuda_with_stream(index).context("initialize Dope CUDA device")
        }
        #[cfg(not(feature = "dope-cuda"))]
        {
            anyhow::bail!("cuda:{index} requested; build with the dope-cuda feature")
        }
    }

    pub fn encode(
        &self,
        prepared: &PreparedTable,
        bundle: Option<&LandmarkBundle>,
        scope: &dope_core::landmark::bundle::LandmarkScope,
    ) -> Result<Dope512> {
        let actual_source = prepared.source_rows_digest();
        ensure!(
            scope.source_rows_digest == actual_source,
            "prepared table differs from landmark source scope"
        );
        let zero = LandmarkBundle::zero();
        let supplied = bundle.unwrap_or(&zero);
        let bundle = LandmarkBundle::from_squashed(
            &supplied.values,
            &supplied.mask,
            dope_core::landmark::bundle::BundleScope {
                subsample_n: supplied.subsample_n,
                source_digest: supplied.source_digest.clone(),
                provenance: supplied.provenance.clone(),
            },
        );
        ensure!(
            bundle.is_zero()
                || (bundle.source_digest == scope.digest()
                    && bundle.subsample_n == scope.subsample.map(|s| s.n)),
            "incompatible landmark source/options scope"
        );
        let mode = if bundle.mask.iter().any(|v| *v) {
            dope_core::landmark::bundle::LandmarkMode::Progressive
        } else {
            dope_core::landmark::bundle::LandmarkMode::None
        };
        let mut result =
            self.encode_conditioning(&build_conditioning(prepared, &bundle, None, mode)?)?;
        result.source_digest = scope.digest();
        Ok(result)
    }

    /// Batched GPU seam for consumers that already hold prepared conditioning tensors.
    /// Returns [B,512] on the input device; input/source scope is the caller's responsibility.
    pub fn encode_tensors(&self, descriptor: &Tensor, columns: &Tensor) -> Result<Tensor> {
        let run = || {
            self.model
                .encode_features_629_tensor(descriptor, columns, "high")
        };
        let full = if let Some(pool) = &self.cpu_pool {
            pool.install(run)?
        } else {
            run()?
        };
        Ok(full.narrow(1, 0, 512)?.contiguous()?)
    }

    pub fn encode_conditioning(&self, input: &DopeConditioning) -> Result<Dope512> {
        ensure!(input.vec256.len() == 256, "Dope descriptor width differs");
        let mut columns = vec![0f32; 64 * 17];
        for (i, column) in input.col_landmarks.iter().take(64).enumerate() {
            for (j, value) in column.iter().take(17).enumerate() {
                columns[i * 17 + j] = *value;
            }
        }
        let descriptor = Tensor::from_slice(&input.vec256, (1, 256), self.device())?;
        let columns = Tensor::from_vec(columns, (1, 64, 17), self.device())?;
        let tensor = self.encode_tensors(&descriptor, &columns)?;
        Ok(Dope512 {
            tensor,
            source_digest: input.bundle.source_digest.clone(),
            conditioning_digest: conditioning_digest(input),
        })
    }

    fn receipt(
        &self,
        embedding: &Dope512,
        bundle: &LandmarkBundle,
        version: u64,
    ) -> LandmarkEncodingReceipt {
        let bundle = bundle.clone();
        LandmarkEncodingReceipt {
            schema: ENCODING_RECEIPT_SCHEMA.into(),
            status: bundle.status,
            mask: bundle.mask,
            subsample_n: bundle.subsample_n,
            source_digest: embedding.source_digest.clone(),
            conditioning_digest: embedding.conditioning_digest.clone(),
            options_digest: None,
            work_budget: None,
            weights_sha256: DOPE_WEIGHTS_SHA256.into(),
            embedding_identity: DOPE512_RECIPE.into(),
            device: format!("{:?}", self.device().location()),
            version,
            outcome: None,
            embedding: None,
        }
    }
}

/// Hash every effective model input, including native metadata beyond the 23 public values.
pub fn conditioning_digest(input: &DopeConditioning) -> String {
    input.digest()
}
