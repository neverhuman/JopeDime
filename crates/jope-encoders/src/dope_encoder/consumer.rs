//! Two optional Rust consumers. Quality is measured independently of these execution surfaces.
use super::{sha256, Dope512, DopeCheckpointIdentity, DopeFusion, EncodingSnapshot};
use crate::jope_policy::JopePolicyModel;
use anyhow::{ensure, Result};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum DopeConsumerMode {
    EncodingOnly,
    JopeDl,
}

pub struct DopeConsumer {
    model: Option<Arc<JopePolicyModel>>,
    cpu_pool: Option<rayon::ThreadPool>,
}

impl DopeConsumer {
    /// The encoder output directly, with no further learned transformation.
    pub fn encoding_only() -> Self {
        Self {
            model: None,
            cpu_pool: None,
        }
    }

    /// Feed that same Dope encoding plus supplied foundation/target components into JOPE's
    /// verified learned context encoder. Load the model on the encoder's owned device/stream.
    pub fn jope_dl(model: Arc<JopePolicyModel>) -> Result<Self> {
        let cpu_pool = if model.device().is_cpu() {
            Some(rayon::ThreadPoolBuilder::new().num_threads(1).build()?)
        } else {
            None
        };
        Ok(Self {
            model: Some(model),
            cpu_pool,
        })
    }

    pub fn mode(&self) -> DopeConsumerMode {
        if self.model.is_some() {
            DopeConsumerMode::JopeDl
        } else {
            DopeConsumerMode::EncodingOnly
        }
    }

    /// Apply to each ordered native-session snapshot; weights/mode remain fixed in this consumer.
    pub fn consume_snapshot(&self, snapshot: &EncodingSnapshot) -> Result<DopeConsumedEncoding> {
        self.consume(&snapshot.embedding, snapshot.fusion.as_ref())
    }

    pub fn consume(
        &self,
        embedding: &Dope512,
        fusion: Option<&DopeFusion>,
    ) -> Result<DopeConsumedEncoding> {
        embedding.checkpoint_identity.validate()?;
        ensure!(
            embedding.tensor.dims() == [1, 512],
            "consumer requires Dope512"
        );
        let (tensor, model_sha, components_digest) = if let Some(model) = &self.model {
            ensure!(
                embedding.tensor.device().same_device(model.device()),
                "Dope embedding and JOPE DL model must share a device/stream"
            );
            let fusion = fusion.ok_or_else(|| {
                anyhow::anyhow!("JOPE DL consumer requires scoped foundation and target components")
            })?;
            ensure!(
                fusion.checkpoint_identity == embedding.checkpoint_identity
                    && fusion.source_digest == embedding.source_digest
                    && fusion.conditioning_digest == embedding.conditioning_digest,
                "consumer fusion differs from Dope checkpoint/source/conditioning"
            );
            ensure!(
                fusion.dataset1536.dims() == [1, 1536] && fusion.policy2048.dims() == [1, 2048],
                "invalid consumer fusion shape"
            );
            let run = || -> Result<Tensor> {
                let target = fusion.policy2048.narrow(1, 1536, 512)?;
                let context = model.context_tensors(&fusion.dataset1536, &target)?;
                Ok(model.context_tensor(&context).clone())
            };
            let tensor = if let Some(pool) = &self.cpu_pool {
                pool.install(run)?
            } else {
                run()?
            };
            (
                tensor,
                Some(model.weights_sha256().to_owned()),
                Some(fusion.components_digest.clone()),
            )
        } else {
            (embedding.tensor.clone(), None, None)
        };
        let recipe = match self.mode() {
            DopeConsumerMode::EncodingOnly => "dope512-direct-consumer/v1",
            DopeConsumerMode::JopeDl
                if self.model.as_ref().is_some_and(|m| m.device().is_cuda()) =>
            {
                "dope512-foundation1024-target512-jope-dl512-cuda-f32/v1"
            }
            DopeConsumerMode::JopeDl => "dope512-foundation1024-target512-jope-dl512/v1",
        };
        let conditioning_digest = sha256(&serde_json::to_vec(&(
            recipe,
            &embedding.checkpoint_identity,
            &embedding.source_digest,
            &embedding.conditioning_digest,
            &model_sha,
            &components_digest,
        ))?);
        Ok(DopeConsumedEncoding {
            tensor,
            receipt: DopeConsumerReceipt {
                mode: self.mode(),
                recipe: recipe.into(),
                checkpoint_identity: embedding.checkpoint_identity.clone(),
                jope_weights_sha256: model_sha,
                components_digest,
                source_digest: embedding.source_digest.clone(),
                conditioning_digest,
            },
        })
    }
}

/// Each consumer returns 512 values. Cache keys include BOTH checkpoint identities, component
/// values and effective conditioner inputs; this is not a frozen Hyperion product identity.
#[derive(Clone, Debug)]
pub struct DopeConsumedEncoding {
    tensor: Tensor,
    receipt: DopeConsumerReceipt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopeConsumerReceipt {
    pub mode: DopeConsumerMode,
    pub recipe: String,
    pub checkpoint_identity: DopeCheckpointIdentity,
    pub jope_weights_sha256: Option<String>,
    pub components_digest: Option<String>,
    pub source_digest: String,
    pub conditioning_digest: String,
}

impl DopeConsumedEncoding {
    pub fn tensor_copy(&self) -> Result<Tensor> {
        Ok(self.tensor.copy()?)
    }
    pub fn receipt(&self) -> &DopeConsumerReceipt {
        &self.receipt
    }
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        ensure!(
            self.tensor.dims() == [1, 512],
            "consumer output width differs"
        );
        let values: Vec<f32> = self.tensor.flatten_all()?.to_vec1()?;
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite consumer output"
        );
        Ok(values)
    }
}
