//! Optional raw features and learned policy contexts, with distinct semantic products.
use super::{
    sha256, ComponentAdmissionIdentity, Dope512, DopeCheckpointIdentity, DopeFusion,
    EncodingSnapshot, LandmarkEncodingReceipt,
};
use crate::bundle::JopePolicyAssetsIdentity;
use crate::jope_policy::{JopePolicyContext, JopePolicyModel};
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
    pub fn encoding_only() -> Self {
        Self {
            model: None,
            cpu_pool: None,
        }
    }

    /// Low-level execution surface. Product sessions also require projected assets and admission.
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

    pub fn projection_identity(&self) -> Option<&JopePolicyAssetsIdentity> {
        self.model
            .as_ref()
            .and_then(|model| model.projection_identity())
    }

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
        let (product, model_sha, components_digest) = if let Some(model) = &self.model {
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
            let policy_sha256 = model.weights_sha256().to_owned();
            (
                DopeConsumerProduct::JopeDl(JopeContext512 {
                    tensor,
                    policy_sha256: policy_sha256.clone(),
                }),
                Some(policy_sha256),
                Some(fusion.components_digest.clone()),
            )
        } else {
            (DopeConsumerProduct::Direct(embedding.clone()), None, None)
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
        let projection_identity = self.projection_identity().cloned();
        let base_digest = sha256(&serde_json::to_vec(&(
            recipe,
            &embedding.checkpoint_identity,
            &embedding.source_digest,
            &embedding.conditioning_digest,
            &model_sha,
            &components_digest,
        ))?);
        // Preserve the pre-projection execution identity for existing low-level SDK users.
        let conditioning_digest = if let Some(identity) = &projection_identity {
            sha256(&serde_json::to_vec(&(
                "jope.dope-policy-projection/v1",
                &base_digest,
                identity,
            ))?)
        } else {
            base_digest
        };
        Ok(DopeConsumedEncoding {
            product,
            receipt: DopeConsumerReceipt {
                mode: self.mode(),
                recipe: recipe.into(),
                checkpoint_identity: embedding.checkpoint_identity.clone(),
                jope_weights_sha256: model_sha,
                components_digest,
                source_digest: embedding.source_digest.clone(),
                input_conditioning_digest: embedding.conditioning_digest.clone(),
                projection_identity,
                component_admission: None,
                conditioning_digest,
            },
        })
    }
}

/// Raw features alone can be assembled into dataset1536/policy2048.
#[derive(Clone, Debug)]
pub enum DopeConsumerProduct {
    Direct(Dope512),
    JopeDl(JopeContext512),
}

/// Already passed through the policy context MLP. This type has no fusion or MLP method.
#[derive(Clone, Debug)]
pub struct JopeContext512 {
    tensor: Tensor,
    policy_sha256: String,
}
impl JopeContext512 {
    pub fn tensor_copy(&self) -> Result<Tensor> {
        Ok(self.tensor.copy()?)
    }
    /// Explicit host bridge to the current CPU decoder. This never invokes the context MLP.
    pub fn decoder_context(&self, model: &JopePolicyModel) -> Result<JopePolicyContext> {
        ensure!(
            model.weights_sha256() == self.policy_sha256,
            "derived context and decoder policy checkpoints differ"
        );
        ensure!(
            model.device().is_cpu(),
            "JOPE decoder seam supports CPU only"
        );
        model.context_from_values(&finite_values(&self.tensor)?)
    }
}

#[derive(Clone, Debug)]
pub struct DopeConsumedEncoding {
    product: DopeConsumerProduct,
    receipt: DopeConsumerReceipt,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DopeConsumerReceipt {
    pub mode: DopeConsumerMode,
    pub recipe: String,
    pub checkpoint_identity: DopeCheckpointIdentity,
    pub jope_weights_sha256: Option<String>,
    pub components_digest: Option<String>,
    pub source_digest: String,
    #[serde(default)]
    pub input_conditioning_digest: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projection_identity: Option<JopePolicyAssetsIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub component_admission: Option<ComponentAdmissionIdentity>,
    pub conditioning_digest: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct DopeConsumerExport {
    pub receipt: DopeConsumerReceipt,
    pub embedding: Vec<f32>,
}

impl DopeConsumedEncoding {
    pub fn product(&self) -> &DopeConsumerProduct {
        &self.product
    }
    fn tensor(&self) -> &Tensor {
        match &self.product {
            DopeConsumerProduct::Direct(raw) => &raw.tensor,
            DopeConsumerProduct::JopeDl(context) => &context.tensor,
        }
    }
    pub fn tensor_copy(&self) -> Result<Tensor> {
        Ok(self.tensor().copy()?)
    }
    pub fn receipt(&self) -> &DopeConsumerReceipt {
        &self.receipt
    }
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        finite_values(self.tensor())
    }
    pub fn to_export(&self) -> Result<DopeConsumerExport> {
        Ok(DopeConsumerExport {
            receipt: self.receipt.clone(),
            embedding: self.to_vec()?,
        })
    }
    pub(crate) fn with_component_admission(
        mut self,
        identity: &ComponentAdmissionIdentity,
    ) -> Result<Self> {
        ensure!(
            self.receipt.mode == DopeConsumerMode::JopeDl,
            "component admission requires a learned context"
        );
        if let Some(existing) = &self.receipt.component_admission {
            ensure!(existing == identity, "consumer admission cannot change");
            return Ok(self);
        }
        self.receipt.conditioning_digest = sha256(&serde_json::to_vec(&(
            "jope.dope-component-admission/v1",
            &self.receipt.conditioning_digest,
            identity,
        ))?);
        self.receipt.component_admission = Some(identity.clone());
        Ok(self)
    }
    pub(super) fn require_snapshot(
        &self,
        raw: &Dope512,
        receipt: &LandmarkEncodingReceipt,
    ) -> Result<()> {
        ensure!(
            receipt.consumer.as_ref() == Some(&self.receipt)
                && self.receipt.source_digest == raw.source_digest
                && self.receipt.checkpoint_identity == raw.checkpoint_identity
                && self.receipt.input_conditioning_digest == raw.conditioning_digest,
            "consumer receipt differs from resident product or raw encoding"
        );
        Ok(())
    }
    pub(super) fn require_same_binding(&self, next: &Self) -> Result<()> {
        let a = &self.receipt;
        let b = &next.receipt;
        ensure!(
            a.mode == b.mode
                && a.recipe == b.recipe
                && a.checkpoint_identity == b.checkpoint_identity
                && a.jope_weights_sha256 == b.jope_weights_sha256
                && a.components_digest == b.components_digest
                && a.source_digest == b.source_digest
                && a.projection_identity == b.projection_identity
                && a.component_admission == b.component_admission,
            "consumer checkpoint, projection, components or recipe cannot change within a session"
        );
        Ok(())
    }
}
fn finite_values(tensor: &Tensor) -> Result<Vec<f32>> {
    ensure!(tensor.dims() == [1, 512], "consumer output width differs");
    let values: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite consumer output"
    );
    Ok(values)
}
