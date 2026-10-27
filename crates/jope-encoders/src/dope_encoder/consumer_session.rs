//! Fixed product binding and resident component reuse across a single encoding session.
use super::{
    AuthenticatedFusionComponents, DopeConsumedEncoding, DopeConsumer, DopeConsumerMode,
    EncodingSnapshot, FitScope, LandmarkArgs, ResidentFusionComponents, SessionOptions,
};
use crate::landmark_args::LandmarkConsumer;
use anyhow::{ensure, Result};
use std::sync::Arc;

pub(super) struct ConsumerSession {
    consumer: Option<Arc<DopeConsumer>>,
    admission: Option<Arc<AuthenticatedFusionComponents>>,
    components: Option<ResidentFusionComponents>,
    cached: Option<DopeConsumedEncoding>,
}
impl ConsumerSession {
    pub(super) fn require_active(&self) -> Result<()> {
        if let Some(admission) = &self.admission {
            admission.require_active()?;
        }
        Ok(())
    }
    pub(super) fn new(
        args: &LandmarkArgs,
        options: &SessionOptions,
        encoder: &super::DopeEncoder,
        fit: &FitScope,
        landmark_scope: &str,
    ) -> Result<Self> {
        match (args.landmark_consumer, options.consumer.as_ref()) {
            (None, None) => ensure!(
                options.authenticated_components.is_none(),
                "component admission requires explicit consumer selection"
            ),
            (Some(LandmarkConsumer::Direct), Some(consumer)) => {
                ensure!(
                    consumer.mode() == DopeConsumerMode::EncodingOnly,
                    "direct selection requires the raw consumer"
                );
                ensure!(
                    options.authenticated_components.is_none(),
                    "direct selection does not consume authenticated foundation/target components"
                );
            }
            (Some(LandmarkConsumer::JopeDl), Some(consumer)) => {
                ensure!(
                    consumer.mode() == DopeConsumerMode::JopeDl
                        && consumer.projection_identity().is_some(),
                    "jope-dl selection requires verified projected policy assets"
                );
                ensure!(
                    options.components.is_none() && options.authenticated_components.is_some(),
                    "jope-dl requires authenticated components; unsigned components are forbidden"
                );
            }
            _ => anyhow::bail!("selected consumer differs from the bound session consumer"),
        }
        let components = if let Some(admission) = &options.authenticated_components {
            admission.require_active()?;
            Some(admission.bind(fit, landmark_scope)?)
        } else {
            options.components.clone()
        };
        Ok(Self {
            consumer: options.consumer.clone(),
            admission: options.authenticated_components.clone(),
            components: components
                .as_ref()
                .map(|c| c.to_device(encoder.device()))
                .transpose()?,
            cached: None,
        })
    }
    /// Authentication is checked even for repeated inputs; the MLP is rerun only for new inputs.
    pub(super) fn apply(&mut self, mut snapshot: EncodingSnapshot) -> Result<EncodingSnapshot> {
        self.require_active()?;
        if let Some(components) = &self.components {
            snapshot = snapshot.with_resident_components(components)?;
        }
        if let Some(consumer) = &self.consumer {
            let output = if let Some(cached) = self.cached.as_ref().filter(|cached| {
                cached.receipt().input_conditioning_digest == snapshot.receipt.conditioning_digest
            }) {
                cached.clone()
            } else {
                let output = consumer.consume_snapshot(&snapshot)?;
                if let Some(admission) = &self.admission {
                    output.with_component_admission(admission.identity())?
                } else {
                    output
                }
            };
            self.require_active()?;
            snapshot.receipt.consumer = Some(output.receipt().clone());
            snapshot.consumer = Some(output.clone());
            self.cached = Some(output);
        }
        Ok(snapshot)
    }
}
