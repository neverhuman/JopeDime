//! Shared product binding; models and controller request catalogs load once per run.
use super::{
    ComponentInputs, DopeConsumer, DopeEncoder, LandmarkArgs, LandmarkConsumer, SessionOptions,
};
use crate::bundle::VerifiedJopePolicyAssets;
use crate::jope_policy::JopePolicyModel;
use anyhow::{ensure, Context, Result};
use std::sync::Arc;

pub fn product_session_options(
    args: &LandmarkArgs,
    encoder: &DopeEncoder,
) -> Result<SessionOptions> {
    product_session_options_with_assets(args, encoder, None)
}

/// A sealed-package adapter may supply the already content-verified selected closure.
pub fn product_session_options_with_assets(
    args: &LandmarkArgs,
    encoder: &DopeEncoder,
    packaged: Option<&VerifiedJopePolicyAssets>,
) -> Result<SessionOptions> {
    args.validate()?;
    let mut options = SessionOptions::default();
    match args.landmark_consumer {
        None => {}
        Some(LandmarkConsumer::Direct) => {
            options.consumer = Some(Arc::new(DopeConsumer::encoding_only()))
        }
        Some(LandmarkConsumer::JopeDl) => {
            let inputs = ComponentInputs::load(
                args.landmark_components
                    .as_deref()
                    .context("jope-dl component requests are required")?,
            )?;
            ensure!(
                packaged.is_none() || args.jope_policy_assets.is_none(),
                "ambiguous policy asset sources"
            );
            let external;
            let assets = if let Some(assets) = packaged {
                assets
            } else {
                external = VerifiedJopePolicyAssets::load(args.jope_policy_assets.as_deref()
                    .context("jope-dl requires --jope-policy-assets or a reviewed packaged policy projection")?)?;
                &external
            };
            let model =
                JopePolicyModel::load_policy_assets_on_device(assets, encoder.device().clone())?;
            options.consumer = Some(Arc::new(DopeConsumer::jope_dl(Arc::new(model))?));
            options.component_inputs = Some(Arc::new(inputs));
        }
    }
    Ok(options)
}

/// Read-only controller configuration is protected from output aliases alongside request/assets.
pub fn component_authority_root() -> std::path::PathBuf {
    jope_paths::jope_state()
}
