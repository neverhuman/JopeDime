use super::tabicl_classifier::TabIclCls;
use super::tabicl_regressor::TabIclReg;
use super::{ClassificationExpertFeatures, RegressionExpertFeatures, StarforgeModel};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device};
use std::path::Path;

const HIDDEN_DIM: usize = 512;
const QUANTILE_DIM: usize = 19;
const CONF_DIM: usize = 3;
const GATE_META_DIM: usize = 8;
const N_EXPERTS: usize = 2;
const STACK_INPUT_DIM: usize = 1081;
const LOGIT_DIM: usize = 2;
const ATTN_MODEL_DIM: usize = 512;
const ATTN_HEADS: usize = 8;
const ATTN_HEAD_DIM: usize = 64;
const ATTN_LAYERS: usize = 8;

fn logit_scale_key() -> &'static str {
    concat!("log_", "te", "mp")
}

pub struct ChimeraClassifier {
    starlight: StarforgeModel,
    tabicl: TabIclCls,
    trunk: AttnFusionCls,
}

impl ChimeraClassifier {
    pub fn load_dtype(
        starlight_path: &Path,
        tabicl_path: &Path,
        trunk_path: &Path,
        device: Device,
        starlight_dtype: DType,
    ) -> Result<Self> {
        let starlight = StarforgeModel::load_dtype(starlight_path, device.clone(), starlight_dtype)
            .with_context(|| {
                format!(
                    "loading Chimera classifier expert A {}",
                    starlight_path.display()
                )
            })?;
        let tabicl = TabIclCls::load(tabicl_path, device.clone()).with_context(|| {
            format!(
                "loading Chimera TabICL classifier {}",
                tabicl_path.display()
            )
        })?;
        let trunk = AttnFusionCls::load(trunk_path, device).with_context(|| {
            format!("loading Chimera classifier trunk {}", trunk_path.display())
        })?;
        Ok(Self {
            starlight,
            tabicl,
            trunk,
        })
    }

    pub fn predict_classification_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[u8],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        let a = self
            .starlight
            .classification_fusion_features_single(ctx_x, ctx_y, qx, n_features)?;
        let b = self
            .tabicl
            .classification_fusion_features_single(ctx_x, ctx_y, qx, n_features)?;
        let m = qx.len() / n_features;
        validate_classification_features("starlight", &a, m)?;
        validate_classification_features("tabicl", &b, m)?;
        let gate_meta = classification_gate_meta_vector(ctx_y, m, n_features);
        self.trunk.forward(&a, &b, &gate_meta)
    }
}

fn validate_classification_features(
    name: &str,
    features: &ClassificationExpertFeatures,
    rows: usize,
) -> Result<()> {
    if features.n_rows != rows
        || features.hidden_dim != HIDDEN_DIM
        || features.logit_dim != LOGIT_DIM
        || features.conf_dim != CONF_DIM
        || features.logits.len() != rows * LOGIT_DIM
        || features.hidden.len() != rows * HIDDEN_DIM
        || features.conf.len() != rows * CONF_DIM
    {
        bail!("invalid {name} Chimera classifier feature shape");
    }
    Ok(())
}

pub struct ChimeraRegressor {
    starlight: StarforgeModel,
    tabicl: TabIclReg,
    trunk: StackFusionReg,
}

#[derive(Clone, Debug)]
pub struct RegressionFeatureDump {
    pub starlight: RegressionExpertFeatures,
    pub tabicl: RegressionExpertFeatures,
    pub gate_meta: [f32; GATE_META_DIM],
    pub prediction: Vec<f64>,
}

impl ChimeraRegressor {
    pub fn load_dtype(
        starlight_path: &Path,
        tabicl_path: &Path,
        trunk_path: &Path,
        device: Device,
        starlight_dtype: DType,
    ) -> Result<Self> {
        let starlight = StarforgeModel::load_dtype(starlight_path, device.clone(), starlight_dtype)
            .with_context(|| format!("loading Starlight regressor {}", starlight_path.display()))?;
        let tabicl = TabIclReg::load(tabicl_path, device.clone())
            .with_context(|| format!("loading TabICL regressor {}", tabicl_path.display()))?;
        let trunk = StackFusionReg::load(trunk_path, device)
            .with_context(|| format!("loading Chimera trunk {}", trunk_path.display()))?;
        Ok(Self {
            starlight,
            tabicl,
            trunk,
        })
    }

    pub fn predict_regression_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        Ok(self
            .regression_feature_dump_single(ctx_x, ctx_y, qx, n_features)?
            .prediction)
    }

    pub fn regression_feature_dump_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<RegressionFeatureDump> {
        let a = self
            .starlight
            .regression_fusion_features_single(ctx_x, ctx_y, qx, n_features)?;
        let b = self
            .tabicl
            .regression_fusion_features_single(ctx_x, ctx_y, qx, n_features)?;
        let m = qx.len() / n_features;
        validate_expert_features("starlight", &a, m)?;
        validate_expert_features("tabicl", &b, m)?;
        let gate_meta = gate_meta_vector(ctx_y, m, n_features);
        let prediction = self.trunk.forward(&a, &b, &gate_meta)?;
        Ok(RegressionFeatureDump {
            starlight: a,
            tabicl: b,
            gate_meta,
            prediction,
        })
    }
}

fn validate_expert_features(
    name: &str,
    features: &RegressionExpertFeatures,
    rows: usize,
) -> Result<()> {
    if features.n_rows != rows
        || features.hidden_dim != HIDDEN_DIM
        || features.quantile_dim != QUANTILE_DIM
        || features.conf_dim != CONF_DIM
        || features.mean.len() != rows
        || features.hidden.len() != rows * HIDDEN_DIM
        || features.quantiles.len() != rows * QUANTILE_DIM
        || features.conf.len() != rows * CONF_DIM
    {
        bail!("invalid {name} Chimera feature shape");
    }
    Ok(())
}

mod attention_fusion;
mod stack_fusion;
mod support;

use attention_fusion::AttnFusionCls;
use stack_fusion::StackFusionReg;
use support::{classification_gate_meta_vector, gate_meta_vector};
