//! Tabular classifier/regressor inference and expert feature extraction.
//!
//! [`StarforgeModel`] accepts flat, row-major feature matrices: `ctx_x` has
//! `ctx_y.len() * n_features` entries and `qx` has a whole number of query rows
//! with the same feature width. Callers select the context/query split and supply
//! the context labels; this module does not select or authorize dataset folds.
//! Prediction methods require a positive feature width (see their panic contracts).
//!
//! Raw weight loaders parse safetensors but do not verify bundle identity or
//! provenance. For the pinned bundle contract, use
//! [`LoadedModelBundle::load`](crate::bundle::LoadedModelBundle::load) and
//! [`read_verified`](crate::bundle::LoadedModelBundle::read_verified) before
//! passing artifact bytes to [`StarforgeModel::load_buffer`].

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

mod attention;
mod chimera;
mod inference;
mod layers;
#[cfg(test)]
mod proptests;
mod tabicl_classifier;
mod tabicl_common;
mod tabicl_regressor;
mod tensor_ops;
#[cfg(test)]
mod tests;

#[cfg(test)]
use inference::{average_ranks_one_based, rank_aggregate};
use inference::{detect_model_kind, median_of_views, regression_query_chunk, target_mean_std};
use tensor_ops::{
    add_train_y_column, add_train_y_icl, apply_feature_view, decompress_zstd, preprocess_group,
    preprocess_group_stable, FeatureView,
};

pub use chimera::{ChimeraClassifier, ChimeraRegressor, RegressionFeatureDump};
pub use tabicl_regressor::TabIclFoundation;

const EMBED: usize = 128;
const CLS: usize = 4;
const ICL: usize = 512;
const ICL_HEADS: usize = 8;
const DEC_HEADS: usize = 6;
const HEAD16: usize = 16;
const HEAD64: usize = 64;
const MAX_CLASSES: usize = 160;
const REGRESSION_BUCKETS: usize = 5000;
const REGRESSION_MAX_TOTAL_ROWS: usize = 1024;

/// Model family detected from checkpoint tensor names.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StarforgeModelKind {
    /// Classification checkpoint.
    Classifier,
    /// Regression checkpoint.
    Regressor,
}

/// Loaded checkpoint and inference device; see the module's input/verification contracts.
pub struct StarforgeModel {
    weights: HashMap<String, Tensor>,
    device: Device,
    autocast_bf16: bool,
    kind: StarforgeModelKind,
}

#[derive(Clone, Debug)]
pub struct ClassificationExpertFeatures {
    pub n_rows: usize,
    pub hidden_dim: usize,
    pub logit_dim: usize,
    pub conf_dim: usize,
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub conf: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct RegressionExpertFeatures {
    pub n_rows: usize,
    pub hidden_dim: usize,
    pub quantile_dim: usize,
    pub conf_dim: usize,
    pub mean: Vec<f32>,
    pub hidden: Vec<f32>,
    pub quantiles: Vec<f32>,
    pub conf: Vec<f32>,
}

impl StarforgeModel {
    /// Load a safetensors file (optionally `.zst` compressed) with F32 inference.
    /// Bundle identity and provenance remain the caller's responsibility.
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        Self::load_dtype(path, device, DType::F32)
    }

    /// Load uncompressed safetensors bytes with F32 inference.
    /// Parsing these bytes does not verify their bundle identity or provenance.
    pub fn load_buffer(bytes: &[u8], device: Device) -> Result<Self> {
        Self::load_buffer_dtype(bytes, device, DType::F32)
    }

    /// Load uncompressed safetensors bytes; `BF16` enables the BF16 inference path.
    /// Parsing these bytes does not verify their bundle identity or provenance.
    pub fn load_buffer_dtype(bytes: &[u8], device: Device, dtype: DType) -> Result<Self> {
        let weights = candle_core::safetensors::load_buffer(bytes, &device)
            .context("loading verified safetensors buffer")?;
        Self::from_weights(weights, device, dtype)
    }

    /// Load a safetensors file, decompressing `.zst` in memory.
    /// `BF16` enables the BF16 inference path; bundle provenance is caller-owned.
    pub fn load_dtype(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        let is_zst = path.extension().and_then(|e| e.to_str()) == Some("zst");
        let weights = if is_zst {
            // Weights are shipped zstd-compressed (`*.safetensors.zst`) to shrink the
            // container image; decompress in-memory at load time (no temp file, no
            // shell) and hand the raw safetensors bytes to candle.
            let compressed =
                std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
            let raw = decompress_zstd(&compressed)
                .with_context(|| format!("decompressing {}", path.display()))?;
            candle_core::safetensors::load_buffer(&raw, &device)
                .with_context(|| format!("loading decompressed {}", path.display()))?
        } else {
            candle_core::safetensors::load(path, &device)
                .with_context(|| format!("loading {}", path.display()))?
        };
        Self::from_weights(weights, device, dtype)
    }

    fn from_weights(
        weights: HashMap<String, Tensor>,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let kind = detect_model_kind(&weights)?;
        Ok(Self {
            weights,
            device,
            autocast_bf16: dtype == DType::BF16,
            kind,
        })
    }

    /// Device selected when the checkpoint was loaded.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Family detected from the loaded checkpoint.
    pub fn kind(&self) -> StarforgeModelKind {
        self.kind
    }

    /// Predict classification scores using one feature view.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a classifier checkpoint.
    pub fn predict_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[u8],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        self.ensure_kind(StarforgeModelKind::Classifier)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad feature matrix length");
        }
        let grouped = preprocess_group(ctx_x, qx, n, m, n_features);
        let x = Tensor::from_vec(grouped, vec![1, n + m, n_features, 6], &self.device)?;
        let y_ids: Vec<u32> = ctx_y.iter().map(|&v| v as u32).collect();
        let y = Tensor::from_vec(y_ids, vec![n], &self.device)?;
        let probs = self.forward(&x, &y, ctx_y, n, m)?;
        Ok(probs.into_iter().map(|v| v as f64).collect())
    }

    /// Predict classification scores by taking each row's median across six views.
    /// Despite the historical name, this returns predictions, not an AUC metric.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a classifier checkpoint.
    pub fn predict_ensemble_auc(
        &self,
        ctx_x: &[f32],
        ctx_y: &[u8],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        self.ensure_kind(StarforgeModelKind::Classifier)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad feature matrix length");
        }
        if m == 0 {
            return Ok(Vec::new());
        }
        let views = [
            FeatureView::Identity,
            FeatureView::QuantileNormal,
            FeatureView::QuantileUniform,
            FeatureView::YeoJohnson,
            FeatureView::Robust,
            FeatureView::SignedLog,
        ];
        let mut per_view = Vec::with_capacity(views.len());
        for view in views {
            let (cx, q) = apply_feature_view(view, ctx_x, qx, n, m, n_features);
            per_view.push(self.predict_single(&cx, ctx_y, &q, n_features)?);
        }
        // Median of the 6 feature views — row-stable, independent of the request batch (mirrors the
        // regression head). The previous `rank_aggregate` ranked each query against the OTHER queries
        // in the same request, so a row scored differently alone vs. inside a larger batch.
        Ok(median_of_views(&per_view, m))
    }

    /// Extract classification expert features using one feature view.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a classifier checkpoint.
    pub fn classification_fusion_features_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[u8],
        qx: &[f32],
        n_features: usize,
    ) -> Result<ClassificationExpertFeatures> {
        self.ensure_kind(StarforgeModelKind::Classifier)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad classification feature matrix length");
        }
        let grouped = preprocess_group(ctx_x, qx, n, m, n_features);
        let x = Tensor::from_vec(grouped, vec![1, n + m, n_features, 6], &self.device)?;
        let y_ids: Vec<u32> = ctx_y.iter().map(|&v| v as u32).collect();
        let y = Tensor::from_vec(y_ids, vec![n], &self.device)?;
        self.forward_classification_features(&x, &y, ctx_y, n, m)
    }

    /// Predict regression values in the supplied context labels' original scale.
    /// Requires at least one context row; labels are standardized internally.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a regression checkpoint.
    pub fn predict_regression_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        self.ensure_kind(StarforgeModelKind::Regressor)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad regression feature matrix length");
        }
        if n == 0 {
            bail!("regression prediction requires at least one context row");
        }
        let (mu, sd) = target_mean_std(ctx_y);
        let ctx_y_z: Vec<f32> = ctx_y.iter().map(|&v| (v - mu) / sd).collect();
        let query_chunk = regression_query_chunk(n, m);
        let mut out = Vec::with_capacity(m);
        for start in (0..m).step_by(query_chunk.max(1)) {
            let end = (start + query_chunk).min(m);
            let q_slice = &qx[start * n_features..end * n_features];
            let grouped = preprocess_group(ctx_x, q_slice, n, end - start, n_features);
            let x = Tensor::from_vec(
                grouped,
                vec![1, n + end - start, n_features, 6],
                &self.device,
            )?;
            let y = Tensor::from_vec(ctx_y_z.clone(), vec![1, n, 1], &self.device)?;
            let pred_z = self.forward_regression(&x, &y, n, end - start)?;
            out.extend(pred_z.into_iter().map(|v| (v * sd + mu) as f64));
        }
        Ok(out)
    }

    /// Extract regression expert features, standardizing context labels internally.
    /// Requires at least one context row.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a regression checkpoint.
    pub fn regression_fusion_features_single(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<RegressionExpertFeatures> {
        self.ensure_kind(StarforgeModelKind::Regressor)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad regression feature matrix length");
        }
        if n == 0 {
            bail!("regression fusion features require at least one context row");
        }
        let (mu, sd) = target_mean_std(ctx_y);
        let ctx_y_z: Vec<f32> = ctx_y.iter().map(|&v| (v - mu) / sd).collect();
        self.regression_fusion_features_normalized(ctx_x, &ctx_y_z, qx, n_features, (mu, sd), false)
    }

    /// The caller supplies the frozen export's already standardized context labels.
    /// This avoids a second normalization and preserves the native FP64-to-FP32 seam.
    /// Returns an error for zero feature width, empty context/query rows, inconsistent
    /// matrix lengths, or non-finite input values.
    pub fn regression_fusion_features_standardized(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<RegressionExpertFeatures> {
        self.ensure_kind(StarforgeModelKind::Regressor)?;
        if n_features == 0
            || ctx_y.is_empty()
            || ctx_x.len() != ctx_y.len() * n_features
            || qx.is_empty()
            || !qx.len().is_multiple_of(n_features)
            || ctx_x.iter().chain(ctx_y).chain(qx).any(|v| !v.is_finite())
        {
            bail!("invalid frozen foundation input");
        }
        self.regression_fusion_features_normalized(ctx_x, ctx_y, qx, n_features, (0.0, 1.0), true)
    }

    fn regression_fusion_features_normalized(
        &self,
        ctx_x: &[f32],
        ctx_y_z: &[f32],
        qx: &[f32],
        n_features: usize,
        target_scale: (f32, f32),
        stable_moments: bool,
    ) -> Result<RegressionExpertFeatures> {
        let (mu, sd) = target_scale;
        let n = ctx_y_z.len();
        let m = qx.len() / n_features;
        let query_chunk = regression_query_chunk(n, m);
        let mut mean = Vec::with_capacity(m);
        let mut hidden = Vec::with_capacity(m * ICL);
        let mut quantiles = Vec::with_capacity(m * 19);
        let mut conf = Vec::with_capacity(m * 3);
        for start in (0..m).step_by(query_chunk.max(1)) {
            let end = (start + query_chunk).min(m);
            let q_slice = &qx[start * n_features..end * n_features];
            let grouped = if stable_moments {
                preprocess_group_stable(ctx_x, q_slice, n, end - start, n_features)
            } else {
                preprocess_group(ctx_x, q_slice, n, end - start, n_features)
            };
            let x = Tensor::from_vec(
                grouped,
                vec![1, n + end - start, n_features, 6],
                &self.device,
            )?;
            let y = Tensor::from_vec(ctx_y_z.to_vec(), vec![1, n, 1], &self.device)?;
            let chunk = self.forward_regression_features(&x, &y, n, end - start, mu, sd)?;
            mean.extend(chunk.mean);
            hidden.extend(chunk.hidden);
            quantiles.extend(chunk.quantiles);
            conf.extend(chunk.conf);
        }
        Ok(RegressionExpertFeatures {
            n_rows: m,
            hidden_dim: ICL,
            quantile_dim: 19,
            conf_dim: 3,
            mean,
            hidden,
            quantiles,
            conf,
        })
    }

    /// Predict regression values by taking each row's median across six views.
    ///
    /// # Panics
    /// Panics if `n_features` is zero for a regression checkpoint.
    pub fn predict_regression_ensemble(
        &self,
        ctx_x: &[f32],
        ctx_y: &[f32],
        qx: &[f32],
        n_features: usize,
    ) -> Result<Vec<f64>> {
        self.ensure_kind(StarforgeModelKind::Regressor)?;
        let n = ctx_y.len();
        let m = qx.len() / n_features;
        if ctx_x.len() != n * n_features || qx.len() != m * n_features {
            bail!("bad regression feature matrix length");
        }
        let views = [
            FeatureView::Identity,
            FeatureView::QuantileNormal,
            FeatureView::QuantileUniform,
            FeatureView::YeoJohnson,
            FeatureView::Robust,
            FeatureView::SignedLog,
        ];
        let mut per_view = Vec::with_capacity(views.len());
        for view in views {
            let (cx, q) = apply_feature_view(view, ctx_x, qx, n, m, n_features);
            per_view.push(self.predict_regression_single(&cx, ctx_y, &q, n_features)?);
        }
        Ok(median_of_views(&per_view, m))
    }

    fn forward(
        &self,
        grouped: &Tensor,
        y_ids: &Tensor,
        y_cpu: &[u8],
        n_train: usize,
        n_query: usize,
    ) -> Result<Vec<f32>> {
        let mut x = self.linear(grouped, "x_embed.weight", Some("x_embed.bias"))?;
        if n_train > 0 {
            let y_col = self
                .w("col_y_encoder.embedding.weight")?
                .embedding(y_ids)?
                .reshape(vec![1, n_train, EMBED])?;
            x = add_train_y_column(&x, &y_col, n_train)?;
        }

        for layer in 0..3 {
            x = self.feature_distribution_layer(&x, layer, n_train)?;
        }
        let mut x = self.column_aggregate(&x)?;
        let y_icl = self
            .w("icl_y_encoder.embedding.weight")?
            .embedding(y_ids)?
            .reshape(vec![1, n_train, ICL])?;
        x = add_train_y_icl(&x, &y_icl, n_train)?;
        for layer in 0..24 {
            x = self.icl_block(&x, layer, n_train)?;
        }
        x = self.rms(&x, "output_norm.weight")?;
        let train = x.narrow(1, 0, n_train)?;
        let test = x.narrow(1, n_train, n_query)?;
        self.decode_many_class(&train, &test, y_cpu)
    }

    fn forward_classification_features(
        &self,
        grouped: &Tensor,
        y_ids: &Tensor,
        y_cpu: &[u8],
        n_train: usize,
        n_query: usize,
    ) -> Result<ClassificationExpertFeatures> {
        let mut x = self.linear(grouped, "x_embed.weight", Some("x_embed.bias"))?;
        if n_train > 0 {
            let y_col = self
                .w("col_y_encoder.embedding.weight")?
                .embedding(y_ids)?
                .reshape(vec![1, n_train, EMBED])?;
            x = add_train_y_column(&x, &y_col, n_train)?;
        }

        for layer in 0..3 {
            x = self.feature_distribution_layer(&x, layer, n_train)?;
        }
        let mut x = self.column_aggregate(&x)?;
        let y_icl = self
            .w("icl_y_encoder.embedding.weight")?
            .embedding(y_ids)?
            .reshape(vec![1, n_train, ICL])?;
        x = add_train_y_icl(&x, &y_icl, n_train)?;
        for layer in 0..24 {
            x = self.icl_block(&x, layer, n_train)?;
        }
        x = self.rms(&x, "output_norm.weight")?;
        let train = x.narrow(1, 0, n_train)?;
        let test = x.narrow(1, n_train, n_query)?;
        self.decode_classification_features(&train, &test, y_cpu)
    }

    fn forward_regression(
        &self,
        grouped: &Tensor,
        y_values: &Tensor,
        n_train: usize,
        n_query: usize,
    ) -> Result<Vec<f32>> {
        let mut x = self.linear(grouped, "x_embed.weight", Some("x_embed.bias"))?;
        if n_train > 0 {
            let y_col =
                self.linear(y_values, "col_y_encoder.weight", Some("col_y_encoder.bias"))?;
            x = add_train_y_column(&x, &y_col, n_train)?;
        }

        for layer in 0..3 {
            x = self.feature_distribution_layer(&x, layer, n_train)?;
        }
        let mut x = self.column_aggregate(&x)?;
        let y_icl = self.linear(y_values, "icl_y_encoder.weight", Some("icl_y_encoder.bias"))?;
        x = add_train_y_icl(&x, &y_icl, n_train)?;
        for layer in 0..24 {
            x = self.icl_block(&x, layer, n_train)?;
        }
        x = self.rms(&x, "output_norm.weight")?;
        let test = x.narrow(1, n_train, n_query)?;
        self.decode_regression(&test)
    }

    fn forward_regression_features(
        &self,
        grouped: &Tensor,
        y_values: &Tensor,
        n_train: usize,
        n_query: usize,
        target_mean: f32,
        target_std: f32,
    ) -> Result<RegressionExpertFeatures> {
        let mut x = self.linear(grouped, "x_embed.weight", Some("x_embed.bias"))?;
        if n_train > 0 {
            let y_col =
                self.linear(y_values, "col_y_encoder.weight", Some("col_y_encoder.bias"))?;
            x = add_train_y_column(&x, &y_col, n_train)?;
        }

        for layer in 0..3 {
            x = self.feature_distribution_layer(&x, layer, n_train)?;
        }
        let mut x = self.column_aggregate(&x)?;
        let y_icl = self.linear(y_values, "icl_y_encoder.weight", Some("icl_y_encoder.bias"))?;
        x = add_train_y_icl(&x, &y_icl, n_train)?;
        for layer in 0..24 {
            x = self.icl_block_full_kv(&x, layer, n_train)?;
        }
        x = self.rms(&x, "output_norm.weight")?;
        let test = x.narrow(1, n_train, n_query)?;
        self.decode_regression_features(&test, target_mean, target_std)
    }

    fn ensure_kind(&self, expected: StarforgeModelKind) -> Result<()> {
        if self.kind != expected {
            bail!(
                "Starforge model kind mismatch: loaded {:?}, requested {:?}",
                self.kind,
                expected
            );
        }
        Ok(())
    }
}
