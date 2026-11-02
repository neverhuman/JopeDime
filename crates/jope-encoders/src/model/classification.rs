use super::*;

impl StarforgeModel {
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
}
