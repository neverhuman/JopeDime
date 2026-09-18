use super::tensor_ops::{add_promote, cat_promote};
use super::{
    ClassificationExpertFeatures, RegressionExpertFeatures, StarforgeModel, CLS, DEC_HEADS, EMBED,
    HEAD64, ICL, ICL_HEADS, MAX_CLASSES, REGRESSION_BUCKETS,
};
use anyhow::Result;
use candle_core::{DType, Tensor};

impl StarforgeModel {
    pub(super) fn feature_distribution_layer(
        &self,
        x: &Tensor,
        layer: usize,
        n_train: usize,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let (b, r, c, e) = (dims[0], dims[1], dims[2], dims[3]);
        let x_flat = x
            .transpose(1, 2)?
            .contiguous()?
            .reshape(vec![b * c, r, e])?;
        let x_train = x_flat.narrow(1, 0, n_train)?;
        let ind = self
            .w(&format!(
                "feature_distribution_embedder.layers.{layer}.inducing_vectors"
            ))?
            .reshape(vec![1, 128, EMBED])?
            .broadcast_as(vec![b * c, 128, EMBED])?;
        let p1 = format!("feature_distribution_embedder.layers.{layer}.cross_attn_block1");
        let hidden = self.cross_attention_block(&ind, &x_train, &p1, true)?;
        let p2 = format!("feature_distribution_embedder.layers.{layer}.cross_attn_block2");
        let out = self.cross_attention_block(&x_flat, &hidden, &p2, false)?;
        Ok(out
            .reshape(vec![b, c, r, e])?
            .transpose(1, 2)?
            .contiguous()?)
    }

    pub(super) fn column_aggregate(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let (b, r, _c, e) = (dims[0], dims[1], dims[2], dims[3]);
        let cls = self
            .w("column_aggregator.cls_tokens")?
            .reshape(vec![1, 1, CLS, EMBED])?
            .broadcast_as(vec![b, r, CLS, EMBED])?;
        let mut x = cat_promote(&[&cls, x], 2)?;
        for block in 0..2 {
            x = self.transformer_block(&x, block)?;
        }
        let cls_part = x.narrow(2, 0, CLS)?;
        let cls_out = self.transformer_block_cross(&cls_part, &x, 2)?;
        let out = self.rms(&cls_out, "column_aggregator.out_ln.weight")?;
        Ok(out.reshape(vec![b, r, CLS * e])?)
    }

    pub(super) fn transformer_block(&self, x: &Tensor, block: usize) -> Result<Tensor> {
        let dims = x.dims();
        let (b, r, s, e) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = x.reshape(vec![b * r, s, e])?;
        let prefix = format!("column_aggregator.blocks.{block}");
        let norm = self.rms(&flat, &format!("{prefix}.layernorm.weight"))?;
        let attn = self.self_attention(&norm, &format!("{prefix}.attention"), true)?;
        let mut out = add_promote(x, &attn.reshape(vec![b, r, s, e])?)?;
        let norm2 = self.rms(&out, &format!("{prefix}.layernorm_mlp.weight"))?;
        let mlp = self.mlp(&norm2, &format!("{prefix}.mlp"))?;
        out = add_promote(&out, &mlp)?;
        Ok(out)
    }

    pub(super) fn transformer_block_cross(
        &self,
        query: &Tensor,
        context: &Tensor,
        block: usize,
    ) -> Result<Tensor> {
        let qd = query.dims();
        let cd = context.dims();
        let (b, r, q, e) = (qd[0], qd[1], qd[2], qd[3]);
        let v = cd[2];
        let prefix = format!("column_aggregator.blocks.{block}");
        let q_flat = self
            .rms(query, &format!("{prefix}.layernorm.weight"))?
            .reshape(vec![b * r, q, e])?;
        let c_flat = self
            .rms(context, &format!("{prefix}.layernorm.weight"))?
            .reshape(vec![b * r, v, e])?;
        let attn =
            self.attention_projected(&q_flat, &c_flat, &format!("{prefix}.attention"), None, true)?;
        let mut out = add_promote(query, &attn.reshape(vec![b, r, q, e])?)?;
        let norm2 = self.rms(&out, &format!("{prefix}.layernorm_mlp.weight"))?;
        let mlp = self.mlp(&norm2, &format!("{prefix}.mlp"))?;
        out = add_promote(&out, &mlp)?;
        Ok(out)
    }

    pub(super) fn icl_block(&self, x: &Tensor, layer: usize, n_train: usize) -> Result<Tensor> {
        self.icl_block_with_test_kv(x, layer, n_train, false)
    }

    /// The reviewed upstream foundation expert in full-KV mode disables the model's
    /// inference-only single-KV-head reduction. Foundation hidden parity must
    /// therefore retain every trained K/V head for query rows.
    pub(super) fn icl_block_full_kv(
        &self,
        x: &Tensor,
        layer: usize,
        n_train: usize,
    ) -> Result<Tensor> {
        self.icl_block_with_test_kv(x, layer, n_train, true)
    }

    fn icl_block_with_test_kv(
        &self,
        x: &Tensor,
        layer: usize,
        n_train: usize,
        full_test_kv: bool,
    ) -> Result<Tensor> {
        let dims = x.dims();
        let (b, r, _) = (dims[0], dims[1], dims[2]);
        let prefix = format!("icl_blocks.{layer}");
        let norm = self.rms(x, &format!("{prefix}.layernorm.weight"))?;
        let attn_prefix = format!("{prefix}.icl_attention");
        let q = self
            .linear(&norm, &format!("{attn_prefix}.q_projection.weight"), None)?
            .reshape(vec![b, r, ICL_HEADS, HEAD64])?;
        let x_train = norm.narrow(1, 0, n_train)?;
        let k = self
            .linear(
                &x_train,
                &format!("{attn_prefix}.k_projection.weight"),
                None,
            )?
            .reshape(vec![b, n_train, ICL_HEADS, HEAD64])?;
        let v = self
            .linear(
                &x_train,
                &format!("{attn_prefix}.v_projection.weight"),
                None,
            )?
            .reshape(vec![b, n_train, ICL_HEADS, HEAD64])?;
        let scaling = format!("{attn_prefix}.softmax_scaling_layer");
        let attn = if n_train < r {
            let q_train = q.narrow(1, 0, n_train)?;
            let out_train = self.scaled_attention(&q_train, &k, &v, Some(&scaling), n_train)?;
            let q_test = q.narrow(1, n_train, r - n_train)?;
            let out_test = if full_test_kv {
                self.scaled_attention(&q_test, &k, &v, Some(&scaling), n_train)?
            } else {
                let k_test = k.narrow(2, 0, 1)?;
                let v_test = v.narrow(2, 0, 1)?;
                self.scaled_attention(&q_test, &k_test, &v_test, Some(&scaling), n_train)?
            };
            cat_promote(&[&out_train, &out_test], 1)?
        } else {
            self.scaled_attention(&q, &k, &v, Some(&scaling), n_train)?
        };
        let attn = self.linear(
            &attn.reshape(vec![b, r, ICL])?,
            &format!("{attn_prefix}.out_projection.weight"),
            None,
        )?;
        let mut out = add_promote(x, &attn)?;
        let norm2 = self.rms(&out, &format!("{prefix}.layernorm_mlp.weight"))?;
        let mlp = self.mlp(&norm2, &format!("{prefix}.mlp"))?;
        out = add_promote(&out, &mlp)?;
        Ok(out)
    }

    pub(super) fn decode_many_class(
        &self,
        train: &Tensor,
        test: &Tensor,
        y_cpu: &[u8],
    ) -> Result<Vec<f32>> {
        let logits = self.decode_many_class_logits(train, test, y_cpu)?;
        let m = logits.dims()[0];
        if m == 0 {
            return Ok(Vec::new());
        }
        let probs = candle_nn::ops::softmax(&logits.reshape(vec![1, m, 2])?, 2)?;
        let class1 = probs
            .narrow(2, 1, 1)?
            .reshape(vec![m])?
            .to_dtype(DType::F32)?;
        Ok(class1.to_vec1::<f32>()?)
    }

    pub(super) fn decode_classification_features(
        &self,
        train: &Tensor,
        test: &Tensor,
        y_cpu: &[u8],
    ) -> Result<ClassificationExpertFeatures> {
        let dims = test.dims();
        let (b, m, hidden_dim) = (dims[0], dims[1], dims[2]);
        if m == 0 {
            return Ok(ClassificationExpertFeatures {
                n_rows: 0,
                hidden_dim,
                logit_dim: 2,
                conf_dim: 3,
                logits: Vec::new(),
                hidden: Vec::new(),
                conf: Vec::new(),
            });
        }
        let logits = self.decode_many_class_logits(train, test, y_cpu)?;
        let logits_vec = logits.flatten_all()?.to_vec1::<f32>()?;
        let hidden = test
            .to_dtype(DType::F32)?
            .reshape(vec![b * m, hidden_dim])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut conf = Vec::with_capacity(b * m * 3);
        for pair in logits_vec.chunks_exact(2) {
            let p1 = softmax_class1(pair[0], pair[1]);
            let p0 = 1.0 - p1;
            let entropy = -(p0.max(1e-6).ln() * p0 + p1.max(1e-6).ln() * p1);
            conf.push(p0.max(p1));
            conf.push(entropy);
            conf.push(p1 - p0);
        }
        Ok(ClassificationExpertFeatures {
            n_rows: b * m,
            hidden_dim,
            logit_dim: 2,
            conf_dim: 3,
            logits: logits_vec,
            hidden,
            conf,
        })
    }

    fn decode_many_class_logits(
        &self,
        train: &Tensor,
        test: &Tensor,
        y_cpu: &[u8],
    ) -> Result<Tensor> {
        let dims = test.dims();
        let (b, m, _) = (dims[0], dims[1], dims[2]);
        let n = train.dims()[1];
        if m == 0 {
            return Tensor::zeros(vec![0, 2], DType::F32, self.device()).map_err(Into::into);
        }
        let q = self
            .linear(
                test,
                "many_class_decoder.q_projection.weight",
                Some("many_class_decoder.q_projection.bias"),
            )?
            .reshape(vec![b, m, DEC_HEADS, HEAD64])?;
        let k = self
            .linear(
                train,
                "many_class_decoder.k_projection.weight",
                Some("many_class_decoder.k_projection.bias"),
            )?
            .reshape(vec![b, n, DEC_HEADS, HEAD64])?;
        let mut one_hot = vec![0f32; b * n * MAX_CLASSES];
        for (row, &label) in y_cpu.iter().take(n).enumerate() {
            one_hot[row * MAX_CLASSES + label as usize] = 1.0;
        }
        let value_dtype = if self.autocast_bf16 {
            DType::BF16
        } else {
            DType::F32
        };
        let v = Tensor::from_vec(one_hot, vec![b, n, MAX_CLASSES], &self.device)?
            .to_dtype(value_dtype)?
            .unsqueeze(2)?
            .broadcast_as(vec![b, n, DEC_HEADS, MAX_CLASSES])?;
        let chunks = MAX_CLASSES.div_ceil(HEAD64);
        let pad = chunks * HEAD64 - MAX_CLASSES;
        let v = if pad > 0 {
            let zeros = Tensor::zeros(vec![b, n, DEC_HEADS, pad], value_dtype, &self.device)?;
            cat_promote(&[&v, &zeros], 3)?
        } else {
            v
        };
        let v_folded = v
            .reshape(vec![b, n, DEC_HEADS, chunks, HEAD64])?
            .permute((0, 3, 1, 2, 4))?
            .reshape(vec![b * chunks, n, DEC_HEADS, HEAD64])?;
        let q_folded = q
            .unsqueeze(1)?
            .broadcast_as(vec![b, chunks, m, DEC_HEADS, HEAD64])?
            .reshape(vec![b * chunks, m, DEC_HEADS, HEAD64])?;
        let k_folded = k
            .unsqueeze(1)?
            .broadcast_as(vec![b, chunks, n, DEC_HEADS, HEAD64])?
            .reshape(vec![b * chunks, n, DEC_HEADS, HEAD64])?;
        let out = self.scaled_attention(
            &q_folded,
            &k_folded,
            &v_folded,
            Some("many_class_decoder.softmax_scaling_layer"),
            n,
        )?;
        let out = out
            .reshape(vec![b, chunks, m, DEC_HEADS, HEAD64])?
            .permute((0, 2, 3, 1, 4))?
            .reshape(vec![b, m, DEC_HEADS, chunks * HEAD64])?
            .narrow(3, 0, MAX_CLASSES)?;
        let logits = (out
            .mean(2)?
            .transpose(0, 1)?
            .to_dtype(DType::F32)?
            .maximum(1e-5)?
            + 3e-5)?
            .log()?;
        logits
            .narrow(2, 0, 2)?
            .reshape(vec![b * m, 2])?
            .to_dtype(DType::F32)
            .map_err(Into::into)
    }

    pub(super) fn decode_regression(&self, test: &Tensor) -> Result<Vec<f32>> {
        let dims = test.dims();
        let (b, m, _) = (dims[0], dims[1], dims[2]);
        if m == 0 {
            return Ok(Vec::new());
        }
        let hidden = self
            .linear(
                test,
                "output_projection.0.weight",
                Some("output_projection.0.bias"),
            )?
            .gelu_erf()?;
        let logits = self
            .linear(
                &hidden,
                "output_projection.2.weight",
                Some("output_projection.2.bias"),
            )?
            .to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax(&logits, 2)?;
        let borders = self.w("regression_borders")?.to_dtype(DType::F32)?;
        let left = borders.narrow(0, 0, REGRESSION_BUCKETS)?;
        let right = borders.narrow(0, 1, REGRESSION_BUCKETS)?;
        let centres = ((&left + &right)? * 0.5)?.reshape(vec![REGRESSION_BUCKETS, 1])?;
        let mean = probs
            .reshape(vec![b * m, REGRESSION_BUCKETS])?
            .matmul(&centres)?
            .reshape(vec![b * m])?;
        Ok(mean.to_vec1::<f32>()?)
    }

    pub(super) fn decode_regression_features(
        &self,
        test: &Tensor,
        target_mean: f32,
        target_std: f32,
    ) -> Result<RegressionExpertFeatures> {
        let dims = test.dims();
        let (b, m, hidden_dim) = (dims[0], dims[1], dims[2]);
        if m == 0 {
            return Ok(RegressionExpertFeatures {
                n_rows: 0,
                hidden_dim,
                quantile_dim: 19,
                conf_dim: 3,
                mean: Vec::new(),
                hidden: Vec::new(),
                quantiles: Vec::new(),
                conf: Vec::new(),
            });
        }
        let hidden = test
            .to_dtype(DType::F32)?
            .reshape(vec![b * m, hidden_dim])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let h = self
            .linear(
                test,
                "output_projection.0.weight",
                Some("output_projection.0.bias"),
            )?
            .gelu_erf()?;
        let logits = self
            .linear(
                &h,
                "output_projection.2.weight",
                Some("output_projection.2.bias"),
            )?
            .to_dtype(DType::F32)?;
        let probs = candle_nn::ops::softmax(&logits, 2)?
            .reshape(vec![b * m, REGRESSION_BUCKETS])?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let borders = self.w("regression_borders")?.to_dtype(DType::F32)?;
        let left = borders.narrow(0, 0, REGRESSION_BUCKETS)?.to_vec1::<f32>()?;
        let right = borders.narrow(0, 1, REGRESSION_BUCKETS)?.to_vec1::<f32>()?;
        let centres: Vec<f32> = left
            .iter()
            .zip(&right)
            .map(|(l, r)| 0.5 * (l + r))
            .collect();
        let q_levels = [
            0.05f32, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
            0.75, 0.80, 0.85, 0.90, 0.95,
        ];
        let rows = b * m;
        let mut mean = Vec::with_capacity(rows);
        let mut quantiles = Vec::with_capacity(rows * q_levels.len());
        let mut conf = Vec::with_capacity(rows * 3);
        for row in 0..rows {
            let p = &probs[row * REGRESSION_BUCKETS..(row + 1) * REGRESSION_BUCKETS];
            let mean_z = p.iter().zip(&centres).map(|(pp, c)| pp * c).sum::<f32>();
            let mut var_z = 0.0f32;
            let mut entropy = 0.0f32;
            for (&pp, &centre) in p.iter().zip(&centres) {
                let d = centre - mean_z;
                var_z += pp * d * d;
                entropy -= pp.max(1e-9).ln() * pp;
            }
            let mean_y = mean_z * target_std + target_mean;
            mean.push(mean_y);
            for &q in &q_levels {
                let idx = quantile_index(p, q);
                quantiles.push(centres[idx] * target_std + target_mean);
            }
            conf.push(var_z.max(0.0).sqrt() * target_std);
            conf.push(entropy);
            conf.push(mean_y);
        }
        Ok(RegressionExpertFeatures {
            n_rows: rows,
            hidden_dim,
            quantile_dim: q_levels.len(),
            conf_dim: 3,
            mean,
            hidden,
            quantiles,
            conf,
        })
    }
}

fn quantile_index(probs: &[f32], level: f32) -> usize {
    let mut cdf = 0.0f32;
    for (idx, &p) in probs.iter().enumerate() {
        cdf += p;
        if cdf >= level {
            return idx;
        }
    }
    probs.len().saturating_sub(1)
}

fn softmax_class1(a: f32, b: f32) -> f32 {
    let mx = a.max(b);
    let ea = (a - mx).exp();
    let eb = (b - mx).exp();
    eb / (ea + eb).max(1e-12)
}
