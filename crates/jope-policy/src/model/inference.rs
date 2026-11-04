use super::*;

impl PolicyModel {
    pub(crate) fn check_context(&self, context: &[f32]) -> Result<()> {
        let c = &self.config;
        ensure!(
            context.len() == c.context_len(),
            "context has {} values, expected {}",
            context.len(),
            c.context_len()
        );
        ensure!(context.iter().all(|v| v.is_finite()), "nonfinite context");
        if c.target_keep_dim < c.target_dim {
            ensure!(
                context[c.dataset_dim + c.target_keep_dim..]
                    .iter()
                    .all(|v| *v == 0.0),
                "nonzero dropped target dimensions"
            );
        }
        Ok(())
    }

    /// `Policy::input` then one context encoder: returns `[1, width]`.
    pub(crate) fn encode_context(&self, prefix: &str, context: &[f32]) -> Result<Tensor> {
        self.check_context(context)?;
        let c = &self.config;
        let mut input = Vec::with_capacity(c.dataset_dim + c.target_keep_dim);
        input.extend(
            context[..c.dataset_dim]
                .iter()
                .map(|v| (*v as f64 / c.dataset_rms) as f32),
        );
        input.extend(
            context[c.dataset_dim..c.dataset_dim + c.target_keep_dim]
                .iter()
                .map(|v| (*v as f64 / c.target_rms) as f32),
        );
        let x = Tensor::from_vec(input, (1, c.dataset_dim + c.target_keep_dim), &self.device)?;
        let h = linear(
            &x,
            self.g(&format!("{prefix}.0.weight"))?,
            self.g(&format!("{prefix}.0.bias"))?,
        )?
        .gelu_erf()?;
        let h = layer_norm(
            &h,
            self.g(&format!("{prefix}.2.weight"))?,
            self.g(&format!("{prefix}.2.bias"))?,
        )?;
        let h = linear(
            &h,
            self.g(&format!("{prefix}.4.weight"))?,
            self.g(&format!("{prefix}.4.bias"))?,
        )?
        .gelu_erf()?;
        layer_norm(
            &h,
            self.g(&format!("{prefix}.6.weight"))?,
            self.g(&format!("{prefix}.6.bias"))?,
        )
    }

    pub(crate) fn token_tensor(&self, tokens: &[Vec<usize>]) -> Result<(usize, usize, Tensor)> {
        let batch = tokens.len();
        ensure!(batch > 0, "empty token batch");
        let length = tokens[0].len();
        ensure!(
            length > 1 && length <= self.config.max_seq && tokens.iter().all(|t| t.len() == length),
            "invalid policy token batch shape"
        );
        let mut flat = Vec::with_capacity(batch * length);
        for row in tokens {
            for &t in row {
                ensure!(t < self.config.vocab_size, "invalid policy token {t}");
                flat.push(t as u32);
            }
        }
        Ok((
            batch,
            length,
            Tensor::from_vec(flat, batch * length, &self.device)?,
        ))
    }

    pub(crate) fn embed(
        &self,
        table: &str,
        ids: &Tensor,
        batch: usize,
        length: usize,
    ) -> Result<Tensor> {
        Ok(self
            .g(table)?
            .index_select(ids, 0)?
            .reshape((batch, length, self.config.width))?)
    }

    /// Decoder hidden states `[batch, length + 1, width]` before `out_norm`.
    pub(crate) fn decoder_states(&self, context: &[f32], tokens: &[Vec<usize>]) -> Result<Tensor> {
        let c = &self.config;
        let (batch, length, ids) = self.token_tensor(tokens)?;
        let z = self.encode_context("encoder", context)?;
        let ctx = linear(&z, self.g("ctx_proj.weight")?, self.g("ctx_proj.bias")?)?
            .reshape((1, 1, c.width))?
            .broadcast_as((batch, 1, c.width))?;
        let tok = self.embed("token_emb.weight", &ids, batch, length)?;
        let seq = length + 1;
        let pos = self
            .g("pos_emb.weight")?
            .narrow(0, 0, seq)?
            .reshape((1, seq, c.width))?;
        let mut h = Tensor::cat(&[&ctx.contiguous()?, &tok], 1)?.broadcast_add(&pos)?;
        // allowed(q, k) = k <= q and key k present (position 0 is the context, always present).
        let mut mask = vec![0f32; batch * seq * seq];
        for (b, row) in tokens.iter().enumerate() {
            for q in 0..seq {
                for k in 0..seq {
                    let present = k == 0 || row[k - 1] != PAD;
                    if k > q || !present {
                        mask[(b * seq + q) * seq + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask, (batch, 1, seq, seq), &self.device)?;
        for layer in 0..c.layers {
            h = self.decoder_layer(&h, layer, &mask)?;
        }
        Ok(h)
    }

    pub(crate) fn decoder_layer(&self, x: &Tensor, layer: usize, mask: &Tensor) -> Result<Tensor> {
        let p = format!("decoder.layers.{layer}");
        let g = |s: &str| self.g(&format!("{p}.{s}"));
        let a = layer_norm(x, g("norm1.weight")?, g("norm1.bias")?)?;
        let attn = self.self_attention(&a, &p, mask)?;
        let attn = linear(
            &attn,
            g("self_attn.out_proj.weight")?,
            g("self_attn.out_proj.bias")?,
        )?;
        let h = (x + attn)?;
        let f = layer_norm(&h, g("norm2.weight")?, g("norm2.bias")?)?;
        let f = linear(&f, g("linear1.weight")?, g("linear1.bias")?)?.gelu_erf()?;
        let f = linear(&f, g("linear2.weight")?, g("linear2.bias")?)?;
        Ok((h + f)?)
    }

    pub(crate) fn self_attention(&self, x: &Tensor, p: &str, mask: &Tensor) -> Result<Tensor> {
        let (b, seq, d) = x.dims3()?;
        let heads = self.config.heads;
        let hd = d / heads;
        let qkv = linear(
            x,
            self.g(&format!("{p}.self_attn.in_proj_weight"))?,
            self.g(&format!("{p}.self_attn.in_proj_bias"))?,
        )?;
        let split = |i: usize| -> Result<Tensor> {
            Ok(qkv
                .narrow(D::Minus1, i * d, d)?
                .reshape((b, seq, heads, hd))?
                .transpose(1, 2)?
                .contiguous()?)
        };
        let (q, k, v) = (split(0)?, split(1)?, split(2)?);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / (hd as f64).sqrt())?;
        let scores = scores.broadcast_add(mask)?.contiguous()?;
        let weights = candle_nn::ops::softmax_last_dim(&scores)?;
        Ok(weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, seq, d))?)
    }

    /// Full logits `[batch, length + 1, vocab_size]` for equal-length token rows (PAD = 0 is
    /// masked as a key). Row `i`, position `t` predicts token `t` of row `i`.
    pub fn logits(&self, context: &PolicyContext, tokens: &[Vec<usize>]) -> Result<Tensor> {
        let h = self.decoder_states(context.values(), tokens)?;
        self.project(&h)
    }

    pub(crate) fn project(&self, h: &Tensor) -> Result<Tensor> {
        let h = layer_norm(h, self.g("out_norm.weight")?, self.g("out_norm.bias")?)?;
        let logits = h.broadcast_matmul(&self.g("token_emb.weight")?.t()?)?;
        Ok(logits.narrow(D::Minus1, 0, self.config.vocab_size)?)
    }

    /// Next-token logits after the last position of each equal-length row: `[batch][vocab]`.
    pub fn next_token_logits(
        &self,
        context: &PolicyContext,
        tokens: &[Vec<usize>],
    ) -> Result<Vec<Vec<f32>>> {
        let h = self.decoder_states(context.values(), tokens)?;
        let last = h.dim(1)? - 1;
        let logits = self.project(&h.narrow(1, last, 1)?.contiguous()?)?;
        Ok(logits
            .squeeze(1)?
            .to_dtype(DType::F32)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?)
    }
}
