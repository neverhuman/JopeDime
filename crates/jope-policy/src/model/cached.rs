//! Optional inference cache; no checkpoint binding or sampling authority is created.
use super::{layer_norm, linear, PolicyModel};
use crate::grammar::PAD;
use anyhow::{ensure, Result};
use candle_core::{Device, Tensor, D};

struct LayerKv {
    key: Tensor,
    value: Tensor,
}

/// A one-token step shares the same weights across active rows. Candle 0.11's
/// broadcast_matmul materializes a contiguous [batch, in, out] weight copy for
/// 3D-by-2D inputs; flattening the rows avoids that copy and uses one GEMM.
/// Full-prefix decoding and cache prefill keep the original operation.
fn step_matmul(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (batch, sequence, input_width) = x.dims3()?;
    ensure!(
        sequence == 1,
        "cached linear expects exactly one new position"
    );
    let output_width = weight.dim(0)?;
    Ok(x.reshape((batch, input_width))?
        .matmul(&weight.t()?)?
        .reshape((batch, 1, output_width))?)
}

fn step_linear(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    Ok(step_matmul(x, weight)?.broadcast_add(bias)?)
}

fn step_project(model: &PolicyModel, h: &Tensor) -> Result<Tensor> {
    let h = layer_norm(h, model.g("out_norm.weight")?, model.g("out_norm.bias")?)?;
    Ok(step_matmul(&h, model.g("token_emb.weight")?)?.narrow(
        D::Minus1,
        0,
        model.config.vocab_size,
    )?)
}

/// A batch-local cache borrowing the exact loaded model. Row IDs refer to the
/// original prefill batch and may only be removed, never reordered or revived.
pub(crate) struct DecodeCache<'a> {
    model: &'a PolicyModel,
    layers: Vec<LayerKv>,
    active: Vec<usize>,
    present: Vec<Vec<bool>>,
    next_position: usize,
}

fn projected_qkv(
    model: &PolicyModel,
    x: &Tensor,
    layer: usize,
    step: bool,
) -> Result<(Tensor, Tensor, Tensor)> {
    let prefix = format!("decoder.layers.{layer}");
    let a = layer_norm(
        x,
        model.g(&format!("{prefix}.norm1.weight"))?,
        model.g(&format!("{prefix}.norm1.bias"))?,
    )?;
    let projection = if step { step_linear } else { linear };
    let qkv = projection(
        &a,
        model.g(&format!("{prefix}.self_attn.in_proj_weight"))?,
        model.g(&format!("{prefix}.self_attn.in_proj_bias"))?,
    )?;
    let (batch, sequence, width) = x.dims3()?;
    let heads = model.config.heads;
    let split = |i: usize| -> Result<Tensor> {
        Ok(qkv
            .narrow(D::Minus1, i * width, width)?
            .reshape((batch, sequence, heads, width / heads))?
            .transpose(1, 2)?
            .contiguous()?)
    };
    Ok((split(0)?, split(1)?, split(2)?))
}

impl<'a> DecodeCache<'a> {
    /// Prefill is the original full decoder, with pre-attention K/V retained.
    /// The small prefix's norm/QKV projections are repeated once to leave the
    /// existing full decoder implementation and operation order untouched.
    pub(crate) fn new(
        model: &'a PolicyModel,
        context: &[f32],
        tokens: &[Vec<usize>],
    ) -> Result<(Self, Vec<Vec<f32>>)> {
        let (batch, length, ids) = model.token_tensor(tokens)?;
        let width = model.config.width;
        let z = model.encode_context("encoder", context)?;
        let ctx = linear(&z, model.g("ctx_proj.weight")?, model.g("ctx_proj.bias")?)?
            .reshape((1, 1, width))?
            .broadcast_as((batch, 1, width))?;
        let tok = model.embed("token_emb.weight", &ids, batch, length)?;
        let sequence = length + 1;
        let pos = model
            .g("pos_emb.weight")?
            .narrow(0, 0, sequence)?
            .reshape((1, sequence, width))?;
        let mut h = Tensor::cat(&[&ctx.contiguous()?, &tok], 1)?.broadcast_add(&pos)?;
        let present = tokens
            .iter()
            .map(|row| {
                std::iter::once(true)
                    .chain(row.iter().map(|&t| t != PAD))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut mask = vec![0.0f32; batch * sequence * sequence];
        for (b, row) in present.iter().enumerate() {
            for q in 0..sequence {
                for (k, &allowed) in row.iter().enumerate() {
                    if k > q || !allowed {
                        mask[(b * sequence + q) * sequence + k] = f32::NEG_INFINITY;
                    }
                }
            }
        }
        let mask = Tensor::from_vec(mask, (batch, 1, sequence, sequence), &model.device)?;
        let mut layers = Vec::with_capacity(model.config.layers);
        for layer in 0..model.config.layers {
            let (_, key, value) = projected_qkv(model, &h, layer, false)?;
            layers.push(LayerKv { key, value });
            h = model.decoder_layer(&h, layer, &mask)?;
        }
        let logits = model
            .project(&h.narrow(1, length, 1)?.contiguous()?)?
            .squeeze(1)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        Ok((
            Self {
                model,
                layers,
                active: (0..batch).collect(),
                present,
                next_position: sequence,
            },
            logits,
        ))
    }

    /// Append one token for each surviving original row. All validation and
    /// tensor work complete before committing state, so refusal cannot corrupt
    /// an otherwise reusable cache. An empty subset permanently closes it.
    pub(crate) fn advance(&mut self, active: &[usize], tokens: &[usize]) -> Result<Vec<Vec<f32>>> {
        ensure!(
            active.len() == tokens.len(),
            "cache token/active row count differs"
        );
        ensure!(
            active.windows(2).all(|pair| pair[0] < pair[1]),
            "cache active rows must be strictly increasing"
        );
        let indices = active
            .iter()
            .map(|id| {
                self.active
                    .binary_search(id)
                    .map(|i| i as u32)
                    .map_err(|_| anyhow::anyhow!("cache row {id} was not active"))
            })
            .collect::<Result<Vec<_>>>()?;
        if active.is_empty() {
            self.active.clear();
            self.layers.clear();
            self.present.clear();
            return Ok(Vec::new());
        }
        ensure!(
            self.next_position <= self.model.config.max_seq,
            "cache sequence exceeds policy limit"
        );
        ensure!(
            tokens.iter().all(|&t| t < self.model.config.vocab_size),
            "cache token outside vocabulary"
        );
        let model = self.model;
        let batch = active.len();
        let width = model.config.width;
        let heads = model.config.heads;
        let head_width = width / heads;
        let gather = if active == self.active.as_slice() {
            None
        } else {
            Some(Tensor::from_vec(indices.clone(), batch, &model.device)?)
        };
        let ids = Tensor::from_vec(
            tokens.iter().map(|&t| t as u32).collect::<Vec<_>>(),
            batch,
            &model.device,
        )?;
        let pos = model
            .g("pos_emb.weight")?
            .narrow(0, self.next_position, 1)?
            .reshape((1, 1, width))?;
        let mut h = model
            .embed("token_emb.weight", &ids, batch, 1)?
            .broadcast_add(&pos)?;
        let present = indices
            .iter()
            .zip(tokens)
            .map(|(&row, &token)| {
                let mut keys = self.present[row as usize].clone();
                keys.push(token != PAD);
                keys
            })
            .collect::<Vec<_>>();
        // There are no future keys in a one-token step. Past PAD keys remain
        // hidden even after subsequent non-PAD tokens or active-row gathers.
        let mask = Tensor::from_vec(
            present
                .iter()
                .flatten()
                .map(|&p| if p { 0.0f32 } else { f32::NEG_INFINITY })
                .collect::<Vec<_>>(),
            (batch, 1, 1, self.next_position + 1),
            &model.device,
        )?;
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, old) in self.layers.iter().enumerate() {
            let prefix = format!("decoder.layers.{layer}");
            let g = |suffix: &str| model.g(&format!("{prefix}.{suffix}"));
            let (q, k, v) = projected_qkv(model, &h, layer, true)?;
            let (old_key, old_value) = match &gather {
                Some(gather) => (
                    old.key.index_select(gather, 0)?,
                    old.value.index_select(gather, 0)?,
                ),
                None => (old.key.clone(), old.value.clone()),
            };
            let key = Tensor::cat(&[&old_key, &k], 2)?;
            let value = Tensor::cat(&[&old_value, &v], 2)?;
            let scores =
                (q.matmul(&key.transpose(2, 3)?.contiguous()?)? / (head_width as f64).sqrt())?;
            let scores = scores.broadcast_add(&mask)?.contiguous()?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let attention = weights
                .matmul(&value)?
                .transpose(1, 2)?
                .contiguous()?
                .reshape((batch, 1, width))?;
            let attention = step_linear(
                &attention,
                g("self_attn.out_proj.weight")?,
                g("self_attn.out_proj.bias")?,
            )?;
            h = (h + attention)?;
            let f = layer_norm(&h, g("norm2.weight")?, g("norm2.bias")?)?;
            let f = step_linear(&f, g("linear1.weight")?, g("linear1.bias")?)?.gelu_erf()?;
            let f = step_linear(&f, g("linear2.weight")?, g("linear2.bias")?)?;
            h = (h + f)?;
            layers.push(LayerKv { key, value });
        }
        let logits = step_project(model, &h)?
            .squeeze(1)?
            .to_device(&Device::Cpu)?
            .to_vec2::<f32>()?;
        self.layers = layers;
        self.active = active.to_vec();
        self.present = present;
        self.next_position += 1;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PolicyConfig;

    fn model(seed: u64) -> PolicyModel {
        let mut config = PolicyConfig::regression(17);
        config.dataset_dim = 8;
        config.target_dim = 4;
        config.target_keep_dim = 4;
        config.width = 16;
        config.heads = 4;
        config.ffn = 32;
        config.layers = 2;
        config.max_seq = 32;
        PolicyModel::random_weights(config, seed).unwrap()
    }

    fn close(actual: &[Vec<f32>], expected: &[Vec<f32>]) {
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.iter().zip(expected) {
            assert_eq!(a.len(), b.len());
            for (&a, &b) in a.iter().zip(b) {
                assert!(
                    a.is_finite() && b.is_finite() && (a - b).abs() < 1e-4,
                    "{a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn cached_step_linear_and_tied_projection_match_full_for_active_batch_shapes() {
        let model = model(14);
        for batch in [1, 3, 17, 64] {
            let x = Tensor::from_vec(
                (0..batch * model.config.width)
                    .map(|i| ((i * 17 % 103) as f32 - 51.0) / 31.0)
                    .collect::<Vec<_>>(),
                (batch, 1, model.config.width),
                &Device::Cpu,
            )
            .unwrap();
            let weight = model
                .g("decoder.layers.0.self_attn.in_proj_weight")
                .unwrap();
            let bias = model.g("decoder.layers.0.self_attn.in_proj_bias").unwrap();
            let full = linear(&x, weight, bias).unwrap();
            let step = step_linear(&x, weight, bias).unwrap();
            assert_eq!(step.dims(), &[batch, 1, 3 * model.config.width]);
            close(
                &step.squeeze(1).unwrap().to_vec2::<f32>().unwrap(),
                &full.squeeze(1).unwrap().to_vec2::<f32>().unwrap(),
            );
            let full = model.project(&x).unwrap();
            let step = step_project(&model, &x).unwrap();
            assert_eq!(step.dims(), &[batch, 1, model.config.vocab_size]);
            close(
                &step.squeeze(1).unwrap().to_vec2::<f32>().unwrap(),
                &full.squeeze(1).unwrap().to_vec2::<f32>().unwrap(),
            );
        }
        let bad = Tensor::zeros(
            (1, 2, model.config.width),
            candle_core::DType::F32,
            &Device::Cpu,
        )
        .unwrap();
        assert!(step_matmul(&bad, model.g("token_emb.weight").unwrap()).is_err());
    }

    #[test]
    fn cached_decoder_matches_full_prefix_across_positions_and_pad_history() {
        let model = model(11);
        let context = vec![0.2f32; 12];
        for prefix_len in [2, 3, 7] {
            let mut rows = (0..3)
                .map(|row| {
                    (0..prefix_len)
                        .map(|j| (row + j + 1) % 17)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            rows[1][1] = PAD;
            let (mut cache, logits) = DecodeCache::new(&model, &context, &rows).unwrap();
            close(&logits, &model.next_token_logits(&context, &rows).unwrap());
            for step in prefix_len..model.config.max_seq {
                let tokens = (0..3)
                    .map(|row| {
                        if (row + step) % 5 == 0 {
                            PAD
                        } else {
                            (row + step) % 16 + 1
                        }
                    })
                    .collect::<Vec<_>>();
                for (row, &token) in rows.iter_mut().zip(&tokens) {
                    row.push(token);
                }
                close(
                    &cache.advance(&[0, 1, 2], &tokens).unwrap(),
                    &model.next_token_logits(&context, &rows).unwrap(),
                );
            }
            assert!(cache.advance(&[0, 1, 2], &[1, 2, 3]).is_err());
        }
        assert!(model.bound_identity().is_none());
    }

    #[test]
    fn cached_decoder_gathers_original_rows_and_refuses_revival_or_reorder() {
        let model = model(12);
        let context = vec![-0.1f32; 12];
        let mut rows = vec![vec![1, 2, 3], vec![1, 4, 5], vec![1, PAD, 7], vec![1, 8, 9]];
        let (mut cache, _) = DecodeCache::new(&model, &context, &rows).unwrap();
        for active in [&[0usize, 2, 3][..], &[2usize, 3][..], &[3usize][..]] {
            let tokens = active.iter().map(|id| id + 10).collect::<Vec<_>>();
            for (&id, &token) in active.iter().zip(&tokens) {
                rows[id].push(token);
            }
            let expected = active
                .iter()
                .map(|&id| rows[id].clone())
                .collect::<Vec<_>>();
            close(
                &cache.advance(active, &tokens).unwrap(),
                &model.next_token_logits(&context, &expected).unwrap(),
            );
        }
        assert!(cache.advance(&[0, 3], &[1, 1]).is_err());
        assert!(cache.advance(&[3, 3], &[1, 1]).is_err());
        assert!(cache.advance(&[3, 2], &[1, 1]).is_err());
        assert!(cache.advance(&[3], &[17]).is_err());
        rows[3].push(4);
        close(
            &cache.advance(&[3], &[4]).unwrap(),
            &model
                .next_token_logits(&context, &[rows[3].clone()])
                .unwrap(),
        );
        assert!(cache.advance(&[], &[]).unwrap().is_empty());
        assert!(cache.advance(&[3], &[4]).is_err());
    }

    #[test]
    fn cached_decoder_keeps_pad_keys_hidden_after_nonpad_steps() {
        let plain = model(13);
        let mut tensors = plain.tensors.clone();
        let table = &tensors["token_emb.weight"];
        let shape = table.dims().to_vec();
        let mut values = table.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (j, value) in values[..plain.config.width].iter_mut().enumerate() {
            *value += 3.0 + j as f32 * 0.2;
        }
        tensors.insert(
            "token_emb.weight".into(),
            Tensor::from_vec(values, shape, &Device::Cpu).unwrap(),
        );
        let shifted =
            PolicyModel::from_tensors(plain.config.clone(), tensors, Device::Cpu).unwrap();
        let context = vec![0.3f32; 12];
        let rows = vec![vec![1, PAD, 2], vec![1, 3, PAD]];
        let (mut a, _) = DecodeCache::new(&plain, &context, &rows).unwrap();
        let (mut b, _) = DecodeCache::new(&shifted, &context, &rows).unwrap();
        for tokens in [&[4usize, 5][..], &[PAD, PAD][..], &[6, 7][..]] {
            let left = a.advance(&[0, 1], tokens).unwrap();
            let right = b.advance(&[0, 1], tokens).unwrap();
            if tokens[0] != PAD {
                // The tied output table also changed PAD's own logit; compare
                // all other columns after a non-PAD query.
                close(
                    &left.iter().map(|row| row[1..].to_vec()).collect::<Vec<_>>(),
                    &right
                        .iter()
                        .map(|row| row[1..].to_vec())
                        .collect::<Vec<_>>(),
                );
            }
        }
    }
}
