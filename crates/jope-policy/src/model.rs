//! Candle forward pass of jope-dime `neural::Policy` (`neural.rs:283-531`), inference only.
//!
//! Tensor names and shapes are exactly the `nn::VarStore` names of the reference model (136
//! tensors for the step-60000 configuration, including the seven scalar `loss_log_vars`, the
//! `runtime_head`, `stability_head` and `reward_head_ablation` that inference does not use);
//! loading refuses any missing, extra, reshaped or non-float32 tensor.
//!
//! Architecture (all float32, eval mode, dropout disabled):
//! - context encoder (`encoder.*` for decoding, `reward_encoder.*` for ranking):
//!   Linear(2048→w) → GELU(erf) → LayerNorm → Linear(w→w) → GELU → LayerNorm;
//! - decoder: `[ctx_proj(z), token_emb(tokens)] + pos_emb`, pre-norm layers with packed QKV
//!   attention under a causal mask that also hides PAD keys, then `out_norm` and the tied
//!   `token_emb` projection narrowed to `vocab_size`;
//! - reward: masked mean of `reward_token_emb` → `reward_norm` = g;
//!   `phi = [g, g·tanh(z_gamma.0(rz)), |g − z_proj(rz)|]`; four `reward_heads` MLPs;
//!   runtime = softplus(`runtime_head`([rz, g])).
//!
//! No numeric parity with the tch implementation is established here (see the module docs).
use crate::context::PolicyContext;
use anyhow::{bail, ensure, Context, Result};
use candle_core::{DType, Device, Tensor, D};
use std::collections::BTreeMap;
use std::path::Path;

use super::config::{CheckpointBinding, PolicyConfig, TensorEntry};
use super::grammar::PAD;
use super::sha256_hex;

/// Separate, default-off structural critic candidate. It cannot alter policy actor weights.
pub mod structured_critic;

const LAYER_NORM_EPS: f64 = 1e-5;
pub(crate) mod cached;
const LOSS_LOG_VARS: [&str; 7] = [
    "seq_nll",
    "reward_bt",
    "reward_listwise",
    "runtime_huber",
    "stability_bce",
    "ablation_bt",
    "routing_ce",
];

/// Every tensor name and shape the reference `Policy::new` registers, sorted by name.
pub fn expected_inventory(c: &PolicyConfig) -> Vec<(String, Vec<usize>)> {
    let w = c.width;
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    let dense = |out: &mut Vec<(String, Vec<usize>)>, name: &str, i: usize, o: usize| {
        out.push((format!("{name}.weight"), vec![o, i]));
        out.push((format!("{name}.bias"), vec![o]));
    };
    let norm = |out: &mut Vec<(String, Vec<usize>)>, name: &str| {
        out.push((format!("{name}.weight"), vec![w]));
        out.push((format!("{name}.bias"), vec![w]));
    };
    for encoder in ["encoder", "reward_encoder"] {
        dense(
            &mut out,
            &format!("{encoder}.0"),
            c.dataset_dim + c.target_keep_dim,
            w,
        );
        norm(&mut out, &format!("{encoder}.2"));
        dense(&mut out, &format!("{encoder}.4"), w, w);
        norm(&mut out, &format!("{encoder}.6"));
    }
    dense(&mut out, "ctx_proj", w, w);
    out.push(("token_emb.weight".into(), vec![c.embedding_rows, w]));
    out.push(("pos_emb.weight".into(), vec![c.max_seq + 1, w]));
    for layer in 0..c.layers {
        let p = format!("decoder.layers.{layer}");
        out.push((format!("{p}.self_attn.in_proj_weight"), vec![3 * w, w]));
        out.push((format!("{p}.self_attn.in_proj_bias"), vec![3 * w]));
        dense(&mut out, &format!("{p}.self_attn.out_proj"), w, w);
        dense(&mut out, &format!("{p}.linear1"), w, c.ffn);
        dense(&mut out, &format!("{p}.linear2"), c.ffn, w);
        norm(&mut out, &format!("{p}.norm1"));
        norm(&mut out, &format!("{p}.norm2"));
    }
    norm(&mut out, "out_norm");
    out.push(("reward_token_emb.weight".into(), vec![c.embedding_rows, w]));
    norm(&mut out, "reward_norm");
    dense(&mut out, "z_gamma.0", w, w);
    dense(&mut out, "z_proj", w, w);
    for head in 0..4 {
        dense(&mut out, &format!("reward_heads.{head}.0"), 3 * w, w);
        dense(&mut out, &format!("reward_heads.{head}.3"), w, 1);
    }
    dense(&mut out, "runtime_head.0", 2 * w, w);
    dense(&mut out, "runtime_head.2", w, 1);
    dense(&mut out, "stability_head.0", 3 * w, w);
    dense(&mut out, "stability_head.2", w, 1);
    dense(&mut out, "reward_head_ablation.0", w, w);
    dense(&mut out, "reward_head_ablation.3", w, 1);
    for name in LOSS_LOG_VARS {
        out.push((format!("loss_log_vars.{name}"), vec![]));
    }
    out.sort();
    out
}

fn linear(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    Ok(x.broadcast_matmul(&w.t()?)?.broadcast_add(b)?)
}

fn layer_norm(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let norm = candle_nn::LayerNorm::new(w.clone(), b.clone(), LAYER_NORM_EPS);
    Ok(candle_nn::Module::forward(&norm, x)?)
}

fn softplus(x: f32) -> f32 {
    // torch.nn.functional.softplus(beta=1, threshold=20).
    if x > 20.0 {
        x
    } else {
        x.exp().ln_1p()
    }
}

/// The checkpoint identity a policy was proven to be when it was loaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundIdentity {
    state_sha256: String,
    parameter_sha256: String,
    policy_safetensors_sha256: String,
    config_sha256: String,
}

impl BoundIdentity {
    /// `config_sha256` claimed by the export receipt the policy was loaded with.
    pub fn config_sha256(&self) -> &str {
        &self.config_sha256
    }

    pub fn state_sha256(&self) -> &str {
        &self.state_sha256
    }

    /// Parameter hash recomputed from the loaded tensors (not copied from the receipt).
    pub fn parameter_sha256(&self) -> &str {
        &self.parameter_sha256
    }

    pub fn policy_safetensors_sha256(&self) -> &str {
        &self.policy_safetensors_sha256
    }
}

/// A loaded policy. Construction always checks the full tensor inventory. Only
/// [`PolicyModel::load_bound`] records a [`BoundIdentity`]; every other constructor yields an
/// unbound model, which `seeds::propose` refuses.
pub struct PolicyModel {
    config: PolicyConfig,
    tensors: BTreeMap<String, Tensor>,
    device: Device,
    bound: Option<BoundIdentity>,
}

impl PolicyModel {
    /// Build from named tensors; refuses any inventory difference or a non-float32 tensor.
    pub fn from_tensors(
        config: PolicyConfig,
        tensors: impl IntoIterator<Item = (String, Tensor)>,
        device: Device,
    ) -> Result<Self> {
        config.validate()?;
        let tensors = tensors.into_iter().collect::<BTreeMap<_, _>>();
        let expected = expected_inventory(&config);
        let actual_names = tensors.keys().cloned().collect::<Vec<_>>();
        let expected_names = expected.iter().map(|e| e.0.clone()).collect::<Vec<_>>();
        if actual_names != expected_names {
            let missing = expected_names
                .iter()
                .filter(|n| !tensors.contains_key(*n))
                .take(4)
                .collect::<Vec<_>>();
            let extra = actual_names
                .iter()
                .filter(|n| !expected_names.contains(n))
                .take(4)
                .collect::<Vec<_>>();
            bail!("policy tensor set differs: missing {missing:?}, unexpected {extra:?}");
        }
        for (name, shape) in &expected {
            let t = &tensors[name];
            ensure!(
                t.dims() == shape.as_slice(),
                "policy tensor {name} has shape {:?}, expected {shape:?}",
                t.dims()
            );
            ensure!(
                t.dtype() == DType::F32,
                "policy tensor {name} is {:?}, expected F32",
                t.dtype()
            );
        }
        let tensors = tensors
            .into_iter()
            .map(|(k, t)| Ok((k, t.to_device(&device)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        Ok(Self {
            config,
            tensors,
            device,
            bound: None,
        })
    }

    /// Load a safetensors file with the given configuration (no identity check).
    pub fn load_safetensors(path: &Path, config: PolicyConfig, device: Device) -> Result<Self> {
        let tensors = candle_core::safetensors::load(path, &Device::Cpu)
            .with_context(|| format!("loading policy safetensors {}", path.display()))?;
        Self::from_tensors(config, tensors, device)
    }

    /// Load the exported step-60000 policy and prove it is the receipt's checkpoint: file
    /// sha256, per-tensor name/shape/dtype/sha256, and the recomputed jope-dime parameter hash.
    /// The returned model carries that proven [`BoundIdentity`].
    pub fn load_bound(path: &Path, binding: &CheckpointBinding, device: Device) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("reading policy {}", path.display()))?;
        let file_sha256 = sha256_hex(&bytes);
        ensure!(
            file_sha256 == binding.policy_safetensors_sha256(),
            "policy safetensors differs from its export receipt"
        );
        let tensors = candle_core::safetensors::load_buffer(&bytes, &Device::Cpu)
            .context("parsing policy safetensors")?;
        drop(bytes);
        let model = Self::from_tensors(binding.config().clone(), tensors, Device::Cpu)?;
        let inventory = model.tensor_inventory()?;
        let mut expected = binding.tensors().to_vec();
        expected.sort_by(|a, b| a.name.cmp(&b.name));
        ensure!(
            inventory == expected,
            "policy tensors differ from the export receipt inventory"
        );
        let parameter_sha256 = model.parameter_sha256()?;
        ensure!(
            parameter_sha256 == binding.parameter_sha256(),
            "recomputed parameter hash differs from the checkpoint state"
        );
        let mut model = if matches!(device, Device::Cpu) {
            model
        } else {
            Self::from_tensors(binding.config().clone(), model.tensors, device)?
        };
        model.bound = Some(BoundIdentity {
            state_sha256: binding.state_sha256().to_string(),
            parameter_sha256,
            policy_safetensors_sha256: file_sha256,
            config_sha256: binding.config_sha256().to_string(),
        });
        Ok(model)
    }

    /// The checkpoint identity proven by [`PolicyModel::load_bound`]; `None` for any other model.
    pub fn bound_identity(&self) -> Option<&BoundIdentity> {
        self.bound.as_ref()
    }

    /// Deterministic untrained weights for tests (`testing`). Never a checkpoint and
    /// never bound.
    #[cfg(any(test, feature = "testing"))]
    pub fn random_weights(config: PolicyConfig, seed: u64) -> Result<Self> {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut tensors = Vec::new();
        for (name, shape) in expected_inventory(&config) {
            let numel = shape.iter().product::<usize>();
            let fan_in = shape.get(1).copied().unwrap_or(1).max(1) as f32;
            let bound = if name.contains("_emb.") {
                1.0
            } else {
                1.0 / fan_in.sqrt()
            };
            let values = (0..numel)
                .map(|_| {
                    let r = rng.gen_range(-bound..bound);
                    if name.contains("norm") && name.ends_with(".weight") {
                        1.0 + 0.1 * r
                    } else {
                        r
                    }
                })
                .collect::<Vec<f32>>();
            let t = if shape.is_empty() {
                Tensor::new(values[0], &Device::Cpu)?
            } else {
                Tensor::from_vec(values, shape, &Device::Cpu)?
            };
            tensors.push((name, t));
        }
        Self::from_tensors(config, tensors, Device::Cpu)
    }

    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn g(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("policy tensor missing: {name}"))
    }

    fn le_bytes(t: &Tensor) -> Result<Vec<u8>> {
        let values = t.flatten_all()?.to_device(&Device::Cpu)?.to_vec1::<f32>()?;
        Ok(values.iter().flat_map(|v| v.to_le_bytes()).collect())
    }

    /// Receipt-shaped inventory of the loaded tensors, sorted by name.
    pub fn tensor_inventory(&self) -> Result<Vec<TensorEntry>> {
        self.tensors
            .iter()
            .map(|(name, t)| {
                let bytes = Self::le_bytes(t)?;
                Ok(TensorEntry {
                    name: name.clone(),
                    dtype: "F32".into(),
                    shape: t.dims().to_vec(),
                    numel: t.elem_count(),
                    bytes: bytes.len(),
                    sha256: sha256_hex(&bytes),
                })
            })
            .collect()
    }

    /// jope-dime parameter hash: sorted names, each followed by its little-endian f32 values.
    pub fn parameter_sha256(&self) -> Result<String> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        for (name, t) in &self.tensors {
            hasher.update(name.as_bytes());
            hasher.update(Self::le_bytes(t)?);
        }
        Ok(hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect())
    }

    fn check_context(&self, context: &[f32]) -> Result<()> {
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
    fn encode_context(&self, prefix: &str, context: &[f32]) -> Result<Tensor> {
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

    fn token_tensor(&self, tokens: &[Vec<usize>]) -> Result<(usize, usize, Tensor)> {
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

    fn embed(&self, table: &str, ids: &Tensor, batch: usize, length: usize) -> Result<Tensor> {
        Ok(self
            .g(table)?
            .index_select(ids, 0)?
            .reshape((batch, length, self.config.width))?)
    }

    /// Decoder hidden states `[batch, length + 1, width]` before `out_norm`.
    fn decoder_states(&self, context: &[f32], tokens: &[Vec<usize>]) -> Result<Tensor> {
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

    fn decoder_layer(&self, x: &Tensor, layer: usize, mask: &Tensor) -> Result<Tensor> {
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

    fn self_attention(&self, x: &Tensor, p: &str, mask: &Tensor) -> Result<Tensor> {
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

    fn project(&self, h: &Tensor) -> Result<Tensor> {
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

    /// `Policy::reward_runtime`: four reward-head outputs and the runtime estimate per row.
    /// Rows are right-padded with PAD; PAD positions are excluded from the program mean.
    pub fn reward_runtime(
        &self,
        context: &PolicyContext,
        tokens: &[Vec<usize>],
    ) -> Result<(Vec<[f32; 4]>, Vec<f32>)> {
        let c = &self.config;
        let (batch, length, ids) = self.token_tensor(tokens)?;
        let rz = self
            .encode_context("reward_encoder", context.values())?
            .broadcast_as((batch, c.width))?
            .contiguous()?;
        let present = tokens
            .iter()
            .flat_map(|row| row.iter().map(|&t| if t == PAD { 0f32 } else { 1f32 }))
            .collect::<Vec<_>>();
        let mask = Tensor::from_vec(present, (batch, length, 1), &self.device)?;
        let emb = self.embed("reward_token_emb.weight", &ids, batch, length)?;
        let summed = emb.broadcast_mul(&mask)?.sum(1)?;
        let count = mask.sum(1)?.clamp(1f32, f32::MAX)?;
        let g = layer_norm(
            &summed.broadcast_div(&count)?,
            self.g("reward_norm.weight")?,
            self.g("reward_norm.bias")?,
        )?;
        let gamma = linear(&rz, self.g("z_gamma.0.weight")?, self.g("z_gamma.0.bias")?)?.tanh()?;
        let projected = linear(&rz, self.g("z_proj.weight")?, self.g("z_proj.bias")?)?;
        let phi = Tensor::cat(&[&g, &(&g * &gamma)?, &(&g - &projected)?.abs()?], 1)?;
        let mut heads = Vec::with_capacity(4);
        for i in 0..4 {
            let h = linear(
                &phi,
                self.g(&format!("reward_heads.{i}.0.weight"))?,
                self.g(&format!("reward_heads.{i}.0.bias"))?,
            )?
            .gelu_erf()?;
            let h = linear(
                &h,
                self.g(&format!("reward_heads.{i}.3.weight"))?,
                self.g(&format!("reward_heads.{i}.3.bias"))?,
            )?;
            heads.push(h.squeeze(1)?.to_device(&Device::Cpu)?.to_vec1::<f32>()?);
        }
        let r = linear(
            &Tensor::cat(&[&rz, &g], 1)?,
            self.g("runtime_head.0.weight")?,
            self.g("runtime_head.0.bias")?,
        )?
        .gelu_erf()?;
        let r = linear(
            &r,
            self.g("runtime_head.2.weight")?,
            self.g("runtime_head.2.bias")?,
        )?;
        let runtime = r
            .squeeze(1)?
            .to_device(&Device::Cpu)?
            .to_vec1::<f32>()?
            .into_iter()
            .map(softplus)
            .collect();
        let rewards = (0..batch)
            .map(|row| [heads[0][row], heads[1][row], heads[2][row], heads[3][row]])
            .collect();
        Ok((rewards, runtime))
    }
}

#[cfg(any(test, feature = "inference-cuda"))]
impl PolicyModel {
    /// Qualification only: fresh batch-local cache versus full-prefix logits. Both paths need
    /// at least two tokens, so the cache is primed with the first two and every later token is
    /// one compared cached step.
    pub fn cache_max_error(&self, context: &PolicyContext, tokens: &[usize]) -> Result<f32> {
        ensure!(
            tokens.len() >= 3,
            "qualification needs a two-token prefix and at least one cached step"
        );
        let mut rows = vec![tokens[..2].to_vec()];
        let (mut cache, first) = cached::DecodeCache::new(self, context, &rows)?;
        let mut maximum = 0f32;
        let mut compare = |a: &[Vec<f32>], b: &[Vec<f32>]| -> Result<()> {
            ensure!(a.len() == b.len(), "cache batch changed");
            for (a, b) in a.iter().zip(b) {
                ensure!(a.len() == b.len(), "cache shape changed");
                for (&x, &y) in a.iter().zip(b) {
                    ensure!(x.is_finite() && y.is_finite(), "nonfinite cache output");
                    maximum = maximum.max((x - y).abs());
                }
            }
            Ok(())
        };
        compare(&first, &self.next_token_logits(context, &rows)?)?;
        for &token in &tokens[2..] {
            rows[0].push(token);
            compare(
                &cache.advance(&[0], &[token])?,
                &self.next_token_logits(context, &rows)?,
            )?;
        }
        Ok(maximum)
    }
}

#[cfg(test)]
mod identity_tests {
    use super::*;

    /// A bound identity labels output with the accepted state only for the released weights
    /// file: the accepted state, parameters and configuration around any other
    /// `policy.safetensors` (for example a re-serialized copy of the same tensors) are refused.
    #[test]
    fn accepted_policy_requires_the_released_weights_file() {
        let mut model = PolicyModel::random_weights(PolicyConfig::regression(401), 3).unwrap();
        let bind = |file: &str| BoundIdentity {
            state_sha256: crate::ACCEPTED_STATE_SHA256.into(),
            parameter_sha256: crate::ACCEPTED_PARAMETER_SHA256.into(),
            policy_safetensors_sha256: file.into(),
            config_sha256: crate::ACCEPTED_CONFIG_SHA256.into(),
        };
        model.bound = Some(bind(
            "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e",
        ));
        crate::require_accepted_policy(&model).unwrap();
        model.bound = Some(bind(&sha256_hex(b"re-serialized accepted tensors")));
        let err = crate::require_accepted_policy(&model).unwrap_err();
        assert!(format!("{err:#}").contains("policy.safetensors"), "{err:#}");
    }
}
