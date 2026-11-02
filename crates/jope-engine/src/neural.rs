use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path};
use tch::{nn, nn::Module, nn::ModuleT, Device, Kind, Tensor};

/// Independent, default-off scoring candidate; does not mutate the actor or its optimizer.
pub mod critic;

mod objectives;
mod optimizer;
mod runtime;
pub use objectives::{
    preference_dpo, preference_pairs, ranking, ranking_pairwise, sequence_logprob, sequence_nll,
};
pub(crate) use optimizer::load_bound_model;
pub use optimizer::AdamW;
pub(super) use runtime::native_error;
pub use runtime::{
    configure_runtime, local_devices, loss_optimizer_parity, nccl_probe, nccl_unique_id, parity,
    training_probe, Bf16, Nccl,
};

pub const RANKING_BACKWARD_ID: &str = "dense_pairwise_v2";
/// Objective literals bound into `training::objective::identity` (W6-T06).
/// Uniform label smoothing over the grammar-legal set in `sequence_nll`.
pub const LABEL_SMOOTHING: f64 = 0.05;
/// Listwise target temperature: `softmax(percentile / T)`.
pub const LISTWISE_TEMPERATURE: f64 = 0.25;
/// Smallest comparison group that receives a listwise term.
pub const LISTWISE_MIN_MEMBERS: usize = 3;
/// A Bradley–Terry pair is up-weighted when either member's percentile reaches this.
pub const BOOST_THRESHOLD: f32 = 0.9;
pub const BOOST_FACTOR: f64 = 2.0;
/// Dropout inside the reward and ablation heads.
pub const REWARD_HEAD_DROPOUT: f64 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub dataset_dim: i64,
    pub target_dim: i64,
    pub target_keep_dim: i64,
    pub width: i64,
    pub layers: usize,
    pub heads: i64,
    pub ffn: i64,
    pub vocab_size: i64,
    pub embedding_rows: i64,
    pub max_seq: i64,
    pub dropout: f64,
    pub dataset_rms: f64,
    pub target_rms: f64,
}
impl Config {
    pub fn regression(vocab_size: i64) -> Self {
        Self {
            dataset_dim: 1536,
            target_dim: 512,
            target_keep_dim: 512,
            width: 512,
            layers: 6,
            heads: 4,
            ffn: 2048,
            vocab_size,
            embedding_rows: vocab_size + 256,
            max_seq: 128,
            dropout: 0.1,
            dataset_rms: 1.0,
            target_rms: 1.0,
        }
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.width > 0
                && self.heads > 0
                && self.width % self.heads == 0
                && self.layers > 0
                && self.vocab_size > 2
                && self.embedding_rows >= self.vocab_size
                && self.target_keep_dim > 0
                && self.target_keep_dim <= self.target_dim
                && self.dataset_rms.is_finite()
                && self.dataset_rms > 0.0
                && self.target_rms.is_finite()
                && self.target_rms > 0.0
                && self.dropout >= 0.0
                && self.dropout < 1.0,
            "invalid policy configuration"
        );
        Ok(())
    }
}
// tch defaults to ReLU Kaiming (gain sqrt(2)); torch.nn.Linear initializes
// with a=sqrt(5), equivalent to uniform +/-1/sqrt(fan_in). Explicitly match it.
fn linear(p: nn::Path, input: i64, output: i64, mut config: nn::LinearConfig) -> nn::Linear {
    let bound = 1.0 / (input as f64).sqrt();
    config.ws_init = nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    nn::linear(p, input, output, config)
}
fn mlp(p: nn::Path, input: i64, width: i64, dropout: Option<f64>) -> nn::SequentialT {
    let mut s = nn::seq_t()
        .add(linear(&p / "0", input, width, Default::default()))
        .add_fn(|x| x.gelu("none"));
    let final_name = if let Some(dropout) = dropout {
        s = s.add_fn_t(move |x, train| x.dropout(dropout, train));
        "3"
    } else {
        "2"
    };
    s.add(linear(&p / final_name, width, 1, Default::default()))
}
fn context_encoder(p: nn::Path, c: &Config) -> nn::SequentialT {
    let d = c.dropout;
    nn::seq_t()
        .add(linear(
            &p / "0",
            c.dataset_dim + c.target_keep_dim,
            c.width,
            Default::default(),
        ))
        .add_fn(|x| x.gelu("none"))
        .add(nn::layer_norm(&p / "2", vec![c.width], Default::default()))
        .add_fn_t(move |x, t| x.dropout(d, t))
        .add(linear(&p / "4", c.width, c.width, Default::default()))
        .add_fn(|x| x.gelu("none"))
        .add(nn::layer_norm(&p / "6", vec![c.width], Default::default()))
}
struct Layer {
    qkv_w: Tensor,
    qkv_b: Tensor,
    out: nn::Linear,
    linear1: nn::Linear,
    linear2: nn::Linear,
    norm1: nn::LayerNorm,
    norm2: nn::LayerNorm,
    heads: i64,
    dropout: f64,
}
impl Layer {
    fn new(p: nn::Path, c: &Config) -> Self {
        let a = &p / "self_attn";
        Self {
            qkv_w: a.var(
                "in_proj_weight",
                &[3 * c.width, c.width],
                nn::Init::Uniform {
                    lo: -(1.5 / c.width as f64).sqrt(),
                    up: (1.5 / c.width as f64).sqrt(),
                },
            ),
            qkv_b: a.zeros("in_proj_bias", &[3 * c.width]),
            out: linear(
                &a / "out_proj",
                c.width,
                c.width,
                nn::LinearConfig {
                    bs_init: Some(nn::Init::Const(0.0)),
                    ..Default::default()
                },
            ),
            linear1: linear(&p / "linear1", c.width, c.ffn, Default::default()),
            linear2: linear(&p / "linear2", c.ffn, c.width, Default::default()),
            norm1: nn::layer_norm(&p / "norm1", vec![c.width], Default::default()),
            norm2: nn::layer_norm(&p / "norm2", vec![c.width], Default::default()),
            heads: c.heads,
            dropout: c.dropout,
        }
    }
    fn forward(&self, x: &Tensor, allowed: &Tensor, train: bool) -> Tensor {
        let (b, l, d) = x.size3().unwrap();
        let h = self.norm1.forward(x);
        let qkv = h.linear(&self.qkv_w, Some(&self.qkv_b)).chunk(3, -1);
        let split = |t: &Tensor| t.view([b, l, self.heads, d / self.heads]).transpose(1, 2);
        let attn = Tensor::scaled_dot_product_attention(
            &split(&qkv[0]),
            &split(&qkv[1]),
            &split(&qkv[2]),
            Some(allowed),
            if train { self.dropout } else { 0.0 },
            false,
            None,
            false,
        )
        .transpose(1, 2)
        .contiguous()
        .view([b, l, d]);
        let h = x + self.out.forward(&attn).dropout(self.dropout, train);
        let ff = self.linear2.forward(
            &self
                .linear1
                .forward(&self.norm2.forward(&h))
                .gelu("none")
                .dropout(self.dropout, train),
        );
        h + ff.dropout(self.dropout, train)
    }
}

/// Isolated experimental program encoder. Production Policy does not construct
/// this module: a separately qualified reranker must opt in with a new artifact.
/// The existing vocabulary is retained; policy parameters are never shared.
pub struct OrderAwareRewardEncoder {
    token: nn::Embedding,
    position: nn::Embedding,
    layers: Vec<Layer>,
    norm: nn::LayerNorm,
    vocabulary_size: i64,
    max_seq: i64,
}

impl OrderAwareRewardEncoder {
    pub const VERSION: &'static str = "parallel.reward_program_attention_2x256x4.v1";

    pub fn new(p: nn::Path, vocabulary_size: i64, max_seq: i64) -> Result<Self> {
        ensure!(
            vocabulary_size > 2 && max_seq > 1,
            "invalid experimental reward encoder shape"
        );
        let mut config = Config::regression(vocabulary_size);
        config.width = 256;
        config.heads = 4;
        config.ffn = 1024;
        config.layers = 2;
        config.max_seq = max_seq;
        let result = Self {
            token: nn::embedding(
                &p / "token",
                vocabulary_size,
                256,
                nn::EmbeddingConfig {
                    padding_idx: 0,
                    ..Default::default()
                },
            ),
            position: nn::embedding(&p / "position", max_seq, 256, Default::default()),
            layers: (0..2)
                .map(|i| Layer::new(&p / "layers" / i.to_string(), &config))
                .collect(),
            norm: nn::layer_norm(&p / "norm", vec![256], Default::default()),
            vocabulary_size,
            max_seq,
        };
        tch::no_grad(|| {
            let _ = result.token.ws.get(0).zero_();
        });
        Ok(result)
    }

    pub fn forward(&self, tokens: &Tensor, present: &Tensor, train: bool) -> Result<Tensor> {
        let (batch, length) = tokens.size2()?;
        ensure!(
            batch > 0
                && length > 0
                && length <= self.max_seq
                && tokens.kind() == Kind::Int64
                && present.kind() == Kind::Bool
                && present.size() == tokens.size()
                && present.device() == tokens.device(),
            "experimental reward token/mask shape changed"
        );
        ensure!(
            tokens.min().int64_value(&[]) >= 0
                && tokens.max().int64_value(&[]) < self.vocabulary_size,
            "experimental reward token outside vocabulary"
        );
        ensure!(
            present.any_dim(1, false).all().int64_value(&[]) != 0,
            "experimental reward program is empty"
        );
        let position = Tensor::arange(length, (Kind::Int64, tokens.device())).unsqueeze(0);
        let mut encoded = self.token.forward(tokens) + self.position.forward(&position);
        // Full program attention; every query sees only present keys.
        let allowed = present.unsqueeze(1).unsqueeze(1);
        for layer in &self.layers {
            encoded = layer.forward(&encoded, &allowed, train);
        }
        let mask = present.to_kind(Kind::Float).unsqueeze(-1);
        Ok(
            (self.norm.forward(&encoded) * &mask).sum_dim_intlist(&[1i64][..], false, Kind::Float)
                / mask.sum_dim_intlist(&[1i64][..], false, Kind::Float),
        )
    }
}

mod policy;
pub use policy::Policy;

#[cfg(test)]
mod tests;
