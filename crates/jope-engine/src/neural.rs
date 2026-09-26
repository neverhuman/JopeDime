use crate::manifest::*;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    ffi::{c_void, CStr},
    fs,
    path::Path,
    time::{Duration, Instant},
};
use tch::{nn, nn::Module, nn::ModuleT, Device, Kind, Tensor};

pub const RANKING_BACKWARD_ID: &str = "dense_pairwise_v2";

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

pub struct Policy {
    pub config: Config,
    encoder: nn::SequentialT,
    reward_encoder: nn::SequentialT,
    ctx: nn::Linear,
    token: nn::Embedding,
    position: nn::Embedding,
    layers: Vec<Layer>,
    out_norm: nn::LayerNorm,
    reward_token: nn::Embedding,
    reward_norm: nn::LayerNorm,
    gamma: nn::Linear,
    projection: nn::Linear,
    reward: Vec<nn::SequentialT>,
    runtime: nn::SequentialT,
    stability: nn::SequentialT,
    ablation: nn::SequentialT,
    pub loss_log_vars: BTreeMap<String, Tensor>,
}
impl Policy {
    pub fn new(p: nn::Path, c: Config) -> Result<Self> {
        c.validate()?;
        let embedding = nn::EmbeddingConfig {
            padding_idx: 0,
            ..Default::default()
        };
        let mut loss_log_vars = BTreeMap::new();
        for name in [
            "seq_nll",
            "reward_bt",
            "reward_listwise",
            "runtime_huber",
            "stability_bce",
            "ablation_bt",
            "routing_ce",
        ] {
            loss_log_vars.insert(name.into(), (&p / "loss_log_vars").zeros(name, &[]));
        }
        let model = Self {
            encoder: context_encoder(&p / "encoder", &c),
            reward_encoder: context_encoder(&p / "reward_encoder", &c),
            ctx: linear(&p / "ctx_proj", c.width, c.width, Default::default()),
            token: nn::embedding(&p / "token_emb", c.embedding_rows, c.width, embedding),
            position: nn::embedding(&p / "pos_emb", c.max_seq + 1, c.width, Default::default()),
            layers: (0..c.layers)
                .map(|i| Layer::new(&p / "decoder" / "layers" / i.to_string(), &c))
                .collect(),
            out_norm: nn::layer_norm(&p / "out_norm", vec![c.width], Default::default()),
            reward_token: nn::embedding(
                &p / "reward_token_emb",
                c.embedding_rows,
                c.width,
                embedding,
            ),
            reward_norm: nn::layer_norm(&p / "reward_norm", vec![c.width], Default::default()),
            gamma: linear(&p / "z_gamma" / "0", c.width, c.width, Default::default()),
            projection: linear(&p / "z_proj", c.width, c.width, Default::default()),
            reward: (0..4)
                .map(|i| {
                    mlp(
                        &p / "reward_heads" / i.to_string(),
                        3 * c.width,
                        c.width,
                        Some(0.05),
                    )
                })
                .collect(),
            runtime: mlp(&p / "runtime_head", 2 * c.width, c.width, None),
            stability: mlp(&p / "stability_head", 3 * c.width, c.width, None),
            ablation: mlp(&p / "reward_head_ablation", c.width, c.width, Some(0.05)),
            loss_log_vars,
            config: c,
        };
        tch::no_grad(|| {
            // torch.nn.Embedding zeroes its padding row; tch leaves it random.
            let _ = model.token.ws.get(0).zero_();
            let _ = model.reward_token.ws.get(0).zero_();
            // torch.nn.TransformerEncoder deep-copies its initial layer.
            let names = [
                "self_attn.in_proj_weight",
                "self_attn.in_proj_bias",
                "self_attn.out_proj.weight",
                "self_attn.out_proj.bias",
                "linear1.weight",
                "linear1.bias",
                "linear2.weight",
                "linear2.bias",
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
            ];
            let parameter = |layer: usize, name: &str| {
                let mut q = &p / "decoder" / "layers" / layer.to_string();
                let parts = name.split('.').collect::<Vec<_>>();
                for part in &parts[..parts.len() - 1] {
                    q = &q / *part;
                }
                q.get(parts[parts.len() - 1]).unwrap()
            };
            for i in 1..model.config.layers {
                for name in names {
                    parameter(i, name).copy_(&parameter(0, name));
                }
            }
        });
        Ok(model)
    }
    fn input(&self, dataset: &Tensor, target: &Tensor) -> Result<Tensor> {
        let c = &self.config;
        ensure!(
            dataset.size2()?.1 == c.dataset_dim
                && target.size2()?.1 == c.target_dim
                && dataset.size()[0] == target.size()[0],
            "context schema mismatch"
        );
        ensure!(
            dataset.isfinite().all().int64_value(&[]) != 0
                && target.isfinite().all().int64_value(&[]) != 0,
            "nonfinite context"
        );
        if c.target_keep_dim < c.target_dim {
            ensure!(
                target
                    .narrow(1, c.target_keep_dim, c.target_dim - c.target_keep_dim)
                    .abs()
                    .max()
                    .double_value(&[])
                    == 0.0,
                "nonzero dropped target dimensions"
            );
        }
        Ok(Tensor::cat(
            &[
                dataset.to_kind(Kind::Float) / c.dataset_rms,
                target.narrow(1, 0, c.target_keep_dim).to_kind(Kind::Float) / c.target_rms,
            ],
            1,
        ))
    }
    pub fn forward(
        &self,
        dataset: &Tensor,
        target: &Tensor,
        tokens: &Tensor,
        pad: &Tensor,
        train: bool,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let c = &self.config;
        let (b, t) = tokens.size2()?;
        ensure!(
            t > 1
                && t <= c.max_seq
                && tokens.min().int64_value(&[]) >= 0
                && tokens.max().int64_value(&[]) < c.vocab_size,
            "invalid policy tokens"
        );
        ensure!(pad.size() == tokens.size(), "padding schema mismatch");
        let input = self.input(dataset, target)?;
        let z = self.encoder.forward_t(&input, train);
        let rz = self.reward_encoder.forward_t(&input, train);
        let mut h = Tensor::cat(
            &[
                self.ctx.forward(&z).unsqueeze(1),
                self.token.forward(tokens),
            ],
            1,
        ) + self
            .position
            .forward(&Tensor::arange(t + 1, (Kind::Int64, tokens.device())))
            .unsqueeze(0);
        let present = Tensor::cat(
            &[
                Tensor::ones([b, 1], (Kind::Bool, tokens.device())),
                pad.shallow_clone(),
            ],
            1,
        );
        let allowed = Tensor::ones([t + 1, t + 1], (Kind::Bool, tokens.device()))
            .tril(0)
            .unsqueeze(0)
            .unsqueeze(0)
            .logical_and(&present.unsqueeze(1).unsqueeze(1));
        for layer in &self.layers {
            h = layer.forward(&h, &allowed, train);
        }
        let logits = self
            .out_norm
            .forward(&h)
            .matmul(&self.token.ws.transpose(0, 1))
            .narrow(-1, 0, c.vocab_size);
        let (reward, runtime, phi, g) = self.reward_runtime_encoded(&rz, tokens, pad, train);
        let stability = self.stability.forward_t(&phi, train).squeeze_dim(-1);
        let ablation = self.ablation.forward_t(&g, train).squeeze_dim(-1);
        Ok((logits, reward, runtime, stability, ablation))
    }

    /// Frozen inference for candidate ranking. This intentionally never runs
    /// the policy context encoder, decoder, stability head, or ablation head.
    /// Parameter names and reward arithmetic are shared with full forward.
    pub fn reward_runtime(
        &self,
        dataset: &Tensor,
        target: &Tensor,
        tokens: &Tensor,
        pad: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let (batch, length) = tokens.size2()?;
        ensure!(
            length > 1
                && length <= self.config.max_seq
                && tokens.min().int64_value(&[]) >= 0
                && tokens.max().int64_value(&[]) < self.config.vocab_size,
            "invalid policy tokens"
        );
        ensure!(pad.size() == tokens.size(), "padding schema mismatch");
        ensure!(dataset.size()[0] == batch, "context/token batch mismatch");
        let input = self.input(dataset, target)?;
        let rz = self.reward_encoder.forward_t(&input, false);
        let (reward, runtime, _, _) = self.reward_runtime_encoded(&rz, tokens, pad, false);
        Ok((reward, runtime))
    }

    fn reward_runtime_encoded(
        &self,
        rz: &Tensor,
        tokens: &Tensor,
        pad: &Tensor,
        train: bool,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let mask = pad.to_kind(Kind::Float).unsqueeze(-1);
        let g = self.reward_norm.forward(
            &((self.reward_token.forward(tokens) * &mask).sum_dim_intlist(
                &[1i64][..],
                false,
                Kind::Float,
            ) / mask
                .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                .clamp_min(1.0)),
        );
        let phi = Tensor::cat(
            &[
                g.shallow_clone(),
                &g * self.gamma.forward(&rz).tanh(),
                (&g - self.projection.forward(&rz)).abs(),
            ],
            1,
        );
        let reward = Tensor::stack(
            &self
                .reward
                .iter()
                .map(|h| h.forward_t(&phi, train).squeeze_dim(-1))
                .collect::<Vec<_>>(),
            1,
        );
        let runtime = self
            .runtime
            .forward_t(
                &Tensor::cat(&[rz.shallow_clone(), g.shallow_clone()], 1),
                train,
            )
            .squeeze_dim(-1)
            .softplus();
        (reward, runtime, phi, g)
    }
}

pub fn sequence_nll(
    logits: &Tensor,
    targets: &Tensor,
    legal: &Tensor,
    pad: &Tensor,
    weights: &Tensor,
) -> Result<Tensor> {
    ensure!(
        legal.any_dim(-1, false).all().int64_value(&[]) != 0,
        "empty legal token set"
    );
    // A row without a supervised position would keep its weight in the
    // denominator with a zero numerator and dilute every other row (W6 G8).
    ensure!(
        pad.any_dim(-1, false).all().int64_value(&[]) != 0,
        "sequence row without supervised positions"
    );
    ensure!(
        legal
            .gather(-1, &targets.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .logical_or(&pad.logical_not())
            .all()
            .int64_value(&[])
            != 0,
        "illegal supervision token"
    );
    let logp = logits
        .to_kind(Kind::Float)
        .masked_fill(&legal.logical_not(), f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float);
    let nll = -logp
        .gather(-1, &targets.unsqueeze(-1), false)
        .squeeze_dim(-1);
    let smooth = -logp.masked_fill(&legal.logical_not(), 0.0).sum_dim_intlist(
        &[-1i64][..],
        false,
        Kind::Float,
    ) / legal
        .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        .clamp_min(1.0);
    let nll: Tensor = nll * 0.95 + smooth * 0.05;
    let nll = nll.masked_fill(&pad.logical_not(), 0.0);
    let per_row = nll.sum_dim_intlist(&[-1i64][..], false, Kind::Float)
        / pad
            .sum_dim_intlist(&[-1i64][..], false, Kind::Float)
            .clamp_min(1.0);
    let weights = weights.clamp_min(0.0);
    Ok((per_row * &weights).sum(Kind::Float) / weights.sum(Kind::Float).clamp_min(1e-8))
}
/// Per-row log-probability of the supervised token sequence under the policy:
/// the pad-masked SUM of grammar-masked token log-probs, with no label
/// smoothing and no length normalisation. This is the quantity a preference
/// objective compares between the live policy and a frozen reference.
/// `sequence_nll` deliberately applies both smoothing and length normalisation
/// and must not be reused for it.
pub fn sequence_logprob(
    logits: &Tensor,
    targets: &Tensor,
    legal: &Tensor,
    pad: &Tensor,
) -> Result<Tensor> {
    ensure!(
        legal
            .gather(-1, &targets.unsqueeze(-1), false)
            .squeeze_dim(-1)
            .logical_or(&pad.logical_not())
            .all()
            .int64_value(&[])
            != 0,
        "illegal supervision token"
    );
    let logp = logits
        .to_kind(Kind::Float)
        .masked_fill(&legal.logical_not(), f64::NEG_INFINITY)
        .log_softmax(-1, Kind::Float)
        .gather(-1, &targets.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .masked_fill(&pad.logical_not(), 0.0);
    Ok(logp.sum_dim_intlist(&[-1i64][..], false, Kind::Float))
}

/// Ordered preference pairs inside each comparison group: the group's best
/// program against every other member, plus each adjacent pair in value order
/// (where the champion-margin information lives). Pairs whose value gap is
/// zero, non-finite, or below `min_gap_fraction` of the group's median
/// non-zero gap are skipped and counted, so a preference objective does not
/// train on fold noise. Uniform pair weight: the |Δvalue| weighting used by the
/// critic's ranking losses is intentionally not reused here.
pub fn preference_pairs(
    values: &[f64],
    groups: &[String],
    min_gap_fraction: f64,
) -> (Vec<i64>, Vec<i64>, usize) {
    let mut grouped: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i);
    }
    let mut winners = Vec::new();
    let mut losers = Vec::new();
    let mut skipped = 0usize;
    for ids in grouped.values() {
        let mut ordered: Vec<usize> = ids
            .iter()
            .copied()
            .filter(|&i| values[i].is_finite())
            .collect();
        // Best first; ties break on the row index so the order is stable.
        ordered.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        if ordered.len() < 2 {
            continue;
        }
        let mut gaps: Vec<f64> = ordered
            .windows(2)
            .map(|w| values[w[0]] - values[w[1]])
            .filter(|g| *g > 0.0)
            .collect();
        gaps.sort_by(|a, b| a.total_cmp(b));
        let floor = if gaps.is_empty() {
            0.0
        } else {
            gaps[gaps.len() / 2] * min_gap_fraction
        };
        let mut push = |w: usize, l: usize| {
            let gap = values[w] - values[l];
            if gap > 0.0 && gap >= floor {
                winners.push(w as i64);
                losers.push(l as i64);
            } else {
                skipped += 1;
            }
        };
        let best = ordered[0];
        for &l in &ordered[1..] {
            push(best, l);
        }
        // Adjacent pairs; (best, second) is already covered above.
        for pair in ordered.windows(2).skip(1) {
            push(pair[0], pair[1]);
        }
    }
    (winners, losers, skipped)
}

/// Direct preference optimisation (Rafailov et al. 2023) over ordered pairs:
/// `-log sigmoid(beta * ((lp_w - ref_w) - (lp_l - ref_l)))`, mean over pairs.
/// The reference log-probabilities are constants, so the gradient lands only
/// on the live policy's own sequence probabilities — the decoder, not the
/// critic. Returns a connected zero when there are no pairs.
pub fn preference_dpo(
    logp: &Tensor,
    logp_ref: &Tensor,
    winners: &[i64],
    losers: &[i64],
    beta: f64,
) -> Result<Tensor> {
    ensure!(winners.len() == losers.len(), "preference pair mismatch");
    ensure!(
        logp.size() == logp_ref.size(),
        "preference log-probability shape mismatch"
    );
    if winners.is_empty() {
        return Ok(logp.sum(Kind::Float) * 0.0);
    }
    let device = logp.device();
    let w = Tensor::from_slice(winners).to_device(device);
    let l = Tensor::from_slice(losers).to_device(device);
    let margin = logp - logp_ref;
    let diff = margin.index_select(0, &w) - margin.index_select(0, &l);
    Ok(-(diff * beta).log_sigmoid().mean(Kind::Float))
}

pub fn ranking(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
) -> Result<(Tensor, Tensor)> {
    ranking_losses(reward, values, percentiles, groups, true)
}

/// The ablation objective uses only Bradley–Terry loss. Avoid constructing
/// per-group listwise graphs and transferring labels for a discarded result.
pub fn ranking_pairwise(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
) -> Result<Tensor> {
    Ok(ranking_losses(reward, values, percentiles, groups, false)?.0)
}

fn ranking_losses(
    reward: &Tensor,
    values: &[f64],
    percentiles: &[f32],
    groups: &[String],
    include_listwise: bool,
) -> Result<(Tensor, Tensor)> {
    ensure!(
        values.len() == groups.len() && percentiles.len() == groups.len(),
        "ranking labels mismatch"
    );
    let mut grouped: BTreeMap<&str, Vec<i64>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        grouped.entry(g).or_default().push(i as i64);
    }
    let mut winners = Vec::new();
    let mut losers = Vec::new();
    let mut weights = Vec::new();
    let mut listwise = Vec::new();
    for ids in grouped.values() {
        for (pos, &a) in ids.iter().enumerate() {
            for &b in &ids[pos + 1..] {
                let va = values[a as usize];
                let vb = values[b as usize];
                if !va.is_finite() || !vb.is_finite() || va == vb {
                    continue;
                }
                let (w, l) = if va > vb { (a, b) } else { (b, a) };
                winners.push(w);
                losers.push(l);
                weights.push(
                    (va - vb).abs()
                        * if percentiles[a as usize].max(percentiles[b as usize]) >= 0.9 {
                            2.0
                        } else {
                            1.0
                        },
                );
            }
        }
        if include_listwise
            && ids.len() >= 3
            && ids.iter().all(|&i| percentiles[i as usize].is_finite())
        {
            let ps = Tensor::from_slice(
                &ids.iter()
                    .map(|&i| percentiles[i as usize])
                    .collect::<Vec<_>>(),
            )
            .to_device(reward.device());
            let target = (&ps / 0.25).softmax(0, Kind::Float);
            let pred = reward
                .index_select(0, &Tensor::from_slice(ids).to_device(reward.device()))
                .to_kind(Kind::Float)
                .log_softmax(0, Kind::Float);
            listwise.push(-(target * pred).sum(Kind::Float));
        }
    }
    let zero = reward.sum(Kind::Float) * 0.0;
    let bt = if winners.is_empty() {
        zero.shallow_clone()
    } else {
        let scale = weights.iter().sum::<f64>() / weights.len() as f64;
        // Repeated index_select entries backpropagate through CUDA atomic adds.
        // Tiny, unordered cancellation errors then become visible AdamW updates
        // to the otherwise translation-invariant reward biases. Broadcast each
        // reward once and reduce a dense comparison matrix in a fixed layout.
        let n = groups.len();
        let mut dense = vec![0f32; n * n];
        for ((&w, &l), &weight) in winners.iter().zip(&losers).zip(&weights) {
            dense[w as usize * n + l as usize] = (weight / scale) as f32;
        }
        let weights = Tensor::from_slice(&dense)
            .view([n as i64, n as i64])
            .to_device(reward.device());
        let scores = reward.to_kind(Kind::Float);
        (((scores.unsqueeze(0) - scores.unsqueeze(1)).softplus()) * weights).sum(Kind::Float)
            / winners.len() as f64
    };
    Ok((
        bt,
        if listwise.is_empty() {
            zero
        } else {
            Tensor::stack(&listwise, 0).mean(Kind::Float)
        },
    ))
}

/// Explicit FP32 AdamW state is portable and resumable; tch's C optimizer does
/// not expose state serialization. No reduced-precision master weights are used.
pub struct AdamW {
    pub step: u64,
    pub lr: f64,
    pub decay: f64,
    pub moments: BTreeMap<String, (Tensor, Tensor)>,
    backend: String,
}
impl AdamW {
    pub fn validate_parameters(&self, vs: &nn::VarStore) -> Result<()> {
        let parameters = vs
            .variables()
            .into_iter()
            .filter(|(_, p)| p.requires_grad())
            .collect::<BTreeMap<_, _>>();
        ensure!(
            self.step > 0
                && self.lr == 1e-4
                && self.decay == 0.01
                && self.moments.len() == parameters.len(),
            "optimizer warm-start configuration/parameter count mismatch"
        );
        for (name, p) in parameters {
            let (m, v) = self
                .moments
                .get(&name)
                .context("optimizer parameter missing")?;
            ensure!(
                m.size() == p.size()
                    && v.size() == p.size()
                    && m.kind() == Kind::Float
                    && v.kind() == Kind::Float,
                "optimizer warm-start shape or dtype mismatch: {name}"
            );
            ensure!(
                m.isfinite().all().int64_value(&[]) == 1
                    && v.isfinite().all().int64_value(&[]) == 1,
                "nonfinite optimizer state: {name}"
            );
        }
        Ok(())
    }
    pub fn new(lr: f64, decay: f64) -> Self {
        Self {
            step: 0,
            lr,
            decay,
            moments: BTreeMap::new(),
            backend: "uninitialized".into(),
        }
    }
    pub fn update(&mut self, vs: &nn::VarStore) -> Result<()> {
        if matches!(vs.device(), Device::Cuda(_)) {
            return self.update_cuda(vs);
        }
        self.update_reference(vs)
    }
    /// Scalar reference remains available for numerical parity checks.
    pub fn update_reference(&mut self, vs: &nn::VarStore) -> Result<()> {
        ensure!(
            self.step == 0 || self.backend == "scalar_fp32_v1",
            "optimizer backend changed; start a new run from model weights instead of resuming"
        );
        let parameters = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        // A rejected update changes nothing: validate every gradient before the
        // step counter, backend, moments or any parameter is touched (W6-T04).
        for p in parameters.values() {
            let g = p.grad();
            if g.defined() {
                ensure!(
                    p.kind() == Kind::Float && g.isfinite().all().int64_value(&[]) != 0,
                    "nonfinite gradient or non-FP32 master parameter"
                );
            }
        }
        self.backend = "scalar_fp32_v1".into();
        self.step += 1;
        let step = self.step as f64;
        tch::no_grad(|| -> Result<()> {
            for (name, mut p) in parameters {
                let g = p.grad();
                if !g.defined() {
                    continue;
                }
                let (m, v) = self
                    .moments
                    .entry(name)
                    .or_insert_with(|| (Tensor::zeros_like(&p), Tensor::zeros_like(&p)));
                m.copy_(&(&*m * 0.9 + &g * 0.1));
                v.copy_(&(&*v * 0.999 + g.square() * 0.001));
                let delta = (&*m / (1.0 - 0.9f64.powf(step)))
                    / ((&*v / (1.0 - 0.999f64.powf(step))).sqrt() + 1e-8);
                p.copy_(&(&p * (1.0 - self.lr * self.decay) - delta * self.lr));
                p.zero_grad();
            }
            Ok(())
        })
    }
    fn update_cuda(&mut self, vs: &nn::VarStore) -> Result<()> {
        ensure!(
            self.step == 0 || self.backend == "cuda_fused_fp32_v1",
            "optimizer backend changed; use warm-start with fresh optimizer state"
        );
        let variables = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        // Validate before creating any moment entry: a rejected update leaves
        // the optimizer state exactly as it was (W6-T04).
        let mut gradients = Vec::new();
        for p in variables.values() {
            let g = p.grad();
            if !g.defined() {
                continue;
            }
            ensure!(
                p.kind() == Kind::Float && g.kind() == Kind::Float,
                "fused AdamW requires FP32 master parameters and gradients"
            );
            gradients.push(g.flatten(0, -1));
        }
        ensure!(!gradients.is_empty(), "no optimizer gradients");
        ensure!(
            Tensor::cat(&gradients, 0).isfinite().all().int64_value(&[]) != 0,
            "nonfinite optimizer gradients"
        );
        let mut parameters = Vec::new();
        let mut first = Vec::new();
        let mut second = Vec::new();
        for (name, p) in variables {
            if !p.grad().defined() {
                continue;
            }
            let (m, v) = self
                .moments
                .entry(name)
                .or_insert_with(|| (Tensor::zeros_like(&p), Tensor::zeros_like(&p)));
            first.push(m.shallow_clone());
            second.push(v.shallow_clone());
            parameters.push(p);
        }
        let pointers = |tensors: &mut Vec<Tensor>| {
            tensors
                .iter_mut()
                .map(|t| t.as_mut_ptr() as *mut c_void)
                .collect::<Vec<_>>()
        };
        let mut p = pointers(&mut parameters);
        let mut m = pointers(&mut first);
        let mut v = pointers(&mut second);
        let step = self.step + 1;
        ensure!(
            unsafe {
                jp_fused_adamw(
                    p.as_mut_ptr(),
                    m.as_mut_ptr(),
                    v.as_mut_ptr(),
                    p.len(),
                    step as i64,
                    self.lr,
                    self.decay,
                )
            } == 0,
            "{}",
            native_error()
        );
        self.step = step;
        self.backend = "cuda_fused_fp32_v1".into();
        Ok(())
    }
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut tensors = Vec::new();
        for (name, (m, v)) in &self.moments {
            tensors.push((format!("m/{name}"), m.shallow_clone()));
            tensors.push((format!("v/{name}"), v.shallow_clone()));
        }
        Tensor::save_multi(&tensors, path)?;
        write_json(
            &path.with_extension("json"),
            &serde_json::json!({"step":self.step,"lr":self.lr,"decay":self.decay,"backend":self.backend,"sha256":file_hash(path)?}),
        )
    }
    pub fn load(path: &Path, device: Device) -> Result<Self> {
        let m: serde_json::Value = read_json(&path.with_extension("json"))?;
        ensure!(
            file_hash(path)? == m["sha256"],
            "optimizer artifact changed"
        );
        let mut tensors = Tensor::load_multi_with_device(path, device)?
            .into_iter()
            .collect::<BTreeMap<_, _>>();
        let names = tensors
            .keys()
            .filter_map(|n| n.strip_prefix("m/").map(String::from))
            .collect::<Vec<_>>();
        let mut moments = BTreeMap::new();
        for n in names {
            moments.insert(
                n.clone(),
                (
                    tensors.remove(&format!("m/{n}")).unwrap(),
                    tensors
                        .remove(&format!("v/{n}"))
                        .context("missing second moment")?,
                ),
            );
        }
        ensure!(tensors.is_empty(), "unpaired optimizer moments");
        Ok(Self {
            step: m["step"].as_u64().context("optimizer step")?,
            lr: m["lr"].as_f64().context("optimizer learning rate")?,
            decay: m["decay"].as_f64().context("optimizer weight decay")?,
            moments,
            backend: m["backend"].as_str().unwrap_or("scalar_fp32_v1").into(),
        })
    }
}

extern "C" {
    fn jp_error() -> *const std::ffi::c_char;
    fn jp_unique_id(bytes: *mut u8) -> i32;
    fn jp_nccl_init(bytes: *const u8, rank: i32, world: i32) -> *mut c_void;
    fn jp_allreduce(comm: *mut c_void, tensor: *mut c_void) -> i32;
    fn jp_nccl_abort(comm: *mut c_void);
    fn jp_bf16(enabled: i32);
    fn jp_fused_adamw(
        parameters: *mut *mut c_void,
        first: *mut *mut c_void,
        second: *mut *mut c_void,
        count: usize,
        step: i64,
        lr: f64,
        decay: f64,
    ) -> i32;
}
fn native_error() -> String {
    unsafe { CStr::from_ptr(jp_error()).to_string_lossy().into_owned() }
}
pub struct Nccl {
    comm: *mut c_void,
    pub world: usize,
}
impl Nccl {
    pub fn init(rank: usize, world: usize, rendezvous: &Path) -> Result<Self> {
        ensure!(rank < world && world <= 8, "invalid rank topology");
        let mut id = [0u8; 128];
        if rank == 0 {
            ensure!(
                !rendezvous.exists(),
                "rendezvous must be unique to this launch"
            );
            ensure!(
                unsafe { jp_unique_id(id.as_mut_ptr()) } == 0,
                "{}",
                native_error()
            );
            atomic_write(rendezvous, &id)?;
        } else {
            let start = Instant::now();
            while !rendezvous.exists() {
                ensure!(
                    start.elapsed() < Duration::from_secs(60),
                    "NCCL rendezvous timeout"
                );
                std::thread::sleep(Duration::from_millis(50));
            }
            id.copy_from_slice(&fs::read(rendezvous)?[..128]);
        }
        let comm = unsafe { jp_nccl_init(id.as_ptr(), rank as i32, world as i32) };
        ensure!(!comm.is_null(), "{}", native_error());
        Ok(Self { comm, world })
    }
    pub fn reduce(&self, t: &mut Tensor) -> Result<()> {
        ensure!(
            unsafe { jp_allreduce(self.comm, t.as_mut_ptr() as *mut c_void) } == 0,
            "{}",
            native_error()
        );
        let mean = &*t / self.world as f64;
        t.copy_(&mean);
        Ok(())
    }
}
impl Drop for Nccl {
    fn drop(&mut self) {
        unsafe {
            jp_nccl_abort(self.comm);
        }
    }
}
pub struct Bf16;
impl Bf16 {
    pub fn enter() -> Self {
        unsafe {
            jp_bf16(1);
        }
        Self
    }
}
impl Drop for Bf16 {
    fn drop(&mut self) {
        unsafe {
            jp_bf16(0);
        }
    }
}

pub fn nccl_probe(rank: usize, world: usize, directory: &Path) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("nccl-probe")?;
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    let comm = Nccl::init(rank, world, &directory.join("nccl.id"))?;
    let mut tensor = Tensor::full([4096], rank as f64 + 1.0, (Kind::Float, Device::Cuda(rank)));
    let start = Instant::now();
    comm.reduce(&mut tensor)?;
    let expected = (world + 1) as f64 / 2.0;
    let error = (&tensor - expected).abs().max().double_value(&[]);
    ensure!(error < 1e-6, "gradient reduction mismatch");
    let values: Vec<f32> = Vec::try_from(tensor.to_device(Device::Cpu))?;
    write_json(
        &directory.join(format!("rank-{rank}.json")),
        &serde_json::json!({"rank":rank,"world":world,"steps":1,"max_abs_error":error,"seconds":start.elapsed().as_secs_f64(),"parameter_sha256":hash(&values.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>())}),
    )
}

pub fn parity(directory: &Path) -> Result<()> {
    let config: Config = read_json(&directory.join("config.json"))?;
    let mut vs = nn::VarStore::new(Device::Cpu);
    let model = Policy::new(vs.root(), config)?;
    vs.load(directory.join("reference.safetensors"))?;
    let tensors = Tensor::read_safetensors(directory.join("fixtures.safetensors"))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| tensors.get(name).context("missing parity tensor");
    let out = tch::no_grad(|| {
        model.forward(
            get("dataset")?,
            get("target")?,
            get("tokens")?,
            get("pad")?,
            false,
        )
    })?;
    let actual = [
        ("logits", out.0),
        ("reward", out.1),
        ("runtime", out.2),
        ("stability", out.3),
        ("ablation", out.4),
    ];
    let mut errors = BTreeMap::new();
    for (name, t) in actual {
        let e = (&t - get(name)?).abs().max().double_value(&[]);
        ensure!(e < 3e-4, "policy parity failed for {name}: {e}");
        errors.insert(name, e);
    }
    write_json(
        &directory.join("policy.json"),
        &serde_json::json!({"schema":"parallel.policy_parity.v1","passed":true,"atol":3e-4,"max_abs_errors":errors,"reference_sha256":file_hash(&directory.join("reference.safetensors"))?}),
    )
}

pub fn loss_optimizer_parity(directory: &Path) -> Result<()> {
    let tensors = Tensor::read_safetensors(directory.join("fixtures.safetensors"))?
        .into_iter()
        .collect::<BTreeMap<_, _>>();
    let get = |name: &str| tensors.get(name).context("missing loss parity tensor");
    let seq = sequence_nll(
        &get("logits")?.narrow(1, 1, 8),
        &get("tokens")?.narrow(1, 1, 8),
        get("legal")?,
        &get("pad")?.narrow(1, 1, 8),
        get("seq_weights")?,
    )?;
    let values: Vec<f64> = Vec::try_from(get("ranking_values")?.to_kind(Kind::Double))?;
    let pct: Vec<f32> = Vec::try_from(get("ranking_percentiles")?)?;
    let (bt, lw) = ranking(
        &get("reward")?.mean_dim(&[-1i64][..], false, Kind::Float),
        &values,
        &pct,
        &vec!["same-cut".to_string(); 3],
    )?;
    let mut errors = BTreeMap::new();
    for (name, t) in [("seq_nll", seq), ("bt", bt), ("listwise", lw)] {
        let e = (t - get(name)?).abs().double_value(&[]);
        ensure!(e < 1e-5, "loss parity {name} failed: {e}");
        errors.insert(name, e);
    }
    let vs = nn::VarStore::new(Device::Cpu);
    let p = vs.root().var_copy("parameter", get("adam_initial")?);
    let mut opt = AdamW::new(0.003, 0.01);
    for i in 0..3 {
        (&p * get("adam_gradients")?.get(i))
            .sum(Kind::Float)
            .backward();
        opt.update(&vs)?;
    }
    let saved = directory.join("optimizer-continuation.ot");
    opt.save(&saved)?;
    let mut resumed = AdamW::load(&saved, Device::Cpu)?;
    for i in 3..5 {
        (&p * get("adam_gradients")?.get(i))
            .sum(Kind::Float)
            .backward();
        resumed.update(&vs)?;
    }
    for (name, actual) in [
        ("adam_after_5", p),
        ("adam_m", resumed.moments["parameter"].0.shallow_clone()),
        ("adam_v", resumed.moments["parameter"].1.shallow_clone()),
    ] {
        let e = (&actual - get(name)?).abs().max().double_value(&[]);
        ensure!(e < 1e-6, "optimizer parity {name} failed: {e}");
        errors.insert(name, e);
    }
    write_json(
        &directory.join("optimizer.json"),
        &serde_json::json!({"schema":"parallel.optimizer_loss_parity.v1","passed":true,"optimizer_resume_step":3,"final_step":resumed.step,"max_abs_errors":errors,"fixtures_sha256":file_hash(&directory.join("fixtures.safetensors"))?}),
    )
}

/// Measured full-architecture BF16 forward/backward/AdamW qualification. These
/// deterministic numerical fixtures never become dataset corpus or policy assets.
pub fn training_probe(
    rank: usize,
    world: usize,
    directory: &Path,
    vocabulary_path: &Path,
    microbatch: usize,
    global_batch: usize,
    steps: usize,
) -> Result<()> {
    crate::task_contract::require_qualified_gpu_experiment("training-probe-rank")?;
    let accumulation = crate::controller::probe_accumulation(world, microbatch, global_batch)?;
    ensure!(
        rank < world && steps > 5,
        "invalid throughput probe rank or steps"
    );
    tch::set_num_threads(2);
    tch::set_num_interop_threads(1);
    tch::manual_seed(SEED as i64);
    let comm = Nccl::init(rank, world, &directory.join("nccl.id"))?;
    let device = Device::Cuda(rank);
    let vs = nn::VarStore::new(device);
    let vocabulary: crate::grammar::Vocab = read_json(vocabulary_path)?;
    let vocabulary_sha256 = file_hash(vocabulary_path)?;
    let vocabulary_size = vocabulary.tokens.len() as i64;
    ensure!(
        vocabulary_size > 3,
        "probe vocabulary needs reserved and program tokens"
    );
    let c = Config::regression(vocabulary_size);
    let model = Policy::new(vs.root(), c)?;
    let mut optimizer = AdamW::new(1e-4, 0.01);
    tch::manual_seed((SEED + rank as u64) as i64);
    let dataset = Tensor::randn([microbatch as i64, 1536], (Kind::Float, device));
    let target = Tensor::randn([microbatch as i64, 512], (Kind::Float, device));
    let tokens = Tensor::randint(
        vocabulary_size - 3,
        [microbatch as i64, 128],
        (Kind::Int64, device),
    ) + 3;
    let pad = Tensor::ones([microbatch as i64, 128], (Kind::Bool, device));
    let started = Instant::now();
    let mut times = Vec::new();
    for step in 0..steps {
        let begun = Instant::now();
        let mut loss_value = 0.0;
        for _ in 0..accumulation {
            let guard = Bf16::enter();
            let (logits, rewards, runtime, stability, ablation) =
                model.forward(&dataset, &target, &tokens, &pad, true)?;
            let logits = logits.narrow(1, 1, 127).to_kind(Kind::Float);
            let labels = tokens.narrow(1, 1, 127);
            let nll = logits
                .reshape([-1, vocabulary_size])
                .cross_entropy_for_logits(&labels.reshape([-1]));
            let mut loss = nll
                + rewards.square().mean(Kind::Float) * 0.01
                + runtime.square().mean(Kind::Float) * 0.01
                + stability.square().mean(Kind::Float) * 0.01
                + ablation.square().mean(Kind::Float) * 0.01;
            for s in model.loss_log_vars.values() {
                loss += s.square() * 0.001;
            }
            loss /= accumulation as f64;
            loss_value += loss.double_value(&[]);
            drop(guard);
            loss.backward();
        }
        let mut variables = vs.variables().into_iter().collect::<BTreeMap<_, _>>();
        let gradients = variables.values().map(|p| p.grad()).collect::<Vec<_>>();
        ensure!(
            gradients.iter().all(|g| g.defined()),
            "undefined rank gradient"
        );
        let mut packed = Tensor::cat(
            &gradients
                .iter()
                .map(|g| g.flatten(0, -1))
                .collect::<Vec<_>>(),
            0,
        );
        ensure!(
            packed.isfinite().all().int64_value(&[]) != 0,
            "nonfinite packed gradients"
        );
        let communication = Instant::now();
        comm.reduce(&mut packed)?;
        let communication_seconds = communication.elapsed().as_secs_f64();
        let norm = packed.norm().double_value(&[]);
        if norm > 1.0 {
            packed /= norm;
        }
        let mut offset = 0;
        for p in variables.values_mut() {
            let mut g = p.grad();
            let size = p.numel() as i64;
            g.copy_(&packed.narrow(0, offset, size).reshape(p.size()));
            offset += size;
        }
        optimizer.update(&vs)?;
        tch::Cuda::synchronize(rank as i64);
        let seconds = begun.elapsed().as_secs_f64();
        times.push(seconds);
        write_json(
            &directory.join(format!("rank-{rank}.json")),
            &serde_json::json!({"schema":"parallel.training_probe_rank.v1","rank":rank,"world":world,"phase":"BF16 architecture qualification","training_data":"numerical qualification fixtures; excluded from corpus","step":step+1,"steps":steps,"microbatch":microbatch,"accumulation":accumulation,"global_batch":global_batch,"loss":loss_value,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"timestamp":crate::telemetry::now()}),
        )?;
        if rank == 0 {
            write_json(
                &directory
                    .parent()
                    .unwrap_or(directory)
                    .join("neural-service.json"),
                &serde_json::json!({"phase":"BF16 architecture qualification","corpus_eligible":false,"step":step+1,"steps":steps,"world":world,"microbatch":microbatch,"global_batch":global_batch,"loss":loss_value,"step_seconds":seconds,"communication_seconds":communication_seconds,"examples_per_second":global_batch as f64/seconds,"elapsed_seconds":started.elapsed().as_secs_f64(),"timestamp":crate::telemetry::now()}),
            )?;
        }
    }
    // Compare real parameters across ranks, after all synchronized updates.
    let mut hasher = sha2::Sha256::new();
    use sha2::Digest;
    for (name, p) in vs.variables().into_iter().collect::<BTreeMap<_, _>>() {
        hasher.update(name.as_bytes());
        let values: Vec<f32> = Vec::try_from(p.to_device(Device::Cpu).flatten(0, -1))?;
        for v in values {
            hasher.update(v.to_le_bytes());
        }
    }
    write_json(
        &directory.join(format!("complete-{rank}.json")),
        &serde_json::json!({"rank":rank,"world":world,"steps":steps,"parameter_sha256":hex::encode(hasher.finalize()),"microbatch":microbatch,"global_batch":global_batch,"seconds":started.elapsed().as_secs_f64(),"step_seconds":times,"vocabulary_sha256":vocabulary_sha256,"vocabulary_size":vocabulary_size,"eligible_checkpoint":false}),
    )
}
/// LibTorch interop configuration may be set only before the first task.
pub fn configure_runtime() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        tch::set_num_threads(2);
        tch::set_num_interop_threads(1);
        tch::jit::set_graph_executor_optimize(false);
    });
}

#[cfg(test)]
mod native_runtime_qualification {
    use super::*;
    #[test]
    fn cpu_optimizer_restore_completes_original_window_identically() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let uninterrupted = nn::VarStore::new(Device::Cpu);
        let p = uninterrupted
            .root()
            .var("weight", &[4], nn::Init::Const(0.25));
        let mut optimizer = AdamW::new(1e-4, 0.01);
        for _ in 0..3 {
            p.square().sum(Kind::Float).backward();
            optimizer.update(&uninterrupted)?;
        }
        uninterrupted.save(dir.path().join("model.ot"))?;
        optimizer.save(&dir.path().join("optimizer.ot"))?;
        let original_target = optimizer.step + 3;
        let mut restored = nn::VarStore::new(Device::Cpu);
        let restored_p = restored.root().var("weight", &[4], nn::Init::Const(0.0));
        restored.load(dir.path().join("model.ot"))?;
        let mut recovered = AdamW::load(&dir.path().join("optimizer.ot"), Device::Cpu)?;
        recovered.validate_parameters(&restored)?;
        while optimizer.step < original_target {
            p.square().sum(Kind::Float).backward();
            optimizer.update(&uninterrupted)?;
        }
        while recovered.step < original_target {
            restored_p.square().sum(Kind::Float).backward();
            recovered.update(&restored)?;
        }
        assert_eq!(recovered.step, original_target);
        assert!(p.equal(&restored_p));
        for (name, (m, v)) in &optimizer.moments {
            let (rm, rv) = &recovered.moments[name];
            assert!(m.equal(rm) && v.equal(rv));
        }
        let mut corrupt = fs::read(dir.path().join("optimizer.ot"))?;
        corrupt[0] ^= 1;
        fs::write(dir.path().join("optimizer.ot"), corrupt)?;
        assert!(AdamW::load(&dir.path().join("optimizer.ot"), Device::Cpu).is_err());
        Ok(())
    }
}
