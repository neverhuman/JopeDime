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
                        Some(REWARD_HEAD_DROPOUT),
                    )
                })
                .collect(),
            runtime: mlp(&p / "runtime_head", 2 * c.width, c.width, None),
            stability: mlp(&p / "stability_head", 3 * c.width, c.width, None),
            ablation: mlp(
                &p / "reward_head_ablation",
                c.width,
                c.width,
                Some(REWARD_HEAD_DROPOUT),
            ),
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

/// W2-T04 (G-10c): a warm-start or resume consumes exactly the checkpoint
/// bytes its state binds. The instrumentation point
/// `manifest::after_verification` replaces the files between their check and
/// their load; the load must still be of the verified bytes.
#[cfg(test)]
mod bound_load_tests {
    use super::*;
    use crate::manifest::verification_hook;

    fn store(value: f64) -> nn::VarStore {
        let vs = nn::VarStore::new(Device::Cpu);
        let _ = vs.root().var("weight", &[4], nn::Init::Const(value));
        vs
    }

    fn weight(vs: &nn::VarStore) -> Vec<f32> {
        Vec::<f32>::try_from(vs.variables()["weight"].flatten(0, -1)).unwrap()
    }

    /// Replace each `(target, bytes)` (new inodes, via rename) the first time
    /// a receiver reports `trigger` verified.
    fn replace_on_verification(
        trigger: &Path,
        replacements: Vec<(std::path::PathBuf, Vec<u8>)>,
    ) -> verification_hook::Guard {
        let trigger = fs::canonicalize(trigger).unwrap();
        let mut done = false;
        verification_hook::install(move |path| {
            if !done && fs::canonicalize(path).ok().as_deref() == Some(trigger.as_path()) {
                done = true;
                for (target, bytes) in &replacements {
                    let staged = target.with_extension("replacement");
                    fs::write(&staged, bytes).unwrap();
                    fs::rename(&staged, target).unwrap();
                }
            }
        })
    }

    #[test]
    fn a_checkpoint_model_replaced_after_its_check_is_never_loaded() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("model.ot");
        store(1.0).save(&path)?;
        let expected = serde_json::json!(file_hash(&path)?);
        let other = dir.path().join("other.ot");
        store(2.0).save(&other)?;
        // Positive control.
        let mut vs = store(0.0);
        load_bound_model(&mut vs, &path, &expected, "initialization model changed")?;
        assert_eq!(weight(&vs), [1.0; 4]);
        // Replaced after its check.
        let mut raced = store(0.0);
        let result = {
            let _hook = replace_on_verification(&path, vec![(path.clone(), fs::read(&other)?)]);
            load_bound_model(&mut raced, &path, &expected, "initialization model changed")
        };
        assert!(
            result.is_ok() && weight(&raced) == [1.0; 4],
            "a model other than the bound bytes was loaded: {result:?} {:?}",
            weight(&raced)
        );
        // Negative control: other bytes at the path are refused with the receiver's wording.
        let mut refused = store(0.0);
        let error = load_bound_model(
            &mut refused,
            &path,
            &expected,
            "initialization model changed",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("initialization model changed"), "{error}");
        assert_eq!(weight(&refused), [0.0; 4]);
        Ok(())
    }

    #[test]
    fn a_checkpoint_optimizer_replaced_after_its_check_is_never_loaded() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let optimizer = |step: u64, value: f64| AdamW {
            step,
            lr: 1e-4,
            decay: 0.01,
            moments: BTreeMap::from([(
                "weight".to_string(),
                (
                    Tensor::full([4], value, (Kind::Float, Device::Cpu)),
                    Tensor::full([4], value * 2.0, (Kind::Float, Device::Cpu)),
                ),
            )]),
            backend: "scalar_fp32_v1".into(),
        };
        let path = dir.path().join("optimizer.ot");
        optimizer(3, 1.0).save(&path)?;
        let expected = serde_json::json!(file_hash(&path)?);
        let expected_metadata = serde_json::json!(file_hash(&path.with_extension("json"))?);
        // A consistent replacement pair: another optimizer and the metadata naming it.
        let other = dir.path().join("other").join("optimizer.ot");
        fs::create_dir_all(other.parent().unwrap())?;
        optimizer(9, 5.0).save(&other)?;
        let replacements = vec![
            (path.clone(), fs::read(&other)?),
            (
                path.with_extension("json"),
                fs::read(other.with_extension("json"))?,
            ),
        ];
        let refusal = "warm-start optimizer missing or changed";
        // Positive control.
        let loaded = AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)?;
        assert_eq!(loaded.step, 3);
        // Replaced after its check.
        let result = {
            let _hook = replace_on_verification(&path, replacements);
            AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)
        };
        assert!(
            matches!(&result, Ok(raced) if raced.step == 3
                && raced.moments["weight"].0.equal(&Tensor::full([4], 1.0, (Kind::Float, Device::Cpu)))),
            "an optimizer other than the bound bytes was loaded: {:?}",
            result.as_ref().map(|o| o.step)
        );
        // Negative control: the replaced pair is refused with the receiver's wording.
        let error = AdamW::load_bound(&path, &expected, &expected_metadata, Device::Cpu, refusal)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains(refusal), "{error}");
        Ok(())
    }

    /// W7 (961ad2333), landed on this API: a checkpoint restored through
    /// `load_bound_model` and `AdamW::load_bound` after one AdamW update is the
    /// saved state and continues training exactly as the original, although
    /// every file it came from is replaced right after its check. The
    /// replacements are refused on the next load, and optimizer bytes the
    /// metadata does not name are refused by `from_verified`.
    #[test]
    fn checkpoint_consumption_uses_verified_tensor_bytes_after_path_replacement() -> Result<()> {
        tch::set_num_threads(1);
        let dir = tempfile::tempdir()?;
        let vs = nn::VarStore::new(Device::Cpu);
        let p = vs.root().var("weight", &[2], nn::Init::Const(2.0));
        let mut optimizer = AdamW::new(1e-4, 0.01);
        p.square().sum(Kind::Float).backward();
        optimizer.update(&vs)?;
        let model = dir.path().join("model.ot");
        let optimizer_path = dir.path().join("optimizer.ot");
        vs.save(&model)?;
        optimizer.save(&optimizer_path)?;
        let state = serde_json::json!({
            "model_sha256": file_hash(&model)?,
            "optimizer_sha256": file_hash(&optimizer_path)?,
            "optimizer_metadata_sha256": file_hash(&optimizer_path.with_extension("json"))?,
        });
        let saved_step = optimizer.step;
        let metadata: serde_json::Value = read_json(&optimizer_path.with_extension("json"))?;
        let optimizer_bytes = fs::read(&optimizer_path)?;
        let replaced = || b"replaced after verification".to_vec();
        let (model_refusal, optimizer_refusal) = (
            "checkpoint model changed",
            "checkpoint optimizer or configuration changed",
        );
        // Each file is replaced as soon as it is verified; the load uses the verified bytes.
        let mut restored = nn::VarStore::new(Device::Cpu);
        let q = restored.root().var("weight", &[2], nn::Init::Const(0.0));
        {
            let _hook = replace_on_verification(&model, vec![(model.clone(), replaced())]);
            load_bound_model(&mut restored, &model, &state["model_sha256"], model_refusal)?;
        }
        let mut resumed = {
            let _hook = replace_on_verification(
                &optimizer_path,
                vec![
                    (optimizer_path.clone(), replaced()),
                    (optimizer_path.with_extension("json"), replaced()),
                ],
            );
            AdamW::load_bound(
                &optimizer_path,
                &state["optimizer_sha256"],
                &state["optimizer_metadata_sha256"],
                Device::Cpu,
                optimizer_refusal,
            )?
        };
        // The replacements are refused on the next load, with the receivers' wording.
        let mut other = nn::VarStore::new(Device::Cpu);
        let _ = other.root().var("weight", &[2], nn::Init::Const(0.0));
        let error = load_bound_model(&mut other, &model, &state["model_sha256"], model_refusal)
            .unwrap_err()
            .to_string();
        assert!(error.contains(model_refusal), "{error}");
        let error = AdamW::load_bound(
            &optimizer_path,
            &state["optimizer_sha256"],
            &state["optimizer_metadata_sha256"],
            Device::Cpu,
            optimizer_refusal,
        )
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
        assert!(error.contains(optimizer_refusal), "{error}");
        // The restored checkpoint is the saved one and continues identically.
        assert_eq!(resumed.step, saved_step);
        assert_eq!((&p - &q).abs().max().double_value(&[]), 0.0);
        p.square().sum(Kind::Float).backward();
        optimizer.update(&vs)?;
        q.square().sum(Kind::Float).backward();
        resumed.update(&restored)?;
        assert_eq!((&p - &q).abs().max().double_value(&[]), 0.0);
        // `from_verified` loads the verified bytes and refuses bytes its metadata does not name.
        assert_eq!(
            AdamW::from_verified(metadata.clone(), optimizer_bytes.clone(), Device::Cpu)?.step,
            saved_step
        );
        let mut corrupt = optimizer_bytes;
        corrupt[0] ^= 1;
        let error = AdamW::from_verified(metadata, corrupt, Device::Cpu)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(error.contains("optimizer artifact changed"), "{error}");
        Ok(())
    }
}
