use super::*;

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
