#![cfg(feature = "neural")]

use jope_parallel::neural::{ranking, sequence_nll, AdamW, Bf16, Config, Policy};
use tch::{nn, Device, Kind, Tensor};

#[test]
fn fresh_initialization_matches_reference_contract() {
    tch::set_num_threads(2);
    tch::manual_seed(20260907);
    let vs = nn::VarStore::new(Device::Cpu);
    let mut config = Config::regression(401);
    config.width = 32;
    config.ffn = 128;
    config.layers = 2;
    let _policy = Policy::new(vs.root(), config.clone()).unwrap();
    let vars = vs.variables();
    assert_eq!(
        vars["token_emb.weight"]
            .get(0)
            .abs()
            .max()
            .double_value(&[]),
        0.0
    );
    assert_eq!(
        vars["reward_token_emb.weight"]
            .get(0)
            .abs()
            .max()
            .double_value(&[]),
        0.0
    );
    for (name, tensor) in &vars {
        if let Some(tail) = name.strip_prefix("decoder.layers.1.") {
            assert_eq!(
                (tensor - &vars[&format!("decoder.layers.0.{tail}")])
                    .abs()
                    .max()
                    .double_value(&[]),
                0.0
            );
        }
        if name.ends_with("weight") && tensor.dim() == 2 && !name.contains("_emb") {
            let fan = tensor.size()[1] as f64;
            let bound = if name.ends_with("in_proj_weight") {
                (1.5 / fan).sqrt()
            } else {
                1.0 / fan.sqrt()
            };
            assert!(
                tensor.abs().max().double_value(&[]) <= bound + 1e-7,
                "{name}"
            );
        }
    }
    assert_eq!(
        vars["decoder.layers.0.self_attn.out_proj.bias"]
            .abs()
            .max()
            .double_value(&[]),
        0.0
    );
}

#[test]
fn bf16_forward_observes_updated_master_weights() {
    if !tch::Cuda::is_available() {
        eprintln!("CUDA unavailable: BF16 cache lifetime test skipped");
        return;
    }
    let device = Device::Cuda(0);
    let input = Tensor::ones([4, 4], (Kind::Float, device));
    let mut weight = Tensor::ones([4, 4], (Kind::Float, device)).set_requires_grad(true);
    let first = {
        let _guard = Bf16::enter();
        input.matmul(&weight)
    };
    assert_eq!(first.kind(), Kind::BFloat16);
    assert_eq!(first.double_value(&[0, 0]), 4.0);
    tch::no_grad(|| {
        let _ = weight.fill_(3.0);
    });
    let second = {
        let _guard = Bf16::enter();
        input.matmul(&weight)
    };
    assert_eq!(
        second.double_value(&[0, 0]),
        12.0,
        "autocast reused a stale BF16 weight after the master parameter changed"
    );
    // A nested scope must preserve its outer scope's enabled state.
    {
        let _outer = Bf16::enter();
        {
            let _inner = Bf16::enter();
        }
        assert_eq!(input.matmul(&weight).kind(), Kind::BFloat16);
    }
    assert_eq!(input.matmul(&weight).kind(), Kind::Float);
}

#[test]
fn actual_policy_learns_and_optimizer_continues_after_reload() {
    if !tch::Cuda::is_available() {
        return;
    }
    tch::set_num_threads(2);
    let device = Device::Cuda(0);
    let mut config = Config::regression(16);
    config.width = 32;
    config.ffn = 128;
    config.layers = 2;
    config.dropout = 0.0;
    let vs = nn::VarStore::new(device);
    let policy = Policy::new(vs.root(), config.clone()).unwrap();
    let mut optimizer = AdamW::new(0.003, 0.01);
    let dataset = Tensor::arange(8 * 1536, (Kind::Float, device))
        .reshape([8, 1536])
        .fmod(17.0)
        / 17.0;
    let target = Tensor::ones([8, 512], (Kind::Float, device));
    let tokens = Tensor::from_slice(&[1i64, 3, 4, 5, 6, 7, 8, 2])
        .unsqueeze(0)
        .repeat([8, 1])
        .to_device(device);
    let pad = tokens.ne(0);
    let legal = Tensor::ones([8, 7, 16], (Kind::Bool, device));
    let weights = Tensor::ones([8], (Kind::Float, device));
    let objective = |model: &Policy| {
        let _guard = Bf16::enter();
        let (logits, _, _, _, _) = model
            .forward(&dataset, &target, &tokens, &pad, false)
            .unwrap();
        sequence_nll(
            &logits.narrow(1, 1, 7),
            &tokens.narrow(1, 1, 7),
            &legal,
            &pad.narrow(1, 1, 7),
            &weights,
        )
        .unwrap()
    };
    let initial = objective(&policy).double_value(&[]);
    for _ in 0..20 {
        objective(&policy).backward();
        optimizer.update(&vs).unwrap();
    }
    let learned = objective(&policy).double_value(&[]);
    assert!(
        learned < initial * 0.5,
        "policy did not learn: {initial} -> {learned}"
    );
    let dir = tempfile::tempdir().unwrap();
    vs.save(dir.path().join("model.ot")).unwrap();
    optimizer.save(&dir.path().join("optimizer.ot")).unwrap();
    let mut resumed_vs = nn::VarStore::new(device);
    let resumed_policy = Policy::new(resumed_vs.root(), config).unwrap();
    resumed_vs.load(dir.path().join("model.ot")).unwrap();
    let mut resumed = AdamW::load(&dir.path().join("optimizer.ot"), device).unwrap();
    for _ in 0..5 {
        objective(&policy).backward();
        optimizer.update(&vs).unwrap();
        objective(&resumed_policy).backward();
        resumed.update(&resumed_vs).unwrap();
    }
    let final_loss = objective(&policy).double_value(&[]);
    assert_eq!(resumed.step, optimizer.step);
    assert_eq!(objective(&resumed_policy).double_value(&[]), final_loss);
    for (name, parameter) in vs.variables() {
        assert_eq!(
            (parameter - &resumed_vs.variables()[&name])
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "optimizer continuation changed {name}"
        );
    }
    eprintln!("actual policy BF16 NLL {initial:.6} -> {learned:.6} -> {final_loss:.6}; reloaded continuation identical");
}

#[test]
fn fused_cuda_adamw_matches_reference_moments_and_parameters() {
    if !tch::Cuda::is_available() {
        return;
    }
    let device = Device::Cuda(0);
    let a = nn::VarStore::new(device);
    let mut b = nn::VarStore::new(device);
    for (name, shape) in [
        ("scalar", vec![]),
        ("vector", vec![19]),
        ("matrix", vec![33, 71]),
    ] {
        let _ = a.root().var(&name, &shape, nn::Init::Const(0.25));
        let _ = b.root().var(&name, &shape, nn::Init::Const(0.25));
    }
    b.copy(&a).unwrap();
    let mut fused = AdamW::new(0.003, 0.01);
    let mut reference = AdamW::new(0.003, 0.01);
    for step in 0..50 {
        for vs in [&a, &b] {
            let loss = Tensor::stack(
                &vs.variables()
                    .values()
                    .map(|p| {
                        let g = (Tensor::arange(p.numel() as i64, (Kind::Float, device)) * 0.013
                            + step as f64 * 0.17)
                            .sin()
                            .reshape(p.size());
                        (p * g).sum(Kind::Float)
                    })
                    .collect::<Vec<_>>(),
                0,
            )
            .sum(Kind::Float);
            loss.backward();
        }
        fused.update(&a).unwrap();
        reference.update_reference(&b).unwrap();
    }
    let mut error = 0.0f64;
    let mut parameter_error = 0.0f64;
    for (name, parameter) in a.variables() {
        parameter_error = parameter_error.max(
            (parameter - &b.variables()[&name])
                .abs()
                .max()
                .double_value(&[]),
        );
        error = error.max(parameter_error);
        let (m, v) = &fused.moments[&name];
        let (rm, rv) = &reference.moments[&name];
        error = error.max((m - rm).abs().max().double_value(&[]));
        error = error.max((v - rv).abs().max().double_value(&[]));
    }
    // Fused multiply-add and bias correction use CUDA FP32 arithmetic; the
    // scalar reference forms bias corrections in f64. Bound accumulated drift.
    assert!(error < 1e-6, "fused AdamW parity error {error}");
    eprintln!(
        "CUDA fused/reference AdamW maximum parameter/moment error after 50 steps: {error:e}; parameters {parameter_error:e}; tolerance 1e-6"
    );
}

#[test]
fn dense_ranking_matches_reference_values_and_gradients() {
    for device in [Device::Cpu, Device::Cuda(0)] {
        if matches!(device, Device::Cuda(_)) && !tch::Cuda::is_available() {
            continue;
        }
        let raw = (0..32)
            .map(|i| ((i * 13 % 23) as f32 - 11.0) * 0.17)
            .collect::<Vec<_>>();
        let values = (0..32)
            .map(|i| ((i * 7 % 11) as f64) / 10.0)
            .collect::<Vec<_>>();
        let percentiles = (0..32).map(|i| (i % 8) as f32 / 7.0).collect::<Vec<_>>();
        let groups = (0..32)
            .map(|i| format!("group-{}", i / 8))
            .collect::<Vec<_>>();
        let mut expected_loss = 0.0;
        let mut expected_gradient = vec![0f64; 32];
        let mut total_weight = 0.0;
        for a in 0..32 {
            for b in a + 1..32 {
                if groups[a] != groups[b] || values[a] == values[b] {
                    continue;
                }
                let (w, l) = if values[a] > values[b] {
                    (a, b)
                } else {
                    (b, a)
                };
                let weight = (values[a] - values[b]).abs()
                    * if percentiles[a].max(percentiles[b]) >= 0.9 {
                        2.0
                    } else {
                        1.0
                    };
                let difference = (raw[l] - raw[w]) as f64;
                expected_loss += weight * difference.exp().ln_1p();
                let derivative = weight / (1.0 + (-difference).exp());
                expected_gradient[l] += derivative;
                expected_gradient[w] -= derivative;
                total_weight += weight;
            }
        }
        expected_loss /= total_weight;
        let expected_gradient = Tensor::from_slice(
            &expected_gradient
                .iter()
                .map(|v| (v / total_weight) as f32)
                .collect::<Vec<_>>(),
        )
        .to_device(device);
        let mut previous: Option<Tensor> = None;
        for _ in 0..10 {
            let reward = Tensor::from_slice(&raw)
                .to_device(device)
                .set_requires_grad(true);
            let (bt, _) = ranking(&reward, &values, &percentiles, &groups).unwrap();
            assert!((bt.double_value(&[]) - expected_loss).abs() < 2e-6);
            bt.backward();
            let gradient = reward.grad();
            assert!(
                (&gradient - &expected_gradient)
                    .abs()
                    .max()
                    .double_value(&[])
                    < 2e-7
            );
            if let Some(previous) = previous {
                assert_eq!(
                    (&gradient - previous).abs().max().double_value(&[]),
                    0.0,
                    "dense pairwise gradient is nondeterministic"
                );
            }
            previous = Some(gradient.detach().copy());
        }
    }
}

#[test]
fn reward_policy_continuation_reproduces_parameters_exactly() {
    if !tch::Cuda::is_available() {
        return;
    }
    tch::set_num_threads(2);
    let device = Device::Cuda(0);
    let mut config = Config::regression(32);
    config.width = 32;
    config.ffn = 128;
    config.layers = 2;
    config.dropout = 0.0;
    let a = nn::VarStore::new(device);
    let first = Policy::new(a.root(), config.clone()).unwrap();
    let mut optimizer = AdamW::new(0.001, 0.01);
    let dataset = Tensor::ones([32, 1536], (Kind::Float, device));
    let target = Tensor::ones([32, 512], (Kind::Float, device));
    let tokens = Tensor::from_slice(
        &(0..32)
            .flat_map(|i| [1i64, 3, 4, 5, 6 + (i % 8) as i64, 14, 15, 2])
            .collect::<Vec<_>>(),
    )
    .reshape([32, 8])
    .to_device(device);
    let pad = tokens.ne(0);
    let values = (0..32).map(|i| (i % 8) as f64 / 7.0).collect::<Vec<_>>();
    let pct = values.iter().map(|&x| x as f32).collect::<Vec<_>>();
    let groups = (0..32)
        .map(|i| format!("cut-{}", i / 8))
        .collect::<Vec<_>>();
    let objective = |policy: &Policy| {
        let _guard = Bf16::enter();
        let (_, rewards, _, _, ablation) = policy
            .forward(&dataset, &target, &tokens, &pad, false)
            .unwrap();
        let (bt, lw) = ranking(
            &rewards.mean_dim(&[1i64][..], false, Kind::Float),
            &values,
            &pct,
            &groups,
        )
        .unwrap();
        let (abt, _) = ranking(&ablation, &values, &pct, &groups).unwrap();
        bt + lw * 0.1 + abt * 0.1
    };
    for _ in 0..10 {
        objective(&first).backward();
        optimizer.update(&a).unwrap();
    }
    let tmp = tempfile::tempdir().unwrap();
    a.save(tmp.path().join("model.ot")).unwrap();
    optimizer.save(&tmp.path().join("optimizer.ot")).unwrap();
    let mut b = nn::VarStore::new(device);
    let second = Policy::new(b.root(), config).unwrap();
    b.load(tmp.path().join("model.ot")).unwrap();
    let mut resumed = AdamW::load(&tmp.path().join("optimizer.ot"), device).unwrap();
    for _ in 0..5 {
        objective(&first).backward();
        optimizer.update(&a).unwrap();
        objective(&second).backward();
        resumed.update(&b).unwrap();
    }
    for (name, parameter) in a.variables() {
        assert_eq!(
            (parameter - &b.variables()[&name])
                .abs()
                .max()
                .double_value(&[]),
            0.0,
            "reward continuation drifted in {name}"
        );
    }
}

#[test]
#[ignore = "explicit full-policy GPU optimizer benchmark"]
fn benchmark_cuda_adamw() {
    assert!(tch::Cuda::is_available());
    tch::set_num_threads(2);
    let device = Device::Cuda(0);
    let vs = nn::VarStore::new(device);
    let _policy = Policy::new(vs.root(), Config::regression(401)).unwrap();
    Tensor::stack(
        &vs.variables()
            .values()
            .map(|p| p.sum(Kind::Float))
            .collect::<Vec<_>>(),
        0,
    )
    .sum(Kind::Float)
    .backward();
    let parameters = vs.variables().len();
    let count = vs.variables().values().map(|p| p.numel()).sum::<usize>();
    let mut results = Vec::new();
    for fused in [false, true] {
        let mut opt = AdamW::new(0.0001, 0.01);
        let mut elapsed = 0.0;
        for step in 0..25 {
            tch::no_grad(|| {
                for p in vs.variables().values() {
                    let _ = p.grad().fill_(0.01);
                }
            });
            tch::Cuda::synchronize(0);
            let start = std::time::Instant::now();
            if fused {
                opt.update(&vs).unwrap();
            } else {
                opt.update_reference(&vs).unwrap();
            }
            tch::Cuda::synchronize(0);
            if step >= 5 {
                elapsed += start.elapsed().as_secs_f64();
            }
        }
        results.push(elapsed / 20.0);
    }
    eprintln!("optimizer benchmark: {parameters} tensors, {count} FP32 master parameters, reference {:.3} ms, fused {:.3} ms, {:.2}x speedup",
        results[0]*1000.0, results[1]*1000.0, results[0]/results[1]);
}
