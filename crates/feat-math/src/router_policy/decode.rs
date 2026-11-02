use super::*;

fn decode_candidate(router: &[f64], cfg: &RouterPolicyConfig, sample: usize) -> (DslExpr, f64) {
    let mut rng = Pcg32::new(
        cfg.seed ^ mix64(sample as u64) ^ router_seed(router),
        0x524f_5554_4552,
    );
    let max_transforms = cfg.max_transforms.clamp(1, 8);
    let (stage_idx, mut score) = select_index(
        router,
        &mut rng,
        sample,
        0,
        "stage_count",
        max_transforms,
        |i| (i + 1).to_string(),
    );
    let transforms = stage_idx + 1;
    let mut stages = Vec::with_capacity(transforms + 1);

    for step in 0..transforms {
        let (idx, s) = select_index(
            router,
            &mut rng,
            sample,
            step + 1,
            "transform",
            TRANSFORM_OPS.len(),
            |i| TRANSFORM_OPS[i].to_string(),
        );
        score += s;
        let op = TRANSFORM_OPS[idx];
        let (mut stage, param_score) =
            transform_stage(router, &mut rng, sample, step + 1, op, cfg.seed);
        score += param_score;
        let (append, append_score) = select_bool(router, &mut rng, sample, step + 1, "append");
        if append {
            stage.params.insert("append".into(), ParamValue::Bool(true));
        }
        score += append_score * 0.25;
        stages.push(stage);
    }

    let (head_idx, head_score) = select_index(
        router,
        &mut rng,
        sample,
        transforms + 1,
        "head",
        HEAD_OPS.len(),
        |i| HEAD_OPS[i].to_string(),
    );
    let head_op = HEAD_OPS[head_idx];
    let (head, param_score) =
        head_stage(router, &mut rng, sample, transforms + 1, head_op, cfg.seed);
    score += head_score + param_score;
    score -= 0.03 * transforms as f64;
    stages.push(head);
    (DslExpr { stages }, score)
}

fn transform_stage(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    op: &str,
    seed: u64,
) -> (Stage, f64) {
    let mut params = BTreeMap::new();
    let mut score = 0.0;
    match op {
        "quantile_bins" => {
            let (v, s) = choose_i64(router, rng, sample, step, "bins", &[4, 8, 16, 32]);
            params.insert("bins".into(), ParamValue::Int(v));
            score += s;
        }
        "bin_onehot" => {
            let (v, s) = choose_i64(router, rng, sample, step, "bins", &[4, 8, 12, 16]);
            params.insert("bins".into(), ParamValue::Int(v));
            score += s;
        }
        "hinge" => {
            let (v, s) = choose_i64(router, rng, sample, step, "knots", &[2, 4, 8]);
            params.insert("knots".into(), ParamValue::Int(v));
            score += s;
        }
        "winsor_clip" => {
            let (v, s) = choose_f64(
                router,
                rng,
                sample,
                step,
                "q",
                &[0.01, 0.02, 0.05, 0.1, 0.15],
            );
            params.insert("q".into(), ParamValue::Float(v));
            score += s;
        }
        "cheb" => {
            let (v, s) = choose_i64(router, rng, sample, step, "degree", &[2, 3, 4, 6, 8]);
            params.insert("degree".into(), ParamValue::Int(v));
            score += s;
        }
        "rff_cos" | "rff_sin" => {
            let (n, sn) = choose_i64(router, rng, sample, step, "n", &[16, 32, 64, 96, 128]);
            let (gamma, sg) = choose_f64(
                router,
                rng,
                sample,
                step,
                "gamma",
                &[0.25, 0.5, 1.0, 2.0, 4.0],
            );
            params.insert("n".into(), ParamValue::Int(n));
            params.insert("gamma".into(), ParamValue::Float(gamma));
            params.insert(
                "seed".into(),
                ParamValue::Int(kernel_seed(seed, sample, step, op)),
            );
            score += sn + sg;
        }
        "rand_proj" => {
            let (k, s) = choose_i64(router, rng, sample, step, "k", &[8, 16, 32, 64, 128]);
            params.insert("k".into(), ParamValue::Int(k));
            params.insert(
                "seed".into(),
                ParamValue::Int(kernel_seed(seed, sample, step, op)),
            );
            score += s;
        }
        "oblique_hinge" => {
            let (n, s) = choose_i64(router, rng, sample, step, "n", &[16, 32, 64, 96]);
            params.insert("n".into(), ParamValue::Int(n));
            params.insert(
                "seed".into(),
                ParamValue::Int(kernel_seed(seed, sample, step, op)),
            );
            score += s;
        }
        "pair_prod" => {
            let (topk, s) = choose_i64(router, rng, sample, step, "topk", &[4, 8, 12, 16, 24]);
            params.insert("topk".into(), ParamValue::Int(topk));
            score += s;
        }
        "target_bin_encode" => {
            let (bins, sb) = choose_i64(router, rng, sample, step, "bins", &[4, 8, 16, 32]);
            let (prior, sp) =
                choose_f64(router, rng, sample, step, "prior", &[1.0, 5.0, 10.0, 20.0]);
            params.insert("bins".into(), ParamValue::Int(bins));
            params.insert("prior".into(), ParamValue::Float(prior));
            score += sb + sp;
        }
        _ => {}
    }
    (
        Stage {
            op: op.to_string(),
            params,
        },
        score,
    )
}

fn head_stage(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    op: &str,
    seed: u64,
) -> (Stage, f64) {
    let mut params = BTreeMap::new();
    let mut score = 0.0;
    match op {
        "ridge" => {
            let (v, s) = choose_f64(
                router,
                rng,
                sample,
                step,
                "lambda_frac",
                &[0.005, 0.01, 0.02, 0.05, 0.1],
            );
            params.insert("lambda_frac".into(), ParamValue::Float(v));
            score += s;
        }
        "elasticnet" => {
            let (alpha, sa) = choose_f64(
                router,
                rng,
                sample,
                step,
                "alpha",
                &[0.0001, 0.001, 0.005, 0.01, 0.05],
            );
            let (l1, sl) = choose_f64(
                router,
                rng,
                sample,
                step,
                "l1_ratio",
                &[0.1, 0.3, 0.5, 0.7, 0.9],
            );
            let (sweeps, ss) = choose_i64(router, rng, sample, step, "sweeps", &[40, 60, 100]);
            params.insert("alpha".into(), ParamValue::Float(alpha));
            params.insert("l1_ratio".into(), ParamValue::Float(l1));
            params.insert("sweeps".into(), ParamValue::Int(sweeps));
            score += sa + sl + ss;
        }
        "histgbm" => {
            let (rounds, sr) =
                choose_i64(router, rng, sample, step, "rounds", &[80, 120, 200, 300]);
            let (depth, sd) = choose_i64(router, rng, sample, step, "depth", &[2, 3, 4, 6]);
            let (lr, sl) = choose_f64(router, rng, sample, step, "lr", &[0.03, 0.05, 0.1, 0.15]);
            let (min_leaf, sm) = choose_i64(router, rng, sample, step, "min_leaf", &[5, 10, 20]);
            params.insert("rounds".into(), ParamValue::Int(rounds));
            params.insert("depth".into(), ParamValue::Int(depth));
            params.insert("lr".into(), ParamValue::Float(lr));
            params.insert("min_leaf".into(), ParamValue::Int(min_leaf));
            score += sr + sd + sl + sm;
        }
        "histgbm2" => {
            let (leaves, sv) = choose_i64(router, rng, sample, step, "leaves", &[15, 31, 63, 127]);
            let (rounds, sr) = choose_i64(router, rng, sample, step, "rounds", &[300, 500, 1000]);
            let (lr, sl) = choose_f64(router, rng, sample, step, "lr", &[0.01, 0.03, 0.05, 0.08]);
            let (bags, sb) = choose_i64(router, rng, sample, step, "bags", &[1, 3, 5]);
            let (row, sw) = choose_f64(
                router,
                rng,
                sample,
                step,
                "row_subsample",
                &[0.6, 0.7, 0.85, 1.0],
            );
            let (col, sc) =
                choose_f64(router, rng, sample, step, "col_subsample", &[0.6, 0.8, 1.0]);
            let (min_leaf, sm) =
                choose_i64(router, rng, sample, step, "min_leaf", &[1, 3, 5, 10, 20]);
            params.insert("leaves".into(), ParamValue::Int(leaves));
            params.insert("rounds".into(), ParamValue::Int(rounds));
            params.insert("lr".into(), ParamValue::Float(lr));
            params.insert("bags".into(), ParamValue::Int(bags));
            params.insert("row_subsample".into(), ParamValue::Float(row));
            params.insert("col_subsample".into(), ParamValue::Float(col));
            params.insert("min_leaf".into(), ParamValue::Int(min_leaf));
            score += sv + sr + sl + sb + sw + sc + sm;
        }
        "krr" => {
            let (m, s) = choose_i64(router, rng, sample, step, "m", &[100, 200, 400, 800]);
            params.insert("m".into(), ParamValue::Int(m));
            score += s;
        }
        "gpr" => {
            let (m, sm) = choose_i64(router, rng, sample, step, "m", &[100, 200, 300, 500]);
            let (steps, ss) = choose_i64(router, rng, sample, step, "steps", &[20, 40, 60]);
            let kernels = ["rbf", "rbf+linear", "rq+linear", "rbf*periodic", "linear"];
            let (kernel_idx, sk) =
                select_index(router, rng, sample, step, "kernel", kernels.len(), |i| {
                    kernels[i].to_string()
                });
            params.insert("m".into(), ParamValue::Int(m));
            params.insert("steps".into(), ParamValue::Int(steps));
            if kernels[kernel_idx] != "rbf" {
                params.insert(
                    "kernel".into(),
                    ParamValue::Str(kernels[kernel_idx].to_string()),
                );
            }
            score += sm + ss + sk;
        }
        "ksmooth" => {
            let (k, sk) = choose_i64(router, rng, sample, step, "k", &[8, 16, 32, 64, 128]);
            let (gamma, sg) = choose_f64(router, rng, sample, step, "gamma", &[0.5, 1.0, 2.0, 4.0]);
            params.insert("k".into(), ParamValue::Int(k));
            params.insert("gamma".into(), ParamValue::Float(gamma));
            score += sk + sg;
        }
        "stack" => {}
        _ => {
            let unused_seed_score = kernel_seed(seed, sample, step, "head") as f64;
            score += unused_seed_score * 0.0;
        }
    }
    (
        Stage {
            op: op.to_string(),
            params,
        },
        score,
    )
}

fn select_bool(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    namespace: &str,
) -> (bool, f64) {
    let labels = ["false", "true"];
    let (idx, score) = select_index(router, rng, sample, step, namespace, labels.len(), |i| {
        labels[i].to_string()
    });
    (idx == 1, score)
}

fn choose_i64(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    namespace: &str,
    choices: &[i64],
) -> (i64, f64) {
    let (idx, score) = select_index(router, rng, sample, step, namespace, choices.len(), |i| {
        choices[i].to_string()
    });
    (choices[idx], score)
}

fn choose_f64(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    namespace: &str,
    choices: &[f64],
) -> (f64, f64) {
    let (idx, score) = select_index(router, rng, sample, step, namespace, choices.len(), |i| {
        format!("{:.8}", choices[i])
    });
    (choices[idx], score)
}

fn select_index<F>(
    router: &[f64],
    rng: &mut Pcg32,
    sample: usize,
    step: usize,
    namespace: &str,
    len: usize,
    label_at: F,
) -> (usize, f64)
where
    F: Fn(usize) -> String,
{
    debug_assert!(len > 0);
    let mut logits = Vec::with_capacity(len);
    for idx in 0..len {
        let label = label_at(idx);
        logits.push(action_logit(router, sample, step, namespace, idx, &label));
    }
    let max = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let mut weights = Vec::with_capacity(len);
    let mut total = 0.0;
    for logit in &logits {
        let w = ((*logit - max) / 0.85).exp().max(1e-12);
        total += w;
        weights.push(w);
    }
    let mut cursor = rng.next_f64() * total;
    for (idx, weight) in weights.iter().enumerate() {
        if cursor <= *weight {
            return (idx, logits[idx]);
        }
        cursor -= *weight;
    }
    let last = len - 1;
    (last, logits[last])
}

fn action_logit(
    router: &[f64],
    sample: usize,
    step: usize,
    namespace: &str,
    action: usize,
    label: &str,
) -> f64 {
    let mut h = 0x9e37_79b9_7f4a_7c15
        ^ mix64(sample as u64)
        ^ mix64((step as u64) << 32)
        ^ mix64(action as u64);
    for b in namespace.bytes().chain(label.bytes()) {
        h = mix64(h ^ b as u64);
    }
    let mut acc = 0.0;
    for k in 0..8 {
        let hk = mix64(h ^ ((k as u64) << 48));
        let idx = (hk as usize) % ROUTER_EMBEDDING_DIMS;
        let weight = (((hk >> 16) & 0xffff) as f64 / 32767.5) - 1.0;
        acc += router[idx] * weight;
    }
    acc / 8.0 + (((h >> 40) & 0xff) as f64 / 1024.0)
}
