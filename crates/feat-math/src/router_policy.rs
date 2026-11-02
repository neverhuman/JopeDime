//! Router-conditioned proposer for canonical pipeline seed genomes.
//!
//! This is the v0 router-to-math-kernel interface: a 128-d dataset vector drives
//! a constrained grammar decoder over the emittable feat-math pipeline subset.
//! The evaluator remains authoritative; every proposed string is parsed,
//! canonicalized, and forced through the existing emitter before it is returned.

use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::transpile::{emit, EmitOptions};
use crate::{GenomeIr, Pcg32};
use anyhow::{bail, Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};

pub const ROUTER_EMBEDDING_DIMS: usize = 128;

const POLICY_NAME: &str = "router_to_math_kernel_v0";
const TRANSFORM_OPS: [&str; 18] = [
    "standardize",
    "robust_scale",
    "rank",
    "blom",
    "quantile_bins",
    "bin_onehot",
    "hinge",
    "winsor_clip",
    "cheb",
    "poly_interact",
    "rff_cos",
    "rff_sin",
    "rand_proj",
    "oblique_hinge",
    "row_stats",
    "pair_prod",
    "target_bin_encode",
    "sigmoid",
];
const HEAD_OPS: [&str; 8] = [
    "ridge",
    "elasticnet",
    "histgbm",
    "histgbm2",
    "krr",
    "gpr",
    "ksmooth",
    "stack",
];

#[derive(Clone, Debug, Serialize)]
pub struct RouterPolicyConfig {
    pub samples: usize,
    pub top_n: usize,
    pub max_transforms: usize,
    pub seed: u64,
}

impl Default for RouterPolicyConfig {
    fn default() -> Self {
        Self {
            samples: 256,
            top_n: 64,
            max_transforms: 6,
            seed: 20260616,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RouterPolicyGenome {
    pub rank: usize,
    pub genome: String,
    pub genome_hash: String,
    pub policy_score: f64,
    pub transforms: usize,
    pub head: String,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct RouterPolicyStats {
    pub router_dims: usize,
    pub requested_samples: usize,
    pub decoded: usize,
    pub valid: usize,
    pub parse_failed: usize,
    pub unsupported: usize,
    pub duplicate: usize,
    pub returned: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct RouterPolicyReport {
    pub schema_version: u32,
    pub policy: &'static str,
    pub router_hash: String,
    pub config: RouterPolicyConfig,
    pub stats: RouterPolicyStats,
    pub genomes: Vec<RouterPolicyGenome>,
}

#[derive(Clone, Debug)]
struct Candidate {
    genome: String,
    genome_hash: String,
    policy_score: f64,
    transforms: usize,
    head: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Reject {
    Parse,
    Unsupported,
}

pub fn parse_router_embedding_text(text: &str) -> Result<Vec<f64>> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        bail!("router embedding is empty");
    }

    if trimmed.starts_with('[') || trimmed.starts_with('{') {
        let value: serde_json::Value =
            serde_json::from_str(trimmed).context("parse router embedding JSON")?;
        let values = embedding_from_json(&value).context(
            "router embedding JSON must be an array, or an object containing router/embedding/vector",
        )?;
        validate_router_embedding(&values)?;
        return Ok(values);
    }

    let mut normalized = String::with_capacity(trimmed.len());
    for ch in trimmed.chars() {
        if ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E') {
            normalized.push(ch);
        } else {
            normalized.push(' ');
        }
    }
    let values: Vec<f64> = normalized
        .split_whitespace()
        .map(|tok| {
            tok.parse::<f64>()
                .with_context(|| format!("invalid router embedding value '{tok}'"))
        })
        .collect::<Result<_>>()?;
    validate_router_embedding(&values)?;
    Ok(values)
}

pub fn validate_router_embedding(router: &[f64]) -> Result<()> {
    if router.len() != ROUTER_EMBEDDING_DIMS {
        bail!(
            "router embedding must have {ROUTER_EMBEDDING_DIMS} dimensions, got {}",
            router.len()
        );
    }
    if router.iter().any(|v| !v.is_finite()) {
        bail!("router embedding contains non-finite values");
    }
    Ok(())
}

pub fn propose_router_genomes(
    router: &[f64],
    cfg: &RouterPolicyConfig,
) -> Result<RouterPolicyReport> {
    validate_router_embedding(router)?;
    validate_config(cfg)?;
    let router_norm = normalize_router(router);
    let router_hash = router_hash(router);
    let mut stats = RouterPolicyStats {
        router_dims: router.len(),
        requested_samples: cfg.samples,
        ..RouterPolicyStats::default()
    };
    let mut seen = HashSet::new();
    let mut candidates = Vec::new();

    for sample in 0..cfg.samples {
        stats.decoded += 1;
        let (dsl, policy_score) = decode_candidate(&router_norm, cfg, sample);
        let transforms = dsl.stages.len().saturating_sub(1);
        let head = dsl
            .stages
            .last()
            .map(|s| s.op.clone())
            .unwrap_or_else(|| "ridge".to_string());
        let source = GenomeIr::Pipeline(dsl).canonical();
        match validate_supported_pipeline(&source) {
            Ok(canonical) => {
                if !seen.insert(canonical.clone()) {
                    stats.duplicate += 1;
                    continue;
                }
                stats.valid += 1;
                candidates.push(Candidate {
                    genome_hash: sha256_hex(&canonical),
                    genome: canonical,
                    policy_score: finite(policy_score),
                    transforms,
                    head,
                });
            }
            Err(Reject::Parse) => stats.parse_failed += 1,
            Err(Reject::Unsupported) => stats.unsupported += 1,
        }
    }

    if candidates.is_empty() {
        bail!("router policy produced no valid supported programs");
    }

    candidates.sort_by(|a, b| {
        b.policy_score
            .total_cmp(&a.policy_score)
            .then_with(|| a.genome_hash.cmp(&b.genome_hash))
    });
    candidates.truncate(cfg.top_n);

    let genomes = candidates
        .into_iter()
        .enumerate()
        .map(|(i, c)| RouterPolicyGenome {
            rank: i + 1,
            genome: c.genome,
            genome_hash: c.genome_hash,
            policy_score: c.policy_score,
            transforms: c.transforms,
            head: c.head,
        })
        .collect::<Vec<_>>();
    stats.returned = genomes.len();

    Ok(RouterPolicyReport {
        schema_version: 1,
        policy: POLICY_NAME,
        router_hash,
        config: cfg.clone(),
        stats,
        genomes,
    })
}

fn validate_config(cfg: &RouterPolicyConfig) -> Result<()> {
    if cfg.samples == 0 {
        bail!("router policy samples must be at least 1");
    }
    if cfg.top_n == 0 {
        bail!("router policy top_n must be at least 1");
    }
    if cfg.max_transforms == 0 {
        bail!("router policy max_transforms must be at least 1");
    }
    if cfg.max_transforms > 8 {
        bail!("router policy max_transforms must be at most 8");
    }
    Ok(())
}

fn embedding_from_json(value: &serde_json::Value) -> Option<Vec<f64>> {
    if let Some(values) = json_number_array(value) {
        return Some(values);
    }
    let obj = value.as_object()?;
    for key in [
        "router",
        "router128",
        "embedding",
        "dataset_vector",
        "vector",
        "vec",
        "features",
    ] {
        if let Some(values) = obj.get(key).and_then(json_number_array) {
            return Some(values);
        }
    }
    None
}

fn json_number_array(value: &serde_json::Value) -> Option<Vec<f64>> {
    let array = value.as_array()?;
    let mut out = Vec::with_capacity(array.len());
    for item in array {
        out.push(item.as_f64()?);
    }
    Some(out)
}

fn validate_supported_pipeline(source: &str) -> std::result::Result<String, Reject> {
    let parsed = GenomeIr::parse(source).map_err(|_| Reject::Parse)?;
    if !matches!(parsed, GenomeIr::Pipeline(_)) {
        return Err(Reject::Parse);
    }
    emit(&parsed, &EmitOptions::default()).map_err(|_| Reject::Unsupported)?;
    Ok(parsed.canonical())
}

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

fn normalize_router(router: &[f64]) -> Vec<f64> {
    let mean = router.iter().sum::<f64>() / router.len().max(1) as f64;
    let var = router
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / router.len().max(1) as f64;
    let scale = var.sqrt().max(1e-6);
    router
        .iter()
        .map(|v| ((*v - mean) / scale).clamp(-4.0, 4.0))
        .collect()
}

fn router_hash(router: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for v in router {
        hasher.update(v.to_le_bytes());
    }
    hex_digest(hasher.finalize())
}

fn router_seed(router: &[f64]) -> u64 {
    let mut h = 0x726f_7574_6572_3132;
    for v in router {
        h = mix64(h ^ v.to_bits());
    }
    h
}

fn kernel_seed(seed: u64, sample: usize, step: usize, label: &str) -> i64 {
    let mut h = mix64(seed ^ mix64(sample as u64) ^ ((step as u64) << 24));
    for b in label.bytes() {
        h = mix64(h ^ b as u64);
    }
    (h % 9999 + 1) as i64
}

fn sha256_hex(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hex_digest(hasher.finalize())
}

fn hex_digest(digest: impl AsRef<[u8]>) -> String {
    let bytes = digest.as_ref();
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn finite(v: f64) -> f64 {
    if v.is_finite() {
        v
    } else {
        0.0
    }
}

fn mix64(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn router() -> Vec<f64> {
        (0..ROUTER_EMBEDDING_DIMS)
            .map(|i| ((i as f64) * 0.17).sin())
            .collect()
    }

    #[test]
    fn parses_text_and_json_embeddings() {
        let csv = (0..ROUTER_EMBEDDING_DIMS)
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(",");
        assert_eq!(
            parse_router_embedding_text(&csv).unwrap().len(),
            ROUTER_EMBEDDING_DIMS
        );

        let json = serde_json::json!({"embedding": router()}).to_string();
        assert_eq!(
            parse_router_embedding_text(&json).unwrap().len(),
            ROUTER_EMBEDDING_DIMS
        );
    }

    #[test]
    fn rejects_wrong_embedding_shape() {
        let err = parse_router_embedding_text("[1, 2, 3]").unwrap_err();
        assert!(format!("{err:#}").contains("128"));
    }

    #[test]
    fn proposal_is_deterministic_and_supported() {
        let cfg = RouterPolicyConfig {
            samples: 32,
            top_n: 8,
            max_transforms: 4,
            seed: 7,
        };
        let a = propose_router_genomes(&router(), &cfg).unwrap();
        let b = propose_router_genomes(&router(), &cfg).unwrap();
        assert_eq!(
            a.genomes
                .iter()
                .map(|g| g.genome.as_str())
                .collect::<Vec<_>>(),
            b.genomes
                .iter()
                .map(|g| g.genome.as_str())
                .collect::<Vec<_>>()
        );
        assert!(!a.genomes.is_empty());
        assert_eq!(a.stats.parse_failed + a.stats.unsupported, 0);
        for genome in &a.genomes {
            let parsed = GenomeIr::parse(&genome.genome).unwrap();
            assert_eq!(parsed.canonical(), genome.genome);
            emit(&parsed, &EmitOptions::default()).unwrap();
        }
    }
}
