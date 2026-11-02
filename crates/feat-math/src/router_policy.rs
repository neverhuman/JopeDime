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

mod decode;
mod hash;
mod parsing;

use decode::*;
use hash::*;
use parsing::*;

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
