use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::jope_generalized_gp::{
    raw_sha256, GenerationZeroV1, GenomeAncestryV1, GenomeOriginV1, GenomeV1, GpTaskV1,
    PopulationMemberV1, ProposalFreezeV1,
};
use crate::search_v3::{canonical_sha256, validate_sha256};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};

const ONE_SIDED_90_Z: f64 = 1.281_551_565_544_600_4;
const LINEAGE_EXPLORATION: f64 = 0.10;

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    (values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

/// One-sided 90% LCB with the two frozen zero-uplift prior observations.
pub fn confidence_score(values: &[f64]) -> Result<f64, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("confidence score cannot consume nonfinite values".into());
    }
    let mut supported = values.to_vec();
    supported.extend([0.0, 0.0]);
    Ok(mean(&supported) - ONE_SIDED_90_Z * sample_sd(&supported) / (supported.len() as f64).sqrt())
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

pub(crate) fn semantic_mutation(parent: &GenomeV1, seed: u64) -> Option<GenomeV1> {
    let source = mutate_canonical_pipeline_v2(&parent.canonical_genome, seed, 6, true).ok()?;
    let child = GenomeV1::from_canonical(GpTaskV1::Regression, &source).ok()?;
    (child.semantic_sha256 != parent.semantic_sha256).then_some(child)
}

pub(crate) fn semantic_crossover(left: &GenomeV1, right: &GenomeV1, seed: u64) -> Option<GenomeV1> {
    if left.semantic_sha256 == right.semantic_sha256 {
        return None;
    }
    let source =
        crossover_canonical_pipelines(&left.canonical_genome, &right.canonical_genome, seed, 6)
            .ok()?;
    let child = GenomeV1::from_canonical(GpTaskV1::Regression, &source).ok()?;
    (child.semantic_sha256 != left.semantic_sha256
        && child.semantic_sha256 != right.semantic_sha256)
        .then_some(child)
}

pub(crate) fn regression_head_superfamily(head_family: &str) -> u8 {
    match head_family {
        "ridge" => 0,
        "elasticnet" => 1,
        "krr" => 2,
        "gpr" => 3,
        // A future checkpoint schema remains deterministic, but the frozen
        // Best5 checkpoint uses the four explicit regression families above.
        other => u8::from_str_radix(&raw_sha256(other.as_bytes())[..2], 16).unwrap() % 4,
    }
}

mod anchors;
mod control;
mod lineage;
mod map_elites;
mod residual;
mod surrogate;

pub use anchors::*;
pub use control::*;
pub use lineage::*;
pub use map_elites::*;
pub use residual::*;
pub use surrogate::*;
