//! A separately identified, order-aware structural ranking candidate.
//!
//! It consumes the declared token sequence, produces one scalar acquisition score, and never
//! reads or updates actor parameters. Uncertainty is absent unless a held-out calibration record
//! is explicitly supplied; no synthetic uncertainty head is created.

use anyhow::{ensure, Result};
use candle_core::{Device, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::linear;
use crate::{is_sha256_hex, sha256_hex};

pub const STRUCTURED_RANK_SCHEMA: &str = "jope.structured-rank.v1";
pub const FEATURE_DIM: usize = 128;
pub const WIDTH: usize = 128;
pub const LAYERS: usize = 2;
pub const MAX_SUPPORTED_TOKENS: usize = 128;
const HALF_DIM: usize = FEATURE_DIM / 2;

/// Evidence about a held-out residual distribution. This is not a learned uncertainty head.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeCalibration {
    pub method: String,
    pub heldout_pool_sha256: String,
    pub examples: usize,
    pub absolute_error_p90: f32,
}

impl OutcomeCalibration {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.method == "heldout_absolute_error_p90_v1",
            "unsupported critic calibration method"
        );
        ensure!(
            is_sha256_hex(&self.heldout_pool_sha256),
            "invalid calibration pool digest"
        );
        ensure!(
            self.examples > 0,
            "critic calibration needs held-out examples"
        );
        ensure!(
            self.absolute_error_p90.is_finite() && self.absolute_error_p90 >= 0.0,
            "invalid held-out absolute error"
        );
        Ok(())
    }
}

/// Candidate ABI and frozen comparison identities. The manifest is independent of the actor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CriticManifest {
    pub schema: String,
    pub candidate_id: String,
    pub actor_checkpoint_sha256: String,
    pub same_pool_sha256: String,
    pub critic_weights_sha256: String,
    pub vocab_size: u32,
    pub token_order: String,
    pub feature_schema: String,
    pub normalizer: String,
    pub max_tokens: usize,
    pub width: usize,
    pub layers: usize,
    pub backend: String,
    pub precision: String,
    pub calibration: Option<OutcomeCalibration>,
}

impl CriticManifest {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == STRUCTURED_RANK_SCHEMA,
            "unsupported structured critic schema"
        );
        ensure!(
            !self.candidate_id.is_empty(),
            "critic candidate ID is empty"
        );
        ensure!(
            is_sha256_hex(&self.actor_checkpoint_sha256),
            "invalid frozen actor digest"
        );
        ensure!(
            is_sha256_hex(&self.same_pool_sha256),
            "invalid comparison pool digest"
        );
        ensure!(
            is_sha256_hex(&self.critic_weights_sha256),
            "invalid critic weight digest"
        );
        ensure!(self.vocab_size > 2, "invalid critic vocabulary size");
        ensure!(
            self.token_order == "declared_native_token_order_v1",
            "unsupported token order"
        );
        ensure!(
            self.feature_schema == "unigram-directed-bigram-128-v1",
            "unsupported structural features"
        );
        ensure!(
            self.normalizer == "sqrt_length_l2_v1",
            "unsupported critic normalizer"
        );
        ensure!(
            self.max_tokens > 0 && self.max_tokens <= MAX_SUPPORTED_TOKENS,
            "unsupported critic maximum sequence length"
        );
        ensure!(
            self.width == WIDTH && self.layers == LAYERS,
            "unsupported critic architecture"
        );
        ensure!(self.backend == "candle_cpu", "unsupported critic backend");
        ensure!(self.precision == "f32", "unsupported critic precision");
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        Ok(())
    }
}

/// Flat, row-major weights for two 128-wide hidden layers and one scalar output.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CriticParameters {
    pub first_weights: Vec<f32>,
    pub first_bias: Vec<f32>,
    pub second_weights: Vec<f32>,
    pub second_bias: Vec<f32>,
    pub output_weights: Vec<f32>,
    pub output_bias: f32,
}

impl CriticParameters {
    pub fn sha256(&self) -> String {
        let mut bytes = Vec::with_capacity((2 * WIDTH * FEATURE_DIM + 2 * WIDTH + WIDTH) * 4 + 4);
        for value in self
            .first_weights
            .iter()
            .chain(&self.first_bias)
            .chain(&self.second_weights)
            .chain(&self.second_bias)
            .chain(&self.output_weights)
            .chain(std::iter::once(&self.output_bias))
        {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        sha256_hex(&bytes)
    }

    fn validate(&self) -> Result<()> {
        ensure!(
            self.first_weights.len() == WIDTH * FEATURE_DIM,
            "critic first layer shape mismatch"
        );
        ensure!(
            self.first_bias.len() == WIDTH,
            "critic first bias shape mismatch"
        );
        ensure!(
            self.second_weights.len() == WIDTH * WIDTH,
            "critic second layer shape mismatch"
        );
        ensure!(
            self.second_bias.len() == WIDTH,
            "critic second bias shape mismatch"
        );
        ensure!(
            self.output_weights.len() == WIDTH,
            "critic output shape mismatch"
        );
        ensure!(
            self.first_weights
                .iter()
                .chain(&self.first_bias)
                .chain(&self.second_weights)
                .chain(&self.second_bias)
                .chain(&self.output_weights)
                .chain(std::iter::once(&self.output_bias))
                .all(|v| v.is_finite()),
            "critic parameters contain non-finite values"
        );
        Ok(())
    }
}

/// Scoring interface with no update method: actor training and critic acquisition stay separate.
pub struct StructuredRankV1 {
    manifest: CriticManifest,
    parameters: CriticParameters,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CriticCandidate<'a> {
    pub program_sha256: &'a str,
    pub tokens: &'a [u32],
    pub actor_score: Option<f32>,
    pub actor_checkpoint_sha256: &'a str,
    pub pool_sha256: &'a str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CriticRankedCandidate<'a> {
    pub program_sha256: &'a str,
    pub acquisition_score: f32,
    pub actor_score: Option<f32>,
    pub actor_checkpoint_sha256: &'a str,
    pub pool_sha256: &'a str,
    pub calibrated_interval: Option<(f32, f32)>,
}

impl StructuredRankV1 {
    pub fn new(manifest: CriticManifest, parameters: CriticParameters) -> Result<Self> {
        manifest.validate()?;
        parameters.validate()?;
        ensure!(
            parameters.sha256() == manifest.critic_weights_sha256,
            "critic weights do not match the manifest"
        );
        Ok(Self {
            manifest,
            parameters,
        })
    }

    pub fn manifest(&self) -> &CriticManifest {
        &self.manifest
    }

    /// Preserve token order in directed adjacent-pair features. Overlength input is rejected;
    /// there is no truncation or implicit EOS insertion.
    pub fn features(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        ensure!(!tokens.is_empty(), "critic refuses an empty token sequence");
        ensure!(
            tokens.len() <= self.manifest.max_tokens,
            "critic refuses overlength input"
        );
        ensure!(
            tokens.iter().all(|token| *token < self.manifest.vocab_size),
            "token outside critic vocabulary"
        );
        let mut out = vec![0.0f32; FEATURE_DIM];
        for token in tokens {
            out[(*token as usize) % HALF_DIM] += 1.0;
        }
        for pair in tokens.windows(2) {
            let bucket = pair_bucket(pair[0], pair[1]);
            out[HALF_DIM + bucket] += 1.0;
        }
        let scale = (tokens.len() as f32).sqrt().recip();
        for value in &mut out {
            *value *= scale;
        }
        Ok(out)
    }

    /// Inference-only Candle CPU scoring. The return contains no fabricated uncertainty.
    pub fn score(&self, tokens: &[u32]) -> Result<f32> {
        let features = self.features(tokens)?;
        let device = Device::Cpu;
        let input = Tensor::from_vec(features, (1, FEATURE_DIM), &device)?;
        let w1 = Tensor::from_vec(
            self.parameters.first_weights.clone(),
            (WIDTH, FEATURE_DIM),
            &device,
        )?;
        let b1 = Tensor::from_vec(self.parameters.first_bias.clone(), (1, WIDTH), &device)?;
        let w2 = Tensor::from_vec(
            self.parameters.second_weights.clone(),
            (WIDTH, WIDTH),
            &device,
        )?;
        let b2 = Tensor::from_vec(self.parameters.second_bias.clone(), (1, WIDTH), &device)?;
        let wo = Tensor::from_vec(self.parameters.output_weights.clone(), (1, WIDTH), &device)?;
        let bo = Tensor::new(self.parameters.output_bias, &device)?.reshape((1, 1))?;
        let h1 = linear(&input, &w1, &b1)?.gelu_erf()?;
        let h2 = linear(&h1, &w2, &b2)?.gelu_erf()?;
        let score = linear(&h2, &wo, &bo)?.to_vec2::<f32>()?[0][0];
        ensure!(score.is_finite(), "critic produced a non-finite score");
        Ok(score)
    }

    /// Rank one frozen pool. Actor scores are carried through unchanged and never enter the
    /// critic score; the actor checkpoint and pool must match the manifest exactly.
    pub fn rank_pool<'a>(
        &self,
        pool_sha256: &str,
        actor_sha256: &str,
        candidates: &[CriticCandidate<'a>],
    ) -> Result<Vec<CriticRankedCandidate<'a>>> {
        ensure!(
            pool_sha256 == self.manifest.same_pool_sha256,
            "critic comparison pool changed"
        );
        ensure!(
            actor_sha256 == self.manifest.actor_checkpoint_sha256,
            "frozen actor changed"
        );
        let mut seen = BTreeSet::new();
        let mut ranked = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            ensure!(
                candidate.pool_sha256 == pool_sha256,
                "candidate pool identity changed"
            );
            ensure!(
                candidate.actor_checkpoint_sha256 == actor_sha256,
                "candidate actor identity changed"
            );
            ensure!(
                is_sha256_hex(candidate.program_sha256),
                "invalid program digest"
            );
            ensure!(
                seen.insert(candidate.program_sha256),
                "duplicate candidate program"
            );
            if let Some(actor_score) = candidate.actor_score {
                ensure!(actor_score.is_finite(), "non-finite actor score");
            }
            let score = self.score(candidate.tokens)?;
            let calibrated_interval = self.manifest.calibration.as_ref().map(|calibration| {
                (
                    score - calibration.absolute_error_p90,
                    score + calibration.absolute_error_p90,
                )
            });
            ranked.push(CriticRankedCandidate {
                program_sha256: candidate.program_sha256,
                acquisition_score: score,
                actor_score: candidate.actor_score,
                actor_checkpoint_sha256: candidate.actor_checkpoint_sha256,
                pool_sha256: candidate.pool_sha256,
                calibrated_interval,
            });
        }
        ranked.sort_by(|a, b| {
            b.acquisition_score
                .total_cmp(&a.acquisition_score)
                .then_with(|| a.program_sha256.cmp(b.program_sha256))
        });
        Ok(ranked)
    }
}

fn pair_bucket(left: u32, right: u32) -> usize {
    // Fixed FNV-1a over a directed pair; changing pair order changes the feature stream.
    let mut hash = 0x811c9dc5u32;
    for byte in left.to_le_bytes().into_iter().chain(right.to_le_bytes()) {
        hash = (hash ^ u32::from(byte)).wrapping_mul(0x01000193);
    }
    hash as usize % HALF_DIM
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(byte: u8) -> String {
        format!("{byte:02x}").repeat(32)
    }

    fn candidate(
        parameters: CriticParameters,
        calibration: Option<OutcomeCalibration>,
    ) -> StructuredRankV1 {
        let manifest = CriticManifest {
            schema: STRUCTURED_RANK_SCHEMA.into(),
            candidate_id: "structural-rank-test-v1".into(),
            actor_checkpoint_sha256: digest(1),
            same_pool_sha256: digest(2),
            critic_weights_sha256: parameters.sha256(),
            vocab_size: 64,
            token_order: "declared_native_token_order_v1".into(),
            feature_schema: "unigram-directed-bigram-128-v1".into(),
            normalizer: "sqrt_length_l2_v1".into(),
            max_tokens: 128,
            width: WIDTH,
            layers: LAYERS,
            backend: "candle_cpu".into(),
            precision: "f32".into(),
            calibration,
        };
        StructuredRankV1::new(manifest, parameters).unwrap()
    }

    fn order_sensitive_parameters() -> CriticParameters {
        let mut p = CriticParameters {
            first_weights: vec![0.0; WIDTH * FEATURE_DIM],
            first_bias: vec![0.0; WIDTH],
            second_weights: vec![0.0; WIDTH * WIDTH],
            second_bias: vec![0.0; WIDTH],
            output_weights: vec![0.0; WIDTH],
            output_bias: 0.0,
        };
        // Pass the directed bigram bucket for 22 -> 11 through both hidden layers to output.
        let bucket = pair_bucket(22, 11);
        p.first_weights[0 * FEATURE_DIM + HALF_DIM + bucket] = 1.0;
        p.second_weights[0 * WIDTH] = 1.0;
        p.output_weights[0] = 1.0;
        p
    }

    #[test]
    fn same_token_multiset_in_different_order_has_distinct_semantics() {
        let ranker = candidate(order_sensitive_parameters(), None);
        let left = [11, 22, 11];
        let right = [11, 11, 22];
        assert_eq!(
            {
                let mut x = left;
                x.sort();
                x
            },
            {
                let mut x = right;
                x.sort();
                x
            }
        );
        assert_ne!(
            ranker.features(&left).unwrap(),
            ranker.features(&right).unwrap()
        );
        assert!(ranker.score(&left).unwrap() > ranker.score(&right).unwrap());
    }

    #[test]
    fn overlength_input_is_refused_without_truncation() {
        let ranker = candidate(order_sensitive_parameters(), None);
        let over = vec![1; MAX_SUPPORTED_TOKENS + 1];
        assert!(ranker
            .features(&over)
            .unwrap_err()
            .to_string()
            .contains("overlength"));
    }

    #[test]
    fn uncertainty_is_absent_without_heldout_calibration() {
        let ranker = candidate(order_sensitive_parameters(), None);
        assert!(ranker.manifest().calibration.is_none());
    }

    #[test]
    fn calibration_requires_a_real_heldout_pool_and_finite_error() {
        let valid = OutcomeCalibration {
            method: "heldout_absolute_error_p90_v1".into(),
            heldout_pool_sha256: digest(3),
            examples: 40,
            absolute_error_p90: 0.25,
        };
        let mut invalid = valid.clone();
        invalid.examples = 0;
        assert!(invalid.validate().is_err());
        candidate(order_sensitive_parameters(), Some(valid));
    }

    #[test]
    fn acquisition_ranking_uses_the_frozen_pool_and_preserves_actor_scores() {
        let ranker = candidate(order_sensitive_parameters(), None);
        let actor = ranker.manifest.actor_checkpoint_sha256.as_str();
        let pool = ranker.manifest.same_pool_sha256.as_str();
        let program_a = digest(8);
        let program_b = digest(9);
        let candidates = [
            CriticCandidate {
                program_sha256: &program_a,
                tokens: &[11, 22, 11],
                actor_score: Some(0.91),
                actor_checkpoint_sha256: actor,
                pool_sha256: pool,
            },
            CriticCandidate {
                program_sha256: &program_b,
                tokens: &[11, 11, 22],
                actor_score: Some(0.12),
                actor_checkpoint_sha256: actor,
                pool_sha256: pool,
            },
        ];
        let ranked = ranker.rank_pool(pool, actor, &candidates).unwrap();
        assert_eq!(ranked[0].actor_score, Some(0.91));
        assert_eq!(ranked[1].actor_score, Some(0.12));
        assert!(ranker.rank_pool(&digest(4), actor, &candidates).is_err());
        assert!(ranker.rank_pool(pool, &digest(5), &candidates).is_err());
    }

    #[test]
    fn unknown_manifest_fields_and_bad_shapes_are_refused() {
        let ranker = candidate(order_sensitive_parameters(), None);
        let mut bad = order_sensitive_parameters();
        bad.first_bias.pop();
        let manifest = ranker.manifest.clone();
        assert!(StructuredRankV1::new(manifest, bad).is_err());
        let value = serde_json::to_value(ranker.manifest()).unwrap();
        let mut with_unknown = value;
        with_unknown["fake_uncertainty"] = serde_json::json!(true);
        assert!(serde_json::from_value::<CriticManifest>(with_unknown).is_err());
    }
}
