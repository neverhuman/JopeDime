//! Versioned native support and behavior-law bridge.
//!
//! Scores use the same legal-token softmax and f32 temperature conversion as `decode`'s
//! sampler. Unknown handed-off selection propensity remains typed unknown and cannot authorize
//! an unbiasedness claim.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

use crate::{
    decode,
    grammar::{BoundVocabulary, State},
};

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum PropensityValue {
    Discrete {
        probability: f64,
        log_probability: f64,
    },
    Density {
        density: f64,
        log_density: f64,
        base_measure: String,
    },
    Unknown {
        reason: String,
    },
}

/// Runtime propensity returned by the scorer. It cannot be deserialized or caller-constructed;
/// only a draw/scorer or the explicit unknown path can produce one.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Propensity {
    value: PropensityValue,
}

impl Propensity {
    pub fn permits_unbiased_claim(&self) -> bool {
        self.validate().is_ok() && !self.is_unknown()
    }

    pub fn value(&self) -> &PropensityValue {
        &self.value
    }

    pub fn is_unknown(&self) -> bool {
        matches!(self.value, PropensityValue::Unknown { .. })
    }

    pub fn validate(&self) -> Result<()> {
        match &self.value {
            PropensityValue::Discrete {
                probability,
                log_probability,
            } => {
                ensure!(
                    probability.is_finite() && (0.0..=1.0).contains(probability),
                    "valid discrete probability required"
                );
                ensure!(
                    log_probability.is_finite() && *log_probability <= 0.0,
                    "valid discrete log propensity required"
                );
                if *probability > 0.0 {
                    ensure!(
                        (probability.ln() - log_probability).abs() <= 1e-12,
                        "discrete propensity log mismatch"
                    );
                } else {
                    ensure!(
                        log_probability.exp() == 0.0,
                        "zero probability must be underflow of a finite log propensity"
                    );
                }
            }
            PropensityValue::Density {
                density,
                log_density,
                base_measure,
            } => {
                ensure!(
                    density.is_finite() && *density > 0.0,
                    "valid density required"
                );
                ensure!(
                    !base_measure.trim().is_empty(),
                    "density base measure required"
                );
                ensure!(
                    log_density.is_finite() && (density.ln() - log_density).abs() <= 1e-12,
                    "density log mismatch"
                );
            }
            PropensityValue::Unknown { .. } => {
                anyhow::bail!("unknown propensity has no unbiasedness authority")
            }
        }
        Ok(())
    }

    fn discrete(probability: f64, log_probability: f64) -> Result<Self> {
        let propensity = Self {
            value: PropensityValue::Discrete {
                probability,
                log_probability,
            },
        };
        propensity.validate()?;
        Ok(propensity)
    }
}

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct BehaviorLaw {
    schema: String,
    vocabulary_sha256: String,
    temperature: f32,
    includes_eos: bool,
    legal_mask: String,
    mixture_branch_probability: Option<f64>,
    fallback_or_rejection: String,
    selection_law: String,
}

impl BehaviorLaw {
    pub fn native(vocabulary_sha256: &str, temperature: f32) -> Result<Self> {
        ensure!(
            crate::is_sha256_hex(vocabulary_sha256),
            "bound vocabulary SHA256 required"
        );
        ensure!(
            temperature.is_finite() && temperature > 0.,
            "positive finite temperature required"
        );
        Ok(Self {
            schema: "jope.behavior-law/native-grammar/v1".into(),
            vocabulary_sha256: vocabulary_sha256.to_owned(),
            temperature,
            includes_eos: true,
            legal_mask: "grammar_state_legal_sorted".into(),
            mixture_branch_probability: Some(1.),
            fallback_or_rejection: "none".into(),
            selection_law: "conditional_token_draw".into(),
        })
    }

    /// Compute the exact conditional token law used by `decode::sample_legal`.
    pub fn score_legal(&self, logits: &[f32], legal: &[usize]) -> Result<Vec<(usize, f64)>> {
        ensure!(
            self.schema == "jope.behavior-law/native-grammar/v1",
            "behavior law version mismatch"
        );
        ensure!(self.includes_eos, "EOS-inclusive law required");
        ensure!(
            crate::is_sha256_hex(&self.vocabulary_sha256),
            "behavior law vocabulary identity required"
        );
        ensure!(
            self.legal_mask == "grammar_state_legal_sorted",
            "unsupported legal mask law"
        );
        ensure!(
            self.mixture_branch_probability == Some(1.),
            "unresolved mixture branch law"
        );
        ensure!(
            self.fallback_or_rejection == "none",
            "selection conditioning changes draw law"
        );
        ensure!(
            self.selection_law == "conditional_token_draw",
            "handed-off selection law is not token draw law"
        );
        ensure!(!legal.is_empty(), "nonempty legal set required");
        decode::score_legal(logits, legal, self.temperature)
    }

    pub fn score_program(
        &self,
        bound: &BoundVocabulary,
        tokens: &[usize],
        logits: &[Vec<f32>],
    ) -> Result<Propensity> {
        ensure!(
            bound.sha256() == self.vocabulary_sha256,
            "behavior law vocabulary mismatch"
        );
        ensure!(
            tokens.len() >= 2 && tokens.last() == Some(&2),
            "complete EOS program required"
        );
        ensure!(
            tokens.len() <= crate::grammar::MAX_PROGRAM_TOKENS,
            "program exceeds native token cap"
        );
        ensure!(
            logits.len() == tokens.len() - 1,
            "one logit row per predicted token required"
        );
        let round_trip = bound.representable_tokens(&bound.vocab().decode(tokens)?)?;
        ensure!(
            round_trip == tokens,
            "native token/program identity mismatch"
        );
        let states = bound.vocab().replay(tokens)?;
        let mut log_probability = 0.0;
        for (index, row) in logits.iter().enumerate() {
            let legal = &states[index];
            let scored = self.score_legal(row, legal)?;
            let selected = tokens[index + 1];
            let probability = scored
                .iter()
                .find(|(id, _)| *id == selected)
                .ok_or_else(|| anyhow::anyhow!("program selected an illegal token"))?
                .1;
            ensure!(
                probability.is_finite() && probability > 0.,
                "token has zero probability"
            );
            log_probability += probability.ln();
        }
        Propensity::discrete(log_probability.exp(), log_probability)
    }

    /// Draw through the production grammar sampler while returning the same law's score.
    pub fn sample_state<R: rand::Rng>(
        &self,
        bound: &BoundVocabulary,
        state: &State,
        logits: &[f32],
        rng: &mut R,
    ) -> Result<(usize, Propensity)> {
        ensure!(
            bound.sha256() == self.vocabulary_sha256,
            "behavior law vocabulary mismatch"
        );
        let legal = state.legal(bound.vocab());
        let scored = self.score_legal(logits, &legal)?;
        let token = decode::sample_legal(logits, &legal, self.temperature, rng)?;
        let probability = scored
            .iter()
            .find(|(id, _)| *id == token)
            .ok_or_else(|| anyhow::anyhow!("sampler returned token outside scored support"))?
            .1;
        let propensity = Propensity::discrete(probability, probability.ln())?;
        Ok((token, propensity))
    }

    pub fn unknown_selection(reason: impl Into<String>) -> Propensity {
        Propensity {
            value: PropensityValue::Unknown {
                reason: reason.into(),
            },
        }
    }
}
