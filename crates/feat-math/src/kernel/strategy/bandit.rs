//! A small, explicit EXP3 behavior law for operator selection.
//!
//! This module owns no RNG, clocks, evaluator, or event store. The caller supplies a uniform
//! variate and stable attempt ID, and receives the exact probability used to select the action.

use serde::{Deserialize, Serialize};
use sha2::Digest;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BanditConfig {
    pub context_id: String,
    pub arm_order: Vec<String>,
    /// Uniform exploration floor. Must be in (0, 1].
    pub epsilon: f64,
    /// EXP3 learning rate. Must be finite and positive.
    pub learning_rate: f64,
    /// Eligible measured rewards are clipped to this closed interval before updating.
    pub reward_min: f64,
    pub reward_max: f64,
}

impl BanditConfig {
    pub fn validate(&self) -> Result<(), BanditError> {
        if self.context_id.is_empty()
            || self.arm_order.is_empty()
            || self.arm_order.iter().any(String::is_empty)
            || self
                .arm_order
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                != self.arm_order.len()
            || !self.epsilon.is_finite()
            || self.epsilon < f64::MIN_POSITIVE * self.arm_order.len() as f64
            || self.epsilon > 1.0
            || !self.learning_rate.is_finite()
            || self.learning_rate <= 0.0
            || !self.reward_min.is_finite()
            || !self.reward_max.is_finite()
            || self.reward_min >= 0.0
            || self.reward_max <= 0.0
            || self.reward_min >= self.reward_max
        {
            return Err(BanditError::InvalidConfig);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub enum BehaviorLaw {
    KnownExp3 { config_sha256: String },
    Unknown { reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionProbability {
    pub arm: String,
    /// Probability after legal-set normalization and uniform exploration floor.
    pub probability: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Selection {
    pub attempt_id: String,
    pub context_id: String,
    pub law: BehaviorLaw,
    pub legal_arms: Vec<String>,
    pub distribution: Vec<ActionProbability>,
    pub chosen_arm: String,
    pub propensity: f64,
    pub log_propensity: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AttemptRecord {
    pub attempt_id: String,
    pub selection: Option<Selection>,
    pub disposition: AttemptDisposition,
    pub rejection_reason: Option<String>,
    pub feedback: Vec<Feedback>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttemptDisposition {
    Chosen,
    Rejected,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Feedback {
    pub status: FeedbackStatus,
    pub raw_reward: Option<f64>,
    pub clipped_reward: Option<f64>,
    pub updated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeedbackStatus {
    EligibleMeasured,
    Delayed,
    Failed,
    Ineligible,
}

#[derive(Clone, Debug, PartialEq)]
pub enum BanditError {
    InvalidConfig,
    UnknownArm(String),
    InvalidDraw,
    DuplicateAttempt(String),
    UnknownAttempt(String),
    FeedbackAlreadyRecorded(String),
    UnavailableAttempt(String),
    InvalidReward,
    UnknownLaw,
    PropensityUnavailable,
}

impl std::fmt::Display for BanditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for BanditError {}

/// Context-stratified EXP3 with a recorded, known action propensity.
///
/// Legal subsets are re-normalized on every selection. The configured arm order determines
/// output and tie behavior, while `uniform_draw` makes replay independent of an internal RNG.
#[derive(Clone, Debug)]
pub struct OperatorBandit {
    config: BanditConfig,
    config_sha256: String,
    log_weights: BTreeMap<String, f64>,
    attempts: Vec<AttemptRecord>,
    attempt_indices: BTreeMap<String, usize>,
}

impl OperatorBandit {
    pub fn new(config: BanditConfig) -> Result<Self, BanditError> {
        config.validate()?;
        let serialized = serde_json::to_vec(&config).expect("validated config is serializable");
        let config_sha256 = format!("{:x}", sha2::Sha256::digest(serialized));
        let log_weights = config
            .arm_order
            .iter()
            .cloned()
            .map(|arm| (arm, 0.0))
            .collect();
        Ok(Self {
            config,
            config_sha256,
            log_weights,
            attempts: Vec::new(),
            attempt_indices: BTreeMap::new(),
        })
    }

    pub fn attempts(&self) -> &[AttemptRecord] {
        &self.attempts
    }

    pub fn behavior_law(&self) -> BehaviorLaw {
        BehaviorLaw::KnownExp3 {
            config_sha256: self.config_sha256.clone(),
        }
    }

    /// Empty legal sets are logged as unavailable, with no fabricated action or probability.
    pub fn select(
        &mut self,
        attempt_id: impl Into<String>,
        context_id: &str,
        legal_arms: &[String],
        uniform_draw: f64,
    ) -> Result<Option<Selection>, BanditError> {
        let attempt_id = attempt_id.into();
        if self.attempt_indices.contains_key(&attempt_id) {
            return Err(BanditError::DuplicateAttempt(attempt_id));
        }
        if context_id != self.config.context_id {
            return Err(BanditError::InvalidConfig);
        }
        if !uniform_draw.is_finite() || !(0.0..1.0).contains(&uniform_draw) {
            return Err(BanditError::InvalidDraw);
        }
        let legal = self.canonical_legal_set(legal_arms)?;
        if legal.is_empty() {
            self.push_attempt(AttemptRecord {
                attempt_id,
                selection: None,
                disposition: AttemptDisposition::Unavailable,
                rejection_reason: None,
                feedback: Vec::new(),
            });
            return Ok(None);
        }

        let distribution = self.distribution_for(&legal);
        let mut cumulative = 0.0;
        let mut chosen = distribution
            .last()
            .expect("legal set is nonempty")
            .arm
            .clone();
        for entry in &distribution {
            cumulative += entry.probability;
            if uniform_draw < cumulative {
                chosen.clone_from(&entry.arm);
                break;
            }
        }
        let propensity = distribution
            .iter()
            .find(|entry| entry.arm == chosen)
            .expect("chosen arm is in distribution")
            .probability;
        let selection = Selection {
            attempt_id: attempt_id.clone(),
            context_id: self.config.context_id.clone(),
            law: self.behavior_law(),
            legal_arms: legal,
            distribution,
            chosen_arm: chosen,
            propensity,
            log_propensity: propensity.ln(),
        };
        self.push_attempt(AttemptRecord {
            attempt_id,
            selection: Some(selection.clone()),
            disposition: AttemptDisposition::Chosen,
            rejection_reason: None,
            feedback: Vec::new(),
        });
        Ok(Some(selection))
    }

    /// Record delayed feedback exactly once. Only eligible measured outcomes update EXP3.
    pub fn record_feedback(
        &mut self,
        attempt_id: &str,
        status: FeedbackStatus,
        reward: Option<f64>,
    ) -> Result<(), BanditError> {
        let index = *self
            .attempt_indices
            .get(attempt_id)
            .ok_or_else(|| BanditError::UnknownAttempt(attempt_id.to_owned()))?;
        let record = &self.attempts[index];
        if record
            .feedback
            .iter()
            .any(|feedback| feedback.status != FeedbackStatus::Delayed)
        {
            return Err(BanditError::FeedbackAlreadyRecorded(attempt_id.to_owned()));
        }
        let selection = record
            .selection
            .as_ref()
            .ok_or_else(|| BanditError::UnavailableAttempt(attempt_id.to_owned()))?
            .clone();
        let (raw_reward, clipped_reward, updated) = match status {
            FeedbackStatus::EligibleMeasured => {
                let reward = reward.ok_or(BanditError::InvalidReward)?;
                if !reward.is_finite() {
                    return Err(BanditError::InvalidReward);
                }
                let clipped = reward.clamp(self.config.reward_min, self.config.reward_max);
                self.update(&selection, clipped)?;
                (Some(reward), Some(clipped), true)
            }
            FeedbackStatus::Delayed | FeedbackStatus::Failed | FeedbackStatus::Ineligible => {
                if reward.is_some() {
                    return Err(BanditError::InvalidReward);
                }
                (None, None, false)
            }
        };
        self.attempts[index].feedback.push(Feedback {
            status,
            raw_reward,
            clipped_reward,
            updated,
        });
        Ok(())
    }

    /// Mark a selected action rejected by a downstream legality or execution check.
    pub fn mark_rejected(
        &mut self,
        attempt_id: &str,
        reason: impl Into<String>,
    ) -> Result<(), BanditError> {
        let index = *self
            .attempt_indices
            .get(attempt_id)
            .ok_or_else(|| BanditError::UnknownAttempt(attempt_id.to_owned()))?;
        let record = &mut self.attempts[index];
        if record.selection.is_none() {
            return Err(BanditError::UnavailableAttempt(attempt_id.to_owned()));
        }
        record.disposition = AttemptDisposition::Rejected;
        record.rejection_reason = Some(reason.into());
        Ok(())
    }

    /// Unknown behavior laws are never accepted as unbiased PG/OPE propensities.
    pub fn propensity_for_pg_ope(
        law: &BehaviorLaw,
        propensity: Option<f64>,
    ) -> Result<f64, BanditError> {
        match law {
            BehaviorLaw::Unknown { .. } => return Err(BanditError::UnknownLaw),
            BehaviorLaw::KnownExp3 { config_sha256 }
                if config_sha256.len() != 64
                    || !config_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) =>
            {
                return Err(BanditError::PropensityUnavailable);
            }
            BehaviorLaw::KnownExp3 { .. } => {}
        }
        let propensity = propensity.ok_or(BanditError::PropensityUnavailable)?;
        if !propensity.is_finite() || !(0.0..=1.0).contains(&propensity) || propensity == 0.0 {
            return Err(BanditError::PropensityUnavailable);
        }
        Ok(propensity)
    }

    fn update(&mut self, selection: &Selection, reward: f64) -> Result<(), BanditError> {
        let k = selection.distribution.len() as f64;
        let reward_scale = self
            .config
            .reward_min
            .abs()
            .max(self.config.reward_max.abs());
        let importance_weight = (reward / reward_scale) / selection.propensity;
        let increment = self.config.learning_rate * importance_weight / k;
        if !increment.is_finite() {
            return Err(BanditError::InvalidReward);
        }
        let weight = self
            .log_weights
            .get_mut(&selection.chosen_arm)
            .expect("selected arm belongs to configured arm set");
        let updated = *weight + increment;
        if !updated.is_finite() {
            return Err(BanditError::InvalidReward);
        }
        *weight = updated;
        Ok(())
    }

    fn canonical_legal_set(&self, legal_arms: &[String]) -> Result<Vec<String>, BanditError> {
        let mut legal = std::collections::BTreeSet::new();
        for arm in legal_arms {
            if !self.log_weights.contains_key(arm) {
                return Err(BanditError::UnknownArm(arm.clone()));
            }
            legal.insert(arm.as_str());
        }
        Ok(self
            .config
            .arm_order
            .iter()
            .filter(|arm| legal.contains(arm.as_str()))
            .cloned()
            .collect())
    }

    fn distribution_for(&self, legal: &[String]) -> Vec<ActionProbability> {
        let max_log_weight = legal
            .iter()
            .map(|arm| self.log_weights[arm])
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = legal
            .iter()
            .map(|arm| (self.log_weights[arm] - max_log_weight).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let k = legal.len() as f64;
        let mut probabilities: Vec<f64> = legal
            .iter()
            .zip(weights)
            .map(|(_, weight)| {
                ((1.0 - self.config.epsilon) * (weight / total) + self.config.epsilon / k)
                    .max(f64::MIN_POSITIVE)
            })
            .collect();
        let probability_sum: f64 = probabilities.iter().sum();
        for probability in &mut probabilities {
            *probability /= probability_sum;
        }
        legal
            .iter()
            .zip(probabilities)
            .map(|(arm, probability)| ActionProbability {
                arm: arm.clone(),
                probability,
            })
            .collect()
    }

    fn push_attempt(&mut self, record: AttemptRecord) {
        let index = self.attempts.len();
        self.attempt_indices
            .insert(record.attempt_id.clone(), index);
        self.attempts.push(record);
    }
}
