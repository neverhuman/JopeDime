//! Pure objective coefficients, eligibility and sticky KL state transitions.
//! This module does not measure model likelihoods or invoke an optimizer.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const MAX_GROUP_SIZE: usize = 64;
pub const KL_HISTORY_CAP: usize = 50;
pub const KL_ROUNDING_FLOOR: f64 = -1e-10;

fn finite_nonempty(values: &[f64]) -> Result<()> {
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "nonempty finite utility vector required"
    );
    ensure!(
        values.len() <= MAX_GROUP_SIZE,
        "group exceeds bounded maximum"
    );
    Ok(())
}

/// Mean-objective RLOO coefficients, matching the committed Python primitive.
pub fn rloo_weights(utilities: &[f64], std_normalize: bool) -> Result<Vec<f64>> {
    finite_nonempty(utilities)?;
    ensure!(!std_normalize, "group-standardized RLOO is unsupported");
    let g = utilities.len();
    if g == 1 {
        return Ok(vec![utilities[0]]);
    }
    let total: f64 = utilities.iter().sum();
    Ok(utilities
        .iter()
        .map(|x| (x - (total - x) / (g - 1) as f64) / g as f64)
        .collect())
}

fn choose(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    (1..=k).fold(1.0, |v, i| v * (n - k + i) as f64 / i as f64)
}

/// Unbiased best-of-B positive-part coefficients. Ties follow source ordering,
/// as in the committed Python implementation; no second division by G applies.
pub fn pkpo_weights(utilities: &[f64], h: f64, best_of: usize) -> Result<Vec<f64>> {
    finite_nonempty(utilities)?;
    ensure!(
        h.is_finite() && (1..=utilities.len()).contains(&best_of),
        "finite threshold and 1 <= B <= G required"
    );
    let g = utilities.len();
    let clipped: Vec<f64> = utilities.iter().map(|x| h.max(*x)).collect();
    if best_of == 1 {
        return Ok(clipped.iter().map(|x| (x - h) / g as f64).collect());
    }
    let mut order: Vec<usize> = (0..g).collect();
    order.sort_by(|a, b| clipped[*a].total_cmp(&clipped[*b]));
    let denominator = choose(g, best_of);
    let mut out = vec![0.0; g];
    let mut lower_sum = 0.0;
    for (z, index) in order.iter().copied().enumerate() {
        let k = z + 1;
        if k < best_of {
            continue;
        }
        let m = k - 1;
        lower_sum += choose(m - 1, best_of - 2) * clipped[order[m - 1]];
        out[index] =
            ((choose(k - 1, best_of - 1) * clipped[index] - lower_sum) / denominator).max(0.0);
    }
    ensure!(
        out.iter().all(|x| x.is_finite()),
        "PKPO coefficient overflow"
    );
    Ok(out)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PerHeadEligibility {
    pub head: String,
    pub train_role: bool,
    pub exact_eos_behavior_logprob: bool,
    pub support_match: bool,
    pub complete_sibling_group: bool,
    pub positive_finite_scale: bool,
    pub reason: Option<String>,
}
impl PerHeadEligibility {
    pub fn eligible(&self) -> bool {
        self.train_role
            && self.exact_eos_behavior_logprob
            && self.support_match
            && self.complete_sibling_group
            && self.positive_finite_scale
            && self.reason.is_none()
    }
    pub fn require(&self) -> Result<()> {
        ensure!(!self.head.trim().is_empty(), "head identity required");
        ensure!(self.eligible(), "head is ineligible: {:?}", self.reason);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AwrInput {
    pub advantage: f64,
    pub pooled_scale: f64,
    pub beta: f64,
    pub max_weight: f64,
    pub exact_eos_behavior_logprob: bool,
    pub support_match: bool,
    pub train_role: bool,
}
#[derive(Clone, Debug, PartialEq)]
pub enum AwrAdmission {
    Eligible {
        normalized_advantage: f64,
        weight: f64,
    },
    Ineligible {
        reason: &'static str,
    },
}
/// AWR requires a measured positive pooled scale and declared behavior support.
pub fn awr_weight(input: &AwrInput) -> AwrAdmission {
    if !input.train_role {
        return AwrAdmission::Ineligible {
            reason: "not_train_role",
        };
    }
    if !input.exact_eos_behavior_logprob {
        return AwrAdmission::Ineligible {
            reason: "missing_exact_eos_behavior_logprob",
        };
    }
    if !input.support_match {
        return AwrAdmission::Ineligible {
            reason: "behavior_support_mismatch",
        };
    }
    if !input.advantage.is_finite() || !input.pooled_scale.is_finite() || input.pooled_scale <= 0.0
    {
        return AwrAdmission::Ineligible {
            reason: "pooled_scale_not_positive_finite",
        };
    }
    if !input.beta.is_finite()
        || input.beta <= 0.0
        || !input.max_weight.is_finite()
        || input.max_weight <= 0.0
    {
        return AwrAdmission::Ineligible {
            reason: "invalid_weight_policy",
        };
    }
    let normalized_advantage = input.advantage / input.pooled_scale;
    let exponent = (normalized_advantage / input.beta).clamp(-745.0, 709.0);
    let weight = exponent.exp().min(input.max_weight);
    if !weight.is_finite() {
        return AwrAdmission::Ineligible {
            reason: "nonfinite_weight",
        };
    }
    AwrAdmission::Eligible {
        normalized_advantage,
        weight,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KlPolicy {
    pub alarm_mean: f64,
    pub hold_mean: f64,
    pub hold_max: f64,
}
impl KlPolicy {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.alarm_mean.is_finite()
                && self.hold_mean.is_finite()
                && self.hold_max.is_finite()
                && self.alarm_mean > 0.0
                && self.alarm_mean <= self.hold_mean
                && self.hold_mean <= self.hold_max,
            "finite ordered positive KL thresholds required"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KlObservation {
    pub epoch: u64,
    pub step: u64,
    pub kl_mean: f64,
    pub kl_max: f64,
    pub alarm: bool,
    pub hold: bool,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StickyKlGuard {
    pub policy_sha256: String,
    pub probe_identity_sha256: String,
    pub last_epoch: Option<u64>,
    pub alarm_count: u64,
    pub first_alarm_epoch: Option<u64>,
    pub tripped: bool,
    pub first_trip_epoch: Option<u64>,
    pub history: Vec<KlObservation>,
}
impl StickyKlGuard {
    pub fn new(policy_sha256: String, probe_identity_sha256: String) -> Result<Self> {
        ensure!(
            is_sha256(&policy_sha256) && is_sha256(&probe_identity_sha256),
            "policy and probe identities must be SHA-256"
        );
        Ok(Self {
            policy_sha256,
            probe_identity_sha256,
            last_epoch: None,
            alarm_count: 0,
            first_alarm_epoch: None,
            tripped: false,
            first_trip_epoch: None,
            history: Vec::new(),
        })
    }
    pub fn check_identity(&self, policy: &str, probe: &str) -> Result<()> {
        ensure!(
            self.policy_sha256 == policy && self.probe_identity_sha256 == probe,
            "KL guard policy/probe identity changed"
        );
        Ok(())
    }
    pub fn observe(
        &mut self,
        policy: &KlPolicy,
        epoch: u64,
        step: u64,
        kl_mean: f64,
        kl_max: f64,
    ) -> Result<()> {
        policy.validate()?;
        ensure!(
            kl_mean.is_finite()
                && kl_max.is_finite()
                && kl_mean >= KL_ROUNDING_FLOOR
                && kl_max >= KL_ROUNDING_FLOOR,
            "finite nonnegative KL required"
        );
        ensure!(
            self.last_epoch.is_none_or(|last| epoch > last),
            "KL epochs must increase strictly"
        );
        let alarm = kl_mean >= policy.alarm_mean;
        let hold = kl_mean >= policy.hold_mean || kl_max >= policy.hold_max;
        if alarm {
            self.alarm_count += 1;
            if self.first_alarm_epoch.is_none() {
                self.first_alarm_epoch = Some(epoch);
            }
        }
        if hold && !self.tripped {
            self.tripped = true;
            self.first_trip_epoch = Some(epoch);
        }
        self.last_epoch = Some(epoch);
        self.history.push(KlObservation {
            epoch,
            step,
            kl_mean,
            kl_max,
            alarm,
            hold,
        });
        if self.history.len() > KL_HISTORY_CAP {
            self.history.remove(0);
        }
        Ok(())
    }
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HeadGradient {
    pub head: String,
    pub active_parameter_count: u64,
    pub gradient_l2: f64,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UpdateAccounting {
    pub computed_updates: u64,
    pub applied_updates: u64,
    pub committed_updates: u64,
    pub active_heads: Vec<HeadGradient>,
    pub parameters_before_sha256: String,
    pub parameters_after_sha256: String,
    pub optimizer_state_sha256: String,
}
impl UpdateAccounting {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.computed_updates >= self.applied_updates
                && self.applied_updates >= self.committed_updates,
            "update counters must satisfy computed >= applied >= committed"
        );
        ensure!(
            is_sha256(&self.parameters_before_sha256)
                && is_sha256(&self.parameters_after_sha256)
                && is_sha256(&self.optimizer_state_sha256),
            "update state digests required"
        );
        ensure!(
            self.active_heads.iter().all(|h| !h.head.trim().is_empty()
                && h.gradient_l2.is_finite()
                && h.gradient_l2 >= 0.0),
            "finite nonnegative per-head gradient records required"
        );
        if self.committed_updates > 0 {
            ensure!(
                self.active_heads
                    .iter()
                    .any(|h| h.active_parameter_count > 0 && h.gradient_l2 > 0.0),
                "committed active update requires a nonzero admitted-head gradient"
            );
            ensure!(
                self.parameters_before_sha256 != self.parameters_after_sha256,
                "decay-only/no-change parameter state cannot evidence learning"
            );
        }
        Ok(())
    }
}
