//! Explicit, digest-bound optimizer recipes and continuation decisions.
//!
//! This module is deliberately independent of the neural trainer. Callers must
//! bind the accepted objective identity and the complete optimizer/scheduler
//! state they actually use; these types do not establish numerical parity.
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const RECIPE_SCHEMA: &str = "jope.optimizer_recipe.v1";
pub const ACCEPTED_CONTROL_ID: &str = "lineage_b_adamw_1e-4_control_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ObjectiveId {
    pub name: String,
    pub version: u32,
    /// SHA-256 of the exact accepted objective manifest supplied by the caller.
    pub manifest_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AdamWRecipe {
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub decay_rule: String,
    pub gradient_clip_norm: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ScheduleState {
    pub kind: String,
    pub step: u64,
    pub total_steps: Option<u64>,
    pub base_learning_rate: f64,
    pub current_learning_rate: f64,
    pub state_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct OptimizerRecipe {
    pub schema: String,
    pub recipe_id: String,
    pub hypothesis_id: String,
    pub preregistration_sha256: Option<String>,
    pub objective: ObjectiveId,
    pub optimizer: AdamWRecipe,
    pub moments_policy: String,
    pub schedule: ScheduleState,
    pub ema_state_sha256: Option<String>,
    pub sibling_baseline: String,
    pub support_policy: String,
    pub kl_reference: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ContinuationDecision {
    ExactResume,
    CarryMoments,
    ResetMomentsWarmStart,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct OptimizerStateIdentity {
    pub moments_sha256: String,
    pub step: u64,
    pub schedule_state_sha256: String,
    pub ema_state_sha256: Option<String>,
}

impl ObjectiveId {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.name.trim().is_empty(), "objective name required");
        ensure!(self.version > 0, "objective version must be positive");
        ensure!(
            is_sha256(&self.manifest_sha256),
            "objective manifest digest required"
        );
        Ok(())
    }
}

impl OptimizerRecipe {
    /// The accepted AdamW control; objective digest and state identities are
    /// explicit arguments because they are owned by the objective/caller.
    pub fn accepted_control(objective: ObjectiveId, schedule_state_sha256: String) -> Result<Self> {
        objective.validate()?;
        ensure!(
            is_sha256(&schedule_state_sha256),
            "schedule state digest required"
        );
        let recipe = Self {
            schema: RECIPE_SCHEMA.into(),
            recipe_id: ACCEPTED_CONTROL_ID.into(),
            hypothesis_id: "accepted_lineage_b_control".into(),
            preregistration_sha256: None,
            objective,
            optimizer: AdamWRecipe {
                learning_rate: 1e-4,
                weight_decay: 0.01,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                decay_rule: "uniform_decoupled_adamw".into(),
                gradient_clip_norm: None,
            },
            moments_policy: "carry_only_exact_same_recipe".into(),
            schedule: ScheduleState {
                kind: "constant".into(),
                step: 0,
                total_steps: None,
                base_learning_rate: 1e-4,
                current_learning_rate: 1e-4,
                state_sha256: schedule_state_sha256,
            },
            ema_state_sha256: None,
            sibling_baseline: "accepted_objective_internal".into(),
            support_policy: "accepted_objective_support".into(),
            kl_reference: "none_in_accepted_control".into(),
        };
        recipe.validate()?;
        Ok(recipe)
    }

    /// An alternative requires its own hypothesis identity and preregistration
    /// digest. Changing one field then necessarily changes the recipe digest.
    pub fn alternative(
        mut self,
        hypothesis_id: String,
        preregistration_sha256: String,
    ) -> Result<Self> {
        ensure!(
            hypothesis_id.trim() != self.hypothesis_id && !hypothesis_id.trim().is_empty(),
            "alternative hypothesis must have a distinct nonempty ID"
        );
        ensure!(
            is_sha256(&preregistration_sha256),
            "alternative preregistration digest required"
        );
        self.hypothesis_id = hypothesis_id.clone();
        self.recipe_id = hypothesis_id;
        self.preregistration_sha256 = Some(preregistration_sha256);
        self.validate()?;
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == RECIPE_SCHEMA,
            "unsupported optimizer recipe schema"
        );
        ensure!(
            !self.recipe_id.trim().is_empty() && !self.hypothesis_id.trim().is_empty(),
            "recipe and hypothesis IDs required"
        );
        self.objective.validate()?;
        let o = &self.optimizer;
        ensure!(
            o.learning_rate.is_finite() && o.learning_rate > 0.0,
            "positive finite learning rate required"
        );
        ensure!(
            o.weight_decay.is_finite() && o.weight_decay >= 0.0,
            "finite nonnegative weight decay required"
        );
        ensure!(
            o.beta1.is_finite()
                && (0.0..1.0).contains(&o.beta1)
                && o.beta2.is_finite()
                && (0.0..1.0).contains(&o.beta2),
            "AdamW beta values must be in [0,1)"
        );
        ensure!(
            o.epsilon.is_finite() && o.epsilon > 0.0,
            "positive finite epsilon required"
        );
        ensure!(!o.decay_rule.trim().is_empty(), "decay rule required");
        if let Some(c) = o.gradient_clip_norm {
            ensure!(
                c.is_finite() && c > 0.0,
                "gradient clip norm must be positive and finite"
            );
        }
        let s = &self.schedule;
        ensure!(
            !s.kind.trim().is_empty()
                && s.base_learning_rate.is_finite()
                && s.base_learning_rate > 0.0
                && s.current_learning_rate.is_finite()
                && s.current_learning_rate > 0.0,
            "explicit finite schedule state required"
        );
        if let Some(n) = s.total_steps {
            ensure!(
                n > 0 && s.step <= n,
                "schedule step outside declared horizon"
            );
        }
        ensure!(is_sha256(&s.state_sha256), "schedule state digest required");
        if let Some(d) = &self.ema_state_sha256 {
            ensure!(is_sha256(d), "EMA state digest malformed");
        }
        if self.recipe_id != ACCEPTED_CONTROL_ID {
            ensure!(
                self.preregistration_sha256
                    .as_deref()
                    .is_some_and(is_sha256),
                "alternative recipe must bind preregistration"
            );
        }
        Ok(())
    }

    pub fn digest_sha256(&self) -> Result<String> {
        self.validate()?;
        let bytes = serde_json::to_vec(self)?;
        Ok(hex::encode(Sha256::digest(bytes)))
    }

    pub fn continuation_from(
        &self,
        parent: &Self,
        child_state: &OptimizerStateIdentity,
        parent_state: &OptimizerStateIdentity,
        decision: ContinuationDecision,
    ) -> Result<()> {
        self.validate()?;
        parent.validate()?;
        match decision {
            ContinuationDecision::ExactResume => {
                ensure!(
                    self == parent,
                    "exact resume requires identical complete recipe"
                );
                ensure!(
                    child_state == parent_state,
                    "exact resume requires identical moments, step, schedule and EMA state"
                );
            }
            ContinuationDecision::CarryMoments => {
                ensure!(
                    self == parent,
                    "moment carry requires identical complete recipe"
                );
                ensure!(
                    child_state.moments_sha256 == parent_state.moments_sha256
                        && child_state.step == parent_state.step,
                    "moment carry requires unchanged moments and optimizer step"
                );
                ensure!(
                    child_state.schedule_state_sha256 == self.schedule.state_sha256
                        && parent_state.schedule_state_sha256 == parent.schedule.state_sha256,
                    "moment carry requires explicitly bound schedule state"
                );
                ensure!(
                    child_state.ema_state_sha256 == parent_state.ema_state_sha256,
                    "moment carry requires unchanged EMA state"
                );
            }
            ContinuationDecision::ResetMomentsWarmStart => {
                ensure!(
                    child_state.moments_sha256 == empty_state_sha256() && child_state.step == 0,
                    "warm start must reset moments and optimizer step"
                );
                ensure!(
                    child_state.schedule_state_sha256 == self.schedule.state_sha256,
                    "warm start must bind its schedule state"
                );
                ensure!(
                    child_state.ema_state_sha256 == self.ema_state_sha256,
                    "warm start EMA state must match recipe"
                );
            }
        }
        Ok(())
    }
}

pub fn empty_state_sha256() -> String {
    hex::encode(Sha256::digest(b"jope.optimizer.empty-state.v1"))
}
pub fn is_sha256(s: &str) -> bool {
    s.len() == 64
        && s.bytes()
            .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}
