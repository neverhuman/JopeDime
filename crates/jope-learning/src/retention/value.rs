//! Read-only value and packing review plans. This module returns data in memory and has no
//! storage or filesystem effects; applying a recommendation belongs to a later reviewed tool.

use super::policy::Policy;
use crate::canon::sha256_hex;
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const VALUE_POLICY_V2_SCHEMA: &str = "jopedime.retention-value-policy/v2";
pub const RETENTION_PLAN_V2_SCHEMA: &str = "jope.retention-plan/v2";
pub const MAX_INVENTORY_AGE_SECS: u64 = 24 * 60 * 60;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ValueWeights {
    pub influence: f64,
    pub novelty: f64,
    pub uncertainty: f64,
    pub reproduction_cost: f64,
    pub storage_cost: f64,
}

impl ValueWeights {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.influence.is_finite() && self.influence >= 0.0,
            "influence weight must be finite and nonnegative"
        );
        for (name, value) in [
            ("novelty", self.novelty),
            ("uncertainty", self.uncertainty),
            ("reproduction_cost", self.reproduction_cost),
            ("storage_cost", self.storage_cost),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} weight must be finite and nonnegative"
            );
        }
        Ok(())
    }
}

/// Versioned view of the validated v1 retention knobs plus explicit value weights.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetentionPolicyV2 {
    pub schema: String,
    pub k_ctx: u64,
    pub k_lineage: u64,
    pub reservoir_rate: f64,
    pub margin_rel: f64,
    pub grace_days: u64,
    pub weights: ValueWeights,
}

impl RetentionPolicyV2 {
    /// Lift only the validated dormant v1 knobs; weights remain caller-supplied and explicit.
    pub fn from_v1(policy: &Policy, weights: ValueWeights) -> Result<Self> {
        let v2 = Self {
            schema: VALUE_POLICY_V2_SCHEMA.into(),
            k_ctx: policy.k_ctx,
            k_lineage: policy.k_lineage,
            reservoir_rate: policy.reservoir_rate,
            margin_rel: policy.margin_rel,
            grace_days: policy.grace_days,
            weights,
        };
        v2.validate()?;
        Ok(v2)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let value: Self = serde_json::from_slice(bytes)?;
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == VALUE_POLICY_V2_SCHEMA,
            "unsupported retention value policy schema"
        );
        ensure!(
            self.k_ctx > 0 && self.k_lineage > 0,
            "keeper limits must be positive"
        );
        ensure!(
            self.reservoir_rate.is_finite()
                && self.reservoir_rate > 0.0
                && self.reservoir_rate <= 1.0,
            "reservoir_rate must be in (0, 1]"
        );
        ensure!(
            self.margin_rel.is_finite() && self.margin_rel >= 0.0,
            "margin_rel must be finite and nonnegative"
        );
        ensure!(self.grace_days > 0, "grace_days must be positive");
        self.weights.validate()
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct PhysicalObject {
    pub bucket: String,
    pub key: String,
    pub version_id: String,
    pub sha256: String,
    pub bytes: u64,
}

impl PhysicalObject {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.bucket.is_empty() && !self.key.is_empty() && !self.version_id.is_empty(),
            "physical object requires bucket, key, and version"
        );
        ensure!(
            self.sha256.len() == 64
                && self
                    .sha256
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "physical object {} has invalid sha256",
            self.key
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LearningSignals {
    /// Measured leave-group-out change in the declared value-model metric.
    pub influence: f64,
    pub novelty: f64,
    pub uncertainty: f64,
    pub reproduction_cost_seconds: f64,
    pub reproduction_probability: f64,
    pub storage_usd_per_month: f64,
    /// Residual surprise is shown for review ordering only.
    pub surprise: f64,
}

impl LearningSignals {
    fn validate(&self) -> Result<()> {
        ensure!(self.influence.is_finite(), "influence must be finite");
        for (name, value) in [
            ("novelty", self.novelty),
            ("uncertainty", self.uncertainty),
            ("reproduction_cost_seconds", self.reproduction_cost_seconds),
            ("reproduction_probability", self.reproduction_probability),
            ("storage_usd_per_month", self.storage_usd_per_month),
            ("surprise", self.surprise),
        ] {
            ensure!(
                value.is_finite() && value >= 0.0,
                "{name} must be finite and nonnegative"
            );
        }
        ensure!(self.novelty <= 1.0, "novelty must be in [0, 1]");
        ensure!(
            self.reproduction_probability <= 1.0,
            "reproduction_probability must be in [0, 1]"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetentionValue {
    pub schema: String,
    pub object: PhysicalObject,
    pub pinned: bool,
    pub score_rows_always_kept: bool,
    pub marginal_value: Option<f64>,
    pub surprise: Option<f64>,
    pub score_basis: String,
}

impl RetentionValue {
    pub fn classify(
        object: &PhysicalObject,
        pinned: bool,
        score_row: bool,
        signals: Option<&LearningSignals>,
        policy: &RetentionPolicyV2,
    ) -> Result<Self> {
        let (marginal_value, surprise, score_basis) = match signals {
            Some(s) => {
                s.validate()?;
                let w = &policy.weights;
                let value = w.influence * s.influence
                    + w.novelty * s.novelty
                    + w.uncertainty * s.uncertainty
                    + w.reproduction_cost
                        * s.reproduction_cost_seconds
                        * s.reproduction_probability
                    - w.storage_cost * s.storage_usd_per_month;
                ensure!(value.is_finite(), "marginal value is not finite");
                (Some(value), Some(s.surprise), "measured-signals-v1".into())
            }
            None => (None, None, "unknown-no-measured-signals".into()),
        };
        Ok(Self {
            schema: "jope.retention-value/v2".into(),
            object: object.clone(),
            pinned: pinned || score_row,
            score_rows_always_kept: score_row,
            marginal_value,
            surprise,
            score_basis,
        })
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ConsumerClosure {
    pub scanned_at_unix: u64,
    pub observed_root_generation: String,
    pub current_root_generation: String,
    pub active_consumers: Vec<String>,
    pub unresolved: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ReplacementProof {
    ExactDuplicate {
        replacement: PhysicalObject,
        content_hash_verified: bool,
        restore_tested: bool,
    },
    Pack {
        replacement: PhysicalObject,
        members: Vec<PhysicalObject>,
        manifest_verified: bool,
        restore_tested: bool,
    },
    Supersession {
        replacement: PhysicalObject,
        qualified_champion: bool,
        relative_margin: f64,
        days_since_dominance: u64,
        score_rows_retained: bool,
        recovery_closure_verified: bool,
        restore_tested: bool,
    },
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct Candidate {
    pub sources: Vec<PhysicalObject>,
    pub hard_t0: bool,
    pub active_pin: bool,
    pub active_promise: bool,
    pub score_row: bool,
    pub inventory_at_unix: Option<u64>,
    pub consumer_closure: Option<ConsumerClosure>,
    pub replacement: Option<ReplacementProof>,
    pub signals: Option<LearningSignals>,
    pub context_rank: Option<u64>,
    pub lineage_rank: Option<u64>,
    pub reservoir_selected: Option<bool>,
    pub ranking_current: bool,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Recommendation {
    KeepT0,
    KeepScoreRows,
    Keep,
    Hold,
    ReviewDuplicate,
    ReviewPack,
    ReviewSupersession,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PlanItem {
    pub source: Vec<PhysicalObject>,
    pub replacement: Option<PhysicalObject>,
    pub recommendation: Recommendation,
    pub reason: String,
    pub unresolved: Vec<String>,
    pub values: Vec<RetentionValue>,
    pub current_bytes: u64,
    pub hypothetical_bytes_after_review: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RetentionPlanV2 {
    pub schema: String,
    pub as_of_unix: u64,
    pub inventory_max_age_secs: u64,
    pub source_inventory_sha256: String,
    pub policy: RetentionPolicyV2,
    pub items: Vec<PlanItem>,
    pub current_objects: u64,
    pub current_bytes: u64,
    pub hypothetical_objects_after_review: u64,
    pub hypothetical_bytes_after_review: u64,
    pub hypothetical_bytes_avoided: u64,
    pub net_bytes_delta: i128,
    pub plan_sha256: String,
}

fn validate_digest(value: &str, field: &str) -> Result<()> {
    ensure!(
        value.len() == 64
            && value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
        "{field} must be a SHA-256 hex digest"
    );
    Ok(())
}

fn sum_bytes(objects: &[PhysicalObject]) -> Result<u64> {
    objects.iter().try_fold(0u64, |sum, object| {
        sum.checked_add(object.bytes)
            .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
    })
}

fn replacement_for(
    candidate: &Candidate,
    policy: &RetentionPolicyV2,
    current_objects: &BTreeSet<PhysicalObject>,
    keeper_objects: &BTreeSet<PhysicalObject>,
) -> Result<Option<(Recommendation, PhysicalObject)>> {
    let Some(proof) = &candidate.replacement else {
        return Ok(None);
    };
    match proof {
        ReplacementProof::ExactDuplicate {
            replacement,
            content_hash_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            if candidate.sources.len() != 1 {
                return Ok(None);
            }
            let source = &candidate.sources[0];
            if *content_hash_verified
                && *restore_tested
                && source.sha256 == replacement.sha256
                && source.bytes == replacement.bytes
                && source != replacement
                && current_objects.contains(replacement)
                && keeper_objects.contains(replacement)
            {
                Ok(Some((Recommendation::ReviewDuplicate, replacement.clone())))
            } else {
                Ok(None)
            }
        }
        ReplacementProof::Pack {
            replacement,
            members,
            manifest_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            let source_set: BTreeSet<_> = candidate.sources.iter().cloned().collect();
            let member_set: BTreeSet<_> = members.iter().cloned().collect();
            if *manifest_verified
                && *restore_tested
                && source_set.len() == candidate.sources.len()
                && source_set == member_set
                && member_set.len() == members.len()
                && !current_objects.contains(replacement)
            {
                Ok(Some((Recommendation::ReviewPack, replacement.clone())))
            } else {
                Ok(None)
            }
        }
        ReplacementProof::Supersession {
            replacement,
            qualified_champion,
            relative_margin,
            days_since_dominance,
            score_rows_retained,
            recovery_closure_verified,
            restore_tested,
        } => {
            replacement.validate()?;
            ensure!(
                relative_margin.is_finite() && *relative_margin >= 0.0,
                "supersession margin must be finite and nonnegative"
            );
            if candidate.sources.len() == 1
                && *qualified_champion
                && *relative_margin >= policy.margin_rel
                && *days_since_dominance >= policy.grace_days
                && *score_rows_retained
                && *recovery_closure_verified
                && *restore_tested
                && current_objects.contains(replacement)
                && keeper_objects.contains(replacement)
            {
                Ok(Some((
                    Recommendation::ReviewSupersession,
                    replacement.clone(),
                )))
            } else {
                Ok(None)
            }
        }
    }
}

fn replacement_gap(
    candidate: &Candidate,
    policy: &RetentionPolicyV2,
    current_objects: &BTreeSet<PhysicalObject>,
    keeper_objects: &BTreeSet<PhysicalObject>,
) -> String {
    match candidate.replacement.as_ref() {
        None => "replacement or recovery proof is missing".into(),
        Some(ReplacementProof::ExactDuplicate {
            replacement,
            content_hash_verified,
            restore_tested,
        }) => {
            if !content_hash_verified {
                "exact duplicate content hash is not independently verified".into()
            } else if !restore_tested {
                "duplicate recovery restore has not been tested".into()
            } else if !current_objects.contains(replacement) {
                "exact primary version is absent from the current inventory".into()
            } else if !keeper_objects.contains(replacement) {
                "exact primary version is not protected as a keeper".into()
            } else {
                "duplicate physical identity, bytes, or key comparison is incomplete".into()
            }
        }
        Some(ReplacementProof::Pack {
            replacement,
            members,
            manifest_verified,
            restore_tested,
        }) => {
            if !manifest_verified {
                "pack manifest does not verify the exact source closure".into()
            } else if !restore_tested {
                "pack recovery restore has not been tested".into()
            } else if current_objects.contains(replacement) {
                "pack output already exists in the current inventory; use an exact-version proof"
                    .into()
            } else if members.iter().cloned().collect::<BTreeSet<_>>()
                != candidate.sources.iter().cloned().collect()
                || members.iter().cloned().collect::<BTreeSet<_>>().len() != members.len()
            {
                "pack member versions do not exactly match the proposed source closure".into()
            } else {
                "pack proof is incomplete".into()
            }
        }
        Some(ReplacementProof::Supersession {
            replacement,
            qualified_champion,
            relative_margin,
            days_since_dominance,
            score_rows_retained,
            recovery_closure_verified,
            restore_tested,
            ..
        }) => {
            if !qualified_champion {
                "superseding champion is not qualified".into()
            } else if !relative_margin.is_finite() || *relative_margin < policy.margin_rel {
                "qualified champion does not meet the configured dominance margin".into()
            } else if *days_since_dominance < policy.grace_days {
                "qualified dominance has not completed the configured grace period".into()
            } else if !score_rows_retained {
                "supersession proof does not retain the score rows".into()
            } else if !recovery_closure_verified {
                "supersession recovery closure is not verified".into()
            } else if !restore_tested {
                "supersession recovery restore has not been tested".into()
            } else if !current_objects.contains(replacement) {
                "qualified replacement version is absent from the current inventory".into()
            } else if !keeper_objects.contains(replacement) {
                "qualified replacement version is not protected as a keeper".into()
            } else {
                "supersession candidate does not have one exact source object".into()
            }
        }
    }
}

impl RetentionPlanV2 {
    /// Build a deterministic, in-memory plan. No operation is performed on source objects.
    pub fn build(
        as_of_unix: u64,
        inventory_max_age_secs: u64,
        source_inventory_sha256: &str,
        policy: RetentionPolicyV2,
        mut candidates: Vec<Candidate>,
    ) -> Result<Self> {
        policy.validate()?;
        validate_digest(source_inventory_sha256, "source inventory digest")?;
        ensure!(
            inventory_max_age_secs > 0,
            "inventory maximum age must be positive"
        );
        candidates.sort_by(|a, b| a.sources.cmp(&b.sources));
        let current_objects: BTreeSet<PhysicalObject> = candidates
            .iter()
            .flat_map(|candidate| candidate.sources.iter().cloned())
            .collect();
        let keeper_objects: BTreeSet<PhysicalObject> = candidates
            .iter()
            .filter(|candidate| {
                candidate.hard_t0
                    || candidate.active_pin
                    || candidate.active_promise
                    || candidate.score_row
                    || candidate.reservoir_selected == Some(true)
                    || candidate.ranking_current
                        && (candidate
                            .context_rank
                            .is_some_and(|rank| rank <= policy.k_ctx)
                            || candidate
                                .lineage_rank
                                .is_some_and(|rank| rank <= policy.k_lineage))
            })
            .flat_map(|candidate| candidate.sources.iter().cloned())
            .collect();
        let mut seen = BTreeSet::new();
        let mut items = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            ensure!(
                !candidate.sources.is_empty(),
                "candidate has no source objects"
            );
            for source in &candidate.sources {
                source.validate()?;
                ensure!(
                    seen.insert(source.clone()),
                    "duplicate physical source in candidates: {}",
                    source.key
                );
            }
            if let Some(signals) = &candidate.signals {
                signals.validate()?;
            }
            let current_bytes = sum_bytes(&candidate.sources)?;
            let pinned = candidate.hard_t0 || candidate.active_pin || candidate.active_promise;
            let values = candidate
                .sources
                .iter()
                .map(|source| {
                    RetentionValue::classify(
                        source,
                        pinned,
                        candidate.score_row,
                        candidate.signals.as_ref(),
                        &policy,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let mut unresolved = Vec::new();
            let mut recommendation = Recommendation::Hold;
            let mut replacement = None;
            let mut reason = String::new();
            if candidate.hard_t0
                || candidate.active_pin
                || candidate.active_promise
                || candidate.score_row
            {
                recommendation = if candidate.hard_t0 {
                    Recommendation::KeepT0
                } else if candidate.score_row {
                    Recommendation::KeepScoreRows
                } else {
                    Recommendation::Keep
                };
                reason = if candidate.hard_t0 {
                    "hard T0 obligation"
                } else if candidate.score_row {
                    "score rows are retained"
                } else if candidate.active_pin {
                    "active evidence pin"
                } else {
                    "active future promise"
                }
                .into();
            } else if !candidate.ranking_current
                || candidate.context_rank.is_none()
                || candidate.lineage_rank.is_none()
                || candidate.reservoir_selected.is_none()
            {
                unresolved
                    .push("current keeper ranking and reservoir evidence is incomplete".into());
                reason = "no change proposed because keeper status is unknown".into();
            } else if candidate.context_rank.unwrap_or(u64::MAX) <= policy.k_ctx
                || candidate.lineage_rank.unwrap_or(u64::MAX) <= policy.k_lineage
                || candidate.reservoir_selected == Some(true)
            {
                recommendation = Recommendation::Keep;
                reason = "retained by context, lineage, or reservoir obligation".into();
            } else {
                match candidate.inventory_at_unix {
                    Some(at) if at <= as_of_unix && as_of_unix - at <= inventory_max_age_secs => {}
                    Some(_) => unresolved.push("inventory is stale for this plan date".into()),
                    None => unresolved.push("current inventory evidence is missing".into()),
                }
                match &candidate.consumer_closure {
                    Some(scan)
                        if scan.scanned_at_unix <= as_of_unix
                            && !scan.observed_root_generation.is_empty()
                            && scan.observed_root_generation == scan.current_root_generation
                            && scan.active_consumers.is_empty()
                            && scan.unresolved.is_empty() => {}
                    Some(scan) => {
                        if scan.scanned_at_unix > as_of_unix {
                            unresolved.push("consumer scan postdates plan".into());
                        }
                        if scan.observed_root_generation.is_empty()
                            || scan.observed_root_generation != scan.current_root_generation
                        {
                            unresolved
                                .push("consumer root generation changed or is unknown".into());
                        }
                        unresolved.extend(
                            scan.active_consumers
                                .iter()
                                .map(|c| format!("active consumer: {c}")),
                        );
                        unresolved
                            .extend(scan.unresolved.iter().map(|g| format!("consumer gap: {g}")));
                    }
                    None => unresolved.push("consumer and recovery closure is missing".into()),
                }
                if let Some((kind, object)) =
                    replacement_for(&candidate, &policy, &current_objects, &keeper_objects)?
                {
                    replacement = Some(object);
                    if unresolved.is_empty() {
                        recommendation = kind;
                        reason = match kind {
                            Recommendation::ReviewDuplicate => "exact duplicate has verified bytes and tested recovery; owner review required".into(),
                            Recommendation::ReviewPack => "pack manifest and restore are verified; owner review required".into(),
                            Recommendation::ReviewSupersession => "qualified champion dominance passed margin and grace; score rows and recovery remain; owner review required".into(),
                            _ => unreachable!(),
                        };
                    }
                } else {
                    unresolved.push(replacement_gap(
                        &candidate,
                        &policy,
                        &current_objects,
                        &keeper_objects,
                    ));
                }
                if !unresolved.is_empty() {
                    reason = "no change proposed because current proof is incomplete".into();
                }
            }
            let replacement_bytes = match recommendation {
                Recommendation::ReviewDuplicate | Recommendation::ReviewSupersession => 0,
                Recommendation::ReviewPack => {
                    replacement.as_ref().map_or(current_bytes, |r| r.bytes)
                }
                _ => current_bytes,
            };
            let hypothetical_bytes_after_review = replacement_bytes;
            items.push(PlanItem {
                source: candidate.sources,
                replacement,
                recommendation,
                reason,
                unresolved,
                values,
                current_bytes,
                hypothetical_bytes_after_review,
            });
        }
        items.sort_by(|a, b| {
            fn priority(item: &PlanItem) -> u8 {
                match item.recommendation {
                    Recommendation::ReviewDuplicate => 0,
                    Recommendation::ReviewSupersession => 1,
                    Recommendation::ReviewPack => 2,
                    Recommendation::Hold => 3,
                    Recommendation::KeepT0 => 4,
                    Recommendation::KeepScoreRows => 5,
                    Recommendation::Keep => 6,
                }
            }
            let max_value = |item: &PlanItem| {
                item.values
                    .iter()
                    .filter_map(|v| v.marginal_value)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            let max_surprise = |item: &PlanItem| {
                item.values
                    .iter()
                    .filter_map(|v| v.surprise)
                    .fold(f64::NEG_INFINITY, f64::max)
            };
            priority(a)
                .cmp(&priority(b))
                .then_with(|| max_value(b).total_cmp(&max_value(a)))
                .then_with(|| max_surprise(b).total_cmp(&max_surprise(a)))
                .then_with(|| a.source.cmp(&b.source))
        });
        let current_objects = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.source.len() as u64)
                .ok_or_else(|| anyhow::anyhow!("object total overflow"))
        })?;
        let current_bytes = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.current_bytes)
                .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
        })?;
        let hypothetical_objects_after_review = items.iter().try_fold(0u64, |n, item| {
            let count = match item.recommendation {
                Recommendation::ReviewDuplicate | Recommendation::ReviewSupersession => 0,
                Recommendation::ReviewPack => 1,
                _ => item.source.len() as u64,
            };
            n.checked_add(count)
                .ok_or_else(|| anyhow::anyhow!("object total overflow"))
        })?;
        let hypothetical_bytes_after_review = items.iter().try_fold(0u64, |n, item| {
            n.checked_add(item.hypothetical_bytes_after_review)
                .ok_or_else(|| anyhow::anyhow!("byte total overflow"))
        })?;
        let hypothetical_bytes_avoided =
            current_bytes.saturating_sub(hypothetical_bytes_after_review);
        let mut plan = Self {
            schema: RETENTION_PLAN_V2_SCHEMA.into(),
            as_of_unix,
            inventory_max_age_secs,
            source_inventory_sha256: source_inventory_sha256.into(),
            policy,
            items,
            current_objects,
            current_bytes,
            hypothetical_objects_after_review,
            hypothetical_bytes_after_review,
            hypothetical_bytes_avoided,
            net_bytes_delta: current_bytes as i128 - hypothetical_bytes_after_review as i128,
            plan_sha256: String::new(),
        };
        let bytes = serde_json::to_vec(&plan)?;
        plan.plan_sha256 = sha256_hex(&bytes);
        Ok(plan)
    }
}
