//! Prospective, fixture-first qualification for best-of-N distillation teachers.
//!
//! This module is deliberately a read-only decision function. It neither trains a model nor
//! grants serving or campaign authority. The contract is a draft until the I16/I19/I30/I35
//! schemas and callers are accepted by the integrator.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const TEACHER_CONTRAST_SCHEMA: &str = "jope.teacher-contrast/v1";
pub const TEACHER_QUALIFICATION_SCHEMA: &str = "jope.teacher-qualification/v1";

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherQualification {
    pub schema: String,
    /// Frozen identifier for the raw-prior versus best-of-N contrast.
    pub contrast_id: String,
    pub raw_prior_id: String,
    pub best_of_n_teacher_id: String,
    pub candidate_ids: Vec<String>,
    pub selected_candidate_id: String,
    /// Explicitly declares the selection law used before outcomes were observed.
    pub selection_law: String,
    pub expected_utility_metric: String,
    pub minimum_mean_gain: f64,
    pub minimum_independent_contexts: usize,
    pub raw_prior_cost: ChargedCost,
    pub teacher_cost: ChargedCost,
    pub prior_support_sha256: String,
    pub teacher_support_sha256: String,
    pub contexts: Vec<DevelopmentContext>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ChargedCost {
    /// Integer microseconds make the frozen resource comparison exact.
    pub cpu_micros: u64,
    pub wall_micros: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DevelopmentContext {
    pub context_id: String,
    /// Must be unique across rows: contexts sharing a development source are not independent.
    pub independence_group_id: String,
    pub raw_prior_expected_utility: f64,
    pub teacher_expected_utility: f64,
    pub selected_root_won: bool,
    pub selected_candidate_id: String,
    pub included_candidate_ids: Vec<String>,
    pub targets: Vec<TeacherTarget>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TeacherTarget {
    pub row_id: String,
    pub action_id: String,
    pub action_probability: f64,
    pub support_sha256: String,
    pub measured: bool,
    pub source_id: String,
    pub source_sha256: String,
    pub evaluation_id: String,
    pub evaluation_sha256: String,
    /// Independently eligible labels can remain useful when the teacher gate fails.
    pub supervised_label_eligible: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DistillationAdmission {
    pub schema: &'static str,
    pub contrast_id: String,
    pub contrast_sha256: String,
    pub qualified: bool,
    pub policy_cross_entropy_authorized: bool,
    pub mean_expected_utility_gain: Option<f64>,
    pub independent_contexts: usize,
    pub eligible_policy_target_rows: Vec<String>,
    pub eligible_supervised_label_rows: Vec<String>,
    pub rejected_policy_target_rows: Vec<String>,
    pub reasons: Vec<String>,
}

/// Decide whether this frozen teacher contrast supports distillation policy targets.
///
/// This gate uses the expected policy utility measured on every independent development
/// context, not the selected root's win/loss indicator. All selected candidates must obey the
/// declared inclusion set, all contexts must use the same measured support and exact provenance,
/// and raw-prior and teacher charged resources must match. It is intentionally conservative:
/// every context's expected utility is included and no missing result is imputed.
pub fn decide(qualification: &TeacherQualification) -> DistillationAdmission {
    let mut reasons = Vec::new();
    let mut context_ids = BTreeSet::new();
    let mut independence_ids = BTreeSet::new();
    let mut expected_gains = Vec::new();
    let candidate_ids: BTreeSet<_> = qualification.candidate_ids.iter().collect();
    let mut policy_rows = Vec::new();
    let mut supervised_rows = Vec::new();
    let mut rejected_rows = Vec::new();
    let mut canonical_support: Option<BTreeSet<&str>> = None;
    let mut all_policy_targets_valid = true;

    if qualification.schema != TEACHER_CONTRAST_SCHEMA {
        reasons.push("unsupported contrast schema".to_owned());
    }
    if qualification.contrast_id.trim().is_empty()
        || qualification.raw_prior_id.trim().is_empty()
        || qualification.best_of_n_teacher_id.trim().is_empty()
        || qualification.expected_utility_metric.trim().is_empty()
    {
        reasons.push("contrast identifiers and utility metric must be nonempty".to_owned());
    }
    if qualification.selection_law != "best_of_n_selected_in_candidate_set/v1" {
        reasons.push("selection law is unsupported".to_owned());
    }
    if qualification.candidate_ids.is_empty()
        || candidate_ids.len() != qualification.candidate_ids.len()
        || !candidate_ids.contains(&&qualification.selected_candidate_id)
        || qualification.selected_candidate_id != qualification.best_of_n_teacher_id
    {
        reasons.push(
            "selected teacher is not uniquely included in the frozen candidate set".to_owned(),
        );
    }
    if qualification.raw_prior_cost != qualification.teacher_cost {
        reasons.push("raw-prior and best-of-N charged resources differ".to_owned());
    }
    if !is_hex64(&qualification.prior_support_sha256)
        || !is_hex64(&qualification.teacher_support_sha256)
        || qualification.prior_support_sha256 != qualification.teacher_support_sha256
    {
        reasons.push("raw-prior and teacher support identities differ or are invalid".to_owned());
    }
    if !qualification.minimum_mean_gain.is_finite() || qualification.minimum_mean_gain < 0.0 {
        reasons.push("minimum expected-utility gain must be finite and nonnegative".to_owned());
    }

    for context in &qualification.contexts {
        if context.context_id.trim().is_empty() || !context_ids.insert(&context.context_id) {
            reasons.push("development context identifiers must be nonempty and unique".to_owned());
        }
        if context.independence_group_id.trim().is_empty()
            || !independence_ids.insert(&context.independence_group_id)
        {
            reasons.push("development contexts must have distinct independence groups".to_owned());
        }
        if !context.raw_prior_expected_utility.is_finite()
            || !context.teacher_expected_utility.is_finite()
        {
            reasons.push(format!(
                "context {} has missing or non-finite expected utility",
                context.context_id
            ));
        } else {
            expected_gains
                .push(context.teacher_expected_utility - context.raw_prior_expected_utility);
        }
        let included: BTreeSet<_> = context.included_candidate_ids.iter().collect();
        if included.len() != context.included_candidate_ids.len()
            || included != candidate_ids
            || !included.contains(&&context.selected_candidate_id)
            || context.selected_candidate_id != qualification.selected_candidate_id
        {
            reasons.push(format!(
                "context {} violates the frozen selected-set inclusion law",
                context.context_id
            ));
        }
        if context.targets.is_empty() {
            reasons.push(format!(
                "context {} has no measured teacher targets",
                context.context_id
            ));
            all_policy_targets_valid = false;
        }
        for target in &context.targets {
            if target.supervised_label_eligible && !supervised_rows.contains(&target.row_id) {
                supervised_rows.push(target.row_id.clone());
            }
            let support: BTreeSet<_> = [target.support_sha256.as_str()].into_iter().collect();
            if canonical_support
                .as_ref()
                .is_some_and(|expected| expected != &support)
            {
                reasons.push(format!("target {} changed measured support", target.row_id));
                all_policy_targets_valid = false;
            } else if canonical_support.is_none() {
                canonical_support = Some(support);
            }
            let valid = !target.row_id.trim().is_empty()
                && !target.action_id.trim().is_empty()
                && target.action_probability.is_finite()
                && target.action_probability > 0.0
                && is_hex64(&target.support_sha256)
                && target.support_sha256 == qualification.teacher_support_sha256
                && target.measured
                && !target.source_id.trim().is_empty()
                && is_hex64(&target.source_sha256)
                && !target.evaluation_id.trim().is_empty()
                && is_hex64(&target.evaluation_sha256);
            if valid {
                if !policy_rows.contains(&target.row_id) {
                    policy_rows.push(target.row_id.clone());
                }
            } else {
                all_policy_targets_valid = false;
                if !rejected_rows.contains(&target.row_id) {
                    rejected_rows.push(target.row_id.clone());
                }
            }
        }
    }

    if qualification.contexts.len() < qualification.minimum_independent_contexts.max(1) {
        reasons.push("too few independent development contexts".to_owned());
    }
    let mean_gain = (!expected_gains.is_empty())
        .then(|| expected_gains.iter().sum::<f64>() / expected_gains.len() as f64);
    if let Some(gain) = mean_gain {
        if !gain.is_finite() || gain <= qualification.minimum_mean_gain {
            reasons
                .push("expected policy utility does not clear the preregistered gain".to_owned());
        }
    } else {
        reasons.push("no complete expected-utility contrast".to_owned());
    }
    if expected_gains.len() != qualification.contexts.len() {
        reasons.push("one or more development contexts lack a usable paired outcome".to_owned());
    }
    if policy_rows.is_empty() || !all_policy_targets_valid {
        reasons.push(
            "teacher targets are not all supported measured targets with exact provenance"
                .to_owned(),
        );
    }

    reasons.sort();
    reasons.dedup();
    let qualified = reasons.is_empty();
    let contrast_sha256 = serde_json::to_vec(qualification)
        .map(|bytes| format!("{:x}", Sha256::digest(bytes)))
        .unwrap_or_default();
    if !qualified {
        // An unqualified teacher never lends policy CE authority. Keep independently eligible
        // supervised labels in the separate output channel.
        for row in &policy_rows {
            if !rejected_rows.contains(row) {
                rejected_rows.push(row.clone());
            }
        }
        policy_rows.clear();
    }

    DistillationAdmission {
        schema: TEACHER_QUALIFICATION_SCHEMA,
        contrast_id: qualification.contrast_id.clone(),
        contrast_sha256,
        qualified,
        policy_cross_entropy_authorized: qualified,
        mean_expected_utility_gain: mean_gain,
        independent_contexts: independence_ids.len(),
        eligible_policy_target_rows: policy_rows,
        eligible_supervised_label_rows: supervised_rows,
        rejected_policy_target_rows: rejected_rows,
        reasons,
    }
}

fn is_hex64(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}
