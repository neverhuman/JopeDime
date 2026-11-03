use super::*;
use std::collections::BTreeMap;

pub(crate) struct PolicyEvaluation {
    pub(crate) archive_leaf_genomes: Vec<String>,
    pub(crate) active_support_leaves: usize,
    pub(crate) tilted_leaf_log_probs: Option<Vec<f64>>,
    pub(crate) kl_source: String,
    pub(crate) kl_malformed_reason: Option<String>,
    pub(crate) policy_archive_ce_loss: Option<f64>,
    pub(crate) decomposed_kl: Option<crate::support::DecomposedSupportKl>,
    pub(crate) seed_update_frozen_by_kl_guard: Option<bool>,
}

pub(crate) fn evaluate_policy_guard(
    round_idx: usize,
    round_leaf_log_prob_fn: Option<RoundLeafLogProbFn<'_>>,
    measured_tilt_policy: Option<&MeasuredTiltPolicyConfig>,
    base_catalog_ordered: &[String],
    archive_best_rmse: &BTreeMap<String, f64>,
    base_best_q: &BTreeMap<String, f64>,
    bootstrap_q: f64,
    base_leaf_count: usize,
    max_conditional_base_kl: f64,
    max_new_support_mass: f64,
    measured_tilt_rounds_admissible: &mut usize,
) -> PolicyEvaluation {
    let archive_leaf_genomes: Vec<String> = archive_best_rmse.keys().cloned().collect();
    let n_new = archive_leaf_genomes.len();
    let active_support_leaves = base_leaf_count + n_new;
    let mut is_base = Vec::with_capacity(active_support_leaves);
    is_base.extend(std::iter::repeat_n(true, base_leaf_count));
    is_base.extend(std::iter::repeat_n(false, n_new));

    let mut tilted_leaf_log_probs: Option<Vec<f64>> = None;
    let (
        kl_source,
        kl_malformed_reason,
        policy_archive_ce_loss,
        decomposed_kl,
        seed_update_frozen_by_kl_guard,
    ) = if let Some(eval_fn) = round_leaf_log_prob_fn {
        let (pol_lp, ref_lp) = eval_fn(round_idx, base_catalog_ordered, &archive_leaf_genomes);
        if pol_lp.len() == active_support_leaves && ref_lp.len() == active_support_leaves {
            match crate::support::decomposed_support_kl(
                &pol_lp,
                &ref_lp,
                &is_base,
                max_conditional_base_kl,
                max_new_support_mass,
            ) {
                Ok(report) => {
                    let frozen = !report.admissible;
                    (
                        "caller_evaluated_leaf_log_probs".to_string(),
                        None,
                        None,
                        Some(report),
                        Some(frozen),
                    )
                }
                Err(err) => (
                    "caller_evaluated_leaf_log_probs_malformed".to_string(),
                    Some(format!("{err:#}")),
                    None,
                    None,
                    Some(true),
                ),
            }
        } else {
            (
                "caller_evaluated_leaf_log_probs_malformed".to_string(),
                Some(format!(
                    "expected {} leaf log-probs, got policy={} ref={}",
                    active_support_leaves,
                    pol_lp.len(),
                    ref_lp.len()
                )),
                None,
                None,
                Some(true),
            )
        }
    } else if let Some(tilt_cfg) = measured_tilt_policy {
        let base_q_vec: Vec<f64> = base_catalog_ordered
            .iter()
            .map(|g| base_best_q.get(g).copied().unwrap_or(bootstrap_q))
            .collect();
        let arch_q_vec: Vec<f64> = archive_leaf_genomes
            .iter()
            .map(|g| 1.0 / (1.0 + archive_best_rmse[g]))
            .collect();
        match measured_tilt_leaf_log_probs(&base_q_vec, &arch_q_vec, bootstrap_q, tilt_cfg)
            .and_then(|(pol_lp, ref_lp, ce_loss)| {
                let report = crate::support::decomposed_support_kl(
                    &pol_lp,
                    &ref_lp,
                    &is_base,
                    max_conditional_base_kl,
                    max_new_support_mass,
                )?;
                Ok((pol_lp, ce_loss, report))
            }) {
            Ok((pol_lp, ce_loss, report)) => {
                let frozen = !report.admissible;
                if report.admissible {
                    *measured_tilt_rounds_admissible += 1;
                    tilted_leaf_log_probs = Some(pol_lp);
                }
                (
                    "measured_tilt_leaf_policy".to_string(),
                    None,
                    Some(ce_loss),
                    Some(report),
                    Some(frozen),
                )
            }
            Err(err) => (
                "measured_tilt_leaf_policy_malformed".to_string(),
                Some(format!("{err:#}")),
                None,
                None,
                Some(true),
            ),
        }
    } else {
        (
            "unmeasured_no_policy_log_probs".to_string(),
            None,
            None,
            None,
            None,
        )
    };

    PolicyEvaluation {
        archive_leaf_genomes,
        active_support_leaves,
        tilted_leaf_log_probs,
        kl_source,
        kl_malformed_reason,
        policy_archive_ce_loss,
        decomposed_kl,
        seed_update_frozen_by_kl_guard,
    }
}
