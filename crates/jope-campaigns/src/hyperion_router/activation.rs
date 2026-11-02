use super::*;

pub(super) fn rmse_slice(truth: &[f64], predictions: &[f64]) -> Result<f64, String> {
    if truth.is_empty() || truth.len() != predictions.len() {
        return Err("RMSE vectors are empty or have different lengths".into());
    }
    require_finite("RMSE truth", truth)?;
    require_finite("RMSE predictions", predictions)?;
    Ok((truth
        .iter()
        .zip(predictions)
        .map(|(truth, prediction)| (truth - prediction).powi(2))
        .sum::<f64>()
        / truth.len() as f64)
        .sqrt())
}

fn rmse_rows(truth: &[f64], predictions: &[f64], rows: &[usize]) -> Result<f64, String> {
    if rows.is_empty() || truth.len() != predictions.len() {
        return Err("fold RMSE rows or vectors are invalid".into());
    }
    let mut sum = 0.0;
    for row in rows {
        let actual = *truth
            .get(*row)
            .ok_or_else(|| "fold RMSE row is out of range".to_string())?;
        let predicted = predictions[*row];
        if !actual.is_finite() || !predicted.is_finite() {
            return Err("fold RMSE contains a nonfinite value".into());
        }
        sum += (actual - predicted).powi(2);
    }
    Ok((sum / rows.len() as f64).sqrt())
}

fn population_standard_deviation(values: &[f64]) -> Result<f64, String> {
    if values.is_empty() {
        return Err("target standard deviation requires values".into());
    }
    require_finite("training targets", values)?;
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let standard_deviation = (values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / values.len() as f64)
        .sqrt();
    if !standard_deviation.is_finite() || standard_deviation <= 0.0 {
        return Err("training target standard deviation is zero or unstable".into());
    }
    Ok(standard_deviation)
}

fn deterministic_draw_index(tag: &str, repetition: usize, draw: usize, len: usize) -> usize {
    let digest = Sha256::digest(format!("{tag}|{repetition}|{draw}").as_bytes());
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    (u64::from_le_bytes(bytes) as usize) % len
}

fn grouped_bootstrap_lower_bound(
    seed: u64,
    route: RouteFamilyV2,
    truth: &[f64],
    raw_predictions: &[f64],
    route_predictions: &[f64],
    row_atomic_groups: &[String],
) -> Result<f64, String> {
    if truth.len() != raw_predictions.len()
        || truth.len() != route_predictions.len()
        || truth.len() != row_atomic_groups.len()
        || truth.is_empty()
    {
        return Err("grouped bootstrap vectors have different lengths".into());
    }
    require_finite("grouped bootstrap truth", truth)?;
    require_finite("grouped bootstrap raw", raw_predictions)?;
    require_finite("grouped bootstrap route", route_predictions)?;
    let mut groups = BTreeMap::<String, Vec<usize>>::new();
    for (row, group) in row_atomic_groups.iter().enumerate() {
        if group.is_empty() {
            return Err("grouped bootstrap contains an empty group ID".into());
        }
        groups.entry(group.clone()).or_default().push(row);
    }
    if groups.len() < MINIMUM_ATOMIC_GROUPS {
        return Err("grouped bootstrap requires at least fourteen atomic groups".into());
    }
    let groups = groups.into_values().collect::<Vec<_>>();
    let tag = format!("hyperion-router-grouped-bootstrap-v2|{seed}|{route:?}");
    let mut improvements = Vec::with_capacity(GROUPED_BOOTSTRAP_REPETITIONS);
    for repetition in 0..GROUPED_BOOTSTRAP_REPETITIONS {
        let mut raw_squared_error = 0.0;
        let mut route_squared_error = 0.0;
        let mut row_count = 0usize;
        for draw in 0..groups.len() {
            let group = &groups[deterministic_draw_index(&tag, repetition, draw, groups.len())];
            for row in group {
                raw_squared_error += (truth[*row] - raw_predictions[*row]).powi(2);
                route_squared_error += (truth[*row] - route_predictions[*row]).powi(2);
                row_count += 1;
            }
        }
        let raw_rmse = (raw_squared_error / row_count as f64).sqrt();
        let route_rmse = (route_squared_error / row_count as f64).sqrt();
        improvements.push(raw_rmse - route_rmse);
    }
    improvements.sort_by(f64::total_cmp);
    let lower_index = (GROUPED_BOOTSTRAP_REPETITIONS * 5) / 100;
    Ok(improvements[lower_index])
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedActivationEvidenceV2 {
    pub schema: String,
    pub seed: u64,
    pub route: RouteFamilyV2,
    pub raw_pooled_rmse: f64,
    pub route_pooled_rmse: f64,
    pub training_target_standard_deviation: f64,
    pub relative_rmse_improvement: f64,
    pub normalized_rmse_improvement: f64,
    pub raw_fold_rmse: Vec<f64>,
    pub route_fold_rmse: Vec<f64>,
    pub fold_wins: u32,
    pub grouped_bootstrap_lower_bound: f64,
    pub worst_fold_regression: f64,
    pub grouped_bootstrap_repetitions: u32,
}

impl SeedActivationEvidenceV2 {
    pub const SCHEMA: &'static str = "HyperionSeedActivationEvidence.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || !ROUTER_SEEDS.contains(&self.seed)
            || self.route == RouteFamilyV2::Raw
            || self.raw_fold_rmse.len() != OUTER_FOLDS
            || self.route_fold_rmse.len() != OUTER_FOLDS
            || self.grouped_bootstrap_repetitions != GROUPED_BOOTSTRAP_REPETITIONS as u32
            || self.fold_wins > OUTER_FOLDS as u32
        {
            return Err("seed activation evidence schema, route, or fold coverage drifted".into());
        }
        let finite = [
            self.raw_pooled_rmse,
            self.route_pooled_rmse,
            self.training_target_standard_deviation,
            self.relative_rmse_improvement,
            self.normalized_rmse_improvement,
            self.grouped_bootstrap_lower_bound,
            self.worst_fold_regression,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite
            || self.raw_pooled_rmse < 0.0
            || self.route_pooled_rmse < 0.0
            || self.training_target_standard_deviation <= 0.0
            || self
                .raw_fold_rmse
                .iter()
                .chain(&self.route_fold_rmse)
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("seed activation evidence contains invalid metrics".into());
        }
        let expected_relative = if self.raw_pooled_rmse > 0.0 {
            (self.raw_pooled_rmse - self.route_pooled_rmse) / self.raw_pooled_rmse
        } else {
            0.0
        };
        let expected_normalized = (self.raw_pooled_rmse - self.route_pooled_rmse)
            / self.training_target_standard_deviation;
        let expected_wins = self
            .raw_fold_rmse
            .iter()
            .zip(&self.route_fold_rmse)
            .filter(|(raw, route)| route < raw)
            .count() as u32;
        let expected_worst = self
            .raw_fold_rmse
            .iter()
            .zip(&self.route_fold_rmse)
            .map(|(raw, route)| {
                if *raw > 0.0 {
                    (route - raw) / raw
                } else if *route == 0.0 {
                    0.0
                } else {
                    f64::INFINITY
                }
            })
            .fold(f64::NEG_INFINITY, f64::max);
        if (self.relative_rmse_improvement - expected_relative).abs() > 1e-12
            || (self.normalized_rmse_improvement - expected_normalized).abs() > 1e-12
            || self.fold_wins != expected_wins
            || (self.worst_fold_regression - expected_worst).abs() > 1e-12
        {
            return Err(
                "seed activation evidence does not reconstruct from its RMSE values".into(),
            );
        }
        Ok(())
    }

    pub fn passes(&self, thresholds: ActivationThresholdsV2) -> bool {
        self.validate().is_ok()
            && thresholds == ActivationThresholdsV2::default()
            && self.relative_rmse_improvement
                >= thresholds.minimum_seed_relative_improvement_micros as f64 / 1_000_000.0
            && self.normalized_rmse_improvement
                >= thresholds.minimum_seed_normalized_improvement_micros as f64 / 1_000_000.0
            && self.fold_wins >= thresholds.minimum_fold_wins
            && self.grouped_bootstrap_lower_bound > 0.0
            && self.worst_fold_regression
                <= thresholds.maximum_fold_regression_micros as f64 / 1_000_000.0
    }
}

pub fn evaluate_seed_activation_v2(
    seed: u64,
    route: RouteFamilyV2,
    truth: &[f64],
    raw_predictions: &[f64],
    route_predictions: &[f64],
    plan: &GroupedOuterPlanV2,
) -> Result<SeedActivationEvidenceV2, String> {
    plan.validate()?;
    if plan.seed != seed || route == RouteFamilyV2::Raw {
        return Err("activation evidence seed or route is invalid".into());
    }
    if truth.len() != plan.row_count
        || raw_predictions.len() != plan.row_count
        || route_predictions.len() != plan.row_count
    {
        return Err("activation predictions do not cover the grouped outer plan".into());
    }
    let raw_pooled_rmse = rmse_slice(truth, raw_predictions)?;
    let route_pooled_rmse = rmse_slice(truth, route_predictions)?;
    let target_sd = population_standard_deviation(truth)?;
    let raw_fold_rmse = plan
        .folds
        .iter()
        .map(|fold| rmse_rows(truth, raw_predictions, &fold.hold_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let route_fold_rmse = plan
        .folds
        .iter()
        .map(|fold| rmse_rows(truth, route_predictions, &fold.hold_rows))
        .collect::<Result<Vec<_>, _>>()?;
    let fold_wins = raw_fold_rmse
        .iter()
        .zip(&route_fold_rmse)
        .filter(|(raw, route)| route < raw)
        .count() as u32;
    let worst_fold_regression = raw_fold_rmse
        .iter()
        .zip(&route_fold_rmse)
        .map(|(raw, route)| {
            if *raw > 0.0 {
                (route - raw) / raw
            } else if *route == 0.0 {
                0.0
            } else {
                f64::INFINITY
            }
        })
        .fold(f64::NEG_INFINITY, f64::max);
    let evidence = SeedActivationEvidenceV2 {
        schema: SeedActivationEvidenceV2::SCHEMA.into(),
        seed,
        route,
        raw_pooled_rmse,
        route_pooled_rmse,
        training_target_standard_deviation: target_sd,
        relative_rmse_improvement: if raw_pooled_rmse > 0.0 {
            (raw_pooled_rmse - route_pooled_rmse) / raw_pooled_rmse
        } else {
            0.0
        },
        normalized_rmse_improvement: (raw_pooled_rmse - route_pooled_rmse) / target_sd,
        raw_fold_rmse,
        route_fold_rmse,
        fold_wins,
        grouped_bootstrap_lower_bound: grouped_bootstrap_lower_bound(
            seed,
            route,
            truth,
            raw_predictions,
            route_predictions,
            &plan.row_atomic_groups,
        )?,
        worst_fold_regression,
        grouped_bootstrap_repetitions: GROUPED_BOOTSTRAP_REPETITIONS as u32,
    };
    evidence.validate()?;
    Ok(evidence)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouteCandidateV2 {
    pub route: RouteFamilyV2,
    pub seeds: Vec<SeedActivationEvidenceV2>,
    pub common_fold_mean_rmse: f64,
    pub common_fold_standard_error: f64,
    pub blend_coefficients: Vec<f64>,
}

impl RouteCandidateV2 {
    pub fn validate(&self, thresholds: ActivationThresholdsV2) -> Result<(), String> {
        if self.route == RouteFamilyV2::Raw
            || self.seeds.len() != ROUTER_SEEDS.len()
            || !self.common_fold_mean_rmse.is_finite()
            || self.common_fold_mean_rmse < 0.0
            || !self.common_fold_standard_error.is_finite()
            || self.common_fold_standard_error < 0.0
        {
            return Err("nonraw route candidate is malformed".into());
        }
        for (index, evidence) in self.seeds.iter().enumerate() {
            evidence.validate()?;
            if evidence.seed != ROUTER_SEEDS[index] || evidence.route != self.route {
                return Err("route candidate seeds disagree on route family".into());
            }
        }
        if self.route == RouteFamilyV2::RawCellBlend {
            if self.blend_coefficients.len() != ROUTER_SEEDS.len()
                || self
                    .blend_coefficients
                    .iter()
                    .any(|alpha| !alpha.is_finite() || !(0.0..=0.5).contains(alpha))
            {
                return Err("blend route lacks two clipped train-only coefficients".into());
            }
        } else if !self.blend_coefficients.is_empty() {
            return Err("non-blend route carries blend coefficients".into());
        }
        if thresholds != ActivationThresholdsV2::default() {
            return Err("activation thresholds cannot be weakened or replaced".into());
        }
        Ok(())
    }

    pub fn robustly_activates(&self, thresholds: ActivationThresholdsV2) -> bool {
        if self.validate(thresholds).is_err()
            || !self
                .seeds
                .iter()
                .all(|evidence| evidence.passes(thresholds))
        {
            return false;
        }
        let median = (self.seeds[0].relative_rmse_improvement
            + self.seeds[1].relative_rmse_improvement)
            * 0.5;
        median >= thresholds.minimum_two_seed_median_improvement_micros as f64 / 1_000_000.0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterDecisionV2 {
    pub schema: String,
    pub route: RouteFamilyV2,
    pub reason: String,
    pub thresholds: ActivationThresholdsV2,
    pub selected_candidate_sha256: Option<String>,
}

impl RouterDecisionV2 {
    pub const SCHEMA: &'static str = "HyperionRouterDecision.v2";

    pub fn identity(&self) -> Result<String, String> {
        if self.schema != Self::SCHEMA
            || self.reason.trim().is_empty()
            || self.thresholds != ActivationThresholdsV2::default()
            || (self.route == RouteFamilyV2::Raw) != self.selected_candidate_sha256.is_none()
        {
            return Err("router decision is malformed or uses nonfrozen thresholds".into());
        }
        if let Some(digest) = &self.selected_candidate_sha256 {
            validate_sha256("selected route candidate", digest)?;
        }
        canonical_sha256(self)
    }
}

/// Apply every gate first, then the common-fold 0.5-SE conservative rule.
/// Preference inside the acceptance band is raw, cell-only, append, blend.
pub fn choose_route_v2(
    raw_common_fold_mean_rmse: f64,
    candidates: &[RouteCandidateV2],
    upstream_failure: Option<&str>,
) -> Result<RouterDecisionV2, String> {
    let thresholds = ActivationThresholdsV2::default();
    if !raw_common_fold_mean_rmse.is_finite() || raw_common_fold_mean_rmse < 0.0 {
        return Err("matched raw common-fold RMSE is invalid".into());
    }
    if let Some(failure) = upstream_failure {
        if failure.trim().is_empty() {
            return Err("upstream failure reason is empty".into());
        }
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: format!("predeclared raw fallback: {failure}"),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    let mut seen = BTreeSet::new();
    for candidate in candidates {
        candidate.validate(thresholds)?;
        if !seen.insert(candidate.route) {
            return Err("route candidate family is duplicated".into());
        }
    }
    let mut passing = candidates
        .iter()
        .filter(|candidate| candidate.robustly_activates(thresholds))
        .collect::<Vec<_>>();
    if passing.is_empty() {
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: "no nonraw route passed both-seed activation gates".into(),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    passing.sort_by(|a, b| {
        a.common_fold_mean_rmse
            .total_cmp(&b.common_fold_mean_rmse)
            .then_with(|| a.route.preference().cmp(&b.route.preference()))
    });
    let best = passing[0];
    let acceptance_limit = best.common_fold_mean_rmse + 0.5 * best.common_fold_standard_error;
    if raw_common_fold_mean_rmse <= acceptance_limit {
        return Ok(RouterDecisionV2 {
            schema: RouterDecisionV2::SCHEMA.into(),
            route: RouteFamilyV2::Raw,
            reason: "matched raw is inside the frozen 0.5-SE acceptance band".into(),
            thresholds,
            selected_candidate_sha256: None,
        });
    }
    let selected = passing
        .into_iter()
        .filter(|candidate| candidate.common_fold_mean_rmse <= acceptance_limit)
        .min_by_key(|candidate| candidate.route.preference())
        .expect("best candidate is inside its own acceptance band");
    Ok(RouterDecisionV2 {
        schema: RouterDecisionV2::SCHEMA.into(),
        route: selected.route,
        reason: "nonraw route passed both seeds and won the frozen 0.5-SE preference rule".into(),
        thresholds,
        selected_candidate_sha256: Some(canonical_sha256(selected)?),
    })
}
