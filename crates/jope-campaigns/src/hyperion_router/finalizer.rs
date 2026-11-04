use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizerEncodingProofV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub selection_sha256: String,
    pub requested_route: RouteFamilyV2,
    pub requested_predictor_sha256: String,
    pub authorized_full_train_cell_sha256: String,
    pub fresh_full_train_cell_sha256: String,
    pub authorized_test_cell_sha256: Option<String>,
    pub fresh_test_cell_sha256: String,
    pub encoder_identity_sha256: String,
    pub cuda_authenticated: bool,
    pub isolated_gpu: bool,
    pub dummy_test_targets: bool,
    pub labels_reached_hyperion: bool,
    pub train_shape_valid: bool,
    pub test_shape_valid: bool,
    pub row_order_valid: bool,
    pub finite_output: bool,
    pub test_row_count: usize,
    pub feature_width: usize,
    pub raw_latency_seconds: f64,
    pub routed_latency_seconds: f64,
}

impl FinalizerEncodingProofV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizerEncodingProof.v2";
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizerRouteDecisionV2 {
    pub schema: String,
    pub requested_route: RouteFamilyV2,
    pub chosen_route: RouteFamilyV2,
    pub chosen_predictor_sha256: String,
    pub decision_preceded_label_metrics: bool,
    pub fallback_reasons: Vec<String>,
}

impl FinalizerRouteDecisionV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizerRouteDecision.v2";
}

/// Validate every encoder/provenance condition without reading `stage.labels`.
/// A bad proof is data, not an exception: it deterministically chooses raw.
pub fn lock_finalizer_route_before_labels_v2(
    stage: &SealedFeatureStageV2,
    selection: &RouterSeedSelectionV2,
    policy: &RouterPolicyFreezeV2,
    proof: &FinalizerEncodingProofV2,
) -> Result<FinalizerRouteDecisionV2, String> {
    stage.validate()?;
    policy.validate()?;
    let policy_sha256 = policy.identity()?;
    selection.validate(&policy_sha256)?;
    let selection_sha256 = selection.identity(&policy_sha256)?;
    let mut failures = Vec::new();
    if proof.schema != FinalizerEncodingProofV2::SCHEMA {
        failures.push("encoding proof schema drift".into());
    }
    if proof.dataset_sha256 != stage.dataset_sha256
        || proof.dataset_sha256 != selection.dataset_sha256
    {
        failures.push("dataset identity mismatch".into());
    }
    if proof.selection_sha256 != selection_sha256
        || proof.requested_route != selection.route
        || proof.requested_predictor_sha256 != selection.selected_predictor_sha256
    {
        failures.push("route identity mismatch".into());
    }
    if proof.encoder_identity_sha256 != policy.encoder_identity_sha256 {
        failures.push("encoder identity mismatch".into());
    }
    if validate_sha256(
        "authorized train cell",
        &proof.authorized_full_train_cell_sha256,
    )
    .is_err()
        || proof.fresh_full_train_cell_sha256 != proof.authorized_full_train_cell_sha256
    {
        failures.push("fresh full-train cell output is not byte-identical to train3.vs".into());
    }
    if validate_sha256("fresh test cell", &proof.fresh_test_cell_sha256).is_err() {
        failures.push("fresh test cell hash is invalid".into());
    }
    if let Some(expected) = &proof.authorized_test_cell_sha256 {
        if validate_sha256("authorized test cell", expected).is_err()
            || &proof.fresh_test_cell_sha256 != expected
        {
            failures.push("fresh test cell output differs from its authorized pre-cache".into());
        }
    }
    if !proof.cuda_authenticated || !proof.isolated_gpu {
        failures.push("CUDA or isolated-GPU proof failed".into());
    }
    if !proof.dummy_test_targets || proof.labels_reached_hyperion {
        failures.push("test labels reached or could reach Hyperion".into());
    }
    if !proof.train_shape_valid
        || !proof.test_shape_valid
        || !proof.row_order_valid
        || !proof.finite_output
        || proof.test_row_count != stage.feature_rows.len()
        || proof.feature_width != stage.feature_count
    {
        failures.push("cell shape, order, finiteness, or row count failed".into());
    }
    if !proof.raw_latency_seconds.is_finite()
        || !proof.routed_latency_seconds.is_finite()
        || proof.raw_latency_seconds < 0.0
        || proof.routed_latency_seconds < 0.0
    {
        failures.push("latency proof is invalid".into());
    }
    let (chosen_route, chosen_predictor_sha256) = if failures.is_empty() {
        (selection.route, selection.selected_predictor_sha256.clone())
    } else {
        (
            RouteFamilyV2::Raw,
            selection.matched_raw_predictor_sha256.clone(),
        )
    };
    Ok(FinalizerRouteDecisionV2 {
        schema: FinalizerRouteDecisionV2::SCHEMA.into(),
        requested_route: selection.route,
        chosen_route,
        chosen_predictor_sha256,
        decision_preceded_label_metrics: true,
        fallback_reasons: failures,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizedSeedScoreV2 {
    pub schema: String,
    pub seed: u64,
    pub route_decision_sha256: String,
    pub old_production_raw_rmse: f64,
    pub matched_raw_gp_rmse: f64,
    pub routed_rmse: f64,
    pub ablation: SeedAblationV2,
    pub labels_reached_hyperion: bool,
    pub labels_reached_prediction_fit: bool,
    pub common_staged_data_and_evaluator: bool,
}

impl FinalizedSeedScoreV2 {
    pub const SCHEMA: &'static str = "HyperionFinalizedSeedScore.v2";
}

pub fn score_after_finalizer_lock_v2(
    stage: &SealedFeatureStageV2,
    seed: u64,
    decision: &FinalizerRouteDecisionV2,
    old_raw_predictions: &[f64],
    matched_raw_predictions: &[f64],
    requested_route_predictions: &[f64],
) -> Result<FinalizedSeedScoreV2, String> {
    stage.validate()?;
    if !ROUTER_SEEDS.contains(&seed)
        || decision.schema != FinalizerRouteDecisionV2::SCHEMA
        || !decision.decision_preceded_label_metrics
        || (decision.chosen_route == RouteFamilyV2::Raw) != !decision.fallback_reasons.is_empty()
            && decision.requested_route != RouteFamilyV2::Raw
    {
        return Err("finalizer score was not preceded by a valid route lock".into());
    }
    let labels = stage.labels_after_route_lock()?;
    for (name, predictions) in [
        ("old raw", old_raw_predictions),
        ("matched raw", matched_raw_predictions),
        ("requested route", requested_route_predictions),
    ] {
        if predictions.len() != labels.len() {
            return Err(format!(
                "{name} predictions do not cover the staged test rows"
            ));
        }
        require_finite(name, predictions)?;
    }
    let routed_predictions = if decision.chosen_route == RouteFamilyV2::Raw {
        matched_raw_predictions
    } else {
        requested_route_predictions
    };
    let old_production_raw_rmse = rmse_slice(labels, old_raw_predictions)?;
    let matched_raw_gp_rmse = rmse_slice(labels, matched_raw_predictions)?;
    let routed_rmse = rmse_slice(labels, routed_predictions)?;
    let ablation = SeedAblationV2::new(old_production_raw_rmse, matched_raw_gp_rmse, routed_rmse)?;
    Ok(FinalizedSeedScoreV2 {
        schema: FinalizedSeedScoreV2::SCHEMA.into(),
        seed,
        route_decision_sha256: canonical_sha256(decision)?,
        old_production_raw_rmse,
        matched_raw_gp_rmse,
        routed_rmse,
        ablation,
        labels_reached_hyperion: false,
        labels_reached_prediction_fit: false,
        common_staged_data_and_evaluator: true,
    })
}
