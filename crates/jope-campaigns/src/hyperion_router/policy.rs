use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedAblationV2 {
    pub old_production_raw_rmse: f64,
    pub matched_raw_gp_rmse: f64,
    pub routed_rmse: f64,
    pub gp_lift: f64,
    pub incremental_hyperion_lift: f64,
    pub total_lift: f64,
}

impl SeedAblationV2 {
    pub fn new(
        old_production_raw_rmse: f64,
        matched_raw_gp_rmse: f64,
        routed_rmse: f64,
    ) -> Result<Self, String> {
        if ![old_production_raw_rmse, matched_raw_gp_rmse, routed_rmse]
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err("ablation RMSE values must be finite and positive".into());
        }
        Ok(Self {
            old_production_raw_rmse,
            matched_raw_gp_rmse,
            routed_rmse,
            gp_lift: (old_production_raw_rmse - matched_raw_gp_rmse) / old_production_raw_rmse,
            incremental_hyperion_lift: (matched_raw_gp_rmse - routed_rmse) / matched_raw_gp_rmse,
            total_lift: (old_production_raw_rmse - routed_rmse) / old_production_raw_rmse,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterPolicyFreezeV2 {
    pub schema: String,
    pub campaign_id: String,
    pub training_adaptation_dataset_sha256s: Vec<String>,
    pub fixed_candidate_bank_sha256: String,
    pub grouped_fold_protocol_sha256: String,
    pub encoder_identity_sha256: String,
    pub search_configuration_sha256: String,
    pub source_tree_sha256: String,
    pub search_binary_sha256: String,
    pub comparator_snapshot_sha256: String,
    pub evaluator_sha256: String,
    pub finalizer_binary_sha256: String,
    pub activation_thresholds: ActivationThresholdsV2,
    pub outer_seeds: Vec<u64>,
    pub outer_folds: u32,
    pub inner_folds: u32,
    pub evaluations_per_stream: u32,
    pub candidate_timeout_seconds: u32,
    pub test_inventory_resolved: bool,
    pub historical_confirmation_identity_reused: bool,
}

impl RouterPolicyFreezeV2 {
    pub const SCHEMA: &'static str = "HyperionRouterPolicyFreeze.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.campaign_id != EXPLORATORY_CAMPAIGN_ID
            || self.training_adaptation_dataset_sha256s.len() != ROUTER_DATASET_COUNT
            || self.activation_thresholds != ActivationThresholdsV2::default()
            || self.outer_seeds != ROUTER_SEEDS
            || self.outer_folds != OUTER_FOLDS as u32
            || self.inner_folds != INNER_FOLDS as u32
            || self.evaluations_per_stream != SEARCH_EVALUATIONS_PER_STREAM
            || self.candidate_timeout_seconds != FIXED_CANDIDATE_TIMEOUT_SECONDS
            || self.test_inventory_resolved
            || self.historical_confirmation_identity_reused
        {
            return Err("router policy reused history, resolved test material, or drifted from frozen rules".into());
        }
        let mut unique = BTreeSet::new();
        for (index, dataset) in self.training_adaptation_dataset_sha256s.iter().enumerate() {
            validate_sha256("policy adaptation dataset", dataset)?;
            if !dataset.starts_with(CURRENT_EXPLORATORY_DATASET_IDS[index])
                || !unique.insert(dataset.as_str())
            {
                return Err(
                    "router policy adaptation cohort is foreign, reordered, or duplicated".into(),
                );
            }
        }
        for (name, value) in [
            (
                "policy candidate bank",
                self.fixed_candidate_bank_sha256.as_str(),
            ),
            (
                "policy grouped folds",
                self.grouped_fold_protocol_sha256.as_str(),
            ),
            ("policy encoder", self.encoder_identity_sha256.as_str()),
            (
                "policy search configuration",
                self.search_configuration_sha256.as_str(),
            ),
            ("policy source tree", self.source_tree_sha256.as_str()),
            ("policy search binary", self.search_binary_sha256.as_str()),
            (
                "policy comparator",
                self.comparator_snapshot_sha256.as_str(),
            ),
            ("policy evaluator", self.evaluator_sha256.as_str()),
            ("policy finalizer", self.finalizer_binary_sha256.as_str()),
        ] {
            validate_sha256(name, value)?;
        }
        Ok(())
    }

    pub fn identity(&self) -> Result<String, String> {
        self.validate()?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterSeedSelectionV2 {
    pub schema: String,
    pub campaign_id: String,
    pub policy_freeze_sha256: String,
    pub dataset_sha256: String,
    pub seed: u64,
    pub route: RouteFamilyV2,
    pub matched_raw_predictor_sha256: String,
    pub selected_predictor_sha256: String,
    pub blend_alpha: Option<f64>,
    pub router_decision_sha256: String,
    pub nested_search_sha256: String,
    pub fold_cell_coverage_sha256: String,
    pub fixed_bank_admission_sha256: String,
    pub selection_finalized: bool,
    pub test_inventory_resolved: bool,
    pub test_material_present: bool,
}

impl RouterSeedSelectionV2 {
    pub const SCHEMA: &'static str = "HyperionRouterSeedSelection.v2";

    pub fn validate(&self, policy_sha256: &str) -> Result<(), String> {
        validate_sha256("selection policy", policy_sha256)?;
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || self.policy_freeze_sha256 != policy_sha256
            || !ROUTER_SEEDS.contains(&self.seed)
            || !self.selection_finalized
            || self.test_inventory_resolved
            || self.test_material_present
        {
            return Err("router selection is not a complete train-only freeze".into());
        }
        for (name, value) in [
            ("selection dataset", self.dataset_sha256.as_str()),
            (
                "selection matched raw predictor",
                self.matched_raw_predictor_sha256.as_str(),
            ),
            (
                "selection selected predictor",
                self.selected_predictor_sha256.as_str(),
            ),
            ("selection decision", self.router_decision_sha256.as_str()),
            (
                "selection nested search",
                self.nested_search_sha256.as_str(),
            ),
            (
                "selection cell coverage",
                self.fold_cell_coverage_sha256.as_str(),
            ),
            (
                "selection fixed-bank admission",
                self.fixed_bank_admission_sha256.as_str(),
            ),
        ] {
            validate_sha256(name, value)?;
        }
        match (self.route, self.blend_alpha) {
            (RouteFamilyV2::Raw, None)
                if self.selected_predictor_sha256 == self.matched_raw_predictor_sha256 =>
            {
                Ok(())
            }
            (RouteFamilyV2::RawCellBlend, Some(alpha))
                if alpha.is_finite() && (0.0..=0.5).contains(&alpha) =>
            {
                Ok(())
            }
            (RouteFamilyV2::CellOnly | RouteFamilyV2::Append, None) => Ok(()),
            _ => Err("selection predictor, route, and blend coefficient disagree".into()),
        }
    }

    pub fn identity(&self, policy_sha256: &str) -> Result<String, String> {
        self.validate(policy_sha256)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterSelectionInventoryV2 {
    pub schema: String,
    pub campaign_id: String,
    pub policy_freeze_sha256: String,
    pub dataset_sha256s: Vec<String>,
    pub selections: Vec<RouterSeedSelectionV2>,
    pub all_selections_precede_test_access: bool,
    pub test_access_count_at_freeze: u32,
}

impl RouterSelectionInventoryV2 {
    pub const SCHEMA: &'static str = "HyperionRouterSelectionInventory.v2";

    pub fn validate(&self, policy: &RouterPolicyFreezeV2) -> Result<(), String> {
        policy.validate()?;
        let policy_sha256 = policy.identity()?;
        if self.schema != Self::SCHEMA
            || !matches!(
                self.campaign_id.as_str(),
                EXPLORATORY_CAMPAIGN_ID | CONFIRMATORY_CAMPAIGN_ID
            )
            || self.policy_freeze_sha256 != policy_sha256
            || self.dataset_sha256s.len() != ROUTER_DATASET_COUNT
            || self.selections.len() != ROUTER_DATASET_COUNT * ROUTER_SEEDS.len()
            || !self.all_selections_precede_test_access
            || self.test_access_count_at_freeze != 0
        {
            return Err(
                "selection inventory is partial, foreign, or was frozen after test access".into(),
            );
        }
        let datasets = self.dataset_sha256s.iter().collect::<BTreeSet<_>>();
        if datasets.len() != ROUTER_DATASET_COUNT {
            return Err("selection inventory dataset hashes are duplicated".into());
        }
        for dataset in &self.dataset_sha256s {
            validate_sha256("selection inventory dataset", dataset)?;
        }
        if self.campaign_id == EXPLORATORY_CAMPAIGN_ID {
            for (index, dataset) in self.dataset_sha256s.iter().enumerate() {
                if !dataset.starts_with(CURRENT_EXPLORATORY_DATASET_IDS[index]) {
                    return Err(
                        "exploratory selection inventory differs from the current twenty".into(),
                    );
                }
            }
        }
        for (index, selection) in self.selections.iter().enumerate() {
            selection.validate(&policy_sha256)?;
            let expected_dataset = &self.dataset_sha256s[index / ROUTER_SEEDS.len()];
            let expected_seed = ROUTER_SEEDS[index % ROUTER_SEEDS.len()];
            if selection.campaign_id != self.campaign_id
                || &selection.dataset_sha256 != expected_dataset
                || selection.seed != expected_seed
            {
                return Err("selection inventory order or 40-slot coverage drifted".into());
            }
            if index % ROUTER_SEEDS.len() == 1
                && self.selections[index - 1].route != selection.route
            {
                return Err("the two seeds selected different route families".into());
            }
        }
        Ok(())
    }

    pub fn identity(&self, policy: &RouterPolicyFreezeV2) -> Result<String, String> {
        self.validate(policy)?;
        canonical_sha256(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConfirmationApplicationFreezeV1 {
    pub schema: String,
    pub campaign_id: String,
    pub frozen_router_policy_sha256: String,
    pub fresh_cohort_freeze_sha256: String,
    pub comparator_snapshot_sha256: String,
    pub encoder_identity_sha256: String,
    pub search_binary_sha256: String,
    pub evaluator_sha256: String,
    pub finalizer_binary_sha256: String,
    pub thresholds_unchanged: bool,
    pub budgets_unchanged: bool,
    pub finalizer_unchanged: bool,
    pub test_inventory_resolved: bool,
}

impl ConfirmationApplicationFreezeV1 {
    pub const SCHEMA: &'static str = "HyperionConfirmationApplicationFreeze.v1";

    pub fn validate(&self, policy: &RouterPolicyFreezeV2) -> Result<(), String> {
        policy.validate()?;
        if self.schema != Self::SCHEMA
            || self.campaign_id != CONFIRMATORY_CAMPAIGN_ID
            || self.frozen_router_policy_sha256 != policy.identity()?
            || self.comparator_snapshot_sha256 != policy.comparator_snapshot_sha256
            || self.encoder_identity_sha256 != policy.encoder_identity_sha256
            || self.search_binary_sha256 != policy.search_binary_sha256
            || self.evaluator_sha256 != policy.evaluator_sha256
            || self.finalizer_binary_sha256 != policy.finalizer_binary_sha256
            || !self.thresholds_unchanged
            || !self.budgets_unchanged
            || !self.finalizer_unchanged
            || self.test_inventory_resolved
        {
            return Err("fresh confirmation does not apply the frozen router unchanged".into());
        }
        validate_sha256("fresh cohort freeze", &self.fresh_cohort_freeze_sha256)
    }

    pub fn identity(&self, policy: &RouterPolicyFreezeV2) -> Result<String, String> {
        self.validate(policy)?;
        canonical_sha256(self)
    }
}

/// A single-open boundary. Labels remain private until a route has been
/// chosen exclusively from encoding proof and the frozen selection.
#[derive(Debug, Clone, PartialEq)]
pub struct SealedFeatureStageV2 {
    pub schema: String,
    pub dataset_sha256: String,
    pub test_sha256: String,
    pub feature_count: usize,
    pub feature_rows: Vec<Vec<f64>>,
    labels: Vec<f64>,
    pub test_access_ordinal: u32,
}

impl SealedFeatureStageV2 {
    pub const SCHEMA: &'static str = "HyperionSealedFeatureStage.v2";

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != Self::SCHEMA
            || self.test_access_ordinal != 1
            || self.feature_count == 0
            || self.feature_rows.is_empty()
            || self.labels.len() != self.feature_rows.len()
            || self.feature_rows.iter().any(|row| {
                row.len() != self.feature_count || row.iter().any(|value| !value.is_finite())
            })
            || self.labels.iter().any(|value| !value.is_finite())
        {
            return Err("sealed V2 feature stage has invalid shape or access boundary".into());
        }
        validate_sha256("sealed V2 dataset", &self.dataset_sha256)?;
        validate_sha256("sealed V2 test", &self.test_sha256)
    }

    pub fn hyperion_dummy_target_rows(&self) -> Result<Vec<Vec<f64>>, String> {
        self.validate()?;
        Ok(self
            .feature_rows
            .iter()
            .map(|features| {
                let mut row = features.clone();
                row.push(0.0);
                row
            })
            .collect())
    }

    pub(super) fn labels_after_route_lock(&self) -> Result<&[f64], String> {
        self.validate()?;
        Ok(&self.labels)
    }
}

pub fn stage_sealed_test_once_v2(
    dataset_sha256: &str,
    test_sha256: &str,
    bytes: &[u8],
    feature_count: usize,
) -> Result<SealedFeatureStageV2, String> {
    validate_sha256("sealed V2 dataset", dataset_sha256)?;
    validate_sha256("sealed V2 test", test_sha256)?;
    if bytes.is_empty() || feature_count == 0 || raw_sha256(bytes) != test_sha256 {
        return Err("sealed test bytes are empty or differ from their authorized hash".into());
    }
    let text =
        std::str::from_utf8(bytes).map_err(|error| format!("sealed test is not UTF-8: {error}"))?;
    let mut feature_rows = Vec::new();
    let mut labels = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let parsed = line
            .split(',')
            .map(|field| field.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>();
        let mut values = match parsed {
            Ok(values) => values,
            Err(_) if line_index == 0 => continue,
            Err(error) => return Err(format!("sealed test row {line_index} is invalid: {error}")),
        };
        if values.len() != feature_count + 1 {
            return Err(format!(
                "sealed test row {line_index} has {} columns, expected {}",
                values.len(),
                feature_count + 1
            ));
        }
        let label = values.pop().expect("feature and target count checked");
        if !label.is_finite() || values.iter().any(|value| !value.is_finite()) {
            return Err(format!("sealed test row {line_index} is nonfinite"));
        }
        feature_rows.push(values);
        labels.push(label);
    }
    let stage = SealedFeatureStageV2 {
        schema: SealedFeatureStageV2::SCHEMA.into(),
        dataset_sha256: dataset_sha256.into(),
        test_sha256: test_sha256.into(),
        feature_count,
        feature_rows,
        labels,
        test_access_ordinal: 1,
    };
    stage.validate()?;
    Ok(stage)
}
