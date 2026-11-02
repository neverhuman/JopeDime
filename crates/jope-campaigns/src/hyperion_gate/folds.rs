use super::*;

pub(crate) fn validate_stage_folds(
    folds: &[PairedFoldEvidenceV1],
    expected_folds: usize,
    arms: &[FeatureViewKindV1],
) -> Result<(), String> {
    if folds.len() != expected_folds {
        return Err(format!(
            "stage requires exactly {expected_folds} paired folds"
        ));
    }
    let mut fold_ids = BTreeSet::new();
    let mut row_ids = BTreeSet::new();
    let mut group_to_fold = BTreeMap::<&str, u32>::new();
    for fold in folds {
        fold.validate(arms)?;
        if fold.row_ids.len() < MINIMUM_FOLD_HOLD_ROWS {
            return Err(format!(
                "fold {} has {} hold rows; minimum is {MINIMUM_FOLD_HOLD_ROWS}",
                fold.fold_index,
                fold.row_ids.len()
            ));
        }
        let hold_group_count = fold.atomic_groups.iter().collect::<BTreeSet<_>>().len();
        if hold_group_count < MINIMUM_FOLD_HOLD_GROUPS {
            return Err(format!(
                "fold {} has {hold_group_count} hold groups; minimum is {MINIMUM_FOLD_HOLD_GROUPS}",
                fold.fold_index
            ));
        }
        if !fold_ids.insert(fold.fold_index) {
            return Err("stage contains duplicate fold identities".into());
        }
        for (row_id, group) in fold.row_ids.iter().zip(&fold.atomic_groups) {
            if !row_ids.insert(row_id.as_str()) {
                return Err("OOF evidence predicts a row more than once".into());
            }
            if group_to_fold
                .insert(group.as_str(), fold.fold_index)
                .is_some_and(|prior| prior != fold.fold_index)
            {
                return Err("atomic group crosses validation folds".into());
            }
        }
    }
    if fold_ids != (0..expected_folds as u32).collect() {
        return Err("stage fold identities are incomplete".into());
    }
    let minimum_groups = (expected_folds * MINIMUM_FOLD_HOLD_GROUPS)
        .max(GateThresholdsV1::default().minimum_atomic_groups as usize);
    if group_to_fold.len() < minimum_groups {
        return Err("stage has insufficient atomic groups".into());
    }
    for fold in folds {
        let fit_rows = row_ids.len().saturating_sub(fold.row_ids.len());
        if fit_rows < MINIMUM_FOLD_FIT_ROWS {
            return Err(format!(
                "fold {} has {fit_rows} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}",
                fold.fold_index
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldPlanV1 {
    pub schema: String,
    pub stage: GateStageV1,
    pub seed: u64,
    pub folds: Vec<GroupedFoldV1>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GroupedFoldV1 {
    pub fold_index: u32,
    pub fit_rows: Vec<usize>,
    pub hold_rows: Vec<usize>,
    pub hold_groups: Vec<String>,
}

pub fn build_grouped_fold_plan(
    stage: GateStageV1,
    atomic_groups: &[String],
) -> Result<GroupedFoldPlanV1, String> {
    let fold_count = match stage {
        GateStageV1::Cv5Screen => 5,
        GateStageV1::Cv7Confirmation => 7,
    };
    let seed = match stage {
        GateStageV1::Cv5Screen => CV5_SEED,
        GateStageV1::Cv7Confirmation => CV7_SEED,
    };
    let mut grouped = BTreeMap::<&str, Vec<usize>>::new();
    for (row, group) in atomic_groups.iter().enumerate() {
        if group.is_empty() {
            return Err("atomic group must not be empty".into());
        }
        grouped.entry(group).or_default().push(row);
    }
    let minimum_groups = (fold_count * MINIMUM_FOLD_HOLD_GROUPS)
        .max(GateThresholdsV1::default().minimum_atomic_groups as usize);
    if grouped.len() < minimum_groups {
        return Err("insufficient atomic groups for the Hyperion gate".into());
    }
    let mut groups = grouped.into_iter().collect::<Vec<_>>();
    groups.sort_by_key(|(group, _)| seed64(&[&seed.to_string(), group]));
    let mut assignments = vec![Vec::<(&str, Vec<usize>)>::new(); fold_count];
    let mut sizes = vec![0usize; fold_count];
    for group in groups {
        let fold = (0..fold_count)
            .min_by_key(|index| (sizes[*index], *index))
            .unwrap();
        sizes[fold] += group.1.len();
        assignments[fold].push(group);
    }
    let all_rows = (0..atomic_groups.len()).collect::<BTreeSet<_>>();
    let folds = assignments
        .into_iter()
        .enumerate()
        .map(|(fold_index, values)| {
            let mut hold_rows = values
                .iter()
                .flat_map(|(_, rows)| rows.iter().copied())
                .collect::<Vec<_>>();
            hold_rows.sort_unstable();
            let hold_set = hold_rows.iter().copied().collect::<BTreeSet<_>>();
            let fit_rows = all_rows.difference(&hold_set).copied().collect();
            let mut hold_groups = values
                .into_iter()
                .map(|(group, _)| group.to_owned())
                .collect::<Vec<_>>();
            hold_groups.sort();
            GroupedFoldV1 {
                fold_index: fold_index as u32,
                fit_rows,
                hold_rows,
                hold_groups,
            }
        })
        .collect::<Vec<_>>();
    for fold in &folds {
        if fold.hold_rows.len() < MINIMUM_FOLD_HOLD_ROWS {
            return Err(format!(
                "fold {} has {} hold rows; minimum is {MINIMUM_FOLD_HOLD_ROWS}",
                fold.fold_index,
                fold.hold_rows.len()
            ));
        }
        if fold.hold_groups.len() < MINIMUM_FOLD_HOLD_GROUPS {
            return Err(format!(
                "fold {} has {} hold groups; minimum is {MINIMUM_FOLD_HOLD_GROUPS}",
                fold.fold_index,
                fold.hold_groups.len()
            ));
        }
        if fold.fit_rows.len() < MINIMUM_FOLD_FIT_ROWS {
            return Err(format!(
                "fold {} has {} fit rows; minimum is {MINIMUM_FOLD_FIT_ROWS}",
                fold.fold_index,
                fold.fit_rows.len()
            ));
        }
    }
    Ok(GroupedFoldPlanV1 {
        schema: "JopeHyperionGroupedFoldPlan.v1".into(),
        stage,
        seed,
        folds,
    })
}

pub(crate) fn validate_plan_feature_feasibility(
    plan: &GroupedFoldPlanV1,
    arms: &[FeatureViewKindV1],
    raw_features: usize,
) -> Result<(), String> {
    for fold in &plan.folds {
        for arm in arms {
            validate_feature_view_feasibility(*arm, raw_features, fold.fit_rows.len()).map_err(
                |error| {
                    format!(
                        "fold {} is structurally ineligible: {error}",
                        fold.fold_index
                    )
                },
            )?;
        }
    }
    Ok(())
}

/// Return the arms that are feasible on every fold of both frozen stages,
/// together with deterministic reasons for each excluded arm.  Eligibility is
/// deliberately arm-local: an unsafe row128 arm cannot disable cell.
pub fn independently_feasible_feature_views(
    cv5: &GroupedFoldPlanV1,
    cv7: &GroupedFoldPlanV1,
    available: &[FeatureViewKindV1],
    raw_features: usize,
) -> (Vec<FeatureViewKindV1>, BTreeMap<FeatureViewKindV1, String>) {
    let mut eligible = Vec::new();
    let mut excluded = BTreeMap::new();
    for arm in available.iter().copied() {
        if arm == FeatureViewKindV1::Raw
            || !FeatureViewKindV1::EXPANDED.contains(&arm)
            || eligible.contains(&arm)
            || excluded.contains_key(&arm)
        {
            excluded.insert(arm, "arm is raw, unknown, or duplicated".into());
            continue;
        }
        let result = validate_plan_feature_feasibility(cv5, &[arm], raw_features)
            .and_then(|()| validate_plan_feature_feasibility(cv7, &[arm], raw_features));
        match result {
            Ok(()) => eligible.push(arm),
            Err(reason) => {
                excluded.insert(arm, reason);
            }
        }
    }
    eligible.sort_unstable();
    (eligible, excluded)
}

#[allow(clippy::too_many_arguments)]
pub fn raw_fallback_decision(
    dataset_id: &str,
    training_sha256: &str,
    predictor_sha256: &str,
    encoder_sha256: &str,
    cv5_plan_sha256: &str,
    cv7_plan_sha256: &str,
    reason: impl Into<String>,
    started: Instant,
) -> JopeHyperionGateDecisionV1 {
    JopeHyperionGateDecisionV1 {
        schema: JopeHyperionGateDecisionV1::SCHEMA.into(),
        dataset_id_sha256: raw_sha256(dataset_id.as_bytes()),
        source_training_sha256: training_sha256.into(),
        raw_predictor_sha256: predictor_sha256.into(),
        encoder_identity_sha256: encoder_sha256.into(),
        cv5_fold_plan_sha256: cv5_plan_sha256.into(),
        cv7_fold_plan_sha256: cv7_plan_sha256.into(),
        thresholds: GateThresholdsV1::default(),
        cv5_candidates: Vec::new(),
        cv5_advanced_arm: None,
        cv7_confirmation: None,
        selected_view: FeatureViewKindV1::Raw,
        raw_fallback_reason: Some(reason.into()),
        fold_receipt_sha256s: Vec::new(),
        encoding_milliseconds: 0,
        scoring_milliseconds: 0,
        wall_clock_milliseconds: u64::try_from(started.elapsed().as_millis())
            .unwrap_or(u64::MAX)
            .min(WALL_CLOCK_BUDGET.as_millis() as u64),
    }
}

pub fn write_gate_artifacts(
    artifact_root: &Path,
    decision: &JopeHyperionGateDecisionV1,
    predictor: &FrozenPredictorV1,
    raw_features: usize,
) -> Result<(), String> {
    decision.validate()?;
    predictor.validate()?;
    fs::create_dir_all(artifact_root)
        .map_err(|error| format!("create gate artifact root: {error}"))?;
    let decision_bytes = canonical_json_bytes(decision)?;
    let decision_sha256 = raw_sha256(&decision_bytes);
    write_immutable(
        &artifact_root.join("gate-decision.v1.json"),
        &decision_bytes,
    )?;
    let feature_view = JopeFeatureViewV1::new(decision.selected_view, raw_features)?;
    let predictor_sha256 = predictor.semantic_sha256()?;
    let bound = JopeFeatureBoundPredictorV1 {
        schema: JopeFeatureBoundPredictorV1::SCHEMA.into(),
        feature_view_sha256: feature_view.identity()?,
        selected_view: feature_view,
        gate_decision_sha256: decision_sha256,
        selected_predictor: predictor.clone(),
        selected_predictor_sha256: predictor_sha256.clone(),
        raw_fallback_predictor: predictor.clone(),
        raw_fallback_predictor_sha256: predictor_sha256,
    };
    bound.validate()?;
    write_immutable(
        &artifact_root.join("feature-bound-predictor.v1.json"),
        &canonical_json_bytes(&bound)?,
    )
}

pub(crate) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if path.exists() {
        let present =
            fs::read(path).map_err(|error| format!("read existing {}: {error}", path.display()))?;
        if present == bytes {
            return Ok(());
        }
        return Err(format!(
            "refusing to replace differing immutable artifact {}",
            path.display()
        ));
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|error| format!("create immutable {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write immutable {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync immutable {}: {error}", path.display()))
}
