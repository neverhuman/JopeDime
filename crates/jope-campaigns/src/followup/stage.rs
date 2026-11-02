use super::*;

pub(super) fn stage_manifest_path() -> PathBuf {
    replacement_root().join("staged/manifest.v1.json")
}

pub(super) fn stage_marker_path() -> PathBuf {
    replacement_root().join("staged/STAGING_COMPLETE")
}

pub(super) fn load_stage() -> Result<StageManifest> {
    let marker: CompletionMarker = serde_json::from_slice(&fs::read(stage_marker_path())?)?;
    if marker.schema != "JopeDimeFollowUpStagingComplete.v1" || marker.status != "COMPLETE" {
        bail!("follow-up staging marker is invalid");
    }
    validate_sha256("staging manifest SHA-256", &marker.manifest_sha256)?;
    let bytes = fs::read(stage_manifest_path())?;
    if sha256_bytes(&bytes) != marker.manifest_sha256 {
        bail!("follow-up staging manifest differs from its marker hash");
    }
    let manifest: StageManifest = serde_json::from_slice(&bytes)?;
    validate_stage_manifest(&manifest)?;
    for entry in &manifest.targets {
        let path = replacement_root().join("staged").join(&entry.staged_file);
        if hash_file(&path)? != entry.new_sha256 {
            bail!("staged target hash mismatch for {}", entry.staged_file);
        }
    }
    Ok(manifest)
}

pub(super) fn validate_stage_manifest(manifest: &StageManifest) -> Result<()> {
    if manifest.schema != "JopeDimeFollowUpStaging.v1"
        || !manifest.exploratory_reused_test
        || manifest.original_panel_ledger_sha256 != ORIGINAL_LEDGER_SHA256
        || manifest.cohort != LOSS_DATASETS
        || manifest.targets.len() != 71
        || manifest.untouched_original_winners.len() != 14
    {
        bail!("follow-up staging manifest has invalid accounting or provenance");
    }
    let mut keys = BTreeSet::new();
    for entry in &manifest.targets {
        validate_sha256("old target SHA-256", &entry.old_sha256)?;
        validate_sha256("new target SHA-256", &entry.new_sha256)?;
        if !keys.insert((entry.kind.clone(), entry.dataset_hash.clone())) {
            bail!("follow-up staging manifest has duplicate targets");
        }
    }
    if manifest
        .targets
        .iter()
        .filter(|entry| entry.kind == "prediction")
        .count()
        != 35
        || manifest
            .targets
            .iter()
            .filter(|entry| entry.kind == "report")
            .count()
            != 35
        || manifest
            .targets
            .iter()
            .filter(|entry| entry.kind == "panel_ledger")
            .count()
            != 1
    {
        bail!("follow-up staging manifest target classes are invalid");
    }
    Ok(())
}

pub(super) fn create_stage(
    registry: &Path,
    freezes: &BTreeMap<String, ValidatedFreeze>,
    source: &Value,
) -> Result<StageManifest> {
    if stage_marker_path().exists() {
        return load_stage();
    }
    if replacement_root().join("staged").exists() {
        bail!("incomplete staged publication requires operator recovery");
    }
    let source_rows = source_outcomes(source)?;
    let baselines = load_baselines()?;
    let incoming = replacement_root().join(format!("staged.incoming-{}", std::process::id()));
    if incoming.exists() {
        bail!(
            "stale incoming publication directory exists: {}",
            incoming.display()
        );
    }
    fs::create_dir_all(&incoming)?;

    let mut targets = Vec::new();
    let mut follow_up_rows = BTreeMap::<String, crate::ScoredDataset>::new();
    for hash in LOSS_DATASETS {
        let old = original_report(registry, hash)?;
        let old_report_path = registry.join(hash).join("reports").join(REPORT_FILE);
        let old_prediction_path = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let old_report_sha256 = hash_file(&old_report_path)?;
        let old_prediction_sha256 = hash_file(&old_prediction_path)?;
        if source_rows[*hash]["report_sha256"].as_str() != Some(old_report_sha256.as_str())
            || source_rows[*hash]["prediction_sha256"].as_str()
                != Some(old_prediction_sha256.as_str())
        {
            bail!("published v1 artifacts drifted from the pinned source ledger for {hash}");
        }
        let mut row = score_dataset(registry, hash, &freezes[*hash], &baselines[*hash])?;
        let generated: Value = serde_json::from_slice(&row.report_bytes)?;
        row.report_bytes = build_v2_report(&old, &freezes[*hash], generated)?;
        row.report_sha256 = sha256_bytes(&row.report_bytes);
        row.outcome["report_sha256"] = json!(row.report_sha256);
        row.outcome["follow_up"] = follow_up_comparison(
            &old,
            &freezes[*hash],
            row.outcome["test_metrics"]["rmse"]
                .as_f64()
                .context("new outcome test RMSE is missing")?,
        )?;
        row.outcome["original_strict_overall_outcome"] =
            source_rows[*hash]["strict_overall_outcome"].clone();
        row.outcome["new_win_among_original_losses"] =
            json!(row.outcome["strict_overall_outcome"] == "win");

        let prediction_file = format!("predictions/{hash}.csv");
        immutable_write(&incoming.join(&prediction_file), &row.prediction_bytes)?;
        targets.push(TargetEntry {
            kind: "prediction".to_string(),
            dataset_hash: Some((*hash).to_string()),
            staged_file: prediction_file,
            archive_file: format!("payload/predictions/{hash}.csv"),
            old_sha256: old_prediction_sha256,
            new_sha256: row.prediction_sha256.clone(),
        });
        let report_file = format!("reports/{hash}.json");
        immutable_write(&incoming.join(&report_file), &row.report_bytes)?;
        targets.push(TargetEntry {
            kind: "report".to_string(),
            dataset_hash: Some((*hash).to_string()),
            staged_file: report_file,
            archive_file: format!("payload/reports/{hash}.json"),
            old_sha256: old_report_sha256,
            new_sha256: row.report_sha256.clone(),
        });
        follow_up_rows.insert((*hash).to_string(), row);
    }

    let mut all_scored = Vec::with_capacity(DATASETS.len());
    let mut outcomes = Vec::with_capacity(DATASETS.len());
    for hash in DATASETS {
        if let Some(row) = follow_up_rows.remove(*hash) {
            outcomes.push(row.outcome.clone());
            all_scored.push(row);
        } else {
            let mut outcome = source_rows[*hash].clone();
            outcome["follow_up_status"] = json!("original_winner_untouched");
            let scores = outcome["rmse_by_algorithm"]
                .as_object()
                .context("source outcome RMSE map is missing")?
                .iter()
                .map(|(name, score)| {
                    Ok((
                        name.clone(),
                        score.as_f64().context("source RMSE is not finite")?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            outcomes.push(outcome.clone());
            all_scored.push(crate::ScoredDataset {
                dataset_hash: (*hash).to_string(),
                predictions: Vec::new(),
                prediction_bytes: Vec::new(),
                prediction_sha256: outcome["prediction_sha256"]
                    .as_str()
                    .context("source prediction hash is missing")?
                    .to_string(),
                report_bytes: Vec::new(),
                report_sha256: outcome["report_sha256"]
                    .as_str()
                    .context("source report hash is missing")?
                    .to_string(),
                scores,
                outcome,
            });
        }
    }
    if all_scored.len() != 42 || outcomes.len() != 42 {
        bail!("updated panel does not contain exactly 42 outcomes");
    }
    let wins = all_scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let losses = all_scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "loss")
        .count();
    let ties = 42 - wins - losses;
    let tabfm_wins = all_scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "win")
        .count();
    let tabfm_losses = all_scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "loss")
        .count();
    let tabfm_ties = 42 - tabfm_wins - tabfm_losses;
    let tabfm_status = match tabfm_wins.cmp(&tabfm_losses) {
        Ordering::Greater => "winning",
        Ordering::Equal => "tied",
        Ordering::Less => "losing",
    };
    let new_wins = all_scored
        .iter()
        .filter(|row| LOSS_DATASETS.contains(&row.dataset_hash.as_str()))
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let ledger = json!({
        "schema": "JopeDimePanelResults.v2",
        "panel_id": ORIGINAL_PANEL_ID,
        "follow_up_panel_id": FOLLOW_UP_PANEL_ID,
        "dataset_count": 42,
        "follow_up": {
            "label": "exploratory, test-adaptive follow-up on the 35 original strict losses",
            "exploratory_reused_test": true,
            "source_panel_ledger_sha256": ORIGINAL_LEDGER_SHA256,
            "cohort_size": 35,
            "new_wins_among_original_losses": new_wins,
            "all_35_follow_up_results_published_without_cherry_picking": true,
        },
        "panel_text_sha256": crate::PANEL_SHA256,
        "baseline_panel_sha256": crate::PANEL_ANALYSIS_SHA256,
        "selection_accounting": {
            "follow_up_evolutionary_slots": FOLLOW_UP_SELECTION_SLOTS,
            "updated_active_panel_evolutionary_slots": ACTIVE_PANEL_SELECTION_SLOTS,
            "historical_evolutionary_slots_across_both_experiments": HISTORICAL_SELECTION_SLOTS,
            "original_experiment_evolutionary_slots": ORIGINAL_SELECTION_SLOTS,
            "follow_up_generation_29_freezes": 35,
            "follow_up_full_cv7_finalist_receipts": FOLLOW_UP_FINALISTS,
            "untouched_original_winner_freezes": 7,
        },
        "strict_overall": {
            "rule": "win iff jope-dime test RMSE is strictly lower than the best baseline test RMSE",
            "wins": wins,
            "losses": losses,
            "ties": ties,
        },
        "tabfm_head_to_head": {
            "wins": tabfm_wins,
            "losses": tabfm_losses,
            "ties": tabfm_ties,
            "status": tabfm_status,
        },
        "pairwise_results": pairwise_results(&all_scored),
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 16.0,
            "method": "Each dataset is a simultaneous 7-algorithm round robin; lower RMSE wins and exact equality ties; panel order is pinned.",
            "standings": elo_standings(&all_scored),
        },
        "outcomes": outcomes,
    });
    let ledger_bytes = canonical_json(&ledger)?;
    immutable_write(&incoming.join("panel-ledger.json"), &ledger_bytes)?;
    targets.push(TargetEntry {
        kind: "panel_ledger".to_string(),
        dataset_hash: None,
        staged_file: "panel-ledger.json".to_string(),
        archive_file: "payload/panel-ledger.json".to_string(),
        old_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        new_sha256: sha256_bytes(&ledger_bytes),
    });

    let mut untouched = Vec::new();
    for hash in DATASETS
        .iter()
        .copied()
        .filter(|hash| !LOSS_DATASETS.contains(hash))
    {
        for (kind, path) in [
            (
                "prediction",
                registry
                    .join(hash)
                    .join("predictions")
                    .join(PREDICTION_FILE),
            ),
            (
                "report",
                registry.join(hash).join("reports").join(REPORT_FILE),
            ),
        ] {
            untouched.push(UntouchedEntry {
                kind: kind.to_string(),
                dataset_hash: hash.to_string(),
                sha256: hash_file(&path)?,
            });
        }
    }
    targets.sort_by(|left, right| {
        let order = |kind: &str| match kind {
            "prediction" => 0,
            "report" => 1,
            "panel_ledger" => 2,
            _ => 3,
        };
        order(&left.kind)
            .cmp(&order(&right.kind))
            .then_with(|| left.dataset_hash.cmp(&right.dataset_hash))
    });
    let manifest = StageManifest {
        schema: "JopeDimeFollowUpStaging.v1".to_string(),
        follow_up_label: "exploratory, test-adaptive follow-up".to_string(),
        exploratory_reused_test: true,
        original_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        cohort: LOSS_DATASETS
            .iter()
            .map(|hash| (*hash).to_string())
            .collect(),
        targets,
        untouched_original_winners: untouched,
    };
    validate_stage_manifest(&manifest)?;
    let manifest_bytes = canonical_json(&serde_json::to_value(&manifest)?)?;
    immutable_write(&incoming.join("manifest.v1.json"), &manifest_bytes)?;
    let marker = CompletionMarker {
        schema: "JopeDimeFollowUpStagingComplete.v1".to_string(),
        status: "COMPLETE".to_string(),
        manifest_sha256: sha256_bytes(&manifest_bytes),
    };
    immutable_write(
        &incoming.join("STAGING_COMPLETE"),
        &canonical_json(&serde_json::to_value(marker)?)?,
    )?;
    fs::create_dir_all(replacement_root())?;
    fs::rename(&incoming, replacement_root().join("staged"))?;
    File::open(replacement_root())?.sync_all()?;
    load_stage()
}
