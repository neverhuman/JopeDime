use super::*;

pub(super) fn preflight_targets(registry: &Path, stage: &StageManifest) -> Result<()> {
    for entry in &stage.targets {
        let path = public_target(registry, entry)?;
        let current = hash_file(&path)?;
        validate_target_hash(&path, &current, &entry.old_sha256, &entry.new_sha256)?;
    }
    for entry in &stage.untouched_original_winners {
        let path = untouched_target(registry, entry)?;
        if hash_file(&path)? != entry.sha256 {
            bail!("untouched original winner drifted: {}", path.display());
        }
    }
    Ok(())
}

pub(super) fn validate_target_hash(path: &Path, current: &str, old: &str, new: &str) -> Result<()> {
    if current != old && current != new {
        bail!(
            "{} has an unauthorized third hash {current}; expected old {old} or new {new}",
            path.display()
        );
    }
    Ok(())
}

pub(super) fn apply_targets(registry: &Path, stage: &StageManifest) -> Result<()> {
    preflight_targets(registry, stage)?;
    for kind in ["prediction", "report", "panel_ledger"] {
        for entry in stage.targets.iter().filter(|entry| entry.kind == kind) {
            let target = public_target(registry, entry)?;
            let current = hash_file(&target)?;
            validate_target_hash(&target, &current, &entry.old_sha256, &entry.new_sha256)?;
            if current == entry.new_sha256 {
                continue;
            }
            let bytes = fs::read(replacement_root().join("staged").join(&entry.staged_file))?;
            if sha256_bytes(&bytes) != entry.new_sha256 {
                bail!("staged replacement changed before publication");
            }
            atomic_replace(&target, &bytes)?;
            if hash_file(&target)? != entry.new_sha256 {
                bail!(
                    "atomic replacement hash audit failed for {}",
                    target.display()
                );
            }
        }
    }
    preflight_targets(registry, stage)?;
    for entry in &stage.targets {
        let path = public_target(registry, entry)?;
        if hash_file(&path)? != entry.new_sha256 {
            bail!("not every staged target reached its new hash");
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn rollback_targets(
    registry: &Path,
    stage: &StageManifest,
    archive: &Path,
) -> Result<()> {
    preflight_targets(registry, stage)?;
    for kind in ["prediction", "report", "panel_ledger"] {
        for entry in stage.targets.iter().filter(|entry| entry.kind == kind) {
            let target = public_target(registry, entry)?;
            let current = hash_file(&target)?;
            validate_target_hash(&target, &current, &entry.old_sha256, &entry.new_sha256)?;
            if current == entry.old_sha256 {
                continue;
            }
            let bytes = fs::read(archive.join(&entry.archive_file))?;
            if sha256_bytes(&bytes) != entry.old_sha256 {
                bail!("rollback archive payload hash mismatch");
            }
            atomic_replace(&target, &bytes)?;
        }
    }
    Ok(())
}

pub(super) fn audit_complete(registry: &Path, stage: &StageManifest) -> Result<()> {
    preflight_targets(registry, stage)?;
    for entry in &stage.targets {
        let target = public_target(registry, entry)?;
        if hash_file(&target)? != entry.new_sha256 {
            bail!(
                "publication target is not at its staged new hash: {}",
                target.display()
            );
        }
        if entry.kind == "report" {
            let report: Value = serde_json::from_slice(&fs::read(&target)?)?;
            if report["schema"] != "JopeDimeDatasetReport.v2"
                || report["provenance"]["exploratory_reused_test"] != true
                || report["comparisons"]["baselines"]
                    .as_object()
                    .map(|value| value.len())
                    != Some(6)
                || report["timings"]["fit_seconds"].as_f64().is_none()
                || report["timings"]["predict_seconds"].as_f64().is_none()
                || report["prediction"]["sha256"].as_str().is_none()
                || report["selection_proxy"]["authority_label"]
                    != "CV7 training-only selection proxy; not final test authority"
            {
                bail!("published v2 report audit failed for {}", target.display());
            }
        }
    }
    let ledger: Value = serde_json::from_slice(&fs::read(original_ledger_path())?)?;
    if ledger["schema"] != "JopeDimePanelResults.v2"
        || ledger["dataset_count"].as_u64() != Some(42)
        || ledger["follow_up"]["exploratory_reused_test"] != true
        || ledger["selection_accounting"]["follow_up_evolutionary_slots"].as_u64()
            != Some(FOLLOW_UP_SELECTION_SLOTS as u64)
        || ledger["selection_accounting"]["updated_active_panel_evolutionary_slots"].as_u64()
            != Some(ACTIVE_PANEL_SELECTION_SLOTS as u64)
        || ledger["selection_accounting"]["historical_evolutionary_slots_across_both_experiments"]
            .as_u64()
            != Some(HISTORICAL_SELECTION_SLOTS as u64)
        || ledger["outcomes"].as_array().map(Vec::len) != Some(42)
        || ledger["pairwise_results"].as_array().map(Vec::len) != Some(21)
        || ledger["elo"]["standings"].as_array().map(Vec::len) != Some(7)
    {
        bail!("published v2 panel ledger audit failed");
    }
    Ok(())
}

pub(super) fn publish_follow_up(
    registry: &Path,
    freezes: &BTreeMap<String, ValidatedFreeze>,
) -> Result<()> {
    if replacement_complete_path().exists() {
        let stage = load_stage()?;
        audit_complete(registry, &stage)?;
        println!("follow-up replacement is already complete and passed the full hash audit");
        return Ok(());
    }
    // Baseline/test data are first opened below, after all 35 freezes and all
    // replay prefixes have passed the training-only validator.
    let source_bytes = authorized_source_ledger_bytes()?;
    let source = validate_compiled_cohort(&source_bytes)?;
    let stage = create_stage(registry, freezes, &source)?;
    let (_pointer, archive) = ensure_archive(registry, &stage)?;
    validate_archive(
        &stage,
        &serde_json::from_slice(&fs::read(replacement_root().join("ARCHIVE_COMPLETE"))?)?,
    )?;
    apply_targets(registry, &stage)?;
    audit_complete(registry, &stage)?;
    let marker = json!({
        "schema": "JopeDimeOfficialReplacementComplete.v1",
        "status": "COMPLETE",
        "stage_manifest_sha256": hash_file(&stage_manifest_path())?,
        "archive_manifest_sha256": archive.file_name().and_then(|name| name.to_str()),
        "reports_replaced": 35,
        "predictions_replaced": 35,
        "untouched_original_winners": 7,
        "panel_ledgers_replaced": 1,
        "exploratory_reused_test": true,
    });
    immutable_write(&replacement_complete_path(), &canonical_json(&marker)?)?;
    println!(
        "published all 35 exploratory follow-up outcomes, one v2 panel ledger, and a complete rollback archive"
    );
    Ok(())
}
