use super::*;

pub(crate) fn publish_new(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("publication path has no parent")?;
    fs::create_dir_all(parent)?;
    if path.exists() {
        bail!("refusing to overwrite {}", path.display());
    }
    let staged = parent.join(format!(
        ".jope-dime-{}-{}.stage",
        std::process::id(),
        &sha256_bytes(bytes)[..16]
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&staged)?;
    file.write_all(bytes)?;
    file.flush()?;
    file.sync_all()?;
    drop(file);
    let mut permissions = fs::metadata(&staged)?.permissions();
    permissions.set_readonly(true);
    fs::set_permissions(&staged, permissions)?;
    fs::hard_link(&staged, path).with_context(|| format!("publish {}", path.display()))?;
    fs::remove_file(staged)?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

pub(crate) fn ensure_publication_targets_absent(registry: &Path, panel_id: &str) -> Result<()> {
    let root = run_root(panel_id);
    let ledger = root.join("jope-dime-panel-results.v1.json");
    if ledger.exists() {
        bail!("refusing to overwrite {}", ledger.display());
    }
    for hash in DATASETS {
        for path in [
            root.join(hash).join("phase2/predictions.csv"),
            registry
                .join(hash)
                .join("predictions")
                .join(PREDICTION_FILE),
            registry.join(hash).join("reports").join(REPORT_FILE),
        ] {
            if path.exists() {
                bail!("refusing to overwrite {}", path.display());
            }
        }
    }
    Ok(())
}

pub(crate) fn score_panel(registry: &Path, panel_id: &str) -> Result<()> {
    // The baseline manifest and every test.csv remain unopened until this
    // complete panel-wide training-only validation returns successfully.
    let frozen = load_validated_panel(registry, panel_id)?;
    let baselines = load_baselines()?;
    ensure_publication_targets_absent(registry, panel_id)?;

    let mut scored = Vec::with_capacity(DATASETS.len());
    for hash in DATASETS {
        eprintln!("[jope-dime] final scoring {hash}");
        scored.push(score_dataset(
            registry,
            hash,
            &frozen[*hash],
            &baselines[*hash],
        )?);
    }
    if scored.len() != 42 || scored.iter().any(|row| row.predictions.is_empty()) {
        bail!("final scoring did not produce exactly 42 complete outcomes");
    }

    let root = run_root(panel_id);
    for row in &scored {
        let local_prediction = root.join(&row.dataset_hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(&row.dataset_hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry
            .join(&row.dataset_hash)
            .join("reports")
            .join(REPORT_FILE);
        publish_new(&local_prediction, &row.prediction_bytes)?;
        publish_new(&public_prediction, &row.prediction_bytes)?;
        publish_new(&public_report, &row.report_bytes)?;
        if hash_file(&local_prediction)? != row.prediction_sha256
            || hash_file(&public_prediction)? != row.prediction_sha256
            || hash_file(&public_report)? != row.report_sha256
        {
            bail!("published artifact hash mismatch for {}", row.dataset_hash);
        }
    }

    let wins = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "win")
        .count();
    let losses = scored
        .iter()
        .filter(|row| row.outcome["strict_overall_outcome"] == "loss")
        .count();
    let ties = 42 - wins - losses;
    let tabfm_wins = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "win")
        .count();
    let tabfm_losses = scored
        .iter()
        .filter(|row| row.outcome["tabfm_head_to_head_outcome"] == "loss")
        .count();
    let tabfm_ties = 42 - tabfm_wins - tabfm_losses;
    let tabfm_status = match tabfm_wins.cmp(&tabfm_losses) {
        Ordering::Greater => "winning",
        Ordering::Equal => "tied",
        Ordering::Less => "losing",
    };
    let outcomes = scored
        .iter()
        .map(|row| row.outcome.clone())
        .collect::<Vec<_>>();
    let ledger = json!({
        "schema": "JopeDimePanelResults.v1",
        "panel_id": panel_id,
        "panel_text_sha256": PANEL_SHA256,
        "baseline_panel_sha256": PANEL_ANALYSIS_SHA256,
        "dataset_count": 42,
        "selection_accounting": {
            "independent_freezes": 42,
            "distinct_selected_semantic_genomes": 42,
            "evolutionary_slots": 25200,
            "full_cv7_finalist_receipts": 420,
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
        "pairwise_results": pairwise_results(&scored),
        "elo": {
            "initial_rating": 1500.0,
            "k_factor": 16.0,
            "method": "Each dataset is a simultaneous 7-algorithm round robin; lower RMSE wins and exact equality ties; panel order is pinned.",
            "standings": elo_standings(&scored),
        },
        "outcomes": outcomes,
    });
    let ledger_path = root.join("jope-dime-panel-results.v1.json");
    publish_new(&ledger_path, &canonical_json(&ledger)?)?;
    println!(
        "published 42 reports and predictions: overall {wins}W/{losses}L/{ties}T; TabFM {tabfm_wins}W/{tabfm_losses}L/{tabfm_ties}T"
    );
    Ok(())
}

pub(crate) fn summarize(registry: &Path, panel_id: &str) -> Result<()> {
    let path = run_root(panel_id).join("jope-dime-panel-results.v1.json");
    let value: Value = serde_json::from_slice(&fs::read(&path)?)?;
    if value["schema"] == "JopeDimePanelResults.v2" {
        if panel_id != "grok4p5-42" {
            bail!("the v2 follow-up ledger is only valid at the original panel path");
        }
        return followup::summarize(registry);
    }
    if value["schema"] != "JopeDimePanelResults.v1"
        || value["dataset_count"].as_u64() != Some(42)
        || value["outcomes"].as_array().map(Vec::len) != Some(42)
        || value["pairwise_results"].as_array().map(Vec::len) != Some(21)
        || value["elo"]["standings"].as_array().map(Vec::len) != Some(7)
        || value["selection_accounting"]["evolutionary_slots"].as_u64() != Some(25_200)
        || value["selection_accounting"]["full_cv7_finalist_receipts"].as_u64() != Some(420)
    {
        bail!("panel ledger is incomplete or invalid");
    }
    for outcome in value["outcomes"].as_array().unwrap() {
        let hash = outcome["dataset_hash"]
            .as_str()
            .context("outcome dataset hash is missing")?;
        if !DATASETS.contains(&hash) {
            bail!("panel ledger contains an unknown dataset");
        }
        let prediction_sha256 = outcome["prediction_sha256"]
            .as_str()
            .context("outcome prediction hash is missing")?;
        let report_sha256 = outcome["report_sha256"]
            .as_str()
            .context("outcome report hash is missing")?;
        let local_prediction = run_root(panel_id).join(hash).join("phase2/predictions.csv");
        let public_prediction = registry
            .join(hash)
            .join("predictions")
            .join(PREDICTION_FILE);
        let public_report = registry.join(hash).join("reports").join(REPORT_FILE);
        if hash_file(&local_prediction)? != prediction_sha256
            || hash_file(&public_prediction)? != prediction_sha256
            || hash_file(&public_report)? != report_sha256
        {
            bail!("published artifact hash mismatch for {hash}");
        }
        let report: Value = serde_json::from_slice(&fs::read(public_report)?)?;
        if report["schema"] != "JopeDimeDatasetReport.v1"
            || report["dataset_hash"].as_str() != Some(hash)
            || report["status"] != "completed"
            || report["prediction"]["sha256"].as_str() != Some(prediction_sha256)
        {
            bail!("published report is invalid for {hash}");
        }
    }
    println!(
        "42 outcomes; strict overall {}W/{}L/{}T; TabFM {}W/{}L/{}T ({})",
        value["strict_overall"]["wins"],
        value["strict_overall"]["losses"],
        value["strict_overall"]["ties"],
        value["tabfm_head_to_head"]["wins"],
        value["tabfm_head_to_head"]["losses"],
        value["tabfm_head_to_head"]["ties"],
        value["tabfm_head_to_head"]["status"]
            .as_str()
            .unwrap_or("unknown"),
    );
    Ok(())
}
