fn rebuild_summary(
    root: &Path,
    run_id: &str,
    scan_output: Option<String>,
) -> Result<TerminalSummary> {
    let manifest_path = root.join("campaign.v1.json");
    let manifest: CampaignManifest = load_json(&manifest_path)?;
    let campaign_sha256 = sha256_file(&manifest_path)?;
    let mut datasets = Vec::with_capacity(manifest.datasets.len());
    for spec in &manifest.datasets {
        let result: DatasetResult = load_json(&final_result_path(root, &spec.dataset_hash))?;
        if !result_matches_campaign(
            &result,
            run_id,
            &campaign_sha256,
            spec,
            &manifest.release_binary_sha256,
        ) {
            bail!(
                "TimeSeries42 dataset result is outdated for {}",
                spec.dataset_hash
            );
        }
        datasets.push(result);
    }
    datasets.sort_by(|left, right| left.dataset_hash.cmp(&right.dataset_hash));
    let logical_evaluations = datasets.iter().map(|row| row.logical_evaluations).sum();
    let valid_evaluations = datasets.iter().map(|row| row.valid_evaluations).sum();
    let summary = TerminalSummary {
        schema: SUMMARY_SCHEMA.to_string(),
        test_adaptive: true,
        benchmark_valid: false,
        run_id: run_id.to_string(),
        base_run_id: manifest.base_run_id.clone(),
        inherited_base_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| row.continuation_status.as_deref() == Some("inherited_base_win"))
                .count()
        }),
        improved_base_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| row.continuation_status.as_deref() == Some("base_win_improved"))
                .count()
        }),
        continuation_strict_wins: manifest.base_run_id.as_ref().map(|_| {
            datasets
                .iter()
                .filter(|row| {
                    matches!(
                        row.continuation_status.as_deref(),
                        Some("gen30_strict_win" | "winall_strict_win")
                    )
                })
                .count()
        }),
        dataset_count: datasets.len(),
        wins: datasets.iter().filter(|row| row.status == "win").count(),
        losses: datasets.iter().filter(|row| row.status == "loss").count(),
        ties: datasets.iter().filter(|row| row.status == "tie").count(),
        exhausted: datasets
            .iter()
            .filter(|row| row.status == "exhausted")
            .count(),
        logical_evaluations,
        valid_evaluations,
        datasets,
        sotarena_scan_output: scan_output,
        completed_at_unix_ms: unix_millis(),
    };
    write_current_json(
        &root.join("terminal-summary.v1.json"),
        &summary,
        "terminal-summary",
    )?;
    write_ranked_csv(root, &summary)?;
    Ok(summary)
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_string()
    }
}

fn write_ranked_csv(root: &Path, summary: &TerminalSummary) -> Result<()> {
    let mut rows = summary.datasets.clone();
    rows.sort_by(|left, right| {
        let left_mase = left.best_jope_mase.unwrap_or(f64::INFINITY);
        let right_mase = right.best_jope_mase.unwrap_or(f64::INFINITY);
        left.status
            .cmp(&right.status)
            .then(left_mase.total_cmp(&right_mase))
            .then(left.dataset_hash.cmp(&right.dataset_hash))
    });
    let mut bytes = b"dataset,status,official_leader,official_mase,best_mase,best_rmse,mase_delta,best_generation,best_slot,semantic_sha256,seed_family\n".to_vec();
    for row in rows {
        bytes.extend_from_slice(
            format!(
                "{},{},{},{:.17},{},{},{},{},{},{},{}\n",
                row.dataset_hash,
                row.status,
                csv_field(&row.official_leader_model),
                row.official_leader_mase,
                row.best_jope_mase
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.best_jope_rmse
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.mase_delta_vs_official
                    .map(|value| format!("{value:.17}"))
                    .unwrap_or_default(),
                row.best_generation
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                row.best_slot
                    .map(|value| value.to_string())
                    .unwrap_or_default(),
                row.best_semantic_sha256.unwrap_or_default(),
                csv_field(row.seed_source_family.as_deref().unwrap_or("")),
            )
            .as_bytes(),
        );
    }
    immutable_or_equal(&root.join("ranked-summary.csv"), &bytes)
}

fn run_sotarena_scan(root: &Path) -> Result<String> {
    let output = Command::new(sotarena_pull_str())
        .args(["scan", "--task", "timeseries"])
        .output()
        .context("run sotarena_pull scan --task timeseries")?;
    let scan_dir = root.join("sotarena-scan");
    fs::create_dir_all(&scan_dir)?;
    fs::write(scan_dir.join("stdout.txt"), &output.stdout)?;
    fs::write(scan_dir.join("stderr.txt"), &output.stderr)?;
    if !output.status.success() {
        bail!(
            "sotarena_pull scan failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn timeseries_report_value(
    spec: &DatasetSpec,
    result: &DatasetResult,
    terminal_summary_path: &Path,
    terminal_summary_sha256: &str,
) -> Value {
    json!({
        "report_schema_version": JOPE_TIMESERIES_REPORT_SCHEMA,
        "algorithm": "jope-dime-regressor",
        "model_name": "jope-dime-regressor",
        "model_class": "aspfm_generated_rust_timeseries_reduction",
        "task_type": "timeseries",
        "status": "completed",
        "dataset_hash": result.dataset_hash,
        "source_path": result.best_receipt_file,
        "benchmark_valid": false,
        "test_adaptive": true,
        "selection_bias": "test-selected",
        "train_rows": spec.train_rows,
        "test_rows": spec.test_rows,
        "n_features": spec.reduction_features,
        "metrics": {
            "mase": result.best_jope_mase,
            "negative_mase": result.best_jope_mase.map(|value| -value),
            "rmse": result.best_jope_rmse,
            "mase_delta_vs_official": result.mase_delta_vs_official,
            "official_leader_mase": result.official_leader_mase,
            "strict_winner": result.strict_winner,
            "best_beats_official": result.best_beats_official
        },
        "comparison": {
            "official_leader_model": result.official_leader_model,
            "official_leader_mase": result.official_leader_mase,
            "generation_zero_best_mase": result.generation_zero_best_mase,
            "terminal_status": result.status,
            "strict_winner": result.strict_winner,
            "continuation_status": result.continuation_status,
            "base_status": result.base_status,
            "base_strict_winner": result.base_strict_winner,
            "base_best_jope_mase": result.base_best_jope_mase
        },
        "genome": result.best_canonical_genome,
        "genome_hash": result.best_semantic_sha256,
        "semantic_sha256": result.best_semantic_sha256,
        "provenance": {
            "run_id": result.run_id,
            "base_run_id": result.base_run_id,
            "best_evidence_run_id": result.best_evidence_run_id,
            "campaign_sha256": result.campaign_sha256,
            "release_binary_sha256": result.release_binary_sha256,
            "terminal_summary_path": terminal_summary_path.to_string_lossy(),
            "terminal_summary_sha256": terminal_summary_sha256,
            "receipt_file": result.best_receipt_file,
            "receipt_sha256": result.best_receipt_sha256,
            "prediction_sha256": result.best_prediction_sha256,
            "best_generation": result.best_generation,
            "best_slot": result.best_slot,
            "logical_evaluations": result.logical_evaluations,
            "valid_evaluations": result.valid_evaluations,
            "completed_generations": result.completed_generations,
            "completed_at_unix_ms": result.completed_at_unix_ms
        },
        "environment": {
            "execution_host": "scq",
            "nodes": NODES,
            "candidate_timeout_seconds": CANDIDATE_TIMEOUT_SECONDS,
            "candidate_threads": SHARD_SIZE,
            "corpus_eval": corpus_eval_str()
        }
    })
}

fn write_report_file(path: &Path, value: &Value) -> Result<String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    immutable_or_equal(path, &bytes)?;
    Ok(sha256_bytes(&bytes))
}

fn write_timeseries_dataset_reports(
    manifest: &CampaignManifest,
    summary: &TerminalSummary,
    summary_path: &Path,
) -> Result<usize> {
    let summary_sha256 = sha256_file(summary_path)?;
    let specs = manifest
        .datasets
        .iter()
        .map(|spec| (spec.dataset_hash.as_str(), spec))
        .collect::<BTreeMap<_, _>>();
    let mut written = 0usize;
    for result in &summary.datasets {
        if result.best_jope_mase.is_none() {
            continue;
        }
        let spec = specs
            .get(result.dataset_hash.as_str())
            .with_context(|| format!("missing spec for {}", result.dataset_hash))?;
        let mut report = timeseries_report_value(spec, result, summary_path, &summary_sha256);
        let checkpoint_path = result.best_generation.map(|generation| {
            dataset_root(&campaign_root(&manifest.run_id), &result.dataset_hash)
                .join("generations")
                .join(format!("generation-{generation:06}"))
                .join("generation.v1.json")
        });
        let checkpoint_sha256 = checkpoint_path
            .as_ref()
            .filter(|path| path.is_file())
            .map(|path| sha256_file(path))
            .transpose()?;
        let object = report
            .as_object_mut()
            .context("timeseries report must be an object")?;
        object.insert("official_combined_rank_eligible".to_string(), json!(false));
        object.insert("ranking_eligible".to_string(), json!(false));
        object.insert(
            "comparator".to_string(),
            json!({
                "lock_sha256": spec.comparator_lock_sha256,
                "accepted_report_count": spec.accepted_external_reports.len(),
                "winner_model": spec.official_leader.model,
                "winner_report_sha256": spec.official_leader.report_sha256,
            }),
        );
        object.insert(
            "checkpoint".to_string(),
            json!({
                "path": checkpoint_path,
                "sha256": checkpoint_sha256,
            }),
        );
        object.insert(
            "confirmation".to_string(),
            json!({
                "strict_beat_confirmed": result.strict_winner,
                "receipt_path": result.best_receipt_file,
                "receipt_sha256": result.best_receipt_sha256,
                "prediction_sha256": result.best_prediction_sha256,
            }),
        );
        let report_path = Path::new(registry_str())
            .join(&result.dataset_hash)
            .join("reports")
            .join(JOPE_TIMESERIES_REPORT_FILE);
        write_report_file(&report_path, &report)?;
        written += 1;
    }
    Ok(written)
}
