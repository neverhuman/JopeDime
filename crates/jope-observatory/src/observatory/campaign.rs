use super::*;

/// Build a bounded, identity-checked observation of one campaign directory.
pub fn campaign(root: &Path) -> Value {
    let cfg = observe(&root.join("config.json"));
    let mut evidence = observe(&root.join("training/progress.json"));
    let loaded = observe(&root.join("training/loaded-receipt.json"));
    let p = evidence.value.clone();
    let identity_matches = cfg
        .sha256
        .as_deref()
        .is_some_and(|s| p["config_sha256"].as_str() == Some(s));
    let schema_matches = p["schema"] == "jope.flywheel-neural-progress/v1";
    let recognized = schema_matches && identity_matches;
    let stale = now() - evidence.observed_at.unwrap_or(0.) > 60.;
    let terminal = p["state"] == "completed" || p["state"] == "failed";
    if !recognized {
        evidence.status = "unqualified".into();
    } else if stale && !terminal {
        evidence.status = "stale".into();
    }
    let mut checkpoint_names = directory_names(&root.join("training/checkpoints"), 20_000);
    checkpoint_names.retain(|s| s.len() == 6 && s.bytes().all(|b| b.is_ascii_digit()));
    checkpoint_names.sort();
    let count = checkpoint_names.len();
    let checkpoints: Vec<_> = checkpoint_names.iter().rev().take(24).map(|name| {
        let manifest = observe(&root.join("training/checkpoints").join(name).join("manifest.json"));
        json!({"id":name,"evidence": {"source":manifest.source,"sha256":manifest.sha256,"status":manifest.status},"identity_matches":cfg.sha256.as_deref().is_some_and(|sha|manifest.value["config_sha256"].as_str()==Some(sha)),"manifest":selected(&manifest.value,&["schema","step","state_bytes","state_sha256","parameter_sha256","parameters","model_config","optimizer","learning_rate","weight_decay","config_sha256"]),"verification":"manifest observed; weights not rehashed by the dashboard"})
    }).collect();
    let mut cfg_public = selected(
        &cfg.value,
        &[
            "host",
            "schema",
            "unit",
            "duration_seconds",
            "closure_reserve_seconds",
            "expected_contexts",
            "expected_unique_teacher_programs",
            "research_optimizer_admission",
            "runtime_versions",
            "interpretation",
        ],
    );
    cfg_public["neural"] = selected(
        &cfg.value["neural"],
        &[
            "run_id",
            "max_updates",
            "max_epochs",
            "batch_size",
            "learning_rate",
            "seed",
            "device",
            "initialization",
            "layers",
            "heads",
            "d_model",
            "d_ff",
            "dropout",
            "max_seq",
            "input_dimension",
            "objective",
        ],
    );
    cfg_public["snapshot"] = selected(&cfg.value["snapshot"], &["sha256", "bytes"]);
    // These counters remain null on invalid/mismatched evidence, never invented zero.
    evidence.value = if recognized {
        selected(
            &p,
            &[
                "run_id",
                "state",
                "at",
                "optimizer_steps",
                "completed_epochs",
                "total_train_presentations",
                "unique_train_data_runs_seen",
                "unique_validation_data_runs_scored",
                "model_generated_programs",
                "novel_programs_executed",
                "published_results",
                "comparison_games",
                "elo",
                "winrate_vs_random",
                "winrate_vs_winner",
                "performance",
                "outbox",
                "proposal_catalog",
                "latest_full_validation",
                "scope",
            ],
        )
    } else {
        Value::Null
    };
    if evidence.value.is_object() {
        evidence.value["latest_full_validation"] = if p["latest_full_validation"].is_object() {
            selected(
                &p["latest_full_validation"],
                &[
                    "schema",
                    "epoch",
                    "context_count",
                    "loss",
                    "dropout",
                    "new_program_rewards_used",
                    "denominators",
                    "numerators",
                    "support",
                    "terms",
                ],
            )
        } else {
            Value::Null
        };
    }
    let source_map = observe(&root.join("source-map.json"));
    let mut terminal_receipt = observe(&root.join("status.json"));
    terminal_receipt.value = selected(
        &terminal_receipt.value,
        &["schema", "state", "at", "config_sha256", "optimizer_steps"],
    );
    json!({"id":root.file_name().unwrap_or_default().to_string_lossy(),"config":cfg_public,"config_sha256":cfg.sha256,"progress":evidence,"identity_matches":identity_matches,"curve":if recognized {curve(&p)} else {vec![]},"counts":selected(&loaded.value["counts"], &["partitions","loaded_kumo_contexts","current_eligible_views","pending_contexts","refused_rows","native_detail_records_declared"]),"membership_sha256":loaded.value["membership_sha256"],"checkpoints":checkpoints,"checkpoint_count_observed":count,"checkpoint_list_truncated":count>24,"source_identity":selected(&source_map.value,&["source_commit","source_main_sha","commit","verified_files"]),"terminal_receipt":terminal_receipt})
}
