use super::details::decimal_count;
use super::evidence::{
    age, campaign_resource_evidence, completed, count_pair, duration, experiment_worker_evidence,
    live_operations, memory_throttle_reason, metric_tail, metrics, number, panel, phase,
    policy_receipt, state, status_age, status_age_with_limit,
};
use super::prelude::*;
use super::tasks::{campaign_task_state, label, training_hold};

pub fn campaign(root: &Path) -> Value {
    let mut run = state(&root.join("runs/status.json"));
    if !run.is_object() {
        run = serde_json::json!({});
    }
    let index_path = run["contexts_index"]
        .as_str()
        .unwrap_or("data/encodings/contexts-v1/index.json");
    let index = state(&root.join(index_path));
    if let Some(encodings) = index["encodings"].as_array() {
        let families: BTreeSet<_> = encodings
            .iter()
            .filter_map(|v| v["family"].as_str())
            .collect();
        run["contexts"] = serde_json::json!({"completed":encodings.len(),"total":encodings.len(),"families":families.len(),"dimensions":number(&index,"dimensions"),"qualification":index["qualification"],"source":index_path});
    }
    if let Some(manifests) = run["prepared_job_manifests"].as_array() {
        let mut views = BTreeSet::new();
        let mut valid = !manifests.is_empty();
        for name in manifests {
            let jobs = name
                .as_str()
                .map(|name| state(&root.join(name)))
                .unwrap_or_default();
            if let Some(rows) = jobs["jobs"].as_array() {
                for row in rows {
                    if let (Some(prepared), Some(cut)) =
                        (row["prepared"].as_str(), row["cut"].as_u64())
                    {
                        if let Some(dataset) = Path::new(prepared).file_name() {
                            views.insert((dataset.to_string_lossy().into_owned(), cut));
                        } else {
                            valid = false;
                        }
                    } else {
                        valid = false;
                    }
                }
            } else {
                valid = false;
            }
        }
        if valid {
            let datasets = views
                .iter()
                .map(|(dataset, _)| dataset)
                .collect::<BTreeSet<_>>()
                .len();
            run["preparation"] = serde_json::json!({"views":views.len(),"datasets":datasets});
            run["contexts"]["total"] = views.len().into();
        }
    }
    run["cycle"] = state(&root.join("runs/cycle-progress.json"));
    run["operations"] = live_operations(root);
    run["continuous"] = state(&root.join("runs/continuous-progress.json"));
    if let Some(path) = run["continuous"]["packing_progress"].as_str() {
        run["continuous_packing"] = state(&root.join(path));
    }
    run["frozen_monitoring"] = state(&root.join("runs/frozen-monitoring.json"));
    run["encoding_pipeline"] = state(&root.join("runs/encoding-pipeline.json"));
    run["campaign_service"] = state(&root.join("runs/campaign-progress.json"));
    run["bootstrap"] = state(&root.join("runs/cohort64-bootstrap-readiness.json"));
    run["corpus_progress"] = state(&root.join("runs/corpus-progress.json"));
    run["runtime_readiness"] = state(&root.join("runs/runtime-readiness.json"));
    // Nothing in this workspace writes the next three receipts. Their writers,
    // `watch_campaign_reports.py`, `opportunistic_proposals.py` and
    // `opportunistic_proposals_v2.py`, were archived by the Python removal P1 (ADR-0006).
    // A receipt left in the state root is history and keeps its saved status; without
    // one, no row is shown.
    run["report_progress"] = state(&root.join("runs/report-progress.json"));
    run["proposal_library"] = state(&root.join("runs/proposal-library.json"));
    run["pure_proposal_library"] = state(&root.join("runs/pure-proposal-library.json"));
    run["seed_experiment"] = state(&root.join("runs/seed-quality-experiment-v1/progress.json"));
    run["seed_quality"] = state(&root.join("runs/seed-quality-experiment-v1/report.json"));
    run["binary_qualification"] = state(&root.join("runs/binary-qualification-v1/progress.json"));
    run["proposal_library_admission"] = state(&root.join("runs/proposal-library-admission.json"));
    if let Some(path) = run["proposal_library_admission"]["receipt"].as_str() {
        let path = root.join(path);
        let receipt = state(&path);
        if receipt["outcome"] == "admitted" {
            run["proposal_library_admission"]["committed"] = true.into();
            run["proposal_library_admission"]["requested_jobs"] = receipt["requested_jobs"].clone();
            run["proposal_library_gp"] = library_gp::progress(root, &path, &receipt);
        }
    }
    run["auxiliary"] = state(&root.join("runs/auxiliary-training.json"));
    if !run["auxiliary"]["maximum_steps"].is_u64() {
        if let Some(output) = run["auxiliary"]["output"].as_str() {
            let spec = state(&root.join(output).join("spec.json"));
            if spec["options"]["max_steps"].is_u64() {
                run["auxiliary"]["maximum_steps"] = spec["options"]["max_steps"].clone();
            }
        }
    }
    run["teacher"] = state(&root.join("runs/teacher-progress.json"));
    run["policy"] = state(&root.join("runs/neural-service.json"));
    run["latest_neural_activity"] = run["policy"].clone();
    run["hyperion_recovery"] =
        state(&root.join("runs/parity/hyperion-reviewed4-recovery-parity.json"));
    let mut cycles = BTreeSet::new();
    for entry in std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.join("cycle.json").is_file() {
            cycles.insert(path);
        }
    }
    if let Some(path) = run["campaign_service"]["current_cycle"].as_str() {
        let current = root.join(path);
        if let Some(parent) = current.parent() {
            for row in state(&parent.join("completed-cycles.json"))["cycles"]
                .as_array()
                .into_iter()
                .flatten()
            {
                if let Some(path) = row["cycle"].as_str() {
                    cycles.insert(root.join(path));
                }
            }
        }
        cycles.insert(current);
    }
    // Display chronology comes from durable receipt times, never directory spelling.
    let current_cycle = run["campaign_service"]["current_cycle"]
        .as_str()
        .map(|p| root.join(p));
    let mut cycles = cycles
        .into_iter()
        .map(|path| {
            let complete = state(&path.join("complete.json"));
            let started = state(&path.join("cycle.json"));
            let timestamp = [
                number(&complete, "completed_at"),
                number(&complete, "timestamp"),
                number(&started, "timestamp"),
                number(&started, "started_at"),
            ]
            .into_iter()
            .max()
            .unwrap_or(0);
            let current = current_cycle.as_ref() == Some(&path);
            (current, timestamp, path)
        })
        .collect::<Vec<_>>();
    cycles.sort();
    let mut reports = Vec::new();
    let mut checkpoints = Vec::new();
    for (_, _, cycle) in cycles.iter().rev().take(16).rev() {
        let report = state(&cycle.join("report-v1/report.json"));
        if report["dl_gp_dl_complete"] == true {
            reports.push(serde_json::json!({"cycle":cycle.file_name().unwrap_or_default().to_string_lossy(),"path":cycle.join("report-v1/report.json"),"records":report["teaching_records"],"predictors":report["fitted_predictors_passed"],"views":report["comparison_views"],"searches":report["comparison_searches"],"validation":report["partitions"]["meta_validation"],"generation":report["generation"],"budget":report["matched_evaluation_budget"],"clock_coverage":report["clock_coverage"]}));
        } else {
            let raw = state(&cycle.join("comparisons.json"));
            let complete = state(&cycle.join("complete.json"));
            if raw["comparisons"].is_array() {
                reports.push(serde_json::json!({"cycle":cycle.file_name().unwrap_or_default().to_string_lossy(),"path":cycle.join("comparisons.json"),"raw_final_scores_only":true,"comparisons":raw["comparisons"],"views":raw["comparisons"].as_array().map_or(0,Vec::len),"complete":complete["dl_gp_dl_complete"],"predictors":complete["predictor_exports_passed"]}));
            }
        }
        for block in ["train-1", "train-2"] {
            let path = cycle.join(block);
            let latest = state(&path.join("latest.json"));
            if latest["checkpoint"].is_string() && !path.join("invalidated.json").exists() {
                let samples = metric_tail(&path.join("metrics.jsonl"), 16 * 1024);
                let last = samples.last().cloned().unwrap_or_default();
                checkpoints.push(serde_json::json!({"name":format!("{} / {block}",cycle.file_name().unwrap_or_default().to_string_lossy()),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":last,"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    if let Some(path) = run["policy"]["run"].as_str().map(|p| root.join(p)) {
        if !checkpoints
            .iter()
            .any(|v| v["path"].as_str() == path.to_str())
            && !path.join("invalidated.json").is_file()
        {
            let latest = state(&path.join("latest.json"));
            if latest["checkpoint"].is_string() && policy_receipt(&run["policy"]) {
                checkpoints.push(serde_json::json!({"name":path.file_name().unwrap_or_default().to_string_lossy(),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":run["policy"],"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    // Standalone real-corpus training remains visible while a diagnostic worker
    // temporarily owns the global neural-service receipt.
    for entry in std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .flatten()
    {
        let path = entry.path();
        if path.join("invalidated.json").is_file()
            || !path.join("latest.json").is_file()
            || checkpoints
                .iter()
                .any(|v| v["path"].as_str() == path.to_str())
        {
            continue;
        }
        let latest = state(&path.join("latest.json"));
        let samples = metric_tail(&path.join("metrics.jsonl"), 16 * 1024);
        if let Some(last) = samples.last().filter(|v| policy_receipt(v)) {
            if latest["checkpoint"].is_string() {
                checkpoints.push(serde_json::json!({"name":path.file_name().unwrap_or_default().to_string_lossy(),"path":path,"checkpoint":latest["checkpoint"],"step":latest["step"],"complete":path.join("complete.json").is_file(),"metrics":last,"initialization":state(&path.join("initialization.json"))}));
            }
        }
    }
    checkpoints.sort_by_key(|v| number(&v["metrics"], "timestamp"));
    run["reports"] = reports.into();
    run["checkpoints"] = checkpoints.into();
    if !policy_receipt(&run["policy"]) {
        if let Some(last) = run["checkpoints"]
            .as_array()
            .and_then(|v| v.last())
            .cloned()
        {
            if policy_receipt(&last["metrics"]) {
                run["policy"] = last["metrics"].clone();
                run["policy"]["checkpoint"] = last["checkpoint"].clone();
                if last["complete"] == true {
                    run["policy"]["status"] = "complete".into();
                }
            }
        }
    }
    if let Some(path) = run["policy"]["run"].as_str() {
        run["learning_curve"] = metrics(&root.join(path).join("metrics.jsonl")).into();
    }
    if let Some(path) = run["family_registry"].as_str() {
        if let Ok(registry) = read_json::<crate::families::Registry>(&root.join(path)) {
            run["family_readiness"] = registry.quota_report();
            run["dataset_names"] = registry
                .families
                .values()
                .flat_map(|family| {
                    family.datasets.iter().map(move |dataset| {
                        (
                            dataset.dataset.clone(),
                            Value::String(format!(
                                "{} / {}",
                                family.family.trim_start_matches("rdatasets-"),
                                dataset.dataset.chars().take(4).collect::<String>()
                            )),
                        )
                    })
                })
                .collect::<serde_json::Map<_, _>>()
                .into();
        }
    }
    run["gpu_wait"] = state(&root.join("runs/gpu-wait.json"));
    // Keep the campaign's completed policy visible while a newer encoding or
    // recovery-check receipt owns the global live neural panel.
    let active_cycle = std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            let modified = path.join("cycle.json").metadata().ok()?.modified().ok()?;
            Some((modified, path))
        })
        .max_by_key(|(modified, _)| *modified);
    let active_path = run["campaign_service"]["current_cycle"]
        .as_str()
        .map(|p| root.join(p))
        .filter(|p| p.join("cycle.json").is_file())
        .or_else(|| active_cycle.map(|(_, p)| p));
    if let Some(path) = active_path {
        run["active_cycle"] = path.to_string_lossy().as_ref().into();
        let exports = state(&path.join("predictor-exports.json"));
        run["predictor_exports"] = exports.clone();
        let mut selected = state(&path.join("train-1/latest.json"));
        let best = state(&path.join("auxiliary/train/best.json"));
        if best["checkpoint"].is_string() {
            selected = best;
        }
        if path.join("train-2/complete.json").is_file() {
            selected = state(&path.join("train-2/latest.json"));
        }
        if let Some(checkpoint) = selected["checkpoint"].as_str() {
            let checkpoint_state = state(&root.join(checkpoint).join("state.json"));
            selected["corpus_sha256"] = checkpoint_state["corpus_sha256"].clone();
            selected["model_sha256"] = checkpoint_state["model_sha256"].clone();
            run["selected_checkpoint"] = selected;
        }
        let cycle_gp = &run["cycle"]["neural_seeded_gp"];
        if exports["completed"].is_number()
            && exports["total"].is_number()
            && number(&exports, "completed") < number(&exports, "total")
            && number(cycle_gp, "total") > 0
            && number(cycle_gp, "completed") == number(cycle_gp, "total")
        {
            run["cycle"]["phase"] = "Fit and save GP predictors".into();
            run["cycle"]["reason"] = format!(
                "{} / {} fitted predictor exports saved; each fit has a 30-second and 8 GiB limit.",
                number(&exports, "completed"),
                number(&exports, "total")
            )
            .into();
        }
        let live_train = state(&root.join("runs/neural-service.json"));
        if live_train["run"]
            .as_str()
            .is_some_and(|p| root.join(p) == path.join("train-2"))
        {
            run["cycle"]["retraining"]["step"] = live_train["step"].clone();
        }
        for block in ["train-1", "train-2"] {
            let receipt = state(&path.join(block).join("latest.json"));
            if path.join(block).join("complete.json").exists() && receipt["step"].is_number() {
                let steps = number(&receipt, "step");
                let policy = serde_json::json!({"phase":"completed policy training block","step":steps,"steps":steps,"world":8,"checkpoint":receipt["checkpoint"]});
                if block == "train-1" {
                    run["fresh_policy"] = policy;
                }
                run["checkpoint"] = receipt;
            }
        }
    }
    let latest = std::fs::read_dir(root.join("runs"))
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path().join("latest.json");
            let modified = path.metadata().ok()?.modified().ok()?;
            let receipt = state(&path);
            receipt["checkpoint"].as_str()?;
            Some((modified, receipt))
        })
        .max_by_key(|(modified, _)| *modified);
    if run["checkpoint"].is_null() {
        if let Some((_, receipt)) = latest {
            run["checkpoint"] = receipt;
        }
    }
    // The bootstrap progress receipt does not change as later corpora grow.
    // Read only the index header after a complete cycle binds its corpus hash.
    if let Some(active) = run["active_cycle"].as_str() {
        if let Some(parent) = Path::new(active).parent() {
            for entry in std::fs::read_dir(parent).into_iter().flatten().flatten() {
                let cycle = entry.path();
                let complete = state(&cycle.join("complete.json"));
                if run["selected_checkpoint"]["corpus_sha256"].is_string()
                    && complete["teaching_corpus_sha256"]
                        == run["selected_checkpoint"]["corpus_sha256"]
                {
                    let mut bytes = vec![0; 2048];
                    if let Ok(mut file) = std::fs::File::open(cycle.join("corpus-2/index.json")) {
                        if let Ok(n) = file.read(&mut bytes) {
                            let header = String::from_utf8_lossy(&bytes[..n]);
                            if let Some((_, suffix)) = header.split_once("\"records\":") {
                                let digits = suffix
                                    .trim_start()
                                    .chars()
                                    .take_while(char::is_ascii_digit)
                                    .collect::<String>();
                                if let Ok(records) = digits.parse::<u64>() {
                                    run["corpus_progress"]["records"] = records.into();
                                    run["corpus_progress"]["output"] =
                                        cycle.join("corpus-2").to_string_lossy().as_ref().into();
                                    run["corpus_progress"]["reason"] =
                                        "Current selected checkpoint corpus; saved index header"
                                            .into();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    run
}
pub(crate) fn goal_phase<'a>(run: &'a Value, neural: &'a Value, gp: &'a Value) -> &'a str {
    if run["continuous"].is_object() {
        if run["operations"]["services"]["jope-parallel-campaign"]["ActiveState"] != "active"
            || now().saturating_sub(number(&run["continuous"], "timestamp")) > 15
        {
            return "Continuous controller unavailable · inspect saved evidence";
        }
        if run["continuous"]["phase"] == "paused" {
            return "Continuous learning paused";
        }
        if completed(neural) {
            return "Continuous learning · scheduling fresh GP targets";
        }
    }
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    if training_hold(run, policy).is_some() {
        return "Automatic training held · inspect task state";
    }
    if number(neural, "timestamp") > 0
        && !completed(neural)
        && now().saturating_sub(number(neural, "timestamp")) <= 15
    {
        return phase(neural);
    }
    if !completed(&run["auxiliary"])
        && number(&run["auxiliary"], "timestamp") > 0
        && now().saturating_sub(number(&run["auxiliary"], "timestamp")) <= 15
    {
        return phase(&run["auxiliary"]);
    }
    let pure = &run["pure_proposal_library"];
    if phase(pure) == "running"
        && number(pure, "timestamp") > 0
        && now().saturating_sub(number(pure, "timestamp")) <= 15
    {
        return "Saving pure policy predictions on eight GPUs";
    }
    let library = &run["proposal_library"];
    if phase(library) == "running"
        && number(library, "timestamp") > 0
        && now().saturating_sub(number(library, "timestamp")) <= 15
    {
        return "Building saved proposal seed library on eight GPUs";
    }
    let pipeline = &run["encoding_pipeline"];
    if !completed(pipeline)
        && number(pipeline, "timestamp") > 0
        && now().saturating_sub(number(pipeline, "timestamp")) <= 15
    {
        return phase(pipeline);
    }
    if !completed(&run["cycle"]) {
        if let Some(value) = run["cycle"]["phase"].as_str() {
            return value;
        }
    }
    if number(gp, "active") > 0 && now().saturating_sub(number(gp, "timestamp")) <= 5 {
        return "GP teaching corpus growing";
    }
    if run["contexts"]["completed"].is_number() {
        return "Encoded corpus ready · next learning cycle";
    }
    phase(run)
}
pub(crate) fn gpu_reason(run: &Value, neural: &Value, hardware: &Hardware) -> String {
    if hardware.gpus.len() != 8 {
        return "GPU telemetry incomplete; inspect Hardware for missing devices.".into();
    }
    if let Some(reason) = memory_throttle_reason(run) {
        if number(&run["campaign_resource_evidence"], "throttled_workers") > 0
            && (hardware.timestamp == 0
                || hardware.timestamp > now()
                || now().saturating_sub(hardware.timestamp) > 5)
        {
            return "GPU activity unknown: current host-memory contention observed, but GPU telemetry is stale.".into();
        }
        return reason;
    }
    let evidence = &run["experiment_worker_evidence"];
    if evidence.is_object() {
        if evidence["inventory_available"] != true
            || evidence["locks_available"] != true
            || now().saturating_sub(number(evidence, "timestamp")) > 5
        {
            return "GPU ownership unknown: worker or scheduler evidence is unavailable or stale."
                .into();
        }
        if number(evidence, "gpu_workers") > 0 {
            if evidence["experimental_gpu_owner"] != true {
                return "GPU ownership uncertain: experimental workers and scheduler lease disagree.".into();
            }
            return format!("{} experimental GPU ranks observed with the scheduler lease; production handoff takes priority.",number(evidence,"gpu_workers"));
        }
        for key in ["seed_experiment", "binary_qualification"] {
            let receipt = &run[key];
            if number(receipt, "gpu_world") > 0
                && !completed(receipt)
                && now().saturating_sub(number(receipt, "timestamp")) <= 15
            {
                return "GPU activity uncertain: a recent experiment receipt has no matching live GPU ranks.".into();
            }
        }
    }
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    if let Some(reason) = training_hold(run, policy) {
        return format!("GPU training held: {reason}; existing tasks require reconciliation.");
    }
    if !completed(neural) && now().saturating_sub(number(neural, "timestamp")) <= 5 {
        return format!("GPU assignment: {}", phase(neural));
    }
    for key in ["binary_qualification", "seed_experiment"] {
        let work = &run[key];
        if work.is_object()
            && number(work, "timestamp") > 0
            && now().saturating_sub(number(work, "timestamp")) <= 15
        {
            return format!(
                "Qualification {}: {} · CPUs 0–7, 62–69 shared by encoding and comparison",
                phase(work),
                work["reason"]
                    .as_str()
                    .unwrap_or("bounded spare-resource window")
            );
        }
    }
    if number(&run["gpu_wait"], "timestamp") > 0
        && now().saturating_sub(number(&run["gpu_wait"], "timestamp")) <= 5
    {
        return format!(
            "Queued GPU work: {} · waiting for current block to finish",
            phase(&run["gpu_wait"])
        );
    }
    let pipeline = &run["encoding_pipeline"];
    if pipeline["phase"].is_string() && !completed(pipeline) {
        return format!(
            "Next encoding work: {} · {} · {}",
            phase(pipeline),
            pipeline["reason"].as_str().unwrap_or("see Data / Events"),
            age(pipeline)
        );
    }
    if completed(&run["auxiliary"])
        && run["cycle"]["phase"] == "GP optimizes cut targets for the next teaching corpus"
    {
        return format!(
            "Auxiliary learning finished; CPU GP still evaluating frozen proposals. {}",
            run["auxiliary"]["reason"]
                .as_str()
                .unwrap_or("Next mandatory GPU block follows GP and corpus packing.")
        );
    }
    if run["cycle"]["phase"]
        .as_str()
        .is_some_and(|p| p.starts_with("Pack ") || p.starts_with("Fit and save"))
    {
        let cycle = &run["cycle"];
        if number(cycle, "timestamp") == 0
            || number(cycle, "timestamp") > now()
            || now().saturating_sub(number(cycle, "timestamp")) > 5
        {
            return format!(
                "GPU wait reason unconfirmed: stale {} receipt.",
                phase(cycle)
            );
        }
        return format!(
            "GPUs waiting for {}: {}",
            phase(cycle),
            stage_work_detail(cycle)
        );
    }
    let pure = &run["pure_proposal_library"];
    if pure.is_object() {
        return format!("Pure policy predictions {}: {} · {} / {} pools saved · separate scope, no GP queue writes · {}",phase(pure),pure["reason"].as_str().unwrap_or("saved training-context predictions await GP admission"),decimal_count(pure,"saved_seed_pools"),decimal_count(pure,"planned_seed_pools"),status_age_with_limit(pure,15));
    }
    let library = &run["proposal_library"];
    if library.is_object() {
        return format!(
            "Proposal library {}: {} · {} / {} pools saved · {}",
            phase(library),
            library["reason"]
                .as_str()
                .unwrap_or("seed generation precedes GP teaching scores"),
            decimal_count(library, "saved_seed_pools"),
            decimal_count(library, "planned_seed_pools"),
            status_age_with_limit(library, 15)
        );
    }
    if run["contexts"]["completed"].is_number() {
        if completed(&run["auxiliary"]) {
            return format!("Auxiliary training complete: {}. GP searches retain their frozen proposal checkpoint.",run["auxiliary"]["reason"].as_str().unwrap_or("bounded learning block saved"));
        }
        return "Encoding queue finished; vectors are saved. GPUs await the next training/proposal block.".into();
    }
    "GPU work receipt unavailable; consult Neural and recent events.".into()
}

pub(crate) fn stage_work_detail(cycle: &Value) -> String {
    if cycle["heartbeat_role"] == "controller_liveness" {
        let work = &cycle["work"];
        format!(
            "controller live; {} {} · work age {}",
            work["operation"]
                .as_str()
                .unwrap_or("awaiting work receipt"),
            work["completed"]
                .as_u64()
                .map(|n| n.to_string())
                .unwrap_or_default(),
            duration(now().saturating_sub(number(cycle, "last_progress_at")))
        )
    } else {
        cycle["reason"]
            .as_str()
            .unwrap_or("CPU preparation precedes the next training block")
            .into()
    }
}
pub(crate) fn milestone_lines(run: &Value, gp: &Value, neural: &Value) -> Vec<Line<'static>> {
    let contexts = &run["contexts"];
    let policy =
        if run["policy"]["corpus_sha256"].is_string() && run["policy"]["invalidated"] != true {
            &run["policy"]
        } else if run["fresh_policy"].is_object() {
            &run["fresh_policy"]
        } else {
            neural
        };
    let gp_total = number(gp, "completed_views")
        + number(gp, "failed_views")
        + gp["active"]
            .as_u64()
            .or_else(|| gp["last_recorded_activity"]["active"].as_u64())
            .unwrap_or(0)
        + number(gp, "queued");
    let cycle = &run["cycle"];
    let mut lines = vec![
        label(
            "1  Encodings on disk",
            format!(
                "{} views · {:.1}% · {} encoded families · {} dimensions · {}",
                count_pair(number(contexts, "completed"), number(contexts, "total")),
                100.0 * number(contexts, "completed") as f64
                    / number(contexts, "total").max(1) as f64,
                number(contexts, "families"),
                number(contexts, "dimensions"),
                if contexts["qualification"] == false {
                    "qualified"
                } else {
                    "qualification cohort"
                }
            ),
        ),
        label(
            "2  GP teaching targets",
            format!(
                "{} search jobs · {} evaluations · {} failed",
                count_pair(number(gp, "completed_views"), gp_total),
                number(gp, "completed_evaluations") + number(gp, "running_evaluations"),
                number(gp, "failed_views")
            ),
        ),
        label(
            "3  Policy checkpoint",
            if policy["invalidated"] == true {
                "previous run invalidated · corrected training pending".into()
            } else if policy["encoding"] == true || !policy["step"].is_number() {
                "awaiting training receipt".into()
            } else {
                format!(
                    "{} steps · {} ranks · {}",
                    count_pair(number(policy, "step"), number(policy, "steps")),
                    number(policy, "world"),
                    status_age(policy)
                )
            },
        ),
        label(
            "4  Neural → GP → DL",
            if run["continuous"].is_object() {
                let (status, detail) = campaign_task_state(run, neural);
                format!("{status} · {detail}")
            } else if cycle["complete"] == true {
                "completed · comparisons and fitted predictors saved".into()
            } else if cycle["phase"].is_string() {
                format!(
                    "{} · proposals {} · GP {} · retrain {}",
                    phase(cycle),
                    count_pair(
                        number(&cycle["proposals"], "completed"),
                        number(&cycle["proposals"], "total")
                    ),
                    count_pair(
                        number(&cycle["neural_seeded_gp"], "completed"),
                        number(&cycle["neural_seeded_gp"], "total")
                    ),
                    count_pair(
                        number(&cycle["retraining"], "step"),
                        number(&cycle["retraining"], "steps")
                    )
                )
            } else {
                "pending neural proposals, GP reevaluation, then retraining".into()
            },
        ),
    ];
    if run["encoding_pipeline"]["phase"].is_string() {
        let pipeline = &run["encoding_pipeline"];
        lines.push(label(
            "Next encoding cohort",
            format!(
                "{} · {} contexts · {}",
                phase(pipeline),
                count_pair(number(pipeline, "completed"), number(pipeline, "total")),
                status_age(pipeline)
            ),
        ));
    }
    if run["continuous"].is_object() {
        let (status, detail) = campaign_task_state(run, neural);
        lines.push(label("Continuous learning", format!("{status} · {detail}")));
    }
    if run["campaign_service"]["phase"].is_string() {
        let campaign = &run["campaign_service"];
        lines.push(label(
            if run["continuous"].is_object() {
                "Historical campaign"
            } else {
                "Managed campaign"
            },
            format!(
                "{} cycles complete · {}",
                number(campaign, "completed_cycles"),
                phase(campaign)
            ),
        ));
    }
    if let Some(checkpoint) = run["selected_checkpoint"]["checkpoint"].as_str() {
        lines.push(label(
            "Selected checkpoint",
            checkpoint.trim_start_matches("runs/"),
        ));
    }
    lines.push(label(
        "24-hour pilot",
        if run["pilot_started"] == true {
            "running · inspect pilot receipt"
        } else {
            "not started · qualification work; full pilot gates remain"
        },
    ));
    lines
}
