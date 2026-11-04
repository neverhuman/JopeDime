use super::state::*;
use super::*;

pub(crate) fn workflow_rows(run: &Value, gp: &Value, neural: &Value) -> Vec<TaskRow> {
    let contexts = &run["contexts"];
    let boot = &run["bootstrap"];
    let cycle = &run["cycle"];
    let policy = if run["policy"].is_object() {
        &run["policy"]
    } else {
        neural
    };
    let live = receipt_task_state;
    let (policy_status, policy_detail) = policy_task_state(run, policy);
    let (campaign_status, campaign_detail) = campaign_task_state(run, neural);
    let mut rows = vec![
        task_row(
            "Saved contexts",
            if number(contexts, "completed") > 0
                && number(contexts, "completed") == number(contexts, "total")
            {
                "DONE"
            } else {
                live(contexts)
            },
            count_pair(number(contexts, "completed"), number(contexts, "total")),
            "2048D · source pointers · frozen encoders".into(),
        ),
        task_row(
            "GP teacher search jobs",
            if gp["current_activity"] == "stale" {
                "STALE"
            } else if gp["current_activity"] == "terminal" {
                "STOPPED"
            } else if gp["phase"] == "waiting" {
                "WAITING"
            } else if gp["phase"] == "completed" {
                "EXHAUSTED"
            } else if gp["phase"] == "paused" {
                "PAUSED"
            } else if number(gp, "active") > 0 {
                live(gp)
            } else {
                "QUEUED"
            },
            format!(
                "{} done / {} active",
                number(gp, "completed_views"),
                decimal_count(gp, "active")
            ),
            format!(
                "{} queued · {} eval/s",
                number(gp, "queued"),
                decimal(gp, "evaluations_per_second")
            ),
        ),
        task_row(
            "64-family bootstrap",
            if boot["compact_receipt_ready"] == true {
                "DONE"
            } else {
                "QUEUED"
            },
            format!(
                "{} train · {} validation",
                number(
                    &boot["partitions"]["meta_train"],
                    "families_with_completed_view"
                ),
                number(
                    &boot["partitions"]["meta_validation"],
                    "families_with_completed_view"
                )
            ),
            "Finite reward-group admission remains separate".into(),
        ),
        task_row(
            "Teaching corpus",
            live(&run["corpus_progress"]),
            format!(
                "{} records · {} / {} families",
                number(&run["corpus_progress"], "records"),
                number(&run["corpus_progress"], "completed"),
                number(&run["corpus_progress"], "total")
            ),
            run["corpus_progress"]["reason"]
                .as_str()
                .unwrap_or("Awaiting finite reward-group admission receipt")
                .into(),
        ),
        if run["continuous"].is_object() {
            task_row(
                "Self-play runtime",
                campaign_status,
                "Continuous supervisor".into(),
                campaign_detail.clone(),
            )
        } else {
            task_row(
                "Self-play runtime",
                live(&run["runtime_readiness"]),
                run["runtime_readiness"]["phase"]
                    .as_str()
                    .unwrap_or("verification pending")
                    .into(),
                run["runtime_readiness"]["reason"]
                    .as_str()
                    .unwrap_or("A complete corpus does not imply the next campaign has launched")
                    .into(),
            )
        },
        task_row(
            "Policy optimization",
            policy_status,
            count_pair(number(policy, "step"), number(policy, "steps")),
            policy_detail,
        ),
        task_row(
            "Auxiliary learning",
            live(&run["auxiliary"]),
            format!(
                "{} / {} steps",
                run["auxiliary"]["completed_optimizer_steps"]
                    .as_u64()
                    .or_else(|| run["auxiliary"]["step"].as_u64())
                    .map(|n| n.to_string())
                    .unwrap_or_else(|| "—".into()),
                decimal_count(&run["auxiliary"], "maximum_steps")
            ),
            run["auxiliary"]["reason"]
                .as_str()
                .or_else(|| run["auxiliary"]["phase"].as_str())
                .unwrap_or("Bounded learning during GP wait; proposal checkpoint stays frozen")
                .into(),
        ),
        task_row(
            "Neural proposals",
            if number(&cycle["proposals"], "total") > 0
                && number(&cycle["proposals"], "completed") == number(&cycle["proposals"], "total")
            {
                "DONE"
            } else {
                "QUEUED"
            },
            count_pair(
                number(&cycle["proposals"], "completed"),
                number(&cycle["proposals"], "total"),
            ),
            "Real context + shuffled context comparison".into(),
        ),
        task_row(
            "DL → GP → DL",
            if cycle["complete"] == true {
                "DONE"
            } else {
                live(cycle)
            },
            format!(
                "{} completed cycles",
                run["reports"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|v| v["raw_final_scores_only"] != true || v["complete"] == true)
                    .count()
            ),
            phase(cycle).into(),
        ),
        task_row(
            "Managed campaign",
            campaign_status,
            format!(
                "{} cycles saved",
                number(&run["campaign_service"], "completed_cycles")
            ),
            campaign_detail,
        ),
    ];
    // The report watcher that wrote this receipt was archived (see `campaign`), so a missing
    // receipt is not a pending one: show the row only for a saved receipt.
    let report = &run["report_progress"];
    if report.is_object() {
        rows.push(task_row(
            "Matched-budget reports",
            report_status(report),
            if phase(report).starts_with("Waiting") && number(report, "total") == 0 {
                "0 completed cycles".into()
            } else {
                count_pair(number(report, "completed"), number(report, "total"))
            },
            report["reason"]
                .as_str()
                .or_else(|| report["phase"].as_str())
                .unwrap_or("Completed-cycle reporting receipt pending")
                .into(),
        ));
    }
    let mut operations_rows = Vec::new();
    if run["operations"].is_object() {
        let services = &run["operations"]["services"];
        let progress = if run["operations"]["gp_units"]
            .as_array()
            .is_some_and(|v| !v.is_empty())
        {
            format!(
                "GP {} · DL {}",
                decimal_count(gp, "active"),
                services["jope-parallel-campaign"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown")
            )
        } else {
            format!(
                "Managed DL {} · GP {}",
                services["jope-parallel-campaign"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown"),
                services["jope-parallel-gp"]["ActiveState"]
                    .as_str()
                    .unwrap_or("unknown")
            )
        };
        operations_rows.push(task_row(
            "Actual services",
            "OBSERVED",
            progress,
            format!(
                "GP units: {} · GPU units: {} · owners: {}",
                run["operations"]["gp_units"],
                run["operations"]["gpu_units"],
                run["operations"]["gpu_owners"]
            ),
        ));
    }
    let continuous = &run["continuous"];
    if continuous.is_object() {
        operations_rows.push(task_row(
            "GPU inactivity",
            continuous["inactivity"].as_str().unwrap_or("OBSERVED"),
            format!(
                "{} idle · {} GPU-s paid",
                duration(number(continuous, "gpu_idle_seconds")),
                number(continuous, "paid_gpu_idle_seconds")
            ),
            continuous["idle_explanation"]
                .as_str()
                .unwrap_or("No active packing explanation")
                .into(),
        ));
        operations_rows.push(task_row(
            "Last committed model",
            "SAVED",
            format!("{} blocks", number(continuous, "completed_training_blocks")),
            continuous["last_committed_checkpoint"]
                .as_str()
                .unwrap_or("unavailable")
                .into(),
        ));
        let pack = &run["continuous_packing"];
        operations_rows.push(task_row(
            "Snapshot packing",
            if !pack.is_object() {
                "WAITING"
            } else if pack["complete"] == true {
                "DONE"
            } else if now().saturating_sub(number(pack, "timestamp")) > 15 {
                "STALE"
            } else {
                "RUNNING"
            },
            format!("{}", pack["work"]["completed"]),
            format!("{}", pack["work"]["operation"]),
        ));
        let monitor = &run["frozen_monitoring"];
        if monitor.is_object() {
            operations_rows.push(task_row(
                "Frozen validation",
                "SAVED",
                format!(
                    "NLL {} · BT {}",
                    decimal(monitor, "sequence_nll"),
                    decimal(monitor, "ranking_bt")
                ),
                format!(
                    "{} examples · panel {} · adaptive validation; separate from training loss",
                    number(monitor, "sampled_examples"),
                    monitor["binding"]["panel_sha256"]
                        .as_str()
                        .unwrap_or("unknown")
                ),
            ));
        }
    }
    let library = &run["proposal_library"];
    if library.is_object() {
        let admission = &run["proposal_library_admission"];
        let observed = &run["proposal_library_gp"];
        let detail = if admission["committed"] == true {
            format!(
                "{} ranked · GP: {} admitted / {} observed evals · {} / {} outputs checked",
                decimal_count(library, "saved_ranked_programs"),
                decimal_count(admission, "requested_jobs"),
                decimal_count(observed, "observed_evaluations"),
                decimal_count(observed, "checked_jobs"),
                decimal_count(admission, "requested_jobs")
            )
        } else if admission.is_object() {
            format!(
                "{} ranked · GP admission {} · 0 admitted / 0 evaluations",
                decimal_count(library, "saved_ranked_programs"),
                phase(admission)
            )
        } else {
            format!(
                "{} ranked programs · GP: {} jobs admitted / {} evaluations · seeds only",
                decimal_count(library, "saved_ranked_programs"),
                decimal_count(library, "gp_jobs_admitted"),
                decimal_count(library, "gp_evaluations")
            )
        };
        rows.insert(
            8,
            task_row(
                "Proposal seed library",
                proposal_library_status(library),
                format!(
                    "{} / {} pools",
                    decimal_count(library, "saved_seed_pools"),
                    decimal_count(library, "planned_seed_pools")
                ),
                detail,
            ),
        );
    }
    let pure = &run["pure_proposal_library"];
    if pure.is_object() {
        let admission = if pure["gp_jobs_admitted"] == 0 {
            "unadmitted".into()
        } else {
            format!("{} admitted", decimal_count(pure, "gp_jobs_admitted"))
        };
        let detail = format!(
            "{} ranked · {} / {} GP evals · separate pure scope",
            decimal_count(pure, "saved_ranked_programs"),
            admission,
            decimal_count(pure, "gp_evaluations")
        );
        rows.insert(
            8 + usize::from(library.is_object()),
            task_row(
                "Pure policy predictions",
                proposal_library_status(pure),
                format!(
                    "{} / {} pools",
                    decimal_count(pure, "saved_seed_pools"),
                    decimal_count(pure, "planned_seed_pools")
                ),
                detail,
            ),
        );
    }
    if run["active_cycle"].is_string() {
        let exports = &run["predictor_exports"];
        rows.push(task_row(
            "Predictor export",
            if number(exports, "total") > 0
                && number(exports, "completed") == number(exports, "total")
            {
                "DONE"
            } else if phase(cycle).starts_with("Fit and save") {
                live(cycle)
            } else {
                "QUEUED"
            },
            count_pair(number(exports, "completed"), number(exports, "total")),
            if phase(cycle).starts_with("Fit and save") {
                stage_work_detail(cycle)
            } else {
                "Fitted predictors · 30-second / 8 GiB limits".into()
            },
        ));
        rows.push(task_row(
            "Corpus packing",
            if phase(cycle).starts_with("Pack ") {
                live(cycle)
            } else {
                "WAITING"
            },
            format!(
                "{} current records",
                decimal_count(&run["corpus_progress"], "records")
            ),
            if phase(cycle).starts_with("Pack ") {
                stage_work_detail(cycle)
            } else {
                "New scores packed before retraining; exact program labels".into()
            },
        ));
    }
    for (key, title, count) in [
        ("seed_experiment", "Seed quality experiment", "completed"),
        ("binary_qualification", "Binary qualification", "completed"),
    ] {
        let v = &run[key];
        if v.is_object() {
            let (progress, detail) =
                if key == "binary_qualification" && run["binary_admitted_contexts"].is_number() {
                    (
                        format!(
                            "{} admitted · {} replayed",
                            decimal_count(run, "binary_admitted_contexts"),
                            decimal_count(&run["binary_verification"], "verified_pairs")
                        ),
                        binary_replay_detail(run),
                    )
                } else {
                    (
                        count_pair(number(v, count), number(v, "total")),
                        format!(
                            "{} · {}",
                            phase(v),
                            v["reason"].as_str().unwrap_or("qualification results only")
                        ),
                    )
                };
            rows.push(task_row(
                title,
                proposal_library_status(v),
                progress,
                detail,
            ));
        }
    }
    if run["hyperion_recovery"]["passed"] == true {
        rows.push(task_row(
            "CUDA recovery",
            "DONE",
            format!(
                "{} comparisons",
                number(&run["hyperion_recovery"], "shared_input_comparisons")
            ),
            "Smaller batches passed · previous vectors preserved".into(),
        ));
    }
    operations_rows.extend(rows);
    operations_rows
}
