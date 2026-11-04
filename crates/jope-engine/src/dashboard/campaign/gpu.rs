use super::*;

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
