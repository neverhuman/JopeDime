use super::*;

pub(super) fn parse_campaign(value: Option<&Value>) -> CampaignSummary {
    let value = value.unwrap_or(&Value::Null);
    let rung_counts = value.get("rung_counts").unwrap_or(&Value::Null);
    let active_progress = value
        .get("active_jobs")
        .and_then(Value::as_array)
        .and_then(|jobs| jobs.first())
        .and_then(|job| u64_at(job, &["progress_evaluations"]));
    let current_rung = active_progress
        .map(|progress| match progress {
            0..=999 => "1000",
            1_000..=1_999 => "2000",
            2_000..=4_999 => "5000",
            _ => "10000",
        })
        .or_else(|| {
            ["10000", "5000", "2000", "1000"]
                .into_iter()
                .find(|rung| u64_at(rung_counts, &[*rung]).unwrap_or(0) > 0)
        })
        .unwrap_or("-")
        .to_owned();
    CampaignSummary {
        stage: string_at(value, &["stage"])
            .unwrap_or("unavailable")
            .to_owned(),
        gate_state: string_at(value, &["gate_state"])
            .unwrap_or("launch_gates_blocked")
            .to_owned(),
        robust_wins: u64_at(value, &["robust_wins"]).unwrap_or(0),
        robust_target: u64_at(value, &["robust_wins_target"]).unwrap_or(7),
        robust_denominator: u64_at(value, &["robust_wins_denominator"]).unwrap_or(22),
        trajectories_complete: u64_at(value, &["completed_trajectory_count"]).unwrap_or(0),
        trajectories_total: u64_at(value, &["trajectory_count"]).unwrap_or(0),
        evaluations_used: u64_at(value, &["evaluations_used"]).unwrap_or(0),
        evaluations_cap: u64_at(value, &["evaluations_cap"]).unwrap_or(200_000),
        current_rung,
        seed_agreement: string_at(value, &["seed_agreement"])
            .unwrap_or("unavailable")
            .to_owned(),
        active_jobs: value
            .get("active_jobs")
            .and_then(Value::as_array)
            .map_or(0, |jobs| jobs.len() as u64),
        license_track: string_at(value, &["license_track"])
            .unwrap_or("separate")
            .to_owned(),
        test_opening_state: string_at(value, &["test_opening_state"])
            .unwrap_or("sealed")
            .to_owned(),
        last_advanced_at_ms: u64_at(value, &["last_advanced_at_ms"])
            .map(u128::from)
            .unwrap_or(0),
        blockers: value
            .get("blockers")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect(),
    }
}

pub(super) fn infrastructure_attempt_number(job_identity: &str) -> u64 {
    job_identity
        .rsplit_once("-infra")
        .and_then(|(_, suffix)| suffix.parse::<u64>().ok())
        .unwrap_or(1)
}

pub(super) fn enrich_gpu_admission(gpus: &mut [GpuRow], value: Option<&Value>) {
    let Some(states) = value
        .and_then(|value| value.get("states"))
        .and_then(Value::as_object)
    else {
        return;
    };
    for (role, state) in states {
        let Some(node) = string_at(state, &["target_node"]) else {
            continue;
        };
        let job_identity = string_at(state, &["immutable_job_identity"]).unwrap_or("-");
        let stage = string_at(state, &["stage"]).unwrap_or("unknown");
        let reason = string_at(state, &["dependency_failure"])
            .or_else(|| string_at(state, &["last_capacity_backoff"]))
            .unwrap_or(stage);
        if let Some(gpu) = gpus.iter_mut().find(|gpu| gpu.node == node) {
            gpu.role = role.clone();
            gpu.attempt = infrastructure_attempt_number(job_identity);
            gpu.admission_reason = reason.to_owned();
        }
    }
}
