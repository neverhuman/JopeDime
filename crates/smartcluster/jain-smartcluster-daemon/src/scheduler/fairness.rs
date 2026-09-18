use std::cmp::Ordering;
use std::collections::BTreeMap;

use jain_smartcluster_core::{TenantId, Timestamp};

use super::{ResourceVector, SchedulerConfig, SchedulingCandidate};

pub(super) fn compare_candidates(
    left: &SchedulingCandidate,
    right: &SchedulingCandidate,
    usage: &BTreeMap<TenantId, ResourceVector>,
    capacity: ResourceVector,
    weights: &BTreeMap<TenantId, u32>,
    now: Timestamp,
    config: SchedulerConfig,
) -> Ordering {
    left.job
        .spec
        .priority
        .rank()
        .cmp(&right.job.spec.priority.rank())
        .then_with(|| {
            fairness_score(left, usage, capacity, weights, now, config).cmp(&fairness_score(
                right, usage, capacity, weights, now, config,
            ))
        })
        .then(left.job.created_at.cmp(&right.job.created_at))
        .then(left.job.id.cmp(&right.job.id))
}

fn fairness_score(
    candidate: &SchedulingCandidate,
    usage: &BTreeMap<TenantId, ResourceVector>,
    capacity: ResourceVector,
    weights: &BTreeMap<TenantId, u32>,
    now: Timestamp,
    config: SchedulerConfig,
) -> i128 {
    let used = usage
        .get(&candidate.job.spec.tenant)
        .copied()
        .unwrap_or_default();
    let weight = u128::from(
        weights
            .get(&candidate.job.spec.tenant)
            .copied()
            .unwrap_or(1_000)
            .max(1),
    );
    let shares = [
        share_micros(used.cpu_millicores, capacity.cpu_millicores, weight),
        share_micros(used.memory_bytes, capacity.memory_bytes, weight),
        share_micros(
            used.gpu_compute_per_mille,
            capacity.gpu_compute_per_mille,
            weight,
        ),
        share_micros(used.gpu_vram_bytes, capacity.gpu_vram_bytes, weight),
    ];
    let dominant = shares.into_iter().max().unwrap_or(0);
    let waited = now
        .as_unix_millis()
        .saturating_sub(candidate.job.created_at.as_unix_millis());
    let age_intervals = waited / config.aging_interval_ms;
    let age_credit = age_intervals.saturating_mul(config.aging_credit_micros);
    i128::from(dominant) - i128::from(age_credit)
}

fn share_micros(used: u64, capacity: u64, weight: u128) -> u64 {
    if capacity == 0 {
        return if used == 0 { 0 } else { u64::MAX };
    }
    let scaled = u128::from(used)
        .saturating_mul(1_000_000)
        .saturating_mul(1_000)
        / u128::from(capacity)
        / weight;
    u64::try_from(scaled).unwrap_or(u64::MAX)
}
