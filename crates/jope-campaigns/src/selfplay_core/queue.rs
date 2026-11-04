use super::*;

/// The checkpoint-fresh queue policy and observed p95.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QueueTargets {
    pub target: i64,
    pub hard_cap: i64,
    pub p95_jobs_per_minute: f64,
}

/// Calculate the nearest-rank p95 and the work-conserving queue horizon.
pub fn dynamic_queue_targets(
    jobs_per_minute: &[f64],
    effective_cluster_slots: i64,
) -> PolicyResult<QueueTargets> {
    if effective_cluster_slots < 0 {
        return Err(error("effective cluster slots cannot be negative"));
    }
    let mut values = Vec::with_capacity(jobs_per_minute.len());
    for &value in jobs_per_minute {
        values.push(validate_finite(value, "jobs per minute")?.max(0.0));
    }
    values.sort_by(f64::total_cmp);
    let p95 = if values.is_empty() {
        0.0
    } else {
        let rank = ((0.95 * values.len() as f64).ceil() as usize).max(1);
        values[rank - 1]
    };
    let slot_horizon = ((i128::from(effective_cluster_slots) * 5 + 3) / 4).min(4_096) as i64;
    Ok(QueueTargets {
        target: slot_horizon.max(2_048),
        hard_cap: 4_096,
        p95_jobs_per_minute: p95,
    })
}

pub fn queue_refill_needed(queued_jobs: i64, target: i64, hard_cap: i64) -> PolicyResult<i64> {
    if target < 0 || hard_cap < target {
        return Err(error(
            "queue hard cap must be at least the nonnegative queue target",
        ));
    }
    if queued_jobs >= target {
        return Ok(0);
    }
    Ok((target - queued_jobs).min(hard_cap - queued_jobs))
}

/// Assign complete source families to stable train/validation splits.
pub fn family_disjoint_split(
    dataset_families: &BTreeMap<String, String>,
    validation_fraction: f64,
    seed: u64,
) -> PolicyResult<BTreeMap<String, String>> {
    if !(0.0..1.0).contains(&validation_fraction) || validation_fraction == 0.0 {
        return Err(error("validation_fraction must be between zero and one"));
    }
    let mut families = BTreeMap::<String, Vec<String>>::new();
    for (dataset, family) in dataset_families {
        if dataset.is_empty() || family.is_empty() {
            return Err(error("dataset and family must be nonempty"));
        }
        families
            .entry(family.clone())
            .or_default()
            .push(dataset.clone());
    }
    if dataset_families.len() < 2 || families.is_empty() {
        return Err(error(
            "family topology cannot produce nonempty disjoint splits",
        ));
    }
    let mut ordered = families.keys().cloned().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        family_order_digest(seed, left)
            .cmp(&family_order_digest(seed, right))
            .then_with(|| left.cmp(right))
    });
    let rounded = (dataset_families.len() as f64 * validation_fraction).round_ties_even() as usize;
    let target = (dataset_families.len() - 1).min(rounded.max(1));
    let mut validation = BTreeSet::new();
    let mut remaining = target;
    for family in ordered {
        let mut members = families[&family].clone();
        members.sort();
        if remaining == 0 {
            break;
        }
        if members.len() <= remaining || validation.is_empty() {
            remaining = remaining.saturating_sub(members.len());
            validation.extend(members);
        }
    }
    if validation.is_empty() || validation.len() == dataset_families.len() {
        return Err(error(
            "family topology cannot produce nonempty disjoint splits",
        ));
    }
    Ok(dataset_families
        .keys()
        .map(|dataset| {
            let split = if validation.contains(dataset) {
                "validation"
            } else {
                "train"
            };
            (dataset.clone(), split.to_owned())
        })
        .collect())
}

pub(crate) fn family_order_digest(seed: u64, family: &str) -> [u8; 32] {
    sha256_bytes(format!("{seed}|{family}").as_bytes())
}
