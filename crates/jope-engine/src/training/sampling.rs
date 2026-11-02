use super::*;

pub const SAMPLER_ID: &str = "family_uniform_champion_plus_seven_quantiles_v2";
pub const REPLAY_SAMPLER_ID: &str = "lineage_fresh_program_champion_plus_seven_v2";
pub const SEARCH_BEATING_SAMPLER_ID: &str = "search_beating_champion_plus_seven_v1";

pub fn active_sampler_id(policy: Option<&Value>) -> &'static str {
    match policy {
        Some(p)
            if p["schema"] == crate::checkpoint_recovery::CONTINUATION_SCHEMA
                && p.get("replay_policy_ref").is_none() =>
        {
            SAMPLER_ID
        }
        Some(p)
            if p["discoveries"] == "teacher_improved"
                || p["discoveries"] == "search_beating"
                || p["search_beating"] == true =>
        {
            SEARCH_BEATING_SAMPLER_ID
        }
        Some(_) => REPLAY_SAMPLER_ID,
        None => SAMPLER_ID,
    }
}
thread_local! {
    static RESIDENT_TENSORS: std::cell::RefCell<Option<(String,Vec<Tensor>)>> = const { std::cell::RefCell::new(None) };
    static RESIDENT_CONTEXT: std::cell::RefCell<Option<(String,Tensor)>> = const { std::cell::RefCell::new(None) };
}

pub fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}
/// Every rank-local comparison contains the best observed teacher and seven diverse comparators.
/// Records are sorted by increasing cut RMSE by the immutable corpus builder. Families receive equal mass.
pub fn sample(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
) -> Vec<usize> {
    assert!(count % 8 == 0 && serial % 8 == 0 && !families.is_empty());
    let families = families.values().collect::<Vec<_>>();
    let mut result = Vec::with_capacity(count);
    for g in serial / 8..(serial + count) / 8 {
        let cycle = g / families.len();
        let family = (g + mix(seed ^ cycle as u64) as usize) % families.len();
        let candidates = families[family];
        let h = mix(seed ^ g as u64);
        let group = &index.groups[candidates[h as usize % candidates.len()]];
        assert!(
            group.records.len() >= 8,
            "teaching comparison needs eight distinct programs"
        );
        result.push(group.records[0]);
        for band in 0..7 {
            let lo = 1 + band * (group.records.len() - 1) / 7;
            let hi = 1 + (band + 1) * (group.records.len() - 1) / 7;
            result.push(
                group.records
                    [lo + mix(h ^ ((band + 1) as u64).wrapping_mul(7919)) as usize % (hi - lo)],
            );
        }
    }
    result
}

/// Alternate equal-size fresh and historical portions, balancing families in each.
pub fn sample_replay(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
    balanced: bool,
) -> Vec<usize> {
    assert!(serial % 8 == 0 && count % 8 == 0);
    if !balanced || fresh.is_empty() {
        return sample(index, families, serial, count, seed);
    }
    let mut result = Vec::with_capacity(count);
    for comparison in serial / 8..(serial + count) / 8 {
        let portion = if comparison % 2 == 0 { fresh } else { families };
        result.extend(sample(index, portion, comparison / 2 * 8, 8, seed));
    }
    result
}

/// A designated fresh comparison must actually consume a newly admitted or
/// improved program. Preserve the champion and seven distinct comparators.
pub fn sample_replay_fresh(
    index: &Index,
    families: &BTreeMap<String, Vec<usize>>,
    fresh: &BTreeMap<String, Vec<usize>>,
    fresh_by_teacher: &BTreeMap<usize, Vec<usize>>,
    serial: usize,
    count: usize,
    seed: u64,
    balanced: bool,
) -> Vec<usize> {
    let mut result = sample_replay(index, families, fresh, serial, count, seed, balanced);
    if !balanced || fresh.is_empty() {
        return result;
    }
    for (n, ids) in result.chunks_mut(8).enumerate() {
        let comparison = serial / 8 + n;
        if comparison % 2 != 0 {
            continue;
        }
        let eligible = &fresh_by_teacher[&ids[0]];
        assert!(!eligible.is_empty(), "fresh group needs a fresh program");
        if !ids.iter().any(|id| eligible.contains(id)) {
            let fresh_id =
                eligible[mix(seed ^ comparison as u64 ^ 0x6672657368) as usize % eligible.len()];
            // Replace the closest sampled score band, keeping the champion.
            let slot = (1..8)
                .min_by_key(|&slot| ids[slot].abs_diff(fresh_id))
                .unwrap();
            ids[slot] = fresh_id;
        }
    }
    result
}
pub(super) fn validation_families(index: &Index) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    for (i, group) in index.groups.iter().enumerate() {
        if group.partition == "meta_validation" {
            ensure!(
                group.records.len() >= 8,
                "validation group lacks comparators"
            );
            families.entry(group.family.clone()).or_default().push(i);
        }
    }
    Ok(families)
}

/// Fixed global validation panel, identical on every evaluation of a corpus
/// (partition v1: 256 rows per rank).
pub fn validation_panel(index: &Index, world: usize, seed: u64) -> Result<Vec<Vec<usize>>> {
    let families = validation_families(index)?;
    ensure!(
        !families.is_empty() && [1, 2, 4, 8].contains(&world),
        "validation panel requires families and a supported world size"
    );
    Ok((0..world)
        .map(|rank| sample(index, &families, rank * 256, 256, seed ^ 0x76616c))
        .collect())
}

/// The fixed validation panel of one attempt.
pub(super) enum ValidationPanel {
    /// Partition v1: 256 rows per rank.
    V1(Vec<Vec<usize>>),
    /// Partition v2: every rank's microbatches of the fixed global panel,
    /// the panel in global order, and its global denominators.
    V2 {
        per_rank: Vec<Vec<Vec<usize>>>,
        global_ids: Vec<usize>,
        denominators: partition::Denominators,
    },
}

/// Partition v2 (W7-T06): one fixed global panel of
/// `partition::VALIDATION_PANEL_ROWS` rows, whatever the rank count, split
/// by `plan` into each rank's microbatches (`[rank][microbatch]` record ids).
/// Its rows are those of the v1 eight-rank panel, so the panel and the
/// globally normalized validation objective do not depend on the rank count.
pub fn validation_panel_partitioned(
    index: &Index,
    plan: &partition::Plan,
    seed: u64,
) -> Result<Vec<Vec<Vec<usize>>>> {
    ensure!(
        plan.global_batch() == partition::VALIDATION_PANEL_ROWS,
        "the validation panel has {} rows",
        partition::VALIDATION_PANEL_ROWS
    );
    let families = validation_families(index)?;
    ensure!(
        !families.is_empty(),
        "validation panel requires families and a supported world size"
    );
    (0..plan.world())
        .map(|rank| {
            Ok(plan
                .slots(rank, seed, 0)?
                .iter()
                .map(|slot| {
                    sample(
                        index,
                        &families,
                        slot.serial as usize,
                        slot.rows,
                        seed ^ 0x76616c,
                    )
                })
                .collect())
        })
        .collect()
}
