use super::*;

pub(crate) fn rank_slots(left: &LogicalSlot, right: &LogicalSlot) -> Ordering {
    let left_score = left.score.as_ref().expect("ranked slot has score");
    let right_score = right.score.as_ref().expect("ranked slot has score");
    right_score
        .auc
        .total_cmp(&left_score.auc)
        .then_with(|| left_score.logloss.total_cmp(&right_score.logloss))
        .then_with(|| left.member.complexity.cmp(&right.member.complexity))
        .then_with(|| {
            left.member
                .semantic_sha256
                .cmp(&right.member.semantic_sha256)
        })
}

pub(crate) fn strict_win(
    candidate_auc: f64,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> bool {
    candidate_auc > leader_auc && candidate_auc > generation_zero_best_auc
}

pub(crate) fn policy_win(
    win_policy: &str,
    candidate_auc: f64,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> bool {
    match win_policy {
        "strictly_greater_auc" => strict_win(candidate_auc, leader_auc, generation_zero_best_auc),
        "exact_auc_1_tie" => candidate_auc.to_bits() == 1.0f64.to_bits(),
        _ => false,
    }
}

pub(crate) fn physical_hostname(node: &str) -> Result<&'static str> {
    NODES
        .iter()
        .position(|candidate| *candidate == node)
        .map(|index| PHYSICAL_HOSTNAMES[index])
        .with_context(|| format!("unknown Binary42 SCQ node {node}"))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn cohort_win(candidate_auc: f64, leader_auc: f64) -> bool {
    if leader_auc == 1.0 {
        candidate_auc == 1.0
    } else {
        candidate_auc > leader_auc
    }
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn qualifying_winner(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    qualifying_winner_for_policy(
        generation,
        ranked,
        seed_semantics,
        "strictly_greater_auc",
        leader_auc,
        generation_zero_best_auc,
    )
}

pub(crate) fn qualifying_winner_for_policy(
    generation: usize,
    ranked: &[LogicalSlot],
    seed_semantics: &BTreeSet<String>,
    win_policy: &str,
    leader_auc: f64,
    generation_zero_best_auc: f64,
) -> Option<QualifyingWinner> {
    if generation == 0 {
        return None;
    }
    ranked.iter().find_map(|slot| {
        let score = slot.score.as_ref()?;
        (!seed_semantics.contains(&slot.member.semantic_sha256)
            && matches!(
                slot.member.ancestry.operation.as_str(),
                "mutation" | "crossover" | "elite"
            )
            && policy_win(win_policy, score.auc, leader_auc, generation_zero_best_auc))
        .then(|| QualifyingWinner {
            generation,
            slot: slot.slot,
            semantic_sha256: slot.member.semantic_sha256.clone(),
            auc: score.auc,
            logloss: score.logloss,
            complexity: slot.member.complexity,
        })
    })
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn shard_members(members: &[MemberRecord]) -> Result<Vec<Vec<MemberRecord>>> {
    shard_members_with_sizes(members, &[SHARD_SIZE; 4])
}

pub(crate) fn shard_members_with_sizes(
    members: &[MemberRecord],
    shard_sizes: &[usize],
) -> Result<Vec<Vec<MemberRecord>>> {
    if members.len() != POPULATION {
        bail!("Binary42 generation requires exactly {POPULATION} members");
    }
    if shard_sizes.len() != NODES.len() || shard_sizes.iter().sum::<usize>() != POPULATION {
        bail!("Binary42 shard sizes do not account for the 60-member population");
    }
    let mut offset = 0;
    let mut shards = Vec::with_capacity(shard_sizes.len());
    for size in shard_sizes {
        shards.push(members[offset..offset + size].to_vec());
        offset += size;
    }
    Ok(shards)
}

pub(crate) fn campaign_shard_sizes(config: &CampaignConfig) -> Vec<usize> {
    if config.shard_sizes.is_empty() {
        vec![SHARD_SIZE; NODES.len()]
    } else {
        config.shard_sizes.clone()
    }
}

pub(crate) fn shard_start(shard_sizes: &[usize], shard: usize) -> Result<usize> {
    if shard >= shard_sizes.len() {
        bail!("Binary42 shard index is outside the configured partition");
    }
    Ok(shard_sizes[..shard].iter().sum())
}

pub(crate) fn assigned_node_for_slot(shard_sizes: &[usize], slot: usize) -> Result<&'static str> {
    let mut end = 0;
    for (index, size) in shard_sizes.iter().enumerate() {
        end += size;
        if slot < end {
            return Ok(NODES[index]);
        }
    }
    bail!("Binary42 slot {slot} is outside the configured shard partition")
}

pub(crate) fn generation_cap_reached(config: &CampaignConfig, generation: usize) -> bool {
    config
        .generation_cap
        .is_some_and(|cap| generation + 1 >= cap)
}

pub(crate) fn failover_node(shard: usize, attempt: usize) -> Result<&'static str> {
    if shard >= NODES.len() || attempt > 2 {
        bail!("Binary42 failover index is outside the immutable policy");
    }
    Ok(NODES[(shard + attempt) % NODES.len()])
}

pub(crate) fn binary_prediction_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:.17}\n").as_bytes());
    }
    bytes
}

pub(crate) fn binary_prediction_roundtrip_bytes(predictions: &[f64]) -> Vec<u8> {
    let mut bytes = b"prediction\n".to_vec();
    for prediction in predictions {
        bytes.extend_from_slice(format!("{prediction:?}\n").as_bytes());
    }
    bytes
}

pub(crate) fn parse_prediction_bytes(bytes: &[u8]) -> Result<Vec<f64>> {
    let text = std::str::from_utf8(bytes).context("prediction object is not UTF-8")?;
    let mut lines = text.lines();
    if lines.next() != Some("prediction") {
        bail!("prediction object has the wrong header");
    }
    lines
        .enumerate()
        .map(|(index, line)| {
            let value = line
                .parse::<f64>()
                .with_context(|| format!("prediction row {} is not numeric", index + 1))?;
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                bail!("prediction row {} is outside [0, 1]", index + 1);
            }
            Ok(value)
        })
        .collect()
}

pub(crate) fn candidate_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.receipt.v1.json")
}

pub(crate) fn prediction_file(generation: usize, slot: usize) -> String {
    format!("candidate-g{generation:06}-s{slot:02}.predictions.csv")
}
