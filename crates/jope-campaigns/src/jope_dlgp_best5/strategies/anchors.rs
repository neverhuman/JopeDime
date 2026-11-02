use super::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnchorObservationV1 {
    pub observation_snapshot_sha256: String,
    pub dataset_sha256: String,
    pub source_family: String,
    pub genome: GenomeV1,
    pub within_dataset_percentile: f64,
    pub strict_win: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RankedAnchorV1 {
    pub genome: GenomeV1,
    pub support_shrunk_percentile_lcb: f64,
    pub worst_percentile: f64,
    pub dataset_support: u32,
    pub source_family_support: u32,
    pub strict_wins: u32,
}

pub fn rank_anchor_observations(
    target_dataset_sha256: &str,
    target_source_family: &str,
    observation_snapshot_sha256: &str,
    observations: &[AnchorObservationV1],
) -> Result<Vec<RankedAnchorV1>, String> {
    validate_sha256("target dataset", target_dataset_sha256)?;
    validate_sha256("observation snapshot", observation_snapshot_sha256)?;
    if target_source_family.trim().is_empty() {
        return Err("target source family must not be empty".into());
    }
    let mut grouped = BTreeMap::<String, Vec<&AnchorObservationV1>>::new();
    for observation in observations {
        validate_sha256(
            "anchor observation snapshot",
            &observation.observation_snapshot_sha256,
        )?;
        validate_sha256("anchor observation dataset", &observation.dataset_sha256)?;
        observation.genome.validate()?;
        if observation.source_family.trim().is_empty()
            || !observation.within_dataset_percentile.is_finite()
            || !(0.0..=1.0).contains(&observation.within_dataset_percentile)
        {
            return Err("anchor observation fields are invalid".into());
        }
        if observation.observation_snapshot_sha256 != observation_snapshot_sha256 {
            return Err("anchor observation comes from a stale snapshot".into());
        }
        if observation.dataset_sha256 == target_dataset_sha256
            || observation.source_family == target_source_family
        {
            continue;
        }
        grouped
            .entry(observation.genome.semantic_sha256.clone())
            .or_default()
            .push(observation);
    }
    let mut ranked = Vec::new();
    for group in grouped.into_values() {
        let mut per_dataset = BTreeMap::<&str, f64>::new();
        let mut families = BTreeSet::new();
        let mut strict_wins = BTreeSet::new();
        for observation in &group {
            per_dataset
                .entry(&observation.dataset_sha256)
                .and_modify(|percentile| {
                    *percentile = percentile.min(observation.within_dataset_percentile)
                })
                .or_insert(observation.within_dataset_percentile);
            families.insert(observation.source_family.as_str());
            if observation.strict_win {
                strict_wins.insert(observation.dataset_sha256.as_str());
            }
        }
        if per_dataset.len() < 3 || families.len() < 2 {
            continue;
        }
        let percentiles = per_dataset.values().copied().collect::<Vec<_>>();
        ranked.push(RankedAnchorV1 {
            genome: group[0].genome.clone(),
            support_shrunk_percentile_lcb: confidence_score(&percentiles)?,
            worst_percentile: percentiles.iter().copied().min_by(f64::total_cmp).unwrap(),
            dataset_support: per_dataset.len() as u32,
            source_family_support: families.len() as u32,
            strict_wins: strict_wins.len() as u32,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .support_shrunk_percentile_lcb
            .total_cmp(&left.support_shrunk_percentile_lcb)
            .then_with(|| right.worst_percentile.total_cmp(&left.worst_percentile))
            .then_with(|| right.dataset_support.cmp(&left.dataset_support))
            .then_with(|| right.strict_wins.cmp(&left.strict_wins))
            .then_with(|| {
                left.genome
                    .semantic_sha256
                    .cmp(&right.genome.semantic_sha256)
            })
    });
    Ok(ranked)
}

/// Builds the exact 64 proposal / 10 anchor / 6 seed population.  Duplicate
/// anchors and seeds are skipped by advancing their ranked streams; a DL slot
/// is never replaced or reduced.
pub fn build_generation_zero(
    proposal: &ProposalFreezeV1,
    ranked_anchors: &[RankedAnchorV1],
    ranked_diversity_seeds: &[GenomeV1],
) -> Result<GenerationZeroV1, String> {
    proposal.validate()?;
    let task = proposal.proposals[0].task;
    let mut population = Vec::with_capacity(80);
    let mut semantics = BTreeSet::new();
    for (rank, genome) in proposal.proposals.iter().enumerate() {
        // Proposal validation already guarantees exactly 64 unique values.
        semantics.insert(genome.semantic_sha256.clone());
        population.push(PopulationMemberV1 {
            slot: population.len() as u32,
            genome: genome.clone(),
            ancestry: GenomeAncestryV1 {
                origin: GenomeOriginV1::ModelProposal,
                source_rank: Some(rank as u32),
                parents: Vec::new(),
                seed: 0,
            },
        });
    }
    let mut anchor_count = 0;
    for (rank, anchor) in ranked_anchors.iter().enumerate() {
        anchor.genome.validate()?;
        if anchor.genome.task == task && semantics.insert(anchor.genome.semantic_sha256.clone()) {
            population.push(PopulationMemberV1 {
                slot: population.len() as u32,
                genome: anchor.genome.clone(),
                ancestry: GenomeAncestryV1 {
                    origin: GenomeOriginV1::GlobalGene,
                    source_rank: Some(rank as u32),
                    parents: Vec::new(),
                    seed: 0,
                },
            });
            anchor_count += 1;
            if anchor_count == 10 {
                break;
            }
        }
    }
    if anchor_count != 10 {
        return Err("ranked anchor stream exhausted before ten unique anchors".into());
    }
    let mut diversity_count = 0;
    for (rank, seed) in ranked_diversity_seeds.iter().enumerate() {
        seed.validate()?;
        if seed.task == task && semantics.insert(seed.semantic_sha256.clone()) {
            population.push(PopulationMemberV1 {
                slot: population.len() as u32,
                genome: seed.clone(),
                ancestry: GenomeAncestryV1 {
                    origin: GenomeOriginV1::DiversitySeed,
                    source_rank: Some(rank as u32),
                    parents: Vec::new(),
                    seed: 0,
                },
            });
            diversity_count += 1;
            if diversity_count == 6 {
                break;
            }
        }
    }
    if diversity_count != 6 {
        return Err("ranked diversity stream exhausted before six unique seeds".into());
    }
    let generation = GenerationZeroV1 {
        schema: GenerationZeroV1::SCHEMA.into(),
        dataset_sha256: proposal.dataset_sha256.clone(),
        proposal_freeze_sha256: proposal.identity()?,
        population,
    };
    generation.validate()?;
    Ok(generation)
}
