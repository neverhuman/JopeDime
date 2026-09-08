use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::jope_generalized_gp::{
    raw_sha256, GenerationZeroV1, GenomeAncestryV1, GenomeOriginV1, GenomeV1, GpTaskV1,
    PopulationMemberV1, ProposalFreezeV1,
};
use crate::search_v3::{canonical_sha256, validate_sha256};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};

const ONE_SIDED_90_Z: f64 = 1.281_551_565_544_600_4;
const LINEAGE_EXPLORATION: f64 = 0.10;

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let center = mean(values);
    (values
        .iter()
        .map(|value| (value - center).powi(2))
        .sum::<f64>()
        / (values.len() - 1) as f64)
        .sqrt()
}

/// One-sided 90% LCB with the two frozen zero-uplift prior observations.
pub fn confidence_score(values: &[f64]) -> Result<f64, String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("confidence score cannot consume nonfinite values".into());
    }
    let mut supported = values.to_vec();
    supported.extend([0.0, 0.0]);
    Ok(mean(&supported) - ONE_SIDED_90_Z * sample_sd(&supported) / (supported.len() as f64).sqrt())
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

pub(crate) fn semantic_mutation(parent: &GenomeV1, seed: u64) -> Option<GenomeV1> {
    let source = mutate_canonical_pipeline_v2(&parent.canonical_genome, seed, 6, true).ok()?;
    let child = GenomeV1::from_canonical(GpTaskV1::Regression, &source).ok()?;
    (child.semantic_sha256 != parent.semantic_sha256).then_some(child)
}

pub(crate) fn semantic_crossover(left: &GenomeV1, right: &GenomeV1, seed: u64) -> Option<GenomeV1> {
    if left.semantic_sha256 == right.semantic_sha256 {
        return None;
    }
    let source =
        crossover_canonical_pipelines(&left.canonical_genome, &right.canonical_genome, seed, 6)
            .ok()?;
    let child = GenomeV1::from_canonical(GpTaskV1::Regression, &source).ok()?;
    (child.semantic_sha256 != left.semantic_sha256
        && child.semantic_sha256 != right.semantic_sha256)
        .then_some(child)
}

pub(crate) fn regression_head_superfamily(head_family: &str) -> u8 {
    match head_family {
        "ridge" => 0,
        "elasticnet" => 1,
        "krr" => 2,
        "gpr" => 3,
        // A future checkpoint schema remains deterministic, but the frozen
        // Best5 checkpoint uses the four explicit regression families above.
        other => u8::from_str_radix(&raw_sha256(other.as_bytes())[..2], 16).unwrap() % 4,
    }
}

/// Matched Confidence-GP control offspring. `ranked` is the true confidence
/// ordering; no other arm state is accepted. Every child is produced by the
/// canonical semantic mutator/crossover and the exact 3/58/19 schedule is
/// preserved under deterministic duplicate retries.
pub fn build_control_offspring(
    ranked: &[GenomeV1],
    generation: u32,
    master_seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    if ranked.len() != 80 || !super::SEARCH_SEEDS.contains(&master_seed) {
        return Err("control offspring requires 80 ranked genomes and seed 101 or 202".into());
    }
    let semantics = ranked
        .iter()
        .map(|genome| {
            genome.validate()?;
            if genome.task != GpTaskV1::Regression {
                return Err("control offspring accepts regression genomes only".into());
            }
            Ok(genome.semantic_sha256.as_str())
        })
        .collect::<Result<BTreeSet<_>, String>>()?;
    if semantics.len() != 80 {
        return Err("control ranking contains semantic duplicates".into());
    }
    let mut population = Vec::with_capacity(80);
    let mut used = BTreeSet::<String>::new();
    for elite in ranked.iter().take(3) {
        used.insert(elite.semantic_sha256.clone());
        population.push(PopulationMemberV1 {
            slot: population.len() as u32,
            genome: elite.clone(),
            ancestry: GenomeAncestryV1 {
                origin: GenomeOriginV1::Elite,
                source_rank: None,
                parents: vec![elite.semantic_sha256.clone()],
                seed: seed64(&[
                    "best5-control",
                    &master_seed.to_string(),
                    &generation.to_string(),
                    "elite",
                    &population.len().to_string(),
                ]),
            },
        });
    }
    for (origin, count) in [
        (GenomeOriginV1::Mutation, 58usize),
        (GenomeOriginV1::Crossover, 19usize),
    ] {
        for offspring in 0..count {
            let mut accepted = None;
            for attempt in 0..100_000usize {
                let seed = seed64(&[
                    "best5-control",
                    &master_seed.to_string(),
                    &generation.to_string(),
                    match origin {
                        GenomeOriginV1::Mutation => "mutation",
                        GenomeOriginV1::Crossover => "crossover",
                        _ => unreachable!(),
                    },
                    &offspring.to_string(),
                    &attempt.to_string(),
                ]);
                // Draw across the complete ranked population. Later generations
                // can legitimately cluster the first forty into one structural
                // family, which otherwise makes the finite semantic crossover
                // stream unable to fill all nineteen required slots.
                let left = seed as usize % ranked.len();
                let right = seed.rotate_left(29) as usize % ranked.len();
                let child = match origin {
                    GenomeOriginV1::Mutation => semantic_mutation(&ranked[left], seed),
                    GenomeOriginV1::Crossover => {
                        semantic_crossover(&ranked[left], &ranked[right], seed)
                    }
                    _ => unreachable!(),
                };
                let Some(child) = child else { continue };
                if !used.insert(child.semantic_sha256.clone()) {
                    continue;
                }
                let parents = match origin {
                    GenomeOriginV1::Mutation => vec![ranked[left].semantic_sha256.clone()],
                    GenomeOriginV1::Crossover => vec![
                        ranked[left].semantic_sha256.clone(),
                        ranked[right].semantic_sha256.clone(),
                    ],
                    _ => unreachable!(),
                };
                accepted = Some((child, parents, seed));
                break;
            }
            let (genome, parents, seed) = accepted.ok_or_else(|| {
                format!("control {origin:?} stream exhausted at offspring {offspring}")
            })?;
            population.push(PopulationMemberV1 {
                slot: population.len() as u32,
                genome,
                ancestry: GenomeAncestryV1 {
                    origin,
                    source_rank: None,
                    parents,
                    seed,
                },
            });
        }
    }
    if population.len() != 80 {
        return Err("control violated the exact 3/58/19 schedule".into());
    }
    for member in &population {
        member.validate()?;
    }
    Ok(population)
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FounderScoreV1 {
    pub genome: GenomeV1,
    pub confidence_score: f64,
}

pub fn select_head_diverse_founders(founders: &[FounderScoreV1]) -> Result<Vec<GenomeV1>, String> {
    if founders.len() < 20 {
        return Err("lineage freeze requires at least 20 founders".into());
    }
    let mut ranked = founders.to_vec();
    for founder in &ranked {
        founder.genome.validate()?;
        if !founder.confidence_score.is_finite() {
            return Err("founder confidence score must be finite".into());
        }
    }
    ranked.sort_by(|left, right| {
        right
            .confidence_score
            .total_cmp(&left.confidence_score)
            .then_with(|| {
                left.genome
                    .semantic_sha256
                    .cmp(&right.genome.semantic_sha256)
            })
    });
    let mut selected = Vec::with_capacity(20);
    let mut seen = BTreeSet::<String>::new();
    let mut family_leaders = BTreeMap::<&str, &FounderScoreV1>::new();
    for founder in &ranked {
        family_leaders
            .entry(&founder.genome.head_family)
            .or_insert(founder);
    }
    for leader in family_leaders.values() {
        if seen.insert(leader.genome.semantic_sha256.clone()) {
            selected.push(leader.genome.clone());
        }
    }
    selected.sort_by(|left, right| {
        let left_score = ranked
            .iter()
            .find(|value| value.genome.semantic_sha256 == left.semantic_sha256)
            .unwrap();
        let right_score = ranked
            .iter()
            .find(|value| value.genome.semantic_sha256 == right.semantic_sha256)
            .unwrap();
        right_score
            .confidence_score
            .total_cmp(&left_score.confidence_score)
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    selected.truncate(20);
    for founder in ranked {
        if selected.len() == 20 {
            break;
        }
        if seen.insert(founder.genome.semantic_sha256.clone()) {
            selected.push(founder.genome);
        }
    }
    if selected.len() != 20 {
        return Err("fewer than 20 unique founders survived diversity selection".into());
    }
    Ok(selected)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageCreditV1 {
    pub founder_semantic_sha256: String,
    /// Paired child-minus-founder improvements only.
    pub paired_improvements: Vec<f64>,
}

impl LineageCreditV1 {
    fn ucb(&self, total_allocations: usize, allocated_this_generation: u32) -> f64 {
        let n = self.paired_improvements.len() + allocated_this_generation as usize;
        let zero_prior_mean = self.paired_improvements.iter().sum::<f64>() / (1 + n) as f64;
        zero_prior_mean
            + LINEAGE_EXPLORATION
                * (2.0 * ((1 + total_allocations) as f64).ln() / (1 + n) as f64).sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LineageOffspringV1 {
    pub founder_semantic_sha256: String,
    pub mutations: u32,
    pub crossovers: u32,
}

pub fn allocate_lineage_offspring(
    credits: &[LineageCreditV1],
) -> Result<Vec<LineageOffspringV1>, String> {
    if credits.len() != 20 {
        return Err("lineage UCB requires the frozen 20 founders".into());
    }
    let mut seen = BTreeSet::new();
    let mut allocations = Vec::with_capacity(20);
    for credit in credits {
        validate_sha256("lineage founder", &credit.founder_semantic_sha256)?;
        if !seen.insert(credit.founder_semantic_sha256.as_str())
            || credit
                .paired_improvements
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err("lineage credits are duplicate or nonfinite".into());
        }
        allocations.push(LineageOffspringV1 {
            founder_semantic_sha256: credit.founder_semantic_sha256.clone(),
            mutations: 1,
            crossovers: 0,
        });
    }
    let total_evidence = credits
        .iter()
        .map(|credit| credit.paired_improvements.len())
        .sum::<usize>();
    for draw in 0..57usize {
        let index = (0..credits.len())
            .min_by(|left, right| {
                credits[*right]
                    .ucb(
                        total_evidence + 20 + draw,
                        allocations[*right].mutations + allocations[*right].crossovers,
                    )
                    .total_cmp(&credits[*left].ucb(
                        total_evidence + 20 + draw,
                        allocations[*left].mutations + allocations[*left].crossovers,
                    ))
                    .then_with(|| {
                        credits[*left]
                            .founder_semantic_sha256
                            .cmp(&credits[*right].founder_semantic_sha256)
                    })
            })
            .unwrap();
        if draw < 38 {
            allocations[index].mutations += 1;
        } else {
            allocations[index].crossovers += 1;
        }
    }
    if allocations.iter().map(|value| value.mutations).sum::<u32>() != 58
        || allocations
            .iter()
            .map(|value| value.crossovers)
            .sum::<u32>()
            != 19
    {
        return Err(
            "lineage allocation violated the 20+38 mutation / 19 crossover schedule".into(),
        );
    }
    Ok(allocations)
}

/// Realize the frozen lineage-UCB allocation. Every mutation is rooted at its
/// frozen founder and every crossover names that founder first, so lineage
/// credit remains recoverable from the ordinary two-parent ancestry contract.
pub fn build_lineage_offspring(
    ranked: &[PopulationMemberV1],
    founders: &[GenomeV1],
    credits: &[LineageCreditV1],
    generation: u32,
    master_seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    if ranked.len() != 80 || founders.len() != 20 || !super::SEARCH_SEEDS.contains(&master_seed) {
        return Err(
            "lineage offspring requires 80 ranked members, 20 founders, and seed 101 or 202".into(),
        );
    }
    let founder_by_semantic = founders
        .iter()
        .map(|founder| {
            founder.validate()?;
            Ok((founder.semantic_sha256.as_str(), founder))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    if founder_by_semantic.len() != founders.len()
        || credits
            .iter()
            .map(|credit| credit.founder_semantic_sha256.as_str())
            .collect::<BTreeSet<_>>()
            != founder_by_semantic.keys().copied().collect()
    {
        return Err("lineage founders and UCB credit identities disagree".into());
    }
    let allocations = allocate_lineage_offspring(credits)?;
    let mut population = ranked
        .iter()
        .take(3)
        .cloned()
        .collect::<Vec<PopulationMemberV1>>();
    for (slot, member) in population.iter_mut().enumerate() {
        member.slot = slot as u32;
    }
    let mut used = population
        .iter()
        .map(|member| member.genome.semantic_sha256.clone())
        .collect::<BTreeSet<_>>();
    for allocation in allocations {
        let founder = founder_by_semantic[allocation.founder_semantic_sha256.as_str()];
        for (operation, count) in [
            (GenomeOriginV1::Mutation, allocation.mutations),
            (GenomeOriginV1::Crossover, allocation.crossovers),
        ] {
            for ordinal in 0..count {
                let mut accepted = None;
                for attempt in 0..100_000usize {
                    let seed = seed64(&[
                        "best5-lineage",
                        &master_seed.to_string(),
                        &generation.to_string(),
                        &allocation.founder_semantic_sha256,
                        match operation {
                            GenomeOriginV1::Mutation => "mutation",
                            GenomeOriginV1::Crossover => "crossover",
                            _ => unreachable!(),
                        },
                        &ordinal.to_string(),
                        &attempt.to_string(),
                    ]);
                    let (child, parents) = if operation == GenomeOriginV1::Mutation {
                        (
                            semantic_mutation(founder, seed),
                            vec![founder.semantic_sha256.clone()],
                        )
                    } else {
                        let mut right_index = seed.rotate_left(29) as usize % founders.len();
                        if founders[right_index].semantic_sha256 == founder.semantic_sha256 {
                            right_index = (right_index + 1) % founders.len();
                        }
                        let right = &founders[right_index];
                        (
                            semantic_crossover(founder, right, seed),
                            vec![
                                founder.semantic_sha256.clone(),
                                right.semantic_sha256.clone(),
                            ],
                        )
                    };
                    let Some(child) = child else { continue };
                    if !used.insert(child.semantic_sha256.clone()) {
                        continue;
                    }
                    accepted = Some((child, parents, seed));
                    break;
                }
                let (genome, parents, seed) = accepted.ok_or_else(|| {
                    format!(
                        "lineage {operation:?} stream exhausted for founder {}",
                        allocation.founder_semantic_sha256
                    )
                })?;
                population.push(PopulationMemberV1 {
                    slot: population.len() as u32,
                    genome,
                    ancestry: GenomeAncestryV1 {
                        origin: operation,
                        source_rank: None,
                        parents,
                        seed,
                    },
                });
            }
        }
    }
    if population.len() != 80 {
        return Err("lineage allocation did not materialize the exact 3/58/19 population".into());
    }
    for member in &population {
        member.validate()?;
    }
    Ok(population)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapOperationV1 {
    Mutation,
    Crossover,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapEliteCandidateV1 {
    pub semantic_sha256: String,
    pub head_superfamily: u8,
    pub transform_count: u8,
    pub score: f64,
}

impl MapEliteCandidateV1 {
    fn niche(&self) -> Result<(u8, u8), String> {
        validate_sha256("MAP-Elites semantic", &self.semantic_sha256)?;
        if self.head_superfamily >= 4 || self.transform_count > 6 || !self.score.is_finite() {
            return Err("MAP-Elites descriptor is outside its frozen grid".into());
        }
        let bucket = match self.transform_count {
            0..=2 => 0,
            3..=4 => 1,
            _ => 2,
        };
        Ok((self.head_superfamily, bucket))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapOffspringSlotV1 {
    pub slot: u32,
    pub operation: MapOperationV1,
    pub parent_niche: (u8, u8),
    pub cross_head_superfamilies: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MapElitesScheduleV1 {
    pub protected_semantic_sha256s: Vec<String>,
    pub slots: Vec<MapOffspringSlotV1>,
}

pub fn build_map_elites_schedule(
    candidates: &[MapEliteCandidateV1],
) -> Result<MapElitesScheduleV1, String> {
    if candidates.len() < 3 {
        return Err("MAP-Elites requires at least three candidates".into());
    }
    let mut ranked = candidates.to_vec();
    for candidate in &ranked {
        candidate.niche()?;
    }
    ranked.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.semantic_sha256.cmp(&right.semantic_sha256))
    });
    let mut protected = BTreeSet::new();
    for candidate in ranked.iter().take(3) {
        protected.insert(candidate.semantic_sha256.clone());
    }
    let mut niche_leaders = BTreeMap::new();
    for candidate in &ranked {
        niche_leaders
            .entry(candidate.niche()?)
            .or_insert(candidate.semantic_sha256.clone());
    }
    protected.extend(niche_leaders.values().cloned());
    if protected.len() > 80 {
        return Err("protected MAP-Elites exceed population".into());
    }
    let niches = niche_leaders.keys().copied().collect::<Vec<_>>();
    let ordinary_slots = 80 - protected.len();
    let mut family_ordinary = [0usize; 4];
    let mut slots = Vec::with_capacity(ordinary_slots);
    let mut niche_cursor = 0usize;
    let mut crossover_ordinal = 0usize;
    for slot in 0..ordinary_slots {
        let operation = if slot % 4 == 3 {
            MapOperationV1::Crossover
        } else {
            MapOperationV1::Mutation
        };
        let mut selected = None;
        for _ in 0..niches.len() {
            let niche = niches[niche_cursor % niches.len()];
            niche_cursor += 1;
            // The cap applies only to ordinary candidates; protected elites
            // were counted separately above.
            if family_ordinary[niche.0 as usize] < 20 {
                selected = Some(niche);
                break;
            }
        }
        let niche =
            selected.ok_or_else(|| "ordinary head-superfamily caps exhausted".to_string())?;
        family_ordinary[niche.0 as usize] += 1;
        let cross_head_superfamilies = if operation == MapOperationV1::Crossover {
            crossover_ordinal += 1;
            crossover_ordinal.is_multiple_of(4)
        } else {
            false
        };
        slots.push(MapOffspringSlotV1 {
            slot: (protected.len() + slot) as u32,
            operation,
            parent_niche: niche,
            cross_head_superfamilies,
        });
    }
    Ok(MapElitesScheduleV1 {
        protected_semantic_sha256s: protected.into_iter().collect(),
        slots,
    })
}

/// Materialize the protected-niche schedule into the next isolated
/// population. Protected elites are copied exactly; all ordinary slots use
/// the canonical semantic operators and the schedule's frozen round-robin
/// parent niches.
pub fn build_map_elites_offspring(
    ranked: &[GenomeV1],
    generation: u32,
    master_seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    if ranked.len() != 80 || !super::SEARCH_SEEDS.contains(&master_seed) {
        return Err("MAP-Elites offspring requires 80 ranked genomes and seed 101 or 202".into());
    }
    let candidates = ranked
        .iter()
        .enumerate()
        .map(|(index, genome)| {
            genome.validate()?;
            Ok(MapEliteCandidateV1 {
                semantic_sha256: genome.semantic_sha256.clone(),
                head_superfamily: regression_head_superfamily(&genome.head_family),
                transform_count: genome.transform_count as u8,
                score: (ranked.len() - index) as f64,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    let schedule = build_map_elites_schedule(&candidates)?;
    let genomes = ranked
        .iter()
        .map(|genome| (genome.semantic_sha256.as_str(), genome))
        .collect::<BTreeMap<_, _>>();
    if genomes.len() != ranked.len() {
        return Err("MAP-Elites ranking contains semantic duplicates".into());
    }
    let mut by_niche = BTreeMap::<(u8, u8), Vec<&GenomeV1>>::new();
    let mut by_family = BTreeMap::<u8, Vec<&GenomeV1>>::new();
    for (candidate, genome) in candidates.iter().zip(ranked) {
        by_niche.entry(candidate.niche()?).or_default().push(genome);
        by_family
            .entry(candidate.head_superfamily)
            .or_default()
            .push(genome);
    }
    let mut population = Vec::with_capacity(80);
    let mut used = BTreeSet::new();
    for semantic in &schedule.protected_semantic_sha256s {
        let genome = (*genomes[semantic.as_str()]).clone();
        used.insert(semantic.clone());
        population.push(PopulationMemberV1 {
            slot: population.len() as u32,
            genome,
            ancestry: GenomeAncestryV1 {
                origin: GenomeOriginV1::Elite,
                source_rank: None,
                parents: vec![semantic.clone()],
                seed: seed64(&[
                    "best5-map-elites",
                    &master_seed.to_string(),
                    &generation.to_string(),
                    "protected",
                    semantic,
                ]),
            },
        });
    }
    for scheduled in &schedule.slots {
        let left_pool = by_niche
            .get(&scheduled.parent_niche)
            .ok_or_else(|| "MAP-Elites scheduled an empty parent niche".to_string())?;
        let mut accepted = None;
        for attempt in 0..100_000usize {
            let seed = seed64(&[
                "best5-map-elites",
                &master_seed.to_string(),
                &generation.to_string(),
                &scheduled.slot.to_string(),
                &attempt.to_string(),
            ]);
            let left = left_pool[seed as usize % left_pool.len()];
            let (child, parents, origin) = match scheduled.operation {
                MapOperationV1::Mutation => (
                    semantic_mutation(left, seed),
                    vec![left.semantic_sha256.clone()],
                    GenomeOriginV1::Mutation,
                ),
                MapOperationV1::Crossover => {
                    let right_pool = if scheduled.cross_head_superfamilies {
                        ranked
                            .iter()
                            .filter(|genome| {
                                regression_head_superfamily(&genome.head_family)
                                    != scheduled.parent_niche.0
                            })
                            .collect::<Vec<_>>()
                    } else {
                        by_family[&scheduled.parent_niche.0].clone()
                    };
                    if right_pool.is_empty() {
                        continue;
                    }
                    let right = right_pool[seed.rotate_left(29) as usize % right_pool.len()];
                    (
                        semantic_crossover(left, right, seed),
                        vec![left.semantic_sha256.clone(), right.semantic_sha256.clone()],
                        GenomeOriginV1::Crossover,
                    )
                }
            };
            let Some(child) = child else { continue };
            if !used.insert(child.semantic_sha256.clone()) {
                continue;
            }
            accepted = Some((child, parents, origin, seed));
            break;
        }
        let (genome, parents, origin, seed) = accepted.ok_or_else(|| {
            format!(
                "MAP-Elites semantic operator stream exhausted at slot {}",
                scheduled.slot
            )
        })?;
        population.push(PopulationMemberV1 {
            slot: population.len() as u32,
            genome,
            ancestry: GenomeAncestryV1 {
                origin,
                source_rank: None,
                parents,
                seed,
            },
        });
    }
    if population.len() != 80 {
        return Err("MAP-Elites did not materialize exactly 80 members".into());
    }
    for member in &population {
        member.validate()?;
    }
    Ok(population)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DescriptorV1 {
    pub head_superfamily: u8,
    pub transform_type_counts: [u8; 8],
    pub transform_count_normalized: f64,
    pub complexity_normalized: f64,
    pub normalized_hyperparameters: [f64; 2],
}

impl DescriptorV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.head_superfamily >= 4
            || self.transform_type_counts.iter().any(|count| *count > 6)
            || [
                self.transform_count_normalized,
                self.complexity_normalized,
                self.normalized_hyperparameters[0],
                self.normalized_hyperparameters[1],
            ]
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("surrogate descriptor is outside its normalized domain".into());
        }
        Ok(())
    }

    fn vector(&self) -> Vec<f64> {
        let mut values = (0..4)
            .map(|family| (family == self.head_superfamily) as u8 as f64)
            .collect::<Vec<_>>();
        values.extend(
            self.transform_type_counts
                .iter()
                .map(|count| *count as f64 / 6.0),
        );
        values.push(self.transform_count_normalized);
        values.push(self.complexity_normalized);
        values.extend(self.normalized_hyperparameters);
        values
    }

    fn distance(&self, other: &Self) -> f64 {
        self.vector()
            .iter()
            .zip(other.vector())
            .map(|(left, right)| (left - right).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurrogateCandidateV1 {
    pub semantic_sha256: String,
    pub descriptor: DescriptorV1,
    /// Present only for truly evaluated historical observations. Acquisition
    /// predictions are never copied into this field.
    pub true_uplift: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurrogatePickV1 {
    pub semantic_sha256: String,
    pub acquisition: f64,
    pub descriptor_novelty: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SurrogateSelectionV1 {
    pub elite_semantic_sha256s: Vec<String>,
    pub ucb: Vec<SurrogatePickV1>,
    pub novelty: Vec<SurrogatePickV1>,
}

fn solve(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    for pivot in 0..rhs.len() {
        let best = (pivot..rhs.len()).max_by(|left, right| {
            matrix[*left][pivot]
                .abs()
                .total_cmp(&matrix[*right][pivot].abs())
        })?;
        if matrix[best][pivot].abs() < 1e-12 {
            return None;
        }
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        let scale = matrix[pivot][pivot];
        for value in matrix[pivot].iter_mut().skip(pivot) {
            *value /= scale;
        }
        rhs[pivot] /= scale;
        let normalized_pivot = matrix[pivot].clone();
        for row in 0..rhs.len() {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for (value, pivot_value) in matrix[row]
                .iter_mut()
                .zip(normalized_pivot.iter())
                .skip(pivot)
            {
                *value -= factor * pivot_value;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    Some(rhs)
}

fn bootstrap_ridge(
    training: &[SurrogateCandidateV1],
    ensemble: usize,
    seed: u64,
) -> Result<Vec<f64>, String> {
    let width = training[0].descriptor.vector().len() + 1;
    let mut xtx = vec![vec![0.0; width]; width];
    let mut xty = vec![0.0; width];
    for draw in 0..training.len() {
        let index = seed64(&[
            "surrogate-bootstrap",
            &seed.to_string(),
            &ensemble.to_string(),
            &draw.to_string(),
        ]) as usize
            % training.len();
        let sample = &training[index];
        let mut x = vec![1.0];
        x.extend(sample.descriptor.vector());
        let y = sample.true_uplift.unwrap();
        for row in 0..width {
            xty[row] += x[row] * y;
            for column in 0..width {
                xtx[row][column] += x[row] * x[column];
            }
        }
    }
    for (index, row) in xtx.iter_mut().enumerate().skip(1) {
        row[index] += 0.01;
    }
    solve(xtx, xty).ok_or_else(|| "surrogate ridge system is singular".into())
}

fn predict(weights: &[f64], descriptor: &DescriptorV1) -> f64 {
    weights[0]
        + weights[1..]
            .iter()
            .zip(descriptor.vector())
            .map(|(weight, value)| weight * value)
            .sum::<f64>()
}

pub fn build_surrogate_selection(
    generated: &[SurrogateCandidateV1],
    historical: &[SurrogateCandidateV1],
    elite_semantic_sha256s: &[String],
    seed: u64,
) -> Result<SurrogateSelectionV1, String> {
    if generated.len() != 320 || historical.len() < 16 || elite_semantic_sha256s.len() != 3 {
        return Err(
            "surrogate search requires 320 offspring, >=16 observations, and 3 elites".into(),
        );
    }
    let mut semantics = BTreeSet::new();
    for candidate in generated.iter().chain(historical) {
        validate_sha256("surrogate candidate", &candidate.semantic_sha256)?;
        candidate.descriptor.validate()?;
        if let Some(value) = candidate.true_uplift {
            if !value.is_finite() {
                return Err("historical surrogate uplift must be finite".into());
            }
        }
    }
    if generated
        .iter()
        .any(|candidate| !semantics.insert(candidate.semantic_sha256.as_str()))
        || generated
            .iter()
            .any(|candidate| candidate.true_uplift.is_some())
        || historical
            .iter()
            .any(|candidate| candidate.true_uplift.is_none())
        || elite_semantic_sha256s.iter().collect::<BTreeSet<_>>().len() != 3
    {
        return Err("surrogate pools are duplicate or lack true observations".into());
    }
    let ensemble = (0..16)
        .map(|index| bootstrap_ridge(historical, index, seed))
        .collect::<Result<Vec<_>, _>>()?;
    let elites = elite_semantic_sha256s.iter().collect::<BTreeSet<_>>();
    let mut acquisition = generated
        .iter()
        .filter(|candidate| !elites.contains(&candidate.semantic_sha256))
        .map(|candidate| {
            let predictions = ensemble
                .iter()
                .map(|weights| predict(weights, &candidate.descriptor))
                .collect::<Vec<_>>();
            let center = mean(&predictions);
            let sd = sample_sd(&predictions);
            (
                candidate,
                SurrogatePickV1 {
                    semantic_sha256: candidate.semantic_sha256.clone(),
                    acquisition: center + sd,
                    descriptor_novelty: 0.0,
                },
            )
        })
        .collect::<Vec<_>>();
    acquisition.sort_by(|left, right| {
        right
            .1
            .acquisition
            .total_cmp(&left.1.acquisition)
            .then_with(|| left.1.semantic_sha256.cmp(&right.1.semantic_sha256))
    });
    let ucb = acquisition
        .iter()
        .take(60)
        .map(|(_, pick)| pick.clone())
        .collect::<Vec<_>>();
    let mut selected = ucb
        .iter()
        .map(|pick| pick.semantic_sha256.as_str())
        .chain(elite_semantic_sha256s.iter().map(String::as_str))
        .collect::<BTreeSet<_>>();
    let mut novelty = Vec::with_capacity(17);
    while novelty.len() < 17 {
        let next = generated
            .iter()
            .filter(|candidate| !selected.contains(candidate.semantic_sha256.as_str()))
            .map(|candidate| {
                let distance = generated
                    .iter()
                    .filter(|other| selected.contains(other.semantic_sha256.as_str()))
                    .map(|other| candidate.descriptor.distance(&other.descriptor))
                    .min_by(f64::total_cmp)
                    .unwrap_or(f64::INFINITY);
                (candidate, distance)
            })
            .min_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.semantic_sha256.cmp(&right.0.semantic_sha256))
            })
            .ok_or_else(|| "novelty pool exhausted".to_string())?;
        selected.insert(next.0.semantic_sha256.as_str());
        novelty.push(SurrogatePickV1 {
            semantic_sha256: next.0.semantic_sha256.clone(),
            acquisition: 0.0,
            descriptor_novelty: next.1,
        });
    }
    Ok(SurrogateSelectionV1 {
        elite_semantic_sha256s: elite_semantic_sha256s.to_vec(),
        ucb,
        novelty,
    })
}

fn genome_descriptor(genome: &GenomeV1) -> DescriptorV1 {
    let transforms = [
        "standardize",
        "robust_scale",
        "log1p_abs",
        "signed_sqrt",
        "clip",
        "pca",
        "select_k",
        "poly2",
    ];
    let mut transform_type_counts = [0u8; 8];
    for (index, transform) in transforms.iter().enumerate() {
        transform_type_counts[index] = genome
            .canonical_genome
            .match_indices(&format!("({transform}"))
            .count()
            .min(6) as u8;
    }
    let digest = hex::decode(&genome.semantic_sha256).unwrap();
    DescriptorV1 {
        head_superfamily: regression_head_superfamily(&genome.head_family),
        transform_type_counts,
        transform_count_normalized: genome.transform_count as f64 / 6.0,
        complexity_normalized: (genome.complexity.saturating_sub(1).min(6)) as f64 / 6.0,
        normalized_hyperparameters: [digest[0] as f64 / 255.0, digest[1] as f64 / 255.0],
    }
}

/// Generate the frozen 320-member semantic acquisition pool and materialize
/// exactly 3 elites, 60 ensemble-UCB choices, and 17 descriptor-novelty
/// choices as the next evaluated population.
pub fn build_surrogate_offspring(
    ranked: &[GenomeV1],
    generation: u32,
    master_seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    if ranked.len() != 80 || !super::SEARCH_SEEDS.contains(&master_seed) {
        return Err("surrogate offspring requires 80 ranked genomes and seed 101 or 202".into());
    }
    for genome in ranked {
        genome.validate()?;
    }
    let historical = ranked
        .iter()
        .enumerate()
        .map(|(index, genome)| SurrogateCandidateV1 {
            semantic_sha256: genome.semantic_sha256.clone(),
            descriptor: genome_descriptor(genome),
            true_uplift: Some((ranked.len() - index) as f64 / ranked.len() as f64),
        })
        .collect::<Vec<_>>();
    let elite_semantics = ranked
        .iter()
        .take(3)
        .map(|genome| genome.semantic_sha256.clone())
        .collect::<Vec<_>>();
    let mut generated_genomes = BTreeMap::<String, (GenomeV1, GenomeAncestryV1)>::new();
    for genome in ranked.iter().take(3) {
        generated_genomes.insert(
            genome.semantic_sha256.clone(),
            (
                genome.clone(),
                GenomeAncestryV1 {
                    origin: GenomeOriginV1::Elite,
                    source_rank: None,
                    parents: vec![genome.semantic_sha256.clone()],
                    seed: seed64(&[
                        "best5-surrogate",
                        &master_seed.to_string(),
                        &generation.to_string(),
                        "elite",
                        &genome.semantic_sha256,
                    ]),
                },
            ),
        );
    }
    for offspring in 3..320usize {
        let mut accepted = None;
        for attempt in 0..100_000usize {
            let seed = seed64(&[
                "best5-surrogate",
                &master_seed.to_string(),
                &generation.to_string(),
                &offspring.to_string(),
                &attempt.to_string(),
            ]);
            let left = &ranked[seed as usize % ranked.len()];
            let crossover = offspring % 4 == 3;
            let (child, parents, origin) = if crossover {
                let right = &ranked[seed.rotate_left(29) as usize % ranked.len()];
                (
                    semantic_crossover(left, right, seed),
                    vec![left.semantic_sha256.clone(), right.semantic_sha256.clone()],
                    GenomeOriginV1::Crossover,
                )
            } else {
                (
                    semantic_mutation(left, seed),
                    vec![left.semantic_sha256.clone()],
                    GenomeOriginV1::Mutation,
                )
            };
            let Some(child) = child else { continue };
            if generated_genomes.contains_key(&child.semantic_sha256) {
                continue;
            }
            accepted = Some((child, parents, origin, seed));
            break;
        }
        let (child, parents, origin, seed) = accepted.ok_or_else(|| {
            format!("surrogate semantic generation exhausted at offspring {offspring}")
        })?;
        generated_genomes.insert(
            child.semantic_sha256.clone(),
            (
                child,
                GenomeAncestryV1 {
                    origin,
                    source_rank: None,
                    parents,
                    seed,
                },
            ),
        );
    }
    let generated = generated_genomes
        .values()
        .map(|(genome, _)| SurrogateCandidateV1 {
            semantic_sha256: genome.semantic_sha256.clone(),
            descriptor: genome_descriptor(genome),
            true_uplift: None,
        })
        .collect::<Vec<_>>();
    let selection = build_surrogate_selection(
        &generated,
        &historical,
        &elite_semantics,
        seed64(&[
            "best5-surrogate-selection",
            &master_seed.to_string(),
            &generation.to_string(),
        ]),
    )?;
    let selected = selection
        .elite_semantic_sha256s
        .iter()
        .cloned()
        .chain(selection.ucb.into_iter().map(|pick| pick.semantic_sha256))
        .chain(
            selection
                .novelty
                .into_iter()
                .map(|pick| pick.semantic_sha256),
        )
        .collect::<Vec<_>>();
    if selected.len() != 80 || selected.iter().collect::<BTreeSet<_>>().len() != 80 {
        return Err("surrogate selection did not produce 80 unique members".into());
    }
    selected
        .into_iter()
        .enumerate()
        .map(|(slot, semantic)| {
            let (genome, ancestry) = generated_genomes.remove(&semantic).ok_or_else(|| {
                "surrogate selected a genome outside its generated pool".to_string()
            })?;
            let member = PopulationMemberV1 {
                slot: slot as u32,
                genome,
                ancestry,
            };
            member.validate()?;
            Ok(member)
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualRowV1 {
    pub stable_row_id: String,
    pub target: f64,
    pub base_prediction: f64,
    pub residual: f64,
    pub fitted_without_row: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualTargetsV1 {
    pub oof_training: Vec<ResidualRowV1>,
    pub validation: Vec<ResidualRowV1>,
    pub inner_folds: u32,
}

/// Recreates five-fold base predictions inside the supplied outer training
/// partition. The callback receives fit row IDs and prediction row IDs; each
/// outer-validation prediction is made by a base fitted on outer training only.
pub fn cross_fitted_residual_targets<F>(
    rows: &[super::TargetRowV1],
    outer_train_row_ids: &[String],
    outer_validation_row_ids: &[String],
    seed: u64,
    mut predict_base: F,
) -> Result<ResidualTargetsV1, String>
where
    F: FnMut(&[String], &[String]) -> Result<Vec<f64>, String>,
{
    let by_id = rows
        .iter()
        .map(|row| {
            row.validate()?;
            Ok((row.stable_row_id.as_str(), row))
        })
        .collect::<Result<BTreeMap<_, _>, String>>()?;
    let outer_train = outer_train_row_ids.iter().collect::<BTreeSet<_>>();
    let outer_validation = outer_validation_row_ids.iter().collect::<BTreeSet<_>>();
    if outer_train.len() != outer_train_row_ids.len()
        || outer_validation.len() != outer_validation_row_ids.len()
        || !outer_train.is_disjoint(&outer_validation)
        || outer_train
            .iter()
            .chain(&outer_validation)
            .any(|row| !by_id.contains_key(row.as_str()))
    {
        return Err("outer residual partition is duplicate, overlapping, or unknown".into());
    }
    let mut groups = BTreeMap::<&str, Vec<&str>>::new();
    for row_id in outer_train_row_ids {
        let row = by_id[row_id.as_str()];
        groups
            .entry(row.atomic_group.as_str())
            .or_default()
            .push(row.stable_row_id.as_str());
    }
    if groups.len() < 5 {
        return Err("five-fold residual OOF requires at least five training groups".into());
    }
    let mut ordered_groups = groups.into_iter().collect::<Vec<_>>();
    ordered_groups.sort_by_key(|(group, _)| seed64(&["residual-oof", &seed.to_string(), group]));
    let mut folds = vec![Vec::<String>::new(); 5];
    for (index, (_, members)) in ordered_groups.into_iter().enumerate() {
        folds[index % 5].extend(members.into_iter().map(str::to_string));
    }
    let mut oof = Vec::with_capacity(outer_train.len());
    for fold in &folds {
        let held_out = fold.iter().collect::<BTreeSet<_>>();
        let fit = outer_train_row_ids
            .iter()
            .filter(|row| !held_out.contains(row))
            .cloned()
            .collect::<Vec<_>>();
        let predictions = predict_base(&fit, fold)?;
        if predictions.len() != fold.len() || predictions.iter().any(|value| !value.is_finite()) {
            return Err("base predictor returned invalid OOF predictions".into());
        }
        for (row_id, prediction) in fold.iter().zip(predictions) {
            let target = by_id[row_id.as_str()].target;
            oof.push(ResidualRowV1 {
                stable_row_id: row_id.clone(),
                target,
                base_prediction: prediction,
                residual: target - prediction,
                fitted_without_row: !fit.contains(row_id),
            });
        }
    }
    oof.sort_by(|left, right| left.stable_row_id.cmp(&right.stable_row_id));
    let validation_predictions = predict_base(outer_train_row_ids, outer_validation_row_ids)?;
    if validation_predictions.len() != outer_validation_row_ids.len()
        || validation_predictions
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("base predictor returned invalid outer-validation predictions".into());
    }
    let mut validation = outer_validation_row_ids
        .iter()
        .zip(validation_predictions)
        .map(|(row_id, prediction)| {
            let target = by_id[row_id.as_str()].target;
            ResidualRowV1 {
                stable_row_id: row_id.clone(),
                target,
                base_prediction: prediction,
                residual: target - prediction,
                fitted_without_row: !outer_train.contains(row_id),
            }
        })
        .collect::<Vec<_>>();
    validation.sort_by(|left, right| left.stable_row_id.cmp(&right.stable_row_id));
    if oof.len() != outer_train.len()
        || oof.iter().any(|row| !row.fitted_without_row)
        || validation.iter().any(|row| !row.fitted_without_row)
    {
        return Err("residual target was predicted by a base fitted on itself".into());
    }
    Ok(ResidualTargetsV1 {
        oof_training: oof,
        validation,
        inner_folds: 5,
    })
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResidualGenomeV1 {
    pub residual_genome: GenomeV1,
    pub eta: f64,
}

impl ResidualGenomeV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.residual_genome.validate()?;
        if ![0.25, 0.5, 0.75, 1.0].contains(&self.eta) {
            return Err("residual eta must be one of 0.25, 0.5, 0.75, 1.0".into());
        }
        Ok(())
    }

    pub fn semantic_sha256(&self, base_semantic_sha256: &str) -> Result<String, String> {
        self.validate()?;
        validate_sha256("residual base semantic", base_semantic_sha256)?;
        canonical_sha256(&(base_semantic_sha256, self))
    }
}

pub fn composite_residual_prediction(
    base: &[f64],
    residual: &[f64],
    eta: f64,
) -> Result<Vec<f64>, String> {
    if base.len() != residual.len()
        || ![0.25, 0.5, 0.75, 1.0].contains(&eta)
        || base.iter().chain(residual).any(|value| !value.is_finite())
    {
        return Err("invalid base/residual composite prediction".into());
    }
    Ok(base
        .iter()
        .zip(residual)
        .map(|(base, residual)| base + eta * residual)
        .collect())
}
