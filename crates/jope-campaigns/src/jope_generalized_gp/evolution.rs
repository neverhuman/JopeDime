use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use feat_math::dsl::{ParamValue, Stage};
use feat_math::evolve::{crossover_canonical_pipelines, mutate_canonical_pipeline_v2};
use feat_math::genome::GenomeIr;

use super::contracts::*;

const ONE_SIDED_90_Z: f64 = 1.281_551_565_544_600_4;

pub fn confidence_score(uplifts: &[f64]) -> Result<f64, String> {
    if uplifts.iter().any(|value| !value.is_finite()) {
        return Err("confidence score cannot consume nonfinite uplift".into());
    }
    let mut supported = Vec::with_capacity(uplifts.len() + 2);
    supported.extend_from_slice(uplifts);
    supported.extend([0.0, 0.0]);
    let mean = supported.iter().sum::<f64>() / supported.len() as f64;
    Ok(mean - ONE_SIDED_90_Z * sample_sd(&supported) / (supported.len() as f64).sqrt())
}

pub fn build_global_gene_ranking(
    task: GpTaskV1,
    target_dataset_sha256: &str,
    observation_snapshot_sha256: &str,
    observations: &[GlobalGeneObservationV1],
) -> Result<Vec<GlobalGeneRankV1>, String> {
    crate::search_v3::validate_sha256("target dataset", target_dataset_sha256)?;
    crate::search_v3::validate_sha256("observation snapshot", observation_snapshot_sha256)?;
    let mut grouped = BTreeMap::<String, Vec<&GlobalGeneObservationV1>>::new();
    for observation in observations {
        observation.validate()?;
        if observation.observation_snapshot_sha256 != observation_snapshot_sha256 {
            return Err("global observation comes from a stale snapshot".into());
        }
        if observation.dataset_sha256 == target_dataset_sha256 || observation.genome.task != task {
            continue;
        }
        grouped
            .entry(observation.genome.semantic_sha256.clone())
            .or_default()
            .push(observation);
    }
    let mut ranked = Vec::new();
    for group in grouped.into_values() {
        let datasets: BTreeSet<_> = group
            .iter()
            .map(|observation| observation.dataset_sha256.as_str())
            .collect();
        let families: BTreeSet<_> = group
            .iter()
            .map(|observation| observation.source_family.as_str())
            .collect();
        if datasets.len() < 3 || families.len() < 2 {
            continue;
        }
        // At most one observation per dataset contributes.  If a malformed
        // snapshot contains duplicates, keeping the lowest percentile is the
        // conservative, deterministic interpretation.
        let mut per_dataset = BTreeMap::<&str, f64>::new();
        for observation in &group {
            per_dataset
                .entry(&observation.dataset_sha256)
                .and_modify(|value| *value = value.min(observation.within_dataset_percentile))
                .or_insert(observation.within_dataset_percentile);
        }
        let percentiles: Vec<_> = per_dataset.values().copied().collect();
        let strict_win_datasets: BTreeSet<_> = group
            .iter()
            .filter(|observation| observation.strict_win)
            .map(|observation| observation.dataset_sha256.as_str())
            .collect();
        ranked.push(GlobalGeneRankV1 {
            genome: group[0].genome.clone(),
            support_shrunk_lower_bound: confidence_score(&percentiles)?,
            worst_percentile: percentiles.iter().copied().min_by(f64::total_cmp).unwrap(),
            dataset_support: datasets.len() as u32,
            source_family_support: families.len() as u32,
            strict_wins: strict_win_datasets.len() as u32,
        });
    }
    ranked.sort_by(|left, right| {
        right
            .support_shrunk_lower_bound
            .total_cmp(&left.support_shrunk_lower_bound)
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

pub fn build_generation_zero(
    proposal_freeze: &ProposalFreezeV1,
    global_ranking: &[GlobalGeneRankV1],
    diversity_ranking: &[GenomeV1],
) -> Result<GenerationZeroV1, String> {
    proposal_freeze.validate()?;
    let task = proposal_freeze.proposals[0].task;
    let mut population = Vec::with_capacity(80);
    let mut semantics = BTreeSet::new();
    for (rank, genome) in proposal_freeze.proposals.iter().enumerate() {
        if genome.task != task || !semantics.insert(genome.semantic_sha256.clone()) {
            return Err("proposal freeze is not unique and task-separated".into());
        }
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
    let mut globals = 0;
    for (rank, candidate) in global_ranking.iter().enumerate() {
        candidate.genome.validate()?;
        if candidate.genome.task == task
            && semantics.insert(candidate.genome.semantic_sha256.clone())
        {
            population.push(PopulationMemberV1 {
                slot: population.len() as u32,
                genome: candidate.genome.clone(),
                ancestry: GenomeAncestryV1 {
                    origin: GenomeOriginV1::GlobalGene,
                    source_rank: Some(rank as u32),
                    parents: Vec::new(),
                    seed: 0,
                },
            });
            globals += 1;
            if globals == 10 {
                break;
            }
        }
    }
    if globals != 10 {
        return Err("global ranking exhausted before ten unique task-valid genes".into());
    }
    let mut diversity = 0;
    for (rank, candidate) in diversity_ranking.iter().enumerate() {
        candidate.validate()?;
        if candidate.task == task && semantics.insert(candidate.semantic_sha256.clone()) {
            population.push(PopulationMemberV1 {
                slot: population.len() as u32,
                genome: candidate.clone(),
                ancestry: GenomeAncestryV1 {
                    origin: GenomeOriginV1::DiversitySeed,
                    source_rank: Some(rank as u32),
                    parents: Vec::new(),
                    seed: 0,
                },
            });
            diversity += 1;
            if diversity == 6 {
                break;
            }
        }
    }
    if diversity != 6 {
        return Err("diversity ranking exhausted before six unique task-valid seeds".into());
    }
    let generation = GenerationZeroV1 {
        schema: GenerationZeroV1::SCHEMA.into(),
        dataset_sha256: proposal_freeze.dataset_sha256.clone(),
        proposal_freeze_sha256: proposal_freeze.identity()?,
        population,
    };
    generation.validate()?;
    for member in &generation.population {
        feat_math::capture_gp_source!(
            "generalized-gp",
            &format!("generation/0/slot/{}", member.slot),
            &member.genome.canonical_genome
        );
    }
    Ok(generation)
}

pub fn rank_population(
    arm: CampaignArmV1,
    evaluations: &[CandidateSplitEvaluationV1],
) -> Result<Vec<RankedCandidateV1>, String> {
    if evaluations.is_empty() {
        return Err("population ranking has no evaluations".into());
    }
    let mut grouped = BTreeMap::<String, Vec<&CandidateSplitEvaluationV1>>::new();
    for evaluation in evaluations {
        evaluation.validate()?;
        if evaluation.arm != arm {
            return Err("an arm cannot consume another arm's ranking state".into());
        }
        grouped
            .entry(evaluation.candidate.semantic_sha256.clone())
            .or_default()
            .push(evaluation);
    }
    let mut ranked = Vec::with_capacity(grouped.len());
    for group in grouped.into_values() {
        let valid = group.iter().all(|evaluation| evaluation.valid);
        let uplifts: Vec<_> = group
            .iter()
            .map(|evaluation| evaluation.paired_uplift)
            .collect();
        let mean = uplifts.iter().sum::<f64>() / uplifts.len() as f64;
        let primary = match arm {
            CampaignArmV1::RotatingMean | CampaignArmV1::LiteRaceControl => mean,
            CampaignArmV1::ConfidenceGp | CampaignArmV1::QualityDiversityGp => {
                confidence_score(&uplifts)?
            }
        };
        ranked.push(RankedCandidateV1 {
            candidate: group[0].candidate.clone(),
            primary_score: primary,
            mean_uplift: mean,
            worst_uplift: uplifts.iter().copied().min_by(f64::total_cmp).unwrap(),
            support: group.len() as u32,
            secondary_metric: group
                .iter()
                .map(|evaluation| evaluation.secondary_metric)
                .sum::<f64>()
                / group.len() as f64,
            runtime_seconds: group
                .iter()
                .map(|evaluation| evaluation.runtime.cpu_seconds)
                .sum(),
            valid,
        });
    }
    ranked.sort_by(compare_ranked);
    Ok(ranked)
}

fn compare_ranked(left: &RankedCandidateV1, right: &RankedCandidateV1) -> Ordering {
    right
        .valid
        .cmp(&left.valid)
        .then_with(|| right.primary_score.total_cmp(&left.primary_score))
        .then_with(|| right.mean_uplift.total_cmp(&left.mean_uplift))
        .then_with(|| right.worst_uplift.total_cmp(&left.worst_uplift))
        .then_with(|| left.secondary_metric.total_cmp(&right.secondary_metric))
        .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
        .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}

fn seed64(parts: &[&str]) -> u64 {
    let digest = raw_sha256(parts.join("|").as_bytes());
    u64::from_be_bytes(hex::decode(&digest[..16]).unwrap().try_into().unwrap())
}

fn tournament<'a>(
    pool: &[&'a RankedCandidateV1],
    seed_sha256: &str,
    generation: u32,
    offspring: usize,
    draw: usize,
) -> &'a RankedCandidateV1 {
    let mut indices = BTreeSet::new();
    let mut attempt = 0usize;
    while indices.len() < 3.min(pool.len()) {
        let seed = seed64(&[
            seed_sha256,
            &generation.to_string(),
            &offspring.to_string(),
            &draw.to_string(),
            &attempt.to_string(),
        ]);
        indices.insert(seed as usize % pool.len());
        attempt += 1;
    }
    indices
        .into_iter()
        .map(|index| pool[index])
        .min_by(|left, right| compare_ranked(left, right))
        .unwrap()
}

fn parent_pool(
    arm: CampaignArmV1,
    ranked: &[RankedCandidateV1],
    offspring: usize,
) -> Vec<&RankedCandidateV1> {
    let top: Vec<_> = ranked.iter().take(40).collect();
    if arm != CampaignArmV1::QualityDiversityGp {
        return top;
    }
    let mut niches = BTreeMap::<String, Vec<&RankedCandidateV1>>::new();
    for candidate in &top {
        niches
            .entry(candidate.candidate.niche())
            .or_default()
            .push(*candidate);
    }
    let values: Vec<_> = niches.into_values().collect();
    let selected = &values[offspring % values.len()];
    if selected.len() >= 3 {
        selected.clone()
    } else {
        top
    }
}

fn repair_head(source: &str, parent: &GenomeV1) -> Option<String> {
    let GenomeIr::Pipeline(mut child) = GenomeIr::parse(source).ok()? else {
        return None;
    };
    let GenomeIr::Pipeline(parent_pipeline) = GenomeIr::parse(&parent.canonical_genome).ok()?
    else {
        return None;
    };
    *child.stages.last_mut()? = parent_pipeline.stages.last()?.clone();
    Some(GenomeIr::Pipeline(child).canonical())
}

fn task_valid_child(task: GpTaskV1, source: String, parent: &GenomeV1) -> Option<GenomeV1> {
    GenomeV1::from_canonical(task, &source).ok().or_else(|| {
        let repaired = repair_head(&source, parent)?;
        GenomeV1::from_canonical(task, &repaired).ok()
    })
}

/// Bounded deterministic refill for saturated discrete mutator streams.  It
/// remains a real pipeline mutation/crossover: crossover interleaves both
/// transform stacks, and the seeded RFF stage supplies a replayable continuous
/// mutation coordinate. The task head is inherited from the left parent.
fn refill_child(
    origin: GenomeOriginV1,
    left: &GenomeV1,
    right: &GenomeV1,
    seed: u64,
) -> Option<String> {
    let GenomeIr::Pipeline(left_pipeline) = GenomeIr::parse(&left.canonical_genome).ok()? else {
        return None;
    };
    let GenomeIr::Pipeline(right_pipeline) = GenomeIr::parse(&right.canonical_genome).ok()? else {
        return None;
    };
    let left_head = left_pipeline.stages.last()?.clone();
    let mut transforms = left_pipeline.stages[..left_pipeline.stages.len() - 1].to_vec();
    if origin == GenomeOriginV1::Crossover {
        let right_transforms = &right_pipeline.stages[..right_pipeline.stages.len() - 1];
        let mut interleaved = Vec::new();
        let width = transforms.len().max(right_transforms.len());
        for index in 0..width {
            if let Some(stage) = transforms.get(index) {
                interleaved.push(stage.clone());
            }
            if let Some(stage) = right_transforms.get(index) {
                interleaved.push(stage.clone());
            }
        }
        transforms = interleaved;
    }
    transforms.truncate(5);
    let mut marker = Stage::new("rff_cos");
    marker.params.insert("n".into(), ParamValue::Int(24));
    marker.params.insert(
        "seed".into(),
        ParamValue::Int((seed & i64::MAX as u64) as i64),
    );
    transforms.push(marker);
    transforms.push(left_head);
    Some(GenomeIr::Pipeline(feat_math::dsl::DslExpr { stages: transforms }).canonical())
}

pub fn build_next_generation(
    arm: CampaignArmV1,
    generation: u32,
    seed_sha256: &str,
    ranked: &[RankedCandidateV1],
) -> Result<Vec<PopulationMemberV1>, String> {
    crate::search_v3::validate_sha256("evolution seed", seed_sha256)?;
    if ranked.len() != 80 {
        return Err("next generation requires exactly 80 ranked candidates".into());
    }
    if ranked
        .windows(2)
        .any(|pair| compare_ranked(&pair[0], &pair[1]) == Ordering::Greater)
    {
        return Err("next generation received an unsorted ranking".into());
    }
    let task = ranked[0].candidate.task;
    if ranked
        .iter()
        .any(|candidate| candidate.candidate.task != task)
    {
        return Err("next generation cannot mix tasks".into());
    }
    let mut population = Vec::with_capacity(80);
    let mut semantics = BTreeSet::new();
    for elite in ranked.iter().take(3) {
        semantics.insert(elite.candidate.semantic_sha256.clone());
        population.push(PopulationMemberV1 {
            slot: population.len() as u32,
            genome: elite.candidate.clone(),
            ancestry: GenomeAncestryV1 {
                origin: GenomeOriginV1::Elite,
                source_rank: None,
                parents: vec![elite.candidate.semantic_sha256.clone()],
                seed: seed64(&[
                    seed_sha256,
                    &generation.to_string(),
                    "elite",
                    &population.len().to_string(),
                ]),
            },
        });
    }
    let mut ordinary_niches = BTreeMap::<String, usize>::new();
    let mut add_offspring = |origin: GenomeOriginV1,
                             count: usize,
                             population: &mut Vec<PopulationMemberV1>|
     -> Result<(), String> {
        for offspring in 0..count {
            let logical = population.len();
            let mut accepted = None;
            for attempt in 0..100_000usize {
                let pool = parent_pool(arm, ranked, logical + attempt);
                let left = tournament(&pool, seed_sha256, generation, logical, attempt * 2);
                let right = tournament(&pool, seed_sha256, generation, logical, attempt * 2 + 1);
                let seed = seed64(&[
                    seed_sha256,
                    &generation.to_string(),
                    match origin {
                        GenomeOriginV1::Mutation => "mutation",
                        GenomeOriginV1::Crossover => "crossover",
                        _ => unreachable!(),
                    },
                    &offspring.to_string(),
                    &attempt.to_string(),
                ]);
                let source = match origin {
                    GenomeOriginV1::Mutation if attempt < 512 => mutate_canonical_pipeline_v2(
                        &left.candidate.canonical_genome,
                        seed,
                        6,
                        true,
                    )
                    .ok(),
                    GenomeOriginV1::Crossover => {
                        if left.candidate.semantic_sha256 == right.candidate.semantic_sha256 {
                            continue;
                        }
                        if attempt < 512 {
                            crossover_canonical_pipelines(
                                &left.candidate.canonical_genome,
                                &right.candidate.canonical_genome,
                                seed,
                                6,
                            )
                            .ok()
                        } else {
                            refill_child(origin, &left.candidate, &right.candidate, seed)
                        }
                    }
                    GenomeOriginV1::Mutation => {
                        refill_child(origin, &left.candidate, &right.candidate, seed)
                    }
                    _ => unreachable!(),
                };
                let Some(source) = source else { continue };
                let Some(child) = task_valid_child(task, source, &left.candidate) else {
                    continue;
                };
                if !semantics.insert(child.semantic_sha256.clone()) {
                    continue;
                }
                let niche = child.niche();
                if arm == CampaignArmV1::QualityDiversityGp
                    && ordinary_niches.get(&niche).copied().unwrap_or(0) >= 20
                {
                    semantics.remove(&child.semantic_sha256);
                    continue;
                }
                let parents = match origin {
                    GenomeOriginV1::Mutation => vec![left.candidate.semantic_sha256.clone()],
                    GenomeOriginV1::Crossover => vec![
                        left.candidate.semantic_sha256.clone(),
                        right.candidate.semantic_sha256.clone(),
                    ],
                    _ => unreachable!(),
                };
                *ordinary_niches.entry(niche).or_default() += 1;
                accepted = Some((child, parents, seed));
                break;
            }
            let (genome, parents, seed) = accepted.ok_or_else(|| {
                format!("{origin:?} stream exhausted while filling offspring {offspring}")
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
        Ok(())
    };
    add_offspring(GenomeOriginV1::Mutation, 58, &mut population)?;
    add_offspring(GenomeOriginV1::Crossover, 19, &mut population)?;
    if population.len() != 80 {
        return Err("offspring assembly violated the 3/58/19 schedule".into());
    }
    for member in &population {
        member.validate()?;
    }
    for member in &population {
        feat_math::capture_gp_source!(
            "generalized-gp",
            &format!("successor/slot/{}", member.slot),
            &member.genome.canonical_genome
        );
    }
    Ok(population)
}

pub fn build_provisional_archive<F>(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    split_plan_sha256: &str,
    ranked_generations: &[Vec<RankedCandidateV1>],
    existing_evaluations: &[CandidateSplitEvaluationV1],
    mut backfill: F,
) -> Result<ArchiveCheckpointV1, String>
where
    F: FnMut(&GenomeV1, u32) -> Result<CandidateSplitEvaluationV1, String>,
{
    crate::search_v3::validate_sha256("archive dataset", dataset_sha256)?;
    crate::search_v3::validate_sha256("archive split plan", split_plan_sha256)?;
    if ranked_generations.len() != 5 || ranked_generations.iter().any(Vec::is_empty) {
        return Err("archive construction requires five nonempty generation rankings".into());
    }
    fn add_candidate(
        candidates: &mut BTreeMap<String, GenomeV1>,
        selected: &mut Vec<String>,
        candidate: &GenomeV1,
    ) {
        candidates
            .entry(candidate.semantic_sha256.clone())
            .or_insert_with(|| candidate.clone());
        if !selected.contains(&candidate.semantic_sha256) {
            selected.push(candidate.semantic_sha256.clone());
        }
    }
    let mut candidates = BTreeMap::<String, GenomeV1>::new();
    let mut selected = Vec::<String>::new();
    // Preserve every generation leader.
    for generation in ranked_generations {
        add_candidate(&mut candidates, &mut selected, &generation[0].candidate);
    }
    // Preserve one leader from every observed head family before score fill.
    let mut family_leaders = BTreeMap::<String, &RankedCandidateV1>::new();
    for candidate in ranked_generations.iter().flatten() {
        family_leaders
            .entry(candidate.candidate.head_family.clone())
            .and_modify(|current| {
                if compare_ranked(candidate, current) == Ordering::Less {
                    *current = candidate;
                }
            })
            .or_insert(candidate);
    }
    for leader in family_leaders.values() {
        add_candidate(&mut candidates, &mut selected, &leader.candidate);
    }
    let all_ranked = rank_population(arm, existing_evaluations)?;
    for leader in &all_ranked {
        add_candidate(&mut candidates, &mut selected, &leader.candidate);
        if selected.len() == 20 {
            break;
        }
    }
    if selected.len() < 20 {
        for generation in ranked_generations {
            for leader in generation {
                add_candidate(&mut candidates, &mut selected, &leader.candidate);
                if selected.len() == 20 {
                    break;
                }
            }
            if selected.len() == 20 {
                break;
            }
        }
    }
    selected.truncate(20);
    if selected.len() != 20 {
        return Err("fewer than 20 unique genomes are available for the archive".into());
    }

    let mut confirmed = Vec::with_capacity(20);
    for semantic in selected {
        let candidate = candidates
            .get(&semantic)
            .cloned()
            .or_else(|| {
                ranked_generations
                    .iter()
                    .flatten()
                    .find(|ranked| ranked.candidate.semantic_sha256 == semantic)
                    .map(|ranked| ranked.candidate.clone())
            })
            .unwrap();
        let mut split_evaluations = Vec::with_capacity(5);
        for split_index in 0..5u32 {
            let evaluation = existing_evaluations
                .iter()
                .find(|evaluation| {
                    evaluation.arm == arm
                        && evaluation.candidate.semantic_sha256 == semantic
                        && evaluation.split_index == split_index
                })
                .cloned()
                .map(Ok)
                .unwrap_or_else(|| backfill(&candidate, split_index))?;
            evaluation.validate()?;
            if evaluation.arm != arm
                || evaluation.dataset_sha256 != dataset_sha256
                || evaluation.split_index != split_index
                || evaluation.candidate.semantic_sha256 != semantic
            {
                return Err("archive backfill returned stale or mismatched evidence".into());
            }
            split_evaluations.push(evaluation);
        }
        confirmed.push((candidate, split_evaluations));
    }
    let all_confirmed: Vec<_> = confirmed
        .iter()
        .flat_map(|(_, evaluations)| evaluations.iter().cloned())
        .collect();
    let confirmed_ranking = rank_population(arm, &all_confirmed)?;
    let rank: BTreeMap<_, _> = confirmed_ranking
        .iter()
        .enumerate()
        .map(|(rank, candidate)| (candidate.candidate.semantic_sha256.as_str(), rank))
        .collect();
    confirmed.sort_by_key(|(candidate, _)| rank[&candidate.semantic_sha256.as_str()]);
    let archive = ArchiveCheckpointV1 {
        schema: ArchiveCheckpointV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        split_plan_sha256: split_plan_sha256.into(),
        members: confirmed
            .into_iter()
            .enumerate()
            .map(|(rank, (candidate, split_evaluations))| ArchiveMemberV1 {
                candidate,
                split_evaluations,
                confirmed_rank: rank as u32,
            })
            .collect(),
    };
    archive.validate()?;
    Ok(archive)
}

fn compare_cv7(left: &Cv7CandidateEvidenceV1, right: &Cv7CandidateEvidenceV1) -> Ordering {
    right
        .valid
        .cmp(&left.valid)
        .then_with(|| right.robust_objective().total_cmp(&left.robust_objective()))
        .then_with(|| left.secondary_metric.total_cmp(&right.secondary_metric))
        .then_with(|| left.runtime_seconds.total_cmp(&right.runtime_seconds))
        .then_with(|| left.candidate.complexity.cmp(&right.candidate.complexity))
        .then_with(|| {
            left.candidate
                .semantic_sha256
                .cmp(&right.candidate.semantic_sha256)
        })
}

pub fn select_cv7_winner(
    arm: CampaignArmV1,
    dataset_sha256: &str,
    archive_sha256: &str,
    cv7_plan_sha256: &str,
    mut finalists: Vec<Cv7CandidateEvidenceV1>,
) -> Result<Cv7SelectionFreezeV1, String> {
    if finalists.len() != 10 {
        return Err("CV7 selection requires exactly ten candidates".into());
    }
    for finalist in &finalists {
        finalist.validate()?;
    }
    finalists.sort_by(compare_cv7);
    if !finalists[0].valid {
        return Err("CV7 cannot freeze a winner when every finalist is invalid".into());
    }
    let selection = Cv7SelectionFreezeV1 {
        schema: Cv7SelectionFreezeV1::SCHEMA.into(),
        arm,
        dataset_sha256: dataset_sha256.into(),
        archive_sha256: archive_sha256.into(),
        cv7_plan_sha256: cv7_plan_sha256.into(),
        winner_semantic_sha256: finalists[0].candidate.semantic_sha256.clone(),
        finalists,
    };
    selection.validate()?;
    Ok(selection)
}
