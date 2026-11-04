use super::*;

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

pub(super) fn compare_ranked(left: &RankedCandidateV1, right: &RankedCandidateV1) -> Ordering {
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
