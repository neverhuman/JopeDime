use super::ranking::compare_ranked;
use super::*;

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
