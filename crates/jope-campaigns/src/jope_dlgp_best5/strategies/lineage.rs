use super::*;

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
