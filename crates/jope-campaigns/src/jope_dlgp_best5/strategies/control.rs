use super::*;

/// Matched Confidence-GP control offspring. `ranked` is the true confidence
/// ordering; no other arm state is accepted. Every child is produced by the
/// canonical semantic mutator/crossover and the exact 3/58/19 schedule is
/// preserved under deterministic duplicate retries.
pub fn build_control_offspring(
    ranked: &[GenomeV1],
    generation: u32,
    master_seed: u64,
) -> Result<Vec<PopulationMemberV1>, String> {
    if ranked.len() != 80 || !super::super::SEARCH_SEEDS.contains(&master_seed) {
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
