use super::*;

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
