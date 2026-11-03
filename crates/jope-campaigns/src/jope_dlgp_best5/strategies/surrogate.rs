use super::*;

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
    if ranked.len() != 80 || !super::super::SEARCH_SEEDS.contains(&master_seed) {
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
