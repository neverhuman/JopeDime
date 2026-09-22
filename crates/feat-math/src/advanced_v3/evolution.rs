//! Deterministic four-island evolution for the test-adaptive V3 campaign.
//!
//! This module does not call the legacy evolutionary engine.  Every child is
//! assembled from typed, bounded stage edits and canonicalized through the
//! public genome parser before it can enter a population or checkpoint.

use crate::dsl::{DslExpr, ParamValue, Stage};
use crate::{GenomeIr, Pcg32};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

const EVOLUTION_VERSION: &str = "golden42-v3-evolution-v1";
const ELITES: usize = 2;
const TOURNAMENT: usize = 3;
const ARCHIVE_LIMIT: usize = 128;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct V3Candidate {
    pub syntax: String,
    pub semantic_hash: String,
    pub lineage: Vec<String>,
    pub island: u8,
    pub operator: String,
    pub step_micros: u32,
    pub generation: usize,
}

impl V3Candidate {
    pub fn complexity(&self) -> usize {
        match GenomeIr::parse(&self.syntax) {
            Ok(GenomeIr::Pipeline(pipeline)) => pipeline.stages.len(),
            Ok(GenomeIr::Portfolio(portfolio)) => {
                portfolio
                    .members
                    .iter()
                    .map(|member| member.stages.len())
                    .sum::<usize>()
                    + 1
            }
            Ok(GenomeIr::Symbolic { .. }) => 1,
            Err(_) => usize::MAX,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct V3ScoredCandidate {
    pub candidate: V3Candidate,
    pub rmse: f64,
    pub novelty: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct V3Archive {
    pub entries: Vec<V3ScoredCandidate>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct V3EvolutionConfig {
    pub seed: u64,
    pub islands: usize,
    pub island_size: usize,
    pub migration_interval: usize,
    pub stagnation_generations: usize,
}

impl Default for V3EvolutionConfig {
    fn default() -> Self {
        Self {
            seed: 42,
            islands: 4,
            island_size: 20,
            migration_interval: 5,
            stagnation_generations: 8,
        }
    }
}

impl V3EvolutionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.islands != 4 || self.island_size != 20 {
            bail!("Golden42 V3 requires four islands of twenty candidates");
        }
        if self.migration_interval == 0 || self.stagnation_generations == 0 {
            bail!("migration and stagnation intervals must be positive");
        }
        Ok(())
    }

    pub fn population(&self) -> usize {
        self.islands.saturating_mul(self.island_size)
    }
}

fn hash(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update((part.len() as u64).to_le_bytes());
        hasher.update(part);
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn canonicalize(source: &str) -> Result<String> {
    let genome = GenomeIr::parse(source)?;
    let canonical = genome.canonical();
    if GenomeIr::parse(&canonical)?.canonical() != canonical {
        bail!("genome canonicalization is not idempotent");
    }
    Ok(canonical)
}

fn make_candidate(
    source: &str,
    island: usize,
    generation: usize,
    operator: &str,
    step_micros: u32,
    parents: &[&V3Candidate],
) -> Result<V3Candidate> {
    let syntax = canonicalize(source)?;
    let semantic_hash = hash(&[EVOLUTION_VERSION.as_bytes(), syntax.as_bytes()]);
    let mut lineage = Vec::new();
    for parent in parents {
        for ancestor in &parent.lineage {
            if !lineage.contains(ancestor) {
                lineage.push(ancestor.clone());
            }
        }
        if !lineage.contains(&parent.semantic_hash) {
            lineage.push(parent.semantic_hash.clone());
        }
    }
    if lineage.is_empty() {
        lineage.push(semantic_hash.clone());
    }
    Ok(V3Candidate {
        syntax,
        semantic_hash,
        lineage,
        island: u8::try_from(island).context("island index exceeds u8")?,
        operator: operator.to_string(),
        step_micros: step_micros.clamp(25_000, 1_000_000),
        generation,
    })
}

/// Enumerate a stable four-island seed population. Invalid and duplicate
/// supplied seeds are ignored; deterministic typed baselines fill every slot.
pub fn initial_population(config: &V3EvolutionConfig, seeds: &[String]) -> Vec<V3Candidate> {
    config
        .validate()
        .expect("invalid Golden42 V3 evolution configuration");
    let mut islands = vec![Vec::with_capacity(config.island_size); config.islands];
    let mut seen = BTreeSet::new();
    for (index, source) in seeds.iter().enumerate() {
        if islands.iter().map(Vec::len).sum::<usize>() == config.population() {
            break;
        }
        let island = index % config.islands;
        if islands[island].len() == config.island_size {
            continue;
        }
        let Ok(candidate) = make_candidate(source, island, 0, "receipt_seed", 150_000, &[]) else {
            continue;
        };
        if seen.insert(candidate.semantic_hash.clone()) {
            islands[island].push(candidate);
        }
    }
    let mut nonce = 0usize;
    for island in 0..config.islands {
        while islands[island].len() < config.island_size {
            let source = baseline(island, nonce, config.seed);
            nonce += 1;
            let candidate = make_candidate(
                &source,
                island,
                0,
                "grammar_baseline",
                150_000 + (nonce as u32 % 20_000),
                &[],
            )
            .expect("typed baseline must compile");
            if seen.insert(candidate.semantic_hash.clone()) {
                islands[island].push(candidate);
            }
        }
    }
    islands.into_iter().flatten().collect()
}

pub fn deterministic_seed_enumeration(config: &V3EvolutionConfig, seeds: &[String]) -> Vec<String> {
    initial_population(config, seeds)
        .into_iter()
        .map(|candidate| candidate.semantic_hash)
        .collect()
}

fn baseline(island: usize, nonce: usize, campaign_seed: u64) -> String {
    match island {
        0 => {
            let bins = [2, 4, 8, 16, 25, 32, 48, 64][nonce % 8];
            let topk = 4 + nonce % 21;
            let alpha = 0.001 + (nonce % 1_000_000) as f64 * 0.0000001;
            format!(
                "(pipe X (target kind=log1p) (quantile_bins append=true bins={bins}) (blom append=true) (pair_prod append=true topk={topk}) (elasticnet alpha={alpha} l1_ratio=0.09375316848524005 sweeps=60))"
            )
        }
        1 => {
            let degree = 1 + nonce % 8;
            let q = 0.005 + (nonce % 1_000_000) as f64 * 0.0000002;
            format!(
                "(pipe X (winsor_clip q={q}) (cheb append=true degree={degree}) (ridge lambda_frac=0.02))"
            )
        }
        2 => {
            let features = 16 + nonce % 49;
            let seed = campaign_seed.wrapping_add(nonce as u64 * 17);
            format!("(pipe X (rff_cos gamma=1.0 n={features} seed={seed}) (ridge_gcv))")
        }
        _ => {
            let knots = 2 + nonce % 7;
            let topk = 4 + nonce % 21;
            let q = 0.005 + (nonce % 1_000_000) as f64 * 0.0000002;
            format!(
                "(pipe X (winsor_clip q={q}) (hinge append=true knots={knots}) (pair_prod append=true topk={topk}) (elasticnet alpha=0.025 l1_ratio=0.1 sweeps=60))"
            )
        }
    }
}

fn ranked(mut values: Vec<V3ScoredCandidate>) -> Vec<V3ScoredCandidate> {
    values.sort_by(|left, right| {
        left.rmse
            .total_cmp(&right.rmse)
            .then_with(|| {
                left.candidate
                    .complexity()
                    .cmp(&right.candidate.complexity())
            })
            .then_with(|| {
                left.candidate
                    .semantic_hash
                    .cmp(&right.candidate.semantic_hash)
            })
    });
    values
}

fn tournament<'a>(pool: &'a [V3ScoredCandidate], rng: &mut Pcg32) -> &'a V3ScoredCandidate {
    let mut best = rng.next_u32() as usize % pool.len();
    for _ in 1..TOURNAMENT {
        let challenger = rng.next_u32() as usize % pool.len();
        if score_order(&pool[challenger], &pool[best]) == Ordering::Less {
            best = challenger;
        }
    }
    &pool[best]
}

fn score_order(left: &V3ScoredCandidate, right: &V3ScoredCandidate) -> Ordering {
    left.rmse
        .total_cmp(&right.rmse)
        .then_with(|| {
            left.candidate
                .complexity()
                .cmp(&right.candidate.complexity())
        })
        .then_with(|| {
            left.candidate
                .semantic_hash
                .cmp(&right.candidate.semantic_hash)
        })
}

fn adapted_step(parent: u32, rng: &mut Pcg32) -> u32 {
    let factor = 0.75 + 0.5 * rng.next_f64();
    ((parent as f64 * factor).round() as u32).clamp(25_000, 1_000_000)
}

pub fn mutate_candidate(
    parent: &V3Candidate,
    seed: u64,
    generation: usize,
    island: usize,
) -> Result<V3Candidate> {
    let GenomeIr::Pipeline(mut pipeline) = GenomeIr::parse(&parent.syntax)? else {
        return make_candidate(
            &baseline(island, seed as usize, seed),
            island,
            generation,
            "stagnation_reseed",
            parent.step_micros,
            &[parent],
        );
    };
    if pipeline.stages.is_empty() {
        bail!("pipeline has no head");
    }
    let mut rng = Pcg32::new(seed, generation as u64 ^ ((island as u64) << 32));
    let transform_count = pipeline.stages.len() - 1;
    let choice = rng.next_u32() % 6;
    match choice {
        0 => tune_head(pipeline.stages.last_mut().expect("head exists"), &mut rng),
        1 if transform_count > 0 => {
            let index = rng.next_u32() as usize % transform_count;
            tune_transform(&mut pipeline.stages[index], &mut rng);
        }
        2 => {
            let mut stage = Stage::new("winsor_clip");
            stage.params.insert(
                "q".to_string(),
                ParamValue::Float(0.005 + 0.245 * rng.next_f64()),
            );
            pipeline.stages.insert(transform_count, stage);
        }
        3 if transform_count > 1 => {
            let index = rng.next_u32() as usize % transform_count;
            pipeline.stages.remove(index);
        }
        4 if transform_count > 1 => {
            let a = rng.next_u32() as usize % transform_count;
            let b = rng.next_u32() as usize % transform_count;
            pipeline.stages.swap(a, b);
        }
        _ => {
            let mut stage = Stage::new("standardize");
            stage.params.insert(
                "append".to_string(),
                ParamValue::Bool(rng.next_u32() & 1 == 1),
            );
            pipeline.stages.insert(transform_count, stage);
        }
    }
    let source = GenomeIr::Pipeline(pipeline).canonical();
    make_candidate(
        &source,
        island,
        generation,
        "mutation",
        adapted_step(parent.step_micros, &mut rng),
        &[parent],
    )
}

pub fn crossover_candidates(
    left: &V3Candidate,
    right: &V3Candidate,
    seed: u64,
    generation: usize,
    island: usize,
) -> Result<V3Candidate> {
    let (GenomeIr::Pipeline(left_pipe), GenomeIr::Pipeline(right_pipe)) = (
        GenomeIr::parse(&left.syntax)?,
        GenomeIr::parse(&right.syntax)?,
    ) else {
        return mutate_candidate(left, seed, generation, island);
    };
    if left_pipe.stages.is_empty() || right_pipe.stages.is_empty() {
        bail!("crossover parent has no head");
    }
    let mut rng = Pcg32::new(seed, generation as u64 ^ 0xC205_50A3);
    let left_transforms = &left_pipe.stages[..left_pipe.stages.len() - 1];
    let right_transforms = &right_pipe.stages[..right_pipe.stages.len() - 1];
    let left_cut = if left_transforms.is_empty() {
        0
    } else {
        rng.next_u32() as usize % (left_transforms.len() + 1)
    };
    let right_cut = if right_transforms.is_empty() {
        0
    } else {
        rng.next_u32() as usize % (right_transforms.len() + 1)
    };
    let mut stages = left_transforms[..left_cut].to_vec();
    stages.extend_from_slice(&right_transforms[right_cut..]);
    stages.truncate(8);
    let head = if rng.next_u32() & 1 == 0 {
        left_pipe.stages.last().expect("head exists").clone()
    } else {
        right_pipe.stages.last().expect("head exists").clone()
    };
    stages.push(head);
    let source = GenomeIr::Pipeline(DslExpr { stages }).canonical();
    make_candidate(
        &source,
        island,
        generation,
        "crossover",
        adapted_step((left.step_micros / 2) + (right.step_micros / 2), &mut rng),
        &[left, right],
    )
}

fn tune_head(head: &mut Stage, rng: &mut Pcg32) {
    match head.op.as_str() {
        "elasticnet" => {
            let alpha = head.get_f64("alpha", 0.001);
            let factor = 0.75 + 0.5 * rng.next_f64();
            head.params.insert(
                "alpha".to_string(),
                ParamValue::Float((alpha * factor).clamp(1e-6, 1.0)),
            );
            head.params.insert(
                "l1_ratio".to_string(),
                ParamValue::Float((0.02 + 0.48 * rng.next_f64()).clamp(0.0, 1.0)),
            );
        }
        "ridge" => {
            head.params.insert(
                "lambda_frac".to_string(),
                ParamValue::Float((0.001 + 0.099 * rng.next_f64()).clamp(1e-6, 1.0)),
            );
        }
        "gpr" => {
            head.params.insert(
                "steps".to_string(),
                ParamValue::Int(1 + (rng.next_u32() % 60) as i64),
            );
        }
        _ => {
            // Keep the head family fixed; a stage insertion in the caller
            // supplies the structural change for parameterless heads.
        }
    }
}

fn tune_transform(stage: &mut Stage, rng: &mut Pcg32) {
    match stage.op.as_str() {
        "quantile_bins" => {
            let values = [2, 4, 8, 16, 25, 32, 48, 64];
            stage.params.insert(
                "bins".to_string(),
                ParamValue::Int(values[rng.next_u32() as usize % values.len()]),
            );
        }
        "pair_prod" => {
            stage.params.insert(
                "topk".to_string(),
                ParamValue::Int(4 + (rng.next_u32() % 21) as i64),
            );
        }
        "winsor_clip" => {
            stage.params.insert(
                "q".to_string(),
                ParamValue::Float(0.005 + 0.245 * rng.next_f64()),
            );
        }
        "cheb" => {
            stage.params.insert(
                "degree".to_string(),
                ParamValue::Int(1 + (rng.next_u32() % 8) as i64),
            );
        }
        "hinge" => {
            stage.params.insert(
                "knots".to_string(),
                ParamValue::Int(2 + (rng.next_u32() % 7) as i64),
            );
        }
        _ => {
            let append = stage.params.get("append") != Some(&ParamValue::Bool(true));
            stage
                .params
                .insert("append".to_string(), ParamValue::Bool(append));
        }
    }
}

/// Produce two elites plus eighteen tournament-3 descendants per island.
/// The optional stagnation counter triggers deterministic grammar immigrants.
pub fn next_generation(
    config: &V3EvolutionConfig,
    generation: usize,
    scored: &[V3ScoredCandidate],
    stagnation: &[usize],
) -> Result<Vec<V3Candidate>> {
    config.validate()?;
    if scored.len() != config.population() || stagnation.len() != config.islands {
        bail!("scored population or stagnation vector has the wrong size");
    }
    let mut by_island = BTreeMap::<usize, Vec<V3ScoredCandidate>>::new();
    for item in scored {
        let island = item.candidate.island as usize;
        if island >= config.islands || !item.rmse.is_finite() {
            bail!("scored candidate has an invalid island or RMSE");
        }
        by_island.entry(island).or_default().push(item.clone());
    }
    for island in 0..config.islands {
        if by_island.get(&island).map_or(0, Vec::len) != config.island_size {
            bail!("island {island} does not contain exactly twenty candidates");
        }
    }
    let ranked_islands: Vec<Vec<V3ScoredCandidate>> = (0..config.islands)
        .map(|island| ranked(by_island.remove(&island).expect("validated island")))
        .collect();
    let mut output = Vec::with_capacity(config.population());
    let mut seen = BTreeSet::new();
    for island in 0..config.islands {
        let pool = &ranked_islands[island];
        for elite in pool.iter().take(ELITES) {
            let mut candidate = elite.candidate.clone();
            candidate.generation = generation;
            candidate.operator = "elite".to_string();
            candidate.island = island as u8;
            seen.insert(candidate.semantic_hash.clone());
            output.push(candidate);
        }
        let mut rng = Pcg32::new(
            config.seed ^ generation as u64,
            0x4953_4C41_4E44_0000 ^ island as u64,
        );
        let descendants = config.island_size - ELITES;
        for slot in 0..descendants {
            let seed = ((rng.next_u32() as u64) << 32) | rng.next_u32() as u64;
            let mut child = if stagnation[island] >= config.stagnation_generations && slot >= 9 {
                make_candidate(
                    &baseline(island, generation * descendants + slot, seed),
                    island,
                    generation,
                    "stagnation_reseed",
                    250_000,
                    &[&pool[0].candidate],
                )?
            } else if generation.is_multiple_of(config.migration_interval) && slot == 0 {
                let migrant =
                    &ranked_islands[(island + config.islands - 1) % config.islands][0].candidate;
                mutate_candidate(migrant, seed, generation, island)?
            } else {
                let left = tournament(pool, &mut rng);
                if rng.next_u32() % 100 < 35 {
                    let right = tournament(pool, &mut rng);
                    crossover_candidates(
                        &left.candidate,
                        &right.candidate,
                        seed,
                        generation,
                        island,
                    )?
                } else {
                    mutate_candidate(&left.candidate, seed, generation, island)?
                }
            };
            if !seen.insert(child.semantic_hash.clone()) {
                child = make_candidate(
                    &baseline(island, generation * 10_000 + slot, seed),
                    island,
                    generation,
                    "semantic_dedup_reseed",
                    child.step_micros,
                    &[&pool[0].candidate],
                )?;
                if !seen.insert(child.semantic_hash.clone()) {
                    bail!("deterministic semantic deduplication produced a collision");
                }
            }
            output.push(child);
        }
    }
    Ok(output)
}

impl V3Archive {
    pub fn update(&mut self, candidates: impl IntoIterator<Item = V3ScoredCandidate>) {
        let mut by_hash = BTreeMap::<String, V3ScoredCandidate>::new();
        for candidate in self.entries.drain(..).chain(candidates) {
            by_hash
                .entry(candidate.candidate.semantic_hash.clone())
                .and_modify(|present| {
                    if score_order(&candidate, present) == Ordering::Less {
                        *present = candidate.clone();
                    }
                })
                .or_insert(candidate);
        }
        let mut entries: Vec<_> = by_hash.into_values().collect();
        entries.sort_by(|left, right| {
            score_order(left, right).then_with(|| right.novelty.total_cmp(&left.novelty))
        });
        entries.truncate(ARCHIVE_LIMIT);
        self.entries = entries;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_population_is_complete_unique_and_reproducible() {
        let config = V3EvolutionConfig::default();
        let first = initial_population(&config, &[]);
        let second = initial_population(&config, &[]);
        assert_eq!(first, second);
        assert_eq!(first.len(), 80);
        assert_eq!(
            first
                .iter()
                .map(|candidate| &candidate.semantic_hash)
                .collect::<BTreeSet<_>>()
                .len(),
            80
        );
        for island in 0..4 {
            assert_eq!(
                first
                    .iter()
                    .filter(|candidate| candidate.island as usize == island)
                    .count(),
                20
            );
        }
    }

    #[test]
    fn one_hundred_thousand_typed_children_compile_without_retry() {
        let config = V3EvolutionConfig::default();
        let population = initial_population(&config, &[]);
        for index in 0..100_000usize {
            let left = &population[index % population.len()];
            let right = &population[(index * 37 + 11) % population.len()];
            let child = if index & 1 == 0 {
                mutate_candidate(left, index as u64, 1 + index / 80, left.island as usize)
            } else {
                crossover_candidates(
                    left,
                    right,
                    index as u64,
                    1 + index / 80,
                    left.island as usize,
                )
            }
            .unwrap();
            assert_eq!(
                GenomeIr::parse(&child.syntax).unwrap().canonical(),
                child.syntax
            );
        }
    }

    #[test]
    fn rankings_and_next_population_are_input_order_independent() {
        let config = V3EvolutionConfig::default();
        let population = initial_population(&config, &[]);
        let scored = population
            .into_iter()
            .enumerate()
            .map(|(index, candidate)| V3ScoredCandidate {
                candidate,
                rmse: 0.1 + index as f64 * 1e-6,
                novelty: (index % 7) as f64,
            })
            .collect::<Vec<_>>();
        let mut reversed = scored.clone();
        reversed.reverse();
        assert_eq!(
            next_generation(&config, 1, &scored, &[0; 4]).unwrap(),
            next_generation(&config, 1, &reversed, &[0; 4]).unwrap()
        );
    }
}
