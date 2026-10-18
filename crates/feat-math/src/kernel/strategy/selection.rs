//! Stable candidate ordering and source-mixture accounting for modern GP.

use std::collections::BTreeMap;

/// One candidate's deterministic identity and measured score. Lower scores are better.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoredCandidate {
    pub proposal_sequence: u64,
    pub genome_id: String,
    pub score: f64,
}

/// Orders candidates by finite score, then proposal sequence, then genome identity.
/// Non-finite scores rank after finite scores and remain eligible only as ordinary candidates.
pub fn rank(candidates: &mut [ScoredCandidate]) {
    candidates.sort_by(|a, b| {
        let left = if a.score.is_finite() {
            a.score
        } else {
            f64::INFINITY
        };
        let right = if b.score.is_finite() {
            b.score
        } else {
            f64::INFINITY
        };
        left.total_cmp(&right)
            .then_with(|| a.proposal_sequence.cmp(&b.proposal_sequence))
            .then_with(|| a.genome_id.cmp(&b.genome_id))
    });
}

/// Requested and admitted source mass before deduplication.
#[derive(Clone, Debug, PartialEq)]
pub struct MixtureSource {
    pub source_id: String,
    pub requested_mass: u64,
    pub admitted_genomes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SourceMass {
    pub source_id: String,
    pub requested_mass: u64,
    pub admitted_mass: u64,
    pub realized_mass: u64,
    pub unique_genomes: usize,
    /// Unknown without realized draw counts; admission ratios are not behavior probabilities.
    pub behavior_probability: Option<f64>,
}

/// Reconstructs requested and admitted pre-dedup mass, then assigns each unique genome to its
/// first source in declaration order for realized post-dedup mass. Duplicate admission counts
/// remain visible through `admitted_mass` and [`multiplicities`].
pub fn account_mixture(sources: &[MixtureSource]) -> Vec<SourceMass> {
    let mut seen = std::collections::BTreeSet::new();
    sources
        .iter()
        .map(|source| {
            let unique_genomes = source
                .admitted_genomes
                .iter()
                .filter(|id| seen.insert(id.as_str()))
                .count();
            SourceMass {
                source_id: source.source_id.clone(),
                requested_mass: source.requested_mass,
                admitted_mass: source.admitted_mass(),
                realized_mass: unique_genomes as u64,
                unique_genomes,
                behavior_probability: None,
            }
        })
        .collect()
}

impl MixtureSource {
    fn admitted_mass(&self) -> u64 {
        self.admitted_genomes.len() as u64
    }
}

/// Deduplicates admissions deterministically while preserving each genome's source multiplicity.
pub fn multiplicities(sources: &[MixtureSource]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for source in sources {
        for genome in &source.admitted_genomes {
            *counts.entry(genome.clone()).or_insert(0) += 1;
        }
    }
    counts
}
