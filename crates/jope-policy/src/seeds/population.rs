use crate::context::PolicyContext;
use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::path::Path;

use super::config::{ChallengerIdentity, CheckpointBinding};
use super::decode::{self, PurePool, SamplingOptions, EXPLORATION_SIZE, POOL_SIZE};
use super::grammar::BoundVocabulary;
use super::model::PolicyModel;
use super::proposal_ledger::{self, ProposalLedger};
use super::ranking::{self, Bindings, Ranking, SELECTION_ID};
use super::{
    context_sha256, is_sha256_hex, refuse_rejected_checkpoint, require_accepted_policy,
    require_accepted_vocabulary, sha256_hex, CHECKPOINT_VOCABULARY_SHA256,
};

/// Schema identifier accepted by jope-dime's `search --seeds` reader.
pub const SEED_SCHEMA: &str = "parallel.seed_population.v1";
/// Number of candidates written: the ranked pool plus grammar-uniform exploration tail.
pub const SEED_COUNT: usize = POOL_SIZE + EXPLORATION_SIZE;
/// Origin label for a candidate sampled from the neural policy.
pub const ORIGIN_NEURAL: &str = "neural_policy";
/// Origin label for a pool candidate drawn by the grammar-uniform fallback.
pub const ORIGIN_NEURAL_FALLBACK: &str = "neural_rejection_fallback";
/// Origin label for a candidate drawn for random exploration.
pub const ORIGIN_RANDOM: &str = "random_exploration";

/// One seed. Fields are private; candidates are only created by sampling and the grammar-uniform
/// generators in this module tree.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeedCandidate {
    pub(super) genome: String,
    pub(super) origin: String,
    /// Normalized critic rank weight. Historical files called it `policy_logprob`; it is never a
    /// behavior/selection propensity.
    #[serde(
        default,
        rename = "rank_log_weight",
        alias = "policy_logprob",
        skip_serializing_if = "Option::is_none"
    )]
    pub(super) rank_log_weight: Option<f64>,
    /// W4-T07: identity of this slot's record in the proposal ledger
    /// ([`ProposalLedger::slot_record_sha256`]): the draw it came from, its original tokens,
    /// scores, parser result and adaptations. Absent in files written before the join; a
    /// population names it for every slot or for none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) proposal_record_sha256: Option<String>,
}

impl SeedCandidate {
    pub(crate) fn new(genome: impl Into<String>, origin: &str) -> Self {
        Self {
            genome: genome.into(),
            origin: origin.into(),
            rank_log_weight: None,
            proposal_record_sha256: None,
        }
    }

    /// The proposal-ledger record this slot joins (W4-T07); `None` for an older file.
    pub fn proposal_record_sha256(&self) -> Option<&str> {
        self.proposal_record_sha256.as_deref()
    }

    pub fn genome(&self) -> &str {
        &self.genome
    }

    pub fn origin(&self) -> &str {
        &self.origin
    }

    pub fn rank_log_weight(&self) -> Option<f64> {
        self.rank_log_weight
    }

    pub fn into_genome(self) -> String {
        self.genome
    }
}

/// Field order matches jope-dime's struct, so pretty JSON output has the same key order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(super) struct PopulationFields {
    schema: String,
    prepared_sha256: String,
    cut: usize,
    training_sha256: String,
    pub(super) context_sha256: String,
    checkpoint_sha256: String,
    pub(super) candidates: Vec<SeedCandidate>,
}

/// Read accessors shared by [`SeedPopulation`] and [`SeedFile`].
macro_rules! population_accessors {
    () => {
        pub fn schema(&self) -> &str {
            &self.0.schema
        }

        pub fn prepared_sha256(&self) -> &str {
            &self.0.prepared_sha256
        }

        pub fn cut(&self) -> usize {
            self.0.cut
        }

        pub fn training_sha256(&self) -> &str {
            &self.0.training_sha256
        }

        pub fn context_sha256(&self) -> &str {
            &self.0.context_sha256
        }

        pub fn checkpoint_sha256(&self) -> &str {
            &self.0.checkpoint_sha256
        }

        pub fn candidates(&self) -> &[SeedCandidate] {
            &self.0.candidates
        }

        /// Origin label → count.
        pub fn origins(&self) -> BTreeMap<String, usize> {
            let mut out = BTreeMap::new();
            for c in &self.0.candidates {
                *out.entry(c.origin.clone()).or_insert(0) += 1;
            }
            out
        }

        /// jope-dime `search` sets `neural_seeded` from this label alone; it is not proof of use.
        pub fn claims_neural(&self) -> bool {
            self.0.candidates.iter().any(|c| c.origin == ORIGIN_NEURAL)
        }
    };
}

/// A population produced in this process by [`propose`] or [`propose_challenger`]. It has no
/// public constructor and no `Deserialize` implementation.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SeedPopulation(pub(super) PopulationFields);

impl SeedPopulation {
    population_accessors!();
}

/// A seed file as admitted by [`validate_for_search`]: what the file claims, not proof of how it
/// was produced. No writer accepts it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(transparent)]
pub struct SeedFile(PopulationFields);

impl SeedFile {
    population_accessors!();
}

impl PartialEq<SeedPopulation> for SeedFile {
    fn eq(&self, other: &SeedPopulation) -> bool {
        self.0 == other.0
    }
}

/// The run identity a seed file is bound to. The checkpoint state is not part of it: it is taken
/// from the [`CheckpointBinding`] that [`propose`] or [`propose_challenger`] verifies against the
/// loaded policy.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeedBinding {
    prepared_sha256: String,
    cut: usize,
    training_sha256: String,
    context_sha256: String,
}

impl SeedBinding {
    /// `prepared_sha256`: sha256 of the run's `prepared.json`; `training_sha256`: sha256 of the
    /// cut's training table; `context_sha256`: sha256 of the 8192-byte conditioning context.
    pub fn new(
        prepared_sha256: &str,
        cut: usize,
        training_sha256: &str,
        context_sha256: &str,
    ) -> Result<Self> {
        for (name, value) in [
            ("prepared_sha256", prepared_sha256),
            ("training_sha256", training_sha256),
            ("context_sha256", context_sha256),
        ] {
            ensure!(is_sha256_hex(value), "seed binding {name} is not a sha256");
        }
        Ok(Self {
            prepared_sha256: prepared_sha256.to_string(),
            cut,
            training_sha256: training_sha256.to_string(),
            context_sha256: context_sha256.to_string(),
        })
    }

    pub fn prepared_sha256(&self) -> &str {
        &self.prepared_sha256
    }

    pub fn cut(&self) -> usize {
        self.cut
    }

    pub fn training_sha256(&self) -> &str {
        &self.training_sha256
    }

    pub fn context_sha256(&self) -> &str {
        &self.context_sha256
    }
}

/// jope-dime `eval_registry::logprob_from_scores`.
pub fn logprob_from_scores(scores: &[f64]) -> Result<Vec<f64>> {
    ensure!(!scores.is_empty(), "empty ranking scores");
    ensure!(
        scores.iter().all(|s| s.is_finite()),
        "nonfinite ranking score"
    );
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let log_z = scores.iter().map(|s| (s - max).exp()).sum::<f64>().ln();
    Ok(scores.iter().map(|s| s - max - log_z).collect())
}

/// jope-dime `gp::attach_policy_logprobs`: weights follow ranked genomes, never seed order.
fn attach_rank_log_weights(candidates: &mut [SeedCandidate], scores: &[f64]) -> Result<()> {
    ensure!(candidates.len() == scores.len(), "rank weight alignment");
    for (candidate, weight) in candidates.iter_mut().zip(logprob_from_scores(scores)?) {
        candidate.rank_log_weight = Some(weight);
    }
    Ok(())
}

/// Assemble the 80-candidate population from a ranked 64-program pool and 16 exploration
/// programs, refusing any mismatch between the ranking, the pool and the binding. Private: the
/// caller must already have proven that `checkpoint_sha256` is the state of the policy that
/// produced `pool` and `ranking`.
pub(super) fn assemble_population(
    binding: &SeedBinding,
    checkpoint_sha256: &str,
    ranking: &Ranking,
    pool: &[SeedCandidate],
    exploration: &[SeedCandidate],
) -> Result<SeedPopulation> {
    ensure!(
        is_sha256_hex(checkpoint_sha256),
        "seed binding checkpoint_sha256 is not a sha256"
    );
    refuse_rejected_checkpoint(checkpoint_sha256, "")?;
    let bindings = ranking.bindings();
    ensure!(
        ranking.selection_id() == SELECTION_ID
            && bindings.checkpoint_sha256() == checkpoint_sha256
            && bindings.conditioning_context_sha256() == binding.context_sha256
            && ranking.pool_size() == POOL_SIZE
            && ranking.chosen_count() == POOL_SIZE
            && ranking.ranked().len() == POOL_SIZE
            && ranking.training_context_only()
            && !ranking.heldout_labels_read()
            && !ranking.gp_scores_read(),
        "ranking identity does not match the seed binding"
    );
    ensure!(
        pool.len() == POOL_SIZE
            && pool
                .iter()
                .all(|c| c.origin == ORIGIN_NEURAL || c.origin == ORIGIN_NEURAL_FALLBACK),
        "pool must hold 64 neural-arm candidates"
    );
    let mut by_genome = pool
        .iter()
        .map(|c| (c.genome.as_str(), c))
        .collect::<BTreeMap<_, _>>();
    ensure!(by_genome.len() == POOL_SIZE, "pool is not unique");
    let mut candidates = ranking
        .selected()
        .iter()
        .map(|p| {
            by_genome
                .remove(p.program.as_str())
                .cloned()
                .context("ranked program absent from the pool")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(by_genome.is_empty(), "pool identity changed while ranking");
    let scores = ranking
        .selected()
        .iter()
        .map(|p| p.predicted_quality)
        .collect::<Vec<_>>();
    attach_rank_log_weights(&mut candidates, &scores)?;
    ensure!(
        exploration.len() == EXPLORATION_SIZE
            && exploration
                .iter()
                .all(|c| c.origin == ORIGIN_RANDOM && c.rank_log_weight.is_none()),
        "exploration tail must hold 16 unscored random_exploration candidates"
    );
    candidates.extend(exploration.iter().cloned());
    let population = PopulationFields {
        schema: SEED_SCHEMA.into(),
        prepared_sha256: binding.prepared_sha256.clone(),
        cut: binding.cut,
        training_sha256: binding.training_sha256.clone(),
        context_sha256: binding.context_sha256.clone(),
        checkpoint_sha256: checkpoint_sha256.to_string(),
        candidates,
    };
    check_population(&population)?;
    Ok(SeedPopulation(population))
}

/// Test support (`testing`): the private assembler, for any state except the accepted
/// one. The result cannot be written ([`write_population`] takes a [`Proposal`]).
#[cfg(any(test, feature = "testing"))]
pub fn assemble_for_testing(
    binding: &SeedBinding,
    checkpoint_sha256: &str,
    ranking: &Ranking,
    pool: &[SeedCandidate],
    exploration: &[SeedCandidate],
) -> Result<SeedPopulation> {
    ensure!(
        !super::is_accepted_state(checkpoint_sha256)
            && !super::is_accepted_state(ranking.bindings().checkpoint_sha256()),
        "refused: the test assembler never stamps the accepted state"
    );
    assemble_population(binding, checkpoint_sha256, ranking, pool, exploration)
}

/// Structural checks shared by the writer and the reader.
pub(super) fn check_population(population: &PopulationFields) -> Result<()> {
    ensure!(
        population.schema == SEED_SCHEMA,
        "seed population schema is not {SEED_SCHEMA}"
    );
    ensure!(
        population.candidates.len() == SEED_COUNT,
        "seed population must hold exactly {SEED_COUNT} candidates, got {}",
        population.candidates.len()
    );
    let unique = population
        .candidates
        .iter()
        .map(|c| c.genome.as_str())
        .collect::<BTreeSet<_>>();
    ensure!(unique.len() == SEED_COUNT, "duplicate initial proposals");
    for candidate in &population.candidates {
        feat_math::GenomeIr::parse(&candidate.genome)
            .with_context(|| format!("seed genome does not parse: {}", candidate.genome))?;
        ensure!(
            candidate.rank_log_weight.is_none_or(f64::is_finite),
            "nonfinite seed rank weight"
        );
        ensure!(
            candidate
                .proposal_record_sha256
                .as_deref()
                .is_none_or(is_sha256_hex),
            "seed proposal record is not a sha256"
        );
    }
    let joined = population
        .candidates
        .iter()
        .filter(|c| c.proposal_record_sha256.is_some())
        .count();
    ensure!(
        joined == 0 || joined == population.candidates.len(),
        "seed population joins only some slots to proposal records"
    );
    Ok(())
}

/// The bytes jope-dime writes (`serde_json::to_vec_pretty`) for a proposal's population.
pub fn population_bytes(proposal: &Proposal) -> Result<Vec<u8>> {
    check_population(&proposal.population.0)?;
    Ok(serde_json::to_vec_pretty(&proposal.population)?)
}

/// Write the proposal's population atomically; returns its sha256 (the search's
/// `seeds_sha256`).
pub fn write_population(path: &Path, proposal: &Proposal) -> Result<String> {
    let bytes = population_bytes(proposal)?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = path
        .file_name()
        .context("seed path has no file name")?
        .to_string_lossy();
    let temporary = parent.join(format!(".{name}.{}.tmp", std::process::id()));
    {
        let mut file = std::fs::File::create(&temporary)
            .with_context(|| format!("creating {}", temporary.display()))?;
        file.write_all(&bytes)?;
        file.sync_all()?;
    }
    std::fs::rename(&temporary, path).with_context(|| format!("publishing {}", path.display()))?;
    Ok(sha256_hex(&bytes))
}

/// The admission jope-dime `search --seeds` applies (`gp.rs:1364-1375` plus its duplicate
/// check), for a file about to be handed to the search child.
pub fn validate_for_search(
    bytes: &[u8],
    prepared_sha256: &str,
    cut: usize,
    training_sha256: &str,
) -> Result<SeedFile> {
    let population: PopulationFields =
        serde_json::from_slice(bytes).context("seed population json")?;
    ensure!(
        population.schema == SEED_SCHEMA
            && population.prepared_sha256 == prepared_sha256
            && population.cut == cut
            && population.training_sha256 == training_sha256
            && population.candidates.len() == SEED_COUNT,
        "neural population identity mismatch"
    );
    check_population(&population)?;
    refuse_rejected_checkpoint(&population.checkpoint_sha256, "")?;
    Ok(SeedFile(population))
}
