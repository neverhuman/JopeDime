//! CPU proposer policy, following the RES-04 retrieval/edit/synthesis schedule.
//!
//! RES-04 measured evaluated/adaptively admitted proposals, not unconditional
//! generator weights. This module never fits a sampler to those marginals or
//! claims M3 from an unmatched generated-only histogram. Raw-bank replay and
//! discovery-loop registration have separate integration/qualification work.

use super::landmark_pools::select_landmark_preferred_pools;
use anyhow::{bail, ensure, Context, Result};
use feat_math::dsl::{DslExpr, ParamValue, Stage};
use feat_math::evolve::api::{
    capability::{self, CapabilityKind, Language, LanguageFilter, Registry},
    parse_canonical_pipeline,
};
use feat_math::GenomeIr;
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha12Rng,
};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

const MAX_DSL_BYTES: usize = 16 * 1024;
const MAX_PARENTS: usize = 48;
const RAW_COORDINATES: &str = "raw_svd_lm48_first12";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BankRole {
    Proposer,
    ModelPrior,
}

#[derive(Clone, Debug)]
pub struct BankContract {
    pub role: BankRole,
    pub coordinate_space: String,
    /// Identity of the admitted roster AND promotion epoch, verified by the provider.
    pub snapshot_sha256: String,
}

#[derive(Clone, Debug)]
pub struct Retrieved {
    pub program: Program,
    pub family_id: String,
    pub dl_train: bool,
    pub distance_squared: f64,
}

/// Until RNK-09 admits a pinned raw snapshot, the provider must return `None`.
/// Never adapt the normalized model-prior bank to this interface.
pub trait ChampionBank {
    fn contract(&self) -> Option<BankContract>;

    /// Return at most `limit` TRAIN records from other families, ordered by
    /// (float64 squared distance, original program SHA), stably for exact ties.
    /// Distances use the raw first twelve SVD coordinates without normalization.
    fn nearest(
        &self,
        raw_lm12: &[f64; 12],
        excluded_family: &str,
        limit: usize,
    ) -> Result<Vec<Retrieved>>;
}

pub struct NoBank;

impl ChampionBank for NoBank {
    fn contract(&self) -> Option<BankContract> {
        None
    }

    fn nearest(&self, _: &[f64; 12], _: &str, _: usize) -> Result<Vec<Retrieved>> {
        bail!("no admitted raw proposer snapshot")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProposalFamily {
    Retrieval,
    Parameter,
    ReplaceHead,
    Insert,
    Delete,
    Crossover,
    Synthesis,
}

impl ProposalFamily {
    pub fn name(self) -> &'static str {
        match self {
            Self::Retrieval => "champion_retrieval",
            Self::Parameter => "edit_parameter",
            Self::ReplaceHead => "edit_replace_head",
            Self::Insert => "edit_insert_stage",
            Self::Delete => "edit_delete_stage",
            Self::Crossover => "edit_crossover",
            Self::Synthesis => "landmark_synthesis",
        }
    }

    fn edit(step: usize) -> Self {
        [
            Self::Parameter,
            Self::ReplaceHead,
            Self::Insert,
            Self::Delete,
            Self::Crossover,
        ][step % 5]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proposal {
    pub program: Program,
    pub memo_key: String,
    pub family: ProposalFamily,
    pub parents: Vec<String>,
}

pub struct Round<'a> {
    pub seed: u64,
    pub round: u64,
    pub exploring: bool,
    pub target_family: &'a str,
    /// Raw SVD lm48, before Stage1B standardization; never prior-bank summaries.
    pub raw_lm48: &'a [f64; 48],
    /// Already ordered by the caller's committed parent policy.
    pub parents: &'a [Program],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Batch {
    pub proposals: Vec<Proposal>,
    pub requested: usize,
    pub attempts: BTreeMap<ProposalFamily, usize>,
    /// Attempts with no eligible output, including inapplicable edit arms.
    pub invalid_or_filtered: usize,
    pub duplicates_or_memo: usize,
    pub synthesis_exhausted: bool,
    pub raw_bank_available: bool,
}

/// Explicit endian encoding, independent of worker count, globals and call order.
pub fn round_rng(seed: u64, round: u64) -> ChaCha12Rng {
    let mut hash = Sha256::new();
    hash.update(seed.to_le_bytes());
    hash.update(round.to_le_bytes());
    ChaCha12Rng::from_seed(hash.finalize().into())
}

mod edits;
mod program;

use edits::{assemble, delete_stage, head_retry_order, index, mutate_parameters};
#[cfg(test)]
use program::digest;
use program::hash_valid;
pub use program::{memo_key, Program};

pub struct Proposer {
    filter: LanguageFilter,
    heads: BTreeSet<String>,
}

impl Proposer {
    pub fn new(languages: &[Language]) -> Result<Self> {
        let filter = capability::allowed_ops(languages)?;
        let heads = Registry::current()
            .entries
            .into_iter()
            .filter(|row| row.kind == CapabilityKind::Head)
            .map(|row| row.name.to_string())
            .collect();
        Ok(Self { filter, heads })
    }

    fn checked_pipeline(&self, program: &Program) -> Result<DslExpr> {
        ensure!(
            program.dsl.len() <= MAX_DSL_BYTES,
            "proposal DSL exceeds byte limit"
        );
        ensure!(
            program.program_sha256 == digest(program.dsl.as_bytes()),
            "program source hash mismatch"
        );
        let pipeline = parse_canonical_pipeline(&program.dsl).map_err(anyhow::Error::msg)?;
        ensure!(
            pipeline
                .stages
                .iter()
                .flat_map(|stage| stage.params.values())
                .all(|value| { !matches!(value, ParamValue::Float(v) if !v.is_finite()) }),
            "proposal parameters must be finite"
        );
        ensure!(
            pipeline
                .stages
                .last()
                .is_some_and(|stage| self.heads.contains(&stage.op)),
            "proposal requires a supported head"
        );
        Ok(pipeline)
    }

    fn pipeline(&self, program: &Program) -> Result<DslExpr> {
        let pipeline = self.checked_pipeline(program)?;
        ensure!(
            self.filter
                .accepts_seed(&GenomeIr::Pipeline(pipeline.clone())),
            "proposal is excluded by the language capability filter"
        );
        Ok(pipeline)
    }

    fn offer(
        &self,
        dsl: String,
        family: ProposalFamily,
        parents: Vec<String>,
        memo: &BTreeSet<String>,
        seen: &mut BTreeSet<String>,
        batch: &mut Batch,
    ) {
        let program = Program::from_dsl(dsl);
        let Ok(pipeline) = self.pipeline(&program) else {
            batch.invalid_or_filtered += 1;
            return;
        };
        let key = digest(GenomeIr::Pipeline(pipeline).canonical().as_bytes());
        if memo.contains(&key) || !seen.insert(key.clone()) {
            batch.duplicates_or_memo += 1;
            return;
        }
        batch.proposals.push(Proposal {
            program,
            memo_key: key,
            family,
            parents,
        });
    }

    /// Generate8 or16 proposals. Admission/evaluation of at most4 is a separate policy.
    pub fn propose(
        &self,
        round: &Round<'_>,
        bank: &dyn ChampionBank,
        memo: &BTreeSet<String>,
    ) -> Result<Batch> {
        ensure!(!round.target_family.is_empty(), "target family is required");
        ensure!(
            round.raw_lm48.iter().all(|v| v.is_finite()),
            "raw lm48 must be finite"
        );
        ensure!(round.parents.len() <= MAX_PARENTS, "parent pool exceeds48");
        let width = if round.exploring { 16 } else { 8 };
        let mut rng = round_rng(round.seed, round.round);
        let mut seen = BTreeSet::new();
        for parent in round.parents {
            self.pipeline(parent).context("invalid parent")?;
            seen.insert(memo_key(&parent.dsl)?);
        }
        let mut parents = round.parents.to_vec();
        let mut batch = Batch {
            proposals: Vec::new(),
            requested: width,
            attempts: BTreeMap::new(),
            invalid_or_filtered: 0,
            duplicates_or_memo: 0,
            synthesis_exhausted: false,
            raw_bank_available: false,
        };
        if let Some(contract) = bank.contract() {
            ensure!(
                contract.role == BankRole::Proposer
                    && contract.coordinate_space == RAW_COORDINATES
                    && hash_valid(&contract.snapshot_sha256),
                "admitted raw proposer snapshot required; model_prior is not retrieval input"
            );
            batch.raw_bank_available = true;
            let quota = 4.min(width / 3);
            let mut raw12 = [0.0; 12];
            raw12.copy_from_slice(&round.raw_lm48[..12]);
            let retrieved = bank.nearest(&raw12, round.target_family, quota)?;
            ensure!(retrieved.len() <= quota, "bank exceeded retrieval quota");
            let mut previous: Option<(f64, &str)> = None;
            for record in &retrieved {
                ensure!(
                    record.dl_train
                        && !record.family_id.is_empty()
                        && record.family_id != round.target_family,
                    "retrieval requires another DL-Train family"
                );
                ensure!(
                    record.distance_squared.is_finite() && record.distance_squared >= 0.0,
                    "invalid retrieval distance"
                );
                if let Some((distance, hash)) = previous {
                    ensure!(
                        record.distance_squared > distance
                            || (record.distance_squared == distance
                                && record.program.program_sha256.as_str() >= hash),
                        "bank nearest order differs from (distance, program hash)"
                    );
                }
                previous = Some((record.distance_squared, &record.program.program_sha256));
                *batch.attempts.entry(ProposalFamily::Retrieval).or_default() += 1;
                let checked = self
                    .checked_pipeline(&record.program)
                    .context("invalid retrieved program")?;
                let usable_parent = self.filter.accepts_seed(&GenomeIr::Pipeline(checked));
                self.offer(
                    record.program.dsl.clone(),
                    ProposalFamily::Retrieval,
                    vec![record.program.program_sha256.clone()],
                    memo,
                    &mut seen,
                    &mut batch,
                );
                if parents.is_empty() && usable_parent {
                    parents.push(record.program.clone());
                }
            }
        }
        let (stage_templates, head_templates) = select_landmark_preferred_pools(round.raw_lm48);
        let pool_stage = |template: &str| -> Result<Stage> {
            let text = format!("(pipe X {template} (ridge))");
            let mut parsed = parse_canonical_pipeline(&text).map_err(anyhow::Error::msg)?;
            ensure!(parsed.stages.len() == 2, "invalid preferred stage template");
            Ok(parsed.stages.remove(0))
        };
        let stages: Vec<Stage> = stage_templates
            .into_iter()
            .map(pool_stage)
            .collect::<Result<Vec<_>>>()?;
        let heads: Vec<Stage> = head_templates
            .into_iter()
            .map(|template| {
                let mut parsed = parse_canonical_pipeline(&format!("(pipe X {template})"))
                    .map_err(anyhow::Error::msg)?;
                ensure!(parsed.stages.len() == 1, "invalid preferred head template");
                Ok(parsed.stages.remove(0))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            !stages.is_empty() && !heads.is_empty(),
            "empty preferred pools"
        );
        if !parents.is_empty() {
            for step in 0..3 * width {
                if batch.proposals.len() == width {
                    break;
                }
                let family = ProposalFamily::edit(step);
                *batch.attempts.entry(family).or_default() += 1;
                let parent = &parents[step % parents.len()];
                let mut parsed = self.pipeline(parent)?;
                let mut head = parsed.stages.pop().expect("validated head");
                let transforms = &mut parsed.stages;
                let mut provenance = vec![parent.program_sha256.clone()];
                let generated = match family {
                    ProposalFamily::Parameter => {
                        for stage in transforms.iter_mut() {
                            mutate_parameters(stage, &mut rng);
                        }
                        mutate_parameters(&mut head, &mut rng);
                        Some(assemble(parsed.stages, head))
                    }
                    ProposalFamily::ReplaceHead => {
                        let first = index(&mut rng, heads.len());
                        let before = batch.proposals.len();
                        let first_dsl = assemble(parsed.stages.clone(), heads[first].clone());
                        // An unchanged selected parent is a no-op, not a
                        // changed head that collided with a known program.
                        // Keep the frozen single rejected draw for that case.
                        let parent_key = memo_key(&parent.dsl)?;
                        let collision = memo_key(&first_dsl).is_ok_and(|key| {
                            key != parent_key && (seen.contains(&key) || memo.contains(&key))
                        });
                        let order = if collision {
                            head_retry_order(first, heads.len(), step, &rng)
                        } else {
                            vec![first]
                        };
                        // A genuine changed-head collision should not consume this typed
                        // mutation slot while a legal, novel head is available.
                        // Keep the initial weighted draw and RNG position; the
                        // finite retry order preserves the pool's weights.
                        for (offset, head_index) in order.into_iter().enumerate() {
                            if offset > 0 {
                                *batch.attempts.entry(family).or_default() += 1;
                            }
                            self.offer(
                                assemble(parsed.stages.clone(), heads[head_index].clone()),
                                family,
                                provenance.clone(),
                                memo,
                                &mut seen,
                                &mut batch,
                            );
                            if batch.proposals.len() > before {
                                break;
                            }
                        }
                        continue;
                    }
                    ProposalFamily::Insert => {
                        let inserted = stages[index(&mut rng, stages.len())].clone();
                        let position = index(&mut rng, transforms.len() + 1);
                        transforms.insert(position, inserted);
                        (transforms.len() <= 6).then(|| assemble(parsed.stages, head))
                    }
                    ProposalFamily::Delete => {
                        delete_stage(transforms, &mut rng).map(|_| assemble(parsed.stages, head))
                    }
                    ProposalFamily::Crossover if parents.len() >= 2 => {
                        let other = &parents[(step + 1) % parents.len()];
                        let mut other_parsed = self.pipeline(other)?;
                        let other_head = other_parsed.stages.pop().expect("validated head");
                        let left_cut = (transforms.len() / 2).max(1).min(transforms.len());
                        let mut cross = transforms[..left_cut].to_vec();
                        cross.extend_from_slice(
                            &other_parsed.stages[other_parsed.stages.len() / 2..],
                        );
                        cross.truncate(5);
                        provenance.push(other.program_sha256.clone());
                        Some(assemble(cross, other_head))
                    }
                    _ => None,
                };
                if let Some(dsl) = generated {
                    self.offer(dsl, family, provenance, memo, &mut seen, &mut batch);
                } else {
                    batch.invalid_or_filtered += 1;
                }
            }
        }
        // Filtering/memo can make the finite source intersection empty. Never
        // reproduce the historical infinite fallback loop in a service worker.
        for _ in 0..64 * width {
            if batch.proposals.len() == width {
                break;
            }
            *batch.attempts.entry(ProposalFamily::Synthesis).or_default() += 1;
            let count = index(&mut rng, 3) + 1;
            let chosen: Vec<Stage> = (0..count)
                .map(|_| {
                    let mut stage = stages[index(&mut rng, stages.len())].clone();
                    mutate_parameters(&mut stage, &mut rng);
                    stage
                })
                .collect();
            let mut head = heads[index(&mut rng, heads.len())].clone();
            mutate_parameters(&mut head, &mut rng);
            self.offer(
                assemble(chosen, head),
                ProposalFamily::Synthesis,
                vec![],
                memo,
                &mut seen,
                &mut batch,
            );
        }
        batch.synthesis_exhausted = batch.proposals.len() < width;
        Ok(batch)
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests;
