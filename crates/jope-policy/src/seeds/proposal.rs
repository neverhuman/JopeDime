use crate::config::{ChallengerIdentity, CheckpointBinding};
use crate::context::PolicyContext;
use crate::decode::{self, PurePool, SamplingOptions};
use crate::grammar::BoundVocabulary;
use crate::model::PolicyModel;
use crate::proposal_ledger::{self, ProposalLedger};
use crate::ranking::{self, Bindings, Ranking};
use crate::{
    context_sha256, refuse_rejected_checkpoint, require_accepted_policy,
    require_accepted_vocabulary, CHECKPOINT_VOCABULARY_SHA256,
};
use anyhow::{ensure, Context, Result};

use super::population::{assemble_population, check_population, SeedBinding, SeedPopulation};

/// Everything a proposal run produced, for the caller's receipt. Fields are private: only
/// [`propose`] and [`propose_challenger`] (and `propose_unbound` with the `testing` feature)
/// construct it.
#[derive(Clone, Debug)]
pub struct Proposal {
    pub(super) population: SeedPopulation,
    pool: PurePool,
    ranking: Ranking,
    ledger: ProposalLedger,
}

impl Proposal {
    /// Return the seed population assembled for this proposal.
    pub fn population(&self) -> &SeedPopulation {
        &self.population
    }

    /// Return the sampled policy pool before exploration candidates are appended.
    pub fn pool(&self) -> &PurePool {
        &self.pool
    }

    /// Return the deterministic ranking of the sampled pool.
    pub fn ranking(&self) -> &Ranking {
        &self.ranking
    }

    /// Identity and reconciled denominators of this proposal (W4-T07), recorded while it ran.
    pub fn ledger(&self) -> &ProposalLedger {
        &self.ledger
    }
}

/// Seeds for the three independent random streams of one proposal.
#[derive(Clone, Debug)]
pub struct ProposalSeeds {
    /// Seed for neural-policy sampling.
    pub sampling: u64,
    /// Seed for grammar-uniform pool fallback.
    pub fallback: u64,
    /// Seed for the exploration tail.
    pub exploration: u64,
}

impl ProposalSeeds {
    /// Derive independent sampling, fallback, and exploration streams from one run seed.
    pub fn derive(seed: u64) -> Self {
        use crate::hashing::mix64;
        Self {
            sampling: mix64(seed),
            fallback: mix64(seed ^ 0x7075_7265_5f66_696c),
            exploration: mix64(seed ^ 0x6578_706c_6f72_6521),
        }
    }
}

/// Sample, rank and assemble the 80 seeds for one run.
///
/// The model must come from [`PolicyModel::load_bound`] with this `checkpoint`, and its
/// recomputed parameter hash must be the accepted step-60000 hash
/// ([`crate::ACCEPTED_PARAMETER_SHA256`]); any other model (including untrained weights) is
/// refused, so a file stamped with the accepted state was produced by the accepted weights. The
/// receipt must name the accepted configuration (claimed and recomputed hash equal to
/// [`crate::ACCEPTED_CONFIG_SHA256`]) and the step-60000 vocabulary
/// ([`crate::CHECKPOINT_VOCABULARY_SHA256`]). The vocabulary's own byte hash must be that
/// vocabulary or its documented product restriction ([`crate::PRODUCT_VOCABULARY_SHA256`]), and
/// the context must hash to `binding.context_sha256()`.
pub fn propose(
    model: &PolicyModel,
    checkpoint: &CheckpointBinding,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    refuse_rejected_checkpoint(checkpoint.state_sha256(), checkpoint.model_sha256())?;
    let bound = require_accepted_policy(model)?;
    ensure!(
        bound.state_sha256() == checkpoint.state_sha256()
            && bound.parameter_sha256() == checkpoint.parameter_sha256()
            && bound.policy_safetensors_sha256() == checkpoint.policy_safetensors_sha256(),
        "refused: loaded policy was bound to a different checkpoint"
    );
    checkpoint.require_accepted_identity()?;
    propose_checked(
        model, checkpoint, vocabulary, context, binding, seeds, sampling,
    )
}

/// Sample, rank and assemble the 80 seeds with a pinned native challenger: a checkpoint that
/// is not the accepted one, loaded through [`PolicyModel::load_bound`] with the challenger's
/// export receipt. The population is stamped with the challenger state. Everything after the
/// identity check is shared with [`propose`], so both arms of a comparison differ only in weights.
#[allow(clippy::too_many_arguments)]
pub fn propose_challenger(
    model: &PolicyModel,
    checkpoint: &CheckpointBinding,
    challenger: &ChallengerIdentity,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    refuse_rejected_checkpoint(checkpoint.state_sha256(), checkpoint.model_sha256())?;
    checkpoint.require_challenger_identity(challenger)?;
    let bound = model
        .bound_identity()
        .context("refused: policy was not loaded through PolicyModel::load_bound")?;
    ensure!(
        bound.state_sha256() == checkpoint.state_sha256()
            && bound.parameter_sha256() == checkpoint.parameter_sha256()
            && bound.policy_safetensors_sha256() == checkpoint.policy_safetensors_sha256()
            && bound.config_sha256() == checkpoint.config_sha256(),
        "refused: loaded policy was bound to a different checkpoint"
    );
    propose_checked(
        model, checkpoint, vocabulary, context, binding, seeds, sampling,
    )
}

/// Test support (`testing`): [`propose`] for untrained weights. Refuses any bound
/// model and any checkpoint other than
/// [`CheckpointBinding::unbound_for_testing`]'s, so its files are stamped with
/// [`crate::testing::UNBOUND_TEST_STATE_SHA256`] and never with a checkpoint state.
#[cfg(any(test, feature = "testing"))]
pub fn propose_unbound(
    model: &PolicyModel,
    checkpoint: &CheckpointBinding,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    ensure!(
        model.bound_identity().is_none()
            && checkpoint.state_sha256() == crate::testing::UNBOUND_TEST_STATE_SHA256,
        "refused: the unbound test path only accepts untrained weights and the test binding"
    );
    propose_checked(
        model, checkpoint, vocabulary, context, binding, seeds, sampling,
    )
}

/// Everything after the model identity check (accepted or challenger).
fn propose_checked(
    model: &PolicyModel,
    checkpoint: &CheckpointBinding,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    ensure!(
        model.config() == checkpoint.config(),
        "loaded policy configuration differs from the checkpoint"
    );
    // The checkpoint must name the step-60000 vocabulary, and the parsed bytes must hash to it or
    // to its documented product restriction; the label the caller bound against is not trusted.
    ensure!(
        checkpoint.vocabulary_sha256() == CHECKPOINT_VOCABULARY_SHA256,
        "refused: checkpoint vocabulary {} is not the step-60000 vocabulary",
        checkpoint.vocabulary_sha256()
    );
    require_accepted_vocabulary(vocabulary)?;
    ensure!(
        match vocabulary.identity() {
            crate::grammar::VocabularyIdentity::Checkpoint =>
                vocabulary.sha256() == checkpoint.vocabulary_sha256(),
            crate::grammar::VocabularyIdentity::ProductRestriction { base_sha256 } =>
                base_sha256 == checkpoint.vocabulary_sha256(),
        },
        "vocabulary is not bound to this checkpoint"
    );
    ensure!(
        context_sha256(context) == binding.context_sha256(),
        "context bytes differ from the seed binding"
    );
    sample_rank_assemble(
        model,
        checkpoint.state_sha256(),
        vocabulary,
        context,
        binding,
        seeds,
        sampling,
    )
}

/// Test support (unit tests only): the sampling, ranking and assembly of [`propose`] under a
/// vocabulary the product path refuses, so the ledger's parser-refusal branch is exercised by
/// draws that were really refused. Untrained, unbound weights and the test state only.
#[cfg(test)]
pub(crate) fn propose_with_vocabulary_for_test(
    model: &PolicyModel,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    ensure!(
        model.bound_identity().is_none(),
        "refused: the unbound test path only accepts untrained weights"
    );
    sample_rank_assemble(
        model,
        crate::testing::UNBOUND_TEST_STATE_SHA256,
        vocabulary,
        context,
        binding,
        seeds,
        sampling,
    )
}

/// Sample, rank, assemble and record, after every identity check of the callers above.
fn sample_rank_assemble(
    model: &PolicyModel,
    state_sha256: &str,
    vocabulary: &BoundVocabulary,
    context: &PolicyContext,
    binding: &SeedBinding,
    seeds: &ProposalSeeds,
    sampling: &SamplingOptions,
) -> Result<Proposal> {
    let task = "regression";
    let vocab = vocabulary.vocab();
    let (pool, mut seen) = decode::pure_pool_checked(
        model,
        vocab,
        context,
        task,
        seeds.sampling,
        seeds.fallback,
        sampling,
    )?;
    let programs = pool
        .candidates()
        .iter()
        .map(|c| c.genome.clone())
        .collect::<Vec<_>>();
    // The checkpoint state was verified against the model by the caller.
    let ranking = ranking::score_pool(
        model,
        vocab,
        context,
        &programs,
        Bindings::unchecked(state_sha256, binding.context_sha256(), vocabulary.sha256())
            .with_vocabulary(vocabulary),
        &ranking::Options::default(),
    )?;
    let (exploration, exploration_draws) =
        decode::random_exploration_recorded(vocab, task, seeds.exploration, &mut seen)?;
    let mut population = assemble_population(
        binding,
        state_sha256,
        &ranking,
        pool.candidates(),
        &exploration,
    )?;
    // Records only: the ledger never changes which programs are proposed, their order, their
    // origins or their weights; it is reconciled against the vocabulary before it is returned.
    let ledger = proposal_ledger::build(&proposal_ledger::ProposalRun {
        task,
        checkpoint_sha256: state_sha256,
        context_sha256: binding.context_sha256(),
        vocabulary_sha256: vocabulary.sha256(),
        pool: &pool,
        exploration_seed: seeds.exploration,
        exploration_draws: &exploration_draws,
        ranking: &ranking,
        population: population.candidates(),
        vocab,
    })?;
    // W4-T07 join: every seed slot names its proposal record, so the candidate ledgers the
    // population seeds (the GP lineage envelope, the controller's candidates) join the draw,
    // tokens, scores and parser result each candidate came from.
    for (slot, candidate) in population.0.candidates.iter_mut().enumerate() {
        candidate.proposal_record_sha256 = Some(ledger.slot_record_sha256(slot)?);
    }
    check_population(&population.0)?;
    ledger.matches_population(population.candidates())?;
    Ok(Proposal {
        population,
        pool,
        ranking,
        ledger,
    })
}
