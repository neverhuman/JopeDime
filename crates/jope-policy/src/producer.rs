//! Private, lossless proposal transport across a pinned bounded worker boundary.
//! Reading this artifact proves consistency, not execution. The controller must
//! authenticate the pinned producer and its completed bounded execution/output
//! receipts before using the reader. `Proposal` remains non-deserializable.
use crate::context::PolicyContext;
use crate::{
    seeds::{self, Proposal, SeedFile},
    sha256_hex,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};

pub const SCHEMA: &str = "asm.private-policy-producer/v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub worker_source_sha256: String,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub cut: usize,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub policy_safetensors_sha256: String,
    pub vocabulary_sha256: String,
    pub sampling_seed: u64,
    pub fallback_seed: u64,
    pub exploration_seed: u64,
    pub sampling_rounds: usize,
    pub sampling_batch: usize,
    pub sampling_temperature: f32,
    /// The native challenger this request loads instead of the accepted checkpoint.
    /// Omitted when absent, so accepted request bytes and identities are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenger: Option<crate::config::ChallengerIdentity>,
    /// For a qualified CPU context: the digest of the controller-issued context admission pins
    /// (`context_producer::AdmissionPins::sha256`). The worker verifies the context bundle
    /// against these pins, never against pins the bundle chose. Omitted for a native context,
    /// so native request bytes and identities are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_admission_sha256: Option<String>,
}
/// All producer inputs known before encoding. The context output digest is
/// attached later and cannot change the search arm's identity or clock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPlan {
    pub worker_source_sha256: String,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub cut: usize,
    pub checkpoint_sha256: String,
    pub policy_safetensors_sha256: String,
    pub vocabulary_sha256: String,
    pub sampling_seed: u64,
    pub fallback_seed: u64,
    pub exploration_seed: u64,
    pub sampling_rounds: usize,
    pub sampling_batch: usize,
    pub sampling_temperature: f32,
    /// The native challenger this request loads instead of the accepted checkpoint.
    /// Omitted when absent, so accepted request bytes and identities are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub challenger: Option<crate::config::ChallengerIdentity>,
}
impl RequestPlan {
    /// The request for a native context: only the context digest is attached.
    pub fn request(&self, context_sha256: String) -> Request {
        self.request_with_admission(context_sha256, None)
    }
    /// The request for a qualified CPU context: the context digest and the digest of the
    /// controller-issued admission pins are attached after encoding.
    pub fn qualified_request(
        &self,
        context_sha256: String,
        context_admission_sha256: String,
    ) -> Request {
        self.request_with_admission(context_sha256, Some(context_admission_sha256))
    }
    /// This plan with exactly the given encoding outputs attached; everything else is the plan.
    pub fn request_with_admission(
        &self,
        context_sha256: String,
        context_admission_sha256: Option<String>,
    ) -> Request {
        Request {
            worker_source_sha256: self.worker_source_sha256.clone(),
            prepared_sha256: self.prepared_sha256.clone(),
            training_sha256: self.training_sha256.clone(),
            cut: self.cut,
            checkpoint_sha256: self.checkpoint_sha256.clone(),
            policy_safetensors_sha256: self.policy_safetensors_sha256.clone(),
            vocabulary_sha256: self.vocabulary_sha256.clone(),
            sampling_seed: self.sampling_seed,
            fallback_seed: self.fallback_seed,
            exploration_seed: self.exploration_seed,
            sampling_rounds: self.sampling_rounds,
            sampling_batch: self.sampling_batch,
            sampling_temperature: self.sampling_temperature,
            challenger: self.challenger.clone(),
            context_sha256,
            context_admission_sha256,
        }
    }
    pub fn validate(&self) -> Result<()> {
        self.request("0".repeat(64)).validate()
    }
}

impl Request {
    pub fn validate(&self) -> Result<()> {
        for digest in [
            &self.worker_source_sha256,
            &self.prepared_sha256,
            &self.training_sha256,
            &self.context_sha256,
            &self.checkpoint_sha256,
            &self.policy_safetensors_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(
                crate::is_sha256_hex(digest),
                "invalid producer request digest"
            );
        }
        if let Some(admission) = &self.context_admission_sha256 {
            ensure!(
                crate::is_sha256_hex(admission)
                    && !admission.bytes().any(|b| b.is_ascii_uppercase()),
                "invalid context admission digest"
            );
        }
        match &self.challenger {
            None => {
                // W2-T02: the released weights file of the accepted semantic manifest.
                ensure!(
                    self.policy_safetensors_sha256
                        == crate::semantic::accepted()?.policy.weights_sha256,
                    "unbound policy safetensors"
                );
                ensure!(
                    self.checkpoint_sha256 == crate::ACCEPTED_STATE_SHA256,
                    "producer requires accepted checkpoint state"
                );
            }
            Some(challenger) => {
                challenger.validate()?;
                ensure!(
                    self.checkpoint_sha256 == challenger.state_sha256
                        && self.policy_safetensors_sha256 == challenger.policy_safetensors_sha256,
                    "producer request differs from its challenger identity"
                );
            }
        }
        ensure!(
            [
                crate::CHECKPOINT_VOCABULARY_SHA256,
                crate::PRODUCT_VOCABULARY_SHA256
            ]
            .contains(&self.vocabulary_sha256.as_str()),
            "unbound producer vocabulary"
        );
        // The same bounds every decode entry enforces (`SamplingOptions::validate`).
        ensure!(
            (1..=crate::decode::MAX_ROUNDS).contains(&self.sampling_rounds)
                && (1..=crate::decode::MAX_BATCH).contains(&self.sampling_batch)
                && self.sampling_temperature.is_finite()
                && self.sampling_temperature > 0.,
            "invalid bounded sampling configuration"
        );
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        self.validate()?;
        Ok(sha256_hex(&serde_json::to_vec(self)?))
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    schema: String,
    request: Request,
    request_sha256: String,
    population_json: String,
    population_sha256: String,
    canonical_recipe_sha256: Vec<String>,
    ranking_backend: String,
    /// The proposal's identity and denominators (W4-T07). Always written by this producer;
    /// an artifact without one is historical: its population is readable, but it carries no
    /// candidate-identity or denominator evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    proposal_ledger: Option<crate::proposal_ledger::ProposalLedger>,
}

/// Called only by the pinned policy worker after `seeds::propose` or
/// `seeds::propose_challenger`. Request seed and sampling fields must be used by that
/// worker when invoking either; their binding is audited executable behavior, not
/// inferable from sampled programs.
fn artifact_bytes(request: &Request, proposal: &Proposal) -> Result<Vec<u8>> {
    let request_sha256 = request.identity()?;
    let population = proposal.population();
    ensure!(
        population.prepared_sha256() == request.prepared_sha256
            && population.training_sha256() == request.training_sha256
            && population.cut() == request.cut
            && population.context_sha256() == request.context_sha256
            && population.checkpoint_sha256() == request.checkpoint_sha256,
        "producer request differs from proposal"
    );
    ensure!(
        proposal.ranking().bindings().vocabulary_sha256() == request.vocabulary_sha256,
        "producer vocabulary mismatch"
    );
    let population_bytes = seeds::population_bytes(proposal)?;
    let canonical_recipe_sha256 = population
        .candidates()
        .iter()
        .map(|c| recipe_hash(c.genome()))
        .collect::<Result<Vec<_>>>()?;
    Ok(serde_json::to_vec(&Envelope {
        schema: SCHEMA.into(),
        request: request.clone(),
        request_sha256,
        population_sha256: sha256_hex(&population_bytes),
        population_json: String::from_utf8(population_bytes)?,
        canonical_recipe_sha256,
        ranking_backend: proposal.ranking().backend().into(),
        proposal_ledger: Some(proposal.ledger().clone()),
    })?)
}

/// Worker entry point after loading the pinned inputs. Uses every request seed
/// and sampling parameter directly; the coordinator runs this producer in a
/// bounded process. This does not establish native CUDA parity.
pub fn produce(
    request: &Request,
    model: &crate::model::PolicyModel,
    checkpoint: &crate::config::CheckpointBinding,
    vocabulary: &crate::grammar::BoundVocabulary,
    context: &PolicyContext,
) -> Result<Vec<u8>> {
    request.validate()?;
    ensure!(
        context.len() == 2048 && context.iter().all(|v| v.is_finite()),
        "producer needs verified2048 context"
    );
    ensure!(
        crate::context_sha256(context) == request.context_sha256
            && vocabulary.sha256() == request.vocabulary_sha256
            && checkpoint.state_sha256() == request.checkpoint_sha256
            && checkpoint.policy_safetensors_sha256() == request.policy_safetensors_sha256,
        "loaded producer input identity mismatch"
    );
    let binding = seeds::SeedBinding::new(
        &request.prepared_sha256,
        request.cut,
        &request.training_sha256,
        &request.context_sha256,
    )?;
    let proposal_seeds = seeds::ProposalSeeds {
        sampling: request.sampling_seed,
        fallback: request.fallback_seed,
        exploration: request.exploration_seed,
    };
    let sampling = crate::decode::SamplingOptions {
        rounds: request.sampling_rounds,
        batch: request.sampling_batch,
        temperature: request.sampling_temperature,
    };
    let proposal = match &request.challenger {
        None => seeds::propose(
            model,
            checkpoint,
            vocabulary,
            context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?,
        Some(challenger) => seeds::propose_challenger(
            model,
            checkpoint,
            challenger,
            vocabulary,
            context,
            &binding,
            &proposal_seeds,
            &sampling,
        )?,
    };
    artifact_bytes(request, &proposal)
}

/// Claims from a producer artifact. This type alone is never producer proof.
pub struct ProducerFile {
    envelope: Envelope,
    population: SeedFile,
}
impl ProducerFile {
    pub fn population(&self) -> &SeedFile {
        &self.population
    }
    pub fn population_sha256(&self) -> &str {
        &self.envelope.population_sha256
    }
    pub fn ranking_backend(&self) -> &str {
        &self.envelope.ranking_backend
    }
    /// The reconciled proposal ledger; `None` for a historical artifact written without one.
    pub fn ledger(&self) -> Option<&crate::proposal_ledger::ProposalLedger> {
        self.envelope.proposal_ledger.as_ref()
    }
    /// Every draw's terminal outcome for the controller's candidate ledger (W4-T03); `None`
    /// for a historical artifact.
    pub fn draw_outcomes(&self) -> Result<Option<crate::proposal_ledger::DrawOutcomes>> {
        self.ledger().map(|l| l.draw_outcomes()).transpose()
    }
}
pub fn read_claims(bytes: &[u8], expected: &Request) -> Result<ProducerFile> {
    let envelope: Envelope = serde_json::from_slice(bytes)?;
    ensure!(
        envelope.schema == SCHEMA
            && envelope.request == *expected
            && envelope.request_sha256 == expected.identity()?,
        "producer request/schema mismatch"
    );
    ensure!(
        sha256_hex(envelope.population_json.as_bytes()) == envelope.population_sha256,
        "producer population bytes changed"
    );
    let population = seeds::validate_for_search(
        envelope.population_json.as_bytes(),
        &expected.prepared_sha256,
        expected.cut,
        &expected.training_sha256,
    )?;
    ensure!(
        population.context_sha256() == expected.context_sha256
            && population.checkpoint_sha256() == expected.checkpoint_sha256,
        "producer context/checkpoint mismatch"
    );
    ensure!(
        envelope.ranking_backend == crate::ranking::BACKEND_CPU,
        "unknown producer numerical backend"
    );
    let hashes = population
        .candidates()
        .iter()
        .map(|c| recipe_hash(c.genome()))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        hashes == envelope.canonical_recipe_sha256,
        "producer canonical recipes changed"
    );
    if let Some(ledger) = &envelope.proposal_ledger {
        // A claim, re-derived here: every count and link from the records, the exact seed
        // population and canonical recipes, and the request's identities and sampling.
        ledger.reconcile()?;
        ledger.matches_population(population.candidates())?;
        ensure!(
            ledger.canonical_recipes() == envelope.canonical_recipe_sha256,
            "proposal ledger canonical recipes differ from the artifact"
        );
        let s = &ledger.sampling;
        ensure!(
            ledger.checkpoint_sha256 == expected.checkpoint_sha256
                && ledger.context_sha256 == expected.context_sha256
                && ledger.vocabulary_sha256 == expected.vocabulary_sha256
                && s.sampling_seed == expected.sampling_seed
                && s.fallback_seed == expected.fallback_seed
                && s.exploration_seed == expected.exploration_seed
                && s.rounds_scheduled == expected.sampling_rounds
                && s.batch == expected.sampling_batch
                && s.temperature == expected.sampling_temperature,
            "proposal ledger belongs to another request"
        );
    } else {
        // A historical artifact has nothing to join a seed slot to.
        ensure!(
            population
                .candidates()
                .iter()
                .all(|c| c.proposal_record_sha256().is_none()),
            "seed population names proposal records without a proposal ledger"
        );
    }
    Ok(ProducerFile {
        envelope,
        population,
    })
}
fn recipe_hash(genome: &str) -> Result<String> {
    let recipe = feat_math::GenomeIr::parse(genome)?;
    Ok(sha256_hex(recipe.canonical().as_bytes()))
}

/// Same token-uniform legal-grammar policy used by rejection fallback and
/// exploration. Each draw uses a fresh, explicitly recorded ChaCha8 seed.
pub fn generic_program(
    vocabulary: &crate::grammar::BoundVocabulary,
    task: &str,
    seed: u64,
) -> Result<Option<String>> {
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    crate::decode::grammar_uniform_program(vocabulary.vocab(), task, &mut rng)
}

#[cfg(test)]
mod tests;

#[cfg(feature = "context-producer")]
mod context_bundle;
#[cfg(feature = "context-producer")]
pub use context_bundle::QualifiedContextBundle;
