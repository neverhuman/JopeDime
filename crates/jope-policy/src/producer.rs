//! Private, lossless proposal transport across a pinned bounded worker boundary.
//! Reading this artifact proves consistency, not execution. The controller must
//! authenticate the pinned producer and its completed bounded execution/output
//! receipts before using the reader. `Proposal` remains non-deserializable.
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
}
impl RequestPlan {
    pub fn request(&self, context_sha256: String) -> Request {
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
            context_sha256,
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
        ensure!(
            self.policy_safetensors_sha256
                == "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e",
            "unbound policy safetensors"
        );
        ensure!(
            self.checkpoint_sha256 == crate::ACCEPTED_STATE_SHA256,
            "producer requires accepted checkpoint state"
        );
        ensure!(
            [
                crate::CHECKPOINT_VOCABULARY_SHA256,
                crate::PRODUCT_VOCABULARY_SHA256
            ]
            .contains(&self.vocabulary_sha256.as_str()),
            "unbound producer vocabulary"
        );
        ensure!(
            (1..=8).contains(&self.sampling_rounds)
                && (1..=128).contains(&self.sampling_batch)
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
}

/// Called only by the pinned policy worker after `seeds::propose`. Request seed
/// and sampling fields must be used by that worker when invoking propose; their
/// binding is audited executable behavior, not inferable from sampled programs.
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
    context: &[f32],
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
    let proposal = seeds::propose(
        model,
        checkpoint,
        vocabulary,
        context,
        &binding,
        &seeds::ProposalSeeds {
            sampling: request.sampling_seed,
            fallback: request.fallback_seed,
            exploration: request.exploration_seed,
        },
        &crate::decode::SamplingOptions {
            rounds: request.sampling_rounds,
            batch: request.sampling_batch,
            temperature: request.sampling_temperature,
        },
    )?;
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
    Ok(ProducerFile {
        envelope,
        population,
    })
}
fn recipe_hash(genome: &str) -> Result<String> {
    let recipe = feat_math::GenomeIr::parse(genome)?;
    Ok(sha256_hex(recipe.canonical().as_bytes()))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn request() -> Request {
        Request {
            worker_source_sha256: sha256_hex(b"synthetic producer source"),
            prepared_sha256: sha256_hex(b"synthetic prepared"),
            training_sha256: sha256_hex(b"synthetic training"),
            cut: 0,
            context_sha256: sha256_hex(b"synthetic context"),
            checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
            policy_safetensors_sha256:
                "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
            vocabulary_sha256: crate::CHECKPOINT_VOCABULARY_SHA256.into(),
            sampling_seed: 1,
            fallback_seed: 2,
            exploration_seed: 3,
            sampling_rounds: 8,
            sampling_batch: 128,
            sampling_temperature: 1.,
        }
    }
    // Deliberately forgeable file claims, never a Proposal or execution proof.
    fn claims() -> Envelope {
        let request = request();
        let genomes = (0..80)
            .map(|i| format!("(pipe X (ridge lambda={}))", i + 1))
            .collect::<Vec<_>>();
        let population = serde_json::json!({"schema":seeds::SEED_SCHEMA,"prepared_sha256":request.prepared_sha256,"training_sha256":request.training_sha256,"cut":request.cut,"context_sha256":request.context_sha256,"checkpoint_sha256":request.checkpoint_sha256,"candidates":genomes.iter().enumerate().map(|(i,g)|serde_json::json!({"genome":g,"origin":if i<64 { seeds::ORIGIN_NEURAL } else { seeds::ORIGIN_RANDOM },"rank_log_weight":null})).collect::<Vec<_>>()});
        let bytes = serde_json::to_vec(&population).unwrap();
        Envelope {
            schema: SCHEMA.into(),
            request_sha256: request.identity().unwrap(),
            request,
            population_json: String::from_utf8(bytes.clone()).unwrap(),
            population_sha256: sha256_hex(&bytes),
            canonical_recipe_sha256: genomes.iter().map(|g| recipe_hash(g).unwrap()).collect(),
            ranking_backend: crate::ranking::BACKEND_CPU.into(),
        }
    }
    #[test]
    fn producer_reader_preserves_80_typed_recipes_but_is_only_claims() -> Result<()> {
        let file = read_claims(&serde_json::to_vec(&claims())?, &request())?;
        assert_eq!(file.population().candidates().len(), 80);
        assert_eq!(
            file.population().candidates()[0].genome(),
            "(pipe X (ridge lambda=1))"
        );
        Ok(())
    }
    #[test]
    fn producer_transport_rejects_seed_request_recipe_and_context_tampering() -> Result<()> {
        let expected = request();
        let mut changed = claims();
        changed.request.sampling_seed += 1;
        assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
        let mut changed = claims();
        changed.canonical_recipe_sha256[0] = sha256_hex(b"different recipe");
        assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
        let mut changed = claims();
        changed.population_json.push(' ');
        assert!(read_claims(&serde_json::to_vec(&changed)?, &expected).is_err());
        let mut different = expected.clone();
        different.context_sha256 = sha256_hex(b"another context");
        assert!(read_claims(&serde_json::to_vec(&claims())?, &different).is_err());
        Ok(())
    }
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

/// Private context admission payload for the separate policy child. The parent
/// authenticates all original bounded receipts before pinning these normalized
/// attestations. This transport verifier alone is not process-execution proof.
#[cfg(feature = "context-producer")]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualifiedContextBundle {
    pub artifact: Vec<u8>,
    pub qualification: Vec<u8>,
    pub reference_manifest: Vec<u8>,
    pub context_execution: Vec<u8>,
    pub case_executions: Vec<Vec<u8>>,
    pub pins: crate::context_producer::AdmissionPins,
}
#[cfg(feature = "context-producer")]
impl QualifiedContextBundle {
    pub fn verify(&self) -> Result<crate::context_producer::VerifiedCpuContext> {
        crate::context_producer::verify_qualified(
            &self.artifact,
            &self.qualification,
            &self.reference_manifest,
            &self.context_execution,
            &self.case_executions,
            &self.pins,
        )
    }
}
