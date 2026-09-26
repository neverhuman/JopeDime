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
                ensure!(
                    self.policy_safetensors_sha256
                        == "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e",
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
    pub(super) fn request() -> Request {
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
            challenger: None,
            context_admission_sha256: None,
        }
    }
    // Deliberately forgeable file claims, never a Proposal or execution proof.
    fn claims() -> Envelope {
        claims_for(request())
    }
    fn claims_for(request: Request) -> Envelope {
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

    fn plan_of(r: &Request) -> RequestPlan {
        RequestPlan {
            worker_source_sha256: r.worker_source_sha256.clone(),
            prepared_sha256: r.prepared_sha256.clone(),
            training_sha256: r.training_sha256.clone(),
            cut: r.cut,
            checkpoint_sha256: r.checkpoint_sha256.clone(),
            policy_safetensors_sha256: r.policy_safetensors_sha256.clone(),
            vocabulary_sha256: r.vocabulary_sha256.clone(),
            sampling_seed: r.sampling_seed,
            fallback_seed: r.fallback_seed,
            exploration_seed: r.exploration_seed,
            sampling_rounds: r.sampling_rounds,
            sampling_batch: r.sampling_batch,
            sampling_temperature: r.sampling_temperature,
            challenger: r.challenger.clone(),
        }
    }
    /// W2-T02 G-03: the controller-issued admission digest is optional (native request bytes and
    /// identities are unchanged), strictly formatted, and part of the request identity.
    #[test]
    fn context_admission_pin_is_optional_and_bound() -> Result<()> {
        let native = request();
        let bytes = serde_json::to_vec(&native)?;
        assert!(!String::from_utf8(bytes.clone())?.contains("context_admission_sha256"));
        let parsed: Request = serde_json::from_slice(&bytes)?;
        assert_eq!(parsed.context_admission_sha256, None);
        assert_eq!(parsed.identity()?, native.identity()?);
        let issued = Request {
            context_admission_sha256: Some(sha256_hex(b"controller pins")),
            ..request()
        };
        issued.validate()?;
        assert_ne!(issued.identity()?, native.identity()?);
        let parsed: Request = serde_json::from_slice(&serde_json::to_vec(&issued)?)?;
        assert_eq!(parsed, issued);
        for bad in [
            "not hex".to_string(),
            sha256_hex(b"controller pins").to_ascii_uppercase(),
        ] {
            assert!(Request {
                context_admission_sha256: Some(bad),
                ..request()
            }
            .validate()
            .is_err());
        }
        // The plan attaches exactly the encoding outputs.
        let plan = plan_of(&native);
        assert_eq!(plan.request(native.context_sha256.clone()), native);
        assert_eq!(
            plan.qualified_request(
                native.context_sha256.clone(),
                sha256_hex(b"controller pins")
            ),
            issued
        );
        // Claims written for one request never satisfy the other.
        assert!(read_claims(&serde_json::to_vec(&claims())?, &issued).is_err());
        read_claims(&serde_json::to_vec(&claims_for(issued.clone()))?, &issued)?;
        Ok(())
    }

    fn challenger() -> crate::config::ChallengerIdentity {
        crate::config::ChallengerIdentity {
            state_sha256: sha256_hex(b"challenger state"),
            parameter_sha256: sha256_hex(b"challenger parameters"),
            config_sha256: crate::ACCEPTED_CONFIG_SHA256.into(),
            policy_safetensors_sha256: sha256_hex(b"challenger weights"),
        }
    }
    fn challenger_request() -> Request {
        let c = challenger();
        Request {
            checkpoint_sha256: c.state_sha256.clone(),
            policy_safetensors_sha256: c.policy_safetensors_sha256.clone(),
            challenger: Some(c),
            ..request()
        }
    }
    fn refused(request: Request, needle: &str) {
        let error = request.validate().unwrap_err();
        assert!(format!("{error:#}").contains(needle), "{error:#}");
    }
    #[test]
    fn challenger_request_is_optional_and_bound() -> Result<()> {
        // Absent: accepted request bytes, parse and identity are unchanged.
        let accepted = request();
        let bytes = serde_json::to_vec(&accepted)?;
        assert!(!String::from_utf8(bytes.clone())?.contains("\"challenger\""));
        let parsed: Request = serde_json::from_slice(&bytes)?;
        assert_eq!(parsed.challenger, None);
        assert_eq!(serde_json::to_vec(&parsed)?, bytes);
        assert_eq!(parsed.identity()?, accepted.identity()?);
        // Present: bound to its own state and weights, and a different request identity.
        let c = challenger();
        let bound = challenger_request();
        bound.validate()?;
        assert_ne!(bound.identity()?, accepted.identity()?);
        refused(
            Request {
                checkpoint_sha256: sha256_hex(b"other state"),
                ..challenger_request()
            },
            "differs from its challenger identity",
        );
        refused(
            Request {
                policy_safetensors_sha256: sha256_hex(b"other weights"),
                ..challenger_request()
            },
            "differs from its challenger identity",
        );
        let mut named_accepted = c.clone();
        named_accepted.state_sha256 = crate::ACCEPTED_STATE_SHA256.into();
        refused(
            Request {
                checkpoint_sha256: crate::ACCEPTED_STATE_SHA256.into(),
                challenger: Some(named_accepted),
                ..challenger_request()
            },
            "names the accepted identity",
        );
        // An accepted-shaped request cannot carry a challenger.
        assert!(Request {
            challenger: Some(c.clone()),
            ..request()
        }
        .validate()
        .is_err());
        // The plan carries the identity into the request.
        let r = challenger_request();
        let plan = plan_of(&r);
        plan.validate()?;
        assert_eq!(plan.request(r.context_sha256.clone()), r);
        assert_eq!(plan.request(r.context_sha256.clone()).challenger, Some(c));
        // The reader binds the population to the challenger state, never the accepted one.
        let file = read_claims(
            &serde_json::to_vec(&claims_for(challenger_request()))?,
            &challenger_request(),
        )?;
        assert_eq!(
            file.population().checkpoint_sha256(),
            challenger_request().checkpoint_sha256
        );
        let mut relabelled = claims_for(challenger_request());
        let mut population: serde_json::Value = serde_json::from_str(&relabelled.population_json)?;
        population["checkpoint_sha256"] = serde_json::json!(crate::ACCEPTED_STATE_SHA256);
        let population = serde_json::to_vec(&population)?;
        relabelled.population_sha256 = sha256_hex(&population);
        relabelled.population_json = String::from_utf8(population)?;
        assert!(read_claims(&serde_json::to_vec(&relabelled)?, &challenger_request()).is_err());
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
    /// Verify the bundle against its carried pins. Those pins are only as trustworthy as the
    /// channel that delivered the bundle: a controller that authenticated them independently
    /// (for example `cpu_context_bridge::load_qualified`) may call this; a worker must use
    /// [`Self::admit_for_request`].
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

    /// Policy-worker admission of a qualified CPU context. The bundle's pins must be exactly the
    /// admission the controller issued in the authenticated request
    /// ([`Request::context_admission_sha256`]); only then is the bundle verified against them.
    /// The final context must then be the frozen assembly of the qualified components under the
    /// accepted transforms, bit-identical to the actual policy input, and deliverable to the
    /// accepted policy for this request's scope.
    pub fn admit_for_request(
        &self,
        request: &Request,
        transforms: &crate::context_producer::FrozenTransforms,
        policy_input: &[u8],
    ) -> Result<crate::context_producer::ContextHandoff> {
        request.validate()?;
        let issued = request.context_admission_sha256.as_deref().ok_or_else(|| {
            anyhow::anyhow!("request carries no controller-issued context admission pin")
        })?;
        ensure!(
            self.pins.sha256()? == issued,
            "context bundle pins are not the controller-issued admission of this request"
        );
        let verified = self.verify()?;
        let context = verified.replay_assembly(transforms)?;
        ensure!(
            context.to_le_bytes() == policy_input,
            "qualified context differs from the actual policy input"
        );
        let handoff = verified.handoff(
            &context,
            crate::context_producer::ContextConsumer::AcceptedPolicyProposal,
        )?;
        handoff.deliver_to_policy(request)?;
        Ok(handoff)
    }
}

#[cfg(all(test, feature = "context-producer"))]
mod bundle_tests {
    use super::*;
    use crate::context_producer::*;
    fn forged_bundle() -> QualifiedContextBundle {
        let artifact = br#"{"not":"checked yet"}"#.to_vec();
        let pins = AdmissionPins {
            source_sha256: "e".repeat(40),
            producer_sha256: "f".repeat(64),
            artifact_sha256: sha256_hex(&artifact),
            qualification_sha256: sha256_hex(b"q"),
            reference_manifest_sha256: sha256_hex(b"m"),
            context_execution_sha256: sha256_hex(b"x"),
            qualification_execution_sha256: vec![sha256_hex(b"c"); 8],
            verification_policy_sha256: verification_policy_sha256(),
            training: TrainingIdentity {
                dataset_sha256: "a".repeat(64),
                prepared_sha256: "b".repeat(64),
                training_sha256: "c".repeat(64),
                fitting_rows_sha256: "d".repeat(64),
                cut: 0,
            },
            input_sha256: "1".repeat(64),
            preprocessing_sha256: "2".repeat(64),
        };
        QualifiedContextBundle {
            artifact,
            qualification: b"q".to_vec(),
            reference_manifest: b"m".to_vec(),
            context_execution: b"x".to_vec(),
            case_executions: vec![b"c".to_vec(); 8],
            pins,
        }
    }
    fn refusal(
        bundle: &QualifiedContextBundle,
        request: &Request,
        transforms: &FrozenTransforms,
    ) -> String {
        match bundle.admit_for_request(request, transforms, &[0; 8192]) {
            Ok(_) => panic!("forged bundle admitted"),
            Err(error) => format!("{error:#}"),
        }
    }

    /// W2-T02 G-03: the worker verifies a bundle against the controller-issued admission in the
    /// request, never against the pins the bundle carries. Pre-fix reproduction (retained in the
    /// w2b-repro-prefix evidence): a bundle that chose its own pins passed every pin gate of
    /// `verify` and was refused only later, while parsing its artifact.
    #[test]
    fn worker_admission_requires_the_request_issued_pins() {
        let transforms = FrozenTransforms::synthetic_for_testing(1.0);
        let bundle = forged_bundle();
        let request = super::tests::request();
        // No admission issued: refused before any verification.
        let error = refusal(&bundle, &request, &transforms);
        assert!(
            error.contains("no controller-issued context admission"),
            "{error}"
        );
        // The controller issued other pins: refused, whatever the bundle claims.
        let mut other = bundle.pins.clone();
        other.input_sha256 = "9".repeat(64);
        let issued_other = Request {
            context_admission_sha256: Some(other.sha256().unwrap()),
            ..request.clone()
        };
        let error = refusal(&bundle, &issued_other, &transforms);
        assert!(
            error.contains("not the controller-issued admission"),
            "{error}"
        );
        // Pins changed after issuance (for example to name a substituted artifact): refused.
        let issued = Request {
            context_admission_sha256: Some(bundle.pins.sha256().unwrap()),
            ..request.clone()
        };
        let mut substituted = bundle.clone();
        substituted.artifact = b"{}".to_vec();
        substituted.pins.artifact_sha256 = sha256_hex(&substituted.artifact);
        let error = refusal(&substituted, &issued, &transforms);
        assert!(
            error.contains("not the controller-issued admission"),
            "{error}"
        );
        // Positive control for the gate: with the issued pins the bundle passes the pin gate and
        // full verification runs (this synthetic bundle then fails on its artifact content, so
        // nothing is admitted).
        let error = refusal(&bundle, &issued, &transforms);
        assert!(
            !error.contains("controller-issued") && error.contains("unknown field"),
            "{error}"
        );
    }
}
