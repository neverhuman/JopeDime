//! Generation-zero policy consumption followed by bounded, callback-driven GP.
//!
//! Every fit goes through `ExecutionPool`; no engine authority or GPU fence is
//! changed here. Producer proof comes from `jope_policy::seeds::Proposal`, never
//! an origin string read from a seed file. A pinned bounded child transports
//! that producer result with bound request and output receipts. The Candle policy's arithmetic remains
//! explicitly unqualified against native CUDA BF16.

use anyhow::{ensure, Context, Result};
use feat_math::evolve;
use jope_policy::seeds::{ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Path, PathBuf},
    sync::atomic::AtomicBool,
};

use crate::{
    execution::{ArmBudget, ExecutionPool, WorkerJob, WorkerOutcome},
    experiment::{EvaluationRequest, EvaluationStatus},
    search_plan::{OofEvidence, SelectionPlan},
    sha256_hex,
    synthesized_recipe::{SynthesizedRecipe, FIT_RANDOMNESS_POLICY, RAW_INPUT_POLICY},
};

pub const PROTOCOL: &str = "asm.bounded-neural-search/v1";
pub const INITIAL_SIZE: usize = 80;
pub const NEURAL_SLOTS: usize = 64;
const ELITES: usize = 4;

/// The controller supplies the full comparison identity (task, dataset, cut,
/// evaluator, vocabulary, preprocessing and protocol), not just a dataset name.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairInputPlan {
    pub selection: SelectionPlan,
    pub preprocessing_sha256: String,
    pub raw_input_policy: String,
    pub fit_randomness_policy: String,
    pub installation: InstallationPins,
    pub proposal: jope_policy::producer::RequestPlan,
    pub encoding: EncodingPlan,
    pub variation_identity: String,
    pub generic_sampling_policy: String,
    pub checkpoint_training_exposure: String,
    pub exposure_evidence_sha256: Option<String>,
    pub prepared_sha256: String,
    pub training_sha256: String,
    pub cut: usize,
    pub vocabulary_sha256: String,
    pub seconds: u64,
    pub max_evaluations: usize,
    pub random_seed: u64,
}
impl PairInputPlan {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.preprocessing_sha256,
            &self.prepared_sha256,
            &self.training_sha256,
            &self.vocabulary_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid comparison digest"
            );
        }
        ensure!(
            self.raw_input_policy == RAW_INPUT_POLICY
                && self.fit_randomness_policy == FIT_RANDOMNESS_POLICY,
            "unsupported input/randomness policy"
        );
        self.selection.validate()?;
        ensure!(
            self.cut == self.selection.outer().fold as usize,
            "context cut differs from frozen outer fold"
        );
        self.installation.validate()?;
        self.encoding.validate()?;
        self.proposal.validate()?;
        ensure!(
            self.proposal.prepared_sha256 == self.prepared_sha256
                && self.proposal.training_sha256 == self.training_sha256
                && self.proposal.cut == self.cut
                && self.proposal.vocabulary_sha256 == self.vocabulary_sha256
                && self.proposal.worker_source_sha256 == self.installation.source_manifest_sha256,
            "producer request differs from pair plan"
        );
        ensure!(
            !self.variation_identity.is_empty() && !self.generic_sampling_policy.is_empty(),
            "missing frozen variation/generic policy"
        );
        ensure!(
            ["known_seen", "known_unseen", "unknown"]
                .contains(&self.checkpoint_training_exposure.as_str()),
            "unknown exposure status"
        );
        ensure!(
            self.exposure_evidence_sha256
                .as_ref()
                .is_none_or(|d| jope_policy::is_sha256_hex(d))
                && (self.checkpoint_training_exposure == "unknown"
                    || self.exposure_evidence_sha256.is_some()),
            "known exposure needs evidence"
        );
        ensure!(
            (1..=120).contains(&self.seconds),
            "search arm must be at most 120 seconds"
        );
        ensure!(
            (INITIAL_SIZE..=1000).contains(&self.max_evaluations),
            "search must admit 80 initial slots and at most 1000 evaluations"
        );
        Ok(())
    }
    pub fn arm(&self, neural: bool) -> Result<ArmBudget> {
        self.validate()?;
        Ok(ArmBudget {
            id_sha256: sha256_hex(&serde_json::to_vec(&(
                PROTOCOL,
                self,
                if neural { "neural" } else { "generic" },
            ))?),
            seconds: self.seconds,
        })
    }
}

/// Encoder inputs are frozen before the neural arm starts. Output digests are
/// absent here because a fresh, bounded encoding has not produced them yet.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EncodingPlan {
    pub input_sha256: String,
    pub source_sha40: String,
    pub program_sha256: String,
    pub source_manifest_sha256: String,
    pub encoder_assets_manifest_sha256: String,
    pub qualification_sha256: Option<String>,
    pub reference_manifest_sha256: Option<String>,
    pub backend: String,
}
impl EncodingPlan {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.source_sha40.len() == 40
                && self.source_sha40.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid encoder source identity"
        );
        ensure!(
            self.reference_manifest_sha256
                .as_ref()
                .is_none_or(|s| jope_policy::is_sha256_hex(s)),
            "invalid reference manifest identity"
        );
        for digest in [
            &self.input_sha256,
            &self.program_sha256,
            &self.source_manifest_sha256,
            &self.encoder_assets_manifest_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid encoding input pin"
            );
        }
        ensure!(
            !self.backend.is_empty()
                && self
                    .qualification_sha256
                    .as_ref()
                    .is_none_or(|d| jope_policy::is_sha256_hex(d)),
            "invalid encoding backend/qualification pin"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextCost {
    pub cached: bool,
    pub production_receipt: crate::execution::Artifact,
    pub original_wall_seconds: Option<f64>,
    pub original_cpu_seconds: Option<f64>,
    pub original_gpu_seconds: Option<f64>,
    /// None means no measured reuse count, not zero production cost.
    pub amortization_uses: Option<u64>,
}
impl ContextCost {
    fn validate(&self) -> Result<()> {
        self.production_receipt.verify()?;
        ensure!(
            [
                self.original_wall_seconds,
                self.original_cpu_seconds,
                self.original_gpu_seconds
            ]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && v >= 0.),
            "invalid original encoding cost"
        );
        ensure!(
            self.amortization_uses.is_none_or(|n| n > 0),
            "invalid encoding reuse count"
        );
        Ok(())
    }
}
/// Outputs attach to a previously registered input plan. Dereferencing exposes
/// that immutable input plan; arm identity never depends on these output hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Comparison {
    pub input_plan: PairInputPlan,
    pub producer_request: jope_policy::producer::Request,
    pub context_receipt_sha256: String,
    pub context_cost: ContextCost,
}
impl std::ops::Deref for Comparison {
    type Target = PairInputPlan;
    fn deref(&self) -> &Self::Target {
        &self.input_plan
    }
}
impl Comparison {
    fn validate(&self) -> Result<()> {
        self.input_plan.validate()?;
        self.producer_request.validate()?;
        ensure!(
            self.producer_request
                == self
                    .proposal
                    .request(self.producer_request.context_sha256.clone())
                && jope_policy::is_sha256_hex(&self.context_receipt_sha256),
            "context outputs changed preregistered producer inputs"
        );
        self.context_cost.validate()
    }
}

/// Installation provenance is frozen before either arm. The controller verifies
/// the private release manifest; producer output cannot choose these pins.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallationPins {
    pub release_manifest_sha256: String,
    pub policy_program_sha256: String,
    pub source_manifest_sha256: String,
    pub checkpoint_receipt_sha256: String,
}
impl InstallationPins {
    fn validate(&self) -> Result<()> {
        for digest in [
            &self.release_manifest_sha256,
            &self.policy_program_sha256,
            &self.source_manifest_sha256,
            &self.checkpoint_receipt_sha256,
        ] {
            ensure!(
                jope_policy::is_sha256_hex(digest),
                "invalid frozen installation pin"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    NeuralPolicy,
    NeuralRejectionFallback,
    RandomExploration,
    Generic,
    Mutation,
    Crossover,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Candidate {
    pub id_sha256: String,
    pub genome: String,
    pub recipe: SynthesizedRecipe,
    /// Canonical executable recipe, including explicitly recorded preprocessing.
    pub program_sha256: String,
    pub generation: usize,
    /// Original proposal slot for generation zero; no invented slot for offspring.
    pub initial_slot: Option<usize>,
    pub origin: Origin,
    pub parents: Vec<String>,
    pub rank_log_weight: Option<f64>,
}
impl Candidate {
    fn new(
        arm: &ArmBudget,
        genome: String,
        generation: usize,
        slot: usize,
        origin: Origin,
        parents: Vec<String>,
        rank_log_weight: Option<f64>,
    ) -> Result<Self> {
        let recipe = SynthesizedRecipe::new(genome.clone())?;
        ensure!(
            rank_log_weight.is_none_or(f64::is_finite),
            "nonfinite policy rank weight"
        );
        let program_sha256 = recipe.executable_sha256.clone();
        let id_sha256 = sha256_hex(&serde_json::to_vec(&(
            &arm.id_sha256,
            generation,
            slot,
            &program_sha256,
            &origin,
            &parents,
        ))?);
        Ok(Self {
            id_sha256,
            genome,
            recipe,
            program_sha256,
            generation,
            initial_slot: (generation == 0).then_some(slot),
            origin,
            parents,
            rank_log_weight,
        })
    }
}

/// Scoring stays in the controller; the worker receives train labels only.
/// `physical_fit_starts` must come from the worker's fit receipt, not exit code,
/// scheduling, or a policy-origin label. Unknown counters remain None.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Feedback {
    pub candidate_sha256: String,
    pub status: EvaluationStatus,
    pub loss: Option<f64>,
    pub physical_fit_starts: Option<u64>,
    pub worker_receipt_sha256: Option<String>,
    pub fitted_artifact_sha256: Option<String>,
    pub reason: Option<String>,
    pub oof: Option<OofEvidence>,
    /// No launched attempt exists; retry is permitted inside the original clock.
    pub retryable_inventory: bool,
}
impl Feedback {
    fn enforce_selection(&mut self, selection: &SelectionPlan, candidate: &Candidate) {
        self.enforce_measurement();
        if self.status == EvaluationStatus::Completed
            && !self.oof.as_ref().is_some_and(|o| {
                o.validate(selection, &candidate.program_sha256, &candidate.id_sha256)
                    .is_ok()
                    && o.total_fit_starts() == self.physical_fit_starts
                    && o.fitted_bundle_sha256().ok() == self.fitted_artifact_sha256
            })
        {
            self.status = EvaluationStatus::Failed;
            self.loss = None;
            self.reason = Some("selection lacks exact complete train-side OOF measurements".into());
        }
    }
    fn enforce_measurement(&mut self) {
        if self.status == EvaluationStatus::Completed
            && !(self.loss.is_some_and(f64::is_finite)
                && self.physical_fit_starts.is_some_and(|n| n > 0)
                && self
                    .worker_receipt_sha256
                    .as_ref()
                    .is_some_and(|s| jope_policy::is_sha256_hex(s))
                && self
                    .fitted_artifact_sha256
                    .as_ref()
                    .is_some_and(|s| jope_policy::is_sha256_hex(s)))
        {
            self.status = EvaluationStatus::Failed;
            self.reason = Some("completed fit lacks finite loss, physical-fit counter, fitted artifact or worker receipt".into());
        }
        if self.status != EvaluationStatus::Completed {
            self.loss = None;
        }
    }
    #[cfg(test)]
    fn exhausted(candidate: &Candidate) -> Self {
        Self {
            candidate_sha256: candidate.id_sha256.clone(),
            status: EvaluationStatus::BudgetExhausted,
            loss: None,
            physical_fit_starts: Some(0),
            worker_receipt_sha256: None,
            fitted_artifact_sha256: None,
            reason: Some("original arm deadline reached before fit admission".into()),
            oof: None,
            retryable_inventory: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub candidate: Candidate,
    pub feedback: Feedback,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchReceipt {
    pub schema: String,
    pub comparison_sha256: String,
    pub selection_sha256: String,
    pub arm: ArmBudget,
    pub deadline_unix_ms: u64,
    pub proposal: Option<ProposalEvidence>,
    pub observations: Vec<Observation>,
    pub stop_reason: String,
}
impl SearchReceipt {
    /// Validate retained population metadata in both live resume and read-only
    /// reports. This does not authenticate worker outputs or proposal origins;
    /// callers must still replay the pinned producer and completed OOF receipts.
    pub fn validate_population(&self, plan: &PairInputPlan) -> Result<()> {
        let neural_arm = plan.arm(true)?;
        let generic_arm = plan.arm(false)?;
        let neural = self.arm.id_sha256 == neural_arm.id_sha256;
        ensure!(
            self.schema == PROTOCOL
                && self.comparison_sha256 == sha256_hex(&serde_json::to_vec(plan)?)
                && self.selection_sha256 == plan.selection.identity()?
                && self.arm.seconds == plan.seconds
                && (neural || self.arm.id_sha256 == generic_arm.id_sha256),
            "search population binding changed"
        );
        ensure!(
            (INITIAL_SIZE..=plan.max_evaluations).contains(&self.observations.len()),
            "search population count changed"
        );
        let mut earlier = BTreeMap::<&str, &Observation>::new();
        let mut programs = BTreeSet::new();
        for (generation, group) in self
            .observations
            .chunk_by(|a, b| a.candidate.generation == b.candidate.generation)
            .enumerate()
        {
            ensure!(
                group[0].candidate.generation == generation
                    && if generation == 0 {
                        group.len() == INITIAL_SIZE
                    } else {
                        group.len() <= INITIAL_SIZE - ELITES
                    },
                "search generations or initial 80 slots changed"
            );
            for (slot, observation) in group.iter().enumerate() {
                let c = &observation.candidate;
                if generation == 0 {
                    let expected_origin = if !neural {
                        c.origin == Origin::Generic
                    } else if slot < NEURAL_SLOTS {
                        matches!(
                            c.origin,
                            Origin::NeuralPolicy | Origin::NeuralRejectionFallback
                        )
                    } else {
                        c.origin == Origin::RandomExploration
                    };
                    ensure!(
                        expected_origin && c.parents.is_empty(),
                        "initial origin/parents changed"
                    );
                    ensure!(
                        neural || c.rank_log_weight.is_none(),
                        "generic slot has policy rank"
                    );
                } else {
                    ensure!(
                        matches!(
                            (&c.origin, c.parents.len()),
                            (Origin::Mutation, 1) | (Origin::Crossover, 2)
                        ) && c.rank_log_weight.is_none(),
                        "offspring origin/parent count changed"
                    );
                    ensure!(
                        c.parents
                            .iter()
                            .all(|id| earlier.get(id.as_str()).is_some_and(|p| {
                                p.candidate.generation < generation
                                    && p.feedback.status == EvaluationStatus::Completed
                            })),
                        "offspring lacks an earlier completed parent"
                    );
                }
                let expected = Candidate::new(
                    &self.arm,
                    c.genome.clone(),
                    generation,
                    slot,
                    c.origin.clone(),
                    c.parents.clone(),
                    c.rank_log_weight,
                )?;
                ensure!(
                    serde_json::to_value(c)? == serde_json::to_value(expected)?,
                    "candidate identity, recipe or slot changed"
                );
                ensure!(
                    observation.feedback.candidate_sha256 == c.id_sha256
                        && earlier.insert(c.id_sha256.as_str(), observation).is_none(),
                    "duplicate or mismatched candidate identity"
                );
                let unique = programs.insert(c.program_sha256.as_str());
                ensure!(
                    (neural && generation == 0) || unique,
                    "generic or offspring canonical recipe repeated"
                );
                if observation.feedback.status == EvaluationStatus::Completed {
                    let mut checked = observation.feedback.clone();
                    checked.enforce_selection(&plan.selection, c);
                    ensure!(
                        serde_json::to_value(checked)?
                            == serde_json::to_value(&observation.feedback)?,
                        "completed parent/candidate lacks exact OOF measurements"
                    );
                }
            }
        }
        Ok(())
    }

    /// After population and execution replay, prove an actual completed path
    /// from a generation-zero NeuralPolicy candidate to a completed offspring.
    /// Exploration and rejection fallback roots never establish neural ancestry.
    pub fn has_fitted_neural_descendant(&self) -> bool {
        let mut completed = BTreeMap::<&str, (usize, bool)>::new();
        for o in &self.observations {
            if o.feedback.status != EvaluationStatus::Completed {
                continue;
            }
            let c = &o.candidate;
            let neural_path = if c.generation == 0 {
                c.origin == Origin::NeuralPolicy
            } else {
                matches!(
                    (&c.origin, c.parents.len()),
                    (Origin::Mutation, 1) | (Origin::Crossover, 2)
                ) && c.parents.iter().all(|id| {
                    completed
                        .get(id.as_str())
                        .is_some_and(|(generation, _)| *generation < c.generation)
                }) && c.parents.iter().any(|id| {
                    completed
                        .get(id.as_str())
                        .is_some_and(|(_, neural)| *neural)
                })
            };
            if c.generation > 0 && neural_path {
                return true;
            }
            completed.insert(c.id_sha256.as_str(), (c.generation, neural_path));
        }
        false
    }
    pub fn winner(&self) -> Option<&Observation> {
        self.observations
            .iter()
            .filter(|o| {
                o.feedback.status == EvaluationStatus::Completed
                    && o.feedback.loss.is_some_and(f64::is_finite)
            })
            .min_by(|a, b| {
                a.feedback
                    .loss
                    .unwrap()
                    .total_cmp(&b.feedback.loss.unwrap())
                    .then_with(|| a.candidate.program_sha256.cmp(&b.candidate.program_sha256))
            })
    }
    pub fn neural_physical_fit_starts(&self) -> Option<u64> {
        self.observations
            .iter()
            .filter(|o| o.candidate.origin == Origin::NeuralPolicy)
            .try_fold(0u64, |sum, o| {
                sum.checked_add(o.feedback.physical_fit_starts?)
            })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProposalEvidence {
    pub request: jope_policy::producer::Request,
    pub context_cost: ContextCost,
    pub producer_job_sha256: String,
    pub producer_output_sha256: String,
    pub population_sha256: String,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub ranking_backend: String,
    /// Accepted checkpoint bytes do not qualify this CPU implementation numerically.
    pub native_cuda_parity_qualified: bool,
}

/// Narrow test seam. Production uses `PoolBackend` below. Implementations must
/// persist begin_arm once and enforce its original deadline on every child.
pub trait Backend {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64>;
    fn now_ms(&self) -> u64;
    fn cancelled(&self) -> bool {
        false
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>>;
    fn produce(&mut self, job: WorkerJob) -> Result<WorkerOutcome>;
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &serde_json::Value) -> Result<()>;
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>>;
}

/// Grammar choice belongs to the task. Classification must supply classification
/// grammar here; the built-in `RegressionVariation` is deliberately explicit.
pub trait Variation {
    fn identity(&self) -> &str;
    fn generic_policy(&self) -> &str;
    fn generic(&mut self, seed: u64) -> Result<Option<String>>;
    fn mutate(&mut self, parent: &str, seed: u64) -> Result<String>;
    fn crossover(&mut self, left: &str, right: &str, seed: u64) -> Result<String>;
}
pub struct RegressionVariation<'a> {
    pub vocabulary: &'a jope_policy::grammar::BoundVocabulary,
}
impl Variation for RegressionVariation<'_> {
    fn identity(&self) -> &str {
        "feat-math.canonical-mutate-crossover/max-transforms4/v1"
    }
    fn generic_policy(&self) -> &str {
        "jope-policy.grammar-uniform-token-chacha8/v1"
    }
    fn generic(&mut self, seed: u64) -> Result<Option<String>> {
        jope_policy::producer::generic_program(self.vocabulary, "regression", seed)
    }
    fn mutate(&mut self, parent: &str, seed: u64) -> Result<String> {
        evolve::mutate_canonical_pipeline(parent, seed, 4).map_err(anyhow::Error::msg)
    }
    fn crossover(&mut self, left: &str, right: &str, seed: u64) -> Result<String> {
        evolve::crossover_canonical_pipelines(left, right, seed, 4).map_err(anyhow::Error::msg)
    }
}

/// A release-pinned policy command. Pins come from the verified installation
/// manifest, not the produced file. The child loads verified native context,
/// calls `producer::produce` with bound model, checkpoint and vocabulary. Context and
/// proposal work inside that command are subject to the same process/cgroup
/// gate, fit cap (normally30s) and original120s arm deadline as any worker.
/// Existing context files may be reused; this API never relabels an old encoder.
#[derive(Serialize, Deserialize)]
pub struct PolicyWorker {
    pub job: WorkerJob,
    pub expected_program_sha256: String,
    pub release_manifest: crate::execution::Artifact,
    pub source_manifest: crate::execution::Artifact,
    pub request_artifact: crate::execution::Artifact,
    pub checkpoint_receipt: crate::execution::Artifact,
    pub request: jope_policy::producer::Request,
    pub context: super::search_context::ContextFiles,
    pub output_name: PathBuf,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralRun {
    pub proposal_execution: WorkerOutcome,
    /// None means proposal failed/timed out/unavailable, not a generic substitute.
    pub search: Option<SearchReceipt>,
}

/// One immutable pair, fixed neural-then-generic execution order, same fit and
/// search budgets, same variation implementation and generic token policy.
/// Errors are retained as inventory; they are never converted to wins or losses.
#[derive(Debug, Serialize, Deserialize)]
pub struct PairedReceipt {
    pub comparison_sha256: String,
    pub neural: Option<NeuralRun>,
    pub neural_error: Option<String>,
    pub generic: Option<SearchReceipt>,
    pub generic_error: Option<String>,
}
fn freeze_pair<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &V,
) -> Result<()> {
    comparison.validate()?;
    ensure!(
        variation.identity() == comparison.variation_identity
            && variation.generic_policy() == comparison.generic_sampling_policy,
        "variation/generic policy differs from frozen pair"
    );
    let value = serde_json::json!({"schema":PROTOCOL,"comparison":comparison,"order":["neural","generic"],"seed_policy":"asm.candidate-request-seed/v1","selection":"five training-side OOF folds","refit":"freeze selection winner before development hold scoring"});
    let slot = ArmBudget {
        id_sha256: sha256_hex(&serde_json::to_vec(&(
            "asm.paired-slot/v1",
            comparison.selection.outer(),
        ))?),
        seconds: comparison.seconds,
    };
    backend.persist(&slot, "paired-registration", &value)?;
    for neural in [true, false] {
        backend.persist(&comparison.arm(neural)?, "paired-plan", &value)?;
    }
    Ok(())
}
/// Call before fresh encoding. Repeating it returns the original persisted
/// deadline. A cached context still retains its original production-cost receipt.
pub fn begin_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
) -> Result<(ArmBudget, u64)> {
    freeze_pair(backend, plan, variation)?;
    let arm = plan.arm(true)?;
    let deadline = backend.begin_arm(&arm)?;
    Ok((arm, deadline))
}
/// Fresh encoding runs through the same bounded child API and original neural
/// deadline. The output may later construct Comparison, never a new input plan.
pub fn encode_for_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    mut job: WorkerJob,
) -> Result<WorkerOutcome> {
    let (arm, _) = begin_pair(backend, plan, variation)?;
    ensure!(
        job.command.program.sha256 == plan.encoding.program_sha256
            && job.request.dataset_sha256 == plan.selection.outer().dataset_sha256
            && job.request.fitting_rows_sha256 == plan.selection.outer().fitting_rows_sha256,
        "encoding worker differs from exact training-side plan"
    );
    for digest in [
        &plan.encoding.input_sha256,
        &plan.encoding.source_manifest_sha256,
        &plan.encoding.encoder_assets_manifest_sha256,
    ] {
        ensure!(
            job.command.inputs.iter().any(|a| &a.sha256 == digest),
            "encoding input missing from pinned worker closure"
        );
    }
    job.arm = Some(arm.clone());
    let job_sha = job.identity()?;
    let result = backend.produce(job)?;
    ensure!(
        result.job_sha256 == job_sha,
        "encoding execution identity differs"
    );
    let value = serde_json::to_value(&result)?;
    backend.persist(
        &arm,
        &format!(
            "encoding-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        ),
        &value,
    )?;
    Ok(result)
}
/// `None` records unavailable context/proposal input and still evaluates generic.
/// Generic results are retained when the neural producer fails or times out.
pub fn run_pair<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &mut V,
    neural_input: Option<(&Comparison, &PolicyWorker)>,
) -> Result<PairedReceipt> {
    begin_pair(backend, plan, variation)?;
    let (neural, neural_error) = if let Some((comparison, worker)) = neural_input {
        let result = (|| {
            ensure!(
                serde_json::to_value(plan)? == serde_json::to_value(&comparison.input_plan)?,
                "neural outputs attached to different input plan"
            );
            run_neural(backend, comparison, variation, worker)
        })();
        match result {
            Ok(v) => (Some(v), None),
            Err(e) => (None, Some(format!("{e:#}"))),
        }
    } else {
        (
            None,
            Some("verified context or proposal input unavailable".into()),
        )
    };
    let (generic, generic_error) = match run_generic(backend, plan, variation) {
        Ok(v) => (Some(v), None),
        Err(e) => (None, Some(format!("{e:#}"))),
    };
    let receipt = PairedReceipt {
        comparison_sha256: sha256_hex(&serde_json::to_vec(plan)?),
        neural,
        neural_error,
        generic,
        generic_error,
    };
    let value = serde_json::to_value(&receipt)?;
    backend.persist(
        &plan.arm(true)?,
        &format!(
            "pair-inventory-{}",
            sha256_hex(&serde_json::to_vec(&value)?)
        ),
        &value,
    )?;
    Ok(receipt)
}

/// Private fitted-recipe identity frozen before the controller opens the outer
/// development scoring labels. It is selection evidence, not an outer win.
#[derive(Debug, Serialize, Deserialize)]
pub struct FrozenRecipe {
    pub candidate: Candidate,
    pub selection_feedback: Feedback,
    pub search_receipt_sha256: String,
    pub refit_request: EvaluationRequest,
}
pub fn freeze_winner<B: Backend>(
    backend: &mut B,
    comparison: &PairInputPlan,
    receipt: &SearchReceipt,
) -> Result<Option<FrozenRecipe>> {
    comparison.validate()?;
    ensure!(
        receipt.arm.id_sha256 == comparison.arm(true)?.id_sha256
            || receipt.arm.id_sha256 == comparison.arm(false)?.id_sha256,
        "winner arm differs from preregistered pair"
    );
    ensure!(
        receipt.comparison_sha256 == sha256_hex(&serde_json::to_vec(comparison)?)
            && receipt.selection_sha256 == comparison.selection.identity()?,
        "winner belongs to different comparison"
    );
    ensure!(
        !matches!(
            receipt.stop_reason.as_str(),
            "admission_pending_original_window" | "cancelled_original_window"
        ),
        "cannot freeze winner while admissions remain retryable"
    );
    let Some(winner) = receipt.winner() else {
        return Ok(None);
    };
    let mut feedback = winner.feedback.clone();
    feedback.enforce_selection(&comparison.selection, &winner.candidate);
    ensure!(
        feedback.status == EvaluationStatus::Completed,
        "winner lacks complete train-side evidence"
    );
    let mut refit_request = comparison.selection.refit_request(
        &winner.candidate.program_sha256,
        &winner.candidate.id_sha256,
    )?;
    refit_request.checkpoint_sha256 = receipt
        .proposal
        .as_ref()
        .map(|p| p.checkpoint_sha256.clone());
    refit_request.validate()?;
    let frozen = FrozenRecipe {
        candidate: winner.candidate.clone(),
        selection_feedback: feedback,
        search_receipt_sha256: sha256_hex(&serde_json::to_vec(receipt)?),
        refit_request,
    };
    backend.persist(
        &receipt.arm,
        "frozen-recipe",
        &serde_json::to_value(&frozen)?,
    )?;
    Ok(Some(frozen))
}

fn evaluate_unique<B: Backend>(
    backend: &mut B,
    arm: &ArmBudget,
    selection: &SelectionPlan,
    candidates: &[Candidate],
) -> Result<Vec<Feedback>> {
    let mut first = BTreeMap::new();
    let mut unique = vec![];
    let mut duplicate = vec![];
    for candidate in candidates {
        if let Some(previous) = first.get(&candidate.program_sha256) {
            duplicate.push(Feedback {
                candidate_sha256: candidate.id_sha256.clone(),
                status: EvaluationStatus::Failed,
                loss: None,
                physical_fit_starts: Some(0),
                worker_receipt_sha256: None,
                fitted_artifact_sha256: None,
                reason: Some(format!("canonical_duplicate_of:{previous}")),
                oof: None,
                retryable_inventory: false,
            });
        } else {
            first.insert(
                candidate.program_sha256.clone(),
                candidate.id_sha256.clone(),
            );
            unique.push(candidate.clone());
        }
    }
    let mut feedback = backend.evaluate(arm, selection, &unique)?;
    for item in &mut feedback {
        let candidate = unique
            .iter()
            .find(|c| c.id_sha256 == item.candidate_sha256)
            .context("unexpected candidate feedback")?;
        item.enforce_selection(selection, candidate);
    }
    feedback.extend(duplicate);
    Ok(feedback)
}

/// Read-only validation shared by execution and recovery reporting. No model
/// is loaded and no command/window is admitted here.
pub fn verify_policy_inputs(
    comparison: &Comparison,
    worker: &PolicyWorker,
) -> Result<super::search_context::VerifiedRegressionContext> {
    comparison.validate()?;
    ensure!(
        worker.request == comparison.producer_request
            && worker.expected_program_sha256 == comparison.installation.policy_program_sha256
            && worker.source_manifest.sha256 == comparison.installation.source_manifest_sha256
            && worker.checkpoint_receipt.sha256
                == comparison.installation.checkpoint_receipt_sha256
            && worker.context.receipt().sha256 == comparison.context_receipt_sha256,
        "policy worker differs from frozen paired installation/request"
    );
    worker.release_manifest.verify()?;
    ensure!(
        worker.release_manifest.sha256 == comparison.installation.release_manifest_sha256,
        "installation release receipt changed"
    );
    let context = worker.context.load_for_comparison(comparison)?;
    context.matches(comparison)?;
    worker.request.validate()?;
    ensure!(
        worker.request.prepared_sha256 == comparison.prepared_sha256
            && worker.request.training_sha256 == comparison.training_sha256
            && worker.request.cut == comparison.cut
            && worker.request.context_sha256 == context.sha256()
            && worker.request.vocabulary_sha256 == comparison.vocabulary_sha256,
        "policy request differs from comparison/context"
    );
    ensure!(
        worker.job.command.program.sha256 == worker.expected_program_sha256
            && jope_policy::is_sha256_hex(&worker.expected_program_sha256),
        "policy executable differs from installation pin"
    );
    worker.checkpoint_receipt.verify()?;
    let checkpoint = jope_policy::config::CheckpointBinding::from_receipt_bytes(&std::fs::read(
        &worker.checkpoint_receipt.path,
    )?)?;
    checkpoint.require_accepted_identity()?;
    ensure!(
        checkpoint.state_sha256() == worker.request.checkpoint_sha256
            && checkpoint.policy_safetensors_sha256() == worker.request.policy_safetensors_sha256,
        "policy checkpoint receipt changed"
    );
    worker.source_manifest.verify()?;
    ensure!(
        worker.request.worker_source_sha256 == worker.source_manifest.sha256,
        "policy source manifest changed"
    );
    worker.request_artifact.verify()?;
    let actual_request: jope_policy::producer::Request =
        serde_json::from_slice(&std::fs::read(&worker.request_artifact.path)?)?;
    ensure!(
        actual_request == worker.request,
        "policy request artifact differs"
    );
    for input in [
        &worker.source_manifest,
        &worker.request_artifact,
        &worker.checkpoint_receipt,
    ]
    .into_iter()
    .chain(worker.context.inputs())
    {
        ensure!(
            worker
                .job
                .command
                .inputs
                .iter()
                .any(|i| i.path == input.path && i.sha256 == input.sha256),
            "policy input missing from executable closure"
        );
    }
    ensure!(
        worker
            .job
            .command
            .inputs
            .iter()
            .any(|i| i.sha256 == worker.request.policy_safetensors_sha256)
            && worker
                .job
                .command
                .inputs
                .iter()
                .any(|i| i.sha256 == worker.request.vocabulary_sha256),
        "policy weights or vocabulary missing from closure"
    );
    ensure!(
        worker.job.outputs.contains(&worker.output_name),
        "policy output was not declared"
    );
    Ok(context)
}

fn run_neural<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &Comparison,
    variation: &mut V,
    worker: &PolicyWorker,
) -> Result<NeuralRun> {
    comparison.validate()?;
    let arm = comparison.arm(true)?;
    let deadline = backend.begin_arm(&arm)?;
    let context = verify_policy_inputs(comparison, worker)?;
    let mut job = worker.job.clone();
    job.arm = Some(arm.clone());
    let expected_job = job.identity()?;
    let expected_request = job.request.identity()?;
    let outcome = backend.produce(job)?;
    ensure!(
        outcome.job_sha256 == expected_job && outcome.request_sha256 == expected_request,
        "policy execution identity differs"
    );
    let nonattempt = retryable_inventory(&outcome)
        && matches!(
            outcome.command_status,
            EvaluationStatus::Cancelled | EvaluationStatus::Unavailable
        );
    let record = serde_json::to_value(&outcome)?;
    let name = if nonattempt {
        format!(
            "proposal-inventory-{}",
            sha256_hex(&serde_json::to_vec(&record)?)
        )
    } else {
        "proposal-execution".into()
    };
    backend.persist(&arm, &name, &record)?;
    if outcome.command_status != EvaluationStatus::Completed {
        return Ok(NeuralRun {
            proposal_execution: outcome,
            search: None,
        });
    }
    outcome.verify_outputs()?;
    let output_path = worker.job.directory.join(&worker.output_name);
    let artifact = outcome
        .outputs
        .iter()
        .find(|o| o.path == output_path)
        .context("policy producer artifact absent")?;
    artifact.verify()?;
    let producer =
        jope_policy::producer::read_claims(&std::fs::read(&artifact.path)?, &worker.request)?;
    let population = producer.population();
    ensure!(
        population.candidates().len() == INITIAL_SIZE,
        "policy must provide80 initial slots"
    );
    let candidates = population
        .candidates()
        .iter()
        .enumerate()
        .map(|(slot, seed)| {
            let origin = match (slot < NEURAL_SLOTS, seed.origin()) {
                (true, ORIGIN_NEURAL) => Origin::NeuralPolicy,
                (true, ORIGIN_NEURAL_FALLBACK) => Origin::NeuralRejectionFallback,
                (false, ORIGIN_RANDOM) => Origin::RandomExploration,
                _ => anyhow::bail!("incorrect origin in proposal slot {slot}"),
            };
            Candidate::new(
                &arm,
                seed.genome().into(),
                0,
                slot,
                origin,
                vec![],
                seed.rank_log_weight(),
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let evidence = ProposalEvidence {
        request: worker.request.clone(),
        context_cost: comparison.context_cost.clone(),
        producer_job_sha256: outcome.job_sha256.clone(),
        producer_output_sha256: artifact.sha256.clone(),
        population_sha256: producer.population_sha256().into(),
        context_sha256: context.sha256().into(),
        checkpoint_sha256: population.checkpoint_sha256().into(),
        ranking_backend: producer.ranking_backend().into(),
        native_cuda_parity_qualified: false,
    };
    let search = run_population(
        backend,
        comparison,
        variation,
        arm,
        deadline,
        candidates,
        Some(evidence),
    )?;
    Ok(NeuralRun {
        proposal_execution: outcome,
        search: Some(search),
    })
}

fn run_generic<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &mut V,
) -> Result<SearchReceipt> {
    let arm = comparison.arm(false)?;
    let deadline = backend.begin_arm(&arm)?;
    if let Some(saved) = backend.load(&arm, "initial")? {
        let candidates = serde_json::from_value(
            saved
                .get(1)
                .context("saved generic initial candidates missing")?
                .clone(),
        )?;
        return run_population(
            backend, comparison, variation, arm, deadline, candidates, None,
        );
    }
    let mut candidates = Vec::with_capacity(INITIAL_SIZE);
    let mut seen = BTreeSet::new();
    for attempt in 0..INITIAL_SIZE * 128 {
        if backend.now_ms() >= deadline {
            anyhow::bail!("generic generation exhausted original arm before initial population");
        }
        let Some(genome) = variation.generic(stream_seed(comparison.random_seed, 0, attempt))?
        else {
            continue;
        };
        let canonical = SynthesizedRecipe::new(genome.clone())?.executable_sha256;
        if seen.insert(canonical) {
            candidates.push(Candidate::new(
                &arm,
                genome,
                0,
                candidates.len(),
                Origin::Generic,
                vec![],
                None,
            )?);
        }
        if candidates.len() == INITIAL_SIZE {
            break;
        }
    }
    ensure!(
        candidates.len() == INITIAL_SIZE,
        "generic grammar cannot supply 80 distinct initial candidates"
    );
    run_population(
        backend, comparison, variation, arm, deadline, candidates, None,
    )
}

fn run_population<B: Backend, V: Variation>(
    backend: &mut B,
    comparison: &PairInputPlan,
    variation: &mut V,
    arm: ArmBudget,
    deadline: u64,
    mut candidates: Vec<Candidate>,
    proposal: Option<ProposalEvidence>,
) -> Result<SearchReceipt> {
    ensure!(
        candidates.len() == INITIAL_SIZE,
        "initial population is not 80"
    );
    let mut seen: BTreeSet<_> = candidates
        .iter()
        .map(|c| c.program_sha256.clone())
        .collect();
    let mut receipt = SearchReceipt {
        schema: PROTOCOL.into(),
        comparison_sha256: sha256_hex(&serde_json::to_vec(comparison)?),
        selection_sha256: comparison.selection.identity()?,
        arm,
        deadline_unix_ms: deadline,
        proposal,
        observations: vec![],
        stop_reason: String::new(),
    };
    backend.persist(
        &receipt.arm,
        "initial",
        &serde_json::to_value((&receipt.proposal, &candidates, deadline))?,
    )?;
    if let Some(value) = backend.load(&receipt.arm, "search-receipt")? {
        let saved: SearchReceipt = serde_json::from_value(value)?;
        saved.validate_population(comparison)?;
        ensure!(
            saved.schema == PROTOCOL
                && saved.comparison_sha256 == receipt.comparison_sha256
                && saved.selection_sha256 == receipt.selection_sha256
                && saved.arm.id_sha256 == receipt.arm.id_sha256
                && saved.deadline_unix_ms == deadline
                && serde_json::to_value(&saved.proposal)?
                    == serde_json::to_value(&receipt.proposal)?,
            "saved search binding changed"
        );
        ensure!(
            saved.observations.len() >= INITIAL_SIZE
                && serde_json::to_value(
                    saved.observations[..INITIAL_SIZE]
                        .iter()
                        .map(|o| &o.candidate)
                        .collect::<Vec<_>>()
                )? == serde_json::to_value(&candidates)?,
            "saved initial population changed"
        );
        // Recheck the execution cache and fitted outputs; no deadline reset and
        // no receipt-only claim of completed work.
        for group in saved
            .observations
            .chunk_by(|a, b| a.candidate.generation == b.candidate.generation)
        {
            let batch = group
                .iter()
                .map(|o| o.candidate.clone())
                .collect::<Vec<_>>();
            let feedback = evaluate_unique(backend, &receipt.arm, &comparison.selection, &batch)?;
            let current = feedback
                .into_iter()
                .map(|f| Ok((f.candidate_sha256.clone(), serde_json::to_value(f)?)))
                .collect::<Result<BTreeMap<_, _>>>()?;
            ensure!(
                current.len() == group.len(),
                "saved execution cache changed"
            );
            for observation in group {
                ensure!(
                    current
                        .get(&observation.candidate.id_sha256)
                        .context("saved worker outcome missing")?
                        == &serde_json::to_value(&observation.feedback)?,
                    "saved fitted evidence changed"
                );
            }
        }
        return Ok(saved);
    }
    for generation in 0..1000 {
        backend.persist(
            &receipt.arm,
            &format!("generation-{generation:04}-scheduled"),
            &serde_json::to_value(&candidates)?,
        )?;
        // The pool returns intact cached successes even after the deadline; new
        // jobs receive explicit budget results. Never turn cached fits into zero.
        let feedback = evaluate_unique(backend, &receipt.arm, &comparison.selection, &candidates)?;
        let mut by_id = BTreeMap::new();
        for item in feedback {
            ensure!(
                by_id.insert(item.candidate_sha256.clone(), item).is_none(),
                "duplicate fit feedback"
            );
        }
        ensure!(
            by_id.len() == candidates.len(),
            "missing or extra fit feedback"
        );
        let mut observations = Vec::with_capacity(candidates.len());
        for candidate in candidates {
            let feedback = by_id
                .remove(&candidate.id_sha256)
                .context("fit feedback bound to different candidate")?;
            observations.push(Observation {
                candidate,
                feedback,
            });
        }
        if observations.iter().any(|o| o.feedback.retryable_inventory) {
            let inventory = serde_json::to_value(&observations)?;
            backend.persist(
                &receipt.arm,
                &format!(
                    "generation-{generation:04}-inventory-{}",
                    sha256_hex(&serde_json::to_vec(&inventory)?)
                ),
                &inventory,
            )?;
            receipt.observations.extend(observations);
            receipt.validate_population(comparison)?;
            receipt.stop_reason = "admission_pending_original_window".into();
            return Ok(receipt); // no immutable observed/final receipt: resume may admit later
        }
        backend.persist(
            &receipt.arm,
            &format!("generation-{generation:04}-observed"),
            &serde_json::to_value(&observations)?,
        )?;
        receipt.observations.extend(observations);
        if let Some(saved) = backend.load(
            &receipt.arm,
            &format!("generation-{:04}-scheduled", generation + 1),
        )? {
            candidates = serde_json::from_value(saved)?;
            ensure!(
                !candidates.is_empty()
                    && candidates.len() <= INITIAL_SIZE - ELITES
                    && receipt.observations.len() + candidates.len() <= comparison.max_evaluations,
                "saved offspring count changed"
            );
            for (slot, child) in candidates.iter().enumerate() {
                ensure!(
                    child.generation == generation + 1
                        && matches!(child.origin, Origin::Mutation | Origin::Crossover)
                        && !child.parents.is_empty(),
                    "saved offspring origin changed"
                );
                ensure!(
                    child
                        .parents
                        .iter()
                        .all(|id| receipt
                            .observations
                            .iter()
                            .any(|o| o.candidate.id_sha256 == *id
                                && o.feedback.status == EvaluationStatus::Completed)),
                    "saved child lacks measured parent"
                );
                let expected = Candidate::new(
                    &receipt.arm,
                    child.genome.clone(),
                    generation + 1,
                    slot,
                    child.origin.clone(),
                    child.parents.clone(),
                    None,
                )?;
                ensure!(
                    serde_json::to_value(child)? == serde_json::to_value(expected)?
                        && seen.insert(child.program_sha256.clone()),
                    "saved offspring identity changed"
                );
            }
            continue;
        }
        if backend.cancelled() {
            receipt.validate_population(comparison)?;
            receipt.stop_reason = "cancelled_original_window".into();
            return Ok(receipt); // terminal attempts stay, but future generations may resume
        }
        if backend.now_ms() >= deadline {
            receipt.stop_reason = "original_deadline".into();
            break;
        }
        if receipt.observations.len() >= comparison.max_evaluations {
            receipt.stop_reason = "evaluation_limit".into();
            break;
        }
        let mut parents = receipt
            .observations
            .iter()
            .filter(|o| o.feedback.status == EvaluationStatus::Completed)
            .collect::<Vec<_>>();
        parents.sort_by(|a, b| {
            a.feedback
                .loss
                .unwrap()
                .total_cmp(&b.feedback.loss.unwrap())
                .then_with(|| a.candidate.program_sha256.cmp(&b.candidate.program_sha256))
        });
        parents.truncate(ELITES);
        if parents.is_empty() {
            receipt.stop_reason = "no_measured_parent".into();
            break;
        }
        let desired =
            (INITIAL_SIZE - ELITES).min(comparison.max_evaluations - receipt.observations.len());
        candidates = Vec::with_capacity(desired);
        for attempt in 0..desired * 128 {
            if backend.now_ms() >= deadline {
                break;
            }
            let seed = stream_seed(comparison.random_seed, generation + 1, attempt);
            let left = &parents[attempt % parents.len()].candidate;
            let right = &parents[(attempt + 1) % parents.len()].candidate;
            let (genome, origin, ancestors) = if attempt % 2 == 0 {
                (
                    variation.mutate(&left.genome, seed),
                    Origin::Mutation,
                    vec![left.id_sha256.clone()],
                )
            } else {
                (
                    variation.crossover(&left.genome, &right.genome, seed),
                    Origin::Crossover,
                    vec![left.id_sha256.clone(), right.id_sha256.clone()],
                )
            };
            let genome = match genome {
                Ok(value) => value,
                Err(error) => {
                    backend.persist(&receipt.arm,&format!("generation-{}-variation-rejected-{attempt:06}",generation+1),&serde_json::json!({"seed":seed,"origin":origin,"parents":ancestors,"error":format!("{error:#}")}))?;
                    continue;
                }
            };
            let canonical = SynthesizedRecipe::new(genome.clone())?.executable_sha256;
            if seen.insert(canonical.clone()) {
                backend.persist(&receipt.arm,&format!("generation-{}-variation-accepted-{:06}",generation+1,candidates.len()),&serde_json::json!({"seed":seed,"attempt":attempt,"variation":comparison.variation_identity,"origin":origin,"parents":ancestors,"canonical_executable_sha256":canonical,"genome":genome}))?;
                candidates.push(Candidate::new(
                    &receipt.arm,
                    genome,
                    generation + 1,
                    candidates.len(),
                    origin,
                    ancestors,
                    None,
                )?);
            }
            if candidates.len() == desired {
                break;
            }
        }
        if candidates.is_empty() {
            receipt.stop_reason = if backend.now_ms() >= deadline {
                "original_deadline"
            } else {
                "search_space_exhausted"
            }
            .into();
            break;
        }
    }
    receipt.validate_population(comparison)?;
    backend.persist(
        &receipt.arm,
        "search-receipt",
        &serde_json::to_value(&receipt)?,
    )?;
    Ok(receipt)
}

fn stream_seed(base: u64, generation: usize, attempt: usize) -> u64 {
    let digest = sha256_hex(format!("{PROTOCOL}/{base}/{generation}/{attempt}").as_bytes());
    u64::from_str_radix(&digest[..16], 16).expect("sha256 hex")
}

/// Concrete bridge to the one shared worker pool. The job builder owns the
/// fit-only input closure and must declare every output. The scorer verifies
/// fitted output bytes and performs development scoring outside the worker.
/// Pool access is message-based in the panel coordinator; only its supervisor
/// thread owns ExecutionPool. This trait does not authorize direct engine fits.
pub trait PoolAccess {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64>;
    fn run(&mut self, jobs: Vec<WorkerJob>, cancelled: &AtomicBool) -> Result<Vec<WorkerOutcome>>;
}
impl PoolAccess for ExecutionPool {
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        ExecutionPool::begin_arm(self, arm)
    }
    fn run(&mut self, jobs: Vec<WorkerJob>, cancelled: &AtomicBool) -> Result<Vec<WorkerOutcome>> {
        ExecutionPool::run(self, jobs, cancelled)
    }
}
pub struct PoolBackend<'a, J, S, P = ExecutionPool> {
    pub pool: &'a mut P,
    pub cancelled: &'a AtomicBool,
    pub journal: PathBuf,
    pub make_job: J,
    pub score: S,
}
impl<J, S, P: PoolAccess> Backend for PoolBackend<'_, J, S, P>
where
    J: FnMut(&Candidate, &EvaluationRequest) -> Result<WorkerJob>,
    S: FnMut(&Candidate, &[WorkerOutcome], &SelectionPlan) -> Result<Feedback>,
{
    fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
        self.pool.begin_arm(arm)
    }
    fn cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::Acquire)
    }
    fn now_ms(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock before epoch")
            .as_millis()
            .try_into()
            .unwrap_or(u64::MAX)
    }
    fn produce(&mut self, job: WorkerJob) -> Result<WorkerOutcome> {
        let mut outcomes = self.pool.run(vec![job], self.cancelled)?;
        ensure!(outcomes.len() == 1, "missing policy execution outcome");
        Ok(outcomes.remove(0))
    }
    fn evaluate(
        &mut self,
        arm: &ArmBudget,
        selection: &SelectionPlan,
        candidates: &[Candidate],
    ) -> Result<Vec<Feedback>> {
        selection.validate()?;
        let mut requests = BTreeMap::new();
        let mut jobs = vec![];
        let mut preparation_failures = BTreeMap::new();
        for (candidate_index, candidate) in candidates.iter().enumerate() {
            let mut prepared = vec![];
            for (fold, expected) in selection
                .selection_requests(&candidate.program_sha256, &candidate.id_sha256)?
                .into_iter()
                .enumerate()
            {
                let mut job = match (self.make_job)(candidate, &expected) {
                    Ok(job) => job,
                    Err(error) => {
                        preparation_failures.insert(
                            candidate.id_sha256.clone(),
                            Feedback {
                                candidate_sha256: candidate.id_sha256.clone(),
                                status: EvaluationStatus::Unavailable,
                                loss: None,
                                physical_fit_starts: Some(0),
                                worker_receipt_sha256: None,
                                fitted_artifact_sha256: None,
                                reason: Some(format!("job preparation unavailable: {error:#}")),
                                oof: None,
                                retryable_inventory: true,
                            },
                        );
                        break;
                    }
                };
                ensure!(
                    job.request.identity()? == expected.identity()?,
                    "fit factory changed train-side request/seed/recipe"
                );
                job.arm = Some(arm.clone());
                prepared.push((fold, expected.identity()?, job));
            }
            if preparation_failures.contains_key(&candidate.id_sha256) {
                continue;
            }
            for (fold, id, job) in prepared {
                ensure!(
                    requests.insert(id, (candidate_index, fold)).is_none(),
                    "candidate jobs reused request identity"
                );
                jobs.push(job);
            }
        }
        let mut outcomes: Vec<BTreeMap<usize, WorkerOutcome>> =
            (0..candidates.len()).map(|_| BTreeMap::new()).collect();
        for outcome in self.pool.run(jobs, self.cancelled)? {
            let (candidate, fold) = requests
                .remove(&outcome.request_sha256)
                .context("unrequested worker outcome")?;
            ensure!(
                outcomes[candidate].insert(fold, outcome).is_none(),
                "duplicate inner fit outcome"
            );
        }
        ensure!(requests.is_empty(), "missing inner fit outcome");
        candidates
            .iter()
            .zip(outcomes)
            .map(|(candidate, folds)| {
                if let Some(failure) = preparation_failures.remove(&candidate.id_sha256) {
                    return Ok(failure);
                }
                let folds = folds.into_values().collect::<Vec<_>>();
                ensure!(folds.len() == 5, "incomplete inner fold inventory");
                let receipt_sha = sha256_hex(&serde_json::to_vec(&folds)?);
                let retryable = folds.iter().any(retryable_inventory);
                if let Some(failed) = folds
                    .iter()
                    .find(|o| o.command_status != EvaluationStatus::Completed)
                {
                    return Ok(Feedback {
                        candidate_sha256: candidate.id_sha256.clone(),
                        status: failed.command_status,
                        loss: None,
                        physical_fit_starts: folds.iter().all(unstarted).then_some(0),
                        worker_receipt_sha256: Some(receipt_sha),
                        fitted_artifact_sha256: None,
                        reason: failed.reason.clone(),
                        oof: None,
                        retryable_inventory: retryable,
                    });
                }
                for outcome in &folds {
                    outcome.verify_outputs()?;
                }
                let mut feedback = match (self.score)(candidate, &folds, selection) {
                    Ok(feedback) => feedback,
                    Err(error) => Feedback {
                        candidate_sha256: candidate.id_sha256.clone(),
                        status: EvaluationStatus::Failed,
                        loss: None,
                        physical_fit_starts: None,
                        worker_receipt_sha256: Some(receipt_sha.clone()),
                        fitted_artifact_sha256: None,
                        reason: Some(format!("controller OOF scoring failed: {error:#}")),
                        oof: None,
                        retryable_inventory: false,
                    },
                };
                ensure!(
                    feedback.candidate_sha256 == candidate.id_sha256,
                    "scorer changed candidate identity"
                );
                feedback.worker_receipt_sha256 = Some(receipt_sha);
                feedback.retryable_inventory = false;
                feedback.enforce_selection(selection, candidate);
                Ok(feedback)
            })
            .collect()
    }
    fn persist(&mut self, arm: &ArmBudget, name: &str, value: &serde_json::Value) -> Result<()> {
        persist_immutable(&self.journal.join(&arm.id_sha256), name, value)
    }
    fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>> {
        let path = self
            .journal
            .join(&arm.id_sha256)
            .join(format!("{name}.json"));
        if !path.exists() {
            return Ok(None);
        }
        ensure!(
            std::fs::symlink_metadata(&path)?.file_type().is_file(),
            "search journal entry is not a physical file"
        );
        Ok(Some(serde_json::from_slice(&std::fs::read(path)?)?))
    }
}

fn retryable_inventory(outcome: &WorkerOutcome) -> bool {
    matches!(
        outcome.command_status,
        EvaluationStatus::Unavailable | EvaluationStatus::Cancelled
    ) && outcome.consumption.gate_released_unix_ms.is_none()
        && outcome.exit_code.is_none()
        && outcome.signal.is_none()
        && !outcome.reason.as_ref().is_some_and(|r| {
            r.starts_with("controller_interrupted:") || r.starts_with("original attempt unknown;")
        })
}

fn unstarted(outcome: &WorkerOutcome) -> bool {
    outcome.consumption.gate_released_unix_ms.is_none()
        && outcome.exit_code.is_none()
        && outcome.signal.is_none()
        && outcome.consumption.cpu_seconds == Some(0.)
        && outcome
            .consumption
            .accounting_reason
            .starts_with("not_started:")
}

fn persist_immutable(directory: &Path, name: &str, value: &serde_json::Value) -> Result<()> {
    use std::{io::Write, os::unix::fs::OpenOptionsExt};
    ensure!(
        directory.is_absolute() && !directory.starts_with("/tmp"),
        "search journal needs durable absolute location"
    );
    ensure!(
        !name.is_empty() && name.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-'),
        "invalid search record name"
    );
    std::fs::create_dir_all(directory)?;
    ensure!(
        std::fs::canonicalize(directory)? == directory,
        "search journal traverses symlink"
    );
    let path = directory.join(format!("{name}.json"));
    let bytes = serde_json::to_vec(value)?;
    if path.exists() {
        ensure!(
            std::fs::symlink_metadata(&path)?.file_type().is_file(),
            "search record is not a physical file"
        );
        ensure!(
            std::fs::read(&path)? == bytes,
            "search resume evidence differs: {}",
            path.display()
        );
        return Ok(());
    }
    let temporary = directory.join(format!(
        ".{name}.{}-{}.pending",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos()
    ));
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .mode(0o600)
        .write(true)
        .open(&temporary)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    std::fs::rename(&temporary, &path)?;
    std::fs::File::open(directory)?.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, rc::Rc};

    struct FixtureBackend {
        time: u64,
        started: Rc<Cell<bool>>,
        deadlines: BTreeMap<String, u64>,
        cache: BTreeMap<String, Feedback>,
        journal: BTreeMap<String, serde_json::Value>,
        dispatched: Vec<Candidate>,
        unknown_counter: bool,
        admission_unavailable: bool,
    }
    impl Default for FixtureBackend {
        fn default() -> Self {
            Self {
                time: 1000,
                started: Rc::new(Cell::new(false)),
                deadlines: BTreeMap::new(),
                cache: BTreeMap::new(),
                journal: BTreeMap::new(),
                dispatched: vec![],
                unknown_counter: false,
                admission_unavailable: false,
            }
        }
    }
    impl Backend for FixtureBackend {
        fn begin_arm(&mut self, arm: &ArmBudget) -> Result<u64> {
            self.started.set(true);
            Ok(*self
                .deadlines
                .entry(arm.id_sha256.clone())
                .or_insert(self.time + arm.seconds * 1000))
        }
        fn now_ms(&self) -> u64 {
            self.time
        }
        fn produce(&mut self, _job: WorkerJob) -> Result<WorkerOutcome> {
            anyhow::bail!("synthetic coordinator fixture never runs a producer")
        }
        fn evaluate(
            &mut self,
            arm: &ArmBudget,
            selection: &SelectionPlan,
            candidates: &[Candidate],
        ) -> Result<Vec<Feedback>> {
            if self.admission_unavailable {
                self.admission_unavailable = false;
                return Ok(candidates
                    .iter()
                    .map(|c| Feedback {
                        candidate_sha256: c.id_sha256.clone(),
                        status: EvaluationStatus::Unavailable,
                        loss: None,
                        physical_fit_starts: Some(0),
                        worker_receipt_sha256: None,
                        fitted_artifact_sha256: None,
                        reason: Some("synthetic missing dependency".into()),
                        oof: None,
                        retryable_inventory: true,
                    })
                    .collect());
            }
            candidates
                .iter()
                .map(|candidate| {
                    if let Some(cached) = self.cache.get(&candidate.id_sha256) {
                        return Ok(cached.clone());
                    }
                    if self.time >= self.deadlines[&arm.id_sha256] {
                        return Ok(Feedback::exhausted(candidate));
                    }
                    self.dispatched.push(candidate.clone());
                    // Synthetic callback evidence; never a policy/native fit claim.
                    let oof = OofEvidence::for_candidate(
                        selection,
                        &candidate.program_sha256,
                        &candidate.id_sha256,
                        vec![1; 5],
                        vec![sha256_hex(candidate.genome.as_bytes()); 5],
                    )?;
                    let fitted_sha = oof.fitted_bundle_sha256()?;
                    let feedback = Feedback {
                        candidate_sha256: candidate.id_sha256.clone(),
                        status: EvaluationStatus::Completed,
                        loss: Some(1. + candidate.initial_slot.unwrap_or(0) as f64 / 100.),
                        physical_fit_starts: if self.unknown_counter { None } else { Some(5) },
                        worker_receipt_sha256: Some(sha256_hex(b"synthetic callback receipt")),
                        fitted_artifact_sha256: Some(fitted_sha),
                        reason: None,
                        oof: Some(oof),
                        retryable_inventory: false,
                    };
                    self.cache
                        .insert(candidate.id_sha256.clone(), feedback.clone());
                    Ok(feedback)
                })
                .collect()
        }
        fn persist(
            &mut self,
            arm: &ArmBudget,
            name: &str,
            value: &serde_json::Value,
        ) -> Result<()> {
            let key = format!("{}/{name}", arm.id_sha256);
            if let Some(old) = self.journal.get(&key) {
                ensure!(old == value, "fixture stale journal");
            } else {
                self.journal.insert(key, value.clone());
            }
            Ok(())
        }
        fn load(&self, arm: &ArmBudget, name: &str) -> Result<Option<serde_json::Value>> {
            Ok(self
                .journal
                .get(&format!("{}/{name}", arm.id_sha256))
                .cloned())
        }
    }
    struct FixtureVariation {
        started: Rc<Cell<bool>>,
    }
    impl Variation for FixtureVariation {
        fn identity(&self) -> &str {
            "synthetic variation/v1"
        }
        fn generic_policy(&self) -> &str {
            "synthetic uniform/v1"
        }
        fn generic(&mut self, seed: u64) -> Result<Option<String>> {
            ensure!(
                self.started.get(),
                "generation happened before original clock"
            );
            Ok(Some(format!(
                "(pipe X (ridge lambda={}))",
                1 + seed % 1_000_000_000
            )))
        }
        fn mutate(&mut self, _parent: &str, seed: u64) -> Result<String> {
            Ok(format!(
                "(pipe X (standardize) (ridge lambda={}))",
                1 + seed % 1_000_000_000
            ))
        }
        fn crossover(&mut self, _left: &str, _right: &str, seed: u64) -> Result<String> {
            Ok(format!(
                "(pipe X (robust_scale) (ridge lambda={}))",
                1 + seed % 1_000_000_000
            ))
        }
    }
    fn comparison(max_evaluations: usize) -> PairInputPlan {
        PairInputPlan {
            selection: crate::search_plan::fixture(),
            preprocessing_sha256: sha256_hex(b"synthetic preprocessing"),
            raw_input_policy: RAW_INPUT_POLICY.into(),
            fit_randomness_policy: FIT_RANDOMNESS_POLICY.into(),
            installation: InstallationPins {
                release_manifest_sha256: sha256_hex(b"synthetic release"),
                policy_program_sha256: sha256_hex(b"synthetic program"),
                source_manifest_sha256: sha256_hex(b"synthetic source"),
                checkpoint_receipt_sha256: sha256_hex(b"synthetic checkpoint"),
            },
            encoding: EncodingPlan {
                input_sha256: sha256_hex(b"synthetic encoding input"),
                source_sha40: "0".repeat(40),
                program_sha256: sha256_hex(b"synthetic encoder program"),
                source_manifest_sha256: sha256_hex(b"synthetic encoder source"),
                encoder_assets_manifest_sha256: sha256_hex(b"synthetic encoder assets"),
                qualification_sha256: None,
                reference_manifest_sha256: None,
                backend: "synthetic fixture".into(),
            },
            proposal: jope_policy::producer::RequestPlan {
                worker_source_sha256: sha256_hex(b"synthetic source"),
                prepared_sha256: sha256_hex(b"prepared"),
                training_sha256: sha256_hex(b"training"),
                cut: 0,
                checkpoint_sha256: jope_policy::ACCEPTED_STATE_SHA256.into(),
                policy_safetensors_sha256:
                    "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
                vocabulary_sha256: jope_policy::PRODUCT_VOCABULARY_SHA256.into(),
                sampling_seed: 1,
                fallback_seed: 2,
                exploration_seed: 3,
                sampling_rounds: 1,
                sampling_batch: 8,
                sampling_temperature: 1.,
            },
            variation_identity: "synthetic variation/v1".into(),
            generic_sampling_policy: "synthetic uniform/v1".into(),
            checkpoint_training_exposure: "unknown".into(),
            exposure_evidence_sha256: None,
            prepared_sha256: sha256_hex(b"prepared"),
            training_sha256: sha256_hex(b"training"),
            cut: 0,
            vocabulary_sha256: jope_policy::PRODUCT_VOCABULARY_SHA256.into(),
            seconds: 120,
            max_evaluations,
            random_seed: 71,
        }
    }
    fn synthetic_initial(arm: &ArmBudget) -> Vec<Candidate> {
        (0..INITIAL_SIZE)
            .map(|slot| {
                Candidate::new(
                    arm,
                    format!("(pipe X (ridge lambda={}))", slot + 1),
                    0,
                    slot,
                    if slot < 60 {
                        Origin::NeuralPolicy
                    } else if slot < 64 {
                        Origin::NeuralRejectionFallback
                    } else {
                        Origin::RandomExploration
                    },
                    vec![],
                    None,
                )
                .unwrap()
            })
            .collect()
    }

    // Metadata-only fixtures: callback counters/hashes are explicitly synthetic,
    // not receipts from models or fitted workers.
    fn metadata_receipt(plan: &PairInputPlan, neural: bool) -> Result<SearchReceipt> {
        let arm = plan.arm(neural)?;
        let candidates = if neural {
            synthetic_initial(&arm)
        } else {
            (0..INITIAL_SIZE)
                .map(|slot| {
                    Candidate::new(
                        &arm,
                        format!("(pipe X (ridge lambda={}))", slot + 1),
                        0,
                        slot,
                        Origin::Generic,
                        vec![],
                        None,
                    )
                })
                .collect::<Result<Vec<_>>>()?
        };
        let mut backend = FixtureBackend::default();
        let deadline = backend.begin_arm(&arm)?;
        let feedback = backend.evaluate(&arm, &plan.selection, &candidates)?;
        Ok(SearchReceipt {
            schema: PROTOCOL.into(),
            comparison_sha256: sha256_hex(&serde_json::to_vec(plan)?),
            selection_sha256: plan.selection.identity()?,
            arm,
            deadline_unix_ms: deadline,
            proposal: None,
            observations: candidates
                .into_iter()
                .zip(feedback)
                .map(|(candidate, feedback)| Observation {
                    candidate,
                    feedback,
                })
                .collect(),
            stop_reason: "synthetic_metadata_fixture".into(),
        })
    }
    fn append_metadata_child(
        receipt: &mut SearchReceipt,
        plan: &PairInputPlan,
        generation: usize,
        slot: usize,
        origin: Origin,
        parents: &[usize],
    ) -> Result<()> {
        let candidate = Candidate::new(
            &receipt.arm,
            format!(
                "(pipe X (standardize) (ridge lambda={}))",
                receipt.observations.len() + 1000
            ),
            generation,
            slot,
            origin,
            parents
                .iter()
                .map(|i| receipt.observations[*i].candidate.id_sha256.clone())
                .collect(),
            None,
        )?;
        let mut backend = FixtureBackend::default();
        backend.begin_arm(&receipt.arm)?;
        let feedback = backend
            .evaluate(
                &receipt.arm,
                &plan.selection,
                std::slice::from_ref(&candidate),
            )?
            .remove(0);
        receipt.observations.push(Observation {
            candidate,
            feedback,
        });
        Ok(())
    }
    fn copy_receipt(receipt: &SearchReceipt) -> SearchReceipt {
        serde_json::from_value(serde_json::to_value(receipt).unwrap()).unwrap()
    }
    fn legacy_candidate_checks_pass(receipt: &SearchReceipt) -> bool {
        receipt.observations.iter().all(|o| {
            o.candidate.recipe.validate().is_ok()
                && o.candidate.program_sha256 == o.candidate.recipe.executable_sha256
                && o.feedback.candidate_sha256 == o.candidate.id_sha256
        })
    }

    #[test]
    fn population_replay_rejects_forged_identity_and_duplicate_or_missing_slots() -> Result<()> {
        let plan = comparison(84);
        let original = metadata_receipt(&plan, true)?;
        original.validate_population(&plan)?;
        let mut forged = copy_receipt(&original);
        forged.observations[0].candidate.id_sha256 = sha256_hex(b"forged candidate");
        forged.observations[0].feedback.candidate_sha256 =
            forged.observations[0].candidate.id_sha256.clone();
        assert!(legacy_candidate_checks_pass(&forged)); // concrete old report gap
        assert!(forged.validate_population(&plan).is_err());
        let mut repeated = copy_receipt(&original);
        repeated.observations[1] = repeated.observations[0].clone();
        assert!(legacy_candidate_checks_pass(&repeated));
        assert!(repeated.validate_population(&plan).is_err());
        let mut missing = copy_receipt(&original);
        missing.observations.remove(79);
        assert!(legacy_candidate_checks_pass(&missing));
        assert!(missing.validate_population(&plan).is_err());
        let mut wrong_slot = copy_receipt(&original);
        wrong_slot.observations[1].candidate.initial_slot = Some(0);
        assert!(legacy_candidate_checks_pass(&wrong_slot));
        assert!(wrong_slot.validate_population(&plan).is_err());
        let mut wrong_genome = copy_receipt(&original);
        wrong_genome.observations[0].candidate.genome = "(pipe X (ridge lambda=999))".into();
        assert!(legacy_candidate_checks_pass(&wrong_genome));
        assert!(wrong_genome.validate_population(&plan).is_err());
        let mut wrong_origin = copy_receipt(&original);
        let candidate = Candidate::new(
            &original.arm,
            original.observations[0].candidate.genome.clone(),
            0,
            0,
            Origin::RandomExploration,
            vec![],
            None,
        )?;
        wrong_origin.observations[0] = Observation {
            feedback: Feedback::exhausted(&candidate),
            candidate,
        };
        assert!(legacy_candidate_checks_pass(&wrong_origin));
        assert!(wrong_origin.validate_population(&plan).is_err());
        Ok(())
    }

    #[test]
    fn offspring_from_exploration_or_fallback_does_not_prove_neural_evolution() -> Result<()> {
        let plan = comparison(84);
        let mut receipt = metadata_receipt(&plan, true)?;
        append_metadata_child(&mut receipt, &plan, 1, 0, Origin::Mutation, &[64])?;
        append_metadata_child(&mut receipt, &plan, 1, 1, Origin::Crossover, &[60, 65])?;
        append_metadata_child(&mut receipt, &plan, 2, 0, Origin::Mutation, &[80])?;
        receipt.validate_population(&plan)?;
        assert!(receipt
            .observations
            .iter()
            .any(|o| o.candidate.origin == Origin::NeuralPolicy
                && o.feedback.status == EvaluationStatus::Completed));
        let legacy_evolved = receipt.observations.iter().any(|o| {
            o.candidate.generation > 0 && o.feedback.status == EvaluationStatus::Completed
        });
        assert!(legacy_evolved); // old combined flag falsely claimed neural evolution
        assert!(!receipt.has_fitted_neural_descendant());
        append_metadata_child(&mut receipt, &plan, 2, 1, Origin::Crossover, &[81, 0])?;
        receipt.validate_population(&plan)?;
        assert!(receipt.has_fitted_neural_descendant());
        Ok(())
    }

    #[test]
    fn population_replay_requires_completed_parents_from_earlier_generations() -> Result<()> {
        let plan = comparison(84);
        let mut receipt = metadata_receipt(&plan, true)?;
        append_metadata_child(&mut receipt, &plan, 1, 0, Origin::Mutation, &[0])?;
        receipt.validate_population(&plan)?;
        assert!(receipt.has_fitted_neural_descendant());
        let mut failed_parent = copy_receipt(&receipt);
        failed_parent.observations[0].feedback =
            Feedback::exhausted(&failed_parent.observations[0].candidate);
        assert!(legacy_candidate_checks_pass(&failed_parent));
        assert!(failed_parent.validate_population(&plan).is_err());
        assert!(!failed_parent.has_fitted_neural_descendant());
        let mut same_generation = copy_receipt(&receipt);
        append_metadata_child(&mut same_generation, &plan, 1, 1, Origin::Mutation, &[80])?;
        assert!(legacy_candidate_checks_pass(&same_generation));
        assert!(same_generation.validate_population(&plan).is_err());
        let mut wrong_arity = metadata_receipt(&plan, true)?;
        append_metadata_child(&mut wrong_arity, &plan, 1, 0, Origin::Mutation, &[0, 1])?;
        assert!(legacy_candidate_checks_pass(&wrong_arity));
        assert!(wrong_arity.validate_population(&plan).is_err());
        let mut skipped_generation = metadata_receipt(&plan, true)?;
        append_metadata_child(&mut skipped_generation, &plan, 2, 0, Origin::Mutation, &[0])?;
        assert!(skipped_generation.validate_population(&plan).is_err());
        Ok(())
    }

    #[test]
    fn population_replay_preserves_neural_duplicates_but_rejects_generic_duplicates() -> Result<()>
    {
        let plan = comparison(80);
        let mut neural = metadata_receipt(&plan, true)?;
        let duplicate = Candidate::new(
            &neural.arm,
            neural.observations[0].candidate.genome.clone(),
            0,
            1,
            Origin::NeuralPolicy,
            vec![],
            None,
        )?;
        neural.observations[1] = Observation {
            feedback: Feedback::exhausted(&duplicate),
            candidate: duplicate,
        };
        neural.validate_population(&plan)?;
        let mut generic = metadata_receipt(&plan, false)?;
        let duplicate = Candidate::new(
            &generic.arm,
            generic.observations[0].candidate.genome.clone(),
            0,
            1,
            Origin::Generic,
            vec![],
            None,
        )?;
        generic.observations[1] = Observation {
            feedback: Feedback::exhausted(&duplicate),
            candidate: duplicate,
        };
        assert!(legacy_candidate_checks_pass(&generic));
        assert!(generic.validate_population(&plan).is_err());
        generic = metadata_receipt(&plan, false)?;
        generic.arm.seconds += 1;
        assert!(generic.validate_population(&plan).is_err());
        Ok(())
    }

    #[test]
    fn resume_rejects_forged_offspring_metadata_before_cached_fit_replay() -> Result<()> {
        let plan = comparison(84);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let mut saved = run_generic(&mut backend, &plan, &mut variation)?;
        let original_deadline = saved.deadline_unix_ms;
        saved.observations[80].candidate.parents = vec![sha256_hex(b"absent parent")];
        // The old completed-receipt path rechecked cached feedback by unchanged
        // candidate ID, but never reconstructed the child's changed parents.
        assert!(legacy_candidate_checks_pass(&saved));
        backend.journal.insert(
            format!("{}/search-receipt", saved.arm.id_sha256),
            serde_json::to_value(saved)?,
        );
        assert!(run_generic(&mut backend, &plan, &mut variation).is_err());
        assert_eq!(backend.dispatched.len(), 84);
        assert_eq!(
            backend.deadlines[&plan.arm(false)?.id_sha256],
            original_deadline
        );
        Ok(())
    }

    #[test]
    fn synthetic_neural_slots_reach_callbacks_and_evolution_with_origins_intact() -> Result<()> {
        let plan = comparison(84);
        let arm = plan.arm(true)?;
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let deadline = backend.begin_arm(&arm)?;
        let report = run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm.clone(),
            deadline,
            synthetic_initial(&arm),
            None,
        )?;
        assert_eq!(backend.dispatched.len(), 84);
        assert_eq!(
            backend
                .dispatched
                .iter()
                .filter(|c| c.generation == 0)
                .count(),
            80
        );
        assert_eq!(
            backend
                .dispatched
                .iter()
                .filter(|c| c.origin == Origin::NeuralPolicy)
                .count(),
            60
        );
        assert_eq!(
            backend
                .dispatched
                .iter()
                .filter(|c| c.origin == Origin::NeuralRejectionFallback)
                .count(),
            4
        );
        assert_eq!(
            backend
                .dispatched
                .iter()
                .filter(|c| c.origin == Origin::RandomExploration)
                .count(),
            16
        );
        assert_eq!(report.neural_physical_fit_starts(), Some(300));
        for child in &backend.dispatched[80..] {
            assert_eq!(child.generation, 1);
            assert!(child.initial_slot.is_none());
            assert!(child
                .parents
                .iter()
                .all(|p| backend.dispatched[..60].iter().any(|c| c.id_sha256 == *p)));
        }
        assert!(report.winner().is_some());
        Ok(())
    }

    #[test]
    fn generic_has_80_slots_and_identical_limits_clock_precedes_generation() -> Result<()> {
        let plan = comparison(80);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let generic = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(generic.observations.len(), 80);
        assert!(generic
            .observations
            .iter()
            .all(|o| o.candidate.origin == Origin::Generic));
        assert_eq!(generic.arm.seconds, plan.arm(true)?.seconds);
        assert_eq!(generic.deadline_unix_ms, 121000);
        assert_ne!(generic.arm.id_sha256, plan.arm(true)?.id_sha256);
        assert!(comparison(79).arm(false).is_err());
        Ok(())
    }

    #[test]
    fn original_expired_window_admits_no_fits_and_does_not_reset() -> Result<()> {
        let plan = comparison(80);
        let arm = plan.arm(true)?;
        let mut backend = FixtureBackend::default();
        let deadline = backend.begin_arm(&arm)?;
        backend.time = deadline + 1;
        assert_eq!(backend.begin_arm(&arm)?, deadline);
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let result = run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm.clone(),
            deadline,
            synthetic_initial(&arm),
            None,
        )?;
        assert!(backend.dispatched.is_empty());
        assert_eq!(result.observations.len(), 80);
        assert!(result
            .observations
            .iter()
            .all(|o| o.feedback.status == EvaluationStatus::BudgetExhausted));
        assert!(result.winner().is_none());
        Ok(())
    }

    #[test]
    fn unknown_physical_fit_is_not_a_win_or_an_evolution_parent() -> Result<()> {
        let plan = comparison(84);
        let mut backend = FixtureBackend {
            unknown_counter: true,
            ..Default::default()
        };
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let report = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(report.observations.len(), 80);
        assert_eq!(report.stop_reason, "no_measured_parent");
        assert!(report.winner().is_none());
        assert!(report
            .observations
            .iter()
            .all(|o| o.feedback.physical_fit_starts.is_none()));
        Ok(())
    }

    #[test]
    fn resume_rechecks_cached_artifacts_and_keeps_original_window() -> Result<()> {
        let plan = comparison(84);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let first = run_generic(&mut backend, &plan, &mut variation)?;
        backend.time = first.deadline_unix_ms + 5000;
        let second = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(backend.dispatched.len(), 84);
        assert_eq!(
            serde_json::to_value(&first)?,
            serde_json::to_value(&second)?
        );
        backend
            .cache
            .values_mut()
            .next()
            .unwrap()
            .fitted_artifact_sha256 = Some(sha256_hex(b"changed"));
        assert!(run_generic(&mut backend, &plan, &mut variation).is_err());
        Ok(())
    }

    #[test]
    fn partial_resume_replays_scheduled_generation_without_new_deadline() -> Result<()> {
        let plan = comparison(84);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let first = run_generic(&mut backend, &plan, &mut variation)?;
        backend
            .journal
            .remove(&format!("{}/search-receipt", first.arm.id_sha256));
        backend.time = first.deadline_unix_ms + 1;
        let resumed = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(resumed.observations.len(), 84);
        assert_eq!(resumed.deadline_unix_ms, first.deadline_unix_ms);
        assert_eq!(backend.dispatched.len(), 84);
        Ok(())
    }

    #[test]
    fn nonfinite_scores_never_rank() {
        let candidate = synthetic_initial(&comparison(80).arm(true).unwrap()).remove(0);
        let mut f = Feedback {
            candidate_sha256: candidate.id_sha256,
            status: EvaluationStatus::Completed,
            loss: Some(f64::NAN),
            physical_fit_starts: Some(1),
            worker_receipt_sha256: Some(sha256_hex(b"worker")),
            fitted_artifact_sha256: Some(sha256_hex(b"artifact")),
            reason: None,
            oof: None,
            retryable_inventory: false,
        };
        f.enforce_measurement();
        assert_eq!(f.status, EvaluationStatus::Failed);
        assert!(f.loss.is_none());
    }
    #[test]
    fn pair_plan_is_frozen_before_clocks_and_rejects_changed_budget_or_sampling() -> Result<()> {
        let plan = comparison(80);
        let mut backend = FixtureBackend::default();
        let variation = FixtureVariation {
            started: backend.started.clone(),
        };
        freeze_pair(&mut backend, &plan, &variation)?;
        assert!(!backend.started.get());
        assert_eq!(backend.journal.len(), 3);
        let mut changed = comparison(84);
        assert!(freeze_pair(&mut backend, &changed, &variation).is_err());
        changed = comparison(80);
        changed.proposal.sampling_seed += 1;
        assert!(freeze_pair(&mut backend, &changed, &variation).is_err());
        Ok(())
    }
    #[test]
    fn unavailable_nonattempt_does_not_poison_resume() -> Result<()> {
        let plan = comparison(80);
        let mut backend = FixtureBackend {
            admission_unavailable: true,
            ..Default::default()
        };
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let pending = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(pending.stop_reason, "admission_pending_original_window");
        assert!(pending.winner().is_none());
        assert!(!backend
            .journal
            .keys()
            .any(|k| k.ends_with("search-receipt") || k.ends_with("observed")));
        backend.time += 100;
        let resumed = run_generic(&mut backend, &plan, &mut variation)?;
        assert_eq!(resumed.deadline_unix_ms, pending.deadline_unix_ms);
        assert_eq!(backend.dispatched.len(), 80);
        assert!(resumed.winner().is_some());
        Ok(())
    }
    #[test]
    fn canonical_alias_slot_keeps_origin_without_duplicate_fit() -> Result<()> {
        let plan = comparison(80);
        let arm = plan.arm(true)?;
        let mut backend = FixtureBackend::default();
        let deadline = backend.begin_arm(&arm)?;
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let mut initial = synthetic_initial(&arm);
        initial[1] = Candidate::new(
            &arm,
            " (pipe X (ridge   lambda=1)) ".into(),
            0,
            1,
            Origin::NeuralPolicy,
            vec![],
            None,
        )?;
        assert_eq!(initial[0].program_sha256, initial[1].program_sha256);
        let receipt = run_population(
            &mut backend,
            &plan,
            &mut variation,
            arm,
            deadline,
            initial,
            None,
        )?;
        assert_eq!(receipt.observations.len(), 80);
        assert_eq!(backend.dispatched.len(), 79);
        assert_eq!(
            receipt.observations[1].candidate.origin,
            Origin::NeuralPolicy
        );
        assert_eq!(
            receipt.observations[1].feedback.physical_fit_starts,
            Some(0)
        );
        assert!(receipt.observations[1]
            .feedback
            .reason
            .as_ref()
            .unwrap()
            .starts_with("canonical_duplicate_of:"));
        Ok(())
    }
    #[test]
    fn outer_score_cannot_rank_and_refit_requires_frozen_oof() -> Result<()> {
        let plan = comparison(80);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let receipt = run_generic(&mut backend, &plan, &mut variation)?;
        let frozen = freeze_winner(&mut backend, &plan, &receipt)?.unwrap();
        assert_eq!(
            frozen.refit_request.fitting_rows_sha256,
            plan.selection.outer().fitting_rows_sha256
        );
        assert_eq!(
            frozen.refit_request.scoring_rows_sha256,
            plan.selection.outer().scoring_rows_sha256
        );
        assert_eq!(
            frozen.refit_request.recipe_sha256,
            frozen.candidate.program_sha256
        );
        let winner = receipt.winner().unwrap();
        let mut bad = winner.feedback.clone();
        bad.oof.as_mut().unwrap().scored_rows_sha256 =
            plan.selection.outer().scoring_rows_sha256.clone();
        bad.enforce_selection(&plan.selection, &winner.candidate);
        assert_eq!(bad.status, EvaluationStatus::Failed);
        assert!(bad.loss.is_none());
        Ok(())
    }
    #[test]
    fn unavailable_neural_context_still_runs_generic_and_preserves_early_clock() -> Result<()> {
        let plan = comparison(80);
        let mut backend = FixtureBackend::default();
        let mut variation = FixtureVariation {
            started: backend.started.clone(),
        };
        let (arm, deadline) = begin_pair(&mut backend, &plan, &variation)?;
        backend.time += 1000;
        let result = run_pair(&mut backend, &plan, &mut variation, None)?;
        assert!(result.neural.is_none());
        assert!(result.neural_error.is_some());
        assert_eq!(backend.deadlines[&arm.id_sha256], deadline);
        let generic = result.generic.unwrap();
        assert_eq!(generic.observations.len(), 80);
        assert_eq!(generic.arm.seconds, arm.seconds);
        Ok(())
    }
}
