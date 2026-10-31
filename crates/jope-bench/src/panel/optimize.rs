//! Bounded paired search on actual frozen training rows. This controller never
//! receives sealed data and never passes development scoring labels to workers.
use super::{
    campaign::{CampaignPlan, Runtime},
    data::DevelopmentData,
    optimize_fit::{self, artifact, digest, immutable, immutable_bytes, FitAdapter},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    context_qualification::{CompletedQualification, EncoderAssets},
    cpu_context_bridge::{CpuContextFiles, ExecutionProof},
    execution::{Artifact, WorkerCommand, WorkerJob, WorkerOutcome},
    experiment::{ColumnKind, EvaluationRequest, EvaluationStage, EvaluationStatus, Experiment},
    neural_search::{
        self, Backend, Comparison, ContextCost, EncodingPlan, InstallationPins, PairInputPlan,
        PolicyWorker, PoolAccess, PoolBackend, Variation,
    },
    search_context::{ContextFiles, TrainingIdentity},
    synthesized_recipe::{FIT_RANDOMNESS_POLICY, RAW_INPUT_POLICY},
};
use jope_policy::{
    context_producer::{self, TrainingInput, TrainingRow},
    producer::{QualifiedContextBundle, RequestPlan},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::Path, sync::atomic::AtomicBool};

pub const SCHEMA: &str = "jope.asm-configuration/v1";
pub const CONFIG_RUNTIME: &str = "asm-configuration";
/// Largest accepted round nonce. The S2 series preregisters at most 12 rounds.
pub const MAX_SEARCH_ROUND: u32 = 64;
/// W3-T02: the W2 producer run's handoff of its context to teacher admission, beside the
/// run's `context-files.json` (`prepare_regression_context`).
pub(crate) const TEACHER_HANDOFF: &str = "teacher-admission-handoff.json";
pub const NEURAL_METHOD: &str = "asm.neural";
pub const GENERIC_METHOD: &str = "asm.generic";
pub const VARIATION: &str = "feat-math.canonical-mutate-crossover/task-grammar/max-transforms4/v1";
pub const GENERIC_POLICY: &str = "jope-policy.grammar-uniform-token-chacha8/v1";
pub const CLASSIFICATION_GENERIC_POLICY: &str = "asm.classification-generic/v1:head=logit_ridge|multiclass_ridge:l2=10^(-4+5*low16/65535):transform=(seed>>16)%4:none|standardize|winsor|both:q=.005+.24*((seed>>18)&255)/255";
pub const CLASSIFICATION_VARIATION: &str =
    "feat-math.v3-head-preserving/mutate-crossover/max-transforms4/v1";
pub fn generic_policy(task: Task) -> &'static str {
    if task == Task::Regression {
        GENERIC_POLICY
    } else {
        CLASSIFICATION_GENERIC_POLICY
    }
}
pub fn variation_policy(task: Task) -> &'static str {
    if task == Task::Regression {
        VARIATION
    } else {
        CLASSIFICATION_VARIATION
    }
}
/// Distinct, reproducible experimental classification grammar. It does not use
/// or extend the accepted regression checkpoint's task/token vocabulary.
pub fn classification_generic(task: Task, seed: u64) -> Result<String> {
    let head = match task {
        Task::Binary => "logit_ridge",
        Task::Multiclass => "multiclass_ridge",
        _ => anyhow::bail!("classification sampler requires a classification task"),
    };
    let l2 = 10.0_f64.powf(-4.0 + 5.0 * (seed & 65535) as f64 / 65535.0);
    let q = 0.005 + 0.24 * ((seed >> 18) & 255) as f64 / 255.0;
    let transforms = match (seed >> 16) % 4 {
        0 => String::new(),
        1 => "(standardize)".into(),
        2 => format!("(winsor_clip q={q})"),
        _ => format!("(standardize) (winsor_clip q={q})"),
    };
    Ok(feat_math::GenomeIr::parse(&format!("(pipe X {transforms} ({head} l2={l2}))"))?.canonical())
}
fn v3_candidate(source: &str) -> Result<feat_math::advanced_v3::V3Candidate> {
    let syntax = feat_math::GenomeIr::parse(source)?.canonical();
    Ok(feat_math::advanced_v3::V3Candidate {
        semantic_hash: jope_pipeline::sha256_hex(syntax.as_bytes()),
        syntax,
        lineage: vec![],
        island: 0,
        operator: "asm-caller-parent".into(),
        step_micros: 150_000,
        generation: 0,
    })
}
fn bounded_classification_child(candidate: feat_math::advanced_v3::V3Candidate) -> Result<String> {
    let feat_math::GenomeIr::Pipeline(pipeline) = feat_math::GenomeIr::parse(&candidate.syntax)?
    else {
        anyhow::bail!("nonpipeline class child")
    };
    ensure!(
        pipeline.stages.len() <= 5,
        "classification variation exceeded four transforms"
    );
    Ok(candidate.syntax)
}

/// All identities are known before either arm. Qualification is separately
/// produced; its absence is an explicit unavailable neural arm, not permission
/// to admit an arbitrary context vector. No user acceptance boolean exists.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Configuration {
    /// Separate Python exports by task; omission preserves accepted configuration bytes.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub experimental: BTreeMap<String, super::experimental_proposal::ExperimentalConfiguration>,
    pub schema: String,
    pub source_sha40: String,
    /// A jope.asm-worker-build-inventory/v1, not the CLI installer receipt.
    pub release_manifest: Artifact,
    pub source_manifest: Artifact,
    pub checkpoint_receipt: Artifact,
    pub policy_weights: Artifact,
    pub vocabulary: Artifact,
    pub frozen_transforms: Artifact,
    pub policy: Runtime,
    pub encoder: Runtime,
    pub encoder_assets: EncoderAssets,
    pub qualification: Option<CompletedQualification>,
    pub seconds: u64,
    /// 80 initial slots plus this bounded total, identical for both arms.
    pub max_evaluations: usize,
    pub sampling_rounds: usize,
    pub sampling_batch: usize,
    pub sampling_temperature: f32,
    pub dataset_wave_width: usize,
    pub scheduling_policy: String,
    pub inner_fold_policy: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching: Option<super::teaching::Settings>,
    /// Reseeds proposal and generic sampling for a repeated round on one frozen
    /// comparison; omission preserves accepted configuration bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_round: Option<u32>,
    /// Earlier arm winners seeded into generation zero of their own arm; omission
    /// preserves accepted configuration bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub incumbents: Option<Incumbents>,
    /// A native challenger checkpoint for the policy worker; omission preserves
    /// accepted configuration bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub native_challenger: Option<NativeChallenger>,
}
/// Pinned arm winners from earlier rounds on the same dataset. Each artifact is a
/// frozen-recipe blob in the teaching state; each list seeds only its own arm.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Incumbents {
    pub neural: Vec<Artifact>,
    pub generic: Vec<Artifact>,
}
/// A preregistered native challenger checkpoint (plan §4.5 DEMO-3): its export receipt, its
/// exported weights and the identity both must bind. The accepted pins stay mandatory and
/// verified; only the policy worker loads these files instead.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeChallenger {
    pub checkpoint_receipt: Artifact,
    pub policy_weights: Artifact,
    pub identity: jope_policy::config::ChallengerIdentity,
}
impl Configuration {
    pub fn validate(&self) -> Result<()> {
        // Every challenger refusal precedes every other check and every file read.
        if let Some(challenger) = &self.native_challenger {
            ensure!(
                self.teaching.is_none()
                    && self.incumbents.is_none()
                    && self.experimental.is_empty(),
                "native challenger runs without teaching, incumbents or experimental proposals"
            );
            challenger.identity.validate()?;
            ensure!(
                challenger.identity.config_sha256 == jope_policy::ACCEPTED_CONFIG_SHA256,
                "native challenger must keep the accepted architecture"
            );
            ensure!(
                challenger.policy_weights.sha256 == challenger.identity.policy_safetensors_sha256,
                "native challenger weights differ from its identity"
            );
            ensure!(
                challenger.checkpoint_receipt.sha256 != self.checkpoint_receipt.sha256,
                "native challenger repeats the accepted checkpoint receipt"
            );
        }
        if let Some(settings) = &self.teaching {
            settings.validate()?;
        }
        ensure!(
            self.schema == SCHEMA
                && self.source_sha40.len() == 40
                && self.source_sha40.bytes().all(|c| c.is_ascii_hexdigit()),
            "invalid ASM configuration/source"
        );
        ensure!(
            (1..=120).contains(&self.seconds)
                && (80..=1000).contains(&self.max_evaluations)
                && (1..=20).contains(&self.dataset_wave_width),
            "ASM budget/wave outside approved bounds"
        );
        ensure!(
            self.search_round
                .is_none_or(|round| round <= MAX_SEARCH_ROUND),
            "search round outside approved bounds"
        );
        if let Some(incumbents) = &self.incumbents {
            let settings = self
                .teaching
                .as_ref()
                .context("incumbent seeding requires a teaching state")?;
            ensure!(
                self.experimental.is_empty(),
                "incumbent seeding requires the accepted proposal plan"
            );
            ensure!(
                !(incumbents.neural.is_empty() && incumbents.generic.is_empty())
                    && incumbents.neural.len() <= neural_search::MAX_INCUMBENTS
                    && incumbents.generic.len() <= neural_search::MAX_INCUMBENTS,
                "incumbent count outside approved bounds"
            );
            let state = jope_learning::state::State::at(&settings.state_root);
            for pin in incumbents.neural.iter().chain(&incumbents.generic) {
                ensure!(
                    jope_pipeline::experiment::sha256(&pin.sha256)
                        && pin.path == state.blob_path(&pin.sha256),
                    "incumbent must pin a frozen-recipe blob in the teaching state"
                );
            }
        }
        ensure!(
            self.scheduling_policy == super::optimize_schedule::POLICY
                && self.inner_fold_policy == optimize_fit::INNER_FOLD_POLICY,
            "unknown selection or scheduling policy"
        );
        ensure!(
            self.policy.argument_prefix.is_empty() && self.encoder.argument_prefix.is_empty(),
            "producer runtimes must pin their dedicated executables"
        );
        ensure!(
            self.frozen_transforms.sha256 == context_producer::TRANSFORMS_SHA256,
            "wrong accepted frozen transforms"
        );
        for a in self.dependencies() {
            ensure!(
                a.path.is_absolute() && jope_pipeline::experiment::sha256(&a.sha256),
                "malformed dependency pin"
            );
        }
        for (task, config) in &self.experimental {
            ensure!(
                task == config.task.as_str(),
                "experimental task inventory key changed"
            );
            config.validate()?;
        }
        self.vocabulary.verify()?;
        jope_policy::grammar::bind_vocabulary(
            &fs::read(&self.vocabulary.path)?,
            jope_policy::CHECKPOINT_VOCABULARY_SHA256,
        )?;
        ensure!(
            self.policy_weights.sha256
                == "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e",
            "wrong accepted policy bytes"
        );
        ensure!(
            (1..=8).contains(&self.sampling_rounds)
                && (1..=128).contains(&self.sampling_batch)
                && self.sampling_temperature.is_finite()
                && self.sampling_temperature > 0.,
            "invalid proposal allowance"
        );
        if let Some(q) = &self.qualification {
            ensure!(
                q.case_outputs.len() == 8 && q.case_executions.len() == 8,
                "incomplete qualification inventory"
            );
        }
        Ok(())
    }
    pub fn verify_worker_builds(&self) -> Result<Vec<Artifact>> {
        self.release_manifest.verify()?;
        let inventory: super::worker_build::WorkerBuildInventory =
            serde_json::from_slice(&fs::read(&self.release_manifest.path)?)?;
        inventory.verify(self)
    }
    /// Check a prepared runtime without fitting or admitting an execution window.
    pub fn verify_prepared(&self) -> Result<bool> {
        self.validate()?;
        self.verify_worker_builds()?;
        for dependency in self.dependencies() {
            dependency.verify()?;
        }
        if let Some(qualification) = &self.qualification {
            qualification.verify(&self.source_sha40, &self.encoder.program)?;
            for proof in &qualification.case_executions {
                for asset in std::iter::once(&self.encoder_assets.configuration)
                    .chain(&self.encoder_assets.files)
                {
                    ensure!(
                        proof
                            .job
                            .command
                            .inputs
                            .iter()
                            .any(|a| a.sha256 == asset.sha256),
                        "qualification encoder assets differ from configuration"
                    );
                }
            }
            return Ok(true);
        }
        Ok(false)
    }
    pub fn dependencies(&self) -> Vec<&Artifact> {
        let mut files = vec![
            &self.release_manifest,
            &self.source_manifest,
            &self.checkpoint_receipt,
            &self.policy_weights,
            &self.vocabulary,
            &self.frozen_transforms,
            &self.encoder_assets.configuration,
            &self.policy.program,
            &self.encoder.program,
        ];
        files.extend(self.encoder_assets.files.iter());
        for runtime in [&self.policy, &self.encoder] {
            files.extend(runtime.inputs.iter());
            files.extend(runtime.package_files.iter());
        }
        if let Some(q) = &self.qualification {
            files.extend([&q.qualification, &q.reference_manifest]);
            files.extend(q.case_outputs.iter());
        }
        for config in self.experimental.values() {
            files.extend(config.dependencies());
        }
        if let Some(challenger) = &self.native_challenger {
            files.extend([&challenger.checkpoint_receipt, &challenger.policy_weights]);
        }
        files
    }
    pub fn dependency_inventory(&self) -> Vec<serde_json::Value> {
        self.dependencies()
            .into_iter()
            .map(|a| match a.verify() {
                Ok(()) => serde_json::json!({"artifact":a,"available":true}),
                Err(e) => {
                    serde_json::json!({"artifact":a,"available":false,"reason":format!("{e:#}")})
                }
            })
            .collect()
    }
    pub fn from_campaign(plan: &CampaignPlan) -> Result<Self> {
        let runtime = plan
            .runtimes
            .get(CONFIG_RUNTIME)
            .context("ASM configuration is not frozen in runtime roster")?;
        ensure!(
            runtime.inputs.len() == 1,
            "ASM configuration requires one pinned artifact"
        );
        runtime.inputs[0].verify()?;
        let value: Self = serde_json::from_slice(&fs::read(&runtime.inputs[0].path)?)?;
        value.validate()?;
        ensure!(
            value.source_sha40 == plan.experiment.source_sha,
            "worker configuration source differs from experiment"
        );
        Ok(value)
    }
}

pub(crate) struct TaskVariation<'a> {
    pub(crate) vocabulary: &'a jope_policy::grammar::BoundVocabulary,
    pub(crate) task: Task,
}
impl Variation for TaskVariation<'_> {
    fn identity(&self) -> &str {
        variation_policy(self.task)
    }
    fn generic_policy(&self) -> &str {
        generic_policy(self.task)
    }
    fn generic(&mut self, seed: u64) -> Result<Option<String>> {
        if self.task == Task::Regression {
            jope_policy::producer::generic_program(self.vocabulary, "regression", seed)
        } else {
            classification_generic(self.task, seed).map(Some)
        }
    }
    fn mutate(&mut self, parent: &str, seed: u64) -> Result<String> {
        if self.task == Task::Regression {
            feat_math::evolve::mutate_canonical_pipeline(parent, seed, 4)
                .map_err(anyhow::Error::msg)
        } else {
            bounded_classification_child(feat_math::advanced_v3::mutate_candidate(
                &v3_candidate(parent)?,
                seed,
                1,
                0,
            )?)
        }
    }
    fn crossover(&mut self, left: &str, right: &str, seed: u64) -> Result<String> {
        if self.task == Task::Regression {
            feat_math::evolve::crossover_canonical_pipelines(left, right, seed, 4)
                .map_err(anyhow::Error::msg)
        } else {
            bounded_classification_child(feat_math::advanced_v3::crossover_candidates(
                &v3_candidate(left)?,
                &v3_candidate(right)?,
                seed,
                1,
                0,
            )?)
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatasetSearch {
    pub dataset: String,
    pub task: Task,
    pub pair: PairInputPlan,
    pub receipt: neural_search::PairedReceipt,
    pub neural: Option<neural_search::FrozenRecipe>,
    pub generic: Option<neural_search::FrozenRecipe>,
    pub neural_unavailable: Option<String>,
    pub neural_retryable_inventory: bool,
    pub final_test_targets_used: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching_admission: Option<Artifact>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub teaching_error: Option<String>,
}

/// Pair sampling seed. An absent round and round 0 keep the historical seed, so
/// accepted plans and K0 draw exactly the proposals they always drew.
pub(crate) fn pair_seed(root: u64, dataset_sha256: &str, round: Option<u32>) -> Result<u64> {
    let seed = jope_pipeline::search_plan::candidate_seed(root, dataset_sha256)?;
    match round {
        None | Some(0) => Ok(seed),
        Some(round) => jope_pipeline::search_plan::candidate_seed(
            seed,
            &digest(&("asm.search-round/v1", round))?,
        ),
    }
}

/// Read-only: verifies the pinned blob in the teaching state without creating it.
pub(crate) fn incumbent_seed(
    state_root: &Path,
    pin: &Artifact,
    selection: &jope_pipeline::search_plan::SelectionPlan,
    neural: bool,
) -> Result<neural_search::IncumbentSeed> {
    let state = jope_learning::state::State::at(state_root);
    ensure!(
        jope_pipeline::experiment::sha256(&pin.sha256) && pin.path == state.blob_path(&pin.sha256),
        "incumbent must pin a frozen-recipe blob in the teaching state"
    );
    // read_blob re-hashes the stored bytes and refuses a corrupt blob.
    let bytes = state
        .read_blob(&pin.sha256)?
        .context("incumbent frozen recipe is absent from the teaching state")?;
    neural_search::IncumbentSeed::from_frozen_recipe(&bytes, &pin.sha256, selection, neural)
}

fn resolve_incumbents(
    configuration: &Configuration,
    selection: &jope_pipeline::search_plan::SelectionPlan,
) -> Result<Option<neural_search::ArmIncumbents>> {
    let Some(pins) = &configuration.incumbents else {
        return Ok(None);
    };
    let settings = configuration
        .teaching
        .as_ref()
        .context("incumbent seeding requires a teaching state")?;
    let resolve = |pins: &[Artifact], neural: bool| {
        pins.iter()
            .map(|pin| incumbent_seed(&settings.state_root, pin, selection, neural))
            .collect::<Result<Vec<_>>>()
    };
    Ok(Some(neural_search::ArmIncumbents {
        neural: resolve(&pins.neural, true)?,
        generic: resolve(&pins.generic, false)?,
    }))
}

/// Receipt and weights the policy worker loads: the native challenger's, else the accepted pins.
pub(crate) fn proposal_files(configuration: &Configuration) -> (&Artifact, &Artifact) {
    configuration.native_challenger.as_ref().map_or(
        (
            &configuration.checkpoint_receipt,
            &configuration.policy_weights,
        ),
        |c| (&c.checkpoint_receipt, &c.policy_weights),
    )
}
/// Read-only: the checkpoint the policy worker will bind, from the configuration's pins. The
/// native challenger's receipt is held to the challenger rules; otherwise the pinned receipt must
/// be the accepted `tools/ot-export` receipt naming the pinned weights. A re-export of the
/// accepted checkpoint by this repository's exporter carries no acceptance authority and is
/// refused here, before any worker starts (W2 G-08, OWNER-DECISIONS.md D13).
pub(crate) fn pinned_checkpoint(
    config: &Configuration,
) -> Result<jope_policy::config::CheckpointBinding> {
    if let Some(challenger) = &config.native_challenger {
        return challenger_checkpoint(challenger);
    }
    config.checkpoint_receipt.verify()?;
    let checkpoint = jope_policy::config::CheckpointBinding::for_proposal(
        &fs::read(&config.checkpoint_receipt.path)?,
        None,
    )?;
    ensure!(
        checkpoint.policy_safetensors_sha256() == config.policy_weights.sha256,
        "policy checksum differs from accepted receipt"
    );
    Ok(checkpoint)
}
/// Read-only: the challenger's export receipt must bind its pinned identity and weights.
pub(crate) fn challenger_checkpoint(
    challenger: &NativeChallenger,
) -> Result<jope_policy::config::CheckpointBinding> {
    challenger.checkpoint_receipt.verify()?;
    let binding = jope_policy::config::CheckpointBinding::for_proposal(
        &fs::read(&challenger.checkpoint_receipt.path)?,
        Some(&challenger.identity),
    )?;
    ensure!(
        binding.policy_safetensors_sha256() == challenger.policy_weights.sha256,
        "native challenger weights differ from its export receipt"
    );
    Ok(binding)
}

/// Reconstruct every pair identity from exact authenticated train values and configuration.
/// This is shared by execution and read-only report admission; no artifact is written.
pub fn build_pair_input(
    campaign: &CampaignPlan,
    configuration: &Configuration,
    dataset: &DevelopmentData,
) -> Result<(super::experimental_proposal::ContextInput, PairInputPlan)> {
    let cut = dataset.cut(0)?;
    let mut outer = super::campaign::request(
        dataset,
        &cut,
        &campaign.methods[dataset.dataset.task.as_str()][0],
        &campaign.experiment,
        EvaluationStage::Screen,
    )?;
    // This is the pre-output pair slot, never a claimed executable recipe.
    outer.recipe_sha256 = digest(&(
        "asm.pair-slot/v1",
        &configuration.schema,
        &configuration.source_manifest.sha256,
    ))?;
    let selection = optimize_fit::selection(&cut, outer)?;
    let rows = cut
        .train
        .rows
        .iter()
        .map(|r| TrainingRow {
            id: r.id.clone(),
            values: r.values.clone(),
            label: r.target.clone(),
        })
        .collect::<Vec<_>>();
    let training_sha256 = context_producer::training_rows_sha256(&rows)?;
    // W1-T01: a member of the committed regression release takes its source
    // and prepared identities from the release admission its loader verified.
    let (dataset_sha256, prepared_sha256) = match &dataset.release {
        Some(release) => (
            release.dataset_sha256.clone(),
            release.development_sha256.clone(),
        ),
        None => (
            dataset.dataset.source.sha256.clone(),
            dataset.dataset.split.development_sha256.clone(),
        ),
    };
    let input = TrainingInput {
        schema: if dataset.dataset.task == Task::Regression {
            "jope.cpu-context-training/regression-v1"
        } else {
            "asm.experimental-classification-input/v1"
        }
        .into(),
        identity: context_producer::TrainingIdentity {
            dataset_sha256,
            prepared_sha256,
            training_sha256: training_sha256.clone(),
            fitting_rows_sha256: cut.train.rows_sha256.clone(),
            cut: 0,
        },
        features: dataset
            .dataset
            .features
            .iter()
            .map(|f| context_producer::Feature {
                name: f.name.clone(),
                kind: match f.kind {
                    ColumnKind::Numeric => context_producer::FeatureKind::Numeric,
                    ColumnKind::Categorical => context_producer::FeatureKind::Categorical,
                },
            })
            .collect(),
        rows,
    };
    let prepared = if dataset.dataset.task == Task::Regression {
        Some(input.prepare()?)
    } else {
        None
    };
    let preprocessing_sha256 = match prepared {
        Some(ref p) => digest(&p.preprocessing)?,
        None => digest(&("asm.classification-context-unavailable/v1", &input.features))?,
    };
    // Classification keeps original labels in its content identity. It is never
    // sent through the regression encoder or ordinal target context.
    let input_sha256 = digest(&input)?;
    let seed = pair_seed(
        campaign.experiment.seed,
        &dataset.dataset.source.sha256,
        configuration.search_round,
    )?;
    let incumbents = resolve_incumbents(configuration, &selection)?;
    // A native challenger changes only the receipt, weights, state and identity the policy
    // worker is pinned to; selection, encoding, seeds and variation are unchanged.
    let (proposal_receipt, proposal_weights) = proposal_files(configuration);
    let plan = PairInputPlan {
        selection,
        preprocessing_sha256,
        raw_input_policy: RAW_INPUT_POLICY.into(),
        fit_randomness_policy: FIT_RANDOMNESS_POLICY.into(),
        installation: InstallationPins {
            release_manifest_sha256: configuration.release_manifest.sha256.clone(),
            policy_program_sha256: configuration.policy.program.sha256.clone(),
            source_manifest_sha256: configuration.source_manifest.sha256.clone(),
            checkpoint_receipt_sha256: proposal_receipt.sha256.clone(),
        },
        proposal: RequestPlan {
            worker_source_sha256: configuration.source_manifest.sha256.clone(),
            prepared_sha256: input.identity.prepared_sha256.clone(),
            training_sha256: training_sha256.clone(),
            cut: 0,
            checkpoint_sha256: configuration
                .native_challenger
                .as_ref()
                .map_or(jope_policy::ACCEPTED_STATE_SHA256, |c| {
                    c.identity.state_sha256.as_str()
                })
                .into(),
            policy_safetensors_sha256: proposal_weights.sha256.clone(),
            vocabulary_sha256: configuration.vocabulary.sha256.clone(),
            sampling_seed: seed,
            fallback_seed: seed.wrapping_add(1),
            exploration_seed: seed.wrapping_add(2),
            sampling_rounds: configuration.sampling_rounds,
            sampling_batch: configuration.sampling_batch,
            sampling_temperature: configuration.sampling_temperature,
            challenger: configuration
                .native_challenger
                .as_ref()
                .map(|c| c.identity.clone()),
        }
        .into(),
        encoding: EncodingPlan {
            input_sha256,
            source_sha40: configuration.source_sha40.clone(),
            program_sha256: configuration.encoder.program.sha256.clone(),
            source_manifest_sha256: configuration.source_manifest.sha256.clone(),
            encoder_assets_manifest_sha256: configuration
                .encoder_assets
                .configuration
                .sha256
                .clone(),
            qualification_sha256: configuration
                .qualification
                .as_ref()
                .map(|q| q.qualification.sha256.clone()),
            reference_manifest_sha256: configuration
                .qualification
                .as_ref()
                .map(|q| q.reference_manifest.sha256.clone()),
            backend: if dataset.dataset.task == Task::Regression {
                context_producer::BACKEND
            } else {
                "asm.experimental-classification-context/unavailable"
            }
            .into(),
        },
        variation_identity: variation_policy(dataset.dataset.task).into(),
        generic_sampling_policy: generic_policy(dataset.dataset.task).into(),
        checkpoint_training_exposure: "unknown".into(),
        exposure_evidence_sha256: None,
        prepared_sha256: input.identity.prepared_sha256.clone(),
        training_sha256,
        cut: 0,
        vocabulary_sha256: configuration.vocabulary.sha256.clone(),
        seconds: configuration.seconds,
        max_evaluations: configuration.max_evaluations,
        random_seed: seed,
        incumbents,
    };
    if let Some(experimental) = configuration
        .experimental
        .get(dataset.dataset.task.as_str())
    {
        let vocabulary = jope_policy::grammar::bind_vocabulary(
            &fs::read(&configuration.vocabulary.path)?,
            jope_policy::CHECKPOINT_VOCABULARY_SHA256,
        )?;
        let mut variation = TaskVariation {
            vocabulary: &vocabulary,
            task: dataset.dataset.task,
        };
        return super::experimental_proposal::bind_pair(
            configuration,
            experimental,
            dataset,
            input,
            plan,
            &mut variation,
        );
    }
    Ok((input.into(), plan))
}

/// Used by the shared-wave controller, and by standalone optimize on one frozen
/// dataset. PoolAccess is either the actual pool or its bounded message proxy.
pub fn search_dataset<P: PoolAccess>(
    campaign: &CampaignPlan,
    configuration: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    pool: &mut P,
    cancel: &AtomicBool,
) -> Result<DatasetSearch> {
    let cut = dataset.cut(0)?;
    let (input, plan) = build_pair_input(campaign, configuration, dataset)?;
    let input_file = immutable(
        &journal
            .join("asm-context-inputs")
            .join(format!("{}.json", plan.encoding.input_sha256)),
        &input,
    )?;
    let vocabulary = jope_policy::grammar::bind_vocabulary(
        &fs::read(&configuration.vocabulary.path)?,
        jope_policy::CHECKPOINT_VOCABULARY_SHA256,
    )?;
    let mut variation = TaskVariation {
        vocabulary: &vocabulary,
        task: dataset.dataset.task,
    };
    let adapter = FitAdapter::prepare(
        dataset,
        &cut,
        plan.selection.clone(),
        campaign.runtimes["rust"].clone(),
        journal,
        true,
    )?;
    let mut backend = PoolBackend {
        pool,
        cancelled: cancel,
        journal: journal.join("asm-search"),
        make_job: |candidate: &neural_search::Candidate, request: &EvaluationRequest| {
            adapter.job(candidate, request)
        },
        score: |candidate: &neural_search::Candidate,
                outcomes: &[WorkerOutcome],
                selection: &jope_pipeline::search_plan::SelectionPlan| {
            adapter.score(candidate, outcomes, selection)
        },
    };
    neural_search::begin_pair(&mut backend, &plan, &variation)?;
    let (pair, unavailable, neural_retryable_inventory) = if matches!(
        plan.proposal,
        neural_search::ProposalPlan::Experimental(_)
    ) {
        let neural_input = super::experimental_proposal::prepare(
            &mut backend,
            &plan,
            &mut variation,
            configuration,
            &input,
            &input_file,
            &campaign.experiment,
            journal,
        );
        let unavailable = neural_input.as_ref().err().map(|e| format!("{e:#}"));
        let retryable = neural_input.is_err()
            && configuration.qualification.is_some()
            && configuration.encoder.unavailable.is_none()
            && !terminal_encoding_failure(journal, &plan)?;
        let pair = neural_search::run_experimental_pair(
            &mut backend,
            &plan,
            &mut variation,
            neural_input
                .as_ref()
                .ok()
                .map(|(worker, context)| (worker, context)),
        )?;
        (pair, unavailable, retryable)
    } else {
        let neural_input = if dataset.dataset.task != Task::Regression {
            Err(anyhow::anyhow!(
                "accepted policy is regression-only; no configured qualified experimental classifier producer"
            ))
        } else {
            prepare_neural(
                &mut backend,
                &plan,
                &variation,
                configuration,
                input.accepted()?,
                &input_file,
                &campaign.experiment,
                journal,
            )
        };
        let unavailable = neural_input.as_ref().err().map(|e| format!("{e:#}"));
        let retryable = neural_input.is_err()
            && dataset.dataset.task == Task::Regression
            && configuration.qualification.is_some()
            && configuration.policy.unavailable.is_none()
            && configuration.encoder.unavailable.is_none()
            && !terminal_encoding_failure(journal, &plan)?;
        let pair = neural_search::run_pair(
            &mut backend,
            &plan,
            &mut variation,
            neural_input
                .as_ref()
                .ok()
                .map(|(comparison, worker)| (comparison, worker)),
        )?;
        (pair, unavailable, retryable)
    };
    let neural = pair
        .neural
        .as_ref()
        .and_then(|n| n.search.as_ref())
        .filter(|r| !pending(r))
        .map(|r| neural_search::freeze_winner(&mut backend, &plan, r))
        .transpose()?
        .flatten();
    let generic = pair
        .generic
        .as_ref()
        .filter(|r| !pending(r))
        .map(|r| neural_search::freeze_winner(&mut backend, &plan, r))
        .transpose()?
        .flatten();
    let mut result = DatasetSearch {
        dataset: dataset.dataset.id.clone(),
        task: dataset.dataset.task,
        pair: plan,
        receipt: pair,
        neural,
        generic,
        neural_unavailable: unavailable,
        neural_retryable_inventory,
        final_test_targets_used: false,
        teaching_admission: None,
        teaching_error: None,
    };
    match super::teaching::admit_dataset(campaign, configuration, dataset, &result, journal) {
        Ok(value) => result.teaching_admission = value,
        Err(error) => result.teaching_error = Some(format!("{error:#}")),
    }
    let value = serde_json::to_value(&result)?;
    immutable(
        &journal
            .join("asm-inventory")
            .join(&dataset.dataset.id)
            .join(format!("{}.json", digest(&value)?)),
        &value,
    )?;
    Ok(result)
}

pub(crate) fn diagnostic(
    plan: &PairInputPlan,
    purpose: &str,
    pin: &str,
) -> Result<EvaluationRequest> {
    neural_search::diagnostic_request(plan, purpose, pin)
}
pub(crate) fn proof(
    experiment: &Experiment,
    journal: &Path,
    job: WorkerJob,
) -> Result<ExecutionProof> {
    let id = job.identity()?;
    Ok(ExecutionProof {
        outcome: artifact(journal.join("finished").join(format!("{id}.json")))?,
        started: artifact(journal.join("started").join(format!("{id}.json")))?,
        window: artifact(journal.join("window.json"))?,
        arm_window: job
            .arm
            .as_ref()
            .map(|a| artifact(journal.join("arms").join(format!("{}.json", a.id_sha256))))
            .transpose()?,
        experiment: experiment.clone(),
        job,
    })
}
#[allow(clippy::too_many_arguments)] // Keep independently pinned execution inputs explicit.
fn prepare_neural<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    config: &Configuration,
    input: &TrainingInput,
    input_file: &Artifact,
    experiment: &Experiment,
    journal: &Path,
) -> Result<(Comparison, PolicyWorker)> {
    config.qualification.as_ref().context(
        "CPU encoder qualification absent; accepted native-component parity remains unmeasured",
    )?;
    ensure!(
        config.encoder.unavailable.is_none() && config.policy.unavailable.is_none(),
        "planned encoder or policy runtime unavailable"
    );
    for file in config.dependencies() {
        file.verify()?;
    }
    let worker_builds = config.verify_worker_builds()?;
    let (proposal_receipt, proposal_weights) = proposal_files(config);
    pinned_checkpoint(config)?;
    let (files, outcome) = prepare_regression_context(
        backend,
        plan,
        variation,
        config,
        input,
        input_file,
        experiment,
        journal,
        &worker_builds,
    )?;
    let context_directory = journal.join("asm-contexts").join(plan.arm(true)?.id_sha256);
    let context_sha256 = files.vector.sha256.clone();
    // The worker admits the qualified bundle only against pins this request issues: exactly the
    // admission prepare_regression_context built from the controller's own inputs and verified.
    let bundle_bytes = fs::read(&files.bundle.path)?;
    ensure!(
        jope_pipeline::sha256_hex(&bundle_bytes) == files.bundle.sha256,
        "qualified context bundle changed after admission"
    );
    let bundle: QualifiedContextBundle = serde_json::from_slice(&bundle_bytes)?;
    let producer_request = plan
        .proposal
        .accepted()?
        .qualified_request(context_sha256, bundle.pins.sha256()?);
    let request_file = immutable(
        &context_directory.join("proposal-request.json"),
        &producer_request,
    )?;
    let request = diagnostic(
        plan,
        "asm.policy-proposal/v1",
        &producer_request.identity()?,
    )?;
    let directory = journal.join("jobs").join(request.identity()?);
    let named = [
        ("--request", &request_file),
        ("--source-manifest", &config.source_manifest),
        ("--context", &files.vector),
        ("--context-receipt", &files.artifact),
        ("--context-bundle", &files.bundle),
        ("--transforms", &config.frozen_transforms),
        ("--checkpoint-receipt", proposal_receipt),
        ("--policy", proposal_weights),
        ("--vocabulary", &config.vocabulary),
    ];
    let mut args = Vec::new();
    let mut inputs = config.policy.inputs.clone();
    inputs.extend(worker_builds);
    inputs.push(config.release_manifest.clone());
    for (name, file) in named {
        args.extend([name.into(), file.path.display().to_string()]);
        inputs.push(file.clone());
    }
    args.extend([
        "--output".into(),
        directory.join("output.json").display().to_string(),
    ]);
    let comparison = Comparison {
        input_plan: plan.clone(),
        producer_request: producer_request.clone(),
        context_receipt_sha256: files.artifact.sha256.clone(),
        context_cost: ContextCost {
            cached: false,
            production_receipt: files.context_execution.outcome.clone(),
            original_wall_seconds: outcome.consumption.wall_seconds,
            original_cpu_seconds: outcome.consumption.cpu_seconds,
            original_gpu_seconds: Some(0.),
            amortization_uses: Some(1),
        },
    };
    let worker = PolicyWorker {
        job: WorkerJob {
            request,
            command: WorkerCommand {
                program: config.policy.program.clone(),
                args,
                inputs,
                environment: config.policy.environment.clone(),
            },
            directory,
            outputs: vec!["output.json".into()],
            arm: Some(plan.arm(true)?),
        },
        expected_program_sha256: config.policy.program.sha256.clone(),
        release_manifest: config.release_manifest.clone(),
        source_manifest: config.source_manifest.clone(),
        request_artifact: request_file,
        checkpoint_receipt: proposal_receipt.clone(),
        request: producer_request,
        context: ContextFiles::QualifiedCpu(Box::new(files)),
        output_name: "output.json".into(),
    };
    immutable(
        &context_directory.join("policy-admission.json"),
        &(&comparison, &worker),
    )?;
    Ok((comparison, worker))
}

fn terminal_encoding_failure(journal: &Path, plan: &PairInputPlan) -> Result<bool> {
    let directory = journal.join("asm-search").join(plan.arm(true)?.id_sha256);
    if !directory.exists() {
        return Ok(false);
    }
    for file in fs::read_dir(directory)? {
        let file = file?;
        if !file
            .file_name()
            .to_string_lossy()
            .starts_with("encoding-inventory-")
        {
            continue;
        }
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(file.path())?)?;
        if outcome.command_status != EvaluationStatus::Completed
            && (outcome.exit_code.is_some() || outcome.consumption.gate_released_unix_ms.is_some())
        {
            let retained = journal
                .join("finished")
                .join(format!("{}.json", outcome.job_sha256));
            if retained.exists() {
                let actual: WorkerOutcome = serde_json::from_slice(&fs::read(retained)?)?;
                ensure!(
                    digest(&actual)? == digest(&outcome)?,
                    "encoding failure receipt changed"
                );
                return Ok(true);
            }
        }
    }
    Ok(false)
}
pub fn pending(receipt: &neural_search::SearchReceipt) -> bool {
    matches!(
        receipt.stop_reason.as_str(),
        "admission_pending_original_window" | "cancelled_original_window"
    )
}

/// Shared native regression producer; this returns qualified context files only.
/// Accepted and experimental policy consumers retain separate admission routes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_regression_context<B: Backend, V: Variation>(
    backend: &mut B,
    plan: &PairInputPlan,
    variation: &V,
    config: &Configuration,
    input: &TrainingInput,
    input_file: &Artifact,
    experiment: &Experiment,
    journal: &Path,
    worker_builds: &[Artifact],
) -> Result<(CpuContextFiles, WorkerOutcome)> {
    let qualification = config
        .qualification
        .as_ref()
        .context("native regression qualification absent")?;
    let request = diagnostic(plan, "asm.fresh-training-context/v1", &input_file.sha256)?;
    let directory = journal.join("jobs").join(request.identity()?);
    let invocation = context_producer::worker::WorkerInvocation {
        mode: context_producer::worker::WorkerMode::Produce,
        input: context_producer::AssetFile {
            path: input_file.path.clone(),
            sha256: input_file.sha256.clone(),
        },
        assets: context_producer::AssetFile {
            path: config.encoder_assets.configuration.path.clone(),
            sha256: config.encoder_assets.configuration.sha256.clone(),
        },
        source_sha40: config.source_sha40.clone(),
        producer_sha256: config.encoder.program.sha256.clone(),
        output: directory.join("output.json"),
    };
    let mut inputs = config.encoder.inputs.clone();
    inputs.extend_from_slice(worker_builds);
    inputs.push(config.release_manifest.clone());
    inputs.extend(config.encoder_assets.files.clone());
    inputs.extend([
        input_file.clone(),
        config.encoder_assets.configuration.clone(),
        config.source_manifest.clone(),
    ]);
    let job = WorkerJob {
        request,
        command: WorkerCommand {
            program: config.encoder.program.clone(),
            args: invocation.arguments(),
            inputs,
            environment: config.encoder.environment.clone(),
        },
        directory,
        outputs: vec!["output.json".into()],
        arm: Some(plan.arm(true)?),
    };
    let outcome = neural_search::encode_for_pair(backend, plan, variation, job.clone())?;
    ensure!(
        outcome.command_status == EvaluationStatus::Completed,
        "fresh encoding did not complete: {:?}: {:?}",
        outcome.command_status,
        outcome.reason
    );
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == job.directory.join("output.json"))
        .context("fresh context output absent")?
        .clone();
    let context_proof = proof(experiment, journal, job)?;
    let context_execution = serde_json::to_vec(&context_proof.attest(input_file, &output)?)?;
    let manifest = fs::read(&qualification.reference_manifest.path)?;
    // The same case-execution attestations the producer registration a teaching freeze
    // preregisters pins (`CompletedQualification::producer_registration`).
    let executions = qualification.case_execution_attestations()?;
    let bundle = QualifiedContextBundle {
        artifact: fs::read(&output.path)?,
        qualification: fs::read(&qualification.qualification.path)?,
        reference_manifest: manifest,
        pins: context_producer::AdmissionPins {
            source_sha256: config.source_sha40.clone(),
            producer_sha256: config.encoder.program.sha256.clone(),
            artifact_sha256: output.sha256.clone(),
            qualification_sha256: qualification.qualification.sha256.clone(),
            reference_manifest_sha256: qualification.reference_manifest.sha256.clone(),
            context_execution_sha256: jope_pipeline::sha256_hex(&context_execution),
            qualification_execution_sha256: executions
                .iter()
                .map(|b| jope_pipeline::sha256_hex(b))
                .collect(),
            verification_policy_sha256: context_producer::verification_policy_sha256(),
            training: input.identity.clone(),
            input_sha256: input_file.sha256.clone(),
            preprocessing_sha256: plan.preprocessing_sha256.clone(),
        },
        context_execution,
        case_executions: executions,
    };
    let admitted = bundle.verify()?;
    let context_directory = journal.join("asm-contexts").join(plan.arm(true)?.id_sha256);
    let vector = immutable_bytes(
        &context_directory.join("context.f32"),
        &admitted
            .context()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let bundle_file = immutable(&context_directory.join("bundle.json"), &bundle)?;
    let files = CpuContextFiles {
        vector,
        artifact: output,
        frozen_transforms: config.frozen_transforms.clone(),
        bundle: bundle_file,
        context_execution: context_proof,
        case_executions: qualification.case_executions.clone(),
        training: TrainingIdentity {
            prepared_sha256: plan.prepared_sha256.clone(),
            training_sha256: plan.training_sha256.clone(),
            selected_training_rows_sha256: plan.selection.outer().fitting_rows_sha256.clone(),
            preprocessing_sha256: plan.preprocessing_sha256.clone(),
            cut: plan.cut,
        },
    };
    // Durable input to train prepare-request; callers need not reconstruct opaque proof pins.
    immutable(&context_directory.join("context-files.json"), &files)?;
    // W3-T02: the run also hands the qualified context off to teacher admission. Teaching
    // admission retains this record with the run's pool receipts (`teaching::admit_dataset`).
    let transforms =
        context_producer::FrozenTransforms::from_bytes(&config.frozen_transforms.read_verified()?)?;
    let handoff = admitted.handoff(
        &admitted.replay_assembly(&transforms)?,
        context_producer::ContextConsumer::TeacherAdmission,
    )?;
    immutable_bytes(
        &context_directory.join(TEACHER_HANDOFF),
        &serde_json::to_vec(&handoff)?,
    )?;
    Ok((files, outcome))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn distinct_classification_sampler_retains_typed_heads_and_eighty_unique_slots() -> Result<()> {
        use feat_math::advanced_v3::{validate_v3_genome, ColumnSpec, DataSchema, RowSemantics};
        let schema = DataSchema {
            columns: vec![ColumnSpec::numeric("x")],
            row_semantics: RowSemantics::Iid,
        };
        for task in [Task::Binary, Task::Multiclass] {
            let mut seen = std::collections::BTreeSet::new();
            for seed in 0..80 {
                let source = classification_generic(task, seed)?;
                let recipe = jope_pipeline::synthesized_recipe::SynthesizedRecipe::new(source)?;
                validate_v3_genome(
                    &feat_math::GenomeIr::parse(&recipe.executable_canonical)?,
                    &super::super::methods::task_spec(task)?,
                    &schema,
                )?;
                seen.insert(recipe.executable_sha256);
            }
            assert_eq!(seen.len(), 80);
            assert_ne!(generic_policy(task), GENERIC_POLICY);
        }
        Ok(())
    }
}
