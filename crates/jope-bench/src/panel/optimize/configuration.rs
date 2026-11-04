use super::*;
use std::collections::BTreeMap;

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
    pub experimental:
        BTreeMap<String, super::super::experimental_proposal::ExperimentalConfiguration>,
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
    pub teaching: Option<super::super::teaching::Settings>,
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
            self.scheduling_policy == super::super::optimize_schedule::POLICY
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
        let inventory: super::super::worker_build::WorkerBuildInventory =
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
