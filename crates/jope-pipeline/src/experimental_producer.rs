//! Separate experimental Python producer admission. Accepted step60000 identities
//! are never generalized here. Only a completed bounded job plus a separately
//! verified native context can construct the private population token.
use crate::{
    classification_bridge::VerifiedExperimentalContext,
    cpu_context_bridge::{self, AdmissionExpectation, CpuContextFiles, ExecutionProof},
    execution::{Artifact, WorkerJob},
    neural_search::{ContextCost, Variation},
    search_context::TrainingIdentity,
    sha256_hex,
    synthesized_recipe::SynthesizedRecipe,
};
use anyhow::{ensure, Context, Result};
use feat_math::{
    advanced_v3::{validate_v3_genome, DataSchema, Objective, TaskSpec},
    GenomeIr,
};
use jope_datasets::catalog::Task;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub const PLAN_SCHEMA: &str = "jope.experimental-proposal-plan/v1";
pub const REQUEST_SCHEMA: &str = "jope.experimental-proposal-request/v1";
pub const OUTPUT_SCHEMA: &str = "jope.experimental-proposal-output/v1";
pub const LINEAGE: &str = "asm.experimental-python-policy-v3/proposals-v1";
pub const NUMERICS: &str = "torch-cpu-f32:threads=1:deterministic:task-conditioned-mean-reward/v1";
pub const SELECTION: &str = "mean-reward-desc-executable-sha256-first64/v1";
pub const BOOTSTRAP: &str = "import json,os,sys,sysconfig; roots=json.loads(sys.argv[1]); stdlib=sysconfig.get_path('stdlib'); sys.path[:]=roots+[stdlib,sysconfig.get_path('platstdlib'),os.path.join(stdlib,'lib-dynload')]; from jope.policy.experimental_proposal import main; main(sys.argv[2:])";
const MAX_METADATA: u64 = 32 << 20;

pub const REGRESSION_CONTEXT: &str = "asm.experimental-regression-context/v1";
/// Only native receipt admission constructs this token. Serialized vectors or
/// experimental checkpoint metadata cannot create qualified context authority.
pub struct VerifiedProposalContext {
    artifact: serde_json::Value,
    training: TrainingIdentity,
    task: Task,
    sha256: String,
    lineage: &'static str,
    regression_encoder: Option<jope_policy::context_producer::EncoderIdentity>,
}
impl VerifiedProposalContext {
    pub fn classification(context: VerifiedExperimentalContext) -> Result<Self> {
        let task = match context.input().task {
            jope_policy::context_producer::classification::ClassificationTask::Binary => {
                Task::Binary
            }
            jope_policy::context_producer::classification::ClassificationTask::Multiclass => {
                Task::Multiclass
            }
        };
        Ok(Self {
            artifact: serde_json::to_value(context.context())?,
            training: context.training().clone(),
            task,
            sha256: context.sha256().into(),
            lineage: crate::search_context::CLASSIFICATION_LINEAGE,
            regression_encoder: None,
        })
    }
    pub fn regression(files: &CpuContextFiles, expected: &AdmissionExpectation) -> Result<Self> {
        let context = cpu_context_bridge::load_qualified(files, expected)?;
        let produced: jope_policy::context_producer::CpuContextArtifact = read(&files.artifact)?;
        let sha256 = jope_policy::context_sha256(context.context());
        let artifact = serde_json::json!({
            "lineage": REGRESSION_CONTEXT,
            "training": expected.training,
            "values": context.context(),
            "context_sha256": sha256,
            "classes": 0,
            "policy_compatible": false,
        });
        Ok(Self {
            artifact,
            training: expected.training.clone(),
            task: Task::Regression,
            sha256,
            lineage: REGRESSION_CONTEXT,
            regression_encoder: Some(produced.encoder),
        })
    }
    pub fn artifact(&self) -> &serde_json::Value {
        &self.artifact
    }
    pub fn training(&self) -> &TrainingIdentity {
        &self.training
    }
    pub fn sha256(&self) -> &str {
        &self.sha256
    }
    fn require_task(&self, task: Task) -> Result<()> {
        ensure!(
            self.task == task,
            "qualified context task differs from experimental checkpoint"
        );
        Ok(())
    }
}

pub fn digest(value: &impl Serialize) -> Result<String> {
    Ok(sha256_hex(&serde_json::to_vec(value)?))
}
pub fn same(a: &impl Serialize, b: &impl Serialize) -> Result<bool> {
    Ok(serde_json::to_value(a)? == serde_json::to_value(b)?)
}
pub fn read<T: serde::de::DeserializeOwned>(artifact: &Artifact) -> Result<T> {
    artifact.verify()?;
    ensure!(
        fs::metadata(&artifact.path)?.len() <= MAX_METADATA,
        "experimental metadata exceeds32MiB"
    );
    Ok(serde_json::from_slice(&fs::read(&artifact.path)?)?)
}
fn hash(value: &str) -> bool {
    crate::experiment::sha256(value)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sampling {
    pub count: usize,
    pub temperature: f64,
    pub top_p: f64,
    pub max_len: usize,
}
impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (64..=512).contains(&self.count)
                && (8..=120).contains(&self.max_len)
                && self.temperature.is_finite()
                && self.temperature > 0.
                && self.temperature <= 2.
                && self.top_p.is_finite()
                && self.top_p > 0.
                && self.top_p <= 1.,
            "invalid experimental sampling allowance"
        );
        Ok(())
    }
}

/// Paths and installed bytes are frozen by the controller before either arm.
/// Python runs the physical interpreter with these explicit import roots under -I.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonRuntime {
    pub manifest: Artifact,
    pub program: Artifact,
    pub source_root: PathBuf,
    pub site_packages: PathBuf,
    pub source_files: Vec<Artifact>,
    pub package_files: Vec<Artifact>,
    pub runtime_sha256: String,
    pub environment: BTreeMap<String, String>,
}
impl PythonRuntime {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            hash(&self.runtime_sha256),
            "invalid Python runtime identity"
        );
        for root in [&self.source_root, &self.site_packages] {
            ensure!(
                root.is_absolute()
                    && root.canonicalize()? == *root
                    && root.is_dir()
                    && !root.starts_with("/tmp"),
                "Python import root is not durable physical directory"
            );
        }
        self.program.verify()?;
        ensure!(
            self.program.path.canonicalize()? == self.program.path,
            "Python program must be physical"
        );
        ensure!(
            !self.source_files.is_empty() && !self.package_files.is_empty(),
            "Python source/package inventory empty"
        );
        let mut paths = BTreeSet::new();
        for artifact in self.source_files.iter().chain(&self.package_files) {
            artifact.verify()?;
            ensure!(
                paths.insert(&artifact.path),
                "duplicate Python dependency path"
            );
        }
        ensure!(
            self.source_files.iter().any(|a| a.path
                == self
                    .source_root
                    .join("jope/policy/experimental_proposal.py")),
            "proposal module absent from source closure"
        );
        ensure!(
            self.environment.keys().all(|key| matches!(
                key.as_str(),
                "PATH" | "LANG" | "LC_ALL" | "PYTHONDONTWRITEBYTECODE"
            )),
            "unreviewed Python environment override"
        );
        let manifest: RuntimeManifest = read(&self.manifest)?;
        ensure!(
            manifest.schema == "jope.experimental-python-runtime/v1"
                && same(&manifest.runtime, &self.without_manifest())?,
            "Python runtime manifest differs from controller pins"
        );
        let identity: serde_json::Value = read(&manifest.identity)?;
        // Python's runtime object contains integers, strings and maps only. Sorted
        // serde_json Value keys match campaign.encoded; no float spelling ambiguity.
        ensure!(
            digest(&identity)? == self.runtime_sha256
                && identity["python"] == serde_json::to_value(&self.program)?,
            "runtime probe interpreter/identity changed"
        );
        let sources: BTreeMap<String, String> =
            serde_json::from_value(identity["sources"].clone())?;
        let expected = self
            .source_files
            .iter()
            .map(|a| {
                let relative = a
                    .path
                    .strip_prefix(self.source_root.join("jope"))
                    .context("source outside pinned Jope tree")?;
                Ok((
                    relative
                        .to_str()
                        .context("source path not UTF8")?
                        .to_owned(),
                    a.sha256.clone(),
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        ensure!(
            sources == expected,
            "runtime source inventory differs from actual pinned files"
        );
        let packages = identity["packages"]
            .as_object()
            .context("runtime package inventory absent")?;
        ensure!(
            packages.len() == manifest.packages.len(),
            "runtime package count changed"
        );
        let mut installed = BTreeMap::new();
        for (name, package) in &manifest.packages {
            let observed = packages
                .get(name)
                .context("pinned distribution absent from runtime probe")?;
            let hashes = package
                .files
                .iter()
                .map(|(relative, artifact)| (relative.clone(), artifact.sha256.clone()))
                .collect::<BTreeMap<_, _>>();
            ensure!(
                observed["version"] == package.version
                    && observed["files"].as_u64() == Some(hashes.len() as u64)
                    && observed["files_sha256"] == digest(&hashes)?,
                "installed distribution inventory changed"
            );
            for artifact in package.files.values() {
                if let Some(old) = installed.insert(artifact.path.clone(), artifact.sha256.clone())
                {
                    ensure!(
                        old == artifact.sha256,
                        "shared installed file hash disagrees"
                    );
                }
            }
        }
        let pinned = self
            .package_files
            .iter()
            .map(|a| (a.path.clone(), a.sha256.clone()))
            .collect::<BTreeMap<_, _>>();
        ensure!(
            installed == pinned,
            "Python package closure is incomplete or contains unmeasured files"
        );
        Ok(())
    }
    fn without_manifest(&self) -> RuntimePins {
        RuntimePins {
            program: self.program.clone(),
            source_root: self.source_root.clone(),
            site_packages: self.site_packages.clone(),
            source_files: self.source_files.clone(),
            package_files: self.package_files.clone(),
            runtime_sha256: self.runtime_sha256.clone(),
            environment: self.environment.clone(),
        }
    }
    pub fn dependencies(&self) -> Vec<&Artifact> {
        [&self.manifest, &self.program]
            .into_iter()
            .chain(&self.source_files)
            .chain(&self.package_files)
            .collect()
    }
    pub fn argv(&self, request: &Artifact, output: &Path) -> Result<Vec<String>> {
        Ok(vec![
            "-I".into(),
            "-B".into(),
            "-c".into(),
            BOOTSTRAP.into(),
            serde_json::to_string(&[&self.source_root, &self.site_packages])?,
            "--request".into(),
            request.path.to_string_lossy().into_owned(),
            "--request-sha256".into(),
            request.sha256.clone(),
            "--output".into(),
            output.to_string_lossy().into_owned(),
        ])
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimePins {
    pub program: Artifact,
    pub source_root: PathBuf,
    pub site_packages: PathBuf,
    pub source_files: Vec<Artifact>,
    pub package_files: Vec<Artifact>,
    pub runtime_sha256: String,
    pub environment: BTreeMap<String, String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeManifest {
    pub schema: String,
    pub runtime: RuntimePins,
    pub identity: Artifact,
    pub packages: BTreeMap<String, DistributionInventory>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionInventory {
    pub version: String,
    /// Keys are importlib.metadata distribution file names, not rewritten basenames.
    pub files: BTreeMap<String, Artifact>,
}

/// All fields exist before encoding. Output context and request paths attach only
/// after the original neural clock is admitted; they cannot replace this plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RequestPlan {
    pub schema: String,
    pub task: Task,
    pub worker_source_sha256: String,
    pub runtime: PythonRuntime,
    pub export_manifest: Artifact,
    pub checkpoint: Artifact,
    pub vocabulary: Artifact,
    pub canonicalizer: Artifact,
    pub canonicalizer_source_sha40: String,
    pub training: TrainingIdentity,
    pub data_schema: DataSchema,
    pub neural_seed: u64,
    pub sampling: Sampling,
    pub generic_bank_sha256: String,
    pub generic_seed: u64,
    pub generic_draws: usize,
    pub generic_policy: String,
}
impl RequestPlan {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == PLAN_SCHEMA
                && matches!(
                    self.task,
                    Task::Regression | Task::Binary | Task::Multiclass
                ),
            "experimental plan requires explicit supported task"
        );
        self.sampling.validate()?;
        ensure!(
            (80..=4096 - self.sampling.count).contains(&self.generic_draws)
                && !self.generic_policy.is_empty(),
            "invalid experimental generic allowance"
        );
        for value in [
            &self.worker_source_sha256,
            &self.generic_bank_sha256,
            &self.runtime.runtime_sha256,
            &self.training.prepared_sha256,
            &self.training.training_sha256,
            &self.training.selected_training_rows_sha256,
            &self.training.preprocessing_sha256,
        ] {
            ensure!(hash(value), "invalid experimental plan digest");
        }
        ensure!(
            self.training.cut < 5
                && self.canonicalizer_source_sha40.len() == 40
                && self
                    .canonicalizer_source_sha40
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit()),
            "invalid experimental cut/source"
        );
        self.data_schema.validate()?;
        for a in [
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
        ] {
            ensure!(
                a.path.is_absolute() && hash(&a.sha256),
                "invalid experimental artifact pin"
            );
        }
        Ok(())
    }
    pub fn dependencies(&self) -> Vec<&Artifact> {
        let mut out = self.runtime.dependencies();
        out.extend([
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
        ]);
        out
    }
    pub fn request(&self, context: Artifact, bank: Artifact) -> Result<Request> {
        self.validate()?;
        ensure!(
            bank.sha256 == self.generic_bank_sha256,
            "experimental bank differs from frozen plan"
        );
        Ok(Request {
            schema: REQUEST_SCHEMA.into(),
            task: self.task,
            export_manifest: self.export_manifest.clone(),
            checkpoint: self.checkpoint.clone(),
            vocabulary: self.vocabulary.clone(),
            context,
            generic_bank: bank,
            canonicalizer: self.canonicalizer.clone(),
            canonicalizer_source_sha40: self.canonicalizer_source_sha40.clone(),
            runtime_sha256: self.runtime.runtime_sha256.clone(),
            training: self.training.clone(),
            data_schema: self.data_schema.clone(),
            neural_seed: self.neural_seed,
            sampling: self.sampling.clone(),
        })
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub schema: String,
    pub task: Task,
    pub export_manifest: Artifact,
    pub checkpoint: Artifact,
    pub vocabulary: Artifact,
    pub context: Artifact,
    pub generic_bank: Artifact,
    pub canonicalizer: Artifact,
    pub canonicalizer_source_sha40: String,
    pub runtime_sha256: String,
    pub training: TrainingIdentity,
    pub data_schema: DataSchema,
    pub neural_seed: u64,
    pub sampling: Sampling,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenericBank {
    pub schema: String,
    pub task: Task,
    pub policy: String,
    pub seed: u64,
    pub candidates: Vec<GenericDraw>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GenericDraw {
    pub seed: u64,
    pub genome: String,
}
/// Same policy as the generic arm; draw seeds and every original program remain
/// deterministic and individually retained. The bank is never supplied by Python.
pub fn generic_bank<V: Variation>(
    task: Task,
    seed: u64,
    draws: usize,
    variation: &mut V,
) -> Result<GenericBank> {
    ensure!(
        (80..=4096).contains(&draws),
        "generic bank count outside bounds"
    );
    let mut candidates = vec![];
    for index in 0..draws {
        let text =
            sha256_hex(format!("asm.experimental-generic-bank/v1/{seed}/{index}").as_bytes());
        let draw = u64::from_str_radix(&text[..16], 16)?;
        let genome = variation
            .generic(draw)?
            .context("typed generic sampler refused bank draw")?;
        ensure!(
            !genome.is_empty() && genome.len() <= 16384,
            "generic program outside bounds"
        );
        candidates.push(GenericDraw { seed: draw, genome });
    }
    Ok(GenericBank {
        schema: "jope.experimental-generic-bank/v1".into(),
        task,
        policy: variation.generic_policy().into(),
        seed,
        candidates,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Worker {
    pub job: WorkerJob,
    pub request: Request,
    pub request_artifact: Artifact,
    pub source_manifest: Artifact,
    pub context_cost: ContextCost,
    pub output_name: PathBuf,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateClaim {
    pub slot: usize,
    pub recipe: SynthesizedRecipe,
    pub origin: String,
    pub rank_log_weight: Option<f64>,
    pub reward_mean: Option<f64>,
    pub reward_std: Option<f64>,
    pub bank_index: Option<usize>,
    pub candidate_seed: Option<u64>,
    pub program_index: usize,
    pub tokens: Option<Vec<usize>>,
    pub sample_index: Option<usize>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claims {
    pub schema: String,
    pub lineage: String,
    pub serving_eligible: bool,
    pub request_sha256: String,
    pub request_identity_sha256: String,
    pub backend: String,
    pub task: Task,
    pub training: TrainingIdentity,
    pub checkpoint_sha256: String,
    pub export_manifest_sha256: String,
    pub export_verification_sha256: String,
    pub training_proof_sha256: String,
    pub parameter_sha256: String,
    pub checkpoint_training_data_kind: String,
    pub vocabulary_sha256: String,
    pub context_artifact_sha256: String,
    pub context_sha256: String,
    pub context_lineage: String,
    pub runtime_sha256: String,
    pub numerical_policy: String,
    pub selection_policy: String,
    pub sampling: Sampling,
    pub neural_seed: u64,
    pub generic_bank_sha256: String,
    pub generic_policy: String,
    pub generic_seed: u64,
    pub canonicalizer: Artifact,
    pub canonicalizer_source_sha40: String,
    pub canonicalization_sha256: String,
    pub neural_candidates: usize,
    pub rejections: Vec<serde_json::Value>,
    pub candidates: Vec<CandidateClaim>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evidence {
    pub schema: String,
    pub pair_sha256: String,
    pub request: Request,
    pub context_cost: ContextCost,
    pub producer_job_sha256: String,
    pub producer_output_sha256: String,
    pub population_sha256: String,
    pub context_sha256: String,
    pub checkpoint_sha256: String,
    pub ranking_backend: String,
    pub export_verification_sha256: String,
    pub training_proof_sha256: String,
    pub parameter_sha256: String,
    pub checkpoint_training_data_kind: String,
    pub execution_proof: ExecutionProof,
}
/// Private constructor: deserializing claims cannot create this token.
pub struct AdmittedPopulation {
    candidates: Vec<CandidateClaim>,
    evidence: Evidence,
}
impl AdmittedPopulation {
    pub fn candidates(&self) -> &[CandidateClaim] {
        &self.candidates
    }
    pub fn evidence(&self) -> &Evidence {
        &self.evidence
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportManifest {
    schema: String,
    lineage: String,
    serving_eligible: bool,
    artifacts: Vec<ExportFile>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExportFile {
    path: PathBuf,
    sha256: String,
}
/// Only the model/vocabulary plus plan and verification metadata are read here.
/// Corpus roots and other categories' checkpoints never enter the proposal job.
pub fn export_metadata(plan: &RequestPlan) -> Result<Vec<Artifact>> {
    let manifest: ExportManifest = read(&plan.export_manifest)?;
    ensure!(
        manifest.schema == "jope.experimental-training-export/v1"
            && manifest.lineage == "asm.experimental-python-policy-v3/v1"
            && !manifest.serving_eligible,
        "not an experimental export"
    );
    let root = plan
        .export_manifest
        .path
        .parent()
        .context("export parent absent")?;
    let mut files = BTreeMap::new();
    for file in manifest.artifacts {
        ensure!(
            !file.path.as_os_str().is_empty()
                && file
                    .path
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)))
                && hash(&file.sha256),
            "invalid export entry"
        );
        ensure!(
            files
                .insert(
                    file.path.clone(),
                    Artifact {
                        path: root.join(file.path),
                        sha256: file.sha256
                    }
                )
                .is_none(),
            "duplicate export path"
        );
    }
    ensure!(
        same(
            files
                .get(&PathBuf::from(format!("{}/policy.pt", plan.task.as_str())))
                .context("export checkpoint absent")?,
            &plan.checkpoint
        )?,
        "checkpoint differs from export"
    );
    ensure!(
        files.iter().any(|(p, a)| p
            .to_string_lossy()
            .starts_with(&format!("{}/vocabulary.", plan.task.as_str()))
            && same(a, &plan.vocabulary).unwrap_or(false)),
        "vocabulary differs from export"
    );
    let training = files
        .get(Path::new("plan.json"))
        .context("training plan absent")?
        .clone();
    let verification = files
        .get(Path::new("verification.json"))
        .context("training verification absent")?
        .clone();
    let value: serde_json::Value = read(&training)?;
    let proof: serde_json::Value = read(&verification)?;
    ensure!(
        value["schema"] == "jope.training-plan/v1"
            && value["lineage"] == "asm.experimental-python-policy-v3/v1"
            && proof["schema"] == "jope.training-verification/v1"
            && proof["lineage"] == value["lineage"]
            && proof["plan_sha256"] == value["plan_sha256"]
            && proof["completed_plan"] == true
            && value["source_sha"] == plan.canonicalizer_source_sha40,
        "incomplete/mismatched training verification"
    );
    ensure!(
        value["category_inputs"][plan.task.as_str()]["inputs"]["vocabulary"]["sha256"]
            == plan.vocabulary.sha256,
        "trained vocabulary changed"
    );
    let records = proof["tasks"]
        .as_array()
        .context("training tasks absent")?
        .iter()
        .filter(|r| r["task"] == plan.task.as_str())
        .collect::<Vec<_>>();
    ensure!(
        records.len() == 1
            && records[0]["planned_passes"]
                .as_u64()
                .is_some_and(|n| n >= 2)
            && records[0]["completed_passes"].as_u64() >= records[0]["planned_passes"].as_u64(),
        "missing optimizer continuation proof"
    );
    let mut metadata = vec![training, verification];
    if plan.task == Task::Regression {
        ensure!(
            value["category_inputs"]["regression"]["data_kind"] == "admitted",
            "experimental regression consumption requires admitted real training"
        );
        let encoder = files
            .iter()
            .find(|(path, _)| path.to_string_lossy().starts_with("regression/encoder."))
            .map(|(_, file)| file.clone())
            .context("trained regression encoder absent")?;
        ensure!(
            value["category_inputs"]["regression"]["inputs"]["encoder"]["sha256"] == encoder.sha256,
            "trained regression encoder identity changed"
        );
        metadata.push(encoder);
    }
    Ok(metadata)
}

pub fn verify_inputs<V: Variation>(
    plan: &RequestPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    variation: &mut V,
) -> Result<GenericBank> {
    plan.validate()?;
    context.require_task(plan.task)?;
    plan.runtime.validate()?;
    worker.context_cost.validate()?;
    ensure!(
        same(context.training(), &plan.training)?,
        "qualified context training rows differ"
    );
    ensure!(
        variation.generic_policy() == plan.generic_policy,
        "experimental generic policy changed"
    );
    let expected_bank = generic_bank(plan.task, plan.generic_seed, plan.generic_draws, variation)?;
    ensure!(
        digest(&expected_bank)? == plan.generic_bank_sha256,
        "generic bank differs from deterministic draw policy"
    );
    let bank: GenericBank = read(&worker.request.generic_bank)?;
    ensure!(
        same(&bank, &expected_bank)?,
        "worker bank changed original seeds/programs"
    );
    let expected = plan.request(
        worker.request.context.clone(),
        worker.request.generic_bank.clone(),
    )?;
    ensure!(
        same(&worker.request, &expected)?,
        "experimental worker request differs from frozen plan"
    );
    let actual: Request = read(&worker.request_artifact)?;
    ensure!(
        same(&actual, &worker.request)?,
        "request artifact differs from declared request"
    );
    let actual_context: serde_json::Value = read(&worker.request.context)?;
    ensure!(
        actual_context == *context.artifact(),
        "context artifact differs from independently qualified native context"
    );
    ensure!(
        worker.source_manifest.sha256 == plan.worker_source_sha256,
        "experimental source manifest changed"
    );
    ensure!(
        same(&worker.job.command.program, &plan.runtime.program)?
            && worker.job.command.args
                == plan.runtime.argv(
                    &worker.request_artifact,
                    &worker.job.directory.join(&worker.output_name)
                )?
            && worker.job.command.environment == plan.runtime.environment,
        "Python invocation differs from frozen physical interpreter/source environment"
    );
    ensure!(
        worker.job.request.task == plan.task
            && worker.job.request.fitting_rows_sha256
                == plan.training.selected_training_rows_sha256
            && worker.job.request.fold as usize == plan.training.cut,
        "proposal diagnostic request changes task/training rows"
    );
    let metadata = export_metadata(plan)?;
    if plan.task == Task::Regression {
        let trained_encoder: jope_policy::context_producer::EncoderIdentity = read(&metadata[2])?;
        ensure!(
            context.regression_encoder.as_ref() == Some(&trained_encoder),
            "proposal encoder differs from the trained regression context encoder"
        );
    }
    let mut required = plan.dependencies();
    required.extend([
        &worker.request_artifact,
        &worker.request.context,
        &worker.request.generic_bank,
        &worker.source_manifest,
    ]);
    required.extend(metadata.iter());
    for artifact in required {
        artifact.verify()?;
        ensure!(
            worker
                .job
                .command
                .inputs
                .iter()
                .any(|a| same(a, artifact).unwrap_or(false)),
            "experimental dependency absent from pinned job closure"
        );
    }
    ensure!(
        worker.output_name == Path::new("proposals.json")
            && [
                "proposals.json",
                "canonicalization-input.json",
                "canonicalization-output.json",
                "canonicalizer.stdout",
                "canonicalizer.stderr"
            ]
            .iter()
            .all(|p| worker.job.outputs.contains(&PathBuf::from(p))),
        "experimental output inventory incomplete"
    );
    Ok(bank)
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Canonicalization {
    schema: String,
    input_sha256: String,
    results: Vec<CanonicalResult>,
}
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CanonicalResult {
    index: usize,
    original_bytes_sha256: String,
    recipe: Option<SynthesizedRecipe>,
    error: Option<String>,
}
fn task_spec(task: Task) -> Result<TaskSpec> {
    match task {
        Task::Regression => TaskSpec::regression(Objective::Rmse),
        Task::Binary => TaskSpec::binary(Objective::RocAuc),
        Task::Multiclass => TaskSpec::multiclass(Objective::MacroF1),
        _ => anyhow::bail!("experimental tabular task required"),
    }
}
fn validate_recipe(recipe: &SynthesizedRecipe, plan: &RequestPlan) -> Result<()> {
    recipe.validate()?;
    validate_v3_genome(
        &GenomeIr::parse(&recipe.executable_canonical)?,
        &task_spec(plan.task)?,
        &plan.data_schema,
    )?;
    Ok(())
}
/// Structural claims checking is private and cannot create admission. The sole
/// public constructor below also requires qualified context and actual pool proof.
fn check_claims(
    plan: &RequestPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    bank: &GenericBank,
    claims: &Claims,
    canonicalization: &Canonicalization,
) -> Result<()> {
    ensure!(
        claims.schema == OUTPUT_SCHEMA
            && claims.lineage == LINEAGE
            && !claims.serving_eligible
            && claims.backend == "torch-cpu"
            && claims.numerical_policy == NUMERICS
            && claims.selection_policy == SELECTION,
        "experimental output lineage/numerics changed"
    );
    ensure!(
        claims.request_sha256 == worker.request_artifact.sha256
            && hash(&claims.request_identity_sha256)
            && claims.task == plan.task
            && same(&claims.training, &plan.training)?
            && claims.checkpoint_sha256 == plan.checkpoint.sha256
            && claims.export_manifest_sha256 == plan.export_manifest.sha256
            && claims.vocabulary_sha256 == plan.vocabulary.sha256
            && claims.context_artifact_sha256 == worker.request.context.sha256
            && claims.context_sha256 == context.sha256()
            && claims.context_lineage == context.lineage
            && claims.runtime_sha256 == plan.runtime.runtime_sha256
            && claims.neural_seed == plan.neural_seed
            && same(&claims.sampling, &plan.sampling)?
            && claims.generic_bank_sha256 == plan.generic_bank_sha256
            && claims.generic_policy == plan.generic_policy
            && claims.generic_seed == plan.generic_seed
            && same(&claims.canonicalizer, &plan.canonicalizer)?
            && claims.canonicalizer_source_sha40 == plan.canonicalizer_source_sha40,
        "experimental output input binding changed"
    );
    for value in [
        &claims.export_verification_sha256,
        &claims.training_proof_sha256,
        &claims.parameter_sha256,
        &claims.canonicalization_sha256,
    ] {
        ensure!(hash(value), "invalid experimental proof digest");
    }
    let metadata = export_metadata(plan)?;
    ensure!(
        claims.export_verification_sha256 == metadata[1].sha256,
        "output training proof differs from export"
    );
    let verification: serde_json::Value = read(&metadata[1])?;
    let record = verification["tasks"]
        .as_array()
        .context("export tasks")?
        .iter()
        .find(|r| r["task"] == plan.task.as_str())
        .context("export task")?;
    ensure!(
        record["parameter_sha256"] == claims.parameter_sha256,
        "actual checkpoint parameter proof differs from export"
    );
    let training: serde_json::Value = read(&metadata[0])?;
    ensure!(
        training["category_inputs"][plan.task.as_str()]["data_kind"]
            == claims.checkpoint_training_data_kind
            && ["synthetic_fixture", "admitted"]
                .contains(&claims.checkpoint_training_data_kind.as_str()),
        "checkpoint training data provenance changed"
    );
    check_candidates(
        plan,
        &claims.candidates,
        claims.neural_candidates,
        bank,
        canonicalization,
    )
}

// This is a structural helper only; no public admission token can be made here.
fn check_candidates(
    plan: &RequestPlan,
    candidates: &[CandidateClaim],
    neural_candidates: usize,
    bank: &GenericBank,
    canonicalization: &Canonicalization,
) -> Result<()> {
    ensure!(
        candidates.len() == 80 && neural_candidates <= 64,
        "experimental initial slot count changed"
    );
    let mut seen = BTreeSet::new();
    let mut previous: Option<&CandidateClaim> = None;
    for (slot, candidate) in candidates.iter().enumerate() {
        ensure!(
            candidate.slot == slot && seen.insert(candidate.recipe.executable_sha256.clone()),
            "reordered or duplicate experimental initial recipe"
        );
        validate_recipe(&candidate.recipe, plan)?;
        let validated = canonicalization
            .results
            .get(candidate.program_index)
            .context("candidate canonicalization slot absent")?;
        ensure!(
            validated.index == candidate.program_index
                && validated.error.is_none()
                && validated.original_bytes_sha256 == candidate.recipe.original_bytes_sha256
                && validated
                    .recipe
                    .as_ref()
                    .is_some_and(|r| same(r, &candidate.recipe).unwrap_or(false)),
            "candidate recipe differs from exact native batch output"
        );
        if slot < neural_candidates {
            ensure!(
                candidate.origin == "neural_policy"
                    && candidate.bank_index.is_none()
                    && candidate.candidate_seed.is_none()
                    && candidate
                        .sample_index
                        .is_some_and(|i| i < plan.sampling.count)
                    && candidate
                        .tokens
                        .as_ref()
                        .is_some_and(|t| !t.is_empty() && t.len() <= plan.sampling.max_len + 4)
                    && candidate.reward_mean.is_some_and(f64::is_finite)
                    && candidate
                        .reward_std
                        .is_some_and(|v| v.is_finite() && v >= 0.)
                    && candidate.rank_log_weight.is_some_and(f64::is_finite),
                "neural slot lacks actual sampled/ranked evidence"
            );
            if let Some(old) = previous {
                ensure!(
                    old.reward_mean.unwrap() > candidate.reward_mean.unwrap()
                        || (old.reward_mean == candidate.reward_mean
                            && old.recipe.executable_sha256 <= candidate.recipe.executable_sha256),
                    "neural rank ordering changed"
                );
            }
            previous = Some(candidate);
        } else {
            ensure!(
                candidate.origin
                    == if slot < 64 {
                        "neural_rejection_fallback"
                    } else {
                        "random_exploration"
                    }
                    && candidate.tokens.is_none()
                    && candidate.sample_index.is_none()
                    && candidate.reward_mean.is_none()
                    && candidate.reward_std.is_none()
                    && candidate.rank_log_weight.is_none(),
                "generic slot invented neural provenance"
            );
        }
    }
    if neural_candidates > 0 {
        let neural = &candidates[..neural_candidates];
        let max = neural[0].reward_mean.unwrap();
        let normalizer = max
            + neural
                .iter()
                .map(|c| (c.reward_mean.unwrap() - max).exp())
                .sum::<f64>()
                .ln();
        ensure!(
            neural.iter().all(|c| (c.rank_log_weight.unwrap()
                - (c.reward_mean.unwrap() - normalizer))
                .abs()
                <= 1e-12),
            "neural rank weights changed"
        );
    }
    // Recreate fallback/exploration ordering from every original draw. Invalid or
    // duplicate entries alone may be skipped; output cannot cherry-pick the bank.
    let mut used = candidates[..neural_candidates]
        .iter()
        .map(|c| c.recipe.executable_sha256.clone())
        .collect::<BTreeSet<_>>();
    let mut cursor = neural_candidates;
    for (index, draw) in bank.candidates.iter().enumerate() {
        let Ok(recipe) = SynthesizedRecipe::new(draw.genome.clone()) else {
            continue;
        };
        if validate_recipe(&recipe, plan).is_err() || !used.insert(recipe.executable_sha256.clone())
        {
            continue;
        }
        let candidate = &candidates[cursor];
        ensure!(
            candidate.bank_index == Some(index)
                && candidate.candidate_seed == Some(draw.seed)
                && same(&candidate.recipe, &recipe)?,
            "fallback/exploration differs from original typed draw order"
        );
        cursor += 1;
        if cursor == 80 {
            break;
        }
    }
    ensure!(
        cursor == 80,
        "generic bank cannot supply complete population"
    );
    Ok(())
}

pub fn admit<V: Variation>(
    pair: &crate::neural_search::PairInputPlan,
    worker: &Worker,
    context: &VerifiedProposalContext,
    proof: &ExecutionProof,
    variation: &mut V,
) -> Result<AdmittedPopulation> {
    let crate::neural_search::ProposalPlan::Experimental(plan) = &pair.proposal else {
        anyhow::bail!("accepted plan cannot enter experimental admission")
    };
    let arm = pair.arm(true)?;
    ensure!(
        worker
            .job
            .arm
            .as_ref()
            .is_some_and(|a| a.id_sha256 == arm.id_sha256 && a.seconds == arm.seconds),
        "experimental execution changed original arm"
    );
    let diagnostic = crate::neural_search::diagnostic_request(
        pair,
        "asm.experimental-python-proposals/v1",
        &worker.request_artifact.sha256,
    )?;
    ensure!(
        same(&worker.job.request, &diagnostic)?,
        "experimental diagnostic request changed comparison identity"
    );
    let bank = verify_inputs(plan, worker, context, variation)?;
    ensure!(
        same(&proof.job, &worker.job)?,
        "proposal execution proof changed pinned job"
    );
    let output_path = worker.job.directory.join(&worker.output_name);
    let outcome: crate::execution::WorkerOutcome = read(&proof.outcome)?;
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == output_path)
        .context("proposal output absent from actual execution")?;
    let attestation = proof.attest(&worker.request_artifact, output)?;
    ensure!(
        attestation.program_sha256 == plan.runtime.program.sha256
            && attestation.threads == 1
            && attestation.cpu_limit == 1,
        "proposal process differs from bounded CPU policy"
    );
    let claims: Claims = read(output)?;
    let canonical_output = outcome
        .outputs
        .iter()
        .find(|a| a.path == worker.job.directory.join("canonicalization-output.json"))
        .context("native batch output absent")?;
    let canonical_input = outcome
        .outputs
        .iter()
        .find(|a| a.path == worker.job.directory.join("canonicalization-input.json"))
        .context("native batch input absent")?;
    let canonicalization: Canonicalization = read(canonical_output)?;
    ensure!(
        canonicalization.schema == "jope.experimental-canonicalization-response/v1"
            && canonicalization.input_sha256 == canonical_input.sha256
            && canonical_output.sha256 == claims.canonicalization_sha256,
        "native canonicalization output binding changed"
    );
    check_claims(plan, worker, context, &bank, &claims, &canonicalization)?;
    let evidence = Evidence {
        schema: "jope.experimental-proposal-evidence/v1".into(),
        pair_sha256: digest(pair)?,
        request: worker.request.clone(),
        context_cost: worker.context_cost.clone(),
        producer_job_sha256: outcome.job_sha256,
        producer_output_sha256: output.sha256.clone(),
        population_sha256: digest(&claims.candidates)?,
        context_sha256: context.sha256().into(),
        checkpoint_sha256: plan.checkpoint.sha256.clone(),
        ranking_backend: NUMERICS.into(),
        export_verification_sha256: claims.export_verification_sha256,
        training_proof_sha256: claims.training_proof_sha256,
        parameter_sha256: claims.parameter_sha256,
        checkpoint_training_data_kind: claims.checkpoint_training_data_kind,
        execution_proof: proof.clone(),
    };
    Ok(AdmittedPopulation {
        candidates: claims.candidates,
        evidence,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use feat_math::advanced_v3::{ColumnSpec, RowSemantics};

    struct FixtureVariation;
    impl Variation for FixtureVariation {
        fn identity(&self) -> &str {
            "synthetic-classifier-fixture/v1"
        }
        fn generic_policy(&self) -> &str {
            "synthetic-classifier-draw/v1"
        }
        fn generic(&mut self, seed: u64) -> Result<Option<String>> {
            Ok(Some(format!(
                "(pipe X (logit_ridge l2={}))",
                1 + seed % 1_000_000
            )))
        }
        fn mutate(&mut self, _: &str, _: u64) -> Result<String> {
            anyhow::bail!("fixture has no mutation")
        }
        fn crossover(&mut self, _: &str, _: &str, _: u64) -> Result<String> {
            anyhow::bail!("fixture has no crossover")
        }
    }
    fn reference() -> Artifact {
        Artifact {
            path: "/private/synthetic-metadata-only".into(),
            sha256: sha256_hex(b"fixture"),
        }
    }
    fn plan() -> RequestPlan {
        RequestPlan {
            schema: PLAN_SCHEMA.into(),
            task: Task::Binary,
            worker_source_sha256: sha256_hex(b"source"),
            runtime: PythonRuntime {
                manifest: reference(),
                program: reference(),
                source_root: "/private/source".into(),
                site_packages: "/private/packages".into(),
                source_files: vec![reference()],
                package_files: vec![reference()],
                runtime_sha256: sha256_hex(b"runtime"),
                environment: BTreeMap::new(),
            },
            export_manifest: reference(),
            checkpoint: reference(),
            vocabulary: reference(),
            canonicalizer: reference(),
            canonicalizer_source_sha40: "a".repeat(40),
            training: TrainingIdentity {
                prepared_sha256: sha256_hex(b"prepared"),
                training_sha256: sha256_hex(b"training"),
                selected_training_rows_sha256: sha256_hex(b"rows"),
                preprocessing_sha256: sha256_hex(b"preprocessing"),
                cut: 0,
            },
            data_schema: DataSchema {
                columns: vec![ColumnSpec::numeric("x")],
                row_semantics: RowSemantics::Iid,
            },
            neural_seed: 3,
            sampling: Sampling {
                count: 64,
                temperature: 0.7,
                top_p: 0.9,
                max_len: 120,
            },
            generic_bank_sha256: sha256_hex(b"bank"),
            generic_seed: 7,
            generic_draws: 100,
            generic_policy: "synthetic-classifier-draw/v1".into(),
        }
    }
    fn population() -> Result<(
        RequestPlan,
        GenericBank,
        Vec<CandidateClaim>,
        Canonicalization,
    )> {
        let plan = plan();
        let bank = generic_bank(Task::Binary, 7, 100, &mut FixtureVariation)?;
        let mut candidates = vec![];
        for (index, draw) in bank.candidates.iter().take(80).enumerate() {
            candidates.push(CandidateClaim {
                slot: index,
                recipe: SynthesizedRecipe::new(draw.genome.clone())?,
                origin: if index < 64 {
                    "neural_rejection_fallback"
                } else {
                    "random_exploration"
                }
                .into(),
                rank_log_weight: None,
                reward_mean: None,
                reward_std: None,
                bank_index: Some(index),
                candidate_seed: Some(draw.seed),
                program_index: index,
                tokens: None,
                sample_index: None,
            });
        }
        let native = Canonicalization {
            schema: "fixture".into(),
            input_sha256: sha256_hex(b"input"),
            results: candidates
                .iter()
                .enumerate()
                .map(|(index, c)| CanonicalResult {
                    index,
                    original_bytes_sha256: c.recipe.original_bytes_sha256.clone(),
                    recipe: Some(c.recipe.clone()),
                    error: None,
                })
                .collect(),
        };
        Ok((plan, bank, candidates, native))
    }
    #[test]
    fn generic_bank_preserves_each_seed_and_never_claims_neural_origin() -> Result<()> {
        let (plan, bank, candidates, native) = population()?;
        assert!(same(
            &bank,
            &generic_bank(Task::Binary, 7, 100, &mut FixtureVariation)?
        )?);
        assert!(!same(
            &bank,
            &generic_bank(Task::Binary, 8, 100, &mut FixtureVariation)?
        )?);
        assert!(generic_bank(Task::Binary, 7, 79, &mut FixtureVariation).is_err());
        check_candidates(&plan, &candidates, 0, &bank, &native)?;
        let mut invented = candidates.clone();
        invented[0].origin = "neural_policy".into();
        assert!(check_candidates(&plan, &invented, 0, &bank, &native).is_err());
        assert!(check_candidates(&plan, &invented, 1, &bank, &native).is_err());
        Ok(())
    }
    #[test]
    fn initial_claims_reject_reordering_duplicates_wrong_head_and_native_tampering() -> Result<()> {
        let (plan, bank, candidates, native) = population()?;
        for changed in [
            {
                let mut c = candidates.clone();
                c.swap(0, 1);
                c
            },
            {
                let mut c = candidates.clone();
                c[1].recipe = c[0].recipe.clone();
                c
            },
            {
                let mut c = candidates.clone();
                c[0].candidate_seed = Some(99);
                c
            },
            {
                let mut c = candidates.clone();
                c[0].recipe = SynthesizedRecipe::new("(pipe X (ridge lambda=1))".into())?;
                c
            },
        ] {
            assert!(check_candidates(&plan, &changed, 0, &bank, &native).is_err());
        }
        let mut changed = native.clone();
        changed.results[0].original_bytes_sha256 = sha256_hex(b"changed");
        assert!(check_candidates(&plan, &candidates, 0, &bank, &changed).is_err());
        assert!(check_candidates(&plan, &candidates[..79], 0, &bank, &native).is_err());
        Ok(())
    }
    #[test]
    fn experimental_plan_cannot_deserialize_as_accepted_or_change_task() -> Result<()> {
        let plan = plan();
        plan.validate()?;
        let bytes = serde_json::to_vec(&plan)?;
        assert!(serde_json::from_slice::<jope_policy::producer::RequestPlan>(&bytes).is_err());
        let wrapped: crate::neural_search::ProposalPlan = serde_json::from_slice(&bytes)?;
        assert!(wrapped.accepted().is_err());
        assert_eq!(serde_json::to_vec(&wrapped)?, bytes);
        let mut regression = plan;
        regression.task = Task::Regression;
        regression.validate()?;
        let bytes = serde_json::to_vec(&regression)?;
        let wrapped: crate::neural_search::ProposalPlan = serde_json::from_slice(&bytes)?;
        assert!(wrapped.accepted().is_err());
        regression.task = Task::Timeseries;
        assert!(regression.validate().is_err());
        Ok(())
    }
    #[test]
    fn experimental_regression_requires_matching_task_and_executable_head() -> Result<()> {
        let mut plan = plan();
        plan.task = Task::Regression;
        plan.validate()?;
        validate_recipe(
            &SynthesizedRecipe::new("(pipe X (ridge lambda=1))".into())?,
            &plan,
        )?;
        assert!(validate_recipe(
            &SynthesizedRecipe::new("(pipe X (logit_ridge l2=1))".into())?,
            &plan
        )
        .is_err());
        // Metadata-only local token fixture tests the dispatch guard; it is not
        // a context admitted through the public native receipt constructor.
        let token = VerifiedProposalContext {
            artifact: serde_json::Value::Null,
            training: plan.training.clone(),
            task: Task::Regression,
            sha256: sha256_hex(b"synthetic-only-context"),
            lineage: REGRESSION_CONTEXT,
            regression_encoder: None,
        };
        token.require_task(Task::Regression)?;
        assert!(token.require_task(Task::Binary).is_err());
        assert!(token.require_task(Task::Multiclass).is_err());
        Ok(())
    }
    #[test]
    fn runtime_inventory_detects_source_and_package_drift_before_launch() -> Result<()> {
        let directory = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/tmp");
        fs::create_dir_all(&directory)?;
        let temporary = tempfile::tempdir_in(directory)?;
        let root = temporary.path().canonicalize()?;
        let source = root.join("source");
        let packages = root.join("packages");
        fs::create_dir_all(source.join("jope/policy"))?;
        fs::create_dir_all(&packages)?;
        let put = |path: PathBuf, bytes: &[u8]| -> Result<Artifact> {
            fs::write(&path, bytes)?;
            Ok(Artifact {
                path,
                sha256: sha256_hex(bytes),
            })
        };
        let program = put(
            root.join("synthetic-physical-python"),
            b"metadata-only fixture",
        )?;
        let module = put(
            source.join("jope/policy/experimental_proposal.py"),
            b"# source fixture",
        )?;
        let package = put(packages.join("synthetic.py"), b"# package fixture")?;
        let hashes = BTreeMap::from([("synthetic.py".to_owned(), package.sha256.clone())]);
        let identity = serde_json::json!({"python":program,"sources":{"policy/experimental_proposal.py":module.sha256},
            "packages":{"synthetic":{"version":"1","files":1,"files_sha256":digest(&hashes)?}}});
        let identity_ref = put(root.join("identity.json"), &serde_json::to_vec(&identity)?)?;
        let mut runtime = PythonRuntime {
            manifest: reference(),
            program,
            source_root: source,
            site_packages: packages,
            source_files: vec![module.clone()],
            package_files: vec![package.clone()],
            runtime_sha256: digest(&identity)?,
            environment: BTreeMap::new(),
        };
        let manifest = RuntimeManifest {
            schema: "jope.experimental-python-runtime/v1".into(),
            runtime: runtime.without_manifest(),
            identity: identity_ref,
            packages: BTreeMap::from([(
                "synthetic".into(),
                DistributionInventory {
                    version: "1".into(),
                    files: BTreeMap::from([("synthetic.py".into(), package.clone())]),
                },
            )]),
        };
        runtime.manifest = put(root.join("runtime.json"), &serde_json::to_vec(&manifest)?)?;
        runtime.validate()?;
        fs::write(&module.path, b"# changed source")?;
        assert!(runtime.validate().is_err());
        fs::write(&module.path, b"# source fixture")?;
        runtime.validate()?;
        fs::write(&package.path, b"# changed package")?;
        assert!(runtime.validate().is_err());
        fs::write(&package.path, b"# package fixture")?;
        runtime
            .environment
            .insert("PYTHONPATH".into(), "/untrusted".into());
        assert!(runtime.validate().is_err());
        Ok(())
    }

    #[test]
    fn physical_python_argv_ignores_ambient_import_paths_and_pins_request() -> Result<()> {
        let plan = plan();
        let request = reference();
        let args = plan
            .runtime
            .argv(&request, Path::new("/private/output.json"))?;
        assert_eq!(&args[..3], &["-I", "-B", "-c"]);
        assert_eq!(args[3], BOOTSTRAP);
        assert_eq!(
            serde_json::from_str::<Vec<PathBuf>>(&args[4])?,
            vec![plan.runtime.source_root, plan.runtime.site_packages]
        );
        assert_eq!(args[7], "--request-sha256");
        assert_eq!(args[8], request.sha256);
        Ok(())
    }
}
