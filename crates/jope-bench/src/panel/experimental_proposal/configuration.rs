use super::*;

pub const SCHEMA: &str = "jope.experimental-python-configuration/v1";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExperimentalConfiguration {
    pub schema: String,
    pub task: Task,
    pub runtime: PythonRuntime,
    pub export_manifest: Artifact,
    pub checkpoint: Artifact,
    pub vocabulary: Artifact,
    pub canonicalizer: Artifact,
    pub canonicalizer_build: Artifact,
    pub canonicalizer_features: Vec<String>,
    pub sampling: Sampling,
    pub generic_draws: usize,
}
impl ExperimentalConfiguration {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && matches!(
                    self.task,
                    Task::Regression | Task::Binary | Task::Multiclass
                ),
            "experimental configuration requires explicit supported task"
        );
        self.sampling.validate()?;
        ensure!(
            (80..=4096 - self.sampling.count).contains(&self.generic_draws),
            "experimental generic bank allowance outside bounds"
        );
        for artifact in self.dependencies() {
            ensure!(
                artifact.path.is_absolute() && jope_pipeline::experiment::sha256(&artifact.sha256),
                "invalid experimental dependency pin"
            );
        }
        Ok(())
    }
    pub fn verify(&self, source_sha40: &str, source_manifest: &Artifact) -> Result<()> {
        ensure!(
            self.schema == SCHEMA
                && matches!(
                    self.task,
                    Task::Regression | Task::Binary | Task::Multiclass
                ),
            "experimental configuration requires explicit supported task"
        );
        self.sampling.validate()?;
        ensure!(
            (80..=4096 - self.sampling.count).contains(&self.generic_draws),
            "experimental bank allowance outside bounds"
        );
        self.runtime.validate()?;
        for a in [
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
            &self.canonicalizer_build,
        ] {
            a.verify()?;
        }
        source_manifest.verify()?;
        let build: super::worker_build::BuildReceipt =
            experimental_producer::read(&self.canonicalizer_build)?;
        ensure!(
            build.schema == "jope.asm-cargo-build/v1"
                && build.exit_code == 0
                && build.source_sha40 == source_sha40
                && build.source_manifest_sha256 == source_manifest.sha256
                && build.started_unix_ms > 0
                && build.finished_unix_ms >= build.started_unix_ms
                && Path::new(&build.working_directory).is_absolute()
                && build.command.iter().any(|a| a == "build")
                && build
                    .command
                    .iter()
                    .any(|a| a.starts_with("--message-format=json")),
            "native canonicalizer lacks a matching successful source build"
        );
        build.cargo_messages.verify()?;
        let built = build
            .executables
            .iter()
            .filter(|e| e.target_name == "jope" && e.sha256 == self.canonicalizer.sha256)
            .collect::<Vec<_>>();
        ensure!(
            built.len() == 1,
            "canonicalizer differs from actual built CLI bytes"
        );
        let features = self
            .canonicalizer_features
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        ensure!(
            features.len() == self.canonicalizer_features.len(),
            "duplicate canonicalizer feature pin"
        );
        let mut matching = 0;
        let mut completed = false;
        for line in fs::read_to_string(&build.cargo_messages.path)?.lines() {
            let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if value["reason"] == "build-finished" && value["success"] == true {
                completed = true;
            }
            if value["reason"] == "compiler-artifact"
                && value["target"]["name"] == "jope"
                && value["executable"].as_str() == built[0].path.to_str()
            {
                ensure!(
                    serde_json::from_value::<BTreeSet<String>>(value["features"].clone())?
                        == features,
                    "canonicalizer build features changed"
                );
                matching += 1;
            }
        }
        ensure!(
            completed && matching == 1,
            "canonicalizer build capture incomplete"
        );
        Ok(())
    }
    pub fn dependencies(&self) -> Vec<&Artifact> {
        let mut a = self.runtime.dependencies();
        a.extend([
            &self.export_manifest,
            &self.checkpoint,
            &self.vocabulary,
            &self.canonicalizer,
            &self.canonicalizer_build,
        ]);
        a
    }
}

/// Untagged serialization preserves existing regression input bytes/hashes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContextInput {
    Accepted(TrainingInput),
    Experimental(ClassificationInput),
}
impl ContextInput {
    pub fn accepted(&self) -> Result<&TrainingInput> {
        match self {
            Self::Accepted(input) => Ok(input),
            Self::Experimental(_) => {
                anyhow::bail!("classification input cannot enter regression producer")
            }
        }
    }
    pub fn experimental(&self) -> Result<&ClassificationInput> {
        match self {
            Self::Experimental(input) => Ok(input),
            Self::Accepted(_) => anyhow::bail!("regression input cannot enter classifier producer"),
        }
    }
}
impl From<TrainingInput> for ContextInput {
    fn from(value: TrainingInput) -> Self {
        Self::Accepted(value)
    }
}

pub fn bind_pair<V: Variation>(
    configuration: &Configuration,
    experimental: &ExperimentalConfiguration,
    dataset: &DevelopmentData,
    input: TrainingInput,
    mut pair: PairInputPlan,
    variation: &mut V,
) -> Result<(ContextInput, PairInputPlan)> {
    ensure!(
        experimental.task == dataset.dataset.task,
        "experimental checkpoint task differs from dataset"
    );
    let (context_input, preprocessing_sha256) = if dataset.dataset.task == Task::Regression {
        let preprocessing = digest(&input.prepare()?.preprocessing)?;
        (ContextInput::Accepted(input), preprocessing)
    } else {
        let class_input = ClassificationInput {
            schema: "jope.cpu-classification-training/v1".into(),
            split_sha256: dataset.dataset.split.sha256.clone(),
            task: match dataset.dataset.task {
                Task::Binary => ClassificationTask::Binary,
                Task::Multiclass => ClassificationTask::Multiclass,
                _ => anyhow::bail!("supported tabular task required"),
            },
            classes: dataset.dataset.classes.clone(),
            identity: input.identity,
            features: input.features,
            rows: input.rows,
        };
        let preprocessing = class_input.preprocessing_sha256()?;
        (ContextInput::Experimental(class_input), preprocessing)
    };
    let bank = experimental_producer::generic_bank(
        dataset.dataset.task,
        pair.random_seed.wrapping_add(1),
        experimental.generic_draws,
        variation,
    )?;
    let request = experimental_producer::RequestPlan {
        schema: experimental_producer::PLAN_SCHEMA.into(),
        task: dataset.dataset.task,
        worker_source_sha256: configuration.source_manifest.sha256.clone(),
        runtime: experimental.runtime.clone(),
        export_manifest: experimental.export_manifest.clone(),
        checkpoint: experimental.checkpoint.clone(),
        vocabulary: experimental.vocabulary.clone(),
        canonicalizer: experimental.canonicalizer.clone(),
        canonicalizer_source_sha40: configuration.source_sha40.clone(),
        training: TrainingIdentity {
            prepared_sha256: pair.prepared_sha256.clone(),
            training_sha256: pair.training_sha256.clone(),
            selected_training_rows_sha256: pair.selection.outer().fitting_rows_sha256.clone(),
            preprocessing_sha256: preprocessing_sha256.clone(),
            cut: pair.cut,
        },
        data_schema: dataset.schema.clone(),
        neural_seed: pair.random_seed,
        sampling: experimental.sampling.clone(),
        generic_bank_sha256: digest(&bank)?,
        generic_seed: bank.seed,
        generic_draws: experimental.generic_draws,
        generic_policy: bank.policy,
    };
    request.validate()?;
    pair.preprocessing_sha256 = preprocessing_sha256;
    pair.vocabulary_sha256 = experimental.vocabulary.sha256.clone();
    pair.installation.policy_program_sha256 = experimental.runtime.program.sha256.clone();
    pair.installation.checkpoint_receipt_sha256 = experimental.export_manifest.sha256.clone();
    pair.proposal = ProposalPlan::Experimental(Box::new(request));
    pair.encoding.input_sha256 = digest(&context_input)?;
    if dataset.dataset.task != Task::Regression {
        pair.encoding.backend = "asm.experimental-classification-neutral-foundation-cpu/v1".into();
    }
    Ok((context_input, pair))
}
