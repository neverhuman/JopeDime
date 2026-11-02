//! Bounded optimize receipt -> measured teaching adapter. Planning freezes the
//! exact inner partition before the original Experiment can be admitted. This
//! module never launches workers, creates a second window or upgrades old fits.
use super::{
    campaign::{CampaignPlan, Runtime},
    data::DevelopmentData,
    optimize::{Configuration, DatasetSearch},
    optimize_fit::{self, FitAdapter},
};
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_learning::{
    canon::{canonical_json, canonical_sha256},
    eligibility::{
        self, ContextEvidence, ContextPlan, ContextRun, FoldPredictions, FreezePlan, FreezeReceipt,
        InnerEvidence, InnerPartition, OofEvidence, ProducerPlan,
    },
    encoder_receipt::{Encoder, Product},
    hwm,
    metric::Metric,
    record::{
        ComparisonContext, Cost, Evaluator, Genome, HwmIdentity, HwmRecord, Provenance, Score,
        HWM_SCHEMA,
    },
    state::State,
};
use jope_pipeline::{
    cpu_context_bridge::CpuContextFiles,
    evaluation_result::{Consumption, EvaluationResult, RESULT_SCHEMA},
    execution::{Artifact, WorkerOutcome},
    experiment::EvaluationStatus,
    neural_search::{FrozenRecipe, SearchReceipt},
};
use jope_policy::context_producer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
const RUNTIME: &str = "asm-teaching-registry";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    pub state_root: PathBuf,
    pub prepared_root: PathBuf,
    pub panel: Artifact,
    pub original_label_role: String,
}
impl Settings {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.state_root.is_absolute()
                && self.prepared_root.is_absolute()
                && !self.state_root.starts_with("/tmp"),
            "teaching requires durable absolute roots"
        );
        ensure!(
            !self.state_root.ancestors().any(|p| p.join(".git").exists()),
            "teaching state must remain outside checkouts"
        );
        ensure!(
            eligibility::effective_label_role(&self.original_label_role, true)? == "development",
            "inner optimization labels must be development data"
        );
        ensure!(
            self.panel.sha256 == super::FROZEN_PANEL_SHA256,
            "teaching panel is not the approved frozen manifest"
        );
        Ok(())
    }
    fn state(&self) -> Result<State> {
        self.validate()?;
        let existing = self
            .state_root
            .ancestors()
            .find(|p| p.exists())
            .context("state ancestor absent")?;
        ensure!(
            existing.canonicalize()? == existing,
            "state root crosses a symlink"
        );
        fs::create_dir_all(&self.state_root)?;
        ensure!(
            self.state_root.canonicalize()? == self.state_root,
            "nonphysical teaching state"
        );
        Ok(State::at(&self.state_root))
    }
}
#[allow(clippy::large_enum_variant)] // At most the fixed thirty registry entries.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum Registration {
    Frozen {
        comparison: HwmIdentity,
        partition: InnerPartition,
        freeze: FreezeReceipt,
        /// W3-T02: the teaching context the freeze preregistered. Absent on registries written
        /// before producer preregistration, whose freezes preregistered no context.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<TeachingContext>,
    },
    Unavailable {
        reason: String,
    },
}
/// W3-T02: whether a dataset's teaching freeze preregistered a W2 context producer, recorded
/// explicitly either way.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", deny_unknown_fields)]
enum TeachingContext {
    /// The freeze preregistered the W2 CPU context producer that passed this W2 qualification.
    /// A teacher needs that producer's `teacher_admission` handoff of the dataset's context.
    Producer { qualification_sha256: String },
    /// No context is used on this dataset, for `reason`. The freeze preregistered no context
    /// encoder, so teacher admission refuses its teachers (`MissingEncoderProvenance`).
    NoContext { reason: String },
}
/// The source family of a W2 teaching context: the W2 CPU context schema. Its encoder is the
/// producer executable under its asset configuration (the CPU producer takes no seed, so 0).
const CONTEXT_SOURCE_FAMILY: &str = context_producer::SCHEMA;
/// The W2 producer every regression teaching freeze of `config` preregisters, read by the W2
/// qualification's own loader from its authenticated bytes, or why there is none.
fn producer_plan(config: &Configuration) -> Result<ProducerPlan> {
    let qualification = config
        .qualification
        .as_ref()
        .context("no completed W2 CPU context qualification is configured")?;
    if let Some(reason) = &config.encoder.unavailable {
        bail!("the W2 context producer runtime is unavailable: {reason}");
    }
    let registration =
        qualification.producer_registration(&config.source_sha40, &config.encoder.program)?;
    // Field for field: the W2 registration is the preregistration teacher admission parses.
    Ok(serde_json::from_value(serde_json::to_value(
        &registration,
    )?)?)
}
/// The teaching context a dataset's freeze preregisters, and the record of that choice. A
/// context is used exactly when the dataset is a regression dataset (the accepted W2 CPU context
/// is regression-only) and the configuration's W2 producer registered; otherwise the freeze
/// preregisters none and the reason is recorded.
fn teaching_context(
    config: &Configuration,
    task: Task,
    producer: &Result<ProducerPlan>,
) -> (Option<ContextPlan>, TeachingContext) {
    let producer = match (task, producer) {
        (Task::Regression, Ok(producer)) => producer,
        (Task::Regression, Err(error)) => {
            return (
                None,
                TeachingContext::NoContext {
                    reason: format!("{error:#}"),
                },
            )
        }
        (task, _) => {
            return (
                None,
                TeachingContext::NoContext {
                    reason: format!(
                        "the accepted W2 CPU context is regression-only; a {} dataset uses no \
                         teaching context",
                        task.as_str()
                    ),
                },
            )
        }
    };
    (
        Some(ContextPlan {
            source_family: CONTEXT_SOURCE_FAMILY.into(),
            encoder: Encoder {
                artifact_sha256: config.encoder.program.sha256.clone(),
                configuration_sha256: config.encoder_assets.configuration.sha256.clone(),
                seed: 0,
                product: Product::TeachingContext2048,
            },
            producer: Some(producer.clone()),
        }),
        TeachingContext::Producer {
            qualification_sha256: producer.qualification_sha256.clone(),
        },
    )
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Registry {
    schema: String,
    panel_sha256: String,
    entries: BTreeMap<String, Registration>,
}
fn registry(plan: &CampaignPlan) -> Result<Registry> {
    let value = plan
        .runtimes
        .get(RUNTIME)
        .context("no pre-admission teaching registry")?;
    ensure!(value.inputs.len() == 1, "teaching registry closure changed");
    value.inputs[0].verify()?;
    let registry: Registry = serde_json::from_slice(&fs::read(&value.inputs[0].path)?)?;
    ensure!(
        registry.schema == "jope.inner-teaching-registry/v1"
            && registry.panel_sha256 == plan.experiment.panel_sha256,
        "teaching registry changed"
    );
    Ok(registry)
}
/// Frozen high-water-mark comparison for one dataset. Configuration contributes
/// only its vocabulary pin; proposal knobs and `search_round` cannot change it.
pub(crate) fn comparison_identity(
    experiment: &jope_pipeline::experiment::Experiment,
    config: &Configuration,
    dataset: &jope_pipeline::experiment::PanelDataset,
    cut_id: String,
) -> HwmIdentity {
    HwmIdentity {
        task: dataset.task.as_str().into(),
        dataset_hash: dataset.source.sha256.clone(),
        cut_id,
        evaluator_sha256: jope_pipeline::sha256_hex(super::EVALUATOR.as_bytes()),
        vocabulary_sha256: config
            .experimental
            .get(dataset.task.as_str())
            .map_or(&config.vocabulary, |c| &c.vocabulary)
            .sha256
            .clone(),
        protocol: experiment.protocol.clone(),
        comparison: Some(ComparisonContext {
            numerical_policy: experiment.numerical_policy.clone(),
            stage: "train_oof".into(),
            aggregation: "mean_fold_loss".into(),
            label_role: "development".into(),
            sealed: false,
            benchmark_valid: false,
            recipe_encoding: "native-canonical-dsl-sha256/v1".into(),
            class_order: dataset.classes.clone(),
            checkpoint_sha256: None,
        }),
    }
}
/// Called only by clock-free plan construction, before final Experiment identity.
/// Unavailable baseline/fold eligibility remains an explicit dataset inventory.
pub fn preregister(plan: &mut CampaignPlan, config: &Configuration) -> Result<()> {
    let Some(settings) = &config.teaching else {
        return Ok(());
    };
    let state = settings.state()?;
    settings.panel.verify()?;
    let bytes = fs::read(&settings.panel.path)?;
    let panel: jope_pipeline::experiment::Panel = serde_json::from_slice(&bytes)?;
    ensure!(
        canonical_sha256(&panel)? == canonical_sha256(&plan.panel)?
            && state.put_blob(&bytes)? == plan.experiment.panel_sha256,
        "teaching panel identity changed"
    );
    let runtime = plan
        .runtimes
        .get("rust")
        .context("native teaching runtime absent")?
        .clone();
    let runtime_sha = state.put_blob(&canonical_json(&runtime)?)?;
    let selected = if plan.scope == super::origin_proof::SCOPE {
        Some(super::origin_proof::selected(plan)?)
    } else {
        None
    };
    // W3-T02: registered once, from the configuration's W2 qualification.
    let producer = producer_plan(config);
    let mut entries = BTreeMap::new();
    for dataset in &plan.panel.datasets {
        if selected.as_ref().is_some_and(|id| id != &dataset.id) {
            continue;
        }
        let (context_plan, context) = teaching_context(config, dataset.task, &producer);
        let result = (|| -> Result<Registration> {
            super::data::load_development(&settings.prepared_root, dataset)?;
            let bytes = fs::read(
                settings
                    .prepared_root
                    .join("development")
                    .join(format!("{}.json", dataset.id)),
            )?;
            ensure!(
                state.put_blob(&bytes)? == dataset.split.development_sha256,
                "prepared bytes changed during retention"
            );
            let partition = eligibility::derive_inner_partition(
                &state,
                &settings.panel.sha256,
                &dataset.id,
                0,
                plan.experiment.seed,
                &runtime.program.sha256,
                &runtime_sha,
            )?;
            let comparison =
                comparison_identity(&plan.experiment, config, dataset, partition.identity()?);
            let freeze = eligibility::freeze_inner_optimization(
                &state,
                &FreezePlan {
                    comparison: comparison.clone(),
                    original_label_role: settings.original_label_role.clone(),
                    panel_sha256: settings.panel.sha256.clone(),
                    dataset_id: dataset.id.clone(),
                    evaluation_seed: plan.experiment.seed,
                    // W3-T02: the W2 producer when a context is used; otherwise none, and
                    // teacher admission refuses as MissingEncoderProvenance.
                    context: context_plan.clone(),
                },
                &partition,
            )?;
            Ok(Registration::Frozen {
                comparison,
                partition,
                freeze,
                context: Some(context.clone()),
            })
        })();
        entries.insert(
            dataset.id.clone(),
            match result {
                Ok(v) => v,
                Err(e) => Registration::Unavailable {
                    reason: format!("{e:#}"),
                },
            },
        );
    }
    for entry in entries.values() {
        if let Registration::Frozen {
            comparison, freeze, ..
        } = entry
        {
            plan.experiment.checkpoints.insert(
                format!("teaching_freeze/{}", freeze.comparison_sha256),
                freeze.freeze_sha256.clone(),
            );
            plan.experiment.checkpoints.insert(
                format!("teaching_vocabulary/{}", freeze.comparison_sha256),
                comparison.vocabulary_sha256.clone(),
            );
        }
    }
    let value = Registry {
        schema: "jope.inner-teaching-registry/v1".into(),
        panel_sha256: plan.experiment.panel_sha256.clone(),
        entries,
    };
    let sha = state.put_blob(&canonical_json(&value)?)?;
    plan.runtimes.insert(
        RUNTIME.into(),
        Runtime {
            program: runtime.program,
            argument_prefix: vec![],
            inputs: vec![Artifact {
                path: state.blob_path(&sha),
                sha256: sha,
            }],
            package_files: vec![],
            environment: Default::default(),
            version: "jope.inner-teaching-registry/v1".into(),
            unavailable: None,
        },
    );
    plan.experiment.methods_sha256 = super::campaign::digest(&(&plan.methods, &plan.runtimes))?;
    Ok(())
}
mod admission;
mod admission_report;

use admission::{handed_off_context, Admission, Report};
use admission_report::terminal_report;
pub use admission_report::{admit_dataset, verify_admission};

#[cfg(test)]
mod tests;
