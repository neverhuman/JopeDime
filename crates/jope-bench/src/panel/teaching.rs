//! Bounded optimize receipt -> measured teaching adapter. Planning freezes the
//! exact inner partition before the original Experiment can be admitted. This
//! module never launches workers, creates a second window or upgrades old fits.
use super::{
    campaign::{CampaignPlan, Runtime},
    data::DevelopmentData,
    optimize::{Configuration, DatasetSearch},
    optimize_fit::{self, FitAdapter},
};
use anyhow::{ensure, Context, Result};
use jope_learning::{
    canon::{canonical_json, canonical_sha256},
    eligibility::{
        self, FoldPredictions, FreezePlan, FreezeReceipt, InnerEvidence, InnerPartition,
        OofEvidence,
    },
    hwm,
    metric::Metric,
    record::{
        ComparisonContext, Cost, Evaluator, Genome, HwmIdentity, HwmRecord, Provenance, Score,
        HWM_SCHEMA,
    },
    state::State,
};
use jope_pipeline::{
    execution::{Artifact, WorkerOutcome},
    experiment::EvaluationStatus,
    experiment_v2::{Consumption, EvaluationResult, RESULT_SCHEMA},
    neural_search::{FrozenRecipe, SearchReceipt},
};
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
    },
    Unavailable {
        reason: String,
    },
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
    let mut entries = BTreeMap::new();
    for dataset in &plan.panel.datasets {
        if selected.as_ref().is_some_and(|id| id != &dataset.id) {
            continue;
        }
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
                    // No preregistered context encoder exists on this route yet, so
                    // its admissions are refused as MissingEncoderProvenance.
                    context: None,
                },
                &partition,
            )?;
            Ok(Registration::Frozen {
                comparison,
                partition,
                freeze,
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
fn retain(state: &State, path: &Path) -> Result<String> {
    state.put_blob(&fs::read(path)?)
}
#[derive(Serialize, Deserialize)]
struct Admission {
    method: String,
    candidate_sha256: String,
    record: Option<HwmRecord>,
    qualified: bool,
    reason: Option<String>,
    qualification: Option<eligibility::Qualification>,
}
#[derive(Serialize, Deserialize)]
struct Report {
    schema: String,
    experiment_sha256: String,
    dataset: String,
    registry_sha256: String,
    original_label_role: String,
    effective_label_role: String,
    baseline_worker_fits: u64,
    entries: Vec<Admission>,
    registration_unavailable: Option<String>,
}
#[allow(clippy::too_many_arguments)] // One explicit controller-owned admission closure.
fn admit_one(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    state: &State,
    registration: &Registration,
    search: &DatasetSearch,
    method: &str,
    frozen: &FrozenRecipe,
    receipt: &SearchReceipt,
) -> Result<Admission> {
    let Registration::Frozen {
        comparison,
        partition,
        freeze,
    } = registration
    else {
        anyhow::bail!("teaching partition unavailable")
    };
    ensure!(
        plan.experiment
            .checkpoints
            .get(&format!("teaching_freeze/{}", freeze.comparison_sha256))
            == Some(&freeze.freeze_sha256),
        "original experiment lacks teaching freeze pin"
    );
    let partition_now = eligibility::derive_inner_partition(
        state,
        &plan.experiment.panel_sha256,
        &dataset.dataset.id,
        0,
        plan.experiment.seed,
        &plan.runtimes["rust"].program.sha256,
        &canonical_sha256(&plan.runtimes["rust"])?,
    )?;
    ensure!(
        partition_now.identity()? == partition.identity()?
            && comparison.vocabulary_sha256
                == config
                    .experimental
                    .get(dataset.dataset.task.as_str())
                    .map_or(&config.vocabulary, |c| &c.vocabulary)
                    .sha256,
        "teaching comparison changed"
    );
    frozen.candidate.recipe.validate()?;
    let receipt_sha = state.put_blob(&serde_json::to_vec(receipt)?)?;
    ensure!(
        receipt_sha == frozen.search_receipt_sha256,
        "frozen winner receipt differs"
    );
    let frozen_sha = state.put_blob(&serde_json::to_vec(frozen)?)?;
    let marker = eligibility::register_inner_candidate(state, freeze, &frozen_sha, &receipt_sha)?;
    let selection_sha = state.put_blob(&serde_json::to_vec(&search.pair.selection)?)?;
    let inner = InnerEvidence {
        selection_sha256: selection_sha,
        frozen_recipe_sha256: frozen_sha.clone(),
        search_receipt_sha256: receipt_sha,
        runtime_sha256: state.put_blob(&canonical_json(&plan.runtimes["rust"])?)?,
        arm_window_sha256: retain(
            state,
            &journal
                .join("arms")
                .join(format!("{}.json", receipt.arm.id_sha256)),
        )?,
        memory_policy_sha256: retain(state, &journal.join("fit-memory-policy.json"))?,
        candidate_freeze_sha256: marker,
    };
    let adapter = FitAdapter::prepare(
        dataset,
        &dataset.cut(0)?,
        search.pair.selection.clone(),
        plan.runtimes["rust"].clone(),
        journal,
        false,
    )?;
    let mut folds = Vec::new();
    let mut losses = Vec::new();
    let mut wall = Some(0f64);
    for request in search.pair.selection.selection_requests(
        &frozen.candidate.program_sha256,
        &frozen.candidate.id_sha256,
    )? {
        let mut job = adapter.job(&frozen.candidate, &request)?;
        job.arm = Some(receipt.arm.clone());
        let job_sha = job.identity()?;
        let worker_path = journal.join("finished").join(format!("{job_sha}.json"));
        let worker: WorkerOutcome = serde_json::from_slice(&fs::read(&worker_path)?)?;
        super::receipt_timing::completed_fit(journal, &job, &worker)?;
        let output =
            optimize_fit::checked_output(&job, &worker, &frozen.candidate.recipe, request.seed)?;
        let score = super::score::score_predictions(
            request.task,
            &request.classes,
            &adapter.folds[request.fold as usize].labels,
            &output.predictions,
        )?;
        losses.push(score.loss);
        wall = wall
            .zip(worker.consumption.wall_seconds)
            .map(|(a, b)| a + b);
        let predictions_sha = state.put_blob(&canonical_json(&output.predictions)?)?;
        let fitted_sha = state.put_blob(&output.artifact.bytes()?)?;
        ensure!(
            fitted_sha == output.artifact_sha256,
            "fitted model changed during retention"
        );
        let worker_sha = retain(state, &worker_path)?;
        let result = EvaluationResult {
            schema: RESULT_SCHEMA.into(),
            request_sha256: request.identity()?,
            status: EvaluationStatus::Completed,
            reason: None,
            primary: Some(score.value),
            secondary: BTreeMap::from([(
                "physical_fit_starts".into(),
                output.physical_fit_starts as f64,
            )]),
            scored_rows: request.scoring_rows,
            predictions_sha256: Some(predictions_sha.clone()),
            fitted_state_sha256: Some(fitted_sha.clone()),
            consumption: Consumption {
                wall_seconds: worker.consumption.wall_seconds,
                cpu_seconds: worker.consumption.cpu_seconds,
                gpu_seconds: None,
                peak_rss_bytes: worker.consumption.peak_rss_bytes,
                cgroup_peak_memory_bytes: worker.consumption.cgroup_peak_memory_bytes,
                sampled_peak_rss_bytes: worker.consumption.sampled_peak_rss_bytes,
                accounting_reason: worker.consumption.accounting_reason.clone(),
            },
            worker_receipt_sha256: worker_sha,
            reused_from_request_sha256: None,
        };
        result.validate(&request)?;
        let started_path = journal.join("started").join(format!("{job_sha}.json"));
        let started: Value = serde_json::from_slice(&fs::read(&started_path)?)?;
        let candidate = FoldPredictions {
            schema: "jope.train-oof-predictions/v1".into(),
            comparison_sha256: comparison.sha256()?,
            program_sha256: frozen.candidate.program_sha256.clone(),
            fold: request.fold,
            training_rows_sha256: request.fitting_rows_sha256.clone(),
            scoring_rows_sha256: request.scoring_rows_sha256.clone(),
            row_ids: search
                .pair
                .selection
                .inner_rows(request.fold as usize)?
                .1
                .to_vec(),
            predictions_sha256: predictions_sha,
            evaluation_request_sha256: state.put_blob(&serde_json::to_vec(&request)?)?,
            worker_job_sha256: state.put_blob(&serde_json::to_vec(&job)?)?,
            fitted_artifact_sha256: fitted_sha,
            execution_receipt_sha256: state.put_blob(&serde_json::to_vec(&result)?)?,
            started_receipt_sha256: retain(state, &started_path)?,
            fit_started_unix: started["started_unix_ms"]
                .as_u64()
                .context("actual fit start absent")?
                / 1000,
        };
        folds.push(state.put_blob(&canonical_json(&candidate)?)?);
    }
    ensure!(losses.len() == 5, "incomplete fivefold teaching candidate");
    let loss = losses.iter().sum::<f64>() / 5.;
    let metric = match dataset.dataset.task {
        jope_datasets::catalog::Task::Regression => Metric::Rmse,
        jope_datasets::catalog::Task::Binary => Metric::Auc,
        jope_datasets::catalog::Task::Multiclass => Metric::Logloss,
        _ => anyhow::bail!("unsupported teaching task"),
    };
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity: comparison.clone(),
        identity_sha256: comparison.sha256()?,
        legacy_identity_sha256: None,
        genome: Genome {
            dsl: frozen.candidate.recipe.executable_canonical.clone(),
            program_sha256: frozen.candidate.program_sha256.clone(),
            semantic_sha256: None,
        },
        score: Score {
            metric,
            value: if metric == Metric::Auc {
                1. - loss
            } else {
                loss
            },
            direction: metric.direction(),
            units: if metric == Metric::Rmse {
                "original_target_units"
            } else {
                "dimensionless"
            }
            .into(),
        },
        cost: Cost {
            runtime_seconds: wall,
            search_seconds: None,
        },
        evaluator: Evaluator {
            sha256: comparison.evaluator_sha256.clone(),
            version: super::EVALUATOR.into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: true,
        provenance: Provenance {
            source: "asm-inner-oof".into(),
            run_id: format!(
                "{}/{}/{}",
                plan.experiment.identity()?,
                receipt.arm.id_sha256,
                frozen.candidate.id_sha256
            ),
            git_sha: plan.experiment.source_sha.clone(),
            bundle_id: search.pair.selection.identity()?,
            record_sha256: frozen_sha.clone(),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: jope_learning::ledger::now_unix(),
    };
    record.validate()?;
    let evidence = OofEvidence {
        freeze: freeze.clone(),
        experiment_blob_sha256: state.put_blob(&serde_json::to_vec(&plan.experiment)?)?,
        window_sha256: retain(state, &journal.join("window.json"))?,
        fold_predictions_sha256: folds,
        inner: Some(inner),
        context: None,
    };
    let result = eligibility::qualify(state, &record, &evidence);
    let (qualified, reason, qualification) = match result {
        Ok(value) => {
            hwm::ingest(
                state,
                None,
                vec![hwm::Candidate {
                    record: record.clone(),
                    blob: fs::read(state.blob_path(&frozen_sha))?,
                    chain_from_source: false,
                }],
                json!({"kind":"asm-inner-oof","experiment_sha256":plan.experiment.identity()?}),
            )?;
            (true, None, Some(value))
        }
        Err(e) => (false, Some(format!("{e:#}")), None),
    };
    Ok(Admission {
        method: method.into(),
        candidate_sha256: frozen.candidate.id_sha256.clone(),
        record: Some(record),
        qualified,
        reason,
        qualification,
    })
}
/// Runs inside the existing controller before candidate outer-refit scheduling.
/// This consumes completed receipts only; no fit or deadline is started here.
pub fn admit_dataset(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
    journal: &Path,
) -> Result<Option<Artifact>> {
    let Some(settings) = &config.teaching else {
        return Ok(None);
    };
    let state = settings.state()?;
    let registry = registry(plan)?;
    let entry = registry
        .entries
        .get(&dataset.dataset.id)
        .context("teaching dataset was not preregistered")?;
    let registry_sha = plan.runtimes[RUNTIME].inputs[0].sha256.clone();
    let key = report_key(plan, dataset, search)?;
    let path = journal
        .join("teaching-admission")
        .join(format!("{key}.json"));
    if path.exists() {
        let file = optimize_fit::artifact(path)?;
        verify_admission(plan, config, dataset, search, journal, &file)?;
        return Ok(Some(file));
    }
    let mut entries = Vec::new();
    for (method, frozen, receipt) in [
        (
            super::optimize::NEURAL_METHOD,
            search.neural.as_ref(),
            search
                .receipt
                .neural
                .as_ref()
                .and_then(|n| n.search.as_ref()),
        ),
        (
            super::optimize::GENERIC_METHOD,
            search.generic.as_ref(),
            search.receipt.generic.as_ref(),
        ),
    ] {
        let (Some(frozen), Some(receipt)) = (frozen, receipt) else {
            continue;
        };
        let result = admit_one(
            plan, config, dataset, journal, &state, entry, search, method, frozen, receipt,
        );
        entries.push(match result {
            Ok(v) => v,
            Err(e) => Admission {
                method: method.into(),
                candidate_sha256: frozen.candidate.id_sha256.clone(),
                record: None,
                qualified: false,
                reason: Some(format!("{e:#}")),
                qualification: None,
            },
        });
    }
    let report = Report {
        schema: "jope.inner-teaching-admission/v1".into(),
        experiment_sha256: plan.experiment.identity()?,
        dataset: dataset.dataset.id.clone(),
        registry_sha256: registry_sha,
        original_label_role: settings.original_label_role.clone(),
        effective_label_role: "development".into(),
        baseline_worker_fits: 0,
        entries,
        registration_unavailable: match entry {
            Registration::Unavailable { reason } => Some(reason.clone()),
            _ => None,
        },
    };
    // Evidence/admission failures are inventory, never an immutable terminal
    // candidate decision. Revalidation may retry within the original experiment.
    let path = if terminal_report(&report) {
        path
    } else {
        journal
            .join("teaching-inventory")
            .join(format!("{key}-{}.json", canonical_sha256(&report)?))
    };
    Ok(Some(optimize_fit::immutable(&path, &report)?))
}
pub fn verify_admission(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
    journal: &Path,
    file: &Artifact,
) -> Result<()> {
    let settings = config
        .teaching
        .as_ref()
        .context("unexpected teaching admission without configuration")?;
    file.verify()?;
    let report: Report = serde_json::from_slice(&fs::read(&file.path)?)?;
    ensure!(
        report.schema == "jope.inner-teaching-admission/v1"
            && report.experiment_sha256 == plan.experiment.identity()?
            && report.dataset == dataset.dataset.id
            && report.registry_sha256
                == plan
                    .runtimes
                    .get(RUNTIME)
                    .context("teaching registry absent")?
                    .inputs[0]
                    .sha256
            && report.effective_label_role == "development"
            && report.original_label_role == settings.original_label_role
            && report.baseline_worker_fits == 0,
        "teaching report identity or label role changed"
    );
    let expected_path = if terminal_report(&report) {
        journal
            .join("teaching-admission")
            .join(format!("{}.json", report_key(plan, dataset, search)?))
    } else {
        journal.join("teaching-inventory").join(format!(
            "{}-{}.json",
            report_key(plan, dataset, search)?,
            canonical_sha256(&report)?
        ))
    };
    ensure!(
        file.path == expected_path,
        "teaching report is not the retained admission artifact"
    );
    let registered = registry(plan)?;
    let registration = registered
        .entries
        .get(&dataset.dataset.id)
        .context("teaching registration absent")?;
    ensure!(
        report.registration_unavailable.as_deref()
            == match registration {
                Registration::Unavailable { reason } => Some(reason.as_str()),
                _ => None,
            },
        "teaching registration failure changed"
    );
    let expected_methods = [
        (super::optimize::NEURAL_METHOD, search.neural.as_ref()),
        (super::optimize::GENERIC_METHOD, search.generic.as_ref()),
    ]
    .into_iter()
    .filter_map(|(method, candidate)| candidate.map(|_| method))
    .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        report.entries.len() == expected_methods.len()
            && report
                .entries
                .iter()
                .map(|e| e.method.as_str())
                .collect::<std::collections::BTreeSet<_>>()
                == expected_methods,
        "teaching report omits or duplicates a selected arm"
    );
    let state = State::at(&settings.state_root);
    for entry in report.entries {
        let candidate = match entry.method.as_str() {
            super::optimize::NEURAL_METHOD => search.neural.as_ref(),
            super::optimize::GENERIC_METHOD => search.generic.as_ref(),
            _ => None,
        }
        .context("teaching candidate absent")?;
        ensure!(
            candidate.candidate.id_sha256 == entry.candidate_sha256,
            "teaching candidate changed"
        );
        ensure!(
            entry.qualified == entry.qualification.is_some()
                && entry.qualified == entry.reason.is_none(),
            "inconsistent teaching decision"
        );
        if let Some(record) = &entry.record {
            let Registration::Frozen { comparison, .. } = registration else {
                anyhow::bail!("record without preregistered comparison")
            };
            ensure!(
                record.identity == *comparison
                    && record.genome.program_sha256 == candidate.candidate.recipe.executable_sha256
                    && record.genome.dsl == candidate.candidate.recipe.executable_canonical
                    && record.provenance.record_sha256
                        == jope_pipeline::sha256_hex(&serde_json::to_vec(candidate)?),
                "teaching record changed selected recipe/comparison"
            );
        }
        if entry.qualified {
            let record = entry.record.context("qualified record absent")?;
            let measured = eligibility::revalidate_qualification(&state, &record)?;
            ensure!(
                Some(measured) == entry.qualification,
                "teaching qualification changed"
            );
        }
    }
    Ok(())
}

fn report_key(
    plan: &CampaignPlan,
    dataset: &DevelopmentData,
    search: &DatasetSearch,
) -> Result<String> {
    canonical_sha256(&(
        plan.experiment.identity()?,
        &dataset.dataset.id,
        search.neural.as_ref().map(|r| &r.candidate.id_sha256),
        search.generic.as_ref().map(|r| &r.candidate.id_sha256),
    ))
}
fn terminal_report(report: &Report) -> bool {
    report.registration_unavailable.is_some() || report.entries.iter().all(|entry| entry.qualified)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn transient_admission_failure_is_inventory_not_terminal() {
        let mut report = Report {
            schema: "synthetic".into(),
            experiment_sha256: "a".repeat(64),
            dataset: "fixture".into(),
            registry_sha256: "b".repeat(64),
            original_label_role: "test".into(),
            effective_label_role: "development".into(),
            baseline_worker_fits: 0,
            registration_unavailable: None,
            entries: vec![Admission {
                method: "asm.generic".into(),
                candidate_sha256: "c".repeat(64),
                record: None,
                qualified: false,
                reason: Some("synthetic missing retained input".into()),
                qualification: None,
            }],
        };
        assert!(!terminal_report(&report));
        // Only the dependency inventory frozen before admission is permanent.
        report.registration_unavailable = Some("fewer than five training groups".into());
        assert!(terminal_report(&report));
    }
}
