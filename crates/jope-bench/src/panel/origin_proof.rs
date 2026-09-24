//! A separately admitted operational proof on one retained regression dataset.
//! It does not modify the thirty-dataset panel or claim comparator wins.
use super::{
    campaign::{self, CampaignPlan, Runtime},
    optimize::{Configuration, DatasetSearch},
};
use anyhow::{ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::{
    execution::{Artifact, ExecutionPool, WorkerOutcome},
    experiment::{EvaluationStage, EvaluationStatus, Resources},
    neural_search::{Origin, SearchReceipt},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{fs, path::Path, sync::atomic::AtomicBool};

pub const SCOPE: &str = "asm-origin-proof";
pub const PROTOCOL: &str = "jope.asm-origin-proof/v1";
const SELECTION_RUNTIME: &str = "asm-origin-proof-selection";
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Selection {
    schema: String,
    dataset: String,
    resources: Artifact,
}
pub(crate) fn selected(plan: &CampaignPlan) -> Result<String> {
    let runtime = plan
        .runtimes
        .get(SELECTION_RUNTIME)
        .context("proof selection absent")?;
    ensure!(
        runtime.inputs.len() == 1,
        "proof selection requires one pinned input"
    );
    runtime.inputs[0].verify()?;
    let value: Selection = serde_json::from_slice(&fs::read(&runtime.inputs[0].path)?)?;
    ensure!(value.schema == PROTOCOL, "unknown proof selection");
    value.resources.verify()?;
    let resources: Resources = serde_json::from_slice(&fs::read(&value.resources.path)?)?;
    ensure!(
        campaign::digest(&resources)? == campaign::digest(&plan.experiment.resources)?,
        "proof resource declaration changed"
    );
    ensure!(
        plan.panel
            .datasets
            .iter()
            .any(|d| d.id == value.dataset && d.task == Task::Regression),
        "proof requires one frozen regression dataset"
    );
    Ok(value.dataset)
}
#[allow(clippy::too_many_arguments)]
pub fn prepare(
    base: &Path,
    base_sha: &str,
    configuration: &Path,
    configuration_sha: &str,
    dataset: &str,
    resources: &Path,
    resources_sha: &str,
    output: &Path,
) -> Result<Value> {
    let mut plan: CampaignPlan = serde_json::from_slice(&fs::read(base)?)?;
    ensure!(
        plan.identity()? == base_sha && plan.scope == "comparator-qualification",
        "proof base plan differs or is already admitted ASM"
    );
    let configuration = Artifact {
        path: fs::canonicalize(configuration)?,
        sha256: configuration_sha.into(),
    };
    configuration.verify()?;
    let config: Configuration = serde_json::from_slice(&fs::read(&configuration.path)?)?;
    config.validate()?;
    let resources = Artifact {
        path: fs::canonicalize(resources)?,
        sha256: resources_sha.into(),
    };
    resources.verify()?;
    let limits: Resources = serde_json::from_slice(&fs::read(&resources.path)?)?;
    limits.validate()?;
    ensure!(
        limits.per_fit_seconds <= 30,
        "proof fit limit exceeds thirty seconds"
    );
    ensure!(
        plan.panel
            .datasets
            .iter()
            .any(|d| d.id == dataset && d.task == Task::Regression),
        "proof dataset is not frozen regression"
    );
    let selection = Selection {
        schema: PROTOCOL.into(),
        dataset: dataset.into(),
        resources,
    };
    let selection_path = output.with_extension("selection.json");
    super::worker::write_private(&selection_path, &selection)?;
    let selection = super::optimize_fit::artifact(fs::canonicalize(selection_path)?)?;
    let program = plan
        .runtimes
        .get("rust")
        .context("native runtime absent")?
        .program
        .clone();
    for (name, input, version) in [
        (
            super::optimize::CONFIG_RUNTIME,
            configuration,
            super::optimize::SCHEMA,
        ),
        (SELECTION_RUNTIME, selection, PROTOCOL),
    ] {
        plan.runtimes.insert(
            name.into(),
            Runtime {
                program: program.clone(),
                argument_prefix: vec![],
                inputs: vec![input],
                package_files: vec![],
                environment: Default::default(),
                version: version.into(),
                unavailable: None,
            },
        );
    }
    plan.experiment.resources = limits;
    plan.experiment.methods_sha256 = campaign::digest(&(&plan.methods, &plan.runtimes))?;
    plan.experiment.protocol = PROTOCOL.into();
    plan.experiment.checkpoints.insert(
        "asm.accepted-policy-state".into(),
        jope_policy::ACCEPTED_STATE_SHA256.into(),
    );
    for (task, experimental) in &config.experimental {
        for (kind, sha) in [
            ("policy", &experimental.checkpoint.sha256),
            ("export", &experimental.export_manifest.sha256),
            ("runtime", &experimental.runtime.runtime_sha256),
        ] {
            plan.experiment
                .checkpoints
                .insert(format!("asm.experimental-{kind}/{task}"), sha.clone());
        }
    }
    plan.scope = SCOPE.into();
    super::teaching::preregister(&mut plan, &config)?;
    let identity = plan.identity()?;
    super::worker::write_private(output, &plan)?;
    Ok(
        json!({"ok":true,"plan":output,"plan_sha256":identity,"scope":SCOPE,"dataset":dataset,"resources":plan.experiment.resources,"neural_slots":64,"exploration_slots":16,"generic_slots":80,"seconds_per_arm":config.seconds,"maximum_evaluations_per_arm":config.max_evaluations,"fits":0,"execution_budget_started":false,"complete_panel_demonstration":false,"dependencies":config.dependency_inventory()}),
    )
}
fn input(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
) -> Result<super::data::DevelopmentData> {
    plan.validate()?;
    ensure!(plan.scope == SCOPE, "not an origin proof plan");
    let (panel, sha) = super::cli::read_panel(manifest)?;
    ensure!(
        sha == plan.experiment.panel_sha256
            && campaign::digest(&panel)? == campaign::digest(&plan.panel)?,
        "proof manifest changed"
    );
    let id = selected(plan)?;
    super::data::load_development(
        datasets,
        plan.panel
            .datasets
            .iter()
            .find(|d| d.id == id)
            .context("selected dataset absent")?,
    )
}
pub fn run(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
) -> Result<Value> {
    plan.validate()?;
    pool.reserve()?;
    pool.admit()?;
    campaign::immutable(&journal.join("campaign.json"), plan)?;
    let dataset = input(plan, manifest, datasets)?;
    let runtime = &plan.runtimes["rust"];
    runtime.program.verify()?;
    for a in runtime.inputs.iter().chain(&runtime.package_files) {
        a.verify()?;
    }
    let attached = super::asm_campaign::attach(
        plan,
        std::slice::from_ref(&dataset),
        journal,
        pool,
        cancel,
        &[],
    )?;
    if !attached.ready {
        return Ok(
            json!({"schema":PROTOCOL,"ok":false,"state":"pending_original_window","dataset":dataset.dataset.id,"outcomes":attached.inventory,"window":pool.window(),"complete_panel_demonstration":false}),
        );
    }
    report(plan, manifest, datasets, journal)
}
fn counts(receipt: Option<&SearchReceipt>) -> Value {
    let Some(receipt) = receipt else {
        return json!({"available":false,"physical_fit_starts":null});
    };
    let origins = [
        Origin::NeuralPolicy,
        Origin::NeuralRejectionFallback,
        Origin::RandomExploration,
        Origin::Generic,
        Origin::Mutation,
        Origin::Crossover,
    ];
    let mut values = serde_json::Map::new();
    for origin in origins {
        let observations: Vec<_> = receipt
            .observations
            .iter()
            .filter(|o| o.candidate.origin == origin)
            .collect();
        let complete = observations
            .iter()
            .filter(|o| o.feedback.status == EvaluationStatus::Completed)
            .count();
        let starts = observations
            .iter()
            .try_fold(0u64, |a, o| a.checked_add(o.feedback.physical_fit_starts?));
        values.insert(serde_json::to_value(&origin).unwrap().as_str().unwrap().to_owned(),json!({"candidate_slots":observations.len(),"completed_five_fold_candidates":complete,"physical_fit_starts":starts,"known_physical_fit_starts":observations.iter().filter_map(|o|o.feedback.physical_fit_starts).sum::<u64>(),"unknown_start_counters":observations.iter().filter(|o|o.feedback.physical_fit_starts.is_none()).count()}));
    }
    let unique = |initial_only: bool, completed_only: bool| {
        receipt
            .observations
            .iter()
            .filter(|o| {
                (!initial_only || o.candidate.generation == 0)
                    && (!completed_only || o.feedback.status == EvaluationStatus::Completed)
            })
            .map(|o| &o.candidate.program_sha256)
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    json!({"available":true,"origins":values,"candidate_slots":receipt.observations.len(),"generation_zero_slots":receipt.observations.iter().filter(|o|o.candidate.generation==0).count(),"unique_canonical_recipes":unique(false,false),"generation_zero_unique_canonical_recipes":unique(true,false),"completed_five_fold_unique_recipes":unique(false,true),"generation_zero_completed_five_fold_unique_recipes":unique(true,true),"selection_loss_aggregation":"unweighted_mean_of_five_inner_fold_losses","stop_reason":receipt.stop_reason,"deadline_unix_ms":receipt.deadline_unix_ms})
}
pub fn report(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
) -> Result<Value> {
    let dataset = input(plan, manifest, datasets)?;
    let retained: CampaignPlan = serde_json::from_slice(&fs::read(journal.join("campaign.json"))?)?;
    ensure!(
        retained.identity()? == plan.identity()?,
        "proof journal belongs to another plan"
    );
    let window: jope_pipeline::experiment::ExecutionWindow =
        serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
    window.validate(&plan.experiment)?;
    let attached = super::asm_campaign::recover(plan, std::slice::from_ref(&dataset), journal)?;
    let value = &attached.inventory[&dataset.dataset.id]["search"];
    let search = if value.is_null() {
        None
    } else {
        Some(serde_json::from_value::<DatasetSearch>(value.clone())?)
    };
    let neural = search
        .as_ref()
        .and_then(|s| s.receipt.neural.as_ref())
        .and_then(|n| n.search.as_ref());
    let generic = search.as_ref().and_then(|s| s.receipt.generic.as_ref());
    let mut predictions = vec![];
    for method in attached
        .methods
        .get(&dataset.dataset.id)
        .into_iter()
        .flatten()
    {
        let cut = dataset.cut(0)?;
        let request = campaign::request(
            &dataset,
            &cut,
            method,
            &plan.experiment,
            EvaluationStage::Screen,
        )?;
        if !journal
            .join("inputs")
            .join(format!("{}.json", request.identity()?))
            .exists()
        {
            predictions.push(json!({"method":method.id(),"status":"unknown_no_refit_input"}));
            continue;
        }
        let fit = campaign::prepare_fit_impl(
            plan,
            &dataset,
            &cut,
            method,
            EvaluationStage::Screen,
            journal,
            false,
        )?;
        let path = journal
            .join("finished")
            .join(format!("{}.json", fit.job.identity()?));
        if !path.exists() {
            predictions.push(json!({"method":method.id(),"status":"unknown_no_finished_refit"}));
            continue;
        }
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(path)?)?;
        if outcome.command_status != EvaluationStatus::Completed {
            predictions.push(json!({"method":method.id(),"status":outcome.command_status,"reason":outcome.reason}));
            continue;
        }
        let (values, model) = campaign::checked_predictions(&fit, &outcome)?;
        let score = super::score::score_predictions(
            dataset.dataset.task,
            &dataset.dataset.classes,
            &cut.hold.labels,
            &values,
        )?;
        predictions.push(json!({"method":method.id(),"status":"completed","fitted_artifact_sha256":model,"development_hold_score":score,"predictions":values,"output_artifacts":outcome.outputs}));
    }
    let genuine = neural.is_some_and(|r| {
        r.observations.iter().any(|o| {
            o.candidate.origin == Origin::NeuralPolicy
                && o.feedback.status == EvaluationStatus::Completed
        })
    });
    let evolved = neural.is_some_and(SearchReceipt::has_fitted_neural_descendant);
    Ok(
        json!({"schema":PROTOCOL,"ok":true,"dataset":dataset.dataset.id,"task":dataset.dataset.task,"window":window,"outcomes":attached.inventory,"neural":counts(neural),"generic":counts(generic),"genuine_neural_origin_fit_proved":genuine,"neural_offspring_fit_proved":evolved,"generation_zero_and_evolution_proved":genuine&&evolved,"refitted_predictions":predictions,"cost":super::asm_campaign::accounting(plan,journal)?,"final_test_targets_used":false,"optimization_visible_labels":"development","test_headline_eligible":false,"complete_panel_demonstration":false,"comparison_wins_claimed":false}),
    )
}
