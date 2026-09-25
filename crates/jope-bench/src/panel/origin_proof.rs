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
    neural_search::{self, IncumbentSeed, Origin, SearchReceipt},
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
    record_checkpoints(&mut plan, &config);
    plan.scope = SCOPE.into();
    super::teaching::preregister(&mut plan, &config)?;
    // A bad incumbent pin is refused here, before any execution window is burned.
    if config.incumbents.is_some() {
        let settings = config
            .teaching
            .as_ref()
            .context("incumbent seeding requires a teaching state")?;
        let descriptor = plan
            .panel
            .datasets
            .iter()
            .find(|d| d.id == dataset)
            .context("proof dataset absent")?;
        let data = super::data::load_development(&settings.prepared_root, descriptor)?;
        // Read-only; verifies every pin against the exact pair selection, then applies
        // the pair checks `begin_pair` runs after admission (per-arm recipe uniqueness).
        let (_, pair) = super::optimize::build_pair_input(&plan, &config, &data)?;
        pair.arm(true)?;
    }
    // A challenger receipt that does not bind its pins is refused before any window is admitted.
    if let Some(challenger) = &config.native_challenger {
        super::optimize::challenger_checkpoint(challenger)?;
    }
    let identity = plan.identity()?;
    super::worker::write_private(output, &plan)?;
    let (kn, kg) = config
        .incumbents
        .as_ref()
        .map_or((0, 0), |i| (i.neural.len(), i.generic.len()));
    Ok(
        json!({"ok":true,"plan":output,"plan_sha256":identity,"scope":SCOPE,"dataset":dataset,"resources":plan.experiment.resources,"neural_slots":64,"exploration_slots":16 - kn,"generic_slots":80 - kg,"incumbent_slots":{"neural":kn,"generic":kg},"search_round":config.search_round,"seconds_per_arm":config.seconds,"maximum_evaluations_per_arm":config.max_evaluations,"fits":0,"execution_budget_started":false,"complete_panel_demonstration":false,"dependencies":config.dependency_inventory()}),
    )
}
/// Checkpoint identities an origin proof preregisters. A native challenger adds its state;
/// the accepted state stays, because the configuration still pins and verifies it.
fn record_checkpoints(plan: &mut CampaignPlan, config: &Configuration) {
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
    if let Some(challenger) = &config.native_challenger {
        plan.experiment.checkpoints.insert(
            "asm.native-challenger-state".into(),
            challenger.identity.state_sha256.clone(),
        );
    }
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
    run_impl(plan, manifest, datasets, journal, pool, cancel, false)
}
/// Runs and retains every fit, but leaves development hold scoring to `report`.
pub fn run_deferred(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
) -> Result<Value> {
    run_impl(plan, manifest, datasets, journal, pool, cancel, true)
}
fn run_impl(
    plan: &CampaignPlan,
    manifest: &Path,
    datasets: &Path,
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
    defer: bool,
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
    let attached = super::asm_campaign::attach_with(
        plan,
        std::slice::from_ref(&dataset),
        journal,
        pool,
        cancel,
        &[],
        !defer,
    )?;
    if !attached.ready {
        return Ok(
            json!({"schema":PROTOCOL,"ok":false,"state":"pending_original_window","dataset":dataset.dataset.id,"outcomes":attached.inventory,"window":pool.window(),"complete_panel_demonstration":false}),
        );
    }
    if defer {
        return Ok(deferred(&dataset.dataset.id, json!(pool.window())));
    }
    report(plan, manifest, datasets, journal)
}
/// Deferred completion: no hold score, no refit predictions, no trivial baseline.
fn deferred(dataset: &str, window: Value) -> Value {
    json!({"schema":PROTOCOL,"ok":true,"state":"complete_report_deferred","dataset":dataset,"window":window,"complete_panel_demonstration":false})
}
fn counts(receipt: Option<&SearchReceipt>, seeds: &[IncumbentSeed]) -> Value {
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
        Origin::Incumbent,
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
    // Incumbents are re-measured earlier winners, reported apart from discoveries.
    let first_incumbent = neural_search::INITIAL_SIZE.saturating_sub(seeds.len());
    let incumbents = seeds
        .iter()
        .enumerate()
        .map(|(offset, seed)| {
            let slot = first_incumbent + offset;
            let feedback = receipt.observations.get(slot).map(|o| &o.feedback);
            json!({"slot":slot,"frozen_recipe_sha256":seed.frozen_recipe_sha256,"program_sha256":seed.program_sha256,"status":feedback.map(|f| f.status),"train_side_loss":feedback.and_then(|f| f.loss),"reason":feedback.and_then(|f| f.reason.as_deref())})
        })
        .collect::<Vec<_>>();
    // Carried forward is decided by program, not origin: the policy may re-propose
    // an incumbent's program in an earlier slot.
    let winner = receipt.winner().map(|w| {
        json!({"candidate_sha256":w.candidate.id_sha256,"program_sha256":w.candidate.program_sha256,"origin":w.candidate.origin,"generation":w.candidate.generation,"neural_policy_ancestor":receipt.neural_policy_lineage(&w.candidate.id_sha256),"carried_forward":seeds.iter().any(|s| s.program_sha256 == w.candidate.program_sha256)})
    });
    json!({"available":true,"origins":values,"candidate_slots":receipt.observations.len(),"generation_zero_slots":receipt.observations.iter().filter(|o|o.candidate.generation==0).count(),"unique_canonical_recipes":unique(false,false),"generation_zero_unique_canonical_recipes":unique(true,false),"completed_five_fold_unique_recipes":unique(false,true),"generation_zero_completed_five_fold_unique_recipes":unique(true,true),"selection_loss_aggregation":"unweighted_mean_of_five_inner_fold_losses","stop_reason":receipt.stop_reason,"deadline_unix_ms":receipt.deadline_unix_ms,"incumbents":incumbents,"winner":winner})
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
    let no_seeds: &[IncumbentSeed] = &[];
    let neural_seeds = search
        .as_ref()
        .map_or(no_seeds, |s| s.pair.arm_incumbents(true));
    let generic_seeds = search
        .as_ref()
        .map_or(no_seeds, |s| s.pair.arm_incumbents(false));
    let methods = attached
        .methods
        .get(&dataset.dataset.id)
        .cloned()
        .unwrap_or_default();
    let predictions = refit_predictions(plan, &dataset, journal, &methods)?;
    let cut = dataset.cut(0)?;
    let trivial = trivial_hold_rmse(&cut.train.targets, &cut.hold.labels)?;
    let genuine = neural.is_some_and(|r| {
        r.observations.iter().any(|o| {
            o.candidate.origin == Origin::NeuralPolicy
                && o.feedback.status == EvaluationStatus::Completed
        })
    });
    let evolved = neural.is_some_and(SearchReceipt::has_fitted_neural_descendant);
    Ok(
        json!({"schema":PROTOCOL,"ok":true,"dataset":dataset.dataset.id,"task":dataset.dataset.task,"window":window,"outcomes":attached.inventory,"neural":counts(neural,neural_seeds),"generic":counts(generic,generic_seeds),"genuine_neural_origin_fit_proved":genuine,"neural_offspring_fit_proved":evolved,"generation_zero_and_evolution_proved":genuine&&evolved,"refitted_predictions":predictions,"trivial_hold_rmse":trivial,"trivial_hold_policy":"fitting_row_mean/sequential_sum_y_over_n/rmse_on_development_fold_0","cost":super::asm_campaign::accounting(plan,journal)?,"final_test_targets_used":false,"optimization_visible_labels":"development","test_headline_eligible":false,"complete_panel_demonstration":false,"comparison_wins_claimed":false}),
    )
}

/// Refit predictions scored once on the development hold rows. Reads retained
/// inputs and finished receipts only; nothing under `observations/` is consulted.
fn refit_predictions(
    plan: &CampaignPlan,
    dataset: &super::data::DevelopmentData,
    journal: &Path,
    methods: &[campaign::Method],
) -> Result<Vec<Value>> {
    let cut = dataset.cut(0)?;
    let mut predictions = vec![];
    for method in methods {
        let request = campaign::request(
            dataset,
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
            dataset,
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
    Ok(predictions)
}

/// Held RMSE of the fitting-row mean. The convention matches the teaching guard's
/// regression baseline (`jope-learning` `eligibility.rs:390-397`): the mean is the
/// sequential sum of y/n in development row order, scored by the shared scorer.
pub(crate) fn trivial_hold_rmse(fitting_targets: &[f64], held_labels: &[String]) -> Result<f64> {
    ensure!(
        !fitting_targets.is_empty() && !held_labels.is_empty(),
        "empty trivial baseline partition"
    );
    let n = fitting_targets.len() as f64;
    let mean = fitting_targets.iter().map(|y| y / n).sum::<f64>();
    ensure!(mean.is_finite(), "nonfinite trivial mean");
    Ok(super::score::score_predictions(
        Task::Regression,
        &[],
        held_labels,
        &super::score::Predictions::Values(vec![mean; held_labels.len()]),
    )?
    .value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::panel::{
        campaign::Method,
        data::{parse_development, DevelopmentData},
        optimize::{self, Incumbents, MAX_SEARCH_ROUND},
        optimize_fit::{self, FitAdapter},
        score, teaching, worker,
    };
    use jope_pipeline::{
        execution::{ObservedConsumption, EXECUTION_SCHEMA},
        experiment::{Experiment, PanelDataset, EXPERIMENT_SCHEMA},
        sha256_hex,
    };
    use std::collections::{BTreeMap, BTreeSet};

    fn pin(name: &str, sha256: &str) -> String {
        format!(r#"{{"path":"/fixture/{name}","sha256":"{sha256}"}}"#)
    }

    /// The accepted pre-DEMO-2 configuration, keys written in declaration order.
    fn accepted_configuration() -> Vec<u8> {
        let runtime = |name: &str, sha256: &str| {
            format!(
                r#"{{"program":{},"argument_prefix":[],"inputs":[],"package_files":[],"environment":{{}},"version":"fixture","unavailable":null}}"#,
                pin(name, sha256)
            )
        };
        let artifacts = [
            ("release_manifest", "1".repeat(64)),
            ("source_manifest", "2".repeat(64)),
            ("checkpoint_receipt", "3".repeat(64)),
            (
                "policy_weights",
                "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e".into(),
            ),
            ("vocabulary", "5".repeat(64)),
            (
                "frozen_transforms",
                jope_policy::context_producer::TRANSFORMS_SHA256.into(),
            ),
        ]
        .map(|(name, sha256)| format!(r#""{name}":{}"#, pin(name, &sha256)))
        .join(",");
        format!(
            r#"{{"schema":"{}","source_sha40":"{}",{artifacts},"policy":{},"encoder":{},"encoder_assets":{{"configuration":{},"files":[]}},"qualification":null,"seconds":120,"max_evaluations":96,"sampling_rounds":4,"sampling_batch":64,"sampling_temperature":1.0,"dataset_wave_width":1,"scheduling_policy":"{}","inner_fold_policy":"{}"}}"#,
            optimize::SCHEMA,
            "0".repeat(40),
            runtime("policy", &"6".repeat(64)),
            runtime("encoder", &"7".repeat(64)),
            pin("encoder_assets", &"8".repeat(64)),
            crate::panel::optimize_schedule::POLICY,
            optimize_fit::INNER_FOLD_POLICY,
        )
        .into_bytes()
    }

    fn configuration(round: Option<u32>) -> Configuration {
        let mut config: Configuration = serde_json::from_slice(&accepted_configuration()).unwrap();
        config.search_round = round;
        config
    }

    /// `policy.safetensors` sha256 of the accepted step-60000 export.
    const ACCEPTED_WEIGHTS: &str =
        "4b4b5a40f03c9226b4b15856fd35b90127c8b6a4d1435b815484cdd050744f1e";

    /// A native challenger fixture: pins only, no file is read.
    fn challenger() -> optimize::NativeChallenger {
        let weights = sha256_hex(b"challenger weights");
        optimize::NativeChallenger {
            checkpoint_receipt: Artifact {
                path: "/fixture/challenger_receipt".into(),
                sha256: sha256_hex(b"challenger receipt"),
            },
            policy_weights: Artifact {
                path: "/fixture/challenger_weights".into(),
                sha256: weights.clone(),
            },
            identity: jope_policy::config::ChallengerIdentity {
                state_sha256: sha256_hex(b"challenger state"),
                parameter_sha256: sha256_hex(b"challenger parameters"),
                config_sha256: jope_policy::ACCEPTED_CONFIG_SHA256.into(),
                policy_safetensors_sha256: weights,
            },
        }
    }

    /// The accepted configuration plus exactly one key.
    fn challenger_configuration() -> Configuration {
        let mut config = configuration(None);
        config.native_challenger = Some(challenger());
        config
    }

    fn pinned(config: &mut Configuration) -> &mut optimize::NativeChallenger {
        config.native_challenger.as_mut().unwrap()
    }

    /// Synthetic regression development partition: 25 rows, one group per row,
    /// five folds of five. Copied from the prepared-data fixture pattern.
    fn development() -> Result<DevelopmentData> {
        let features = json!([{"name":"x","kind":"numeric"}]);
        let source = "a".repeat(64);
        let rows = (0..25usize)
            .map(|i| json!({"id":format!("{i:064x}"),"source_index":i,"group":format!("{:064x}", i + 100),"fold":i % 5,"values":[i.to_string()],"target":(3 * i + i % 4).to_string()}))
            .collect::<Vec<_>>();
        let table = json!({"schema":"jope.prepared-table/v1","dataset":"fixture","task":Task::Regression,"source_sha256":source,"partition":"development","features":features,"classes":[],"rows":rows});
        let bytes = serde_json::to_vec(&table)?;
        let dataset: PanelDataset = serde_json::from_value(json!({
            "id":"fixture","task":Task::Regression,"metric":"rmse",
            "source":{"url":"fixture","sha256":source,"bytes":1,"metadata_url":"fixture","metadata_sha256":"b".repeat(64)},
            "rows":30,"features":features,"classes":[],"group_columns":[],
            "split":{"protocol":"synthetic-fixture","sha256":"c".repeat(64),"development_sha256":sha256_hex(&bytes),"sealed_sha256":"d".repeat(64),"development_rows":25,"sealed_rows":5,"fold_rows":[5,5,5,5,5]},
            "license":{"original":"synthetic","evidence_url":"fixture","repository":"synthetic","note":"synthetic fixture only"},
            "checkpoint_exposure":"synthetic","description":"synthetic fixture","group_limitations":"synthetic groups"
        }))?;
        parse_development(&bytes, &dataset)
    }

    fn plan() -> Result<CampaignPlan> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        // Frozen descriptors only; no dataset file is opened here.
        let (panel, _) = crate::panel::cli::read_panel(&root.join("benchmarks/panel.json"))?;
        let methods = BTreeMap::from([(
            "regression".to_owned(),
            vec![Method::Rust {
                recipe: crate::panel::methods::rust_methods(Task::Regression)?.remove(0),
            }],
        )]);
        let runtimes = BTreeMap::from([(
            "rust".to_owned(),
            Runtime {
                program: Artifact {
                    path: "/synthetic/not-executed".into(),
                    sha256: "2".repeat(64),
                },
                argument_prefix: vec!["benchmark".into()],
                inputs: vec![],
                package_files: vec![],
                environment: BTreeMap::new(),
                version: "fixture".into(),
                unavailable: None,
            },
        )]);
        let experiment = Experiment {
            schema: EXPERIMENT_SCHEMA.into(),
            panel_sha256: crate::panel::FROZEN_PANEL_SHA256.into(),
            methods_sha256: campaign::digest(&(&methods, &runtimes))?,
            source_sha: "c".repeat(40),
            checkpoints: BTreeMap::new(),
            seed: 42,
            resources: Resources {
                workers: 1,
                cpus: 1,
                memory_bytes: 32 << 20,
                swap_bytes: 0,
                gpu_tasks: 0,
                threads_per_worker: 1,
                total_seconds: 120,
                reporting_seconds: 5,
                per_fit_seconds: 30,
            },
            expected_artifacts: vec!["finished".into()],
            protocol: PROTOCOL.into(),
            numerical_policy: score::NUMERICAL_POLICY.into(),
        };
        Ok(CampaignPlan {
            schema: campaign::PLAN_SCHEMA.into(),
            experiment,
            panel,
            methods,
            runtimes,
            scope: SCOPE.into(),
        })
    }

    fn scratch() -> Result<tempfile::TempDir> {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        fs::create_dir_all(&root)?;
        Ok(tempfile::Builder::new()
            .prefix("origin-proof-")
            .tempdir_in(fs::canonicalize(root)?)?)
    }

    /// In-process native refit with the retained input, output and receipt layout
    /// the pool writes. No subprocess, window or scheduler is involved.
    fn refit(plan: &CampaignPlan, data: &DevelopmentData, journal: &Path) -> Result<()> {
        let cut = data.cut(0)?;
        let fit = campaign::prepare_fit_impl(
            plan,
            data,
            &cut,
            &plan.methods["regression"][0],
            EvaluationStage::Screen,
            journal,
            true,
        )?;
        let retained = journal
            .join("inputs")
            .join(format!("{}.json", fit.request.identity()?));
        let output = fit.job.directory.join("output.json");
        worker::fit(&retained, &output)?;
        let outcome = WorkerOutcome {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: fit.job.identity()?,
            request_sha256: fit.request.identity()?,
            command_status: EvaluationStatus::Completed,
            reason: None,
            exit_code: Some(0),
            signal: None,
            outputs: vec![optimize_fit::artifact(output)?],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                accounting_reason: "synthetic in-process fixture fit".into(),
                ..Default::default()
            },
        };
        worker::write_private(
            &journal
                .join("finished")
                .join(format!("{}.json", outcome.job_sha256)),
            &outcome,
        )
    }

    fn has_key(value: &Value, key: &str) -> bool {
        match value {
            Value::Object(map) => map.iter().any(|(k, v)| k == key || has_key(v, key)),
            Value::Array(items) => items.iter().any(|v| has_key(v, key)),
            _ => false,
        }
    }

    #[test]
    fn accepted_configuration_bytes_unchanged_when_new_fields_absent() -> Result<()> {
        let bytes = accepted_configuration();
        let config: Configuration = serde_json::from_slice(&bytes)?;
        assert_eq!(serde_json::to_vec(&config)?, bytes);
        let text = String::from_utf8(bytes.clone())?;
        assert!(
            !text.contains("search_round")
                && !text.contains("incumbents")
                && !text.contains("native_challenger")
        );
        let mut round = config;
        round.search_round = Some(1);
        let mut expected = bytes[..bytes.len() - 1].to_vec();
        expected.extend_from_slice(br#","search_round":1}"#);
        assert_eq!(serde_json::to_vec(&round)?, expected);
        let unknown = format!(r#"{},"search_rounds":1}}"#, &text[..text.len() - 1]);
        assert!(serde_json::from_str::<Configuration>(&unknown).is_err());
        // A native challenger is exactly one appended key, fields in declaration order.
        let c = challenger();
        let mut expected = bytes[..bytes.len() - 1].to_vec();
        expected.extend_from_slice(
            format!(
                r#","native_challenger":{{"checkpoint_receipt":{},"policy_weights":{},"identity":{{"state_sha256":"{}","parameter_sha256":"{}","config_sha256":"{}","policy_safetensors_sha256":"{}"}}}}}}"#,
                pin("challenger_receipt", &c.checkpoint_receipt.sha256),
                pin("challenger_weights", &c.policy_weights.sha256),
                c.identity.state_sha256,
                c.identity.parameter_sha256,
                c.identity.config_sha256,
                c.identity.policy_safetensors_sha256,
            )
            .as_bytes(),
        );
        assert_eq!(serde_json::to_vec(&challenger_configuration())?, expected);
        let decoded: Configuration = serde_json::from_slice(&expected)?;
        assert_eq!(serde_json::to_vec(&decoded)?, expected);
        // Unknown keys are refused inside the challenger and inside its identity.
        let mut extra = serde_json::to_value(challenger_configuration())?;
        extra["native_challenger"]["extra"] = json!(1);
        assert!(serde_json::from_value::<Configuration>(extra).is_err());
        let mut extra = serde_json::to_value(challenger_configuration())?;
        extra["native_challenger"]["identity"]["extra"] = json!(1);
        assert!(serde_json::from_value::<Configuration>(extra).is_err());
        // The accepted pair carries no challenger key.
        let (plan, data) = (plan()?, development()?);
        let (_, pair) = optimize::build_pair_input(&plan, &configuration(None), &data)?;
        assert!(!serde_json::to_string(&pair)?.contains("\"challenger\""));
        Ok(())
    }

    #[test]
    fn native_challenger_refusals_precede_file_checks() {
        let settings: teaching::Settings = serde_json::from_value(json!({
            "state_root": "/fixture/state",
            "prepared_root": "/fixture/prepared",
            "panel": {"path": "/fixture/panel.json", "sha256": crate::panel::FROZEN_PANEL_SHA256},
            "original_label_role": "development",
        }))
        .unwrap();
        type ConfigurationEdit<'a> = &'a dyn Fn(&mut Configuration);
        let cases: [(ConfigurationEdit<'_>, &str); 9] = [
            (
                &|c: &mut Configuration| {
                    pinned(c).identity.state_sha256 = jope_policy::ACCEPTED_STATE_SHA256.into();
                },
                "challenger names the accepted identity",
            ),
            (
                &|c: &mut Configuration| {
                    pinned(c).identity.parameter_sha256 =
                        jope_policy::ACCEPTED_PARAMETER_SHA256.into();
                },
                "challenger names the accepted identity",
            ),
            (
                &|c: &mut Configuration| {
                    let challenger = pinned(c);
                    challenger.identity.policy_safetensors_sha256 = ACCEPTED_WEIGHTS.into();
                    challenger.policy_weights.sha256 = ACCEPTED_WEIGHTS.into();
                },
                "challenger names the accepted identity",
            ),
            (
                &|c: &mut Configuration| {
                    pinned(c).identity.state_sha256 = jope_policy::REJECTED_STATE_SHA256.into();
                },
                "rejected step-69000",
            ),
            (
                &|c: &mut Configuration| pinned(c).identity.config_sha256 = sha256_hex(b"x"),
                "must keep the accepted architecture",
            ),
            (
                &|c: &mut Configuration| {
                    pinned(c).policy_weights.sha256 = sha256_hex(b"other weights");
                },
                "weights differ from its identity",
            ),
            (
                &|c: &mut Configuration| pinned(c).checkpoint_receipt.sha256 = "3".repeat(64),
                "repeats the accepted checkpoint receipt",
            ),
            (
                // Its values are irrelevant: this check runs before the teaching settings'.
                &|c: &mut Configuration| c.teaching = Some(settings.clone()),
                "without teaching, incumbents or experimental",
            ),
            (
                &|c: &mut Configuration| {
                    c.incumbents = Some(Incumbents {
                        neural: vec![],
                        generic: vec![],
                    });
                },
                "without teaching, incumbents or experimental",
            ),
        ];
        for (edit, needle) in cases {
            let mut config = challenger_configuration();
            edit(&mut config);
            let error = format!("{:#}", config.validate().unwrap_err());
            assert!(error.contains(needle), "{error}");
        }
        // Control: a well-formed challenger passes every challenger check and fails later, at the
        // absent vocabulary file, for a reason that is not about the challenger.
        let error = format!("{:#}", challenger_configuration().validate().unwrap_err());
        assert!(!error.contains("challenger"), "{error}");
        // Both challenger pins are dependencies: form-checked here, verified with the rest.
        let pins = challenger();
        let challenged = challenger_configuration();
        let dependencies = challenged.dependencies();
        assert_eq!(
            dependencies.len(),
            configuration(None).dependencies().len() + 2
        );
        for pin in [&pins.checkpoint_receipt, &pins.policy_weights] {
            assert!(dependencies
                .iter()
                .any(|a| a.path == pin.path && a.sha256 == pin.sha256));
        }
        // The dependency loop runs before the vocabulary read, so these fail there, not later.
        let malformed: [&dyn Fn(&mut Configuration); 3] = [
            &|c: &mut Configuration| pinned(c).policy_weights.path = "relative".into(),
            &|c: &mut Configuration| pinned(c).checkpoint_receipt.path = "relative".into(),
            // Only the dependency loop checks the receipt pin's format.
            &|c: &mut Configuration| {
                pinned(c).checkpoint_receipt.sha256 =
                    sha256_hex(b"challenger receipt").to_uppercase();
            },
        ];
        for edit in malformed {
            let mut config = challenger_configuration();
            edit(&mut config);
            let error = format!("{:#}", config.validate().unwrap_err());
            assert!(error.contains("malformed dependency pin"), "{error}");
        }
    }

    #[test]
    fn native_challenger_changes_only_its_pins_in_the_pair() -> Result<()> {
        let plan = plan()?;
        let data = development()?;
        // `arm()` runs `Request::validate`, which requires a checkpoint vocabulary pin;
        // `build_pair_input` never reads the vocabulary file.
        let mut accepted = configuration(None);
        accepted.vocabulary.sha256 = jope_policy::CHECKPOINT_VOCABULARY_SHA256.into();
        let mut challenged = challenger_configuration();
        challenged.vocabulary.sha256 = jope_policy::CHECKPOINT_VOCABULARY_SHA256.into();
        let (_, a) = optimize::build_pair_input(&plan, &accepted, &data)?;
        let (_, c) = optimize::build_pair_input(&plan, &challenged, &data)?;
        assert_ne!(optimize_fit::digest(&a)?, optimize_fit::digest(&c)?);
        for neural in [true, false] {
            assert_ne!(a.arm(neural)?.id_sha256, c.arm(neural)?.id_sha256);
        }
        let pins = challenger();
        let proposal = c.proposal.accepted()?;
        assert_eq!(proposal.checkpoint_sha256, pins.identity.state_sha256);
        assert_eq!(
            proposal.policy_safetensors_sha256,
            pins.identity.policy_safetensors_sha256
        );
        assert_eq!(proposal.challenger, Some(pins.identity.clone()));
        assert_eq!(
            c.installation.checkpoint_receipt_sha256,
            sha256_hex(b"challenger receipt")
        );
        // Everything else (selection, encoding, seeds, preprocessing) is identical.
        let strip = |pair: &neural_search::PairInputPlan| -> Result<Value> {
            let mut value = serde_json::to_value(pair)?;
            let proposal = value["proposal"]
                .as_object_mut()
                .context("accepted proposal object")?;
            for key in [
                "checkpoint_sha256",
                "policy_safetensors_sha256",
                "challenger",
            ] {
                proposal.remove(key);
            }
            value["installation"]
                .as_object_mut()
                .context("installation object")?
                .remove("checkpoint_receipt_sha256");
            Ok(value)
        };
        assert_eq!(strip(&a)?, strip(&c)?);
        // The teaching comparison ignores the checkpoint, which is why teaching is refused.
        let comparison = |config: &Configuration| {
            teaching::comparison_identity(&plan.experiment, config, &data.dataset, "e".repeat(64))
                .sha256()
        };
        assert_eq!(comparison(&accepted)?, comparison(&challenged)?);
        let (receipt, weights) = optimize::proposal_files(&challenged);
        assert_eq!(receipt.sha256, pins.checkpoint_receipt.sha256);
        assert_eq!(weights.sha256, pins.policy_weights.sha256);
        let (receipt, weights) = optimize::proposal_files(&accepted);
        assert_eq!(receipt.sha256, "3".repeat(64));
        assert_eq!(weights.sha256, ACCEPTED_WEIGHTS);
        Ok(())
    }

    #[test]
    fn origin_proof_records_the_native_challenger_state() -> Result<()> {
        let keys = |plan: &CampaignPlan| {
            plan.experiment
                .checkpoints
                .keys()
                .cloned()
                .collect::<BTreeSet<_>>()
        };
        let mut accepted = plan()?;
        record_checkpoints(&mut accepted, &configuration(None));
        assert_eq!(
            keys(&accepted),
            BTreeSet::from(["asm.accepted-policy-state".to_owned()])
        );
        let mut challenged = plan()?;
        record_checkpoints(&mut challenged, &challenger_configuration());
        assert_eq!(
            keys(&challenged),
            BTreeSet::from([
                "asm.accepted-policy-state".to_owned(),
                "asm.native-challenger-state".to_owned(),
            ])
        );
        assert_eq!(
            challenged.experiment.checkpoints["asm.native-challenger-state"],
            challenger().identity.state_sha256
        );
        // The fixture plan is not a complete admitted plan; the experiment identity is what
        // binds the preregistered checkpoints.
        assert_ne!(
            accepted.experiment.identity()?,
            challenged.experiment.identity()?
        );
        Ok(())
    }

    #[test]
    fn warm_demo_refuses_a_native_challenger_plan() {
        crate::panel::demo_asm::refuse_native_challenger(&configuration(None)).unwrap();
        let error = format!(
            "{:#}",
            crate::panel::demo_asm::refuse_native_challenger(&challenger_configuration())
                .unwrap_err()
        );
        assert!(error.contains("not a demo route"), "{error}");
    }

    #[test]
    fn challenger_receipt_must_bind_its_pins() -> Result<()> {
        let scratch = scratch()?;
        let mut challenger = challenger();
        let identity = challenger.identity.clone();
        let receipt = |accepted: bool, output: &str| {
            json!({
                "schema": jope_policy::config::EXPORT_RECEIPT_SCHEMA,
                "checkpoint": {
                    "accepted_identity": accepted,
                    "carried_bindings": {
                        "corpus_sha256": sha256_hex(b"corpus"),
                        "vocabulary_sha256": jope_policy::CHECKPOINT_VOCABULARY_SHA256,
                    },
                    "config": jope_policy::config::PolicyConfig::regression(401),
                    "config_sha256": jope_policy::ACCEPTED_CONFIG_SHA256,
                    "expected_state_sha256": identity.state_sha256,
                    "model_sha256": sha256_hex(b"challenger model"),
                    "parameter_sha256": identity.parameter_sha256,
                    "parameter_sha256_recomputed": identity.parameter_sha256,
                    "state_sha256": identity.state_sha256,
                    "step": 61600,
                    "verified_bindings": ["model_sha256", "config_sha256", "autocast_cache_lifetime", "parameter_sha256"],
                },
                "policy": {"output": {"sha256": output}, "tensor_count": 0, "tensors": []},
            })
        };
        let write = |name: &str, value: &Value| -> Result<Artifact> {
            let path = scratch.path().join(name);
            let bytes = serde_json::to_vec(value)?;
            fs::write(&path, &bytes)?;
            Ok(Artifact {
                path,
                sha256: sha256_hex(&bytes),
            })
        };
        let weights = identity.policy_safetensors_sha256.clone();
        challenger.checkpoint_receipt = write("receipt.json", &receipt(false, &weights))?;
        let binding = optimize::challenger_checkpoint(&challenger)?;
        assert_eq!(binding.state_sha256(), identity.state_sha256);
        // The pin must match the file's bytes.
        let mut stale = challenger.clone();
        stale.checkpoint_receipt.sha256 = sha256_hex(b"other receipt");
        assert!(optimize::challenger_checkpoint(&stale).is_err());
        // The receipt must name the pinned weights.
        let mut other = challenger.clone();
        other.checkpoint_receipt = write(
            "other-weights.json",
            &receipt(false, &sha256_hex(b"other weights")),
        )?;
        let error = optimize::challenger_checkpoint(&other).unwrap_err();
        assert!(
            format!("{error:#}").contains("names other challenger weights"),
            "{error:#}"
        );
        // A receipt that claims the accepted identity is refused.
        let mut claimed = challenger.clone();
        claimed.checkpoint_receipt = write("accepted.json", &receipt(true, &weights))?;
        assert!(optimize::challenger_checkpoint(&claimed).is_err());
        Ok(())
    }

    #[test]
    fn round_zero_is_legacy_and_rounds_reseed() -> Result<()> {
        let sha = "a".repeat(64);
        let legacy = jope_pipeline::search_plan::candidate_seed(42, &sha)?;
        assert_eq!(optimize::pair_seed(42, &sha, None)?, legacy);
        assert_eq!(optimize::pair_seed(42, &sha, Some(0))?, legacy);
        let seeds = [None, Some(1), Some(2), Some(3)]
            .into_iter()
            .map(|round| optimize::pair_seed(42, &sha, round))
            .collect::<Result<BTreeSet<_>>>()?;
        assert_eq!(seeds.len(), 4);
        Ok(())
    }

    #[test]
    fn round_bound_precedes_file_checks() {
        let error = configuration(Some(MAX_SEARCH_ROUND + 1))
            .validate()
            .unwrap_err();
        assert!(format!("{error:#}").contains("search round outside approved bounds"));
        // A later check reads the absent vocabulary file and fails for another reason.
        let error = configuration(Some(3)).validate().unwrap_err();
        assert!(!format!("{error:#}").contains("search round outside approved bounds"));
    }

    #[test]
    fn comparison_identity_equal_across_rounds_0_to_3() -> Result<()> {
        let plan = plan()?;
        let data = development()?;
        let mut identities = BTreeSet::new();
        let mut selections = BTreeSet::new();
        let mut stripped = BTreeSet::new();
        let mut pairs = Vec::new();
        for round in [None, Some(0), Some(1), Some(2), Some(3)] {
            let config = configuration(round);
            // The inner partition takes no configuration (teaching.rs derive_inner_partition).
            let comparison = teaching::comparison_identity(
                &plan.experiment,
                &config,
                &data.dataset,
                "e".repeat(64),
            );
            identities.insert(comparison.sha256()?);
            let (_, pair) = optimize::build_pair_input(&plan, &config, &data)?;
            selections.insert(pair.selection.identity()?);
            let mut value = serde_json::to_value(&pair)?;
            value
                .as_object_mut()
                .context("pair plan object")?
                .remove("random_seed");
            let proposal = value["proposal"]
                .as_object_mut()
                .context("accepted proposal object")?;
            for key in ["sampling_seed", "fallback_seed", "exploration_seed"] {
                proposal.remove(key);
            }
            stripped.insert(value.to_string());
            pairs.push(pair);
        }
        assert_eq!(identities.len(), 1);
        assert_eq!(selections.len(), 1);
        assert_eq!(stripped.len(), 1);
        let seeds = |pair: &neural_search::PairInputPlan| -> Result<[u64; 4]> {
            let proposal = pair.proposal.accepted()?;
            Ok([
                pair.random_seed,
                proposal.sampling_seed,
                proposal.fallback_seed,
                proposal.exploration_seed,
            ])
        };
        let legacy = seeds(&pairs[0])?;
        assert_eq!(
            serde_json::to_vec(&pairs[1])?,
            serde_json::to_vec(&pairs[0])?
        );
        for pair in &pairs[2..] {
            let reseeded = seeds(pair)?;
            assert!(reseeded.iter().zip(legacy).all(|(a, b)| *a != b));
        }
        Ok(())
    }

    #[test]
    fn incumbent_pin_must_be_an_intact_teaching_blob() -> Result<()> {
        let scratch = scratch()?;
        let state_root = scratch.path().join("state");
        let state = jope_learning::state::State::at(&state_root);
        fs::create_dir_all(state.blob_path("x").parent().context("blob directory")?)?;
        let plan = plan()?;
        let data = development()?;
        let cut = data.cut(0)?;
        let outer = campaign::request(
            &data,
            &cut,
            &plan.methods["regression"][0],
            &plan.experiment,
            EvaluationStage::Screen,
        )?;
        let selection = optimize_fit::selection(&cut, outer)?;
        let at_blob = |sha256: String| Artifact {
            path: state.blob_path(&sha256),
            sha256,
        };
        let absent = sha256_hex(b"absent frozen recipe");
        let corrupt = sha256_hex(b"x");
        fs::write(state.blob_path(&corrupt), b"not x")?;
        let not_a_recipe = state.put_blob(b"{}")?;
        // An intact blob pinned somewhere other than its teaching-state address.
        let misplaced = Artifact {
            path: scratch.path().join("elsewhere"),
            sha256: not_a_recipe.clone(),
        };
        for (artifact, reason) in [
            (
                misplaced,
                Some("incumbent must pin a frozen-recipe blob in the teaching state"),
            ),
            (at_blob(absent), Some("absent from the teaching state")),
            (at_blob(corrupt), Some("is corrupt")),
            (at_blob(not_a_recipe), None),
        ] {
            let error =
                optimize::incumbent_seed(&state_root, &artifact, &selection, false).unwrap_err();
            let error = format!("{error:#}");
            match reason {
                Some(reason) => assert!(error.contains(reason), "{error}"),
                // Passed every teaching-state check; refused as a malformed recipe.
                None => assert!(
                    !error.contains("teaching state") && !error.contains("is corrupt"),
                    "{error}"
                ),
            }
        }
        // Structural and pure: refused before any file check reads the pin.
        let mut config = configuration(None);
        config.incumbents = Some(Incumbents {
            neural: vec![at_blob(sha256_hex(b"frozen recipe"))],
            generic: vec![],
        });
        let error = config.validate().unwrap_err();
        assert!(format!("{error:#}").contains("incumbent seeding requires a teaching state"));
        // The same path binding in the pure configuration check.
        config.teaching = Some(teaching::Settings {
            state_root: "/fixture/state".into(),
            prepared_root: "/fixture/prepared".into(),
            panel: Artifact {
                path: "/fixture/panel.json".into(),
                sha256: crate::panel::FROZEN_PANEL_SHA256.into(),
            },
            original_label_role: "development".into(),
        });
        config.incumbents = Some(Incumbents {
            neural: vec![Artifact {
                path: "/fixture/elsewhere".into(),
                sha256: sha256_hex(b"frozen recipe"),
            }],
            generic: vec![],
        });
        let error = config.validate().unwrap_err();
        assert!(format!("{error:#}")
            .contains("incumbent must pin a frozen-recipe blob in the teaching state"));
        Ok(())
    }

    #[test]
    fn trivial_hold_rmse_formula() -> Result<()> {
        let labels = ["0".to_owned(), "6".to_owned()];
        // Mean 1/4 + 2/4 + 3/4 + 6/4 = 3; RMSE sqrt((9 + 9) / 2) = 3.
        assert_eq!(trivial_hold_rmse(&[1., 2., 3., 6.], &labels)?, 3.0);
        // Pin the guard's Σ(y/n) in row order (jope-learning eligibility.rs:395). For
        // 1, 2, 4 it ends one ulp below (Σy)/n = fl(7/3); reversed, it rounds to fl(7/3).
        let zero = ["0".to_owned()];
        let scored = |mean: f64| -> Result<u64> {
            let predictions = score::Predictions::Values(vec![mean]);
            let result = score::score_predictions(Task::Regression, &[], &zero, &predictions)?;
            Ok(result.value.to_bits())
        };
        let mut means = vec![];
        for ys in [[1., 2., 4.], [4., 2., 1.]] {
            let n = ys.len() as f64;
            let mut mean = 0.;
            for y in ys {
                mean += y / n;
            }
            assert_eq!(trivial_hold_rmse(&ys, &zero)?.to_bits(), scored(mean)?);
            means.push(mean);
        }
        // Checker controls: the fixture separates row-order Σ(y/n) from (Σy)/n and
        // from any order-free sum, and the scorer keeps the one-ulp difference.
        assert_ne!(means[0], 7. / 3.);
        assert_eq!(means[1], 7. / 3.);
        assert_ne!(scored(means[0])?, scored(7. / 3.)?);
        assert!(trivial_hold_rmse(&[], &labels).is_err());
        assert!(trivial_hold_rmse(&[1.], &[]).is_err());
        Ok(())
    }

    #[test]
    fn held_label_mutation_changes_scores_only() -> Result<()> {
        let scratch = scratch()?;
        let journal = scratch.path();
        let plan = plan()?;
        let config = configuration(None);
        let data = development()?;
        refit(&plan, &data, journal)?;
        let mut mutated = data.clone();
        for (i, row) in mutated.rows.iter_mut().enumerate() {
            if row.fold == 0 {
                row.target = (1000 + i).to_string();
            }
        }
        let (context, pair) = optimize::build_pair_input(&plan, &config, &data)?;
        let (mutated_context, mutated_pair) = optimize::build_pair_input(&plan, &config, &mutated)?;
        assert_eq!(
            optimize_fit::digest(&pair)?,
            optimize_fit::digest(&mutated_pair)?
        );
        assert_eq!(
            optimize_fit::digest(&context)?,
            optimize_fit::digest(&mutated_context)?
        );
        let rust = plan.runtimes["rust"].clone();
        let original = FitAdapter::prepare(
            &data,
            &data.cut(0)?,
            pair.selection.clone(),
            rust.clone(),
            journal,
            true,
        )?;
        // Replay with write disabled: every retained fold input must match byte for byte.
        let replayed = FitAdapter::prepare(
            &mutated,
            &mutated.cut(0)?,
            pair.selection.clone(),
            rust,
            journal,
            false,
        )?;
        for (a, b) in original.folds.iter().zip(&replayed.folds) {
            assert_eq!(a.file.sha256, b.file.sha256);
            assert_eq!(a.labels, b.labels);
        }
        let method = &plan.methods["regression"][0];
        let request = |data: &DevelopmentData| -> Result<String> {
            let cut = data.cut(0)?;
            campaign::request(
                data,
                &cut,
                method,
                &plan.experiment,
                EvaluationStage::Screen,
            )?
            .identity()
        };
        assert_eq!(request(&data)?, request(&mutated)?);
        let methods = plan.methods["regression"].clone();
        let before = refit_predictions(&plan, &data, journal, &methods)?;
        let after = refit_predictions(&plan, &mutated, journal, &methods)?;
        assert_eq!(before.len(), 1);
        assert_eq!(before[0]["status"], "completed");
        assert_eq!(before[0]["predictions"], after[0]["predictions"]);
        assert_eq!(
            before[0]["fitted_artifact_sha256"],
            after[0]["fitted_artifact_sha256"]
        );
        assert_ne!(
            before[0]["development_hold_score"],
            after[0]["development_hold_score"]
        );
        let trivial = |data: &DevelopmentData| -> Result<f64> {
            let cut = data.cut(0)?;
            trivial_hold_rmse(&cut.train.targets, &cut.hold.labels)
        };
        assert_ne!(trivial(&data)?, trivial(&mutated)?);
        // Negative control: a fitting-row label does reach the retained fit input.
        let mut fitting = data.clone();
        fitting
            .rows
            .iter_mut()
            .find(|row| row.fold == 1)
            .context("fold-one row")?
            .target = "12345".into();
        let error = refit_predictions(&plan, &fitting, journal, &methods).unwrap_err();
        assert!(format!("{error:#}").contains("retained fit input changed original panel rows"));
        Ok(())
    }

    #[test]
    fn report_scoring_ignores_observations() -> Result<()> {
        let scratch = scratch()?;
        let journal = scratch.path();
        let plan = plan()?;
        let data = development()?;
        refit(&plan, &data, journal)?;
        let methods = plan.methods["regression"].clone();
        let before = refit_predictions(&plan, &data, journal, &methods)?;
        // The key score_batch would write for this refit.
        let request = campaign::request(
            &data,
            &data.cut(0)?,
            &methods[0],
            &plan.experiment,
            EvaluationStage::Screen,
        )?;
        let observations = journal.join("observations");
        fs::create_dir_all(&observations)?;
        fs::write(
            observations.join(format!("{}.json", request.identity()?)),
            b"arbitrary bytes, not an observation",
        )?;
        assert_eq!(refit_predictions(&plan, &data, journal, &methods)?, before);
        Ok(())
    }

    #[test]
    fn deferred_output_has_no_hold_score() -> Result<()> {
        let value = deferred("fixture", json!(null));
        let keys = value
            .as_object()
            .context("deferred object")?
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            keys,
            BTreeSet::from([
                "complete_panel_demonstration",
                "dataset",
                "ok",
                "schema",
                "state",
                "window",
            ])
        );
        assert_eq!(value["state"], "complete_report_deferred");
        for key in [
            "development_hold_score",
            "trivial_hold_rmse",
            "refitted_predictions",
            "predictions",
        ] {
            assert!(!has_key(&value, key), "{key}");
        }
        // Checker control: the same walk finds a hold score in real refit output.
        let scratch = scratch()?;
        let plan = plan()?;
        let data = development()?;
        refit(&plan, &data, scratch.path())?;
        let predictions =
            refit_predictions(&plan, &data, scratch.path(), &plan.methods["regression"])?;
        assert!(has_key(
            &Value::Array(predictions),
            "development_hold_score"
        ));
        Ok(())
    }

    #[test]
    fn deferred_attachment_scores_no_hold_row() -> Result<()> {
        let scratch = scratch()?;
        let journal = scratch.path();
        let plan = plan()?;
        let data = development()?;
        refit(&plan, &data, journal)?;
        let fit = campaign::prepare_fit_impl(
            &plan,
            &data,
            &data.cut(0)?,
            &plan.methods["regression"][0],
            EvaluationStage::Screen,
            journal,
            false,
        )?;
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(
            journal
                .join("finished")
                .join(format!("{}.json", fit.job.identity()?)),
        )?)?;
        let observation = journal
            .join("observations")
            .join(format!("{}.json", fit.request.identity()?));
        let fits = [fit];
        let screen = |ready: bool, score_screen: bool| {
            crate::panel::asm_campaign::screen_observations(
                ready,
                score_screen,
                &fits,
                vec![outcome.clone()],
                std::slice::from_ref(&data),
                journal,
            )
        };
        for (ready, score_screen) in [(true, false), (false, true), (false, false)] {
            assert!(screen(ready, score_screen)?.is_empty());
        }
        assert!(!journal.join("observations").try_exists()?);
        // A deferred journal refuses scoring even if the caller asks for it.
        let marker = journal.join(crate::panel::campaign_cli::DEFERRAL_MARKER);
        fs::write(&marker, b"{}")?;
        let error = screen(true, true).unwrap_err();
        assert!(format!("{error:#}").contains("defers development hold scoring"));
        assert!(!journal.join("observations").try_exists()?);
        // Checker control: the same fixture is scored and written once allowed.
        fs::remove_file(&marker)?;
        assert_eq!(screen(true, true)?.len(), 1);
        assert!(observation.try_exists()?);
        Ok(())
    }
}
