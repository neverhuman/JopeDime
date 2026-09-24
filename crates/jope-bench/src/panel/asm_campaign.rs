//! Frozen-panel attachment for paired ASM arms. Missing neural recipes remain
//! explicit inventory entries; a generic winner is never renamed as neural ASM.
use super::{
    campaign::{self, CampaignPlan, Method},
    data::DevelopmentData,
    optimize::{self, Configuration, DatasetSearch},
    optimize_fit::{artifact, digest, immutable, FitAdapter},
    protocol::Observation,
};
use anyhow::{ensure, Context, Result};
use jope_pipeline::{
    execution::{ExecutionPool, WorkerOutcome},
    experiment::{EvaluationStage, EvaluationStatus},
    neural_search::{self, Origin, SearchReceipt},
};
use serde_json::{json, Value};
use std::{collections::BTreeMap, fs, path::Path, sync::atomic::AtomicBool};

pub struct Attachment {
    pub methods: BTreeMap<String, Vec<Method>>,
    pub screen: Vec<Observation>,
    pub aliases: BTreeMap<String, String>,
    pub inventory: Value,
    pub ready: bool,
}
fn methods(search: &DatasetSearch) -> Vec<Method> {
    [
        (
            optimize::NEURAL_METHOD,
            search.neural.as_ref(),
            Some(search.pair.proposal.checkpoint_sha256().to_owned()),
        ),
        (optimize::GENERIC_METHOD, search.generic.as_ref(), None),
    ]
    .into_iter()
    .filter_map(|(id, winner, checkpoint_sha256)| {
        winner.map(|w| Method::Synthesized {
            id: id.into(),
            recipe: w.candidate.recipe.clone(),
            checkpoint_sha256,
        })
    })
    .collect()
}
fn settled(search: &DatasetSearch) -> bool {
    !search.neural_retryable_inventory && !search.receipt.neural.as_ref().and_then(|n|n.search.as_ref()).is_some_and(optimize::pending)
        && !search.receipt.generic.as_ref().is_some_and(optimize::pending)
        // A proposal absent because context is explicitly unqualified is known.
        // Other command non-attempts remain resumable in the original window.
        && !search.receipt.neural.as_ref().is_some_and(|n|n.search.is_none() && matches!(n.proposal_execution.command_status,EvaluationStatus::Unavailable|EvaluationStatus::Cancelled) && n.proposal_execution.consumption.gate_released_unix_ms.is_none())
        && search.receipt.generic_error.is_none()
}
fn arms_expired(plan: &neural_search::PairInputPlan, journal: &Path) -> Result<bool> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as u64;
    for neural in [true, false] {
        let arm = plan.arm(neural)?;
        let path = journal.join("arms").join(format!("{}.json", arm.id_sha256));
        if !path.exists() {
            return Ok(false);
        }
        let deadline = super::receipt_timing::arm_deadline(
            journal,
            &arm,
            &plan.selection.outer().experiment_sha256,
        )?;
        if now < deadline {
            return Ok(false);
        }
    }
    Ok(true)
}
fn exhausted(pool: &ExecutionPool, plan: &CampaignPlan) -> Result<bool> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as u64;
    Ok(pool.window().is_some_and(|w| {
        now >= w
            .deadline_unix_ms
            .saturating_sub(plan.experiment.resources.reporting_seconds * 1000)
    }))
}
pub fn attach(
    plan: &CampaignPlan,
    data: &[DevelopmentData],
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
    baseline: &[Observation],
) -> Result<Attachment> {
    let config = Configuration::from_campaign(plan)?;
    let order = super::optimize_schedule::dataset_order(
        &data.iter().map(|d| d.dataset.task).collect::<Vec<_>>(),
    );
    immutable(
        &journal.join("asm-scheduling.json"),
        &json!({"policy":config.scheduling_policy,"wave_width":config.dataset_wave_width,"dataset_indices":order,"global_window":pool.window()}),
    )?;
    let mut generated = BTreeMap::new();
    let mut inventory = serde_json::Map::new();
    let mut ready = true;
    for wave in order.chunks(config.dataset_wave_width) {
        let mut active = Vec::new();
        for &i in wave {
            let dataset = &data[i];
            let path = journal
                .join("asm-final")
                .join(format!("{}.json", dataset.dataset.id));
            if path.exists() {
                let value: Value = serde_json::from_slice(&fs::read(path)?)?;
                if !value["search"].is_null() {
                    let search: DatasetSearch = serde_json::from_value(value["search"].clone())?;
                    verify_search(plan, &config, dataset, journal, &search)?;
                    generated.insert(dataset.dataset.id.clone(), methods(&search));
                }
                inventory.insert(dataset.dataset.id.clone(), value);
            } else if exhausted(pool, plan)? {
                let value = json!({"state":"global_budget_exhausted","search":null,"reason":"dataset was not admitted before original campaign fitting cutoff","neural":"unknown_unstarted_or_unfrozen","generic":"unknown_unstarted_or_unfrozen"});
                immutable(&path, &value)?;
                inventory.insert(dataset.dataset.id.clone(), value);
            } else {
                active.push(i);
            }
        }
        if active.is_empty() {
            continue;
        }
        let wave = active.as_slice();
        let config = &config;
        let tasks = wave
            .iter()
            .map(|&i| {
                let dataset = &data[i];
                move |proxy: &mut super::optimize_schedule::Proxy<DatasetSearch>| {
                    optimize::search_dataset(plan, config, dataset, journal, proxy, cancel)
                }
            })
            .collect();
        let results = super::optimize_schedule::wave(pool, cancel, tasks)?;
        for (&i, result) in wave.iter().zip(results) {
            let dataset = &data[i];
            match result {
                Ok(search) => {
                    let done = settled(&search)
                        || arms_expired(&search.pair, journal)?
                        || exhausted(pool, plan)?;
                    let value = json!({"state":if done {"terminal"}else{"pending_original_window"},"search":search});
                    if done {
                        immutable(
                            &journal
                                .join("asm-final")
                                .join(format!("{}.json", dataset.dataset.id)),
                            &value,
                        )?;
                    } else {
                        ready = false;
                    }
                    if done {
                        generated.insert(dataset.dataset.id.clone(), methods(&search));
                    }
                    inventory.insert(dataset.dataset.id.clone(), value);
                }
                Err(error) => {
                    let terminal = exhausted(pool, plan)?;
                    let value = json!({"state":if terminal{"global_budget_exhausted"}else{"pending_original_window"},"search":null,"reason":format!("{error:#}"),"neural":"unknown_no_completed_recipe","generic":"unknown_no_completed_recipe"});
                    if terminal {
                        immutable(
                            &journal
                                .join("asm-final")
                                .join(format!("{}.json", dataset.dataset.id)),
                            &value,
                        )?;
                    } else {
                        ready = false;
                    }
                    inventory.insert(dataset.dataset.id.clone(), value);
                }
            }
        }
    }
    // All thirty inventory entries survive a partially admitted wave/campaign.
    ensure!(
        inventory.len() == data.len()
            && (plan.scope != "asm-panel" || data.len() == plan.panel.datasets.len()),
        "missing frozen dataset outcome"
    );
    if !ready {
        return Ok(Attachment {
            methods: generated,
            screen: vec![],
            aliases: BTreeMap::new(),
            inventory: Value::Object(inventory),
            ready: false,
        });
    }
    let mut fits = Vec::new();
    let mut screen = Vec::new();
    let mut aliases = BTreeMap::new();
    for dataset in data {
        let cut = dataset.cut(0)?;
        for method in generated.get(&dataset.dataset.id).into_iter().flatten() {
            let expected = campaign::request(
                dataset,
                &cut,
                method,
                &plan.experiment,
                EvaluationStage::Screen,
            )?;
            if let Some(existing) = baseline
                .iter()
                .find(|o| o.request.identity().ok() == expected.identity().ok())
            {
                let mut reused = existing.clone();
                reused.method = method.id().into();
                aliases.insert(
                    format!("{}/{}", dataset.dataset.id, method.id()),
                    existing.method.clone(),
                );
                screen.push(reused);
            } else {
                fits.push(campaign::prepare_fit(
                    plan,
                    dataset,
                    &cut,
                    method,
                    EvaluationStage::Screen,
                    journal,
                )?);
            }
        }
    }
    immutable(&journal.join("asm-roster.json"), &generated)?;
    immutable(&journal.join("asm-screen-aliases.json"), &aliases)?;
    immutable(
        &journal.join("asm-screen-jobs.json"),
        &fits.iter().map(|f| &f.job).collect::<Vec<_>>(),
    )?;
    let outcomes = pool.run(fits.iter().map(|f| f.job.clone()).collect(), cancel)?;
    if outcomes.iter().any(|o| {
        !journal
            .join("finished")
            .join(format!("{}.json", o.job_sha256))
            .exists()
    }) {
        ready = false;
    }
    if ready {
        screen.extend(campaign::score_batch(&fits, outcomes, data, journal)?);
    }
    Ok(Attachment {
        methods: generated,
        screen,
        aliases,
        inventory: Value::Object(inventory),
        ready,
    })
}

/// Authentication reads the same frozen rows and every completed inner fit.
/// This code creates no input, output, arm clock, model, or worker process.
pub fn recover(
    plan: &CampaignPlan,
    data: &[DevelopmentData],
    journal: &Path,
) -> Result<Attachment> {
    let config = Configuration::from_campaign(plan)?;
    let mut generated = BTreeMap::new();
    let mut inventory = serde_json::Map::new();
    for dataset in data {
        let path = journal
            .join("asm-final")
            .join(format!("{}.json", dataset.dataset.id));
        if !path.exists() {
            inventory.insert(dataset.dataset.id.clone(),json!({"state":"unknown_or_resumable","neural":"no_final_outcome","generic":"no_final_outcome"}));
            continue;
        }
        let value: Value = serde_json::from_slice(&fs::read(path)?)?;
        if !value["search"].is_null() {
            let search: DatasetSearch = serde_json::from_value(value["search"].clone())?;
            verify_search(plan, &config, dataset, journal, &search)?;
            generated.insert(dataset.dataset.id.clone(), methods(&search));
        } else {
            ensure!(
                value["state"] == "global_budget_exhausted",
                "unproved terminal missing arm"
            );
            let window: jope_pipeline::experiment::ExecutionWindow =
                serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
            window.validate(&plan.experiment)?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64;
            ensure!(
                now >= window.deadline_unix_ms - plan.experiment.resources.reporting_seconds * 1000,
                "claimed deadline was not reached"
            );
        }
        inventory.insert(dataset.dataset.id.clone(), value);
    }
    let roster_path = journal.join("asm-roster.json");
    if roster_path.exists() {
        let retained: BTreeMap<String, Vec<Method>> =
            serde_json::from_slice(&fs::read(roster_path)?)?;
        ensure!(
            digest(&retained)? == digest(&generated)?,
            "generated roster differs from authenticated frozen winners"
        );
    }
    let alias_path = journal.join("asm-screen-aliases.json");
    let aliases: BTreeMap<String, String> = if alias_path.exists() {
        serde_json::from_slice(&fs::read(alias_path)?)?
    } else {
        BTreeMap::new()
    };
    for (key, baseline) in &aliases {
        let (dataset_id, method_id) = key.split_once('/').context("invalid alias")?;
        let dataset = data
            .iter()
            .find(|d| d.dataset.id == dataset_id)
            .context("unknown alias dataset")?;
        let native = plan.methods[dataset.dataset.task.as_str()]
            .iter()
            .find(|m| m.id() == baseline)
            .context("alias source not a comparator")?;
        let method = generated
            .get(dataset_id)
            .and_then(|ms| ms.iter().find(|m| m.id() == method_id))
            .context("alias target absent")?;
        ensure!(
            campaign::request(
                dataset,
                &dataset.cut(0)?,
                native,
                &plan.experiment,
                EvaluationStage::Screen
            )?
            .identity()?
                == campaign::request(
                    dataset,
                    &dataset.cut(0)?,
                    method,
                    &plan.experiment,
                    EvaluationStage::Screen
                )?
                .identity()?,
            "alias changes executable recipe or comparison"
        );
    }
    Ok(Attachment {
        methods: generated,
        screen: vec![],
        aliases,
        inventory: Value::Object(inventory),
        ready: true,
    })
}
fn verify_search(
    plan: &CampaignPlan,
    config: &Configuration,
    dataset: &DevelopmentData,
    journal: &Path,
    search: &DatasetSearch,
) -> Result<()> {
    ensure!(
        search.dataset == dataset.dataset.id
            && search.task == dataset.dataset.task
            && !search.final_test_targets_used,
        "wrong search dataset or label closure"
    );
    let cut = dataset.cut(0)?;
    let (input, expected) = optimize::build_pair_input(plan, config, dataset)?;
    ensure!(
        digest(&expected)? == digest(&search.pair)?,
        "paired plan differs from exact frozen training cells, labels or full configuration"
    );
    let input_file = artifact(
        journal
            .join("asm-context-inputs")
            .join(format!("{}.json", expected.encoding.input_sha256)),
    )?;
    ensure!(
        input_file.sha256 == expected.encoding.input_sha256
            && fs::read(&input_file.path)? == serde_json::to_vec(&input)?,
        "context worker input differs from exact frozen training rows"
    );
    let adapter = FitAdapter::prepare(
        dataset,
        &cut,
        search.pair.selection.clone(),
        plan.runtimes["rust"].clone(),
        journal,
        false,
    )?;
    for (neural, receipt, frozen) in [
        (
            true,
            search
                .receipt
                .neural
                .as_ref()
                .and_then(|r| r.search.as_ref()),
            search.neural.as_ref(),
        ),
        (
            false,
            search.receipt.generic.as_ref(),
            search.generic.as_ref(),
        ),
    ] {
        if let Some(receipt) = receipt {
            verify_receipt(&adapter, journal, &search.pair, receipt, &plan.experiment)?;
            ensure!(
                receipt.arm.id_sha256 == search.pair.arm(neural)?.id_sha256,
                "arm assignment changed"
            );
            if neural {
                config.verify_worker_builds()?;
                if matches!(
                    search.pair.proposal,
                    neural_search::ProposalPlan::Experimental(_)
                ) {
                    let vocabulary = jope_policy::grammar::bind_vocabulary(
                        &fs::read(&config.vocabulary.path)?,
                        jope_policy::CHECKPOINT_VOCABULARY_SHA256,
                    )?;
                    let mut variation = optimize::TaskVariation {
                        vocabulary: &vocabulary,
                        task: dataset.dataset.task,
                    };
                    super::experimental_proposal::verify_report(
                        config,
                        &input,
                        search,
                        receipt,
                        journal,
                        &mut variation,
                    )?;
                } else {
                    verify_producer(journal, search, receipt)?;
                }
            }
            if let Some(frozen) = frozen {
                let winner = receipt
                    .winner()
                    .context("frozen winner lacks measurement")?;
                ensure!(
                    digest(&frozen.candidate)? == digest(&winner.candidate)?
                        && digest(&frozen.selection_feedback)? == digest(&winner.feedback)?
                        && frozen.search_receipt_sha256 == digest(receipt)?,
                    "frozen winner differs from minimum train-side OOF loss"
                );
                let persisted: Value = serde_json::from_slice(&fs::read(
                    journal
                        .join("asm-search")
                        .join(&receipt.arm.id_sha256)
                        .join("frozen-recipe.json"),
                )?)?;
                ensure!(
                    persisted == serde_json::to_value(frozen)?,
                    "frozen selection evidence changed"
                );
            } else {
                ensure!(
                    receipt.winner().is_none() || optimize::pending(receipt),
                    "omitted measured frozen winner"
                );
            }
        } else {
            ensure!(frozen.is_none(), "winner without arm execution");
        }
    }
    ensure!(
        config.teaching.is_none()
            == (search.teaching_admission.is_none() && search.teaching_error.is_none())
            && !(search.teaching_admission.is_some() && search.teaching_error.is_some()),
        "teaching configuration is missing an admission or explicit failure"
    );
    if let Some(file) = &search.teaching_admission {
        super::teaching::verify_admission(plan, config, dataset, search, journal, file)?;
    }
    Ok(())
}
fn verify_receipt(
    adapter: &FitAdapter,
    journal: &Path,
    plan: &neural_search::PairInputPlan,
    receipt: &SearchReceipt,
    experiment: &jope_pipeline::experiment::Experiment,
) -> Result<()> {
    receipt.validate_population(plan)?;
    ensure!(
        receipt.deadline_unix_ms
            == super::receipt_timing::arm_deadline(
                journal,
                &receipt.arm,
                &plan.selection.outer().experiment_sha256
            )?,
        "search receipt extended or changed original clipped deadline"
    );
    let path = journal
        .join("asm-search")
        .join(&receipt.arm.id_sha256)
        .join("search-receipt.json");
    if path.exists() {
        let persisted: Value = serde_json::from_slice(&fs::read(path)?)?;
        ensure!(
            persisted == serde_json::to_value(receipt)?,
            "search receipt differs from original completion"
        );
    } else {
        ensure!(
            optimize::pending(receipt)
                && (arms_expired(plan, journal)? || original_cutoff_reached(experiment, journal)?),
            "missing final receipt before original arm deadlines"
        );
    }
    for observation in &receipt.observations {
        if observation.feedback.status != EvaluationStatus::Completed {
            continue;
        }
        let requests = plan.selection.selection_requests(
            &observation.candidate.program_sha256,
            &observation.candidate.id_sha256,
        )?;
        let mut outcomes = Vec::new();
        for request in requests {
            let mut job = adapter.job(&observation.candidate, &request)?;
            job.arm = Some(receipt.arm.clone());
            let file = artifact(
                journal
                    .join("finished")
                    .join(format!("{}.json", job.identity()?)),
            )?;
            outcomes.push(serde_json::from_slice::<WorkerOutcome>(&fs::read(
                file.path,
            )?)?);
        }
        let actual = adapter.score(&observation.candidate, &outcomes, &plan.selection)?;
        ensure!(
            digest(&actual)? == digest(&observation.feedback)?,
            "retained OOF fitness differs from fitted predictor replay"
        );
    }
    Ok(())
}
fn verify_producer(journal: &Path, search: &DatasetSearch, receipt: &SearchReceipt) -> Result<()> {
    let directory = journal
        .join("asm-contexts")
        .join(search.pair.arm(true)?.id_sha256);
    let (comparison, worker): (neural_search::Comparison, neural_search::PolicyWorker) =
        serde_json::from_slice(&fs::read(directory.join("policy-admission.json"))?)?;
    ensure!(
        digest(&comparison.input_plan)? == digest(&search.pair)?,
        "proposal admission changed pair inputs"
    );
    neural_search::verify_policy_inputs(&comparison, &worker)?;
    let outcome = &search
        .receipt
        .neural
        .as_ref()
        .context("neural execution absent")?
        .proposal_execution;
    let mut job = worker.job.clone();
    job.arm = Some(search.pair.arm(true)?);
    ensure!(
        outcome.command_status == EvaluationStatus::Completed
            && outcome.exit_code == Some(0)
            && outcome.reason.is_none()
            && outcome.job_sha256 == job.identity()?,
        "policy proposal did not complete pinned command"
    );
    let original: WorkerOutcome = serde_json::from_slice(&fs::read(
        journal
            .join("finished")
            .join(format!("{}.json", job.identity()?)),
    )?)?;
    ensure!(
        digest(outcome)? == digest(&original)?,
        "proposal execution receipt changed"
    );
    super::receipt_timing::completed_fit(journal, &job, outcome)?;
    outcome.verify_outputs()?;
    let output = outcome
        .outputs
        .iter()
        .find(|a| a.path == job.directory.join(&worker.output_name))
        .context("proposal output absent")?;
    let producer = jope_policy::producer::read_claims(&fs::read(&output.path)?, &worker.request)?;
    let evidence = receipt
        .proposal
        .as_ref()
        .context("neural provenance absent")?
        .accepted()?;
    ensure!(
        evidence.producer_job_sha256 == outcome.job_sha256
            && evidence.producer_output_sha256 == output.sha256
            && evidence.population_sha256 == producer.population_sha256(),
        "proposal provenance changed"
    );
    for observation in receipt
        .observations
        .iter()
        .filter(|o| o.candidate.generation == 0)
    {
        let slot = observation
            .candidate
            .initial_slot
            .context("generation-zero slot absent")?;
        let seed = producer
            .population()
            .candidates()
            .get(slot)
            .context("proposal slot out of range")?;
        let origin = match seed.origin() {
            jope_policy::seeds::ORIGIN_NEURAL => Origin::NeuralPolicy,
            jope_policy::seeds::ORIGIN_NEURAL_FALLBACK => Origin::NeuralRejectionFallback,
            jope_policy::seeds::ORIGIN_RANDOM => Origin::RandomExploration,
            _ => anyhow::bail!("untrusted proposal origin"),
        };
        ensure!(
            observation.candidate.genome == seed.genome()
                && observation.candidate.origin == origin
                && observation.candidate.rank_log_weight == seed.rank_log_weight(),
            "generation-zero origin/genome/rank differs from actual producer output"
        );
    }
    Ok(())
}

/// Join retained attempts before aggregating cost. A started command without a
/// finished receipt remains an unknown attempt, never an implicit zero or fit.
struct CommandInventory {
    attempts: Vec<(jope_pipeline::execution::WorkerJob, Option<WorkerOutcome>)>,
    /// No started command exists to authenticate these against an experiment.
    /// They remain explicit non-launched inventory and cannot add predictors.
    unstarted_terminal: Vec<WorkerOutcome>,
}
fn command_inventory(journal: &Path, experiment_sha256: &str) -> Result<CommandInventory> {
    let mut started = BTreeMap::new();
    for entry in fs::read_dir(journal.join("started"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|s| s != "json") {
            continue;
        }
        ensure!(entry.file_type()?.is_file(), "nonphysical started receipt");
        let record: Value = serde_json::from_slice(&fs::read(entry.path())?)?;
        let job: jope_pipeline::execution::WorkerJob =
            serde_json::from_value(record["job"].clone())?;
        let id = job.identity()?;
        ensure!(
            entry.file_name() == format!("{id}.json").as_str()
                && record["job_sha256"] == id
                && job.request.experiment_sha256 == experiment_sha256
                && job.directory == journal.join("jobs").join(job.request.identity()?),
            "started cost receipt changed experiment/job/directory"
        );
        ensure!(
            started.insert(id, job).is_none(),
            "duplicate started cost receipt"
        );
    }
    let mut finished = BTreeMap::new();
    for entry in fs::read_dir(journal.join("finished"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|s| s != "json") {
            continue;
        }
        ensure!(entry.file_type()?.is_file(), "nonphysical finished receipt");
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(entry.path())?)?;
        ensure!(
            entry.file_name() == format!("{}.json", outcome.job_sha256).as_str()
                && jope_pipeline::experiment::sha256(&outcome.job_sha256)
                && jope_pipeline::experiment::sha256(&outcome.request_sha256),
            "misnamed or invalid finished cost receipt"
        );
        outcome.verify_outputs()?;
        ensure!(
            [
                outcome.consumption.cpu_seconds,
                outcome.consumption.wall_seconds
            ]
            .into_iter()
            .flatten()
            .all(|v| v.is_finite() && v >= 0.),
            "invalid worker accounting"
        );
        if let Some(job) = started.get(&outcome.job_sha256) {
            ensure!(
                job.request.identity()? == outcome.request_sha256,
                "finished cost receipt changed request"
            );
        } else {
            ensure!(
                outcome.command_status != EvaluationStatus::Completed
                    && outcome.consumption.gate_released_unix_ms.is_none()
                    && outcome.exit_code.is_none()
                    && outcome.signal.is_none()
                    && outcome.outputs.is_empty()
                    && outcome.consumption.cpu_seconds.is_none_or(|v| v == 0.)
                    && outcome.consumption.wall_seconds.is_none_or(|v| v == 0.),
                "launched command lost started cost receipt"
            );
        }
        ensure!(
            finished
                .insert(outcome.job_sha256.clone(), outcome)
                .is_none(),
            "duplicate finished cost receipt"
        );
    }
    let attempts = started
        .into_iter()
        .map(|(id, job)| (job, finished.remove(&id)))
        .collect();
    Ok(CommandInventory {
        attempts,
        unstarted_terminal: finished.into_values().collect(),
    })
}

/// Include incomplete OOF candidates: their successful constituent fits still
/// cost resources and produce predictors even if five-fold selection failed.
/// Proposal/context commands are accounted separately from fitted predictors.
pub fn accounting(plan: &CampaignPlan, journal: &Path) -> Result<Value> {
    accounting_for_experiment(&plan.experiment.identity()?, journal)
}
fn accounting_for_experiment(experiment_sha256: &str, journal: &Path) -> Result<Value> {
    use std::collections::BTreeSet;
    let inventory = command_inventory(journal, experiment_sha256)?;
    let missing_finished = inventory
        .attempts
        .iter()
        .filter(|(_, outcome)| outcome.is_none())
        .count();
    let unstarted_terminal = inventory.unstarted_terminal.iter().map(|outcome|
        json!({"job_sha256":outcome.job_sha256,"request_sha256":outcome.request_sha256,
            "status":outcome.command_status,"reason":outcome.reason,
            "cpu_seconds":outcome.consumption.cpu_seconds,"wall_seconds":outcome.consumption.wall_seconds,
            "basis":"no started receipt; inventory only, not an authenticated launched command or fitted predictor"})
    ).collect::<Vec<_>>();
    let mut cpu = 0.;
    let mut wall = 0.;
    let mut unknown_cpu = 0usize;
    let mut unknown_wall = 0usize;
    let mut commands = 0usize;
    let mut fitting_outputs = 0usize;
    let mut predictors = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    let mut diagnostic_commands = 0usize;
    let mut fitted_export_commands = 0usize;
    for (job, outcome) in inventory.attempts {
        commands += 1;
        if job.request.stage == EvaluationStage::Diagnostic {
            if job
                .command
                .args
                .iter()
                .any(|arg| arg == "fitted-export-worker")
            {
                fitted_export_commands += 1;
            } else {
                diagnostic_commands += 1;
            }
        }
        let Some(outcome) = outcome else {
            unknown_cpu += 1;
            unknown_wall += 1;
            continue;
        };
        match outcome.consumption.cpu_seconds {
            Some(v) => cpu += v,
            None => unknown_cpu += 1,
        }
        match outcome.consumption.wall_seconds {
            Some(v) => wall += v,
            None => unknown_wall += 1,
        }
        if job.request.stage == EvaluationStage::Diagnostic {
            continue;
        }
        if outcome.command_status != EvaluationStatus::Completed {
            continue;
        }
        let output = outcome
            .outputs
            .iter()
            .find(|a| a.path == job.directory.join("output.json"))
            .context("fitted output absent")?;
        let output: Value = serde_json::from_slice(&fs::read(&output.path)?)?;
        let fitted = match output.get("schema").and_then(Value::as_str) {
            Some("jope.panel-synthesized-worker-output/v1") => {
                let value: super::synthesized::SynthesizedWorkerOutput =
                    serde_json::from_value(output)?;
                value.artifact.validate()?;
                ensure!(
                    value.artifact_sha256 == jope_pipeline::sha256_hex(&value.artifact.bytes()?),
                    "generated cost artifact changed"
                );
                Some(value.artifact_sha256)
            }
            Some("jope.panel-rust-worker-output/v1") => {
                let value: super::worker::RustWorkerOutput = serde_json::from_value(output)?;
                value.artifact.validate()?;
                ensure!(
                    value.artifact_sha256 == jope_pipeline::sha256_hex(&value.artifact.bytes()?),
                    "native cost artifact changed"
                );
                Some(value.artifact_sha256)
            }
            _ => {
                if output.get("contract").and_then(Value::as_str)
                    == Some(super::methods::ADAPTER_CONTRACT)
                {
                    let hash = output["fitted"]["sha256"]
                        .as_str()
                        .context("external fitted checksum absent")?;
                    ensure!(
                        output["fitted"]["reload_verified"] == true
                            && outcome
                                .outputs
                                .iter()
                                .any(|a| a.path == job.directory.join("output.json.fitted.pkl")
                                    && a.sha256 == hash),
                        "external cost artifact absent"
                    );
                    Some(hash.to_owned())
                } else {
                    None
                }
            }
        };
        if let Some(hash) = fitted {
            ensure!(
                jope_pipeline::experiment::sha256(&hash),
                "invalid fitted output identity"
            );
            predictors.insert(hash);
            recipes.insert(job.request.recipe_sha256);
            fitting_outputs += 1;
        }
    }
    Ok(
        json!({"worker_command_receipts":commands,"context_or_proposal_commands":diagnostic_commands,"fitted_export_commands":fitted_export_commands,"completed_fitted_output_commands":fitting_outputs,
        "unique_fitted_predictors":predictors.len(),"unique_reusable_recipes":recipes.len(),"known_worker_cpu_seconds":cpu,"unknown_cpu_attempts":unknown_cpu,
        "known_sum_worker_wall_seconds":wall,"unknown_wall_attempts":unknown_wall,
        "started_commands_missing_finished_receipts":missing_finished,
        "unstarted_terminal_inventory":unstarted_terminal,
        "cpu_seconds_per_unique_predictor_known":(!predictors.is_empty()).then(||cpu/predictors.len() as f64),
        "cpu_seconds_per_unique_recipe_known":(!recipes.is_empty()).then(||cpu/recipes.len() as f64),
        "basis":"union of retained started and finished commands; started-only attempts have unknown CPU/wall and no fitted-output claim; unstarted terminal inventory is separate; unique complete fitted artifact bytes; no behavioral-equivalence claim",
        "controller_cpu_seconds":null,"gpu_seconds":null,"one_hour_ten_million_token_scenario":{"supplied_seconds":3600,"supplied_tokens":10000000,"token_usage_measured":null,"token_cost_comparison":"not_comparable_without_actual_token_and_price_receipts"}}),
    )
}

fn original_cutoff_reached(
    experiment: &jope_pipeline::experiment::Experiment,
    journal: &Path,
) -> Result<bool> {
    let window: jope_pipeline::experiment::ExecutionWindow =
        serde_json::from_slice(&fs::read(journal.join("window.json"))?)?;
    window.validate(experiment)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_millis() as u64;
    Ok(now
        >= window
            .deadline_unix_ms
            .saturating_sub(experiment.resources.reporting_seconds * 1000))
}

#[cfg(test)]
mod accounting_tests {
    use super::*;
    use jope_pipeline::{
        execution::{ObservedConsumption, WorkerCommand, WorkerJob, EXECUTION_SCHEMA},
        experiment::{EvaluationRequest, Metric, REQUEST_SCHEMA},
    };
    use std::path::PathBuf;

    fn scratch() -> tempfile::TempDir {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/tmp");
        fs::create_dir_all(&root).unwrap();
        let directory = tempfile::Builder::new()
            .prefix("accounting-union-")
            .tempdir_in(fs::canonicalize(root).unwrap())
            .unwrap();
        fs::create_dir(directory.path().join("started")).unwrap();
        fs::create_dir(directory.path().join("finished")).unwrap();
        directory
    }
    fn job(journal: &Path, seed: u64, stage: EvaluationStage) -> WorkerJob {
        let request = EvaluationRequest {
            schema: REQUEST_SCHEMA.into(),
            experiment_sha256: "a".repeat(64),
            dataset_sha256: "b".repeat(64),
            split_sha256: "c".repeat(64),
            fitting_rows_sha256: "d".repeat(64),
            scoring_rows_sha256: "e".repeat(64),
            fitting_rows: 10,
            scoring_rows: 2,
            recipe_sha256: "f".repeat(64),
            evaluator_sha256: "1".repeat(64),
            checkpoint_sha256: None,
            task: jope_datasets::catalog::Task::Regression,
            metric: Metric::Rmse,
            classes: vec![],
            fold: 0,
            stage,
            seed,
            numerical_policy: "synthetic-metadata-only".into(),
        };
        WorkerJob {
            directory: journal.join("jobs").join(request.identity().unwrap()),
            request,
            command: WorkerCommand {
                program: jope_pipeline::execution::Artifact {
                    path: "/synthetic/not-executed".into(),
                    sha256: "2".repeat(64),
                },
                args: vec![],
                inputs: vec![],
                environment: Default::default(),
            },
            outputs: vec!["output.json".into()],
            arm: None,
        }
    }
    fn started(journal: &Path, job: &WorkerJob) -> PathBuf {
        let id = job.identity().unwrap();
        let path = journal.join("started").join(format!("{id}.json"));
        fs::write(
            &path,
            serde_json::to_vec(&json!({
                "job_sha256":id,"job":job,"started_unix_ms":100,"fit_deadline_unix_ms":120,
                "cgroup":"/synthetic/not-executed"
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }
    fn failed(job: &WorkerJob) -> WorkerOutcome {
        WorkerOutcome {
            schema: EXECUTION_SCHEMA.into(),
            job_sha256: job.identity().unwrap(),
            request_sha256: job.request.identity().unwrap(),
            command_status: EvaluationStatus::Failed,
            reason: Some("synthetic retained failure".into()),
            exit_code: Some(1),
            signal: None,
            outputs: vec![],
            stdout: None,
            stderr: None,
            consumption: ObservedConsumption {
                gate_released_unix_ms: Some(100),
                observed_exit_unix_ms: Some(101),
                cpu_seconds: Some(1.),
                wall_seconds: Some(2.),
                accounting_reason: "synthetic known cost fixture".into(),
                ..Default::default()
            },
        }
    }
    fn finished(journal: &Path, outcome: &WorkerOutcome) -> PathBuf {
        let path = journal
            .join("finished")
            .join(format!("{}.json", outcome.job_sha256));
        fs::write(&path, serde_json::to_vec(outcome).unwrap()).unwrap();
        path
    }
    #[test]
    fn started_only_command_is_unknown_cost_without_a_fitted_predictor() -> Result<()> {
        let scratch = scratch();
        let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
        started(scratch.path(), &job);
        let value = accounting_for_experiment(&"a".repeat(64), scratch.path())?;
        assert_eq!(value["worker_command_receipts"], 1);
        assert_eq!(value["started_commands_missing_finished_receipts"], 1);
        assert_eq!(value["unknown_cpu_attempts"], 1);
        assert_eq!(value["unknown_wall_attempts"], 1);
        assert_eq!(value["unique_fitted_predictors"], 0);
        assert_eq!(value["completed_fitted_output_commands"], 0);
        assert!(value["cpu_seconds_per_unique_predictor_known"].is_null());
        Ok(())
    }
    #[test]
    fn matched_failure_missing_final_and_unstarted_refusal_are_counted_separately() -> Result<()> {
        let scratch = scratch();
        let known = job(scratch.path(), 1, EvaluationStage::Diagnostic);
        let unknown = job(scratch.path(), 2, EvaluationStage::TrainOof);
        let refused = job(scratch.path(), 3, EvaluationStage::TrainOof);
        started(scratch.path(), &known);
        started(scratch.path(), &unknown);
        finished(scratch.path(), &failed(&known));
        let mut outcome = failed(&refused);
        outcome.command_status = EvaluationStatus::BudgetExhausted;
        outcome.exit_code = None;
        outcome.consumption = ObservedConsumption {
            accounting_reason: "pre-gate setup cost unknown".into(),
            ..Default::default()
        };
        finished(scratch.path(), &outcome);
        let value = accounting_for_experiment(&"a".repeat(64), scratch.path())?;
        assert_eq!(value["worker_command_receipts"], 2);
        assert_eq!(value["context_or_proposal_commands"], 1);
        assert_eq!(value["known_worker_cpu_seconds"], 1.);
        assert_eq!(value["known_sum_worker_wall_seconds"], 2.);
        assert_eq!(value["unknown_cpu_attempts"], 1);
        assert_eq!(value["unknown_wall_attempts"], 1);
        assert_eq!(
            value["unstarted_terminal_inventory"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert!(value["unstarted_terminal_inventory"][0]["cpu_seconds"].is_null());
        assert_eq!(value["unique_fitted_predictors"], 0);
        Ok(())
    }
    #[test]
    fn accounting_rejects_mismatched_started_identity_or_experiment() -> Result<()> {
        let scratch = scratch();
        let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
        let path = started(scratch.path(), &job);
        assert!(accounting_for_experiment(&"b".repeat(64), scratch.path()).is_err());
        let mut record: Value = serde_json::from_slice(&fs::read(&path)?)?;
        record["job_sha256"] = json!("3".repeat(64));
        fs::write(&path, serde_json::to_vec(&record)?)?;
        assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
        Ok(())
    }
    #[test]
    fn accounting_rejects_wrong_request_and_missing_started_for_launched_outcome() -> Result<()> {
        let scratch = scratch();
        let job = job(scratch.path(), 1, EvaluationStage::TrainOof);
        let path = started(scratch.path(), &job);
        let mut outcome = failed(&job);
        outcome.request_sha256 = "4".repeat(64);
        finished(scratch.path(), &outcome);
        assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
        outcome.request_sha256 = job.request.identity()?;
        finished(scratch.path(), &outcome);
        accounting_for_experiment(&"a".repeat(64), scratch.path())?;
        fs::remove_file(path)?;
        assert!(accounting_for_experiment(&"a".repeat(64), scratch.path()).is_err());
        Ok(())
    }
}
