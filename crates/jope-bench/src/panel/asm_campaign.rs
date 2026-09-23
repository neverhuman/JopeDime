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
            Some(search.pair.proposal.checkpoint_sha256.clone()),
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
                verify_producer(journal, search, receipt)?;
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
        .context("neural provenance absent")?;
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

/// Include incomplete OOF candidates: their successful constituent fits still
/// cost resources and produce predictors even if five-fold selection failed.
/// Proposal/context commands are accounted separately from fitted predictors.
pub fn accounting(plan: &CampaignPlan, journal: &Path) -> Result<Value> {
    use std::collections::BTreeSet;
    let mut cpu = 0.;
    let mut wall = 0.;
    let mut unknown_cpu = 0usize;
    let mut unknown_wall = 0usize;
    let mut commands = 0usize;
    let mut fitting_outputs = 0usize;
    let mut predictors = BTreeSet::new();
    let mut recipes = BTreeSet::new();
    let mut diagnostic_commands = 0usize;
    for entry in fs::read_dir(journal.join("finished"))? {
        let entry = entry?;
        if entry.path().extension().is_none_or(|s| s != "json") {
            continue;
        }
        let outcome: WorkerOutcome = serde_json::from_slice(&fs::read(entry.path())?)?;
        ensure!(
            entry.file_name() == format!("{}.json", outcome.job_sha256).as_str(),
            "misnamed finished receipt"
        );
        let started = journal
            .join("started")
            .join(format!("{}.json", outcome.job_sha256));
        // Refused admissions have no execution/predictor, but explicit zero cost.
        if !started.exists() {
            ensure!(
                outcome.consumption.gate_released_unix_ms.is_none(),
                "launched command lost started receipt"
            );
            continue;
        }
        let record: Value = serde_json::from_slice(&fs::read(started)?)?;
        let job: jope_pipeline::execution::WorkerJob =
            serde_json::from_value(record["job"].clone())?;
        ensure!(
            job.identity()? == outcome.job_sha256
                && job.request.experiment_sha256 == plan.experiment.identity()?,
            "cost receipt changed experiment/job"
        );
        commands += 1;
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
        match outcome.consumption.cpu_seconds {
            Some(v) => cpu += v,
            None => unknown_cpu += 1,
        }
        match outcome.consumption.wall_seconds {
            Some(v) => wall += v,
            None => unknown_wall += 1,
        }
        if job.request.stage == EvaluationStage::Diagnostic {
            diagnostic_commands += 1;
            continue;
        }
        if outcome.command_status != EvaluationStatus::Completed {
            continue;
        }
        outcome.verify_outputs()?;
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
        json!({"worker_command_receipts":commands,"context_or_proposal_commands":diagnostic_commands,"completed_fitted_output_commands":fitting_outputs,
        "unique_fitted_predictors":predictors.len(),"unique_reusable_recipes":recipes.len(),"known_worker_cpu_seconds":cpu,"unknown_cpu_attempts":unknown_cpu,
        "known_sum_worker_wall_seconds":wall,"unknown_wall_attempts":unknown_wall,
        "cpu_seconds_per_unique_predictor_known":(!predictors.is_empty()).then(||cpu/predictors.len() as f64),
        "cpu_seconds_per_unique_recipe_known":(!recipes.is_empty()).then(||cpu/recipes.len() as f64),
        "basis":"all retained started/finished commands, including incomplete OOF candidates; unique complete fitted artifact bytes; no behavioral-equivalence claim",
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
