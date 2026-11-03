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
/// Development hold rows are scored only for a settled attachment whose caller
/// asked for screen scores. A journal that defers scoring to `report` is refused
/// here, before any hold label is read, whatever the caller passed.
pub(crate) fn screen_observations(
    ready: bool,
    score_screen: bool,
    fits: &[campaign::PendingFit],
    outcomes: Vec<WorkerOutcome>,
    data: &[DevelopmentData],
    journal: &Path,
) -> Result<Vec<Observation>> {
    if !(ready && score_screen) {
        return Ok(vec![]);
    }
    ensure!(
        !journal
            .join(super::campaign_cli::DEFERRAL_MARKER)
            .try_exists()?,
        "journal defers development hold scoring to report"
    );
    campaign::score_batch(fits, outcomes, data, journal)
}
pub fn attach(
    plan: &CampaignPlan,
    data: &[DevelopmentData],
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
    baseline: &[Observation],
) -> Result<Attachment> {
    attach_with(plan, data, journal, pool, cancel, baseline, true)
}
/// With `score_screen` false the refits still run and retain their `finished/`
/// receipts, but no development hold row is scored and no observation is written.
pub fn attach_with(
    plan: &CampaignPlan,
    data: &[DevelopmentData],
    journal: &Path,
    pool: &mut ExecutionPool,
    cancel: &AtomicBool,
    baseline: &[Observation],
    score_screen: bool,
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
    screen.extend(screen_observations(
        ready,
        score_screen,
        &fits,
        outcomes,
        data,
        journal,
    )?);
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
mod verification;
use verification::verify_search;

mod accounting;
pub use accounting::accounting;

#[cfg(test)]
mod accounting_tests;
