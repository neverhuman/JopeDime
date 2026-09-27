//! A bounded rank's training run, derived only from its admitted plan (W7-D2).
//!
//! Bounded ranks train. A rank whose frozen plan selects training ([`TrainingSelection`])
//! receives the admitted plan, its frozen launch record and the admitted NCCL bootstrap of its
//! verified full-group admission. [`prepare`] checks them against each other and writes the
//! run's `run.json` from the plan alone, with the execution partition the plan selects (v2 for
//! three ranks). The single-host `train_controlled` launcher and its v1 `run.json` are not
//! involved and are unchanged.
//!
//! Everything here happens before any device work and is checked on the CPU:
//! * the plan must select training, and the bootstrap must be this plan's, attempt's, rank's
//!   and world's;
//! * the rank's own copy of the corpus index (and parent checkpoint state) must be exactly the
//!   planned bytes, and the parent's optimizer step must be the plan's committed start;
//! * `run.json` is published without replacement: an existing one must be byte-identical (the
//!   other ranks on a host write the same bytes);
//! * each dispatch gets a fresh launch directory. A launch directory that already exists is
//!   refused, and one holding a shared `nccl.id` (the `train_controlled` rendezvous) is refused
//!   by name.
//!
//! [`TrainingSelection`]: super::contract::TrainingSelection
use super::{
    contract::{BoundedRunContract, ClosureEntry},
    launch::{self, AdmittedBootstrap, LaunchRecord},
    rendezvous::publish_immutable,
};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

pub const TRAINING_RUN_SCHEMA: &str = "jope.bounded_training_run.v1";
/// Where a host keeps its bounded training runs, under its input root.
pub const RUNS_DIRECTORY: &str = "bounded-runs";

/// Everything one bounded rank trains from, all derived from its admitted plan.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundedTraining {
    /// The `run.json` written from the plan.
    pub spec: Value,
    pub global_rank: usize,
    pub world: usize,
    pub microbatch: usize,
    pub global_batch: usize,
    pub steps: usize,
    pub checkpoint_steps: usize,
    pub input_root: PathBuf,
    /// Relative to `input_root`.
    pub corpus: PathBuf,
    /// Relative to `input_root`.
    pub initialize: Option<PathBuf>,
    pub run_dir: PathBuf,
    pub launch_dir: PathBuf,
}

fn closure_entry<'a>(contract: &'a BoundedRunContract, path: &Path) -> Result<&'a ClosureEntry> {
    contract
        .artifact_closure
        .iter()
        .find(|e| e.path == path)
        .with_context(|| format!("the plan's input closure has no `{}`", path.display()))
}

/// The run's `run.json`, derived from the frozen plan alone.
pub fn training_run_spec(contract: &BoundedRunContract, plan_sha256: &str) -> Result<Value> {
    let t = contract
        .training
        .as_ref()
        .context("the plan selects no training; a bounded training rank trains only a plan that selects its partition")?;
    let partition = contract
        .execution_partition()?
        .context("the plan selects no execution partition")?;
    let corpus = closure_entry(contract, &t.corpus.join("index.json"))?;
    let initialization = t
        .initialize
        .as_ref()
        .map(|p| closure_entry(contract, &p.join("state.json")))
        .transpose()?;
    let c = &contract.counters;
    let target = c
        .committed_updates_at_start
        .checked_add(c.planned_updates)
        .context("step window overflow")?;
    let hosts: BTreeMap<&str, Value> = contract
        .participants
        .iter()
        .map(|p| {
            (
                p.host.host.as_str(),
                json!({
                    "machine_id": p.host.machine_id,
                    "boot_id": p.host.boot_id,
                    "source_commit": p.host.source_commit,
                    "executable_sha256": p.host.executable_sha256,
                }),
            )
        })
        .collect();
    Ok(json!({
        "schema": TRAINING_RUN_SCHEMA,
        "run": contract.run,
        "attempt": contract.attempt,
        "plan_sha256": plan_sha256,
        "world": contract.world_size(),
        "microbatch": t.microbatch,
        "global_batch": t.global_batch,
        "execution_partition": partition,
        "corpus": {"path": t.corpus, "index_sha256": corpus.sha256},
        "corpus_sha256": corpus.sha256,
        "initialization": initialization.map(|e| json!({"path": t.initialize, "state_sha256": e.sha256})),
        "initialization_sha256": initialization.map(|e| e.sha256.clone()),
        "step_window": {
            "initial_optimizer_step": c.committed_updates_at_start,
            "requested_step_increment": c.planned_updates,
            "target_optimizer_step": target,
        },
        "checkpoint_steps": t.checkpoint_steps,
        "deadline_unix_ms": contract.original_deadline_unix_ms,
        "hosts": hosts,
        "assignment": "frozen_plan",
        "communicator": "admitted_bootstrap",
    }))
}

/// Read a planned input from this host's copy (bounded by its planned size, no symlink) and
/// check its bytes.
fn read_planned(root: &Path, entry: &ClosureEntry) -> Result<Vec<u8>> {
    let bytes = launch::read_bounded(&root.join(&entry.path), entry.bytes)
        .with_context(|| format!("planned input `{}`", entry.path.display()))?;
    ensure!(
        bytes.len() as u64 == entry.bytes && crate::manifest::hash(&bytes) == entry.sha256,
        "this host's `{}` is not the planned bytes",
        entry.path.display()
    );
    Ok(bytes)
}

/// Check the plan, the launch record and the admitted bootstrap against each other, verify
/// this host's planned inputs, write `run.json` from the plan and create this dispatch's
/// fresh launch directory. Nothing touches a device.
pub fn prepare(
    contract: &BoundedRunContract,
    plan_sha256: &str,
    record: &LaunchRecord,
    bootstrap: &AdmittedBootstrap,
    input_root: &Path,
) -> Result<BoundedTraining> {
    launch::verify_launch_record(contract, plan_sha256, record)?;
    ensure!(
        bootstrap.plan_sha256() == plan_sha256
            && bootstrap.attempt() == contract.attempt
            && bootstrap.global_rank() == record.global_rank
            && bootstrap.world_size() == record.world_size,
        "the admitted bootstrap and the plan disagree: the bootstrap is plan {} attempt {} rank {} of {}, the launch is plan {plan_sha256} attempt {} rank {} of {}",
        bootstrap.plan_sha256(),
        bootstrap.attempt(),
        bootstrap.global_rank(),
        bootstrap.world_size(),
        contract.attempt,
        record.global_rank,
        record.world_size
    );
    let spec = training_run_spec(contract, plan_sha256)?;
    let t = contract.training.as_ref().context("training selection")?;
    ensure!(input_root.is_absolute(), "the input root must be absolute");
    read_planned(
        input_root,
        closure_entry(contract, &t.corpus.join("index.json"))?,
    )?;
    if let Some(initialize) = &t.initialize {
        let state = read_planned(
            input_root,
            closure_entry(contract, &initialize.join("state.json"))?,
        )?;
        let state: Value = serde_json::from_slice(&state).context("parent checkpoint state")?;
        ensure!(
            state["step"].as_u64() == Some(contract.counters.committed_updates_at_start),
            "the parent checkpoint is at optimizer step {}, not the plan's committed start {}",
            state["step"],
            contract.counters.committed_updates_at_start
        );
    }
    let run_dir = input_root
        .join(RUNS_DIRECTORY)
        .join(&contract.run)
        .join(format!("attempt-{}", contract.attempt));
    fs::create_dir_all(&run_dir)?;
    ensure!(
        !run_dir.join("nccl.id").try_exists()?,
        "a shared nccl.id in the bounded run directory is refused; bounded ranks take their communicator only from the admission"
    );
    publish_immutable(&run_dir, Path::new("run.json"), &serde_json::to_vec(&spec)?)
        .context("the bounded run directory already holds another plan's run.json")?;
    let launch_dir = run_dir.join(format!("launch-{}", record.dispatch_id));
    if launch_dir.try_exists()? {
        ensure!(
            !launch_dir.join("nccl.id").try_exists()?,
            "a shared nccl.id (the train_controlled rendezvous) in launch {} is refused; bounded ranks take their communicator only from the admission",
            launch_dir.display()
        );
        anyhow::bail!(
            "launch {} already exists; a bounded rank never reuses a launch directory",
            launch_dir.display()
        );
    }
    fs::create_dir(&launch_dir)?;
    let usize_of = |v: u64, what: &str| -> Result<usize> {
        usize::try_from(v).with_context(|| format!("{what} does not fit this host"))
    };
    Ok(BoundedTraining {
        spec,
        global_rank: record.global_rank as usize,
        world: record.world_size as usize,
        microbatch: t.microbatch as usize,
        global_batch: t.global_batch as usize,
        steps: usize_of(contract.counters.planned_updates, "planned updates")?,
        checkpoint_steps: usize_of(t.checkpoint_steps, "checkpoint steps")?,
        input_root: input_root.to_path_buf(),
        corpus: t.corpus.clone(),
        initialize: t.initialize.clone(),
        run_dir,
        launch_dir,
    })
}
