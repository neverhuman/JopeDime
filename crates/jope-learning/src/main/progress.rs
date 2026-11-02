use super::{print_json, Common};
use anyhow::{bail, Context, Result};
use clap::Subcommand;
use jope_learning::mirror::{mirror_url, Mirror};
use jope_learning::progress::cadence::Clock;
use jope_learning::progress::inventory::{RunManifest, WorkSpec};
use jope_learning::progress::product::{Config, Driver, Step, SystemClock};
use jope_learning::progress::publish::{reusable, Publication};
use jope_learning::progress::rebuild::{read_summary, summarize};
use jope_learning::progress::reconcile;
use jope_learning::progress::remote::open_publishing_store;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(Subcommand)]
pub(super) enum ProgressCmd {
    /// Print the outbox limits and cadence in effect, and their approval state.
    Config {
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Freeze a run's planned census (its manifest) and record it for publication.
    Init {
        #[command(flatten)]
        run: RunArgs,
        /// Full git commit of the source the run executes.
        #[arg(long)]
        source_commit: String,
        #[arg(long, default_value_t = 3)]
        max_attempts: u32,
        /// A planned unit, `STAGE:SUBJECT[:INPUT_SHA256,...]`; repeatable.
        #[arg(long = "unit")]
        units: Vec<String>,
    },
    /// A published engine corpus directory: appended, then flushed at once (corpus freeze).
    Corpus {
        #[command(flatten)]
        run: RunArgs,
        dir: PathBuf,
    },
    /// The checkpoint an engine training run's latest.json publishes, under the checkpoint
    /// cadence: mapped as the next W8 update when one is due, otherwise counted.
    Checkpoint {
        #[command(flatten)]
        run: RunArgs,
        /// The engine root the training run and its checkpoint paths are relative to.
        #[arg(long)]
        engine_root: PathBuf,
        /// The training run directory, relative to --engine-root.
        #[arg(long)]
        training_run: PathBuf,
        /// The published corpus the run trains on.
        #[arg(long)]
        corpus: PathBuf,
        /// The optimizer step the run started from.
        #[arg(long, default_value_t = 0)]
        start_step: u64,
        /// Controlled shutdown: checkpoint any counted step not yet in a W8 checkpoint, flush.
        #[arg(long)]
        shutdown: bool,
    },
    /// Attempt 1 of a planned unit was scheduled.
    Schedule {
        #[command(flatten)]
        run: RunArgs,
        #[arg(long)]
        unit: String,
    },
    /// A retry of the previous attempt's recorded failure was scheduled.
    Retry {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        attempt: AttemptArgs,
    },
    /// A completed checkpoint-recovery receipt: the attempt's ranks were reaped.
    Recovery {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        attempt: AttemptArgs,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// A no-replace engine publication is the attempt's result.
    Result {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        attempt: AttemptArgs,
        #[arg(long)]
        engine_root: PathBuf,
        /// The published artifact, relative to --engine-root.
        #[arg(long)]
        artifact: PathBuf,
        /// The digest the engine published it under.
        #[arg(long)]
        sha256: String,
    },
    /// Link the completed attempt to its decision in this host's HWM ledger.
    Decision {
        #[command(flatten)]
        run: RunArgs,
        #[command(flatten)]
        attempt: AttemptArgs,
        /// The ledger batch, `yyyy/mm/<batch>.jsonl` relative to the HWM ledger.
        #[arg(long)]
        batch: String,
        #[arg(long)]
        seq: u64,
    },
    /// Flush if the cadence says so; a new process flushes whatever it finds owed.
    Flush {
        #[command(flatten)]
        run: RunArgs,
    },
    /// Follow an engine training run on the wall clock: count and map its published
    /// checkpoints under the checkpoint cadence and flush under the flush cadence, until the run
    /// writes complete.json; then a controlled shutdown.
    Follow {
        #[command(flatten)]
        run: RunArgs,
        #[arg(long)]
        engine_root: PathBuf,
        /// The training run directory, relative to --engine-root.
        #[arg(long)]
        training_run: PathBuf,
        #[arg(long)]
        corpus: PathBuf,
        #[arg(long, default_value_t = 0)]
        start_step: u64,
        /// Milliseconds between observations.
        #[arg(long, default_value_t = 5_000)]
        poll_ms: u64,
        /// Stop after this many observations even if the run has not completed (0: no bound).
        #[arg(long, default_value_t = 0)]
        max_polls: u64,
    },
    /// Read-only: the committed head and the census and summary derived from it.
    Status {
        #[arg(long)]
        run: String,
    },
    /// Reconcile a publication that lost a race: adopt the winner's committed closure, then
    /// commit this host's remaining work on top of it. Read-only unless --apply; exit 1 when the
    /// rebased commit would not pass the receiving checks.
    Reconcile {
        #[command(flatten)]
        run: RunArgs,
        #[arg(long)]
        apply: bool,
    },
}

/// The store progress is published to: only the one named by `--store`/`JOPE_LEARNING_URL`.
fn progress_store(common: &Common) -> Result<Mirror> {
    match mirror_url(common.store.clone()) {
        Some(url) => open_publishing_store(&url),
        None => bail!(
            "progress publishes only to an explicitly named store: pass --store or set \
             JOPE_LEARNING_URL. Nothing was written"
        ),
    }
}

fn progress_config(path: Option<&PathBuf>) -> Result<Config> {
    path.map_or_else(|| Ok(Config::adopted()), |p| Config::load(p))
}

fn parse_unit(spec: &str) -> Result<WorkSpec> {
    let mut parts = spec.splitn(3, ':');
    let (Some(stage), Some(subject)) = (parts.next(), parts.next()) else {
        bail!("--unit {spec}: expected STAGE:SUBJECT[:INPUT_SHA256,...]");
    };
    let mut inputs: Vec<String> = parts
        .next()
        .map(|i| i.split(',').map(str::to_string).collect())
        .unwrap_or_default();
    inputs.sort();
    Ok(WorkSpec {
        stage: stage.into(),
        subject: subject.into(),
        inputs,
    })
}

fn print_step(driver: &Driver, step: &Step) -> Result<()> {
    print_json(&json!({
        "run": driver.manifest().value().run_id,
        "manifest": driver.manifest().digest(),
        "step": step,
        "pending": driver.outbox().pending_entries(),
        "config": driver.config().report(),
    }))
}

fn open_driver<'r>(state: &State, run: &RunArgs, remote: &'r Mirror) -> Result<Driver<'r>> {
    let config = progress_config(run.config.as_ref())?;
    Driver::open(
        state,
        remote,
        config,
        &run.run,
        &run.publisher,
        SystemClock.now_ms(),
    )
}

pub(super) fn run_progress(state: &State, common: &Common, cmd: ProgressCmd) -> Result<bool> {
    let clock = SystemClock;
    match cmd {
        ProgressCmd::Config { config } => {
            print_json(&progress_config(config.as_ref())?.report())?;
            Ok(true)
        }
        ProgressCmd::Init {
            run,
            source_commit,
            max_attempts,
            units,
        } => {
            let remote = progress_store(common)?;
            let specs = units
                .iter()
                .map(|u| parse_unit(u))
                .collect::<Result<Vec<_>>>()?;
            let manifest = RunManifest::new(
                &run.run,
                &run.publisher,
                &source_commit,
                max_attempts,
                specs,
            )?;
            let units: Vec<Value> = manifest
                .units
                .iter()
                .map(|u| json!({"work_id": u.work_id, "stage": u.spec.stage, "subject": u.spec.subject}))
                .collect();
            let config = progress_config(run.config.as_ref())?;
            let mut driver = Driver::init(
                state,
                &remote,
                config,
                manifest,
                &run.publisher,
                clock.now_ms(),
            )?;
            let step = driver.tick(&clock)?;
            print_json(&json!({
                "run": run.run,
                "manifest": driver.manifest().digest(),
                "units": units,
                "step": step,
                "config": driver.config().report(),
            }))?;
            Ok(true)
        }
        ProgressCmd::Corpus { run, dir } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.corpus(&dir, &clock)?;
            print_step(&driver, &step)?;
            Ok(step.flushed)
        }
        ProgressCmd::Checkpoint {
            run,
            engine_root,
            training_run,
            corpus,
            start_step,
            shutdown,
        } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = if shutdown {
                driver.shutdown(&engine_root, &training_run, &corpus, start_step, &clock)?
            } else {
                driver.checkpoint(&engine_root, &training_run, &corpus, start_step, &clock)?
            };
            print_step(&driver, &step)?;
            Ok(step.publication.is_none() || step.flushed)
        }
        ProgressCmd::Schedule { run, unit } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.schedule(&unit)?;
            print_step(&driver, &step)?;
            Ok(true)
        }
        ProgressCmd::Retry { run, attempt } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.retry(&attempt.unit, attempt.attempt)?;
            print_step(&driver, &step)?;
            Ok(true)
        }
        ProgressCmd::Recovery {
            run,
            attempt,
            receipt,
        } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.recovery(&attempt.unit, attempt.attempt, &receipt)?;
            print_step(&driver, &step)?;
            Ok(true)
        }
        ProgressCmd::Result {
            run,
            attempt,
            engine_root,
            artifact,
            sha256,
        } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.result(
                &attempt.unit,
                attempt.attempt,
                &engine_root,
                &artifact,
                &sha256,
            )?;
            print_step(&driver, &step)?;
            Ok(true)
        }
        ProgressCmd::Decision {
            run,
            attempt,
            batch,
            seq,
        } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.decision(&attempt.unit, attempt.attempt, &batch, seq, &clock)?;
            print_step(&driver, &step)?;
            Ok(true)
        }
        ProgressCmd::Flush { run } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let step = driver.tick(&clock)?;
            print_step(&driver, &step)?;
            Ok(step.staged == 0 && driver.outbox().pending_entries() == 0)
        }
        ProgressCmd::Follow {
            run,
            engine_root,
            training_run,
            corpus,
            start_step,
            poll_ms,
            max_polls,
        } => {
            let remote = progress_store(common)?;
            let mut driver = open_driver(state, &run, &remote)?;
            let (mut polls, mut events) = (0u64, Vec::new());
            let mut completed;
            loop {
                completed = engine_root
                    .join(&training_run)
                    .join("complete.json")
                    .is_file();
                let observed = if completed {
                    driver.shutdown(&engine_root, &training_run, &corpus, start_step, &clock)?
                } else {
                    driver.checkpoint(&engine_root, &training_run, &corpus, start_step, &clock)?
                };
                let flushed = driver.tick(&clock)?;
                for step in [observed, flushed] {
                    if step.publication.is_some()
                        || matches!(
                            step.checkpoint,
                            Some(jope_learning::progress::product::CheckpointStep::Mapped { .. })
                        )
                    {
                        events.push(step);
                    }
                }
                polls += 1;
                if completed || (max_polls > 0 && polls >= max_polls) {
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(poll_ms));
            }
            let owed = driver.staged() + driver.outbox().pending_entries();
            print_json(&json!({
                "run": run.run,
                "polls": polls,
                "completed": completed,
                "events": events,
                "head": driver.head(),
                "owed": owed,
                "config": driver.config().report(),
            }))?;
            Ok(owed == 0)
        }
        ProgressCmd::Reconcile { run, apply } => {
            let remote = progress_store(common)?;
            let config = progress_config(run.config.as_ref())?;
            let plan = if apply {
                reconcile::apply(
                    state,
                    &remote,
                    config,
                    &run.run,
                    &run.publisher,
                    clock.now_ms(),
                )?
            } else {
                reconcile::plan(state, &remote, &run.run, &run.publisher, config.limits)?
            };
            print_json(&plan)?;
            let applied = plan.applied.as_ref().map_or(!apply, |a| {
                matches!(
                    a.publication,
                    Some(Publication::Advanced { .. } | Publication::AlreadyPublished { .. })
                )
            });
            Ok(plan.recommit.is_empty() || (plan.admissible && applied))
        }
        ProgressCmd::Status { run } => {
            let remote = progress_store(common)?;
            let Some(committed) = reusable(&remote, &run)? else {
                print_json(&json!({"run": run, "head": null, "committed": false}))?;
                return Ok(false);
            };
            let (summary, census) = summarize(&committed)?;
            let (freshness, _) = read_summary(&remote, &run)?;
            print_json(&json!({
                "run": run,
                "head": committed.head,
                "summary": summary,
                "reconciled": census.reconciled(),
                "violations": census.violations,
                "units": census.units,
                "stored_summary": freshness,
            }))?;
            Ok(true)
        }
    }
}
