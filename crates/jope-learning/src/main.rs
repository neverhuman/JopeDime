//! `jope-learning`: the learning store's `hwm` and `scores` command groups.

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use jope_learning::hwm;
use jope_learning::mirror::{mirror_url, Mirror};
use jope_learning::progress::cadence::Clock;
use jope_learning::progress::inventory::{RunManifest, WorkSpec};
use jope_learning::progress::product::{Config, Driver, Step, SystemClock};
use jope_learning::progress::publish::{reusable, Publication};
use jope_learning::progress::rebuild::{read_summary, summarize};
use jope_learning::progress::reconcile;
use jope_learning::progress::remote::open_publishing_store;
use jope_learning::revocation;
use jope_learning::scores;
use jope_learning::sources::{champions, genebank, pipeline, snapshot, sotarena};
use jope_learning::state::State;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "jope-learning",
    version,
    about = "JOPE-Dime learning store: high-water-mark and score ledgers"
)]
struct Cli {
    #[command(flatten)]
    common: Common,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Args, Clone)]
struct Common {
    /// Local state tree (ADR-0003 default: $JOPE_STATE, else $XDG_STATE_HOME/jope, else
    /// ~/.local/state/jope, else <jope_home>/state).
    #[arg(long, global = true)]
    state: Option<PathBuf>,
    /// object_store URL mirrored after every write (default: $JOPE_LEARNING_URL; none = local only).
    #[arg(long, global = true)]
    store: Option<String>,
}

#[derive(Subcommand)]
enum Cmd {
    /// High-water-mark ledger and pointers.
    Hwm {
        #[command(subcommand)]
        cmd: HwmCmd,
    },
    /// Score ledger.
    Scores {
        #[command(subcommand)]
        cmd: ScoresCmd,
    },
    /// Print a committed schema as derived from the Rust types.
    Schema {
        #[arg(value_enum)]
        which: Which,
    },
    /// W8 durable progress: the engine's published events, committed on an explicit store.
    ///
    /// An s3:// store needs no code change, only configuration:
    /// JOPE_LEARNING_URL=s3://<bucket>/<run-scoped prefix> (or --store); JOPE_AWS_PROFILE=<scoped
    /// publisher profile> (or AWS_PROFILE, or AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
    /// [+ AWS_SESSION_TOKEN]); AWS_REGION (us-east-2 when unset). A bare bucket, JOPE_DATASETS_ANON=1
    /// and AWS_CONDITIONAL_PUT other than `etag` are refused before anything is written.
    Progress {
        #[command(subcommand)]
        cmd: ProgressCmd,
    },
    /// W12-T05: the revocations this host's consumers refuse, imported from the claims and
    /// revocation graph.
    Revocations {
        #[command(subcommand)]
        cmd: RevocationsCmd,
    },
}

#[derive(Subcommand)]
enum RevocationsCmd {
    /// Import the revocation closure of a verified claims record into this host's append-only
    /// revocation ledger. Read-only unless --apply. Nothing is ever un-revoked.
    Import {
        /// The claims record (jope.core-ip-claims/v1).
        #[arg(long)]
        claims: PathBuf,
        /// The repository root its evidence paths are relative to (default: the record's fifth
        /// ancestor, as for `jope-records verify-claims`).
        #[arg(long)]
        root: Option<PathBuf>,
        #[arg(long)]
        apply: bool,
    },
    /// Read-only: how many revoked digests this host holds, and the ledger head.
    Status,
}

/// The run a progress verb acts for, and who publishes it.
#[derive(Args, Clone)]
struct RunArgs {
    #[arg(long)]
    run: String,
    /// The run's publisher: must be the one its manifest names.
    #[arg(long)]
    publisher: String,
    /// A jope.progress.config/v1 file naming every outbox limit and cadence value (default: the
    /// values adopted by owner delegation on 2026-09-26).
    #[arg(long)]
    config: Option<PathBuf>,
}

/// An attempt of a planned unit.
#[derive(Args, Clone)]
struct AttemptArgs {
    /// The unit's work id (from `progress init`).
    #[arg(long)]
    unit: String,
    #[arg(long)]
    attempt: u32,
}

#[derive(Subcommand)]
enum ProgressCmd {
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

#[derive(Clone, Copy, ValueEnum)]
enum Which {
    Hwm,
    Score,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum HwmSource {
    /// A directory of parallel.cut_champion.v1 files (`<key>.json` or `<key>/current.json`).
    Champions,
    /// A genebank SQLite database (best_scores), opened read-only.
    Genebank,
    /// A lineage-A corpus-manifest.json plus --rows streams.
    Snapshot,
    /// A `jope pipeline run` (or evolve round) receipt, re-derived from its run directory first:
    /// its champion under the pipeline comparison identity (never teaching-eligible).
    Pipeline,
}

#[derive(Subcommand)]
enum HwmCmd {
    /// Map a source into candidates, append one sealed batch, move improved pointers.
    Ingest {
        #[arg(long, value_enum)]
        source: HwmSource,
        path: PathBuf,
        /// Provenance bundle id (default: the input's basename; genebank: the db sha256).
        #[arg(long)]
        bundle_id: Option<String>,
        /// snapshot: `<split>=<path>` (train|validation; `-` = stdin), repeatable.
        #[arg(long = "rows")]
        rows: Vec<String>,
        /// Print the full reject list instead of a count.
        #[arg(long)]
        show_rejects: bool,
    },
    /// The pointers of a dataset (best first), or the raw blob of one cut.
    Show {
        dataset_hash: String,
        #[arg(long)]
        cut: Option<String>,
        #[arg(long)]
        task: Option<String>,
        /// With --cut: write the record blob bytes (the original current.json) to stdout.
        #[arg(long)]
        raw: bool,
    },
    /// Write a blob's bytes to stdout.
    Blob { sha256: String },
    /// The latest ledger event for an identity_sha256 or a legacy Identity::key().
    Find { id: String },
    /// Walk every chain; exit 1 on any break.
    Verify {
        /// Champion event directories (`<key>/event-<sha>.json`) to walk engine chains through;
        /// repeatable.
        #[arg(long = "events")]
        events: Vec<PathBuf>,
    },
    /// Diagnostic HWM archive; this command does not admit teaching records.
    Freeze {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        include_unbound: bool,
    },
    /// Ledger head, pointer count and tree size.
    Status,
    /// Roll a lagging HWM ledger head and lagging pointers forward to the ledger tip (the
    /// recovery after a crash between a batch and its pointer writes). Read-only unless --apply;
    /// exit 1 when a pointer holds a record the ledger never accepted there.
    Recover {
        #[arg(long)]
        apply: bool,
    },
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum ScoreSource {
    /// A SOTArena results directory: every `<version>/scores.csv`.
    Sotarena,
}

#[derive(Subcommand)]
enum ScoresCmd {
    /// Append every row of a source as sealed batches (one per track version).
    Append {
        #[arg(long, value_enum)]
        source: ScoreSource,
        path: PathBuf,
        /// git sha recorded on every row (default: `git rev-parse HEAD` of the path, else unbound).
        #[arg(long)]
        git_sha: Option<String>,
    },
    /// Walk the batch chain and re-validate every row; exit 1 on any break.
    VerifyLedger,
    /// Query the ledger.
    Query {
        #[arg(long)]
        best_per_dataset: bool,
        #[arg(long)]
        track: Option<String>,
        #[arg(long)]
        version: Option<String>,
        #[arg(long)]
        include_invalid: bool,
    },
    Status,
}

fn print_json<T: serde::Serialize>(v: &T) -> Result<()> {
    let mut out = std::io::stdout().lock();
    serde_json::to_writer_pretty(&mut out, v)?;
    out.write_all(b"\n")?;
    Ok(())
}

fn open_mirror(common: &Common) -> Result<Option<Mirror>> {
    mirror_url(common.store.clone())
        .map(|u| Mirror::open(&u))
        .transpose()
}

fn basename(p: &std::path::Path) -> String {
    p.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| p.display().to_string())
}

fn run(cli: Cli) -> Result<bool> {
    let state = State::resolve(cli.common.state.clone())?;
    match cli.cmd {
        Cmd::Schema { which } => {
            let v = match which {
                Which::Hwm => jope_learning::record::derived_hwm_schema(),
                Which::Score => jope_learning::record::derived_score_schema(),
            };
            print_json(&v)?;
            Ok(true)
        }
        Cmd::Hwm { cmd } => run_hwm(&state, &cli.common, cmd),
        Cmd::Scores { cmd } => run_scores(&state, &cli.common, cmd),
        Cmd::Progress { cmd } => run_progress(&state, &cli.common, cmd),
        Cmd::Revocations { cmd } => match cmd {
            RevocationsCmd::Import {
                claims,
                root,
                apply,
            } => {
                let report = revocation::import(&state, &claims, root.as_deref(), apply)?;
                print_json(&report)?;
                Ok(true)
            }
            RevocationsCmd::Status => {
                let revocations = revocation::Revocations::load(&state)?;
                print_json(&json!({"state": state.root(), "status": revocations.status()}))?;
                Ok(true)
            }
        },
    }
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

fn run_progress(state: &State, common: &Common, cmd: ProgressCmd) -> Result<bool> {
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

fn run_hwm(state: &State, common: &Common, cmd: HwmCmd) -> Result<bool> {
    match cmd {
        HwmCmd::Ingest {
            source,
            path,
            bundle_id,
            rows,
            show_rejects,
        } => {
            let mirror = open_mirror(common)?;
            let (candidates, source_value, scan_json) = match source {
                HwmSource::Champions => {
                    let bundle = bundle_id.unwrap_or_else(|| basename(&path));
                    let (c, scan) = champions::scan(&path, &bundle)?;
                    let sv = champions::scan_source(&path, &bundle, &scan);
                    (c, sv, serde_json::to_value(&scan)?)
                }
                HwmSource::Genebank => {
                    let bundle = match bundle_id {
                        Some(b) => b,
                        None => jope_learning::sources::file_sha256(&path)?,
                    };
                    let (c, scan) = genebank::scan(&path, &bundle)?;
                    let sv = genebank::scan_source(&path, &bundle, &scan);
                    (c, sv, serde_json::to_value(&scan)?)
                }
                HwmSource::Snapshot => {
                    let mut splits: Vec<(String, Box<dyn Read>)> = Vec::new();
                    for spec in &rows {
                        let (split, p) = spec
                            .split_once('=')
                            .with_context(|| format!("--rows {spec}: expected <split>=<path>"))?;
                        let reader: Box<dyn Read> = if p == "-" {
                            Box::new(std::io::stdin())
                        } else {
                            Box::new(std::fs::File::open(p).with_context(|| format!("open {p}"))?)
                        };
                        splits.push((split.to_string(), reader));
                    }
                    if splits.is_empty() {
                        // Fall back to the manifest's own train/validation paths when present.
                        let (m, _) = snapshot::read_manifest(&path)?;
                        let _ = m;
                        bail!("--rows <split>=<path> is required (train and/or validation)");
                    }
                    let (c, scan) = snapshot::scan(&path, splits)?;
                    let sv = snapshot::scan_source(&path, &scan);
                    (c, sv, serde_json::to_value(&scan)?)
                }
                HwmSource::Pipeline => {
                    let (c, scan) = pipeline::scan(&path, bundle_id.as_deref())?;
                    let sv = pipeline::scan_source(&path, &scan);
                    (c, sv, serde_json::to_value(&scan)?)
                }
            };
            let report = hwm::ingest(state, mirror.as_ref(), candidates, source_value)?;
            let mut out = serde_json::to_value(&report)?;
            out["scan"] = scan_json;
            if !show_rejects {
                out["rejects"] = json!({"count": report.rejects.len(), "note": "--show-rejects lists them; every reject is also a ledger event"});
            }
            print_json(&out)?;
            Ok(true)
        }
        HwmCmd::Show {
            dataset_hash,
            cut,
            task,
            raw,
        } => {
            let entries = hwm::show(state, &dataset_hash, task.as_deref(), cut.as_deref())?;
            if raw {
                if cut.is_none() {
                    bail!("--raw needs --cut");
                }
                if entries.len() != 1 {
                    bail!("{} pointers match; use --task", entries.len());
                }
                let sha = &entries[0].record.provenance.record_sha256;
                let blob = state
                    .read_blob(sha)?
                    .with_context(|| format!("blob {sha} missing"))?;
                std::io::stdout().lock().write_all(&blob)?;
                return Ok(true);
            }
            print_json(&entries)?;
            Ok(!entries.is_empty())
        }
        HwmCmd::Blob { sha256 } => {
            let blob = state
                .read_blob(&sha256)?
                .with_context(|| format!("blob {sha256} missing"))?;
            std::io::stdout().lock().write_all(&blob)?;
            Ok(true)
        }
        HwmCmd::Find { id } => {
            let e = hwm::find_identity(state, &id)?;
            print_json(&e)?;
            Ok(e.is_some())
        }
        HwmCmd::Verify { events } => {
            let report = hwm::verify(state, &events)?;
            print_json(&report)?;
            eprintln!(
                "hwm verify: {} events ({} accepted, {} rejected), {} identities, {} pointers, {} blobs, {} champion chain links walked, {} breaks",
                report.events,
                report.accepted,
                report.rejected,
                report.identities,
                report.pointers_on_disk,
                report.blobs_checked,
                report.champion_chain_links_walked,
                report.total_breaks()
            );
            Ok(report.ok())
        }
        HwmCmd::Freeze {
            out,
            include_unbound,
        } => {
            let m = hwm::freeze(state, &out, include_unbound)?;
            print_json(&json!({
                "out": out,
                "scope": m.scope,
                "teaching_eligible": m.teaching_eligible,
                "pointers": m.pointers,
                "included": m.included,
                "excluded_sealed": m.excluded_sealed,
                "excluded_unbound": m.excluded_unbound,
                "ledger_head": m.ledger_head,
            }))?;
            Ok(true)
        }
        HwmCmd::Recover { apply } => {
            let report = if apply {
                hwm::recover(state)?
            } else {
                hwm::recover_plan(state)?
            };
            print_json(&report)?;
            Ok(report.refused.is_empty())
        }
        HwmCmd::Status => {
            let ledger = jope_learning::ledger::Ledger::open(state.root(), "hwm");
            let (bytes, files) = if state.root().exists() {
                hwm::tree_size(state.root())?
            } else {
                (0, 0)
            };
            print_json(&json!({
                "state": state.root(),
                "hwm_head": ledger.head()?,
                "pointers": state.pointers()?.len(),
                "tree_bytes": bytes,
                "tree_files": files,
            }))?;
            Ok(true)
        }
    }
}

fn run_scores(state: &State, common: &Common, cmd: ScoresCmd) -> Result<bool> {
    match cmd {
        ScoresCmd::Append {
            source: ScoreSource::Sotarena,
            path,
            git_sha,
        } => {
            let mirror = open_mirror(common)?;
            let git_sha = git_sha
                .or_else(|| sotarena::git_sha_of(&path))
                .unwrap_or_else(|| jope_learning::record::UNBOUND.into());
            // Two phases, and the split is the point. Everything that can refuse the import —
            // parsing every CSV, and proving that every archive path the run would write is a
            // plain path component inside the state root — happens first, with nothing written.
            // Only then does the second phase mutate anything, so one bad `track_id` in the last
            // version cannot leave the first version's archive already overwritten.
            struct Pending {
                version: String,
                track: String,
                bytes: Vec<u8>,
                rows: Vec<jope_learning::record::ScoreRecord>,
                scan: jope_learning::sources::sotarena::TrackScan,
            }
            let mut pending = Vec::new();
            for (version, csv) in sotarena::track_versions(&path)? {
                let bytes = std::fs::read(&csv)?;
                let fallback_ts = std::fs::metadata(&csv)?
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                    .map_or(0, |d| d.as_secs());
                let (rows, mut scan) =
                    sotarena::parse_scores(&bytes, &version, &git_sha, fallback_ts)?;
                scan.scores_csv = csv.display().to_string();
                // `track_id` is a column of a file this crate did not write. It names a directory
                // under the state root, so it is checked as a path component before it is ever
                // joined to one; `State::archive_dir` re-checks it and proves containment.
                let track = rows
                    .first()
                    .and_then(|r| r.source.track.clone())
                    .unwrap_or_else(|| version.clone());
                for (what, seg) in [("track_id", &track), ("version", &version)] {
                    if !jope_learning::state::plain_segment(seg) {
                        bail!(
                            "{}: unsafe {what} {seg:?} — a path component must be non-empty, must \
                             not be `.` or `..`, and may only contain ASCII letters, digits, `.`, \
                             `-` and `_`. Nothing was written.",
                            csv.display()
                        );
                    }
                }
                pending.push(Pending {
                    version,
                    track,
                    bytes,
                    rows,
                    scan,
                });
            }
            let mut reports = Vec::new();
            for Pending {
                version,
                track,
                bytes,
                rows,
                scan,
            } in pending
            {
                let mut scans = BTreeMap::new();
                scans.insert(version.clone(), scan.clone());
                let source = sotarena::scan_source(&path, &git_sha, &scans);
                // The published file, byte-identical, next to its hash — written inside the
                // `scores` lock and only after the mirror preflight has had its chance to refuse.
                let sha = scan.scores_sha256.clone();
                let rep = scores::append(state, mirror.as_ref(), rows, source, |state| {
                    let dir = state.archive_dir("sotarena", &track, &version)?;
                    jope_learning::ledger::atomic_write(&dir.join("scores.csv"), &bytes)?;
                    jope_learning::ledger::atomic_write(
                        &dir.join("scores.csv.sha256"),
                        format!("{sha}  scores.csv\n").as_bytes(),
                    )
                })?;
                let mut v = serde_json::to_value(&rep)?;
                v["scan"] = serde_json::to_value(&scan)?;
                reports.push(v);
            }
            print_json(&reports)?;
            Ok(true)
        }
        ScoresCmd::VerifyLedger => {
            let report = scores::verify(state)?;
            print_json(&report)?;
            eprintln!(
                "scores verify-ledger: {} batches, {} rows, {} breaks",
                report.ledger.batches,
                report.rows,
                report.total_breaks()
            );
            Ok(report.ok())
        }
        ScoresCmd::Query {
            best_per_dataset,
            track,
            version,
            include_invalid,
        } => {
            if !best_per_dataset {
                bail!("only --best-per-dataset is implemented");
            }
            let best = scores::best_per_dataset(
                state,
                track.as_deref(),
                version.as_deref(),
                include_invalid,
            )?;
            print_json(&best)?;
            Ok(true)
        }
        ScoresCmd::Status => {
            let ledger = jope_learning::ledger::Ledger::open(state.root(), "scores");
            let v: Value = json!({"state": state.root(), "scores_head": ledger.head()?});
            print_json(&v)?;
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}
