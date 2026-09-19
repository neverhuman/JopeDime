//! `jope-learning`: the learning store's `hwm` and `scores` command groups.

use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand, ValueEnum};
use jope_learning::hwm;
use jope_learning::mirror::{mirror_url, Mirror};
use jope_learning::scores;
use jope_learning::sources::{champions, genebank, snapshot, sotarena};
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
    /// Local state tree (default: $JOPE_STATE, else ~/.local/state/jope).
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
    /// Snapshot restricted to HWM winners (sealed excluded; unbound excluded unless asked).
    Freeze {
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        include_unbound: bool,
    },
    /// Ledger head, pointer count and tree size.
    Status,
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
                "pointers": m.pointers,
                "included": m.included,
                "excluded_sealed": m.excluded_sealed,
                "excluded_unbound": m.excluded_unbound,
                "ledger_head": m.ledger_head,
            }))?;
            Ok(true)
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
            let mut reports = Vec::new();
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
                // The published file, byte-identical, next to its hash.
                let published = state
                    .root()
                    .join("scores/sotarena")
                    .join(
                        rows.first()
                            .and_then(|r| r.source.track.clone())
                            .unwrap_or_else(|| version.clone()),
                    )
                    .join(&version);
                jope_learning::ledger::atomic_write(&published.join("scores.csv"), &bytes)?;
                jope_learning::ledger::atomic_write(
                    &published.join("scores.csv.sha256"),
                    format!("{}  scores.csv\n", scan.scores_sha256).as_bytes(),
                )?;
                let mut scans = BTreeMap::new();
                scans.insert(version.clone(), scan.clone());
                let source = sotarena::scan_source(&path, &git_sha, &scans);
                let rep = scores::append(state, mirror.as_ref(), rows, source)?;
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
