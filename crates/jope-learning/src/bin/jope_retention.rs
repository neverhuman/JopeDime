//! `jope_retention classify`: the S3 retention brain's read-only dry run (gap S3-RET).
//!
//! Reads local snapshots only and writes a plan directory. It makes no S3 calls and has no apply
//! path. Any unreadable or inconsistent input aborts with exit 2 and writes nothing.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use jope_learning::retention::inputs::InputSpec;
use jope_learning::retention::plan::{build, write, Config};
use jope_learning::retention::refs::Role;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "jope_retention",
    about = "S3 retention brain: read-only dry-run classifier"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Classify every object group and write a dry-run plan.
    Classify(ClassifyArgs),
}

#[derive(clap::Args)]
struct ClassifyArgs {
    /// Plan time, RFC 3339 UTC (YYYY-MM-DDTHH:MM:SSZ); the live-writer window counts back from it.
    #[arg(long)]
    as_of: String,
    /// S3 object ledger, PATH[@SHA256PREFIX].
    #[arg(long)]
    ledger: String,
    /// GP results index, PATH[@SHA256PREFIX].
    #[arg(long)]
    gp_index: String,
    /// Extra input pinned in the lock only, ROLE=PATH[@SHA256PREFIX] (repeatable).
    #[arg(long = "pin")]
    pins: Vec<String>,
    /// Reference file, ROLE=PATH[@SHA256PREFIX]; roles: val263 sealed prereg evidence
    /// main-manifest train frozen24 backlog seed (repeatable).
    #[arg(long = "refs")]
    refs: Vec<String>,
    /// Roster manifest whose `members` pin the --refs files in its directory, PATH[@SHA256PREFIX].
    #[arg(long)]
    roster_manifest: Option<String>,
    /// Directory scanned best-effort for s3 references, ROLE=DIR (repeatable).
    #[arg(long = "scan")]
    scans: Vec<String>,
    /// Files larger than this are counted but not read by --scan.
    #[arg(long, default_value_t = 4 * 1024 * 1024)]
    scan_max_bytes: u64,
    /// Directory name never entered by --scan, e.g. this lane's own output dir (repeatable).
    #[arg(long = "scan-skip-dir")]
    scan_skip_dirs: Vec<String>,
    /// Directory of `<bucket>.json` lifecycle configurations.
    #[arg(long)]
    lifecycle_dir: Option<PathBuf>,
    /// Policy JSON (default: the embedded retention/policy.v1.json).
    #[arg(long)]
    policy: Option<PathBuf>,
    /// Previous plan JSONL to diff against.
    #[arg(long)]
    previous_plan: Option<PathBuf>,
    /// Source commit recorded in the inputs lock.
    #[arg(long)]
    source_commit: Option<String>,
    /// Output directory; must not exist or be empty.
    #[arg(long)]
    out_dir: PathBuf,
}

fn spec(role: &str, s: &str) -> Result<InputSpec> {
    InputSpec::parse(&format!("{role}={s}"))
}

fn run(a: ClassifyArgs) -> Result<()> {
    let scans = a
        .scans
        .iter()
        .map(|s| {
            let (role, dir) = s
                .split_once('=')
                .with_context(|| format!("--scan {s:?} is not ROLE=DIR"))?;
            Ok((Role::parse(role)?, PathBuf::from(dir)))
        })
        .collect::<Result<Vec<_>>>()?;
    let cfg = Config {
        as_of: a.as_of,
        ledger: spec("ledger", &a.ledger)?,
        gp_index: spec("gp-index", &a.gp_index)?,
        extra_inputs: a
            .pins
            .iter()
            .map(|s| InputSpec::parse(s))
            .collect::<Result<_>>()?,
        refs: a
            .refs
            .iter()
            .map(|s| InputSpec::parse(s))
            .collect::<Result<_>>()?,
        roster_manifest: a
            .roster_manifest
            .as_deref()
            .map(|s| spec("roster-manifest", s))
            .transpose()?,
        scans,
        scan_max_bytes: a.scan_max_bytes,
        scan_skip_dirs: a.scan_skip_dirs,
        lifecycle_dir: a.lifecycle_dir,
        policy: a.policy,
        previous_plan: a.previous_plan,
        source_commit: a.source_commit,
    };
    let built = build(&cfg)?;
    write(&built, &a.out_dir)?;
    let s = &built.summary;
    println!(
        "plan {} sha256 {} rows {}",
        a.out_dir.join(s.plan_file).display(),
        s.plan_sha256,
        s.rows
    );
    for (action, t) in &s.by_action {
        println!(
            "{action:>12}: {:>6} groups {:>10} objects {:>9.1} GB  now ${:>8.2}/mo  after ${:>8.2}/mo",
            t.groups, t.objects, t.gb, t.now_usd_mo, t.after_usd_mo
        );
    }
    println!(
        "hazards {} notes {} findings {}",
        s.hazards.len(),
        s.notes.len(),
        s.findings.len()
    );
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let Cmd::Classify(args) = cli.cmd;
    if let Err(e) = run(args) {
        eprintln!("jope_retention: FAIL-CLOSED, no plan written: {e:#}");
        std::process::exit(2);
    }
}
