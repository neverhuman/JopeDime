//! `jope_retention classify`: the S3 retention brain's read-only dry run (gap S3-RET), plus
//! `inspect-parquet` for checking local S3 Inventory files.
//!
//! Reads local snapshots only and writes a plan directory. It makes no S3 calls and has no apply
//! path. Any unreadable or inconsistent input aborts with exit 2 and writes nothing.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use jope_learning::retention::inputs::InputSpec;
use jope_learning::retention::parquet::{FileReader, PageCounts, Values};
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
    Classify(Box<ClassifyArgs>),
    /// Decode every page of local Parquet files (e.g. S3 Inventory reports) and print their
    /// structure, page encodings and a few rows as JSON lines.
    InspectParquet {
        files: Vec<PathBuf>,
        /// Rows of the first row group to print.
        #[arg(long, default_value_t = 3)]
        sample: usize,
    },
    /// Distill S3 context-level GP records from `GP_RESULTS_INDEX.json` into a validated
    /// `gp_distilled/v1` Parquet file and verify readback + E3 key reconciliation.
    DistillGpIndex {
        #[arg(long)]
        gp_index: PathBuf,
        #[arg(long, default_value = "train")]
        split_role: String,
        #[arg(long)]
        source_commit: Option<String>,
        #[arg(long)]
        receipt: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(clap::Args)]
struct ClassifyArgs {
    /// Plan time, RFC 3339 UTC (YYYY-MM-DDTHH:MM:SSZ); the live-writer window counts back from it.
    #[arg(long)]
    as_of: String,
    /// S3 object ledger, `PATH[@SHA256PREFIX]`.
    #[arg(long)]
    ledger: String,
    /// GP results index, `PATH[@SHA256PREFIX]`.
    #[arg(long)]
    gp_index: String,
    /// Extra input pinned in the lock only, `ROLE=PATH[@SHA256PREFIX]` (repeatable).
    #[arg(long = "pin")]
    pins: Vec<String>,
    /// Reference file, `ROLE=PATH[@SHA256PREFIX]`; roles: val263 sealed prereg evidence
    /// main-manifest train frozen24 backlog seed (repeatable).
    #[arg(long = "refs")]
    refs: Vec<String>,
    /// Roster manifest whose `members` pin the --refs files in its directory, `PATH[@SHA256PREFIX]`.
    #[arg(long)]
    roster_manifest: Option<String>,
    /// Directory scanned best-effort for s3 references, ROLE=DIR (repeatable).
    #[arg(long = "scan")]
    scans: Vec<String>,
    /// Files larger than this are not parsed by --scan. They are streamed for in-scope bucket
    /// names instead, and one that names a bucket fails the run (it could hide a pin).
    #[arg(long, default_value_t = 512 * 1024 * 1024)]
    scan_max_bytes: u64,
    /// Directory name never entered by --scan, e.g. this lane's own output dir (repeatable).
    #[arg(long = "scan-skip-dir")]
    scan_skip_dirs: Vec<String>,
    /// Directory of `<bucket>.json` lifecycle configurations.
    #[arg(long)]
    lifecycle_dir: Option<PathBuf>,
    /// Local copy of an S3 Inventory report manifest.json (its manifest.checksum and data/ files
    /// alongside), `PATH[@SHA256PREFIX]`; one per bucket (repeatable).
    #[arg(long = "inventory")]
    inventories: Vec<String>,
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
        inventories: a
            .inventories
            .iter()
            .map(|s| spec("inventory-manifest", s))
            .collect::<Result<_>>()?,
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
    let res = match cli.cmd {
        Cmd::Classify(args) => {
            run(*args).map_err(|e| format!("FAIL-CLOSED, no plan written: {e:#}"))
        }
        Cmd::InspectParquet { files, sample } => {
            inspect(&files, sample).map_err(|e| format!("{e:#}"))
        }
        Cmd::DistillGpIndex {
            gp_index,
            split_role,
            source_commit,
            receipt,
            out,
        } => distill_gp_index(
            &gp_index,
            &split_role,
            source_commit.as_deref(),
            receipt.as_deref(),
            &out,
        )
        .map_err(|e| format!("{e:#}")),
    };
    if let Err(e) = res {
        eprintln!("jope_retention: {e}");
        std::process::exit(2);
    }
}

fn distill_gp_index(
    gp_index: &std::path::Path,
    split_role: &str,
    source_commit: Option<&str>,
    receipt_path: Option<&std::path::Path>,
    out: &std::path::Path,
) -> Result<()> {
    use anyhow::ensure;
    use jope_learning::retention::distilled::{
        distill_gp_results_index_with_stats, read, reconcile, write, SplitRole,
    };
    use std::collections::BTreeSet;
    use std::fs::OpenOptions;
    use std::io::Write;

    if let Some(sc) = source_commit {
        ensure!(
            sc.len() == 40 && sc.bytes().all(|b| b.is_ascii_hexdigit()),
            "--source-commit must be a 40-hex commit digest"
        );
    }
    let bytes =
        std::fs::read(gp_index).with_context(|| format!("cannot read {}", gp_index.display()))?;
    let index: serde_json::Value = serde_json::from_slice(&bytes)
        .with_context(|| format!("cannot parse {}", gp_index.display()))?;
    let role = SplitRole::parse(split_role)?;
    let known: BTreeSet<String> = [
        "cv_geomean_rmse_ratio",
        "best_rust_rmse",
        "validation_rmse",
        "shadow_pool_q",
        "l6",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    let index_sha = jope_learning::canon::sha256_hex(&bytes);
    let (rows, raw_keys, stats) = distill_gp_results_index_with_stats(&index, role, &known)?;
    let parquet_bytes = write(
        &rows,
        &known,
        &[("gp_results_index_sha256".into(), index_sha.clone())],
    )?;
    let readback = read(&parquet_bytes, &known)?;
    reconcile(&readback, &raw_keys)?;
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(out)
        .with_context(|| format!("create {} (outputs are never overwritten)", out.display()))?;
    f.write_all(&parquet_bytes)?;
    f.sync_all()?;
    let parquet_sha = jope_learning::canon::sha256_hex(&parquet_bytes);
    let body = serde_json::json!({
        "schema": "jope.gp-index-distill-receipt/v1",
        "gp_index_path": gp_index.display().to_string(),
        "split_role": role.as_str(),
        "out": out.display().to_string(),
        "parquet_bytes": parquet_bytes.len(),
        "parquet_sha256": parquet_sha,
        "gp_results_index_sha256": index_sha,
        "total_index_runs": stats.total_index_runs,
        "distilled_rows": stats.distilled_rows,
        "skipped_unscored_runs": stats.skipped_unscored_runs,
        "skipped_incomplete_provenance_runs": stats.skipped_incomplete_provenance_runs,
        "reconciled_s3_keys": stats.reconciled_s3_keys,
    });
    if let Some(rp) = receipt_path {
        let exe = std::env::current_exe().context("current_exe")?;
        let bin_sha = jope_learning::canon::sha256_hex(&std::fs::read(&exe)?);
        let receipt_sha =
            jope_learning::canon::sha256_hex(&jope_learning::canon::canonical_json(&body)?);
        let doc = serde_json::json!({
            "receipt_sha256": receipt_sha,
            "source_commit": source_commit,
            "binary_sha256": bin_sha,
            "receipt": body,
        });
        let mut rf = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(rp)
            .with_context(|| format!("create {} (outputs are never overwritten)", rp.display()))?;
        let mut pretty = serde_json::to_vec_pretty(&doc)?;
        pretty.push(b'\n');
        rf.write_all(&pretty)?;
        rf.sync_all()?;
    }
    println!("{body}");
    Ok(())
}

fn show(v: &Values, i: usize) -> serde_json::Value {
    match v {
        Values::Bool(x) => serde_json::json!(x[i]),
        Values::I64(x) => serde_json::json!(x[i]),
        Values::F64(x) => serde_json::json!(x[i]),
        Values::Bytes(x) => serde_json::json!(x[i]
            .as_ref()
            .map(|b| String::from_utf8_lossy(b).to_string())),
    }
}

/// Decodes every page of every column (a full decoder check) and prints one JSON line per file.
fn inspect(files: &[PathBuf], sample: usize) -> Result<()> {
    for f in files {
        let bytes = std::fs::read(f).with_context(|| format!("cannot read {}", f.display()))?;
        let r = FileReader::open(&bytes)?;
        let mut pages: Vec<PageCounts> = vec![PageCounts::default(); r.columns.len()];
        let mut nulls = vec![0u64; r.columns.len()];
        let mut rows_sample = Vec::new();
        for g in 0..r.row_groups.len() {
            let mut cols = Vec::new();
            for (c, (pc, nc)) in pages.iter_mut().zip(nulls.iter_mut()).enumerate() {
                let (v, counts) = r.read_column(g, c)?;
                pc.add(&counts);
                *nc += (0..v.len()).filter(|&i| show(&v, i).is_null()).count() as u64;
                cols.push(v);
            }
            if g == 0 {
                for i in 0..sample.min(r.row_groups[0].rows as usize) {
                    let row: serde_json::Map<String, serde_json::Value> = r
                        .columns
                        .iter()
                        .zip(&cols)
                        .map(|(m, v)| (m.name.clone(), show(v, i)))
                        .collect();
                    rows_sample.push(row);
                }
            }
        }
        let line = serde_json::json!({
            "file": f.display().to_string(),
            "bytes": bytes.len(),
            "sha256": jope_learning::canon::sha256_hex(&bytes),
            "created_by": r.created_by,
            "num_rows": r.num_rows,
            "row_groups": r.row_groups.iter().map(|g| g.rows).collect::<Vec<_>>(),
            "columns": r.columns.iter().zip(pages.iter().zip(&nulls)).map(|(c, (p, n))| serde_json::json!({
                "name": c.name, "physical": c.physical, "required": c.required,
                "timestamp_ms_divisor": c.timestamp_ms_divisor, "nulls": n, "pages": p
            })).collect::<Vec<_>>(),
            "sample": rows_sample,
        });
        println!("{line}");
    }
    Ok(())
}
