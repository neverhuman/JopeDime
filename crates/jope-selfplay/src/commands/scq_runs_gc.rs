//! Retention for `smart_cluster/var/runs`, which grows without bound.
//!
//! scq stages every job's inputs into its own run directory and never reclaims
//! them. On 2026-08-04 that directory held **132,351 run dirs / 523 GB on
//! xbabe2** -- the single largest consumer on the box and the reason it sat at
//! 90 % disk -- because each eval job stages its own 2.2 MB copy of
//! `corpus_eval` alongside its CSVs.
//!
//! The shape of a run directory is what makes this cheap to fix:
//!
//! ```text
//! jte-<id>/attempts/0/inputs/   5.1 MB   <- 98 %: corpus_eval + train.csv + test.csv
//! jte-<id>/attempts/0/outputs/   28 KB   <- the receipt: the gene and its fitness
//! jte-<id>/attempts/0/{rc,stdout.log,scq.token}
//! ```
//!
//! Everything we actually care about long term is the receipt. The staged inputs
//! are pure re-derivable artifacts: they are content-addressed copies of files
//! that still exist at their source, scq re-stages them per attempt on a retry,
//! and **nothing reads them once the job has run** (verified across `jope-dime`:
//! ingestion resolves `attempts/*/outputs/receipt` via `local_receipt_dir` and
//! never touches `inputs/`).
//!
//! So the policy is two-tier, safest first:
//!
//! 1. **Reclaim `inputs/` from terminal jobs.** ~98 % of the bytes, and it cannot
//!    lose a result because the receipt lives in `outputs/`.
//! 2. **Purge whole run dirs past a retention window.** By then the receipt has
//!    long since been ingested into the corpus database, which is the durable
//!    home for the gene and its cost.
//!
//! Both tiers refuse to touch a job scq still considers live, and refuse to touch
//! anything younger than `--min-age-minutes`, so an in-flight or just-finished
//! job is never disturbed.

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, SystemTime};

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_campaigns::paths;
use serde_json::Value;

/// scq job states that mean "this job may still write here". A run directory
/// belonging to one of these is never touched, at either tier.
const LIVE_STATES: &[&str] = &[
    "queued",
    "starting",
    "dispatching",
    "running",
    "preempting",
    "held",
];

#[derive(Parser, Debug)]
#[command(
    name = "scq-runs-gc",
    about = "Reclaim staged job inputs from smart_cluster/var/runs, keeping receipts."
)]
struct Args {
    #[arg(long, default_value_os_t = paths::scq_runs_root())]
    runs_root: PathBuf,
    #[arg(long, default_value_os_t = paths::scq_binary())]
    scq: PathBuf,
    /// Never touch a run directory younger than this. Guards against racing a
    /// job that scq has not yet reported, and against clock skew.
    #[arg(long, default_value_t = 60)]
    min_age_minutes: u64,
    /// Tier 2: remove the whole run directory once it is older than this. The
    /// receipt has been ingested into the corpus database long before.
    #[arg(long, default_value_t = 14)]
    purge_after_days: u64,
    /// Bound the work per invocation so a timer run stays short and interruptible.
    ///
    /// This counts directories actually ACTED ON, not directories looked at.
    /// Counting scans instead makes the tool re-walk the same already-clean
    /// prefix on every run and never reach the tail: `read_dir` returns a stable
    /// order, so a scan-bounded run spends its whole budget on directories it
    /// already emptied. Measured: six consecutive scan-bounded runs reclaimed
    /// 1.05 GiB, then 0.04, then 0.00 four times, while 508 GB remained
    /// untouched further down the directory.
    #[arg(long, default_value_t = 20_000)]
    max_dirs: usize,
    /// Report what would be reclaimed and change nothing.
    #[arg(long)]
    dry_run: bool,
}

pub fn main(args: Vec<String>) -> Result<()> {
    let args = Args::parse_from(&args);
    ensure!(
        args.runs_root.is_dir(),
        "runs root is not a directory: {}",
        args.runs_root.display()
    );
    // Resolve once so every later path can be proven to live underneath it.
    let runs_root = args
        .runs_root
        .canonicalize()
        .with_context(|| format!("canonicalize {}", args.runs_root.display()))?;

    let live = live_job_ids(&args.scq)?;
    let now = SystemTime::now();
    let min_age = Duration::from_secs(args.min_age_minutes * 60);
    let purge_age = Duration::from_secs(args.purge_after_days * 86_400);

    let mut scanned = 0_usize;
    let mut skipped_live = 0_usize;
    let mut skipped_young = 0_usize;
    let mut inputs_reclaimed = 0_usize;
    let mut dirs_purged = 0_usize;
    let mut bytes = 0_u64;

    for entry in fs::read_dir(&runs_root)?.flatten() {
        // Budget is spent on work, not on looking. See `max_dirs`.
        if inputs_reclaimed + dirs_purged >= args.max_dirs {
            break;
        }
        let path = entry.path();
        if !entry.file_type().map(|k| k.is_dir()).unwrap_or(false) {
            continue;
        }
        let Some(job_id) = path.file_name().and_then(|v| v.to_str()) else {
            continue;
        };
        scanned += 1;

        if live.contains(job_id) {
            skipped_live += 1;
            continue;
        }
        let age = match dir_age(&path, now) {
            Some(age) => age,
            None => continue,
        };
        if age < min_age {
            skipped_young += 1;
            continue;
        }

        // Tier 2: past the window, the whole directory goes.
        if age >= purge_age {
            let freed = dir_bytes(&path);
            if !args.dry_run {
                remove_under(&runs_root, &path)?;
            }
            bytes += freed;
            dirs_purged += 1;
            continue;
        }

        // Tier 1: drop the staged inputs, keep outputs/receipt, rc and logs.
        let attempts = path.join("attempts");
        let Ok(reader) = fs::read_dir(&attempts) else {
            continue;
        };
        let mut touched = false;
        for attempt in reader.flatten() {
            let inputs = attempt.path().join("inputs");
            if !inputs.is_dir() {
                continue;
            }
            let freed = dir_bytes(&inputs);
            if !args.dry_run {
                remove_under(&runs_root, &inputs)?;
            }
            bytes += freed;
            touched = true;
        }
        if touched {
            inputs_reclaimed += 1;
        }
    }

    println!(
        "{}",
        serde_json::json!({
            "schema": "ScqRunsGcReport.v1",
            "dry_run": args.dry_run,
            "runs_root": runs_root,
            "scanned": scanned,
            "skipped_live": skipped_live,
            "skipped_younger_than_min_age": skipped_young,
            "inputs_reclaimed_dirs": inputs_reclaimed,
            "whole_dirs_purged": dirs_purged,
            "bytes_reclaimed": bytes,
            "gib_reclaimed": format!("{:.2}", bytes as f64 / (1024.0 * 1024.0 * 1024.0)),
        })
    );
    Ok(())
}

/// Every job scq still knows about in a non-terminal state.
///
/// Fails closed: if scq cannot be reached we return an error rather than an
/// empty set, because an empty set would make every run directory look reclaimable.
fn live_job_ids(scq: &Path) -> Result<HashSet<String>> {
    let output = Command::new(scq)
        .args(["--json", "ls"])
        .output()
        .with_context(|| format!("run {} --json ls", scq.display()))?;
    if !output.status.success() {
        bail!(
            "scq --json ls failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    let value: Value = serde_json::from_slice(&output.stdout).context("parse scq --json ls")?;
    let jobs = value
        .pointer("/Jobs/jobs")
        .and_then(Value::as_array)
        .context("scq --json ls has no /Jobs/jobs array")?;
    Ok(jobs
        .iter()
        .filter(|job| {
            job.get("state")
                .and_then(Value::as_str)
                .is_some_and(|state| LIVE_STATES.contains(&state))
        })
        .filter_map(|job| job.get("id").and_then(Value::as_str).map(str::to_owned))
        .collect())
}

/// Age from the newest mtime we can see on the directory itself. `None` when the
/// directory has vanished or its time is in the future (clock skew) -- both mean
/// "leave it alone".
fn dir_age(path: &Path, now: SystemTime) -> Option<Duration> {
    let modified = fs::metadata(path).ok()?.modified().ok()?;
    now.duration_since(modified).ok()
}

/// Sum of regular-file sizes beneath `path`. Symlinks are counted but never
/// followed, so a link can neither inflate the total nor lead the walk out.
fn dir_bytes(path: &Path) -> u64 {
    let mut total = 0_u64;
    let mut stack = vec![path.to_path_buf()];
    while let Some(current) = stack.pop() {
        let Ok(reader) = fs::read_dir(&current) else {
            continue;
        };
        for entry in reader.flatten() {
            let Ok(kind) = entry.file_type() else {
                continue;
            };
            if kind.is_dir() {
                stack.push(entry.path());
            } else if let Ok(meta) = entry.metadata() {
                total += meta.len();
            }
        }
    }
    total
}

/// Delete `target`, refusing anything that is not strictly beneath `root`.
///
/// This is the blast-radius guard: a symlinked or otherwise escaping path must
/// never be removed, because this tool runs unattended on a timer.
fn remove_under(root: &Path, target: &Path) -> Result<()> {
    let resolved = target
        .canonicalize()
        .with_context(|| format!("canonicalize {}", target.display()))?;
    ensure!(
        resolved.starts_with(root) && resolved != root,
        "refusing to remove {} which is not strictly inside {}",
        resolved.display(),
        root.display()
    );
    fs::remove_dir_all(&resolved).with_context(|| format!("remove {}", resolved.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Write;
    use tempfile::tempdir;

    fn run_dir(root: &Path, job: &str, input_bytes: usize) -> PathBuf {
        let attempt = root.join(job).join("attempts/0");
        fs::create_dir_all(attempt.join("inputs")).unwrap();
        fs::create_dir_all(attempt.join("outputs/receipt")).unwrap();
        File::create(attempt.join("inputs/corpus_eval"))
            .unwrap()
            .write_all(&vec![0_u8; input_bytes])
            .unwrap();
        File::create(attempt.join("outputs/receipt/receipt.json"))
            .unwrap()
            .write_all(b"{\"gene\":\"(pipe X (ridge))\",\"score\":0.5}")
            .unwrap();
        root.join(job)
    }

    /// The receipt is the whole point: reclaiming inputs must never touch it.
    #[test]
    fn reclaiming_inputs_preserves_the_receipt() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let job = run_dir(root, "jte-aaaa", 4096);

        let inputs = job.join("attempts/0/inputs");
        let receipt = job.join("attempts/0/outputs/receipt/receipt.json");
        assert!(inputs.is_dir() && receipt.is_file());

        remove_under(root, &inputs).unwrap();

        assert!(!inputs.exists(), "staged inputs are reclaimable");
        assert!(receipt.is_file(), "the gene and its fitness must survive");
        assert!(job.join("attempts/0/outputs").is_dir());
    }

    /// The blast-radius guard. This runs unattended on a timer, so a path that
    /// escapes the runs root must be refused rather than deleted.
    #[test]
    fn removal_outside_the_runs_root_is_refused() {
        let directory = tempdir().unwrap();
        let root = directory.path().join("runs");
        let outside = directory.path().join("precious");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        File::create(outside.join("keep.txt")).unwrap();

        assert!(remove_under(&root, &outside).is_err());
        assert!(outside.join("keep.txt").is_file());

        // The root itself is never removable either.
        assert!(remove_under(&root, &root).is_err());
        assert!(root.is_dir());
    }

    #[test]
    fn directory_bytes_counts_nested_files() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        run_dir(root, "jte-bbbb", 1024);
        let total = dir_bytes(&root.join("jte-bbbb"));
        assert!(total >= 1024, "expected the staged input to be counted");
    }

    /// A live job's directory must be untouchable: scq may still write to it.
    #[test]
    fn live_states_cover_every_non_terminal_scq_state() {
        for state in ["queued", "starting", "dispatching", "running", "held"] {
            assert!(LIVE_STATES.contains(&state), "{state} must be protected");
        }
        // Terminal states are what this tool exists to reclaim.
        for state in ["completed", "failed", "cancelled"] {
            assert!(!LIVE_STATES.contains(&state));
        }
    }
}
