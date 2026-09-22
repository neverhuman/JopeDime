//! `jope-arena`: the SOTArena side of JopeDime.
//!
//! Roots come from the environment (`SOTARENA_ROOT`, `SOTARENA_REPORTER_ROOT`,
//! `SOTARENA_BENCHMARK_CACHE`, falling back to `JOPE_HOME` / `JOPE_STATE`); every subcommand also
//! takes an explicit `--root`.

use std::io::Write;
use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use jope_arena::leaderboard::Ladder;
use jope_arena::{family, fetch, io, leaderboard, paths, publish, reporter, scores, verify};

#[derive(Parser)]
#[command(
    name = "jope-arena",
    about = "SOTArena fetch, Elo ladders, score emission and publication",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Recompute a track's ladder from `results/<track>[/<version>]/scores.csv`.
    Elo {
        /// Track id (`tabular-regression-v2`).
        #[arg(long)]
        track: String,
        /// Results version directory; default: the newest present, else the flat layout.
        #[arg(long)]
        version: Option<String>,
        /// Benchmark root (default `SOTARENA_ROOT`).
        #[arg(long, env = paths::var::SOTARENA_ROOT)]
        root: Option<PathBuf>,
        /// `published`: every rankable row, byte-for-byte the benchmark's own ladder.
        /// `benchmark-valid`: only rows whose `benchmark_valid` is True after `--corrections`.
        #[arg(long, value_enum, default_value_t = Ladder::Published)]
        ladder: Ladder,
        /// A `jope.arena-protocol-corrections/v1` file applied to the rows before ranking.
        #[arg(long)]
        corrections: Option<PathBuf>,
        /// Write the leaderboard CSV here (default: stdout).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Write the `jope.arena-elo-snapshot/v2` JSON here.
        #[arg(long)]
        json: Option<PathBuf>,
        /// Fail unless the ladder equals the committed CSV (`--committed`, else the
        /// `leaderboard.csv` next to the scores, which is only the published ladder).
        #[arg(long)]
        check: bool,
        /// The committed ladder `--check` compares against.
        #[arg(long, requires = "check")]
        committed: Option<PathBuf>,
    },
    /// Turn a campaign's receipts into `scores.csv` rows.
    EmitScores {
        /// Campaign output directory (searched recursively for receipts).
        #[arg(long)]
        campaign: PathBuf,
        /// Track id for every row.
        #[arg(long)]
        track: String,
        /// Method id for every row.
        #[arg(long)]
        method_id: String,
        /// `blind`, `public-development` or `aspfm-generated`.
        #[arg(long)]
        protocol: String,
        /// Metric id.
        #[arg(long, default_value = "rmse")]
        metric: String,
        /// `source_revision` column.
        #[arg(long, default_value = "")]
        source_revision: String,
        /// `notes` column.
        #[arg(long, default_value = "")]
        notes: String,
        /// Write the CSV here (default: stdout).
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Check every file of a results directory against its `evidence-manifest.json`.
    VerifyResults {
        /// The results directory.
        dir: PathBuf,
        /// Print the full JSON report.
        #[arg(long)]
        json: bool,
        /// Also fail on files present in the directory but absent from the manifest.
        #[arg(long)]
        strict: bool,
    },
    /// Copy a results directory into an object store as `scores/sotarena/<track>/<version>/`.
    Publish {
        /// Track id.
        #[arg(long)]
        track: String,
        /// Version label (immutable once published).
        #[arg(long)]
        version: String,
        /// Object-store URL (`file:///path`, or `s3://bucket/prefix` with the `s3` feature).
        #[arg(long)]
        store: String,
        /// Results directory (default: `results/<track>` under `SOTARENA_ROOT`).
        #[arg(long)]
        results: Option<PathBuf>,
        /// Benchmark root (default `SOTARENA_ROOT`).
        #[arg(long, env = paths::var::SOTARENA_ROOT)]
        root: Option<PathBuf>,
    },
    /// Re-hash a published version against its `MANIFEST.json`.
    VerifyPublished {
        /// Track id.
        #[arg(long)]
        track: String,
        /// Version label.
        #[arg(long)]
        version: String,
        /// Object-store URL.
        #[arg(long)]
        store: String,
    },
    /// Download and verify the reporter's datasets.
    Fetch {
        /// Task id (required unless `--all`).
        #[arg(long, required_unless_present = "all", conflicts_with = "all")]
        task: Option<String>,
        /// One dataset (hash or id) of `--task`.
        #[arg(long, requires = "task", conflicts_with = "all")]
        dataset: Option<String>,
        /// Every downloadable dataset.
        #[arg(long)]
        all: bool,
        /// Cache directory (default `SOTARENA_BENCHMARK_CACHE`, else `$JOPE_HOME/benchmarks/cache`).
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Reporter root (default `SOTARENA_REPORTER_ROOT`).
        #[arg(long, env = paths::var::SOTARENA_REPORTER_ROOT)]
        root: Option<PathBuf>,
    },
    /// The reporter's family report (`sotarena.report.v2`) as JSON.
    Report {
        /// Reporter root (default `SOTARENA_REPORTER_ROOT`).
        #[arg(long, env = paths::var::SOTARENA_REPORTER_ROOT)]
        root: Option<PathBuf>,
        /// Write here (default: stdout).
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

fn benchmark_root(root: Option<PathBuf>) -> PathBuf {
    root.unwrap_or_else(paths::sotarena_root)
}

fn reporter_root(root: Option<PathBuf>) -> PathBuf {
    root.unwrap_or_else(paths::sotarena_reporter_root)
}

fn write_out(out: Option<&PathBuf>, text: &str) -> Result<()> {
    match out {
        Some(path) => {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
        }
        None => std::io::stdout()
            .write_all(text.as_bytes())
            .map_err(Into::into),
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Elo {
            track,
            version,
            root,
            ladder,
            corrections,
            out,
            json,
            check,
            committed,
        } => {
            let root = benchmark_root(root);
            let (board, location) = leaderboard::compute(
                &root,
                &track,
                version.as_deref(),
                ladder,
                corrections.as_deref(),
            )?;
            let csv = leaderboard::to_csv(&board.rows)?;
            let scores_path = io::resolve_input(&location.dir.join("scores.csv"))
                .context("scores.csv vanished")?;
            if let Some(path) = &json {
                let snapshot = leaderboard::Snapshot::new(&board)?;
                let mut text = serde_json::to_string_pretty(&snapshot)?;
                text.push('\n');
                write_out(Some(path), &text)?;
            }
            write_out(out.as_ref(), &csv)?;
            let top = board.rows.first();
            eprintln!(
                "track={} version={} ladder={} results={} scores_sha256={} corrections={} excluded_rows={} rankable_rows={} datasets={} methods={} rank1={} elo={}",
                board.track_id,
                board.version.as_deref().unwrap_or("(flat)"),
                board.ladder.as_str(),
                scores_path.display(),
                board.scores_sha256,
                board
                    .corrections
                    .as_ref()
                    .map(|c| format!("{}@{}:{}rows", c.file, &c.sha256[..12], c.rows_corrected))
                    .unwrap_or_else(|| "none".to_owned()),
                board.excluded_rows,
                board.rankable_rows,
                board.datasets_scored,
                board.rows.len(),
                top.map(|r| r.method_id.as_str()).unwrap_or("-"),
                top.map(|r| r.elo.as_str()).unwrap_or("-"),
            );
            if check {
                let committed_path =
                    committed.unwrap_or_else(|| location.dir.join("leaderboard.csv"));
                let committed = io::read_text(&committed_path).with_context(|| {
                    format!(
                        "--check needs a committed ladder at {} (pass --committed for a non-published ladder)",
                        committed_path.display()
                    )
                })?;
                if committed != csv {
                    let theirs = leaderboard::parse_csv(&committed)?;
                    let first = theirs
                        .iter()
                        .zip(&board.rows)
                        .position(|(a, b)| a != b)
                        .map(|i| format!("first differing row: {}", i + 2))
                        .unwrap_or_else(|| {
                            format!("row counts {} vs {}", theirs.len(), board.rows.len())
                        });
                    bail!(
                        "recomputed {} ladder differs from the committed {} ({first})",
                        board.ladder.as_str(),
                        committed_path.display()
                    );
                }
                eprintln!(
                    "check=ok (byte-identical to the committed {})",
                    committed_path.display()
                );
            }
            Ok(())
        }
        Command::EmitScores {
            campaign,
            track,
            method_id,
            protocol,
            metric,
            source_revision,
            notes,
            out,
        } => {
            let rows = scores::emit_scores(&scores::Emission {
                track_id: track,
                method_id,
                protocol,
                metric,
                source_revision,
                notes,
                root: campaign,
            })?;
            write_out(out.as_ref(), &scores::write_scores(&rows)?)?;
            eprintln!("emitted {} score rows", rows.len());
            Ok(())
        }
        Command::VerifyResults { dir, json, strict } => {
            let report = verify::verify_results(&dir)?;
            if json {
                let mut text = serde_json::to_string_pretty(&report)?;
                text.push('\n');
                write_out(None, &text)?;
            }
            for file in &report.files {
                if file.status != "ok" {
                    eprintln!("{}: {}", file.status, file.path);
                }
            }
            for path in &report.unlisted {
                eprintln!("unlisted: {path}");
            }
            eprintln!(
                "verify-results {}: ok={} missing={} absent={} mismatched={} unlisted={}",
                report.results_dir,
                report.ok,
                report.missing,
                report.absent,
                report.mismatched,
                report.unlisted.len()
            );
            if !report.passed() {
                bail!("results directory does not match its evidence manifest");
            }
            if strict && !report.passed_strict() {
                bail!("results directory holds files the evidence manifest does not list");
            }
            Ok(())
        }
        Command::Publish {
            track,
            version,
            store,
            results,
            root,
        } => {
            let results = match results {
                Some(dir) => dir,
                None => leaderboard::resolve_results(&benchmark_root(root), &track, None)?.dir,
            };
            let manifest = publish::publish_blocking(&store, &results, &track, &version)?;
            eprintln!(
                "published {} files under {}/{}",
                manifest.files.len(),
                store.trim_end_matches('/'),
                manifest.prefix
            );
            Ok(())
        }
        Command::VerifyPublished {
            track,
            version,
            store,
        } => {
            let manifest = publish::verify_published_blocking(&store, &track, &version)?;
            eprintln!(
                "verified {} published files under {}",
                manifest.files.len(),
                manifest.prefix
            );
            Ok(())
        }
        Command::Fetch {
            task,
            dataset,
            all,
            cache_dir,
            root,
        } => {
            let root = reporter_root(root);
            let config = reporter::load_config(&root)?;
            let datasets = reporter::load_datasets(&root, &config)?;
            let selected =
                fetch::fetch_targets(&config, &datasets, task.as_deref(), dataset.as_deref())?;
            let scope_datasets = match task.as_deref() {
                Some(task) => datasets[task].len(),
                None => datasets.values().map(std::collections::BTreeMap::len).sum(),
            };
            let metadata_only = if dataset.is_some() {
                0
            } else {
                scope_datasets - selected.len()
            };
            if all {
                let summary = fetch::corpus_summary(&datasets);
                eprintln!(
                    "corpus: {} datasets, {} downloadable ({} files, {} bytes), {} metadata-only",
                    summary.datasets,
                    summary.downloadable_datasets,
                    summary.files,
                    summary.bytes,
                    summary.metadata_only_datasets
                );
            }
            if metadata_only > 0 {
                println!("skipping {metadata_only} metadata-only datasets");
            }
            let cache = fetch::external_cache(&root, cache_dir)?;
            let report =
                fetch::fetch_into(&fetch::HttpDownloader, &selected, &cache, &mut |line| {
                    println!("{line}");
                })?;
            println!(
                "verified {} datasets ({} files, {} bytes; {} downloaded) into {}; skipped {metadata_only} metadata-only datasets",
                report.datasets,
                report.files,
                report.bytes,
                report.downloaded,
                cache.display()
            );
            Ok(())
        }
        Command::Report { root, out } => {
            let report = family::render_report(&reporter_root(root))?;
            let mut text = serde_json::to_string_pretty(&report)?;
            text.push('\n');
            write_out(out.as_ref(), &text)
        }
    }
}
