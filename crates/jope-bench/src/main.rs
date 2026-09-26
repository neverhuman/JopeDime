//! `jope-bench`: same-cut benchmark over a dataset tree, then a local Elo track from the reports.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use jope_bench::cuts::{self, FoldPlan};
use jope_bench::emit::{self, Stage};
use jope_bench::load::{discover, find_manifest, load_dataset};
use jope_bench::methods::{self, SearchConfig, RESAMPLED_SEARCH_METHOD_ID};
use jope_bench::protocol::{self, Protocol};
use jope_bench::report::{Report, Stamp};
use jope_bench::task::Task;
use jope_bench::{DEFAULT_FOLDS, DEFAULT_SEED};

/// Ceiling on the default `run --workers`: the agreed per-host CPU budget, so a bare run on a
/// 128-core host takes 20 cores rather than all of them. An explicit `--workers` is not capped.
const DEFAULT_MAX_WORKERS: usize = 20;

/// Rayon threads for `run`: an explicit `--workers` as given (at least 1); otherwise the available
/// cores capped at [`DEFAULT_MAX_WORKERS`], and 1 when the core count is unknown.
fn default_workers(explicit: Option<usize>, available: Option<usize>) -> usize {
    explicit
        .unwrap_or_else(|| available.unwrap_or(1).min(DEFAULT_MAX_WORKERS))
        .max(1)
}

#[derive(Parser)]
#[command(
    name = "jope-bench",
    about = "Same-cut benchmark: baselines and a searched program scored on identical cuts",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Frozen-panel inputs, recipes and planning through the product benchmark contract.
    Panel(jope_bench::panel::cli::BenchmarkCli),
    /// Benchmark every dataset under a directory tree and write one report per dataset.
    Run {
        /// A dataset directory (metadata.json + train.csv) or a directory of them.
        #[arg(long)]
        datasets: PathBuf,
        /// Output directory for `<dataset_hash>.bench-report.json` files.
        #[arg(long)]
        out: PathBuf,
        /// Fold seed (the product pipeline's default).
        #[arg(long, default_value_t = DEFAULT_SEED)]
        seed: u64,
        /// Folds.
        #[arg(long, default_value_t = DEFAULT_FOLDS)]
        folds: usize,
        /// The cut every method is screened on.
        #[arg(long, default_value_t = 0)]
        screen_fold: usize,
        /// How many screened methods are re-scored on every cut.
        #[arg(long, default_value_t = 5)]
        top: usize,
        /// Heads to benchmark as single-head programs (comma-separated; default: every head of
        /// each dataset's task, checked against that task's grammar).
        #[arg(long, value_delimiter = ',')]
        heads: Option<Vec<String>>,
        /// Extra program: `NAME=<dsl>` or `NAME=@<file>` (repeatable).
        #[arg(long = "program")]
        programs: Vec<String>,
        /// External engine as `NAME=<command...>` (repeatable; whitespace-split, no quoting).
        /// The command follows the jope.bench-adapter/v1 contract; `<command> --version` is
        /// probed before any dataset runs. Reference adapters: crates/jope-bench/adapters/
        /// (`python3 crates/jope-bench/adapters/xgboost_adapter.py` and siblings).
        #[arg(long = "adapter")]
        adapters: Vec<String>,
        /// Wall-clock cap per adapter invocation, in seconds.
        #[arg(long, default_value_t = 600.0)]
        adapter_timeout: f64,
        /// Also run the search on the screen cut's training rows and benchmark its program
        /// (regression datasets only; others are skipped when this is set).
        #[arg(long)]
        search: bool,
        /// In the deep stage, re-run the search on every cut's training rows and score each
        /// cut's own program (measures the procedure, not the one program found on the screen
        /// cut). The method is then `afm.lite-race.v1.resampled`.
        #[arg(long, requires = "search")]
        resample_search: bool,
        /// Search population.
        #[arg(long, default_value_t = 32)]
        search_pop: usize,
        /// Search generations.
        #[arg(long, default_value_t = 12)]
        search_gens: usize,
        /// Search wall-clock cap in seconds (0 = none).
        #[arg(long, default_value_t = 0.0)]
        search_seconds: f64,
        /// Search seed.
        #[arg(long, default_value_t = 0x1EA7_BEEF)]
        search_seed: u64,
        /// Rayon threads (default: min(20, available cores); 20 is the agreed per-host budget).
        #[arg(long)]
        workers: Option<usize>,
        /// Git head to stamp into reports (default: `GITHUB_SHA` or none).
        #[arg(long, env = "GITHUB_SHA")]
        git_sha: Option<String>,
    },
    /// Turn a directory of reports into local `jope-arena` tracks (`<track>-screen`, `<track>-deep`).
    Ladder {
        /// Directory holding `*.bench-report.json`.
        #[arg(long)]
        reports: PathBuf,
        /// Root to write `tracks/`, `registry/` and `results/` under.
        #[arg(long)]
        track_root: PathBuf,
        /// Track id base.
        #[arg(long, default_value = "jope-same-cut-v1")]
        track: String,
    },
    /// The dependency-free constant adapter (jope.bench-adapter/v1) the adapter-contract tests
    /// run: `--version`, or `<train.csv> <predict.csv> <out.csv> <task>`.
    #[command(hide = true, disable_help_flag = true)]
    MeanAdapter {
        #[arg(num_args = 0.., trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::MeanAdapter { args } => {
            std::process::exit(jope_bench::adapter::mean_adapter(&args));
        }
        Command::Panel(cli) => {
            if jope_bench::panel::cli::execute(cli) != std::process::ExitCode::SUCCESS {
                bail!("panel command failed");
            }
            Ok(())
        }
        Command::Run {
            datasets,
            out,
            seed,
            folds,
            screen_fold,
            top,
            heads,
            programs,
            adapters,
            adapter_timeout,
            search,
            resample_search,
            search_pop,
            search_gens,
            search_seconds,
            search_seed,
            workers,
            git_sha,
        } => {
            let workers = default_workers(
                workers,
                std::thread::available_parallelism().map(usize::from).ok(),
            );
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build_global()
                .context("rayon pool")?;
            let mut extra_methods = Vec::new();
            for spec in &programs {
                extra_methods.push(methods::program(spec)?);
            }
            for text in &adapters {
                let mut spec = jope_bench::adapter::parse_spec(text, adapter_timeout)?;
                spec.probe()?;
                eprintln!(
                    "adapter {}: {} ({})",
                    spec.name,
                    spec.version.as_deref().unwrap_or("?"),
                    spec.command.join(" ")
                );
                extra_methods.push(methods::adapter(spec));
            }
            let search_config = search.then_some(SearchConfig {
                pop: search_pop,
                gens: search_gens,
                seed: search_seed,
                seconds: search_seconds,
            });
            let protocol = Protocol { screen_fold, top };
            let dirs = discover(&datasets)?;
            if let Some(heads) = &heads {
                // Explicit heads are checked against every discovered dataset's task before
                // any report is written, so a mixed-task tree cannot abort halfway through.
                let mut tasks = std::collections::BTreeSet::new();
                for dir in &dirs {
                    if let Some(manifest) = find_manifest(dir) {
                        if let Ok(row) = jope_datasets::cache::read_manifest(&manifest) {
                            tasks.extend(Task::parse(row.task.as_str()));
                        }
                    }
                }
                for task in tasks {
                    methods::baselines(heads, task).with_context(|| {
                        format!("--heads against the {} datasets", task.as_str())
                    })?;
                }
            }
            std::fs::create_dir_all(&out)?;
            let mut written = 0;
            let mut skipped = 0;
            for dir in dirs {
                let dataset = match load_dataset(&dir) {
                    Ok(dataset) => dataset,
                    Err(error) => {
                        eprintln!("skip {}: {error:#}", dir.display());
                        skipped += 1;
                        continue;
                    }
                };
                let task = dataset.bench_task()?;
                if search_config.is_some() && task != Task::Regression {
                    eprintln!(
                        "skip {} ({}): --search is regression-only and this dataset is {}",
                        dataset.row.id,
                        dir.display(),
                        task.as_str()
                    );
                    skipped += 1;
                    continue;
                }
                let mut methods = match &heads {
                    Some(heads) => methods::baselines(heads, task)
                        .with_context(|| format!("{} ({})", dataset.row.id, task.as_str()))?,
                    None => methods::default_baselines(task)?,
                };
                methods.extend(extra_methods.iter().cloned());
                let plan = FoldPlan::new(&dataset.row.dataset_hash, seed, folds);
                let cuts = match cuts::build(
                    &plan,
                    &dataset.row.dataset_hash,
                    &dataset.train_sha256,
                    dataset.x.len(),
                ) {
                    Ok(cuts) => cuts,
                    Err(error) => {
                        eprintln!("skip {}: {error:#}", dataset.row.id);
                        skipped += 1;
                        continue;
                    }
                };
                if let Some(config) = &search_config {
                    let screen = cuts.get(screen_fold).context("screen fold out of range")?;
                    let (x_train, y_train) =
                        jope_bench::eval::take(&dataset.x, &dataset.y, &screen.train);
                    let mut found = methods::search(&x_train, &y_train, config)?;
                    eprintln!(
                        "{}: search {:.1}s -> {}",
                        dataset.row.id, found.produce_seconds, found.dsl
                    );
                    if resample_search {
                        found.resample = Some(config.clone());
                        found.id = RESAMPLED_SEARCH_METHOD_ID.to_owned();
                    }
                    methods.push(found);
                }
                let started = std::time::Instant::now();
                let results =
                    protocol::run(&dataset.x, &dataset.y, &cuts, methods, &protocol, task)?;
                let report = Report::new(
                    &dataset,
                    task,
                    &plan,
                    &cuts,
                    &protocol,
                    results,
                    Stamp {
                        workers,
                        git_sha: git_sha.clone(),
                        search: search_config.clone(),
                    },
                );
                let path = out.join(report.file_name());
                let sha = report.write(&path)?;
                written += 1;
                eprintln!(
                    "{} [{}] rows={} features={} cuts={} methods={} seconds={:.1} -> {} ({})",
                    report.dataset_id,
                    report.task,
                    report.rows,
                    report.features,
                    report.cuts.len(),
                    report.methods.len(),
                    started.elapsed().as_secs_f64(),
                    path.display(),
                    &sha[..12]
                );
                for m in report.deep_ranked().iter().take(3) {
                    let deep = m.deep.as_ref().expect("ranked");
                    let (ladder_metric, _) = task.ladder_metric();
                    // The ladder metric and the primary score are one metric for the
                    // classification tasks; print it once there.
                    let ladder_column = if ladder_metric == task.score_metric() {
                        String::new()
                    } else {
                        format!(
                            "oof_{ladder_metric}={} ",
                            deep.oof
                                .get(ladder_metric)
                                .map(|v| format!("{v:.6}"))
                                .unwrap_or_default()
                        )
                    };
                    eprintln!(
                        "  deep #{} {:<28} {}oof_{}={} penalised={} seconds={:.2}{}",
                        m.deep_rank.unwrap_or(0),
                        m.method.id,
                        ladder_column,
                        task.score_metric(),
                        deep.oof_score
                            .map(|v| format!("{v:.4}"))
                            .unwrap_or_default(),
                        deep.robust_fitness
                            .map(|v| format!("{v:.4}"))
                            .unwrap_or_default(),
                        deep.seconds,
                        if deep.produce_seconds > 0.0 {
                            format!(" produce={:.1}s", deep.produce_seconds)
                        } else {
                            String::new()
                        }
                    );
                }
            }
            if written == 0 {
                bail!("no dataset was benchmarked ({skipped} skipped)");
            }
            eprintln!(
                "wrote {written} reports under {} ({skipped} datasets skipped)",
                out.display()
            );
            Ok(())
        }
        Command::Ladder {
            reports,
            track_root,
            track,
        } => {
            let mut paths: Vec<PathBuf> = std::fs::read_dir(&reports)
                .with_context(|| format!("read {}", reports.display()))?
                .filter_map(|entry| entry.ok().map(|entry| entry.path()))
                .filter(|path| path.to_string_lossy().ends_with(".bench-report.json"))
                .collect();
            paths.sort();
            if paths.is_empty() {
                bail!("no *.bench-report.json under {}", reports.display());
            }
            let loaded: Vec<(PathBuf, Report)> = paths
                .into_iter()
                .map(|path| Report::read(&path).map(|report| (path, report)))
                .collect::<Result<_>>()?;
            for stage in [Stage::Screen, Stage::Deep] {
                let scores = emit::write_track(&track_root, &track, stage, &loaded, &reports)?;
                eprintln!(
                    "wrote {} ({} datasets); rank it with: jope-arena elo --track {track}-{} --root {}",
                    scores.display(),
                    loaded.len(),
                    stage.suffix(),
                    track_root.display()
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{default_workers, DEFAULT_MAX_WORKERS};

    #[test]
    fn default_run_workers_cap_at_the_host_budget() {
        assert_eq!(default_workers(None, Some(128)), DEFAULT_MAX_WORKERS);
        assert_eq!(default_workers(None, Some(8)), 8);
        assert_eq!(default_workers(None, None), 1);
    }

    #[test]
    fn explicit_run_workers_are_not_capped_but_never_zero() {
        assert_eq!(default_workers(Some(64), Some(128)), 64);
        assert_eq!(default_workers(Some(0), Some(128)), 1);
    }
}
