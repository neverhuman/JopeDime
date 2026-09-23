//! `jope-bench`: same-cut benchmark over a dataset tree, then a local Elo track from the reports.

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use jope_bench::cuts::{self, FoldPlan};
use jope_bench::emit::{self, Stage};
use jope_bench::load::{discover, load_dataset};
use jope_bench::methods::{self, SearchConfig, DEFAULT_HEADS};
use jope_bench::protocol::{self, Protocol};
use jope_bench::report::{Report, Stamp};
use jope_bench::{DEFAULT_FOLDS, DEFAULT_SEED};

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
        /// Heads to benchmark as single-head programs (comma-separated; default: every distinct head).
        #[arg(long, value_delimiter = ',')]
        heads: Option<Vec<String>>,
        /// Extra program: `NAME=<dsl>` or `NAME=@<file>` (repeatable).
        #[arg(long = "program")]
        programs: Vec<String>,
        /// Also run the search on the screen cut's training rows and benchmark its program.
        #[arg(long)]
        search: bool,
        /// In the deep stage, re-run the search on every cut's training rows and score each
        /// cut's own program (measures the procedure, not the one program found on the screen cut).
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
        /// Rayon threads (default: all cores).
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
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Run {
            datasets,
            out,
            seed,
            folds,
            screen_fold,
            top,
            heads,
            programs,
            search,
            resample_search,
            search_pop,
            search_gens,
            search_seconds,
            search_seed,
            workers,
            git_sha,
        } => {
            let workers = workers
                .unwrap_or_else(|| {
                    std::thread::available_parallelism()
                        .map(usize::from)
                        .unwrap_or(1)
                })
                .max(1);
            rayon::ThreadPoolBuilder::new()
                .num_threads(workers)
                .build_global()
                .context("rayon pool")?;
            let heads: Vec<String> =
                heads.unwrap_or_else(|| DEFAULT_HEADS.iter().map(|h| (*h).to_owned()).collect());
            let mut base_methods = methods::baselines(&heads)?;
            for spec in &programs {
                base_methods.push(methods::program(spec)?);
            }
            let search_config = search.then_some(SearchConfig {
                pop: search_pop,
                gens: search_gens,
                seed: search_seed,
                seconds: search_seconds,
            });
            let protocol = Protocol { screen_fold, top };
            let dirs = discover(&datasets)?;
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
                if dataset.task() != "regression" {
                    eprintln!(
                        "skip {} ({}): task {} is not benchmarked yet (regression only)",
                        dataset.row.id,
                        dir.display(),
                        dataset.task()
                    );
                    skipped += 1;
                    continue;
                }
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
                let mut methods = base_methods.clone();
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
                    }
                    methods.push(found);
                }
                let started = std::time::Instant::now();
                let results = protocol::run(&dataset.x, &dataset.y, &cuts, methods, &protocol)?;
                let report = Report::new(
                    &dataset,
                    &plan,
                    &cuts,
                    &protocol,
                    search_config.clone(),
                    results,
                    Stamp {
                        workers,
                        git_sha: git_sha.clone(),
                    },
                );
                let path = out.join(report.file_name());
                let sha = report.write(&path)?;
                written += 1;
                eprintln!(
                    "{} rows={} features={} cuts={} methods={} seconds={:.1} -> {} ({})",
                    report.dataset_id,
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
                    eprintln!(
                        "  deep #{} {:<24} oof_rmse={} oof_r2={} fitness={} seconds={:.2}{}",
                        m.deep_rank.unwrap_or(0),
                        m.method.id,
                        deep.oof_rmse.map(|v| format!("{v:.6}")).unwrap_or_default(),
                        deep.oof_r2.map(|v| format!("{v:.4}")).unwrap_or_default(),
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
