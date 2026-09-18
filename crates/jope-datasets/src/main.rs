//! `jope-datasets`: ingest, fetch, verify and link content-addressed datasets.

use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use jope_datasets::cache::{link_from, verify_cache, Fetcher};
use jope_datasets::catalog::Task;
use jope_datasets::ingest::{self, Source};
use jope_datasets::paths::{now_compact, offline_env, Cache};
use jope_datasets::repo::verify_repo;
use jope_datasets::store::Store;
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "jope-datasets",
    version,
    about = "JOPE-Dime dataset catalog tool"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum Which {
    Catalog,
    Collection,
}

#[derive(Subcommand)]
enum Cmd {
    /// Enumerate, hash and de-duplicate a source tree; upload unless --dry-run.
    Ingest {
        #[arg(long, value_enum)]
        source: Source,
        #[arg(long)]
        root: PathBuf,
        /// Hash and report only; upload nothing.
        #[arg(long)]
        dry_run: bool,
        /// Report directory (default: $JOPE_HOME/datasets/ingest/<timestamp>).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Skip blobs the report directory's ledger already verified.
        #[arg(long)]
        resume: bool,
        #[arg(long, default_value_t = 16)]
        jobs: usize,
        /// Hash at most this many candidates (smoke tests).
        #[arg(long)]
        limit: Option<usize>,
    },
    /// Download a dataset id, every dataset of a task, or a collection into the cache.
    Fetch {
        selector: String,
        /// Serve only the cache (also JOPE_OFFLINE=1).
        #[arg(long)]
        offline: bool,
        /// Print the registry directory (JOPE_REGISTRY) after fetching.
        #[arg(long)]
        print_registry: bool,
    },
    /// Re-hash cached blobs and check links; or check a committed dataset repository.
    Verify {
        /// Check blob sizes instead of re-hashing.
        #[arg(long)]
        quick: bool,
        /// Validate every cached manifest against the schema and licence rule.
        #[arg(long)]
        catalog: bool,
        /// The committed dataset directory (datasets/test) to check for CI.
        #[arg(long)]
        repo: Option<PathBuf>,
        /// Optional SOTArena registry/datasets.json to cross-check sha256s.
        #[arg(long)]
        registry: Option<PathBuf>,
    },
    /// Seed the cache from a local tree by hashing (no download).
    Link {
        #[arg(long)]
        from: PathBuf,
        /// Only these ids / directory names (comma separated).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Task for registry-layout directories whose meta.json has no task_type.
        #[arg(long)]
        task: Option<String>,
    },
    /// Print the JSON Schema derived from the Rust types.
    Schema { which: Which },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest {
            source,
            root,
            dry_run,
            out,
            resume,
            jobs,
            limit,
        } => {
            let cache = Cache::from_env();
            let out = out.unwrap_or_else(|| cache.ingest_dir().join(now_compact()));
            let opts = ingest::Options {
                source,
                root,
                out,
                dry_run,
                resume,
                jobs,
                limit,
                store_url: None,
            };
            let summary = ingest::run(&opts)?;
            print!("{}", ingest::describe(&summary));
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Fetch {
            selector,
            offline,
            print_registry,
        } => {
            let cache = Cache::from_env();
            let offline = offline || offline_env();
            let store = if offline {
                None
            } else {
                Some(Store::from_env()?)
            };
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            let fetcher = Fetcher {
                cache: &cache,
                store: store.as_ref(),
            };
            let report = rt.block_on(fetcher.fetch(&selector))?;
            eprintln!(
                "fetched {} datasets: {} files downloaded, {} already cached, {} errors",
                report.datasets,
                report.files_fetched,
                report.files_cached,
                report.errors.len()
            );
            for e in &report.errors {
                eprintln!("error: {e}");
            }
            if print_registry {
                println!("{}", cache.registry_dir().display());
            }
            Ok(if report.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Cmd::Verify {
            quick,
            catalog,
            repo,
            registry,
        } => {
            let mut failed = false;
            if let Some(repo) = repo {
                let rep = verify_repo(&repo, registry.as_deref())?;
                for w in &rep.warnings {
                    eprintln!("warning: {w}");
                }
                for e in &rep.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "repo {}: {} datasets, {} bytes, collection {}, {} errors, {} warnings",
                    repo.display(),
                    rep.datasets,
                    rep.total_bytes,
                    rep.collection.as_deref().unwrap_or("-"),
                    rep.errors.len(),
                    rep.warnings.len()
                );
                failed |= !rep.errors.is_empty();
            } else {
                let cache = Cache::from_env();
                let rep = verify_cache(&cache, quick);
                for e in &rep.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "cache {}: {} blobs, {} datasets, {} errors",
                    cache.root.display(),
                    rep.blobs,
                    rep.datasets,
                    rep.errors.len()
                );
                failed |= !rep.errors.is_empty();
                if catalog {
                    let rows = jope_datasets::cache::cached_rows(&cache);
                    let mut n = 0;
                    for row in &rows {
                        for e in row.invariant_errors() {
                            eprintln!("error: {}: {e}", row.id);
                            n += 1;
                        }
                    }
                    println!("catalog: {} manifests, {} rule errors", rows.len(), n);
                    failed |= n > 0;
                }
            }
            Ok(if failed {
                ExitCode::FAILURE
            } else {
                ExitCode::SUCCESS
            })
        }
        Cmd::Link { from, only, task } => {
            let cache = Cache::from_env();
            let task = match task {
                Some(t) => {
                    Some(Task::parse(&t).ok_or_else(|| anyhow::anyhow!("unknown task {t}"))?)
                }
                None => None,
            };
            let rep = link_from(&cache, &from, &only, task)?;
            for e in &rep.errors {
                eprintln!("error: {e}");
            }
            println!(
                "linked {} datasets into {} ({} skipped, collections: {:?}, {} errors)",
                rep.datasets,
                cache.registry_dir().display(),
                rep.skipped,
                rep.collections,
                rep.errors.len()
            );
            Ok(if rep.errors.is_empty() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
        Cmd::Schema { which } => {
            let v = match which {
                Which::Catalog => jope_datasets::catalog::derived_catalog_schema(),
                Which::Collection => jope_datasets::catalog::derived_collection_schema(),
            };
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(ExitCode::SUCCESS)
        }
    }
}
