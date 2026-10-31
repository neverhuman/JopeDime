//! The `jope-datasets` command line (also mounted as `jope datasets …` by the umbrella CLI):
//! ingest, fetch, verify and link content-addressed datasets.

use crate::cache::{link_from, verify_cache, Fetcher};
use crate::catalog::Task;
use crate::ingest::{self, Source};
use crate::paths::{now_compact, offline_env, Cache};
use crate::regression_release::{self, FamilyRegistry, RegressionRelease};
use crate::repo::verify_repo;
use crate::store::Store;
use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "jope-datasets",
    version,
    about = "JOPE-Dime dataset catalog tool"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
pub enum Which {
    Catalog,
    Collection,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// Enumerate, hash and de-duplicate a source tree; upload unless --dry-run.
    Ingest {
        #[arg(long, value_enum)]
        source: Source,
        #[arg(long)]
        root: PathBuf,
        /// Hash and report only; upload nothing.
        #[arg(long)]
        dry_run: bool,
        /// Report directory (default: `$JOPE_HOME/datasets/ingest/<timestamp>`).
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
        /// Resolve against exactly this published catalog version (`v<N>`);
        /// `LATEST`, cached rows and cached manifests are never consulted.
        #[arg(long, requires = "catalog_sha256")]
        catalog_version: Option<String>,
        /// SHA-256 of the pinned version's `catalog.jsonl` bytes.
        #[arg(long, requires = "catalog_version")]
        catalog_sha256: Option<String>,
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
    /// Derive or verify the immutable ten-regression release. Reads only exact
    /// pre-staged local objects; never fetches, never substitutes.
    RegressionRelease {
        #[command(subcommand)]
        action: ReleaseAction,
    },
    /// Record the historical GPU-study staging from its retained evidence and
    /// verify input closures against the corrected fitting-only protocol.
    /// Local reads only; nothing is fetched or rewritten.
    StagingDisposition {
        #[command(subcommand)]
        action: StagingAction,
    },
}

#[derive(Subcommand)]
pub enum StagingAction {
    /// Derive the disposition and its corrected protocol; outputs must not exist.
    Derive {
        /// Retained study directory (plan, journal, manifests, projected inputs).
        #[arg(long)]
        study_root: PathBuf,
        /// Repository root holding docs/internal/core-ip/data/.
        #[arg(long)]
        repository_root: PathBuf,
        #[arg(long)]
        disposition: PathBuf,
        #[arg(long)]
        protocol: PathBuf,
    },
    /// Reopen both documents; optionally re-derive them from the evidence and
    /// replay-check an input closure, or a corrected run with its closure.
    /// A rejected closure or run exits 1.
    Verify {
        #[arg(long)]
        disposition: PathBuf,
        #[arg(long)]
        protocol: PathBuf,
        #[arg(long, requires = "repository_root")]
        study_root: Option<PathBuf>,
        #[arg(long, requires = "study_root")]
        repository_root: Option<PathBuf>,
        /// A pinned fitting-input manifest to check against the corrected protocol.
        #[arg(long)]
        closure: Option<PathBuf>,
        /// A corrected run document (requires --closure).
        #[arg(long, requires = "closure")]
        run: Option<PathBuf>,
    },
    /// Emit the producer's corrected-run attestation for a fitting-input
    /// manifest it projected (W1-T07), under the committed corrected protocol.
    /// It declares a fitting-only projection: recorded and required by the
    /// inference planner, not independently proven. The closure is checked
    /// first, so a historical closure cannot be attested; the output must not
    /// exist.
    Attest {
        /// The fitting-input manifest the producer projected.
        #[arg(long)]
        closure: PathBuf,
        /// The run's own admitted execution window (experiment sha256), if any.
        #[arg(long)]
        window: Option<String>,
        #[arg(long)]
        output: PathBuf,
    },
}

fn create_new(path: &std::path::Path, bytes: &[u8]) -> Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| anyhow::anyhow!("{}: {e}; records are immutable", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn run_staging(action: StagingAction) -> Result<ExitCode> {
    use crate::staging_disposition::{
        attest_corrected_run, committed, derive_disposition, verify_closure, verify_corrected_run,
        CorrectedProtocol, HistoricalDisposition, ProjectionAttestation,
    };
    match action {
        StagingAction::Attest {
            closure,
            window,
            output,
        } => {
            let (d, p) = committed()?;
            let manifest = std::fs::read(&closure)?;
            let run = attest_corrected_run(
                &p,
                &manifest,
                ProjectionAttestation {
                    inputs_pre_projected: true,
                    outer_scoring_labels_parsed: false,
                    development_targets_parsed_outside_fitting_rows: false,
                    sealed_tables_opened: false,
                },
                window,
            )?;
            let bytes = run.encode()?;
            let accepted = verify_corrected_run(&d, &p, &bytes, &manifest)?;
            create_new(&output, &bytes)?;
            println!(
                "{}",
                serde_json::json!({"schema":"jope.corrected-run-attestation/v1",
                    "run_id":run.run_id,"run_sha256":crate::hash::sha256_hex(&bytes),
                    "protocol_id":p.protocol_id,"manifest_sha256":accepted["manifest_sha256"],
                    "objects_checked":accepted["objects_checked"],
                    "attestation":"producer-declared; recorded and required, not independently proven",
                    "network_access":false})
            );
            Ok(ExitCode::SUCCESS)
        }
        StagingAction::Derive {
            study_root,
            repository_root,
            disposition,
            protocol,
        } => {
            let derived = derive_disposition(&study_root, &repository_root)?;
            let corrected = CorrectedProtocol::derive(&derived)?;
            let (d, p) = (derived.encode()?, corrected.encode()?);
            create_new(&disposition, &d)?;
            create_new(&protocol, &p)?;
            println!(
                "{}",
                serde_json::json!({"schema":"jope.staging-disposition-derivation/v1",
                    "disposition_id":derived.disposition_id,"disposition_sha256":crate::hash::sha256_hex(&d),
                    "protocol_id":corrected.protocol_id,"protocol_sha256":crate::hash::sha256_hex(&p),
                    "historical_closure_manifest_sha256":derived.input_closure.manifest_sha256,
                    "forbidden_closure_objects":corrected.forbidden_closure_sha256.len(),
                    "contamination_inferred":derived.classification.contamination_inferred,
                    "network_access":false})
            );
            Ok(ExitCode::SUCCESS)
        }
        StagingAction::Verify {
            disposition,
            protocol,
            study_root,
            repository_root,
            closure,
            run,
        } => {
            let d_bytes = std::fs::read(&disposition)?;
            let d = HistoricalDisposition::open(&d_bytes)?;
            let p = CorrectedProtocol::open(&std::fs::read(&protocol)?, &d)?;
            let mut report = serde_json::json!({"schema":"jope.staging-disposition-verification/v1",
                "disposition_id":d.disposition_id,"protocol_id":p.protocol_id,"network_access":false});
            if let (Some(study), Some(repository)) = (study_root, repository_root) {
                let rederived = derive_disposition(&study, &repository)?.encode()?;
                anyhow::ensure!(
                    rederived == d_bytes,
                    "re-derivation from the retained evidence does not reproduce the disposition"
                );
                report["rederived_bytes_equal"] = serde_json::json!(true);
            }
            let mut accepted = true;
            if let Some(closure) = closure {
                let manifest = std::fs::read(&closure)?;
                report["closure_sha256"] = serde_json::json!(crate::hash::sha256_hex(&manifest));
                let decision = match &run {
                    Some(run) => verify_corrected_run(&d, &p, &std::fs::read(run)?, &manifest),
                    None => verify_closure(&p, &manifest),
                };
                match decision {
                    Ok(value) => report["closure_decision"] = value,
                    Err(e) => {
                        accepted = false;
                        report["closure_decision"] =
                            serde_json::json!({"accepted":false,"reason":format!("{e:#}")});
                    }
                }
            }
            println!("{report}");
            Ok(if accepted {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            })
        }
    }
}

#[derive(Subcommand)]
pub enum ReleaseAction {
    /// Derive a new release document; the output path must not exist.
    Derive {
        /// Frozen panel manifest (benchmarks/panel.json).
        #[arg(long)]
        panel: PathBuf,
        /// Repository-relative preparation implementation (benchmarks/prepare.py).
        #[arg(long)]
        preparation_source: PathBuf,
        /// Canonical family/exposure registry document.
        #[arg(long)]
        registry: PathBuf,
        /// Preparation root holding downloads/, metadata/, development/, splits/, sealed/.
        #[arg(long)]
        staged_root: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Reopen a release byte-exactly and re-derive it from the staged objects.
    Verify {
        #[arg(long)]
        release: PathBuf,
        #[arg(long)]
        registry: PathBuf,
        #[arg(long)]
        panel: PathBuf,
        #[arg(long)]
        preparation_source: PathBuf,
        #[arg(long)]
        staged_root: PathBuf,
    },
}

fn release_sources(
    panel: &std::path::Path,
    preparation_source: &std::path::Path,
    registry: &std::path::Path,
) -> Result<(Vec<u8>, String, Vec<u8>, FamilyRegistry)> {
    let source_path = preparation_source
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("preparation source path is not UTF-8"))?
        .to_owned();
    let registry = FamilyRegistry::open(&std::fs::read(registry)?)?;
    Ok((
        std::fs::read(panel)?,
        source_path,
        std::fs::read(preparation_source)?,
        registry,
    ))
}

fn run_release(action: ReleaseAction) -> Result<ExitCode> {
    match action {
        ReleaseAction::Derive {
            panel,
            preparation_source,
            registry,
            staged_root,
            output,
        } => {
            let (panel, source_path, source, registry) =
                release_sources(&panel, &preparation_source, &registry)?;
            let staged = regression_release::load_staged(&staged_root, &panel)?;
            let release = regression_release::derive(&regression_release::ReleaseSources {
                panel: &panel,
                preparation_source_path: &source_path,
                preparation_source: &source,
                registry: &registry,
                staged: &staged,
            })?;
            let bytes = release.encode()?;
            {
                use std::io::Write;
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&output)
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "{}: {e}; releases are immutable, choose a new path",
                            output.display()
                        )
                    })?;
                file.write_all(&bytes)?;
                file.sync_all()?;
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema": "jope.regression-release-derivation/v1",
                    "release_id": release.release_id,
                    "release_sha256": crate::hash::sha256_hex(&bytes),
                    "members": release.members.iter().map(|m| &m.dataset).collect::<Vec<_>>(),
                    "exclusions": release.exclusions.len(),
                    "family_registry_sha256": release.family_registry_sha256,
                    "panel_sha256": release.derived_from.panel_sha256,
                    "sealed_partitions_parsed": false,
                    "network_access": false,
                })
            );
            Ok(ExitCode::SUCCESS)
        }
        ReleaseAction::Verify {
            release,
            registry,
            panel,
            preparation_source,
            staged_root,
        } => {
            let release_bytes = std::fs::read(&release)?;
            let release = RegressionRelease::open(&release_bytes)?;
            let (panel, source_path, source, registry) =
                release_sources(&panel, &preparation_source, &registry)?;
            anyhow::ensure!(
                registry.sha256()? == release.family_registry_sha256,
                "registry is not the one bound by the release"
            );
            regression_release::check_against_panel(&release, &panel)?;
            let staged = regression_release::load_staged(&staged_root, &panel)?;
            for member in &release.members {
                let objects = staged
                    .get(&member.dataset)
                    .ok_or_else(|| anyhow::anyhow!("{}: staged objects absent", member.dataset))?;
                release.verify_staged(&member.dataset, objects)?;
            }
            let rederived = regression_release::derive(&regression_release::ReleaseSources {
                panel: &panel,
                preparation_source_path: &source_path,
                preparation_source: &source,
                registry: &registry,
                staged: &staged,
            })?;
            anyhow::ensure!(
                rederived.encode()? == release_bytes,
                "re-derivation does not reproduce the release bytes"
            );
            let mut blocked = serde_json::Map::new();
            for member in &release.members {
                blocked.insert(
                    member.dataset.clone(),
                    serde_json::json!(registry.never_seen_claim_blockers(&member.dataset)?.len()),
                );
            }
            println!(
                "{}",
                serde_json::json!({
                    "schema": "jope.regression-release-verification/v1",
                    "release_id": release.release_id,
                    "release_sha256": crate::hash::sha256_hex(&release_bytes),
                    "members_verified": release.members.len(),
                    "rederived_bytes_equal": true,
                    "never_seen_claim_blockers": blocked,
                    "sealed_partitions_parsed": false,
                    "network_access": false,
                })
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

/// Run a parsed command line; errors are printed as `error: …` and exit with status 2, a
/// failed check exits with status 1.
pub fn exec(cli: Cli) -> ExitCode {
    match run(cli) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(2)
        }
    }
}

pub fn run(cli: Cli) -> Result<ExitCode> {
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
            catalog_version,
            catalog_sha256,
        } => {
            let pin = match (catalog_version, catalog_sha256) {
                (Some(version), Some(sha256)) => {
                    Some(crate::cache::CatalogPin::new(&version, &sha256)?)
                }
                _ => None,
            };
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
            let report = match &pin {
                Some(pin) => rt.block_on(fetcher.fetch_pinned(&selector, pin))?,
                None => rt.block_on(fetcher.fetch(&selector))?,
            };
            if let Some(pin) = &report.catalog {
                eprintln!("catalog pinned: {} sha256 {}", pin.version, pin.sha256);
            }
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
                    let rows = crate::cache::cached_rows(&cache);
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
                Which::Catalog => crate::catalog::derived_catalog_schema(),
                Which::Collection => crate::catalog::derived_collection_schema(),
            };
            println!("{}", serde_json::to_string_pretty(&v)?);
            Ok(ExitCode::SUCCESS)
        }
        Cmd::RegressionRelease { action } => run_release(action),
        Cmd::StagingDisposition { action } => run_staging(action),
    }
}
