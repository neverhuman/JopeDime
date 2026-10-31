//! The `jope-models` command line (also mounted as `jope models …` by the umbrella CLI).

use crate::cache::ModelsCache;
use crate::fetch::{FetchOptions, Fetcher};
use crate::index::Repo;
use anyhow::{anyhow, Result};
use clap::{Parser, Subcommand};
use jope_datasets::paths::offline_env;
use std::path::PathBuf;
use std::process::ExitCode;

/// Parsed command-line options for `jope-models`.
#[derive(Parser)]
#[command(
    name = "jope-models",
    version,
    about = "JOPE-Dime model bundles: fetch, verify, publish and locate content-addressed weights",
    long_about = "Bundles live in JOPE_MODELS_URL (default s3://veox-jopedime-models-use2) as \
blobs/sha256/<sha256> plus bundles/<id>/manifest.json; the checkout commits only manifests \
(models/bundles/<id>.json, models/manifest.json, models/PINNED). Fetched files land under \
$JOPE_HOME/models (default ${XDG_CACHE_HOME:-~/.cache}/jope/models); JOPE_OFFLINE=1 serves \
only the cache."
)]
pub struct Cli {
    /// Checkout whose `models/` directory contains committed manifests and `PINNED`.
    #[arg(long, global = true, default_value = ".")]
    pub repo: PathBuf,
    /// Command to execute.
    #[command(subcommand)]
    pub cmd: Cmd,
}

/// Supported `jope-models` commands.
#[derive(Subcommand)]
pub enum Cmd {
    /// Download a bundle (or every bundle in models/PINNED) into the cache, verifying sha256.
    Fetch {
        /// Bundle identifier to fetch when `--pinned` is not set.
        bundle_id: Option<String>,
        /// Fetch every bundle listed in the repository's models/PINNED file.
        #[arg(long, conflicts_with = "bundle_id")]
        pinned: bool,
        /// Only these files (exact bundle path or a directory prefix); repeatable.
        #[arg(long = "component")]
        components: Vec<String>,
        /// Serve only the cache (also JOPE_OFFLINE=1).
        #[arg(long)]
        offline: bool,
        /// Also fetch files whose licence says `redistribution: prohibited`.
        #[arg(long)]
        accept_restricted: bool,
    },
    /// Re-hash a populated cache; or check the committed manifests without any download.
    Verify {
        /// Bundle identifier to verify when `--all` is not set.
        bundle_id: Option<String>,
        /// Every cached bundle plus the blob directory.
        #[arg(long, conflicts_with = "bundle_id")]
        all: bool,
        /// Check sizes instead of re-hashing.
        #[arg(long)]
        quick: bool,
        /// Validate the repository's models/manifest.json, models/bundles/*.json, and models/PINNED against the schema and
        /// each other; touches no network and no cache.
        #[arg(long, conflicts_with_all = ["bundle_id", "all", "quick"])]
        manifests_only: bool,
    },
    /// Upload a bundle's blobs create-only, then its manifest last (maintainers).
    Publish {
        /// The bundle manifest (jope.model-bundle/v2).
        manifest: PathBuf,
        /// Directory holding every manifest `path`.
        #[arg(long)]
        from: PathBuf,
        /// Store URL (default JOPE_MODELS_URL).
        #[arg(long)]
        store: Option<String>,
        /// Write the machine-readable verified publication receipt JSON to this path (create-only).
        #[arg(long)]
        receipt_out: Option<PathBuf>,
        /// Require a backend physical VersionId on every published object.
        #[arg(long)]
        require_version_id: bool,
    },
    /// Print the cache path of a bundle (or of one file in it); fails when it is not fetched.
    Path {
        /// Bundle identifier whose cached path is requested.
        bundle_id: String,
        /// Optional bundle-relative file path within the cached bundle.
        path: Option<String>,
    },
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

fn runtime() -> Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

fn status(failed: bool) -> ExitCode {
    if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Execute a parsed command and return its process exit status.
pub fn run(cli: Cli) -> Result<ExitCode> {
    let repo = Repo::detect(&cli.repo);
    match cli.cmd {
        Cmd::Fetch {
            bundle_id,
            pinned,
            components,
            offline,
            accept_restricted,
        } => {
            let ids: Vec<String> = if pinned {
                let repo = repo.as_ref().ok_or_else(|| {
                    anyhow!(
                        "--pinned needs <repo>/models/PINNED (--repo {})",
                        cli.repo.display()
                    )
                })?;
                repo.read_pinned()?
            } else {
                vec![bundle_id.ok_or_else(|| anyhow!("give a bundle id or --pinned"))?]
            };
            let cache = ModelsCache::from_env();
            let offline = offline || offline_env();
            let store = if offline {
                None
            } else {
                Some(crate::open_store(None)?)
            };
            let fetcher = Fetcher {
                cache: &cache,
                store: store.as_ref(),
                repo: repo.as_ref(),
            };
            let opts = FetchOptions {
                components,
                accept_restricted,
            };
            let rt = runtime()?;
            let mut failed = false;
            let mut out = std::io::stdout().lock();
            for id in &ids {
                let report = rt.block_on(fetcher.fetch(id, &opts, &mut out))?;
                for e in &report.errors {
                    eprintln!("error: {e}");
                }
                eprintln!(
                    "fetched {}: {} files downloaded ({} bytes), {} already cached, {} errors{}",
                    report.bundle_id,
                    report.files_fetched,
                    report.bytes,
                    report.files_cached,
                    report.errors.len(),
                    if report.restricted_accepted.is_empty() {
                        String::new()
                    } else {
                        format!(
                            "; restricted accepted: {}",
                            report.restricted_accepted.join(", ")
                        )
                    }
                );
                failed |= !report.errors.is_empty();
                if report.errors.is_empty() {
                    println!("{}", cache.bundle_dir(id).display());
                }
            }
            Ok(status(failed))
        }
        Cmd::Verify {
            bundle_id,
            all,
            quick,
            manifests_only,
        } => {
            if manifests_only {
                let repo = repo.ok_or_else(|| {
                    anyhow!("no models/manifest.json under {}", cli.repo.display())
                })?;
                let r = crate::repo::verify_manifests(&repo)?;
                for (id, files, bytes) in &r.bundles {
                    println!("ok {id} ({files} files, {bytes} bytes)");
                }
                for e in &r.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "manifests: {} bundles, {} pinned, {} errors",
                    r.bundles.len(),
                    r.pinned.len(),
                    r.errors.len()
                );
                return Ok(status(!r.errors.is_empty()));
            }
            let cache = ModelsCache::from_env();
            let ids = match bundle_id {
                Some(id) => vec![id],
                None if all => cache.list(),
                None => return Err(anyhow!("give a bundle id, --all or --manifests-only")),
            };
            let mut failed = false;
            for id in &ids {
                let r = cache.verify(id, quick)?;
                for e in &r.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "{}: {} files verified, {} not fetched, {} errors",
                    r.bundle_id,
                    r.files_checked,
                    r.files_absent,
                    r.errors.len()
                );
                failed |= !r.errors.is_empty();
            }
            if all {
                let b = cache.blobs().verify(quick);
                for e in &b.errors {
                    eprintln!("error: {e}");
                }
                println!(
                    "blobs {}: {} blobs, {} errors",
                    cache.blobs().dir.display(),
                    b.blobs,
                    b.errors.len()
                );
                failed |= !b.errors.is_empty();
            }
            Ok(status(failed))
        }
        Cmd::Publish {
            manifest,
            from,
            store,
            receipt_out,
            require_version_id,
        } => {
            let store = crate::open_store(store.as_deref())?;
            let rt = runtime()?;
            let opts = if require_version_id {
                jope_datasets::verified_object::VerificationOptions::require_physical_version_id()
            } else {
                jope_datasets::verified_object::VerificationOptions::default()
            };
            let r = rt.block_on(crate::publish::publish_with_options(
                &store, &manifest, &from, opts,
            ))?;
            if let Some(out_path) = receipt_out {
                if let Some(parent) = out_path.parent() {
                    if !parent.as_os_str().is_empty() {
                        std::fs::create_dir_all(parent)?;
                    }
                }
                let mut file = std::fs::OpenOptions::new()
                    .write(true)
                    .create_new(true)
                    .open(&out_path)
                    .map_err(|e| anyhow!("create receipt {}: {e}", out_path.display()))?;
                let mut bytes = serde_json::to_vec_pretty(&r)?;
                bytes.push(b'\n');
                std::io::Write::write_all(&mut file, &bytes)?;
            }
            println!(
                "published {} to {}: {} blobs uploaded ({} bytes), {} already present, manifest {}",
                r.bundle_id,
                store.url,
                r.blobs_created,
                r.bytes_uploaded,
                r.blobs_existing,
                r.manifest
            );
            Ok(ExitCode::SUCCESS)
        }
        Cmd::Path { bundle_id, path } => {
            let cache = ModelsCache::from_env();
            let p = match &path {
                Some(rel) => cache.file_path(&bundle_id, rel),
                None => cache.bundle_dir(&bundle_id),
            };
            println!("{}", p.display());
            if !p.exists() {
                eprintln!(
                    "error: {} is not fetched (run `jope models fetch {bundle_id}`)",
                    p.display()
                );
                return Ok(ExitCode::FAILURE);
            }
            Ok(ExitCode::SUCCESS)
        }
    }
}
