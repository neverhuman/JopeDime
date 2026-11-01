//! `jope-dist`: build, sign, verify and report on `jopedime` packages (deploy/appliance/README.md).
//!
//! | Command | Does |
//! |---|---|
//! | `lock` | derive `deploy/appliance/models.lock.json` from the profiles, or `--check` it |
//! | `fixture` | write the deterministic CI model files (or regenerate their manifests) |
//! | `notices` | the third-party notice inventory for the appliance's closure |
//! | `audit` | the dependency fence for the shipped package and features |
//! | `pack` | executable + locked assets → a development-signed package, or a release stage |
//! | `seal` | stage + cosign signature of `signing-message.bin` → the release package |
//! | `verify` | every byte of a package, against trusted keys independent of the package |
//! | `inspect` | manifest, footer and signer |
//! | `size-report` | region and asset sizes; budget check |
//! | `sbom-models` | CycloneDX components for the packaged assets |
//! | `provenance` | the SLSA v1 provenance statement bound to the sealed file's sha256 |
use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use jope_dist::{audit, fixture, notices, pack, profile, report};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

#[derive(Parser)]
#[command(
    name = "jope-dist",
    version,
    about = "Build-time packer and verifier for the jopedime appliance"
)]
struct Cli {
    /// Repository root (profiles, lock, LICENSE and Cargo.lock are resolved against it).
    #[arg(long, global = true, default_value = ".")]
    repo: PathBuf,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Derive the model lock from every profile, or fail if the committed lock is stale.
    Lock {
        #[arg(long)]
        check: bool,
    },
    /// Write the fixture model files under `<models>/bundles/`.
    Fixture {
        #[arg(long, required_unless_present = "write_manifests")]
        models: Option<PathBuf>,
        /// Regenerate deploy/appliance/fixtures/bundles/*.json instead.
        #[arg(long)]
        write_manifests: bool,
    },
    /// Write the third-party notice inventory.
    Notices {
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value = "x86_64-unknown-linux-gnu")]
        target: String,
        #[arg(long, value_delimiter = ',', default_value = "embedded")]
        features: Vec<String>,
    },
    /// Check the dependency fence.
    Audit {
        #[arg(long, default_value = "deploy/appliance/dependencies-deny.toml")]
        policy: PathBuf,
    },
    /// Pack an executable and a profile's locked assets.
    Pack {
        #[arg(long)]
        profile: PathBuf,
        #[arg(long)]
        code: PathBuf,
        #[arg(long)]
        models: PathBuf,
        #[arg(long)]
        notices: PathBuf,
        #[arg(long)]
        source_sha: String,
        #[arg(long, default_value = "x86_64-unknown-linux-gnu")]
        target: String,
        /// The release key's public PEM (default: docs/release/signing.pub, compiled in).
        #[arg(long)]
        release_key: Option<PathBuf>,
        /// Development scope: the signed package to write.
        #[arg(long, conflicts_with_all = ["stage", "rehearsal"])]
        out: Option<PathBuf>,
        /// Release scopes: the stage directory to create for external signing.
        #[arg(long, conflicts_with = "rehearsal")]
        stage: Option<PathBuf>,
        /// Internal full-profile inspection only; never signable or sealable.
        #[arg(long)]
        rehearsal: Option<PathBuf>,
        #[arg(long)]
        receipt: Option<PathBuf>,
    },
    /// Seal a stage with the release signature of its signing message.
    Seal {
        #[arg(long)]
        code: PathBuf,
        #[arg(long)]
        stage: PathBuf,
        /// `cosign sign-blob --output-signature` file (base64 DER), raw DER, or r||s.
        #[arg(long)]
        signature: PathBuf,
        #[arg(long)]
        release_key: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        receipt: Option<PathBuf>,
    },
    /// Verify every byte of a package.
    Verify(ReadArgs),
    /// Print a package's manifest, footer and signer.
    Inspect(ReadArgs),
    /// Report sizes; with --budget, fail above budget + tolerance.
    SizeReport {
        #[command(flatten)]
        read: ReadArgs,
        #[arg(long)]
        budget: Option<PathBuf>,
    },
    /// Write CycloneDX components for the packaged assets.
    SbomModels {
        #[command(flatten)]
        read: ReadArgs,
        #[arg(long)]
        out: PathBuf,
    },
    /// Write the SLSA v1 provenance statement.
    Provenance {
        #[command(flatten)]
        read: ReadArgs,
        #[arg(long)]
        tag: String,
        #[arg(long)]
        builder_image: String,
        #[arg(long)]
        builder_id: String,
        #[arg(long, default_value = "dist")]
        cargo_profile: String,
        #[arg(long, default_value = "embedded")]
        features: String,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Args)]
struct ReadArgs {
    #[arg(long)]
    file: PathBuf,
    /// An additional trusted public key (PEM); repeatable.
    #[arg(long = "trusted-key")]
    trusted_keys: Vec<PathBuf>,
    /// Trust the development key (fixture packages).
    #[arg(long)]
    development_key: bool,
    /// Do not trust the compiled-in release key.
    #[arg(long)]
    no_release_key: bool,
}

impl ReadArgs {
    fn keys(&self) -> Result<Vec<jope_assets::TrustedKey>> {
        report::trusted(
            &self.trusted_keys,
            self.development_key,
            !self.no_release_key,
        )
    }
}

fn emit(value: &Value, out: Option<&Path>) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    match out {
        Some(path) => std::fs::write(path, text)?,
        None => print!("{text}"),
    }
    Ok(())
}

fn run(cli: Cli) -> Result<bool> {
    let repo = cli.repo;
    let profiles = repo.join("deploy/appliance/profiles");
    let lock_path = repo.join("deploy/appliance/models.lock.json");
    let manifest = repo.join("Cargo.toml");
    match cli.command {
        Command::Lock { check } => {
            let lock = profile::derive_lock(&profiles, &repo)?;
            if check {
                let existing = std::fs::read_to_string(&lock_path)
                    .map_err(|e| anyhow::anyhow!("read {}: {e}", lock_path.display()))?;
                if existing != lock.to_text() {
                    eprintln!(
                        "{} is stale: run `jope-dist lock` and review the diff",
                        lock_path.display()
                    );
                    return Ok(false);
                }
            } else {
                std::fs::write(&lock_path, lock.to_text())?;
            }
            for (name, p) in &lock.profiles {
                eprintln!(
                    "{name}: {} ({} asset bytes, {} methods)",
                    p.status,
                    p.total_bytes,
                    p.methods.len()
                );
                for blocker in &p.blockers {
                    eprintln!("  blocked: {blocker}");
                }
            }
            Ok(true)
        }
        Command::Fixture {
            models,
            write_manifests,
        } => {
            let manifests = repo.join("deploy/appliance/fixtures/bundles");
            if write_manifests {
                fixture::write_manifests(&manifests)?;
            } else {
                fixture::write(&models.expect("required by clap"), &manifests)?;
            }
            Ok(true)
        }
        Command::Notices {
            out,
            target,
            features,
        } => {
            let features: Vec<String> = features.into_iter().filter(|f| !f.is_empty()).collect();
            let inventory = notices::collect(&manifest, "jope-appliance", &target, &features)?;
            emit(&serde_json::to_value(&inventory)?, Some(&out))?;
            eprintln!(
                "{} crates, {} distinct texts",
                inventory.crates.len(),
                inventory.texts.len()
            );
            Ok(true)
        }
        Command::Audit { policy } => {
            let policy = audit::Policy::load(&repo.join(policy))?;
            let violations = audit::audit(&manifest, &policy)?;
            for v in &violations {
                eprintln!("{v}");
            }
            if violations.is_empty() {
                eprintln!(
                    "{}: dependency fence clean for {} feature set(s)",
                    policy.package,
                    policy.feature_sets.len()
                );
            }
            Ok(violations.is_empty())
        }
        Command::Pack {
            profile,
            code,
            models,
            notices,
            source_sha,
            target,
            release_key,
            out,
            stage,
            rehearsal,
            receipt,
        } => {
            let request = pack::PackRequest {
                profile: &profile,
                lock: &lock_path,
                repo: &repo,
                code: &code,
                models: &models,
                notices: &notices,
                source_sha: &source_sha,
                target: &target,
                release_key: release_key.as_deref(),
            };
            let output = match (&out, &stage, &rehearsal) {
                (Some(out), None, None) => pack::Output::Development { out },
                (None, Some(dir), None) => pack::Output::Stage { dir },
                (None, None, Some(dir)) => pack::Output::Rehearsal { dir },
                _ => anyhow::bail!("give exactly one of --out, --stage or --rehearsal"),
            };
            pack::pack(&request, output, receipt.as_deref())?;
            Ok(true)
        }
        Command::Seal {
            code,
            stage,
            signature,
            release_key,
            out,
            receipt,
        } => {
            pack::seal(
                &code,
                &stage,
                &signature,
                release_key.as_deref(),
                &out,
                receipt.as_deref(),
            )?;
            Ok(true)
        }
        Command::Verify(read) => {
            emit(&report::verify_file(&read.file, &read.keys()?)?, None)?;
            Ok(true)
        }
        Command::Inspect(read) => {
            emit(&report::inspect(&read.file, &read.keys()?)?, None)?;
            Ok(true)
        }
        Command::SizeReport { read, budget } => {
            let budget = budget.map(|b| repo.join(b));
            let (value, ok) = report::size_report(&read.file, &read.keys()?, budget.as_deref())?;
            emit(&value, None)?;
            Ok(ok)
        }
        Command::SbomModels { read, out } => {
            emit(&report::sbom_models(&read.file, &read.keys()?)?, Some(&out))?;
            Ok(true)
        }
        Command::Provenance {
            read,
            tag,
            builder_image,
            builder_id,
            cargo_profile,
            features,
            out,
        } => {
            let value = report::provenance(
                &read.file,
                &read.keys()?,
                &report::ProvenanceInputs {
                    tag: &tag,
                    builder_image: &builder_image,
                    builder_id: &builder_id,
                    cargo_profile: &cargo_profile,
                    features: &features,
                },
            )?;
            emit(&value, Some(&out))?;
            Ok(true)
        }
    }
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("jope-dist: {error:#}");
            ExitCode::from(1)
        }
    }
}
