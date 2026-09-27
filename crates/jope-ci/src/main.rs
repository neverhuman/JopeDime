//! `jope-ci`: CI and host-maintenance tooling (see the library documentation).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use jope_ci::cargo_artifact::{self, Build, Failure};

#[derive(Parser)]
#[command(name = "jope-ci", version, about = "CI and host-maintenance tooling")]
struct Cli {
    #[command(subcommand)]
    command: Top,
}

#[derive(Subcommand)]
enum Top {
    /// Resolve Cargo executables/reports without assuming the target-directory layout.
    ///
    /// build writes diagnostics to stderr and exactly one executable path to stdout. select
    /// consumes successful container build messages, translating a mounted root.
    CargoArtifact {
        #[command(subcommand)]
        command: Artifact,
    },
}

#[derive(Args)]
#[group(required = true, multiple = false)]
struct Target {
    #[arg(long)]
    bin: Option<String>,
    #[arg(long)]
    example: Option<String>,
}

impl Target {
    fn kind_and_name(&self) -> (&'static str, &str) {
        match (&self.bin, &self.example) {
            (Some(bin), _) => ("bin", bin),
            (None, Some(example)) => ("example", example),
            (None, None) => unreachable!("clap requires --bin or --example"),
        }
    }
}

#[derive(Args)]
struct Manifest {
    #[arg(long, default_value = "Cargo.toml")]
    manifest_path: String,
    #[arg(long)]
    target_dir: Option<String>,
}

#[derive(Subcommand)]
enum Artifact {
    Build {
        #[command(flatten)]
        manifest: Manifest,
        #[arg(long)]
        package: String,
        #[command(flatten)]
        target: Target,
        #[arg(long, default_value = "dev")]
        profile: String,
        #[arg(long)]
        features: Vec<String>,
        #[arg(long)]
        no_default_features: bool,
    },
    /// read a successful container build
    Select {
        #[arg(long)]
        package: String,
        #[command(flatten)]
        target: Target,
        #[arg(long)]
        metadata: PathBuf,
        #[arg(long)]
        messages: PathBuf,
        #[arg(long, num_args = 2, value_names = ["CONTAINER_ROOT", "HOST_ROOT"])]
        path_map: Option<Vec<PathBuf>>,
    },
    TargetDir {
        #[command(flatten)]
        manifest: Manifest,
    },
    /// bind nextest's store to Cargo's effective target
    NextestConfig {
        #[arg(long, default_value = ".config/nextest.toml")]
        source: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[command(flatten)]
        manifest: Manifest,
    },
}

fn artifact(command: Artifact) -> Result<String, Failure> {
    match command {
        Artifact::Build {
            manifest,
            package,
            target,
            profile,
            features,
            no_default_features,
        } => {
            let (kind, name) = target.kind_and_name();
            let path = cargo_artifact::build(&Build {
                manifest_path: &manifest.manifest_path,
                target_dir: manifest.target_dir.as_deref(),
                package: &package,
                kind,
                name,
                profile: &profile,
                features: &features,
                no_default_features,
            })?;
            Ok(path.display().to_string())
        }
        Artifact::Select {
            package,
            target,
            metadata,
            messages,
            path_map,
        } => {
            let (kind, name) = target.kind_and_name();
            let map = path_map
                .as_deref()
                .map(|pair| (Path::new(&pair[0]), Path::new(&pair[1])));
            let path = cargo_artifact::select(&metadata, &messages, &package, kind, name, map)?;
            Ok(path.display().to_string())
        }
        Artifact::TargetDir { manifest } => cargo_artifact::target_directory(
            &manifest.manifest_path,
            manifest.target_dir.as_deref(),
        ),
        Artifact::NextestConfig {
            source,
            output,
            manifest,
        } => {
            let target = cargo_artifact::target_directory(
                &manifest.manifest_path,
                manifest.target_dir.as_deref(),
            )?;
            cargo_artifact::nextest_config(&source, &output, Path::new(&target))?;
            Ok(String::new())
        }
    }
}

fn main() -> ExitCode {
    match Cli::parse().command {
        Top::CargoArtifact { command } => match artifact(command) {
            Ok(line) if line.is_empty() => ExitCode::SUCCESS,
            Ok(line) => {
                println!("{line}");
                ExitCode::SUCCESS
            }
            Err(Failure::Exit(code)) => ExitCode::from(u8::try_from(code & 0xff).unwrap_or(1)),
            Err(Failure::Refused(message)) => {
                eprintln!("cargo-artifact: {message}");
                ExitCode::from(1)
            }
        },
    }
}
