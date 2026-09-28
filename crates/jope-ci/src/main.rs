//! `jope-ci`: CI and host-maintenance tooling (see the library documentation).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use jope_ci::build_report;
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
    /// Report Cargo candidates and allocated disk usage without changing scanned files.
    ///
    /// Report only: `--delete` is refused before anything is read. Cargo markers never establish
    /// that contents are disposable.
    BuildReport {
        #[command(subcommand)]
        command: Report,
    },
}

#[derive(Subcommand)]
enum Report {
    /// Print ROOT's outermost Cargo candidates as JSON.
    Scan {
        root: PathBuf,
        /// Only candidates whose newest content is at least this many whole hours old.
        #[arg(long, value_parser = build_report::whole_hours)]
        stale: Option<u64>,
    },
    /// Scan the two approved roots, measure the configured roots and keep a receipt under
    /// CUSTODY (outside /tmp and Git working trees).
    Report {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        custody: PathBuf,
        /// Where the wrapper scripts live, for the receipt's checksums (default: this
        /// executable's directory; the wrappers pass their own).
        #[arg(long)]
        tools_dir: Option<PathBuf>,
        /// One of the two approved Cargo roots (given twice; host layout, so the installed
        /// wrapper `jope-build-report.sh` passes them). The configuration must name exactly these.
        #[arg(long = "approved-root", value_name = "DIR", required = true)]
        approved_roots: Vec<String>,
    },
}

fn build_report_command(command: Report) -> ExitCode {
    let failed = |error: build_report::Failure| {
        eprintln!("report failed: {error}");
        ExitCode::from(1)
    };
    match command {
        Report::Scan { root, stale } => {
            let measure = |path: &Path, privileged: bool| {
                build_report::allocated(path, privileged, &build_report::never_interrupted)
            };
            match build_report::scan(&root, stale, &build_report::never_interrupted, &measure) {
                Ok(document) => {
                    println!("{}", jope_records::pyjson::dumps(&document, false, Some(2)));
                    ExitCode::SUCCESS
                }
                Err(error) => failed(error),
            }
        }
        Report::Report {
            config,
            custody,
            tools_dir,
            approved_roots,
        } => {
            let interrupt = match build_report::signal_interrupt() {
                Ok(interrupt) => interrupt,
                Err(error) => return failed(build_report::Failure::new(error.to_string())),
            };
            // The wrappers' directory (default: this executable's). An executable that cannot be
            // located refuses; it is never taken to be the working directory.
            let tools = match tools_dir {
                Some(tools) => tools,
                None => match std::env::current_exe().and_then(|p| p.canonicalize()) {
                    Ok(exe) => exe.parent().map(Path::to_path_buf).unwrap_or(exe),
                    Err(error) => {
                        return failed(build_report::Failure::new(format!(
                            "cannot locate this executable for the tools directory: {error}"
                        )))
                    }
                },
            };
            let scan = |root: &str, interrupt: build_report::Interrupt<'_>| {
                let measure = |path: &Path, privileged: bool| {
                    build_report::allocated(path, privileged, interrupt)
                };
                build_report::scan(Path::new(root), Some(48), interrupt, &measure)
            };
            let measure =
                |path: &Path, privileged: bool, interrupt: build_report::Interrupt<'_>| {
                    build_report::allocated(path, privileged, interrupt)
                };
            let hooks = build_report::Hooks {
                scan: &scan,
                measure: &measure,
            };
            match build_report::report(
                &config,
                &custody,
                &tools,
                &approved_roots,
                &hooks,
                &interrupt,
            ) {
                Ok(0) => ExitCode::SUCCESS,
                Ok(_) => ExitCode::from(1),
                Err(error) => failed(error),
            }
        }
    }
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

/// The line a subcommand prints on success (`None`: it prints nothing).
fn artifact(command: Artifact) -> Result<Option<String>, Failure> {
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
            Ok(Some(path.display().to_string()))
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
            Ok(Some(path.display().to_string()))
        }
        // `print(meta["target_directory"])`: an empty directory still prints its newline.
        Artifact::TargetDir { manifest } => cargo_artifact::target_directory(
            &manifest.manifest_path,
            manifest.target_dir.as_deref(),
        )
        .map(Some),
        Artifact::NextestConfig {
            source,
            output,
            manifest,
        } => {
            // `nextest_config(args.source, args.output, meta["target_directory"])`: the key is
            // looked up first; its type matters only once the output is created.
            let target = cargo_artifact::target_value(
                &manifest.manifest_path,
                manifest.target_dir.as_deref(),
            )?;
            cargo_artifact::nextest_config(&source, &output, &target)?;
            Ok(None)
        }
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|a| a == "build-report") && args.iter().any(|a| a == "--delete") {
        eprintln!("{}", build_report::DELETE_REFUSAL);
        return ExitCode::from(2);
    }
    match Cli::parse().command {
        Top::BuildReport { command } => build_report_command(command),
        Top::CargoArtifact { command } => match artifact(command) {
            Ok(None) => ExitCode::SUCCESS,
            Ok(Some(line)) => {
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
