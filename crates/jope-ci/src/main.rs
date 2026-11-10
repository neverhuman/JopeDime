//! `jope-ci`: CI and host-maintenance tooling (see the library documentation).

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Args, Parser, Subcommand};
use jope_ci::build_report;
use jope_ci::cargo_artifact::{self, Build, Failure};
use jope_ci::test_partition;

// The argument grammar is argparse's where the two agree and stricter where they do not (review
// round seven of PR #117): a repeated option other than `--features` (argparse kept the last value)
// and a prefix of an option (argparse expanded it) are usage errors, and a subcommand named `help`
// is refused as argparse refused it (`invalid choice`), at every level. No option collects repeated
// values the Python stored once. (Plain comments: a doc comment here would become help text.)
#[derive(Parser)]
#[command(
    name = "jope-ci",
    version,
    about = "CI and host-maintenance tooling",
    disable_help_subcommand = true
)]
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
    /// Prove that split CI jobs run exactly the tests of the single job each one replaced.
    ///
    /// Prints a `jope.test-partition/v1` report; the exit status is 0 only when it passed.
    TestPartition {
        #[command(subcommand)]
        command: Partition,
    },
}

#[derive(Subcommand)]
enum Partition {
    /// Workflow, group and scope declarations (hygiene lane).
    Static {
        /// Also require that nothing declared at this revision is missing (empty: no base; the
        /// all-zero new-push SHA compares with origin/main).
        #[arg(long)]
        base: Option<String>,
        /// The checkout to check (default: the one this binary was built from).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Write the report here too (exclusive create).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Downloaded rust-test shard evidence (ci.yml aggregate job).
    #[command(group = clap::ArgGroup::new("evidence").required(true).multiple(false))]
    RustShards {
        /// The downloaded CI artifacts of this run and attempt.
        #[arg(long, group = "evidence")]
        receipts: Option<PathBuf>,
        /// Each shard's evidence directory, in slice order (aggregate.py's verified receipts).
        #[arg(long = "shard-dir", value_name = "DIR", group = "evidence")]
        shard_dirs: Vec<PathBuf>,
        /// Write the report here too (exclusive create).
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Any listings, e.g. the base command against the shards.
    Collections {
        /// An unsharded listing (repeatable; all must be identical).
        #[arg(long, required = true)]
        unsharded: Vec<PathBuf>,
        /// A partitioned listing (repeatable).
        #[arg(long, required = true)]
        shard: Vec<PathBuf>,
        /// Write the report here too (exclusive create).
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

fn test_partition_command(command: Partition) -> ExitCode {
    let (kind, report, output) = match command {
        Partition::Static { base, root, output } => {
            let root = root.unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."));
            (
                "static",
                test_partition::static_check(&root, base.as_deref()),
                output,
            )
        }
        Partition::RustShards {
            receipts,
            shard_dirs,
            output,
        } => {
            let dirs = match receipts {
                Some(receipts) => test_partition::rust_shard_dirs(&receipts),
                None => test_partition::named_shard_dirs(&shard_dirs),
            };
            (
                "rust-shards",
                dirs.and_then(|dirs| test_partition::verify_shards(&dirs)),
                output,
            )
        }
        Partition::Collections {
            unsharded,
            shard,
            output,
        } => {
            let shards: Vec<test_partition::Shard> = shard
                .into_iter()
                .map(|path| test_partition::Shard {
                    name: path.display().to_string(),
                    collection: path,
                    junit: None,
                })
                .collect();
            (
                "collections",
                test_partition::compare(&unsharded, &shards),
                output,
            )
        }
    };
    let report = report.unwrap_or_else(|error| test_partition::failure(kind, error));
    match test_partition::emit(&report, output.as_deref()) {
        Ok(code) => ExitCode::from(code),
        Err(error) => {
            eprintln!("test-partition: {error}");
            ExitCode::from(1)
        }
    }
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

// `--bin` or `--example`. An empty `--bin` is refused (a usage error): the Python chose the kind by
// truth (`"bin" if args.bin else "example"`, name `args.bin or args.example`), so `--bin ""` asked
// for an example named `None`, which `select` matched only in a message without a target name and
// `build` crashed on (`TypeError`); the port never selects or builds a bin named "" in its place.
// An empty `--example` is kept, as the Python kept it.
#[derive(Args)]
#[group(required = true, multiple = false)]
struct Target {
    #[arg(long, value_parser = clap::builder::NonEmptyStringValueParser::new())]
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
        // One pair, given once: a second `--path-map` is a usage error (argparse's `nargs=2` store
        // kept the last pair; clap's default for a list appends it, and the first pair was used).
        #[arg(
            long,
            num_args = 2,
            value_names = ["CONTAINER_ROOT", "HOST_ROOT"],
            action = clap::ArgAction::Set
        )]
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
            // The executable as text: one whose resolved name is not UTF-8 has refused.
            cargo_artifact::build(&Build {
                manifest_path: &manifest.manifest_path,
                target_dir: manifest.target_dir.as_deref(),
                package: &package,
                kind,
                name,
                profile: &profile,
                features: &features,
                no_default_features,
            })
            .map(Some)
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
            cargo_artifact::select(&metadata, &messages, &package, kind, name, map).map(Some)
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

/// An argument that is not UTF-8, refused before anything is read (exit status 2, a usage error).
/// The Python tools took one surrogate-escaped and used it as a path or a name, printing and
/// recording it as a `str`; the ports read their arguments as text, so this is stricter. The
/// program's own path (`argv[0]`) is not read.
fn non_utf8_argument(args: &[OsString]) -> Option<String> {
    let (n, arg) = args
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, a)| a.to_str().is_none())?;
    Some(format!(
        "jope-ci: argument {n} {} is not UTF-8; the ports read their arguments as text",
        jope_records::py::os_repr(arg)
    ))
}

fn main() -> ExitCode {
    // `std::env::args()` panics on an argument that is not UTF-8; every argument is read as bytes
    // first. `"--delete" in sys.argv` is refused before anything else, whatever the other
    // arguments hold, as the Python refused it.
    let args: Vec<OsString> = std::env::args_os().collect();
    if args.get(1).is_some_and(|a| a == "build-report") && args.iter().any(|a| a == "--delete") {
        eprintln!("{}", build_report::DELETE_REFUSAL);
        return ExitCode::from(2);
    }
    if let Some(refusal) = non_utf8_argument(&args) {
        eprintln!("{refusal}");
        return ExitCode::from(2);
    }
    match Cli::parse_from(args).command {
        Top::BuildReport { command } => build_report_command(command),
        Top::TestPartition { command } => test_partition_command(command),
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
