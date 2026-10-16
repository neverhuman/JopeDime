//! `jope-records`: verify and render the committed core-IP remediation records.
//!
//! Every subcommand keeps the flags, defaults and printed JSON summary of the Python tool it
//! replaces. Default paths are resolved inside the repository given by `--repo` (default: the
//! git checkout containing the working directory).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};
use jope_records::pyjson::{self, Value};
use jope_records::{
    claim_inventory, claims, conformance, coverage, dispositions, elo_history, findings, inventory,
    layout, paper_trace, parity, progress_view, py, readme, repo_layout, scope_check, task_status,
    test_inventory,
};

#[derive(Parser)]
#[command(
    name = "jope-records",
    version,
    about = "Verify and render the committed core-IP remediation records"
)]
struct Cli {
    /// Repository root the default record paths are resolved in (default: the git toplevel of
    /// the working directory).
    #[arg(long, global = true, value_name = "DIR")]
    repo: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check the findings register's source custody and coverage (inventory integrity only).
    VerifyFindings {
        /// The findings directory (default: docs/internal/core-ip/findings).
        #[arg(long)]
        root: Option<PathBuf>,
    },
    /// Validate finding dispositions against hash-bound evidence (W12-T02).
    VerifyDispositions {
        /// The findings directory (default: docs/internal/core-ip/findings).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Also require every cited commit in the local repository.
        #[arg(long)]
        require_git: bool,
    },
    /// Validate scoped publication claims against the evidence dependency graph (W12-T01/T05).
    VerifyClaims {
        /// The claims record (default: docs/internal/core-ip/claims/claims.json).
        #[arg(long)]
        claims: Option<PathBuf>,
        /// Fail when custody evidence is not present on this host.
        #[arg(long)]
        require_custody: bool,
    },
    /// Generate the capability/KPI sentence inventory from the verified claims and check (or
    /// rewrite) claims/INVENTORY.md (W12-T01).
    ClaimInventory {
        /// The claims record (default: docs/internal/core-ip/claims/claims.json).
        #[arg(long)]
        claims: Option<PathBuf>,
        /// Rewrite INVENTORY.md.
        #[arg(long)]
        write: bool,
    },
    /// Check (or regenerate) the public synthetic conformance pack, scan it for private
    /// canaries and replay every fixture through its receiving validator (W12-T07).
    Conformance {
        /// Rewrite the generated pack files (never deletes a file).
        #[arg(long)]
        write: bool,
    },
    /// Check the maintained regression test inventory against the CI coverage map (W0).
    VerifyTestInventory {
        /// The inventory (default: docs/internal/core-ip/findings/test-inventory.json).
        #[arg(long)]
        inventory: Option<PathBuf>,
    },
    /// Validate the task-status record and check (or rewrite) STATUS.md (W12-T03).
    TaskStatus {
        /// The work-orders directory (default: docs/internal/core-ip/work-orders).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Rewrite STATUS.md.
        #[arg(long)]
        write: bool,
    },
    /// Validate the committed records and check (or rewrite) PROGRESS.md (W12-T03).
    ProgressView {
        /// The core-IP directory (default: docs/internal/core-ip).
        #[arg(long)]
        core: Option<PathBuf>,
        /// Rewrite work-orders/PROGRESS.md.
        #[arg(long)]
        write: bool,
    },
    /// Verify the accepted policy tensor inventory against its asset (W2-T06; host-only asset).
    VerifyInventory {
        /// The inventory record (default: docs/internal/core-ip/data/tensors.json).
        #[arg(long)]
        record: Option<PathBuf>,
        /// Optional torch optimizer archive of the same checkpoint.
        #[arg(long)]
        optimizer: Option<PathBuf>,
    },
    /// Replay the retained historical Elo ladders and keep unrecoverable history open (W12-T04).
    EloHistory {
        /// The histories record (default: docs/internal/core-ip/elo-history/histories.json).
        #[arg(long)]
        record: Option<PathBuf>,
        /// Fail when a reproducible history cannot be replayed on this host.
        #[arg(long)]
        require_all: bool,
        /// Only report the iteration-cap sensitivity of one reproducible history.
        #[arg(long, value_name = "HISTORY")]
        sensitivity: Option<String>,
    },
    /// Static CI execution map guard: deploy/ci/coverage.json against Cargo and git.
    /// Discovery is neither collection nor execution.
    Coverage {
        #[arg(value_enum)]
        command: CoverageCommand,
        /// The mapping (default: deploy/ci/coverage.json).
        #[arg(long)]
        mapping: Option<PathBuf>,
    },
    /// Assert deploy/ci/rust-test-scope.sh partitions the workspace exactly.
    RustScopeCheck,
    /// Capability-parity guard for docs/parity/capability-matrix.md.
    ParityCheck {
        /// The matrix (default: docs/parity/capability-matrix.md).
        #[arg(long)]
        matrix: Option<PathBuf>,
        /// The tree the rows are resolved against (default: the repository).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Parsed `jope-parallel --help` output to cross-check the clap subcommands against.
        #[arg(long)]
        help_text: Option<PathBuf>,
    },
    /// Repository layout guard (hygiene lane): root Markdown, tracked logs, member directory vs
    /// package name, duplicate package names, temp-dir use in test code, Cargo manifests not
    /// readable as TOML 1.0 and tracked paths that are not UTF-8, against the reviewed exceptions
    /// in deploy/ci/repo-layout-allow.txt.
    RepoLayout {
        /// The checkout to check (default: the repository).
        #[arg(long)]
        root: Option<PathBuf>,
        /// Print the current `temp-dir <path> <count>` lines of test code instead of checking
        /// (the generator of the allowlist's temp-dir entries).
        #[arg(long)]
        print_temp_dir: bool,
    },
    /// Check the white paper's printed numbers against its claim ledger and trace, and the
    /// README's version and "At a glance" counts against the files their footnotes name.
    PaperTrace {
        /// Also re-hash every claim source against its pin (host mode: the sources live on
        /// xbabe2, under the roots `docs/paper/claims.json` declares).
        #[arg(long)]
        sources: bool,
        /// Re-render Figure 1 (`docs/paper/figures/why-it-matters.svg`) from its data and the
        /// ledger before checking.
        #[arg(long)]
        write_figure: bool,
    },
    /// Render the README's measured-results blocks from committed evidence.
    ReadmeBenchmarks {
        /// Exit 1 if README.md is out of date (CI).
        #[arg(long, conflicts_with = "write", required_unless_present = "write")]
        check: bool,
        /// Rewrite the blocks in place.
        #[arg(long)]
        write: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum CoverageCommand {
    Discover,
    Validate,
}

/// What a command prints: the JSON summary of a record verifier, or the text report of a
/// hygiene guard with its own exit status.
enum Output {
    Summary(Value),
    Report { out: String, err: String, ok: bool },
}

fn report((out, ok): (String, bool)) -> Output {
    Output::Report {
        out,
        err: String::new(),
        ok,
    }
}

fn repository(cli_repo: Option<&Path>) -> jope_records::Result<PathBuf> {
    match cli_repo {
        Some(repo) => Ok(repo.to_path_buf()),
        None => {
            let cwd = std::env::current_dir()
                .map_err(|e| jope_records::Error::Shape(format!("current directory: {e}")))?;
            layout::git_toplevel(&cwd)
        }
    }
}

fn run(cli: Cli) -> jope_records::Result<Output> {
    let repo = || repository(cli.repo.as_deref());
    Ok(Output::Summary(match cli.command {
        Command::VerifyFindings { root } => {
            let root = match root {
                Some(root) => root,
                None => layout::findings(&repo()?),
            };
            findings::verify(&root)?.to_json()
        }
        Command::VerifyDispositions { root, require_git } => {
            let root = match root {
                Some(root) => root,
                None => layout::findings(&repo()?),
            };
            dispositions::verify(&root, require_git, None)?.to_json()
        }
        Command::VerifyClaims {
            claims: path,
            require_custody,
        } => {
            let path = match path {
                Some(path) => path,
                None => layout::claims(&repo()?),
            };
            claims::verify(&path, None, require_custody)?.to_json()
        }
        Command::ClaimInventory {
            claims: path,
            write,
        } => {
            let path = match path {
                Some(path) => path,
                None => layout::claims(&repo()?),
            };
            let (text, summary) = claim_inventory::render(&path, None)?;
            claim_inventory::check_or_write(&claim_inventory::target(&path), &text, write)?;
            summary
        }
        Command::Conformance { write } => {
            let repo = repo()?;
            conformance::check_or_write(&repo, write)?;
            let scanned = conformance::scan(&repo)?;
            let mut summary = conformance::replay(&repo)?;
            if let Some(dict) = summary.as_dict_mut() {
                dict.insert("files_scanned", scanned.into());
            }
            summary
        }
        Command::VerifyTestInventory { inventory } => {
            let repo = repo()?;
            let inventory = inventory.unwrap_or_else(|| layout::test_inventory(&repo));
            test_inventory::verify(&inventory, &repo)?.to_json()
        }
        Command::TaskStatus { root, write } => {
            let root = match root {
                Some(root) => root,
                None => layout::work_orders(&repo()?),
            };
            let record = task_status::verify(&root)?;
            let text = task_status::render(&record)?;
            task_status::check_or_write(&root, &text, write)?;
            task_status::summary(&record)
        }
        Command::ProgressView { core, write } => {
            let core = match core {
                Some(core) => core,
                None => layout::core_ip(&repo()?),
            };
            let data = progress_view::collect(&core)?;
            let text = progress_view::render(&data)?;
            progress_view::check_or_write(&core, &text, write)?;
            progress_view::summary(&data)
        }
        Command::VerifyInventory { record, optimizer } => {
            let repo = repo()?;
            let record_path = record.unwrap_or_else(|| repo.join(inventory::RECORD));
            let record = py::load_json(&record_path)?;
            let asset = record
                .get("path")
                .and_then(Value::as_str)
                .ok_or_else(|| jope_records::Error::Shape("record: path is not a string".into()))?;
            let raw = py::read_bytes(Path::new(asset))?;
            let source = py::read_text(&repo.join(inventory::MODEL_SOURCE))?;
            inventory::verify(
                &record,
                &raw,
                &source,
                optimizer.as_deref(),
                &inventory::Spec::accepted(),
            )?
        }
        Command::EloHistory {
            record,
            require_all,
            sensitivity,
        } => {
            let repo = repo()?;
            let record = record.unwrap_or_else(|| repo.join(elo_history::RECORD));
            let zstd = elo_history::Zstd::Builtin;
            if let Some(history) = sensitivity {
                return elo_history::sensitivity_of(&record, &repo, &history, zstd)
                    .map(Output::Summary);
            }
            let summary = elo_history::verify(&record, &repo, zstd)?;
            if require_all {
                summary.require_all()?;
            }
            summary.to_json()
        }
        Command::Coverage { command, mapping } => {
            let repo = repo()?;
            return Ok(match command {
                CoverageCommand::Discover => report((coverage::discover_report(&repo)?, true)),
                CoverageCommand::Validate => {
                    let mapping = mapping.unwrap_or_else(|| repo.join(coverage::MAPPING));
                    report(coverage::validate_report(&repo, &mapping)?)
                }
            });
        }
        Command::RustScopeCheck => return Ok(report(scope_check::check(&repo()?)?)),
        Command::ParityCheck {
            matrix,
            root,
            help_text,
        } => {
            let root = match root {
                Some(root) => root,
                None => repo()?,
            };
            let matrix = matrix.unwrap_or_else(|| root.join(parity::MATRIX));
            return Ok(report(parity::check(&matrix, &root, help_text.as_deref())?));
        }
        Command::RepoLayout {
            root,
            print_temp_dir,
        } => {
            let root = match root {
                Some(root) => root,
                None => repo()?,
            };
            return Ok(report(repo_layout::run(&root, print_temp_dir)?));
        }
        Command::PaperTrace {
            sources,
            write_figure,
        } => {
            return Ok(report(paper_trace::check(&repo()?, sources, write_figure)?));
        }
        Command::ReadmeBenchmarks { write, .. } => {
            let (message, ok) = readme::run(&repo()?, write)?;
            // The out-of-date notice goes to stderr, like the Python renderer's.
            return Ok(if ok {
                report((message, true))
            } else {
                Output::Report {
                    out: String::new(),
                    err: message,
                    ok: false,
                }
            });
        }
    }))
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(Output::Summary(summary)) => {
            println!("{}", pyjson::dumps(&summary, true, None));
            ExitCode::SUCCESS
        }
        Ok(Output::Report { out, err, ok }) => {
            print!("{out}");
            eprint!("{err}");
            if ok {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Err(error) => {
            eprintln!("jope-records: {error}");
            ExitCode::FAILURE
        }
    }
}
