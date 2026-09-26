//! `jope-records`: verify and render the committed core-IP remediation records.
//!
//! Every subcommand keeps the flags, defaults and printed JSON summary of the Python tool it
//! replaces. Default paths are resolved inside the repository given by `--repo` (default: the
//! git checkout containing the working directory).

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use jope_records::pyjson::{self, Value};
use jope_records::{
    claims, dispositions, findings, layout, progress_view, task_status, test_inventory,
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

fn run(cli: Cli) -> jope_records::Result<Value> {
    let repo = || repository(cli.repo.as_deref());
    Ok(match cli.command {
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
    })
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(summary) => {
            println!("{}", pyjson::dumps(&summary, true, None));
            ExitCode::SUCCESS
        }
        Err(error) => {
            eprintln!("jope-records: {error}");
            ExitCode::FAILURE
        }
    }
}
