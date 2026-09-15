//! Explicit maintenance entrypoint; never signs policy or releases an original held fit.
use anyhow::Result;
use clap::{Parser, Subcommand};
use jope_parallel::eval_registry::Authority;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    root: PathBuf,
    #[arg(long)]
    witness: PathBuf,
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Inspect,
    Prepare,
    Commit {
        #[arg(long)]
        policy: PathBuf,
    },
}
fn main() -> Result<()> {
    let args = Args::parse();
    let value = match args.command {
        Command::Inspect => Authority::native_recovery_candidate(&args.root, &args.witness)?,
        Command::Prepare => {
            Authority::recovery_reconcile(&args.root, &args.witness)?;
            Authority::native_recovery_candidate(&args.root, &args.witness)?
        }
        Command::Commit { policy } => {
            serde_json::json!({"schema":"jope.native_held_epoch_committed.v1","epoch":Authority::commit_native_held_epoch(&args.root,&args.witness,&policy)?,"original_identities_remain_held":true,"new_namespace_activated":false})
        }
    };
    println!("{}", serde_json::to_string(&value)?);
    Ok(())
}
