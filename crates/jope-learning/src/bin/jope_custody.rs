//! Local-only custody dry run and credential-free v2 archive readback.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_learning::{
    canon::{is_hex64, sha256_hex},
    custody::{self, CapturedMember, MAX_MANIFEST},
};
use serde::Deserialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    process::Command as Process,
    sync::Arc,
};

#[derive(Parser)]
#[command(about = "Pack and verify local nonqualifying custody evidence; no S3 adapter")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    DryRun {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        input_sha: String,
        #[arg(long)]
        output: PathBuf,
    },
    Readback {
        #[arg(long)]
        manifest: PathBuf,
    },
    VerifyComplete {
        #[arg(long)]
        directory: PathBuf,
        #[arg(long)]
        receipt_sha: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    rows: Vec<Value>,
    #[serde(default)]
    facts: Vec<Value>,
    budgets: Value,
    members: Vec<Member>,
    #[serde(default)]
    pricing: Option<PriceInput>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PriceInput {
    requests: Vec<Value>,
    rates: Value,
    as_of: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Member {
    row: Value,
    body_path: PathBuf,
    body_sha256: String,
}

fn outside_worktrees(output: &Path) -> Result<()> {
    // Resolve existing ancestors so a symlink cannot route new bulk into a checkout.
    let mut ancestor = output;
    let mut suffix = Vec::new();
    while !ancestor.exists() {
        suffix.push(ancestor.file_name().context("output filename")?.to_owned());
        ancestor = ancestor.parent().context("output ancestor")?;
    }
    let mut resolved = ancestor.canonicalize()?;
    for part in suffix.iter().rev() {
        ensure!(part != ".." && part != ".", "normalized output required");
        resolved.push(part);
    }
    let listing = Process::new("git")
        .args(["worktree", "list", "--porcelain", "-z"])
        .output()?;
    ensure!(
        listing.status.success(),
        "dry-run requires the claimed product checkout for output admission"
    );
    for entry in listing.stdout.split(|b| *b == 0) {
        if let Some(path) = entry.strip_prefix(b"worktree ") {
            let tree = PathBuf::from(std::str::from_utf8(path)?);
            let tree = tree.canonicalize().unwrap_or(tree);
            ensure!(
                !resolved.starts_with(tree),
                "bulk custody output must be outside registered checkouts"
            );
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let report = match Args::parse().command {
        Command::Readback { manifest } => custody::verify_saved(&manifest)?,
        Command::VerifyComplete {
            directory,
            receipt_sha,
        } => custody::verify_complete(&directory, &receipt_sha)?,
        Command::DryRun {
            input,
            input_sha,
            output,
        } => {
            ensure!(is_hex64(&input_sha), "pinned input SHA required");
            ensure!(
                output.is_absolute(),
                "absolute local custody output required"
            );
            outside_worktrees(&output)?;
            let raw = custody::bounded_read(&input, 32 * 1024 * 1024)?;
            ensure!(sha256_hex(&raw) == input_sha, "input SHA mismatch");
            let data: Input = serde_json::from_slice(&raw)?;
            ensure!(
                !data.members.is_empty() && data.members.len() <= 512,
                "bounded member roster"
            );
            let mut dedup = BTreeMap::<String, Arc<[u8]>>::new();
            let mut captured = Vec::new();
            let mut bytes = 0u64;
            for member in data.members {
                ensure!(is_hex64(&member.body_sha256), "pinned body SHA required");
                let path = if member.body_path.is_absolute() {
                    member.body_path
                } else {
                    input
                        .parent()
                        .context("input directory")?
                        .join(member.body_path)
                };
                let body = custody::bounded_read(&path, 64 * 1024 * 1024)?;
                ensure!(sha256_hex(&body) == member.body_sha256, "body SHA mismatch");
                let body = if let Some(existing) = dedup.get(&member.body_sha256) {
                    ensure!(
                        existing.as_ref() == body.as_slice(),
                        "content digest collision"
                    );
                    existing.clone()
                } else {
                    bytes = bytes
                        .checked_add(body.len() as u64)
                        .context("payload overflow")?;
                    ensure!(bytes <= 64 * 1024 * 1024, "bounded unique payload");
                    let body = Arc::<[u8]>::from(body);
                    dedup.insert(member.body_sha256, body.clone());
                    body
                };
                captured.push(CapturedMember {
                    row: member.row,
                    body,
                });
            }
            custody::dry_run(
                &output,
                &custody::DryRunRequest {
                    rows: &data.rows,
                    facts: &data.facts,
                    budgets: &data.budgets,
                    members: &captured,
                    input_sha: Some(&input_sha),
                    pricing: data.pricing.as_ref().map(|p| custody::Pricing {
                        requests: &p.requests,
                        rates: &p.rates,
                        as_of: &p.as_of,
                    }),
                },
                &mut |_, _| Ok(()),
            )?
        }
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
