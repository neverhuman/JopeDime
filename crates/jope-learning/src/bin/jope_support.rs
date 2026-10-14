//! Local-only frozen support tool. No program execution, learner or remote-store interface.
use anyhow::{ensure, Context, Result};
use clap::{Parser, Subcommand};
use jope_learning::{
    canon::canonical_json,
    support::{
        Candidate, FreezeRequest, FrozenSupport, Likelihood, LogitsRow, MismatchReport,
        MismatchRow, Trie,
    },
};
use serde::{de::DeserializeOwned, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[derive(Parser)]
#[command(about = "Freeze and inspect nonqualifying successor action support")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    Freeze {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Extend {
        #[arg(long)]
        support: PathBuf,
        #[arg(long)]
        sha: String,
        #[arg(long)]
        candidates: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    Verify {
        #[arg(long)]
        support: PathBuf,
        #[arg(long)]
        sha: String,
    },
    Children {
        #[arg(long)]
        support: PathBuf,
        #[arg(long)]
        sha: String,
        #[arg(long, value_delimiter = ',')]
        prefix: Vec<u16>,
    },
    Index {
        #[arg(long)]
        support: PathBuf,
        #[arg(long)]
        sha: String,
    },
    Score {
        #[arg(long)]
        support: PathBuf,
        #[arg(long)]
        sha: String,
        #[arg(long, value_delimiter = ',')]
        tokens: Vec<u16>,
        #[arg(long)]
        logits: PathBuf,
        #[arg(long, default_value_t = 1.)]
        temperature: f64,
    },
    Mismatch {
        #[arg(long)]
        rows: PathBuf,
    },
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let mut raw = Vec::new();
    fs::File::open(path)?
        .take(1024 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    ensure!(raw.len() <= 1024 * 1024 * 1024, "bounded input required");
    serde_json::from_slice(&raw).context("parse pinned input")
}

fn support(path: &Path, sha: &str) -> Result<FrozenSupport> {
    let value: FrozenSupport = read(path)?;
    ensure!(
        value.support_sha256 == sha,
        "caller-pinned support identity required"
    );
    value.verify()?;
    Ok(value)
}

fn publish(path: &Path, value: &FrozenSupport) -> Result<()> {
    value.verify()?;
    // Create-only single-file artifact: complete bytes precede any usable identity.
    // A torn write cannot satisfy the caller's pinned digest and semantic verifier.
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(
        !path.is_symlink() && !parent.is_symlink(),
        "output symlink refused"
    );
    let mut bytes = canonical_json(value)?;
    bytes.push(b'\n');
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(&bytes)?;
    file.sync_all()?;
    fs::File::open(parent)?.sync_all()?;
    Ok(())
}

fn print<T: Serialize>(value: &T) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

fn main() -> Result<()> {
    match Args::parse().command {
        Command::Freeze { input, output } => {
            let frozen = FrozenSupport::freeze(read::<FreezeRequest>(&input)?)?;
            publish(&output, &frozen)?;
            print(
                &serde_json::json!({"support_sha256":frozen.support_sha256,"programs":frozen.programs.len(),"qualified":false}),
            )
        }
        Command::Extend {
            support: path,
            sha,
            candidates,
            output,
        } => {
            let frozen = support(&path, &sha)?.extend(read::<Vec<Candidate>>(&candidates)?)?;
            publish(&output, &frozen)?;
            print(
                &serde_json::json!({"support_sha256":frozen.support_sha256,"programs":frozen.programs.len(),"qualified":false}),
            )
        }
        Command::Verify { support: path, sha } => {
            let frozen = support(&path, &sha)?;
            print(
                &serde_json::json!({"passed":true,"support_sha256":frozen.support_sha256,"qualified":false}),
            )
        }
        Command::Children {
            support: path,
            sha,
            prefix,
        } => {
            let frozen = support(&path, &sha)?;
            print(&Trie::new(&frozen)?.children(&prefix)?)
        }
        Command::Index { support: path, sha } => {
            let frozen = support(&path, &sha)?;
            let trie = Trie::new(&frozen)?;
            let mut seen = BTreeSet::new();
            let mut tables = BTreeMap::new();
            for program in &frozen.programs {
                for pos in 1..program.tokens.len() {
                    let prefix = program.tokens[..pos].to_vec();
                    if seen.insert(prefix.clone()) {
                        ensure!(
                            seen.len() <= 2_000_000,
                            "bounded exported prefix table required; use children for larger supports"
                        );
                        tables.insert(
                            prefix
                                .iter()
                                .map(u16::to_string)
                                .collect::<Vec<_>>()
                                .join(","),
                            trie.children(&prefix)?,
                        );
                    }
                }
            }
            print(
                &serde_json::json!({"support_sha256":frozen.support_sha256,"vocabulary_sha256":frozen.vocabulary.sha256,"max_seq":frozen.vocabulary.max_seq,"programs":frozen.programs,"children":tables,"qualified":false}),
            )
        }
        Command::Score {
            support: path,
            sha,
            tokens,
            logits,
            temperature,
        } => {
            let frozen = support(&path, &sha)?;
            print(&Likelihood::teacher_forced(
                &Trie::new(&frozen)?,
                &tokens,
                &read::<Vec<LogitsRow>>(&logits)?,
                temperature,
            )?)
        }
        Command::Mismatch { rows } => print(&MismatchReport::from_rows(
            &read::<Vec<MismatchRow>>(&rows)?,
        )?),
    }
}
