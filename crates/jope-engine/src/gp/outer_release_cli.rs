//! `jope-parallel outer-release …`: the command-line verbs of the outer score
//! release authority ([`super::outer_release`]). Every verb is a thin parse of
//! its arguments followed by one library call; the authority's rules (grant
//! only for an exported sealed selection, one recorded attempt per grant,
//! withheld result, single release, hash-chained ledger) are unchanged.
use super::{handoff, outer_release::Authority};
use anyhow::Result;
use serde_json::{json, Value};
use std::path::PathBuf;

#[derive(clap::Subcommand, Debug)]
pub enum OuterReleaseCommand {
    /// Create an empty release authority with its own instance identity.
    Init {
        #[arg(long)]
        authority: PathBuf,
    },
    /// Issue the one scoring grant for an exported, sealed and independently
    /// replayed selection. The sealed label file is named, never opened.
    Grant {
        #[arg(long)]
        authority: PathBuf,
        /// The declared artifact root of the completed search.
        #[arg(long)]
        root: PathBuf,
        /// The completed search output directory.
        #[arg(long)]
        search_output: PathBuf,
        #[arg(long)]
        seeds: Option<PathBuf>,
        #[arg(long)]
        vocabulary: Option<PathBuf>,
        /// The exported predictor (its `*.handoff.json` sits beside it).
        #[arg(long)]
        predictor: PathBuf,
        /// The sealed final-test file, under `--root`.
        #[arg(long)]
        test: PathBuf,
    },
    /// Score a granted subject on exactly the granted labels. The score is
    /// withheld; `--receipt` receives the outcome without metrics.
    Score {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        grant: PathBuf,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        test: PathBuf,
        #[arg(long)]
        receipt: PathBuf,
    },
    /// Make one withheld score visible at `--output`, exactly once.
    Release {
        #[arg(long)]
        authority: PathBuf,
        #[arg(long)]
        grant_id: String,
        #[arg(long)]
        output: PathBuf,
    },
    /// Print the authority's verified event ledger.
    Events {
        #[arg(long)]
        authority: PathBuf,
    },
}

/// Run one verb; returns the JSON document the CLI prints.
pub fn run(command: OuterReleaseCommand) -> Result<Value> {
    Ok(match command {
        OuterReleaseCommand::Init { authority } => {
            let created = Authority::initialize(&authority)?;
            json!({"authority":authority,"instance":created.instance()})
        }
        OuterReleaseCommand::Grant {
            authority,
            root,
            search_output,
            seeds,
            vocabulary,
            predictor,
            test,
        } => {
            let grant =
                Authority::open(&authority)?.grant(&super::outer_release::GrantRequest {
                    search: handoff::Inputs {
                        root: &root,
                        output: &search_output,
                        seeds: seeds.as_deref(),
                        vocabulary: vocabulary.as_deref(),
                    },
                    predictor: &predictor,
                    test: &test,
                })?;
            let body: Value = crate::manifest::read_json(&grant)?;
            json!({"grant":grant,"grant_id":body["grant_id"]})
        }
        OuterReleaseCommand::Score {
            authority,
            grant,
            model,
            test,
            receipt,
        } => Authority::open(&authority)?.score(&grant, &model, &test, &receipt)?,
        OuterReleaseCommand::Release {
            authority,
            grant_id,
            output,
        } => Authority::open(&authority)?.release(&grant_id, &output)?,
        OuterReleaseCommand::Events { authority } => {
            json!({"events":Authority::open(&authority)?.events()?})
        }
    })
}
