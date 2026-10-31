//! Internal command boundary. Durable discovery capture remains QLT-03-F02.

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Args)]
pub struct ResearchArgs {
    #[command(subcommand)]
    command: ResearchCommand,
}

#[derive(Subcommand)]
enum ResearchCommand {
    /// Record per-round training observations from the discovery producer.
    Record {
        /// Destination for committed training records.
        #[arg(long)]
        out: PathBuf,
        /// Evaluate every eligible proposal while preserving policy admission.
        #[arg(long)]
        evaluate_all_proposals: bool,
    },
}

pub fn run(args: ResearchArgs) -> anyhow::Result<()> {
    // Authenticate before any producer input or output can be accessed.
    let _package = crate::bootstrap::open()?;
    match args.command {
        ResearchCommand::Record {
            out,
            evaluate_all_proposals,
        } => record(out, evaluate_all_proposals),
    }
}

fn record(_out: PathBuf, _evaluate_all_proposals: bool) -> anyhow::Result<()> {
    // F02 must supply the trusted corpus/evaluator and checkpoint commit hooks.
    // The in-memory RoundCapture controller is not that committed producer.
    Err(jope_runtime::Failure::backend(
        "research record is unavailable: the committed discovery producer is not integrated (QLT-03-F02)",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::{Cli, Command};
    use clap::{CommandFactory, Parser};

    #[test]
    fn internal_help_and_record_options_are_available() {
        assert!(Cli::command()
            .render_help()
            .to_string()
            .contains("research"));
        let help = Cli::try_parse_from(["jopedime", "research", "record", "--help"])
            .err()
            .unwrap();
        assert_eq!(help.kind(), clap::error::ErrorKind::DisplayHelp);
        assert!(help.to_string().contains("--evaluate-all-proposals"));
        for all in [false, true] {
            let mut argv = vec!["jopedime", "research", "record", "--out", "records.jsonl"];
            if all {
                argv.push("--evaluate-all-proposals");
            }
            let Command::Research(args) = Cli::try_parse_from(argv).unwrap().command else {
                panic!("research dispatch expected");
            };
            let ResearchCommand::Record {
                out,
                evaluate_all_proposals,
            } = args.command;
            assert_eq!(out, PathBuf::from("records.jsonl"));
            assert_eq!(evaluate_all_proposals, all);
        }
        assert!(Cli::try_parse_from(["jopedime", "research", "record"]).is_err());
    }

    #[test]
    fn unavailable_producer_never_creates_or_overwrites_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        for all in [false, true] {
            for existing in [false, true] {
                if existing {
                    std::fs::write(&path, b"retained evidence\n").unwrap();
                }
                let error = record(path.clone(), all).unwrap_err();
                assert_eq!(
                    jope_runtime::error::kind_of(&error),
                    jope_runtime::FailureKind::Backend
                );
                if existing {
                    assert_eq!(std::fs::read(&path).unwrap(), b"retained evidence\n");
                    std::fs::remove_file(&path).unwrap();
                } else {
                    assert!(!path.exists());
                }
            }
        }
    }

    #[test]
    fn dispatch_authenticates_before_reaching_the_unavailable_producer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let args = ResearchArgs {
            command: ResearchCommand::Record {
                out: path.clone(),
                evaluate_all_proposals: true,
            },
        };
        let error = run(args).unwrap_err();
        assert_eq!(
            jope_runtime::error::kind_of(&error),
            if crate::bootstrap::EMBEDDED {
                jope_runtime::FailureKind::Integrity
            } else {
                jope_runtime::FailureKind::Backend
            }
        );
        assert!(!path.exists());
    }
}
