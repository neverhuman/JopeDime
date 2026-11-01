//! Internal source-approved TRAIN recording command boundary.

use clap::{Args, Subcommand};
use std::path::PathBuf;

#[derive(Args)]
pub struct ResearchArgs {
    #[command(subcommand)]
    command: ResearchCommand,
}

#[derive(Subcommand)]
enum ResearchCommand {
    /// Serve the explicitly admitted frozen r782 Q1 evaluator; not study authority.
    ServeQ1 {
        /// Existing release-authenticated r782 extraction (all signatures rechecked).
        #[arg(long)]
        bundle_dir: PathBuf,
    },
    /// Record per-round training observations from the discovery producer.
    Record {
        /// Opaque, source-approved job ID (not a path or JSON plan).
        #[arg(long)]
        job: String,
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
        ResearchCommand::ServeQ1 { bundle_dir } => {
            let forwards = jope_service::engine::serve_q1_qualification(&bundle_dir)?;
            println!(
                "{}",
                serde_json::json!({
                    "schema": "jope.q1-evaluator-activity/v1",
                    "source_commit": crate::bootstrap::source_sha40(),
                    "learned_forward_count": forwards,
                    "qualification_granted": false,
                })
            );
            Ok(())
        }
        ResearchCommand::Record {
            job,
            out,
            evaluate_all_proposals,
        } => record(&job, out, evaluate_all_proposals),
    }
}

fn record(job: &str, out: PathBuf, evaluate_all: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        job == "qlt03-slump85-v1",
        "no source-approved research job matches this ID"
    );
    let root = jope_paths::jopedime_home().join("research/jobs").join(job);
    let receipt = jope_service::research::record_catalog_job(job, &root, &out, evaluate_all)?;
    println!("{}", serde_json::to_string(&receipt)?);
    Ok(())
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
            let mut argv = vec![
                "jopedime",
                "research",
                "record",
                "--job",
                "qlt03-slump85-v1",
                "--out",
                "records.jsonl",
            ];
            if all {
                argv.push("--evaluate-all-proposals");
            }
            let Command::Research(args) = Cli::try_parse_from(argv).unwrap().command else {
                panic!("research dispatch expected");
            };
            let ResearchCommand::Record {
                job,
                out,
                evaluate_all_proposals,
            } = args.command
            else {
                panic!("record dispatch expected");
            };
            assert_eq!(job, "qlt03-slump85-v1");
            assert_eq!(out, PathBuf::from("records.jsonl"));
            assert_eq!(evaluate_all_proposals, all);
        }
        assert!(Cli::try_parse_from(["jopedime", "research", "record"]).is_err());
    }

    #[test]
    fn q1_entry_requires_explicit_bundle_and_refuses_untrusted_assets() {
        assert!(Cli::try_parse_from(["jopedime", "research", "serve-q1"]).is_err());
        let directory = tempfile::tempdir().unwrap();
        let release = directory.path().join("missing-release");
        let Command::Research(args) = Cli::try_parse_from([
            "jopedime",
            "research",
            "serve-q1",
            "--bundle-dir",
            release.to_str().unwrap(),
        ])
        .unwrap()
        .command
        else {
            panic!("research dispatch expected");
        };
        assert!(
            matches!(&args.command, ResearchCommand::ServeQ1 { bundle_dir } if bundle_dir == &release)
        );
        assert!(run(args).is_err());
        assert!(!release.exists());
    }

    #[test]
    fn unknown_job_never_creates_or_overwrites_records() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        for all in [false, true] {
            for existing in [false, true] {
                if existing {
                    std::fs::write(&path, b"retained evidence\n").unwrap();
                }
                let error = record("../unknown", path.clone(), all).unwrap_err();
                assert!(error
                    .to_string()
                    .contains("no source-approved research job"));
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
    fn dispatch_authenticates_before_reaching_the_producer() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("records.jsonl");
        let args = ResearchArgs {
            command: ResearchCommand::Record {
                job: "qlt03-slump85-v1".into(),
                out: path.clone(),
                evaluate_all_proposals: true,
            },
        };
        let error = run(args).unwrap_err();
        if crate::bootstrap::EMBEDDED {
            assert_eq!(
                jope_runtime::error::kind_of(&error),
                jope_runtime::FailureKind::Integrity
            );
        } else {
            assert!(!error.to_string().is_empty());
        }
        assert!(!path.exists());
    }
}
