//! User-facing commands and flags (SPECIFICATION.md §3.3).
use clap::ValueEnum;
use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

mod arguments;
mod command;

pub use arguments::*;
pub use command::{ApplianceCommand, Cli};

#[cfg(all(test, feature = "engine"))]
#[path = "serve.rs"]
mod serve_lifecycle_tests;

#[cfg(test)]
mod serve_tests {
    use super::{ApplianceCommand, Cli};
    use clap::Parser;

    #[test]
    fn serve_accepts_the_specified_lifecycle_flags() {
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve"]).unwrap().command,
            ApplianceCommand::Serve {
                status: false,
                stop: false,
                idle_exit: None
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve", "--status"])
                .unwrap()
                .command,
            ApplianceCommand::Serve {
                status: true,
                stop: false,
                idle_exit: None
            }
        ));
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "serve", "--stop"])
                .unwrap()
                .command,
            ApplianceCommand::Serve {
                status: false,
                stop: true,
                idle_exit: None
            }
        ));
        assert!(
            matches!(Cli::try_parse_from(["jopedime", "serve", "--idle-exit", "2s"]).unwrap().command, ApplianceCommand::Serve { status: false, stop: false, idle_exit: Some(value) } if value == "2s")
        );
        assert!(Cli::try_parse_from(["jopedime", "serve", "--status", "--stop"]).is_err());
        assert!(Cli::try_parse_from(["jopedime", "serve", "--stop", "--idle-exit", "2s"]).is_err());
    }
}

#[cfg(all(test, feature = "mcp"))]
mod batch_cli_tests {
    use super::{ApplianceCommand, Cli, DatasetCommand, ObjectiveCommand};
    use clap::Parser;

    #[test]
    fn parses_bat04_command_shapes() {
        assert!(
            matches!(Cli::try_parse_from(["jopedime", "predict", "--model", "run_01ARZ3NDEKTSV4RRFFQ69G5FAV", "--input", "in.csv", "--output", "-"]).unwrap().command, ApplianceCommand::Predict { output, .. } if output.as_os_str() == std::ffi::OsStr::new("-"))
        );
        assert!(
            matches!(Cli::try_parse_from(["jopedime", "dataset", "add", "train.csv", "--name", "train"]).unwrap().command, ApplianceCommand::Dataset { command: DatasetCommand::Add { name: Some(name), .. } } if name == "train")
        );
        assert!(
            matches!(Cli::try_parse_from(["jopedime", "objective", "validate", "mae"]).unwrap().command, ApplianceCommand::Objective { command: ObjectiveCommand::Validate { spec, .. } } if spec == "mae")
        );
        assert!(matches!(
            Cli::try_parse_from(["jopedime", "gc", "--dry-run"])
                .unwrap()
                .command,
            ApplianceCommand::Gc { dry_run: true, .. }
        ));
        assert!(matches!(
            Cli::try_parse_from([
                "jopedime",
                "export",
                "run_01ARZ3NDEKTSV4RRFFQ69G5FAV",
                "--kind",
                "bundle",
                "--lang",
                "rust",
                "--out",
                "bundle"
            ])
            .unwrap()
            .command,
            ApplianceCommand::Export { .. }
        ));
    }
}

#[cfg(all(test, feature = "client"))]
mod batch_run_tests {
    use super::{ApplianceCommand, Cli};
    use clap::Parser;

    #[test]
    fn bat03_commands_parse_all_card_subcommands() {
        let run = "run_01ARZ3NDEKTSV4RRFFQ69G5FAV";
        for args in [
            vec!["jopedime", "control", run, "pause"],
            vec!["jopedime", "control", run, "continue"],
            vec!["jopedime", "control", run, "cancel"],
            vec!["jopedime", "control", run, "effort", "high"],
            vec!["jopedime", "restart", run, "--new-seed", "--warm"],
            vec!["jopedime", "runs", "list"],
            vec!["jopedime", "runs", "delete", run],
            vec![
                "jopedime",
                "runs",
                "export-feedback",
                run,
                "--out",
                "feedback.jsonl",
            ],
            vec![
                "jopedime", "code", run, "--alg", "alg_0001", "--lang", "python",
            ],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert!(matches!(
                cli.command,
                ApplianceCommand::Control { .. }
                    | ApplianceCommand::Restart { .. }
                    | ApplianceCommand::Runs { .. }
                    | ApplianceCommand::Code { .. }
            ));
        }
    }
}
