//! `jope`: the umbrella command line.
//!
//! Three verbs, each delegating at the library level (no child process) to the
//! crate that owns the command surface, so nothing is defined twice:
//!
//! * `jope engine …`   — `jope_parallel::cli`: every `jope-parallel` subcommand
//!   unchanged, including the engine's global `--root`;
//! * `jope selfplay …` — `jope_selfplay::TABLE`: the flywheel control-plane programs;
//! * `jope campaign …` — `jope_campaigns::TABLE`: the benchmark-campaign programs.

use clap::{Parser, Subcommand};
use std::ffi::OsString;

#[derive(Parser)]
#[command(
    name = "jope",
    version,
    about = "JopeDime: the search/eval engine, the flywheel control plane and the benchmark campaigns under one command"
)]
struct Jope {
    #[command(subcommand)]
    verb: Verb,
}

#[derive(Subcommand)]
enum Verb {
    /// The search/eval engine: every jope-parallel subcommand, unchanged (`jope engine --root <dir> search …`; `jope engine --help` lists them)
    #[command(disable_help_flag = true, disable_version_flag = true)]
    Engine {
        /// Passed to the engine as-is
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
        args: Vec<OsString>,
    },
    /// The flywheel control plane: `jope selfplay <program> …` (`jope selfplay --help` lists the programs)
    #[command(disable_help_flag = true, disable_version_flag = true)]
    Selfplay {
        /// The program and its arguments
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
        args: Vec<String>,
    },
    /// The benchmark campaigns: `jope campaign <program> …` (`jope campaign --help` lists the programs)
    #[command(disable_help_flag = true, disable_version_flag = true)]
    Campaign {
        /// The program and its arguments
        #[arg(trailing_var_arg = true, allow_hyphen_values = true, num_args = 0..)]
        args: Vec<String>,
    },
}

fn with_argv0(argv0: &str, args: Vec<String>) -> Vec<String> {
    std::iter::once(argv0.to_owned()).chain(args).collect()
}

fn main() -> anyhow::Result<()> {
    match Jope::parse().verb {
        Verb::Engine { args } => jope_parallel::cli::run_from(
            std::iter::once(OsString::from("jope engine")).chain(args),
        ),
        Verb::Selfplay { args } => jope_selfplay::TABLE.run(with_argv0("jope selfplay", args)),
        Verb::Campaign { args } => jope_campaigns::TABLE.run(with_argv0("jope campaign", args)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_semver() {
        let version = env!("CARGO_PKG_VERSION");
        let parts: Vec<_> = version.split('.').collect();
        assert_eq!(parts.len(), 3, "expected major.minor.patch");
        assert!(parts
            .iter()
            .all(|part| part.chars().all(|c| c.is_ascii_digit())));
    }

    #[test]
    fn engine_arguments_pass_through_untouched() {
        let jope = Jope::try_parse_from([
            "jope", "engine", "--root", "/work", "search", "--prepared", "/v", "--help",
        ])
        .unwrap();
        match jope.verb {
            Verb::Engine { args } => assert_eq!(
                args,
                ["--root", "/work", "search", "--prepared", "/v", "--help"]
                    .map(OsString::from)
                    .to_vec()
            ),
            _ => panic!("expected the engine verb"),
        }
    }

    #[test]
    fn program_verbs_keep_their_own_help_flags() {
        let jope = Jope::try_parse_from(["jope", "selfplay", "--help"]).unwrap();
        assert!(matches!(jope.verb, Verb::Selfplay { args } if args == ["--help"]));
        let jope = Jope::try_parse_from(["jope", "campaign", "corpus_eval", "--threads", "1"]).unwrap();
        assert!(matches!(jope.verb, Verb::Campaign { args } if args == ["corpus_eval", "--threads", "1"]));
        assert_eq!(with_argv0("jope campaign", vec!["x".into()]), ["jope campaign", "x"]);
    }
}
