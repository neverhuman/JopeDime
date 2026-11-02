use crate::cli::{default_workers, Cli, Command, DEFAULT_MAX_WORKERS};
use clap::Parser;

#[test]
fn original_automl_cli_preserves_explicit_selection_and_seed() {
    let cli = Cli::try_parse_from([
        "jope-bench",
        "run",
        "--datasets",
        "data",
        "--out",
        "reports",
        "--automl",
        "gpt6-astra-max",
        "--seed",
        "37",
    ])
    .unwrap();
    match cli.command {
        Command::Run { automl, seed, .. } => {
            assert_eq!(automl, ["gpt6-astra-max"]);
            assert_eq!(seed, 37);
        }
        _ => panic!("expected run command"),
    }
    assert!(Cli::try_parse_from([
        "jope-bench",
        "run",
        "--datasets",
        "data",
        "--out",
        "reports",
        "--automl",
        "unknown",
    ])
    .is_err());
    assert!(matches!(
        Cli::try_parse_from(["jope-bench", "automl"])
            .unwrap()
            .command,
        Command::Automl
    ));
}

#[test]
fn default_run_workers_cap_at_the_host_budget() {
    assert_eq!(default_workers(None, Some(128)), DEFAULT_MAX_WORKERS);
    assert_eq!(default_workers(None, Some(8)), 8);
    assert_eq!(default_workers(None, None), 1);
}

#[test]
fn explicit_run_workers_are_not_capped_but_never_zero() {
    assert_eq!(default_workers(Some(64), Some(128)), 64);
    assert_eq!(default_workers(Some(0), Some(128)), 1);
}
