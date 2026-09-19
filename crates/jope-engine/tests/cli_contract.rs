//! The engine's external command-line contract.
//!
//! `jain-core` (`crates/feat-core/src/jope_invention.rs`) spawns
//! `<bin> --root <work> search --prepared … --cut 0 --output … --evaluations …
//! --seconds … --rng-seed … --qualification --vocabulary <json>` and reads the
//! `search.json` the engine writes (`schema: "parallel.search.v1"`) into its own
//! `jain.jope-search-receipt/v1` receipt. This file pins that argv shape, the
//! global `--root`, the search summary's schema and fields, and the full
//! subcommand list (`tests/fixtures/subcommands.txt`).

use clap::Parser;
use jope_parallel::cli::{command, Cli, Cmd};
use std::path::PathBuf;

/// The exact argv `jain-core` builds (`jope_invention.rs`), argv[0] included.
const JAIN_CORE_ARGV: [&str; 18] = [
    "jope-parallel",
    "--root",
    "/work",
    "search",
    "--prepared",
    "/work/prepared",
    "--cut",
    "0",
    "--output",
    "/work/search",
    "--evaluations",
    "1000",
    "--seconds",
    "600",
    "--rng-seed",
    "20260616",
    "--qualification",
    "--vocabulary",
];

#[test]
fn jain_core_search_argv_parses_with_the_global_root() {
    let mut argv = JAIN_CORE_ARGV.to_vec();
    argv.push("/work/vocabulary.json");
    let cli = Cli::try_parse_from(&argv).expect("the jain-core argv shape must keep parsing");
    assert_eq!(cli.root, PathBuf::from("/work"), "--root is the engine's global flag");
    match cli.command {
        Cmd::Search {
            prepared,
            cut,
            output,
            evaluations,
            seconds,
            qualification,
            seeds,
            vocabulary,
            rng_seed,
        } => {
            assert_eq!(prepared, PathBuf::from("/work/prepared"));
            assert_eq!(cut, 0);
            assert_eq!(output, PathBuf::from("/work/search"));
            assert_eq!(evaluations, 1000);
            assert_eq!(seconds, 600);
            assert!(qualification);
            assert_eq!(seeds, None, "jain-core never passes --seeds");
            assert_eq!(vocabulary, Some(PathBuf::from("/work/vocabulary.json")));
            assert_eq!(rng_seed, 20260616);
        }
        _ => panic!("expected the search subcommand"),
    }
}

#[test]
fn root_defaults_to_the_working_directory_and_stays_global() {
    let cli = Cli::try_parse_from(["jope-parallel", "doctor"]).unwrap();
    assert_eq!(cli.root, PathBuf::from("."));
    assert!(matches!(cli.command, Cmd::Doctor));
    // `--root` after the subcommand is not the contract and must not be accepted silently.
    assert!(Cli::try_parse_from(["jope-parallel", "doctor", "--root", "/x"]).is_err());
    let engine = command();
    let search = engine.find_subcommand("search").expect("search subcommand");
    let flags: Vec<_> = search
        .get_arguments()
        .filter_map(|arg| arg.get_long())
        .collect();
    for flag in [
        "prepared",
        "cut",
        "output",
        "evaluations",
        "seconds",
        "qualification",
        "seeds",
        "vocabulary",
        "rng-seed",
    ] {
        assert!(flags.contains(&flag), "search lost --{flag}: {flags:?}");
    }
}

/// The summary jain-core reads back is a `json!` literal in `gp.rs`; pin its
/// schema and the fields the consumer deserialises (`SearchJson` there).
#[test]
fn search_summary_schema_and_consumer_fields_are_pinned() {
    let gp = include_str!("../src/gp.rs");
    assert!(
        gp.contains(r#""schema":"parallel.search.v1""#),
        "search.json schema literal moved or changed"
    );
    for field in [
        "best_program",
        "best_validation_rmse",
        "completed_evaluations",
        "unique_programs",
        "wall_seconds",
        "vocabulary_sha256",
        "final_tests_opened",
        "seeds_sha256",
        "rng_seed",
    ] {
        assert!(gp.contains(&format!("\"{field}\":")), "search.json lost `{field}`");
    }
}

/// Every subcommand, walked from the clap `Command`, against the committed list.
/// A line is `<name>`, optionally suffixed ` [hidden]` and/or ` [neural]`
/// (present only with `--features neural`).
#[test]
fn every_subcommand_is_listed() {
    let fixture = include_str!("fixtures/subcommands.txt");
    let expected: Vec<String> = fixture
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .filter(|line| cfg!(feature = "neural") || !line.contains("[neural]"))
        .map(|line| line.replace(" [neural]", ""))
        .collect();
    let actual: Vec<String> = command()
        .get_subcommands()
        .map(|sub| {
            if sub.is_hide_set() {
                format!("{} [hidden]", sub.get_name())
            } else {
                sub.get_name().to_owned()
            }
        })
        .collect();
    assert_eq!(
        actual, expected,
        "the engine subcommand list changed; update tests/fixtures/subcommands.txt if intended\nactual:\n{}",
        actual.join("\n")
    );
    let total = fixture
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .count();
    assert_eq!(total, 88, "the engine carries 88 subcommands");
}
