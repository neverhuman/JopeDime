use super::*;

/// Round three, integers (`negative-r3-digits.json`): Python 3.12 refuses to convert an integer of
/// more than 4300 decimal digits (`sys.get_int_max_str_digits()`), so `json.loads` of the metadata
/// or of a Cargo message, and `tomllib` of the nextest config, refused one even in a field nothing
/// reads; 4300 digits, either sign, and a float of any length are read. The TOML case refuses on
/// both sides, in the `toml` crate's words (it reads no integer beyond 64 bits).
#[test]
fn cargo_artifact_matches_round_three_digits() {
    replay_round(
        "negative-r3-digits.json",
        14,
        &["toml-int-4301-digits"],
        &[],
        &[],
        &[],
    );
}

/// Round three, paths (`negative-r3-paths.json`): `Path(host) / path.relative_to(container)` with
/// POSIX's distinct `//` root (exactly two leading slashes; three or more are `/`), in both
/// directions and at the root itself, and the paths the refusals print.
#[test]
fn cargo_artifact_matches_round_three_paths() {
    replay_round("negative-r3-paths.json", 15, &[], &[], &[], &[]);
}

/// Round seven: argument grammars argparse accepted (exit status 0) and the port's parser refuses
/// (exit status 2, the text given), stricter:
///
/// - a repeated option other than `--features` (argparse's store kept the last value, `nargs=2`
///   `--path-map` the last pair; clap refuses the second);
/// - an empty `--bin`, however spelled (argparse's truth test made it an example named `None`,
///   which these messages hold as a target whose name is `null` or missing);
/// - a prefix of an option (argparse expanded any unambiguous one);
/// - a value that looks like a negative number after a space (argparse took `-1` as the value;
///   clap reads it as an option; `--bin=-1` is read by both).
const R7_STRICTER: &[(&str, &str)] = &[
    ("select-r7-path-map-twice-last-valid", MULTIPLE),
    ("select-r7-path-map-twice-identical", MULTIPLE),
    ("select-r7-package-twice", MULTIPLE),
    ("select-r7-bin-twice", MULTIPLE),
    ("select-r7-example-twice", MULTIPLE),
    ("select-r7-metadata-twice", MULTIPLE),
    ("select-r7-messages-twice", MULTIPLE),
    ("build-r7-profile-twice", MULTIPLE),
    ("build-r7-manifest-path-twice", MULTIPLE),
    ("build-r7-target-dir-twice", MULTIPLE),
    ("build-r7-no-default-features-twice", MULTIPLE),
    ("target-dir-r7-target-dir-twice", MULTIPLE),
    ("target-dir-r7-manifest-path-twice", MULTIPLE),
    ("nextest-config-r7-output-twice", MULTIPLE),
    ("nextest-config-r7-source-twice", MULTIPLE),
    ("select-r7-bin-twice-last-empty", MULTIPLE),
    ("select-r7-bin-empty-example-null-name", EMPTY_BIN),
    ("select-r7-bin-empty-example-no-name", EMPTY_BIN),
    ("select-r7-bin-equals-empty", EMPTY_BIN),
    (
        "select-r7-abbrev-package",
        "unexpected argument '--pack' found",
    ),
    (
        "select-r7-bin-negative-number",
        "unexpected argument '-1' found",
    ),
];

/// Round seven: grammars both refuse, the port with its parser (exit status 2). The port accepted
/// the first six (exit status 0) before this round. The review's four: a repeated `--path-map` used
/// the first pair (clap appended the second where argparse's store replaced it), so a first pair
/// that maps the executable printed it where the Python's last pair did not (`is not in the
/// subpath`, or a host root without it); an empty `--bin` selected and built a bin named "" where
/// the Python asked for an example named `None` (`found 0`) or crashed building (`TypeError`). The
/// two this round's sweep found: `help` printed help where argparse refused the choice. The port
/// refused the rest before this round too, the empty `--bin` beside `--example` in other words (the
/// group conflict).
const R7_BOTH_REFUSE: &[(&str, &str)] = &[
    ("select-r7-path-map-twice-first-valid", MULTIPLE),
    ("select-r7-path-map-twice-host-differs", MULTIPLE),
    ("select-r7-bin-empty-bin-target", EMPTY_BIN),
    ("build-r7-bin-empty", EMPTY_BIN),
    ("help-r7", "unrecognized subcommand 'help'"),
    ("help-r7-select", "unrecognized subcommand 'help'"),
    ("select-r7-package-twice-last-bad", MULTIPLE),
    (
        "select-r7-path-map-equals-form",
        "2 values required for '--path-map",
    ),
    (
        "select-r7-path-map-one-value",
        "2 values required for '--path-map",
    ),
    (
        "select-r7-path-map-three-values",
        "unexpected argument '/x' found",
    ),
    ("select-r7-bin-empty-and-example", EMPTY_BIN),
    (
        "select-r7-abbrev-ambiguous",
        "unexpected argument '--me' found",
    ),
    (
        "select-r7-package-option-like",
        "unexpected argument '-d' found",
    ),
    (
        "build-r7-flag-with-value",
        "unexpected value 'x' for '--no-default-features' found",
    ),
    (
        "select-r7-terminator-then-value",
        "unexpected argument 'x' found",
    ),
    ("version-r7", "unexpected argument '--version' found"),
    ("select-r7-help-word", "unexpected argument 'help' found"),
];

/// clap's refusal of a second occurrence of an option.
const MULTIPLE: &str = "cannot be used multiple times";
/// clap's refusal of an empty `--bin`.
const EMPTY_BIN: &str = "a value is required for '--bin <BIN>' but none was supplied";

/// Round seven of the review of PR #117 (`negative-r7.json`): the argument grammar, argparse's
/// against clap's. Every case the Python refused, the port refuses; the controls (one
/// `--path-map`, repeated `--features`, an empty `--example`, `--package`, `--target-dir` or
/// `--profile`, `=` forms, `--bin=-1`) print what the Python printed.
#[test]
fn cargo_artifact_matches_round_seven() {
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join("negative-r7.json")).unwrap())
            .unwrap();
    let stricter_names: std::collections::BTreeSet<_> =
        R7_STRICTER.iter().map(|(name, _)| *name).collect();
    let both_refuse_names: std::collections::BTreeSet<_> =
        R7_BOTH_REFUSE.iter().map(|(name, _)| *name).collect();
    assert_eq!(R7_STRICTER.len(), 21, "reviewed stricter-case inventory");
    assert_eq!(R7_BOTH_REFUSE.len(), 17, "reviewed both-refuse inventory");
    assert_eq!(
        stricter_names.len(),
        R7_STRICTER.len(),
        "duplicate stricter case"
    );
    assert_eq!(
        both_refuse_names.len(),
        R7_BOTH_REFUSE.len(),
        "duplicate both-refuse case"
    );
    assert!(
        stricter_names.is_disjoint(&both_refuse_names),
        "one case cannot be both stricter and refused by both"
    );

    let usage_names: std::collections::BTreeSet<_> =
        stricter_names.union(&both_refuse_names).copied().collect();
    let mut observed_usage = std::collections::BTreeSet::new();
    let mut python_accepted = std::collections::BTreeSet::new();
    let mut python_refused = std::collections::BTreeSet::new();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        if usage_names.contains(name) {
            assert!(
                observed_usage.insert(name),
                "duplicate fixture case: {name}"
            );
            if case["exit"].as_i64().unwrap() == 0 {
                python_accepted.insert(name);
            } else {
                python_refused.insert(name);
            }
        }
    }
    assert_eq!(
        observed_usage, usage_names,
        "every reviewed case is present"
    );
    assert_eq!(
        python_accepted, stricter_names,
        "historical Python acceptance set must equal the reviewed stricter cases"
    );
    assert_eq!(
        python_refused, both_refuse_names,
        "historical Python refusal set must equal the reviewed refusal cases"
    );

    let usage: Vec<(&str, &str)> = R7_STRICTER.iter().chain(R7_BOTH_REFUSE).copied().collect();
    replay_round("negative-r7.json", 47, &[], &[], &[], &usage);
}
