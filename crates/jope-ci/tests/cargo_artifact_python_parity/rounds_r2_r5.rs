use super::*;

/// Round two: both refuse with exit status 1 and the port's words differ, because the Python
/// printed a `tomllib.TOMLDecodeError` and the port reads TOML with the `toml` crate, whose errors
/// are worded differently (the port does not reproduce `tomllib`'s messages).
const R2_MESSAGE_DIFFERS: &[&str] = &["toml-invalid-value"];

#[test]
fn cargo_artifact_matches_round_two() {
    replay_round("negative-r2.json", 40, R2_MESSAGE_DIFFERS, &[], &[], &[]);
}

/// Round five (`negative-r5.json`): both refuse, the Python in `tomllib`'s words and the port in
/// the `toml` crate's or its own: the five dates and times `tomllib` refused and the crate read
/// (a seconds field of 60, the year 0000), now refused as a TOML type the port does not read, and
/// three the crate refuses too.
const R5_MESSAGE_DIFFERS: &[&str] = &[
    "toml-r5-dt-second-60",
    "toml-r5-dt-leap-second-datetime",
    "toml-r5-dt-year-zero-date",
    "toml-r5-dt-year-zero-offset",
    "toml-r5-dt-year-zero-local",
    "toml-r5-dt-feb-30",
    "toml-r5-dt-hour-24",
    "toml-r5-datetime-then-invalid",
];

/// Round five: nextest configurations the Python accepted and the port refuses, writing nothing
/// (stricter): every date, time or datetime (the two TOML readers disagree on which are valid),
/// and floats and arrays, which the committed configuration does not use.
const R5_STRICTER: &[&str] = &[
    "toml-r5-dt-offset",
    "toml-r5-dt-offset-fraction",
    "toml-r5-dt-local",
    "toml-r5-dt-local-date",
    "toml-r5-dt-local-time",
    "toml-r5-dt-space-separator",
    "toml-r5-dt-in-array",
    "toml-r5-dt-in-inline-table",
    "toml-r5-dt-in-table",
    "toml-r5-dt-in-array-of-tables",
    "toml-r5-float",
    "toml-r5-float-inf",
    "toml-r5-float-in-inline-table",
    "toml-r5-array",
    "toml-r5-array-empty",
    "toml-r5-array-of-tables",
    "toml-r5-array-nested-deep",
];

/// Round five of the review of PR #117 (`negative-r5.json`): the `toml` crate read TOML dates and
/// times that `tomllib` refused, so `nextest-config` wrote a configuration where the Python
/// refused; the port now reads only the value types the committed configuration uses. And
/// `select`'s `is_file()` of an executable under a directory that may not be searched raised,
/// which the Python printed as the `OSError`; the port refused in other words and now prints it.
#[test]
fn cargo_artifact_matches_round_five() {
    replay_round(
        "negative-r5.json",
        31,
        R5_MESSAGE_DIFFERS,
        R5_STRICTER,
        &[],
        &[],
    );
}
