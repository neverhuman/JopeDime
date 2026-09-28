//! `jope-ci cargo-artifact` against the outputs of the Python it replaced.
//!
//! `tests/fixtures/python-parity/cargo-artifact/python-outputs.json` holds what
//! `deploy/ci/cargo-artifact.py` (archived at 8a55c4fef) printed in the differential run
//! `py-p1-u2-diff-20260927T023243Z`, before it was deleted, with the lane root replaced by
//! `@ROOT@`. The inputs it ran on are committed beside it: the Cargo message stream of a real
//! `cargo build -p jope-paths --bin jope-units-render --message-format=json-render-diagnostics`,
//! that workspace's `cargo metadata` reduced to the fields `select` reads, and the
//! `.config/nextest.toml` of the time. Re-running the archived Python on exactly these committed
//! inputs reproduced every captured output (run `py-p1-fixtures-u2-20260927T033630Z`).
//!
//! Each case runs the real binary on the same inputs, materialised under a fresh directory that
//! takes the place of `@ROOT@`. One difference is intended and is the only one allowed:
//! `usage`, the argument-parser wording (argparse vs clap). Both exit 2 and name the missing
//! `--metadata` and `--messages`. (The OS error of an existing output file, `nextest-exists`, is
//! Python's `[Errno 17] File exists: '<path>'` on both sides.)
//!
//! # Malformed inputs
//!
//! `tests/fixtures/python-parity/cargo-artifact/negative.json` holds what the same Python, as it
//! stood at `origin/main` `f46a9048f` (unmodified, run from its git bytes in a scratch root), did
//! with malformed Cargo metadata and message streams for `select`, `target-dir`, `nextest-config`
//! and `build` (behind a `cargo` on PATH that prints the case's metadata and messages): the exit
//! status, stdout, and either its `cargo-artifact: ...` message or the last line of its
//! traceback. Its `source` and `capture` fields name the custody run and the command. Every case
//! is held to one contract (`judge`): the port is never more permissive. Where the Python
//! succeeded or printed its own message, the port exits the same way and prints the identical
//! stdout and stderr; where the Python crashed with a traceback, the port refuses (exit status 1,
//! one `cargo-artifact: <field>: <what failed and the offending type>` line, nothing on stdout;
//! the text is not CPython's). The exceptions are listed in `STRICTER` (inputs the Python accepted
//! and the port refuses). A refused `nextest-config` leaves the output file exactly as the Python
//! left it: absent, or empty where its crash came after `open("x")` had created it.
//!
//! `negative-r2.json` holds the second round of such cases (the differential review of the
//! fail-closed ports): library messages the Python caught and printed (`OSError`,
//! `UnicodeDecodeError`, `JSONDecodeError` positions, `tomllib`), `repr` of non-printable
//! characters, `--path-map` normalisation, the `nextest-config` order of refusals, NaN and big
//! integer ids. It is replayed by `cargo_artifact_matches_round_two`, with its own exceptions.
//!
//! `negative-r3-digits.json` and `negative-r3-paths.json` hold the third round (review 5333747872
//! of PR #117), captured the same way: Python 3.12's limit of 4300 digits on integer conversion in
//! every JSON and TOML reader, and POSIX's distinct `//` root in `--path-map` and in the printed
//! paths. They are replayed by `cargo_artifact_matches_round_three_digits` and `..._paths`.
//!
//! `negative-r5.json` holds round five (the fail-closed sweep): `nextest-config` sources holding
//! TOML dates and times (five `tomllib` refused and the `toml` crate read), floats and arrays,
//! which the port now refuses as types the committed configuration does not use (`R5_STRICTER`
//! where the Python accepted them), and `select` of an executable under a directory that may not
//! be searched, whose `OSError` the Python printed. `cargo_artifact_matches_round_five` replays it.
//!
//! `negative-r6.json` holds round six (lossy path conversion): `select` and `build` of an
//! executable whose resolved name is not UTF-8, captured under `en_US.UTF-8` (the Python's print
//! raised `UnicodeEncodeError`), `C.UTF-8` and C (it printed the raw bytes), arguments that are not
//! UTF-8, a resolved name longer than `PATH_MAX` and a TOML table `tomllib` read and the `toml`
//! crate refuses. `cargo_artifact_matches_round_six` replays it.
//!
//! `negative-r7.json` holds round seven (the argument grammar, argparse's against clap's): repeated
//! options (a repeated `--path-map` among them), empty values (`--bin ""`, which argparse's truth
//! test made an example named `None`), prefixes, `=` forms, values that look like options or
//! negative numbers, and the word `help` where a command is expected.
//! `cargo_artifact_matches_round_seven` replays it.

use serde_json::Value;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/python-parity/cargo-artifact")
}

fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn cargo_artifact_matches_the_captured_python_outputs() {
    let fixture = fixture();
    let captured: Value =
        serde_json::from_str(&fs::read_to_string(fixture.join("python-outputs.json")).unwrap())
            .unwrap();
    let cases = captured["cases"].as_array().unwrap();
    let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "select",
            "select-map",
            "select-badmap",
            "select-unfinished",
            "select-failed",
            "select-wrongpkg",
            "select-example",
            "select-ambiguous",
            "usage",
            "nextest-config",
            "nextest-exists"
        ],
        "every captured case is replayed"
    );

    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-parity-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    let materialise = |name: &str| {
        fs::read_to_string(fixture.join(name))
            .unwrap()
            .replace("@ROOT@", &root_text)
    };
    fs::create_dir_all(root.join("target/debug")).unwrap();
    executable(
        &root.join("target/debug/jope-units-render"),
        "#!/bin/sh\nexit 0\n",
    );
    fs::write(root.join("metadata.json"), materialise("metadata.json")).unwrap();
    let build = materialise("build.jsonl");
    // The message-stream variants of the differential run, derived the same way.
    let unfinished: String = build
        .split_inclusive('\n')
        .filter(|line| !line.contains("\"build-finished\""))
        .collect();
    let failed = build.replace("\"success\":true", "\"success\":false");
    for (name, text) in [
        ("build", build.clone()),
        ("unfinished", unfinished),
        ("failed", failed),
        ("doubled", format!("{build}{build}")),
    ] {
        fs::write(root.join(format!("{name}.jsonl")), text).unwrap();
    }
    // `nextest-config` asks Cargo for the target directory; this Cargo answers as the lane's did.
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    executable(
        &bin.join("cargo"),
        &format!("#!/bin/sh\nprintf '{{\"target_directory\": \"%s\"}}\\n' '{root_text}/target'\n"),
    );
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let output_file = root.join(format!("{name}.toml"));
        if name == "nextest-exists" {
            fs::write(&output_file, "").unwrap();
        }
        let output_text = output_file.to_str().unwrap().to_owned();
        let messages = root.join(format!("{}.jsonl", case["messages"].as_str().unwrap()));
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a.as_str()
                    .unwrap()
                    .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                    .replace("@MESSAGES@", messages.to_str().unwrap())
                    .replace("@SOURCE@", fixture.join("nextest.toml").to_str().unwrap())
                    .replace("@OUT@", &output_text)
                    .replace("@ROOT@", &root_text)
            })
            .collect();
        let run = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root)
            .output()
            .unwrap();
        let normalise = |bytes: &[u8]| {
            String::from_utf8_lossy(bytes)
                .replace(&output_text, "@OUT@")
                .replace(&root_text, "@ROOT@")
        };
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let (want_stdout, want_stderr) = (
            case["stdout"].as_str().unwrap(),
            case["stderr"].as_str().unwrap(),
        );
        assert_eq!(
            Some(i64::from(run.status.code().unwrap())),
            case["exit"].as_i64(),
            "{name}: exit status; stderr {stderr}"
        );
        match name {
            "usage" => {
                // Allowed difference: argument-parser wording.
                assert_eq!((stdout.as_str(), want_stdout), ("", ""), "{name}");
                assert!(
                    want_stderr.starts_with("usage: cargo-artifact.py select ")
                        && want_stderr.contains(
                            "error: the following arguments are required: --metadata, --messages"
                        ),
                    "{name}: captured Python stderr {want_stderr}"
                );
                assert!(
                    stderr.contains("the following required arguments were not provided")
                        && stderr.contains("--metadata <METADATA>")
                        && stderr.contains("--messages <MESSAGES>"),
                    "{name}: Rust stderr {stderr}"
                );
            }
            "nextest-exists" => {
                // The OS error of an existing output, as the Python printed it; the file stays.
                assert_eq!((stdout.as_str(), want_stdout), ("", ""), "{name}");
                assert_eq!(
                    want_stderr, "cargo-artifact: [Errno 17] File exists: '@OUT@'\n",
                    "{name}: captured Python stderr"
                );
                assert_eq!(stderr, want_stderr, "{name}: stderr");
                assert_eq!(fs::read_to_string(&output_file).unwrap(), "", "{name}");
            }
            _ => {
                assert_eq!(stdout, want_stdout, "{name}: stdout");
                assert_eq!(stderr, want_stderr, "{name}: stderr");
            }
        }
        if name == "nextest-config" {
            let written = fs::read_to_string(&output_file)
                .unwrap()
                .replace(&root_text, "@ROOT@");
            assert_eq!(
                written,
                captured["nextest_output"].as_str().unwrap(),
                "{name}: the relocated nextest config is byte-identical"
            );
        }
    }
}

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

/// Round six: what the Python accepted and the port refuses, writing nothing (stricter):
///
/// - an executable whose resolved name is not UTF-8, under C and `C.UTF-8`, where the Python's
///   `surrogateescape` stdout printed its raw bytes (the port refuses in every locale, with the
///   `UnicodeEncodeError` the Python printed under `en_US.UTF-8`, checked below);
/// - a resolved name longer than `PATH_MAX`, which `realpath(3)` refuses (`[Errno 36]`) and
///   Python's `resolve()` built and printed;
/// - a TOML document `tomllib` read and the `toml` crate refuses (a dotted key extending a table a
///   header created implicitly: `[a.b.c]`, `[a]`, `b.d = 1`).
const R6_STRICTER: &[&str] = &[
    "select-r6-resolved-ff-cu",
    "select-r6-resolved-ff-c",
    "select-r6-resolved-run-cu",
    "select-r6-resolved-run-c",
    "select-r6-resolved-eacute-cu",
    "select-r6-resolved-eacute-c",
    "select-r6-resolved-partial-cu",
    "select-r6-resolved-partial-c",
    "select-r6-resolved-mid-cu",
    "select-r6-resolved-mid-c",
    "build-r6-resolved-ff-cu",
    "select-r6-map-host-link-ff-cu",
    "select-r6-resolved-name-too-long",
    "toml-r6-implicit-then-dotted",
];

/// Round six: an argument that is not UTF-8, which the port refuses before reading anything (exit
/// status 2, `jope-ci: argument N ... is not UTF-8`); the Python took it surrogate-escaped and
/// printed the executable (stricter), or refused printing a resolved name that is not UTF-8 (both
/// refuse, in other words).
const R6_ARGV_REFUSED: &[&str] = &[
    "select-r6-map-host-arg-ff-en",
    "select-r6-map-host-arg-ff-cu",
    "select-r6-metadata-arg-ff-en",
    "select-r6-metadata-arg-ff-cu",
];

/// Round six of the review of PR #117 (`negative-r6.json`): an executable whose resolved name is
/// not UTF-8 (reached through a link to a directory named with the byte `0xff`, a run of such
/// bytes, one after `é`, an incomplete sequence, bytes between others), by `select`, `build` and
/// through a `--path-map` host link. The port printed a lossy conversion (U+FFFD, a path that does
/// not exist) and exited 0 where the Python, under `en_US.UTF-8`, raised `UnicodeEncodeError`
/// printing it; it now refuses with that message, position for position, in every locale. The
/// UTF-8 controls (a link, a name that is not ASCII) print as before.
#[test]
fn cargo_artifact_matches_round_six() {
    let results = replay_round(
        "negative-r6.json",
        29,
        &[],
        R6_STRICTER,
        R6_ARGV_REFUSED,
        &[],
    );
    // Under C and `C.UTF-8` the port refuses with the message the Python printed under
    // `en_US.UTF-8` for the same name.
    for (name, (exit, stdout, stderr)) in &results {
        let Some(base) = name
            .strip_suffix("-cu")
            .or_else(|| name.strip_suffix("-c"))
            .filter(|_| name.contains("-resolved-") || name.contains("-map-host-link-"))
        else {
            continue;
        };
        let (_, _, en) = &results[&format!("{base}-en")];
        assert_eq!(
            (*exit, stdout.as_str(), stderr),
            (1, "", en),
            "{name}: the port refuses as it refuses {base}-en"
        );
    }
}

/// A fixture string with `@Xhh@` for the byte `hh` (a name that is not UTF-8), as its bytes.
fn unescape(text: &str) -> std::ffi::OsString {
    use std::os::unix::ffi::OsStringExt;
    let mut out = vec![];
    let mut rest = text;
    while let Some(at) = rest.find("@X") {
        match (rest.get(at + 2..at + 4), rest.get(at + 4..at + 5)) {
            (Some(hex), Some("@")) if hex.bytes().all(|b| b.is_ascii_hexdigit()) => {
                out.extend_from_slice(&rest.as_bytes()[..at]);
                out.push(u8::from_str_radix(hex, 16).unwrap());
                rest = &rest[at + 5..];
            }
            _ => {
                out.extend_from_slice(&rest.as_bytes()[..at + 2]);
                rest = &rest[at + 2..];
            }
        }
    }
    out.extend_from_slice(rest.as_bytes());
    std::ffi::OsString::from_vec(out)
}

/// A codec position the fixture wrote `@ROOT+k@` (k characters past the capture's root) at the
/// same place past this root, `root_chars` characters long.
fn place_positions(text: &str, root_chars: usize) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("@ROOT+") {
        let after = &rest[at + "@ROOT+".len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        assert_eq!(&after[digits..=digits], "@", "{text}");
        out.push_str(&rest[..at]);
        out.push_str(&(root_chars + after[..digits].parse::<usize>().unwrap()).to_string());
        rest = &after[digits + 1..];
    }
    out.push_str(rest);
    out
}

/// The capture's `deepdir`: a directory whose absolute name is exactly `len` bytes under `base`
/// (`deep`, then components of 200 `d`, one of 100 when needed, and one of `e`).
fn deepdir(base: &str, len: usize) -> String {
    let mut dir = format!("{base}/deep");
    while len - dir.len() > 260 {
        dir = format!("{dir}/{}", "d".repeat(200));
    }
    let mut rest = len - dir.len() - 1;
    if rest > 200 {
        dir = format!("{dir}/{}", "d".repeat(100));
        rest = len - dir.len() - 1;
    }
    format!("{dir}/{}", "e".repeat(rest))
}

/// Round six's inputs under `root`, created once and reused by later cases: `dirs` (each holding
/// the executable `demo`), `symlinks`, a `deep` directory reached through a link, and a copy of the
/// metadata (`metadata_copy_to`).
fn round_six_tree(case: &Value, root: &Path) {
    for dir in case["dirs"].as_array().into_iter().flatten() {
        let dir = root.join(unescape(dir.as_str().unwrap()));
        if !dir.join("demo").exists() {
            fs::create_dir_all(&dir).unwrap();
            executable(&dir.join("demo"), "#!/bin/sh\nexit 0\n");
        }
    }
    for link in case["symlinks"].as_array().into_iter().flatten() {
        let path = root.join(link["path"].as_str().unwrap());
        if fs::symlink_metadata(&path).is_err() {
            std::os::unix::fs::symlink(unescape(link["target"].as_str().unwrap()), path).unwrap();
        }
    }
    if let Some(deep) = case.get("deep") {
        let link = root.join(deep["link"].as_str().unwrap());
        if fs::symlink_metadata(&link).is_err() {
            let dir = deepdir(
                root.to_str().unwrap(),
                deep["length"].as_u64().unwrap() as usize,
            );
            fs::create_dir_all(&dir).unwrap();
            // The executable's own name is past PATH_MAX: created from inside the directory.
            let made = Command::new("/bin/sh")
                .args(["-c", "cd \"$1\" && cp -p \"$2\" \"$3\"", "sh"])
                .arg(&dir)
                .arg(root.join("bin/demo"))
                .arg(deep["file"].as_str().unwrap())
                .status()
                .unwrap();
            assert!(made.success());
            std::os::unix::fs::symlink(&dir, &link).unwrap();
        }
    }
    if let Some(to) = case.get("metadata_copy_to") {
        fs::copy(
            root.join("metadata.json"),
            root.join(unescape(to.as_str().unwrap())),
        )
        .unwrap();
    }
}

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
    let usage: Vec<(&str, &str)> = R7_STRICTER.iter().chain(R7_BOTH_REFUSE).copied().collect();
    replay_round("negative-r7.json", 47, &[], &[], &[], &usage);
    // Each list holds what its documentation says the Python did.
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join("negative-r7.json")).unwrap())
            .unwrap();
    let python_exit = |name: &str| {
        doc["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .map(|c| c["exit"].as_i64().unwrap())
            .unwrap()
    };
    for (name, _) in R7_STRICTER {
        assert_eq!(python_exit(name), 0, "{name}: listed as stricter");
    }
    for (name, _) in R7_BOTH_REFUSE {
        assert_ne!(python_exit(name), 0, "{name}: listed as refused by both");
    }
}

/// Replay every case of a round fixture (the layout of `negative-r2.json`); `message_differs` names
/// the cases where both refuse in different words, `stricter` those the Python accepted and the
/// port refuses (writing nothing), `argv_refused` those holding an argument that is not UTF-8,
/// which the port refuses before reading anything, and `usage_refused` those the port's argument
/// parser refuses (exit status 2, clap's `error: ...` containing the text given, nothing on stdout,
/// nothing written), whatever the Python did. A case's `nosearch_mode` is the mode of
/// `@ROOT@/nosearch`, which holds an executable `demo`, during its run. Round six's cases add
/// names that are not UTF-8 (`@Xhh@`, in arguments too), links, a deep directory and the locale
/// the Python ran under (see [`round_six_tree`]); a codec position `@ROOT+k@` counts this root.
/// Returns each case's exit status, stdout and stderr (the root written `@ROOT@`).
fn replay_round(
    file: &str,
    count: usize,
    message_differs: &[&str],
    stricter: &[&str],
    argv_refused: &[&str],
    usage_refused: &[(&str, &str)],
) -> std::collections::BTreeMap<String, (i64, String, String)> {
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join(file)).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        count,
        "{file}: every captured case is replayed"
    );
    for name in message_differs
        .iter()
        .chain(stricter)
        .chain(argv_refused)
        .chain(usage_refused.iter().map(|(name, _)| name))
    {
        assert!(
            cases.iter().any(|c| c["name"] == *name),
            "{file}: {name} is not a case"
        );
    }
    let unhex = |text: &str| -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    };
    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-round-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    fs::create_dir(root.join("bin")).unwrap();
    executable(&root.join("bin/demo"), "#!/bin/sh\nexit 0\n");
    fs::create_dir(root.join("nosearch")).unwrap();
    executable(&root.join("nosearch/demo"), "#!/bin/sh\nexit 0\n");
    let shims = root.join("shims");
    fs::create_dir(&shims).unwrap();
    executable(
        &shims.join("cargo"),
        "#!/bin/bash\ncase \"$1\" in\n  metadata) cat \"$CA_ROOT/metadata.json\"; e=$(cat \"$CA_ROOT/metadata.exit\") ;;\n  build) cat \"$CA_ROOT/messages.jsonl\"; e=$(cat \"$CA_ROOT/build.exit\") ;;\nesac\n[ \"$e\" = kill ] && kill -9 $$\nexit \"$e\"\n",
    );
    let path = format!(
        "{}:{}",
        shims.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    let mut results = std::collections::BTreeMap::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        for file in ["metadata.json", "messages.jsonl", "source.toml", "out.toml"] {
            let _ = fs::remove_file(root.join(file));
        }
        // Codec positions counted past the capture's root, placed past this one.
        let mut case = case.clone();
        if let Some(stderr) = case.get("stderr").and_then(Value::as_str) {
            case["stderr"] = Value::from(place_positions(stderr, root_text.chars().count()));
        }
        // A deep directory has a fixed absolute length, so the components the Python printed
        // under the capture's root are those `deepdir` gives under this one.
        if let (Some(deep), Some(stdout)) = (
            case.get("deep").cloned(),
            case.get("stdout").and_then(Value::as_str),
        ) {
            let file = deep["file"].as_str().unwrap();
            assert!(
                stdout.starts_with("@ROOT@/deep/") && stdout.ends_with(&format!("/{file}\n")),
                "{name}: {stdout}"
            );
            let here = deepdir(&root_text, deep["length"].as_u64().unwrap() as usize)
                .replacen(&root_text, "@ROOT@", 1);
            case["stdout"] = Value::from(format!("{here}/{file}\n"));
        }
        let case = &case;
        for (key, file) in [
            ("metadata", "metadata.json"),
            ("messages", "messages.jsonl"),
            ("source", "source.toml"),
        ] {
            let bytes = match (case.get(format!("{key}_hex")), case.get(key)) {
                (Some(hex), _) => unhex(hex.as_str().unwrap()),
                (None, Some(text)) => text
                    .as_str()
                    .unwrap()
                    .replace("@ROOT@", &root_text)
                    .into_bytes(),
                (None, None) => continue,
            };
            fs::write(root.join(file), bytes).unwrap();
        }
        for (key, file) in [
            ("metadata_exit", "metadata.exit"),
            ("build_exit", "build.exit"),
        ] {
            let exit = case.get(key).and_then(Value::as_str).unwrap_or("0");
            fs::write(root.join(file), exit).unwrap();
        }
        if case["output_exists"] == true {
            fs::write(root.join("out.toml"), "").unwrap();
        }
        round_six_tree(case, &root);
        let argv: Vec<std::ffi::OsString> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                unescape(
                    &a.as_str()
                        .unwrap()
                        .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                        .replace("@MESSAGES@", root.join("messages.jsonl").to_str().unwrap())
                        .replace("@SOURCE@", root.join("source.toml").to_str().unwrap())
                        .replace("@OUT@", root.join("out.toml").to_str().unwrap())
                        .replace("@ROOT@", &root_text),
                )
            })
            .collect();
        let nosearch = case
            .get("nosearch_mode")
            .and_then(Value::as_str)
            .map(|m| u32::from_str_radix(m, 8).unwrap());
        if let Some(mode) = nosearch {
            fs::set_permissions(root.join("nosearch"), fs::Permissions::from_mode(mode)).unwrap();
        }
        // A user who may search anything (root) cannot reproduce a refused search.
        let searchable = nosearch.is_some() && fs::metadata(root.join("nosearch/demo")).is_ok();
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-ci"));
        command
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env("CA_ROOT", &root)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root);
        // The locale the Python ran under (the port reads none).
        match case.get("locale").and_then(Value::as_str) {
            Some("en") => {
                command
                    .env_remove("LC_ALL")
                    .env_remove("LC_CTYPE")
                    .env_remove("LC_MESSAGES")
                    .env("LANG", "en_US.UTF-8");
            }
            Some("cu") => {
                command.env("LC_ALL", "C.UTF-8");
            }
            Some("c") => {
                command.env("LC_ALL", "C");
            }
            _ => {}
        }
        let run = command.output().unwrap();
        fs::set_permissions(root.join("nosearch"), fs::Permissions::from_mode(0o755)).unwrap();
        if searchable {
            eprintln!("{name}: skipped, this user may search a directory of that mode");
            continue;
        }
        let normalise = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&root_text, "@ROOT@");
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let exit = i64::from(run.status.code().unwrap());
        record(case, exit, &stdout, &stderr);
        results.insert(name.to_owned(), (exit, stdout.clone(), stderr.clone()));
        let usage = usage_refused.iter().find(|(n, _)| *n == name).map(|u| u.1);
        let verdict = if let Some(text) = usage {
            // Refused by the argument parser before anything is read or written.
            if exit == 2
                && stdout.is_empty()
                && stderr.starts_with("error: ")
                && stderr.contains(text)
            {
                Ok(())
            } else {
                Err(format!(
                    "listed as a usage refusal ({text:?}), but the port exited {exit} {stdout:?} {stderr:?}"
                ))
            }
        } else if argv_refused.contains(&name) {
            // Refused before anything is read: one usage line, nothing on stdout, no output file.
            let refused = exit == 2
                && stdout.is_empty()
                && stderr.starts_with("jope-ci: argument ")
                && stderr.contains(" is not UTF-8")
                && stderr.lines().count() == 1;
            if refused && (case["exit"] == 0 || case["exit"] == 1) {
                Ok(())
            } else {
                Err(format!(
                    "listed as an argument refusal, but the Python exited {} and the port {exit} {stderr:?}",
                    case["exit"]
                ))
            }
        } else if stricter.contains(&name) {
            // The Python accepted; the port refuses (one `cargo-artifact:` line, nothing on stdout)
            // and leaves no output (checked below). The `toml` crate's parse error of a nextest
            // source spans lines (it quotes the line).
            let refused = exit == 1
                && stdout.is_empty()
                && stderr.starts_with("cargo-artifact: ")
                && (stderr.lines().count() == 1 || case.get("source").is_some());
            if refused && case["exit"] == 0 {
                Ok(())
            } else {
                Err(format!(
                    "listed as stricter, but the Python exited {} and the port {exit} {stderr:?}",
                    case["exit"]
                ))
            }
        } else if message_differs.contains(&name) {
            let refused = exit == 1
                && stdout.is_empty()
                && stderr.starts_with("cargo-artifact: ")
                && case["exit"] == 1
                && case["stdout"] == "";
            if refused {
                Ok(())
            } else {
                Err(format!(
                    "both must refuse: the port {exit} {stdout:?} {stderr:?}"
                ))
            }
        } else {
            judge(case, exit, &stdout, &stderr)
        };
        if let Err(why) = verdict {
            failures.push(format!("{name}: {why}"));
        }
        // The output file as the Python left it (absent, empty, or the written configuration).
        let left = fs::read_to_string(root.join("out.toml"))
            .ok()
            .map(|t| t.replace(&root_text, "@ROOT@"));
        let python_left =
            if stricter.contains(&name) || argv_refused.contains(&name) || usage.is_some() {
                None
            } else {
                case["output_after"].as_str().map(str::to_owned)
            };
        if left != python_left {
            failures.push(format!(
                "{name}: the output is {left:?}, the Python left {python_left:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
    results
}

/// Malformed inputs the Python accepted (exit status 0) and the port refuses: a
/// `target_directory` that is not a string, which the Python printed as `5` or `None`. (Its
/// `nextest-config` crashed on the same values.)
const STRICTER: &[&str] = &["target-dir-int", "target-dir-null"];

/// Hold one case to the contract (see the module documentation); `Err` names the disagreement.
fn judge(case: &Value, exit: i64, stdout: &str, stderr: &str) -> Result<(), String> {
    let name = case["name"].as_str().unwrap();
    let want_exit = case["exit"].as_i64().unwrap();
    let refused = exit == 1
        && stdout.is_empty()
        && stderr.starts_with("cargo-artifact: ")
        && stderr.ends_with('\n')
        && stderr.lines().count() == 1;
    if STRICTER.contains(&name) {
        return if want_exit == 0 && refused {
            Ok(())
        } else {
            Err(format!(
                "listed as stricter, but the Python exited {want_exit} and the port {exit} ({stderr:?})"
            ))
        };
    }
    if exit != want_exit {
        return Err(format!(
            "exit status: the Python {want_exit}, the port {exit}; stdout {stdout:?}, stderr {stderr:?}"
        ));
    }
    if let Some(raised) = case.get("raised") {
        return if refused {
            Ok(())
        } else {
            Err(format!(
                "the Python raised {raised}; the port did not refuse: stdout {stdout:?}, stderr {stderr:?}"
            ))
        };
    }
    let want_stdout = case["stdout"].as_str().unwrap();
    if stdout != want_stdout {
        return Err(format!(
            "stdout: the Python {want_stdout:?}, the port {stdout:?}"
        ));
    }
    let want_stderr = case["stderr"].as_str().unwrap();
    if stderr != want_stderr {
        return Err(format!(
            "stderr: the Python {want_stderr:?}, the port {stderr:?}"
        ));
    }
    Ok(())
}

/// With `JOPE_NEGATIVE_REPORT` naming a file, every replayed case appends one tab-separated line
/// to it: the fixture, the case, the Python's exit status and message (its exception line, or its
/// stderr or stdout line) and the port's. The custody runs keep this table; the test never reads
/// it.
fn record(case: &Value, exit: i64, stdout: &str, stderr: &str) {
    let Some(path) = std::env::var_os("JOPE_NEGATIVE_REPORT") else {
        return;
    };
    let first = |text: &str| -> String { text.lines().next().unwrap_or("").to_owned() };
    // Round six's stdout that is not UTF-8 (the raw bytes the Python printed under C and
    // `C.UTF-8`) is recorded as its hex.
    let python = match case.get("raised") {
        Some(raised) => format!("raised {}", raised.as_str().unwrap()),
        None if !case["stderr"].as_str().unwrap().is_empty() => {
            format!("refused {}", first(case["stderr"].as_str().unwrap()))
        }
        None => match case.get("stdout").and_then(Value::as_str) {
            Some(stdout) => format!("printed {}", first(stdout)),
            None => format!("printed the bytes {}", case["stdout_hex"].as_str().unwrap()),
        },
    };
    let port = if stderr.is_empty() {
        format!("printed {}", first(stdout))
    } else {
        format!("refused {}", first(stderr))
    };
    let line = format!(
        "cargo-artifact/negative.json\t{}\t{}\t{python}\t{exit}\t{port}\n",
        case["name"].as_str().unwrap(),
        case["exit"]
    );
    use std::io::Write as _;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap()
        .write_all(line.as_bytes())
        .unwrap();
}

#[test]
fn cargo_artifact_refuses_every_input_the_python_refused() {
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join("negative.json")).unwrap())
            .unwrap();
    assert!(
        doc["source"].is_string() && doc["capture"].is_string(),
        "negative.json names its capture run and command"
    );
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 50, "every captured case is replayed");
    for name in STRICTER {
        assert!(
            cases.iter().any(|c| c["name"] == *name),
            "{name} is not a case"
        );
    }

    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-negative-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    fs::create_dir(root.join("bin")).unwrap();
    executable(&root.join("bin/demo"), "#!/bin/sh\nexit 0\n");
    fs::write(
        root.join("nextest.toml"),
        doc["nextest_source"].as_str().unwrap(),
    )
    .unwrap();
    let shims = root.join("shims");
    fs::create_dir(&shims).unwrap();
    executable(
        &shims.join("cargo"),
        "#!/bin/bash\ncase \"$1\" in\n  metadata) cat \"$CA_ROOT/metadata.json\" ;;\n  build) cat \"$CA_ROOT/messages.jsonl\" ;;\nesac\n",
    );
    let path = format!(
        "{}:{}",
        shims.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let output_file = root.join("out.toml");
        let _ = fs::remove_file(&output_file);
        fs::write(
            root.join("metadata.json"),
            case["metadata"]
                .as_str()
                .unwrap()
                .replace("@ROOT@", &root_text),
        )
        .unwrap();
        let mut messages = case["messages"]
            .as_str()
            .unwrap()
            .replace("@ROOT@", &root_text)
            .into_bytes();
        let mut case = case.clone();
        if let Some(bytes) = case.get("append_bytes").cloned() {
            // The codec position counts the stream Cargo printed, whose `@ROOT@` was the capture's
            // root: here the appended byte sits after this root, so the Python's message names the
            // position it would have had on these bytes.
            let stderr = case["stderr"].as_str().unwrap();
            let at = stderr.find("in position ").unwrap() + "in position ".len();
            let digits = stderr[at..].bytes().take_while(u8::is_ascii_digit).count();
            let placed = format!(
                "{}{}{}",
                &stderr[..at],
                messages.len(),
                &stderr[at + digits..]
            );
            case["stderr"] = Value::from(placed);
            messages.extend(
                bytes
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| u8::try_from(b.as_u64().unwrap()).unwrap()),
            );
        }
        fs::write(root.join("messages.jsonl"), messages).unwrap();
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a.as_str()
                    .unwrap()
                    .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                    .replace("@MESSAGES@", root.join("messages.jsonl").to_str().unwrap())
                    .replace("@SOURCE@", root.join("nextest.toml").to_str().unwrap())
                    .replace("@OUT@", output_file.to_str().unwrap())
            })
            .collect();
        let run = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env("CA_ROOT", &root)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root)
            .output()
            .unwrap();
        let normalise = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&root_text, "@ROOT@");
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let exit = i64::from(run.status.code().unwrap());
        record(&case, exit, &stdout, &stderr);
        if let Err(why) = judge(&case, exit, &stdout, &stderr) {
            failures.push(format!("{name}: {why}"));
        }
        // A refused `nextest-config` leaves the output as the Python did: its `TypeError` came
        // after `open("x")` had created the (still empty) file; every other refusal came first.
        if name.starts_with("nextest-config") && exit != 0 {
            let left = fs::read(&output_file).ok();
            let python_left = case.get("raised").map(|_| Vec::new());
            if left != python_left {
                failures.push(format!(
                    "{name}: the output is {left:?}, the Python left {python_left:?}"
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Negative control: `judge` fails an altered expectation, a report in place of a refusal and
    // a success in place of a crash.
    let ok = cases.iter().find(|c| c["name"] == "select-ok").unwrap();
    let ok_stdout = ok["stdout"].as_str().unwrap();
    assert_eq!(judge(ok, 0, ok_stdout, ""), Ok(()));
    assert!(judge(ok, 0, &ok_stdout.replace("demo", "dema"), "").is_err());
    assert!(judge(ok, 1, "", "cargo-artifact: x\n").is_err());
    let crash = cases
        .iter()
        .find(|c| c["name"] == "executable-int")
        .unwrap();
    assert_eq!(judge(crash, 1, "", "cargo-artifact: x\n"), Ok(()));
    assert!(judge(crash, 0, ok_stdout, "").is_err());
    assert!(judge(crash, 1, ok_stdout, "cargo-artifact: x\n").is_err());
    let message = cases.iter().find(|c| c["name"] == "success-one").unwrap();
    assert!(judge(message, 1, "", "cargo-artifact: another message\n").is_err());
}
