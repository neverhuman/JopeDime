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

#[path = "cargo_artifact_python_parity/captured.rs"]
mod captured;
#[path = "cargo_artifact_python_parity/negative.rs"]
mod negative;
#[path = "cargo_artifact_python_parity/replay.rs"]
mod replay;
#[path = "cargo_artifact_python_parity/round_six.rs"]
mod round_six;
#[path = "cargo_artifact_python_parity/rounds_r2_r5.rs"]
mod rounds_r2_r5;
#[path = "cargo_artifact_python_parity/rounds_r3_r7.rs"]
mod rounds_r3_r7;

use replay::replay_round;
