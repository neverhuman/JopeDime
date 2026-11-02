//! The four `jope-records` hygiene guards against the outputs of the Python they replaced.
//!
//! `tests/fixtures/python-parity/python-outputs.json` holds what `deploy/ci/coverage.py`,
//! `rust-test-scope-check.py`, `parity-census.py check` and `readme-benchmarks.py` (archived at
//! 8a55c4fef) printed in the differential run `py-p1-u3-diff-20260927T024147Z`, before they were
//! deleted. The inputs are committed beside them, taken from that run's tree (the U2 commit as
//! first written on `rr2/python-p1`, replayed on landing as the commit whose subject ends in
//! "(P1 U2)", with a different `deploy/ci/coverage.json`, plus the run's working changes;
//! recovered from an export of it).
//! These files are the record of those inputs; the commit map at the end of
//! `docs/internal/core-ip/work-orders/evidence/PY-P1-python-removal.md` names the original:
//!
//! - `coverage/`: the coverage map (`mapping.json`) and the nine malformed variants the run
//!   derived from it, stored as `ed` scripts with the SHA-256 of the file the Python read; the
//!   discovery inputs (`cargo metadata` reduced to what discovery reads, the repository file
//!   list, the workflow texts); `discovered.json`, Python's discovery output, which is also the
//!   inventory every `validate` case checks the map against.
//! - `parity/`: the capability matrix (the home paths in seven notes cells written `~/`, which
//!   the no-home-paths lint requires) and its eight variants as `ed` scripts, the `jope-parallel
//!   --help` text, and `tree.json`: what `parity-check` reads from the tree (binaries and
//!   examples, clap subcommands, Python tools, `jope-py` commands, systemd units, and whether each
//!   row's target existed), captured from the export by `capture_parity_tree`.
//! - `readme/`: every input the README renderer reads at that tree, the stale README of the run
//!   and what `--write` produced from it (the committed README of the time).
//! - `scope/`: the workspace package names, what `rust-test-scope.sh` emitted for both scopes, and
//!   the script text.
//!
//! The only allowed differences are the two remediation hints that name the Rust command instead
//! of the retired script (`parity-preamble`, `readme-stale`), asserted as exact substitutions.
//!
//! # Malformed inputs
//!
//! The `negative*.json` fixtures beside those hold what the same four tools, as they stood at
//! `origin/main` `f46a9048f` (unmodified, run from their git bytes in a scratch root), did with
//! malformed inputs: each case is a variant of a committed input (an `ed` script against it plus
//! the SHA-256 of the file the Python read, or for the README renderer a set of such edits across
//! its evidence files), with the exit status, stdout and either the message the Python printed on
//! stderr or the last line of its traceback. The `source` and `capture` fields name the custody
//! run and the command. The fixtures are:
//!
//! - `coverage/negative-validate.json` (variants of `coverage/mapping.json`, validated against
//!   `coverage/discovered.json`) and `coverage/negative-discover.json` (variants of
//!   `coverage/metadata.json`);
//! - `scope/negative.json`, variants of `scope/metadata.json` (the workspace packages of
//!   `scope/scope.json` as `cargo metadata` reports them);
//! - `parity/negative.json`, variants of `parity/negative-metadata.json` checked against
//!   `parity/negative-matrix.md` in a tree that holds only the two target files the matrix names;
//! - `readme/negative.json`, edits of the `readme/root` evidence rendered with `--write` from
//!   `readme/README.stale.md`.
//!
//! Every case is held to one contract (`compare`): the port is never more permissive than the
//! Python. Where the Python exited 0, the port prints the same stdout (or, for discovery, output
//! with the same SHA-256) and exits 0; where it printed a failing report, the port prints the
//! identical report and exits 1; where it refused with a message (`SystemExit`), the port refuses
//! with the identical message (which the binary prints after `jope-records: `); where it crashed
//! with a traceback, the port refuses too (exit status 1, nothing on stdout), with an error that
//! names the field and the offending type. That error is not compared with CPython's exception
//! text. The only exceptions are listed in `RUST_STRICTER`: malformed inputs the Python accepted
//! and the port refuses. `negative_replay_detects_a_changed_outcome` shows `compare` failing on
//! altered expectations.
//!
//! # Round two
//!
//! A differential review of those ports raised further classes, captured the same way (`source`
//! and `capture` name the run):
//!
//! - `process.json`: the coverage, scope and parity guards run end to end by the binary, each on a
//!   tree and behind a `cargo` (and `git`) printing recorded output: Cargo's stdout and stderr as
//!   `text=True` decoded them (a BOM, UTF-16, bytes that are not UTF-8 in either stream), Python's
//!   `\s`, `\w` and `\d` in the workflow, clap, `--help` and argparse readers, `int()` of Unicode
//!   digits, `repr` of non-printable characters, a killed script, CRLF sources, names that are not
//!   UTF-8, and the order of each guard's steps. `judge_process` holds each case to the contract,
//!   with `PROCESS_REFUSES_REPORT` as its only exception.
//! - `readme/negative-r2.json`: NaN and infinities, Python 3.12's `sorted`, `max` and `sum` of
//!   the same-cut seconds, NaN RMSEs tied by id, `a == b == c`, `repr`, and `$` before a final
//!   newline, replayed by `readme_benchmarks_matches_round_two`.
//!
//! # Round three
//!
//! Review 5333747872 of PR #117 found two classes the port accepted and the Python refused,
//! captured the same way: Python 3.12's limit of 4300 digits on integer conversion
//! (`process-r3-digits.json`, `readme/negative-r3.json`) and POSIX's distinct `//` root in path
//! comparisons (`process-r3-paths.json`), each with the boundary cases the Python accepted.
//!
//! An addendum after a further verifier run appends nine cases to `readme/negative-r3.json`, each
//! naming its own `source` and `capture`: `int()` of a README CSV cell (`datasets_scored`) counts
//! leading zeros, so 4301 digits written with leading zeros, a sign, surrounding spaces or `0_`
//! separators refused where 4300 rendered. Where the Python raised that limit's `ValueError`, the
//! port's refusal must carry CPython's message, unless the strict number grammar refused the text
//! first (`REFUSED_BEFORE_THE_LIMIT`).
//!
//! # Round four
//!
//! Review 5334300806 found `int()` and `float()` of a README CSV cell accepting U+001F around the
//! digits, which `str.strip()` removes but Python's conversions refused. Rather than match
//! Python's conversions case by case, every port now reads text as a number with one strict
//! grammar (`py::decimal`, `py::int`, `py::decimal_int`, `py::float`): ASCII whitespace, sign and
//! digits (and for a float, a decimal point and exponent) only, a subset of what Python read.
//! `readme/negative-r4.json` holds the reviewer's cells and the boundary cases of each class; the
//! inputs Python read that the grammar refuses (Unicode whitespace, underscores, other Unicode
//! digits, `nan` and the infinities) are listed in `RUST_STRICTER`, `PROCESS_STRICTER` and
//! `PROCESS_REFUSES_REPORT`, each with its reason.
//!
//! # Round five
//!
//! A verifier found `parity-check` reading a permission error as "absent" where Python 3.12's
//! pathlib raised (`Path.exists()`, `is_dir()`, `is_file()`), and the sweep that followed found
//! the `src/bin` fallback skipping entries its globs listed (a directory, a dangling link or a bare
//! `.rs` named like a source; a `Cargo.toml` that is not a readable file) and the README renderer
//! skipping a report whose name is not UTF-8. `process-r5-sweep.json` and `readme/negative-r5.json`
//! hold what the Python did, each case's tree (and the modes it ran under) as data. A directory
//! that cannot be listed, which pathlib's globs skipped, now refuses: `PROCESS_STRICTER` lists
//! those cases.
//!
//! # Round eight
//!
//! Review 5337854418 found the README renderer's correction totals promoting an integer sum that
//! left 64 bits to a float, so `[2^63-1, 1, -(2^63-1), 45]` totalled 45 where Python's exact 46
//! refused. `readme/negative-r8.json` holds that case and the class around it (every other
//! integer sum, difference, division and comparison the renderer makes on evidence values), each
//! with the outcome of the original; none is listed as stricter.

use jope_records::pyjson::{self, Dict, Value};
use jope_records::{coverage, parity, py, readme, scope_check};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/python-parity")
}

fn text(rel: &str) -> String {
    fs::read_to_string(fixture().join(rel)).unwrap()
}

fn json(rel: &str) -> Value {
    pyjson::parse_str(&text(rel)).unwrap()
}

fn s(value: &Value) -> &str {
    value.as_str().unwrap()
}

/// The captured cases by name: (exit status, stdout, stderr). A long stdout is kept in the file
/// `stdout_file` names instead.
fn captured() -> BTreeMap<String, (i64, String, String)> {
    let doc = json("python-outputs.json");
    let cases: BTreeMap<String, (i64, String, String)> = doc
        .get("cases")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|c| {
            let stdout = match c.get("stdout_file") {
                Some(file) => text(s(file)),
                None => s(c.get("stdout").unwrap()).to_owned(),
            };
            (
                s(c.get("name").unwrap()).to_owned(),
                (
                    c.get("exit").and_then(Value::as_int).unwrap(),
                    stdout,
                    s(c.get("stderr").unwrap()).to_owned(),
                ),
            )
        })
        .collect();
    assert_eq!(cases.len(), 25, "every captured case is present");
    cases
}

/// Apply a `diff -e` script: hunks bottom-up, `Na`, `N[,M]c` and `N[,M]d` only.
fn apply_ed(base: &str, script: &str) -> String {
    let mut lines: Vec<String> = base.split_inclusive('\n').map(str::to_owned).collect();
    let mut commands = script.lines();
    while let Some(command) = commands.next() {
        let (range, op) = command.split_at(command.len() - 1);
        let (start, end): (usize, usize) = match range.split_once(',') {
            Some((a, b)) => (a.parse().unwrap(), b.parse().unwrap()),
            None => {
                let n = range.parse().unwrap();
                (n, n)
            }
        };
        let mut added = vec![];
        if op == "a" || op == "c" {
            for line in commands.by_ref() {
                if line == "." {
                    break;
                }
                added.push(format!("{line}\n"));
            }
        }
        match op {
            "a" => drop(lines.splice(start..start, added)),
            "c" => drop(lines.splice(start - 1..end, added)),
            "d" => drop(lines.splice(start - 1..end, Vec::<String>::new())),
            _ => panic!("unsupported ed command {command}"),
        }
    }
    lines.concat()
}

/// A variant's text, checked against its recorded SHA-256: the file the Python read (for the
/// matrix, that file with the notes' home paths written `~/`).
fn variants(rel: &str, base: &str) -> Vec<(String, String)> {
    json(rel)
        .as_list()
        .unwrap()
        .iter()
        .map(|v| {
            let name = s(v.get("name").unwrap()).to_owned();
            let text = apply_ed(base, s(v.get("ed").unwrap()));
            assert_eq!(
                py::sha256_hex(text.as_bytes()),
                s(v.get("sha256").unwrap()),
                "{rel} {name}: the replayed input is not the one the Python read"
            );
            (name, text)
        })
        .collect()
}

fn check(name: &str, want: &(i64, String, String), ok: bool, stdout: &str) {
    assert_eq!(want.0, if ok { 0 } else { 1 }, "{name}: exit status");
    assert_eq!(stdout, want.1, "{name}: stdout");
    assert_eq!(want.2, "", "{name}: the Python wrote nothing on stderr");
}

#[path = "python_parity/capture.rs"]
mod capture;
#[path = "python_parity/negative.rs"]
mod negative;
#[path = "python_parity/process.rs"]
mod process;

use capture::copy_tree;
use negative::{compare, variant, Rust};
