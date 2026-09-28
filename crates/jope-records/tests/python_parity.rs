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

#[test]
fn coverage_validate_matches_the_captured_python_outputs() {
    let cases = captured();
    let discovered = json("coverage/discovered.json");
    let discovered = discovered
        .get("items")
        .and_then(Value::as_dict)
        .unwrap()
        .clone();
    let base = text("coverage/mapping.json");
    let mut replayed = vec![("validate".to_owned(), base.clone())];
    replayed.extend(variants("coverage/variants.json", &base));
    assert_eq!(replayed.len(), 10);
    for (name, mapping) in replayed {
        let case = format!("cov-{name}");
        let (stdout, ok) =
            coverage::validate_text(&pyjson::parse_str(&mapping).unwrap(), &discovered).unwrap();
        check(&case, &cases[&case], ok, &stdout);
    }
}

#[test]
fn coverage_discover_matches_the_captured_python_output() {
    let cases = captured();
    let metadata = json("coverage/metadata.json");
    let files = text("coverage/files.txt");
    let workflows = json("coverage/workflows.json");
    let read = |name: &str| -> jope_records::Result<String> {
        Ok(s(workflows
            .get(name)
            .unwrap_or_else(|| panic!("no workflow text for {name}")))
        .to_owned())
    };
    let discovered =
        coverage::discover_from(&metadata, files.lines(), Path::new("@ROOT@"), &read).unwrap();
    let stdout = coverage::discover_document(discovered);
    check("cov-discover", &cases["cov-discover"], true, &stdout);
}

#[test]
fn rust_scope_check_matches_the_captured_python_output() {
    let cases = captured();
    let scope = json("scope/scope.json");
    let packages: BTreeSet<String> = scope
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p).to_owned())
        .collect();
    let (stdout, ok) = scope_check::evaluate(
        &packages,
        s(scope.get("product_stdout").unwrap()),
        s(scope.get("deferred_stdout").unwrap()),
        s(scope.get("script").unwrap()),
    )
    .unwrap();
    check("scope", &cases["scope"], ok, &stdout);
}

/// What `parity-check` read from the tree, as `capture_parity_tree` recorded it.
fn recorded_tree() -> (parity::Tree, BTreeMap<String, bool>) {
    let doc = json("parity/tree.json");
    let strings = |key: &str| -> Vec<String> {
        doc.get(key)
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .map(|v| s(v).to_owned())
            .collect()
    };
    let bins = doc
        .get("bins")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|b| {
            let b: Vec<String> = b
                .as_list()
                .unwrap()
                .iter()
                .map(|v| s(v).to_owned())
                .collect();
            (b[0].clone(), b[1].clone(), b[2].clone(), b[3].clone())
        })
        .collect();
    let jope_py = match doc.get("jope_py") {
        Some(Value::Null) | None => None,
        Some(_) => Some(strings("jope_py")),
    };
    let tree = parity::Tree {
        clap: strings("clap"),
        bins,
        tools: strings("tools"),
        jope_py,
        units: strings("units"),
    };
    let exists = doc
        .get("targets_exist")
        .and_then(Value::as_dict)
        .unwrap()
        .iter()
        .map(|(k, v)| (k.to_owned(), *v == Value::Bool(true)))
        .collect();
    (tree, exists)
}

#[test]
fn parity_check_matches_the_captured_python_outputs() {
    let cases = captured();
    let (tree, exists) = recorded_tree();
    let exists = |head: &str| -> bool {
        *exists
            .get(head)
            .unwrap_or_else(|| panic!("tree.json does not record whether {head} existed"))
    };
    let base = text("parity/capability-matrix.md");
    let help_doc = json("parity/help.json");
    let help = s(help_doc.get("text").unwrap()).to_owned();
    assert_eq!(
        py::sha256_hex(help.as_bytes()),
        s(help_doc.get("sha256").unwrap()),
        "parity/help.json: not the help text the Python read"
    );
    let mut replayed = vec![
        ("parity".to_owned(), base.clone(), None),
        ("parity-help".to_owned(), base.clone(), Some(help.as_str())),
    ];
    for (name, matrix) in variants("parity/variants.json", &base) {
        replayed.push((format!("parity-{name}"), matrix, None));
    }
    assert_eq!(replayed.len(), 10);
    for (name, matrix, help) in replayed {
        let (stdout, ok) = parity::evaluate(&matrix, &tree, help, &exists).unwrap();
        let mut want = cases[&name].clone();
        if name == "parity-preamble" {
            // Allowed difference: the remediation hint names the Rust source, not the census.
            let python = "(re-run `deploy/ci/parity-census.py census`, which emits it)";
            let rust = "(restore it from COUNTS_PREAMBLE in crates/jope-records/src/parity.rs)";
            assert_eq!(want.1.matches(python).count(), 1, "{name}: captured hint");
            want.1 = want.1.replace(python, rust);
        }
        check(&name, &want, ok, &stdout);
    }
}

fn copy_tree(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let target = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_tree(&entry.path(), &target);
        } else {
            fs::copy(entry.path(), &target).unwrap();
        }
    }
}

#[test]
fn readme_benchmarks_matches_the_captured_python_outputs() {
    let cases = captured();
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("root");
    copy_tree(&fixture().join("readme/root"), &root);
    let readme_path = root.join("README.md");
    let stale = text("readme/README.stale.md");
    let written = text("readme/README.written.md");
    // --check on the stale README: exit 1, the out-of-date notice on stderr.
    fs::write(&readme_path, &stale).unwrap();
    let (message, ok) = readme::run(&root, false).unwrap();
    let want = &cases["readme-stale"];
    assert_eq!((want.0, ok), (1, false), "readme-stale: exit status");
    assert_eq!(want.1, "", "readme-stale: stdout");
    // Allowed difference: the remediation hint names the Rust command, not the retired script.
    let python = "Run: python3 deploy/ci/readme-benchmarks.py --write\n";
    let rust = "Run: cargo run -q --locked -p jope-records -- readme-benchmarks --write\n";
    assert_eq!(
        want.2.matches(python).count(),
        1,
        "readme-stale: captured hint"
    );
    assert_eq!(
        message,
        want.2.replace(python, rust),
        "readme-stale: notice"
    );
    // --write from the stale README reproduces what the Python wrote, byte for byte.
    let (message, ok) = readme::run(&root, true).unwrap();
    check("readme-write", &cases["readme-write"], ok, &message);
    assert_eq!(fs::read_to_string(&readme_path).unwrap(), written);
    // --check on the result.
    let (message, ok) = readme::run(&root, false).unwrap();
    check("readme", &cases["readme"], ok, &message);
}

/// Record what `parity-check` reads from a tree, for the replay above. Run it on the export of the
/// differential run's tree:
///
/// ```sh
/// JOPE_PARITY_PIN_ROOT=<export> cargo test -p jope-records --test python_parity -- --ignored capture_parity_tree
/// ```
#[test]
#[ignore = "fixture capture: needs JOPE_PARITY_PIN_ROOT"]
fn capture_parity_tree() {
    let root = PathBuf::from(std::env::var("JOPE_PARITY_PIN_ROOT").unwrap())
        .canonicalize()
        .unwrap();
    let tree = parity::scan_tree(&root).unwrap();
    let base = text("parity/capability-matrix.md");
    let mut heads = BTreeSet::new();
    let mut matrices = vec![base.clone()];
    matrices.extend(
        variants("parity/variants.json", &base)
            .into_iter()
            .map(|(_, m)| m),
    );
    for matrix in &matrices {
        heads.extend(parity::target_heads(matrix));
    }
    let strings =
        |items: &[String]| Value::List(items.iter().map(|i| Value::from(i.as_str())).collect());
    let mut exists = Dict::new();
    for head in heads {
        let present = root.join(&head).exists();
        exists.insert(head, Value::Bool(present));
    }
    let doc: Dict = [
        ("clap", strings(&tree.clap)),
        (
            "bins",
            Value::List(
                tree.bins
                    .iter()
                    .map(|(a, b, c, d)| strings(&[a.clone(), b.clone(), c.clone(), d.clone()]))
                    .collect(),
            ),
        ),
        ("tools", strings(&tree.tools)),
        (
            "jope_py",
            tree.jope_py.as_deref().map_or(Value::Null, strings),
        ),
        ("units", strings(&tree.units)),
        ("targets_exist", Value::Dict(exists)),
    ]
    .into_iter()
    .collect();
    fs::write(
        fixture().join("parity/tree.json"),
        pyjson::dumps(&Value::Dict(doc), false, Some(1)) + "\n",
    )
    .unwrap();
}

// ------------------------------------------------------------------------------------------------
// Malformed inputs (see the module documentation).

/// Malformed inputs the Python accepted (exit status 0) and the port refuses, by fixture and case.
/// Being stricter than the Python is allowed; being more permissive never is.
const RUST_STRICTER: &[(&str, &str, &str)] = &[
    (
        "coverage/negative-validate.json",
        "scope-bigint",
        "the port's JSON reader refuses integers beyond 64 bits",
    ),
    (
        "coverage/negative-discover.json",
        "feature-deps-int",
        "a feature dependency that is not a string is refused rather than sorted",
    ),
    (
        "coverage/negative-discover.json",
        "required-int-member",
        "a required feature that is not a string is refused rather than sorted",
    ),
    (
        "parity/negative.json",
        "package-name-int",
        "a package name that is not a string is refused rather than sorted and printed",
    ),
    (
        "parity/negative.json",
        "target-name-int",
        "a target name that is not a string is refused rather than sorted and printed",
    ),
    (
        "readme/negative.json",
        "valid-rankable-float",
        "`rankable_rows` must be an integer; the Python printed `format(8718.0, ',')`",
    ),
    // Round four: the strict number grammar (`py::decimal`, `py::float`).
    (
        "readme/negative-r2.json",
        "csv-average-rank-nan",
        "`float()` of `nan` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r3.json",
        "csv-int-underscores-4300-digits",
        "`int()` of digits with `_` separators is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-nbsp",
        "U+00A0 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-em-space",
        "U+2003 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-ideographic-space",
        "U+3000 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-underscore",
        "`int()` of digits with a `_` separator is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-arabic-indic",
        "`int()` of Arabic-Indic digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "published-scored-fullwidth",
        "`int()` of fullwidth digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "samecut-rank-nbsp",
        "U+00A0 around an integer is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-nbsp",
        "U+00A0 around a float is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-em-space",
        "U+2003 around a float is outside the strict number grammar (ASCII whitespace only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-underscore",
        "`float()` of digits with a `_` separator is outside the strict number grammar",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-arabic-indic",
        "`float()` of Arabic-Indic digits is outside the strict number grammar (ASCII digits only)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-nan",
        "`float()` of `nan` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-inf",
        "`float()` of `inf` is outside the strict number grammar (no committed ladder writes it)",
    ),
    (
        "readme/negative-r4.json",
        "valid-average-minus-infinity",
        "`float()` of `-Infinity` is outside the strict number grammar (no committed ladder writes it)",
    ),
];

/// README cases where both refuse with an explicit message and the port's words differ (round
/// five): the Python counted the reports it globbed or found none, the port refuses the entry
/// itself.
const README_MESSAGE_DIFFERS: &[(&str, &str, &str)] = &[
    (
        "readme/negative-r5.json",
        "readme-r5-report-name-not-utf8",
        "the Python read the report and refused the count; the port refuses the undecodable name",
    ),
    (
        "readme/negative-r5.json",
        "readme-r5-reports-unlistable",
        "the Python's glob listed nothing (\"no reports\"); the port refuses to list the directory",
    ),
];

/// Cases where the Python refused an integer of more than 4300 digits and the port refuses the
/// text earlier, because it is outside the strict number grammar: the refusal is not required to
/// carry CPython's message about the limit.
const REFUSED_BEFORE_THE_LIMIT: &[(&str, &str, &str)] = &[(
    "readme/negative-r3.json",
    "csv-int-underscores-4301-digits",
    "`_` separators are outside the strict number grammar, which refuses the text before it counts digits",
)];

/// What the port did with one input.
#[derive(Debug)]
enum Rust {
    /// A report on stdout and whether the command succeeded (exit status 0 or 1).
    Report(String, bool),
    /// A refusal: the error the binary prints after `jope-records: `, exit status 1.
    Refused(String),
}

impl Rust {
    fn of(result: jope_records::Result<(String, bool)>) -> Self {
        match result {
            Ok((stdout, ok)) => Rust::Report(stdout, ok),
            Err(error) => Rust::Refused(error.to_string()),
        }
    }

    fn exit(&self) -> i64 {
        match self {
            Rust::Report(_, true) => 0,
            _ => 1,
        }
    }
}

/// The negative cases of a fixture, by name.
fn negative(rel: &str) -> Vec<(String, Value)> {
    let doc = json(rel);
    assert!(
        doc.get("source").and_then(Value::as_str).is_some()
            && doc.get("capture").and_then(Value::as_str).is_some(),
        "{rel}: names its capture run and command"
    );
    doc.get("cases")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|c| (s(c.get("name").unwrap()).to_owned(), c.clone()))
        .collect()
}

/// A case's input: its `ed` script applied to `base`, checked against the recorded SHA-256 of the
/// file the Python read (the base itself when the case has no script).
fn variant(case: &Value, base: &str) -> String {
    let Some(ed) = case.get("ed") else {
        return base.to_owned();
    };
    let text = apply_ed(base, s(ed));
    assert_eq!(
        py::sha256_hex(text.as_bytes()),
        s(case.get("sha256").unwrap()),
        "{}: the replayed input is not the one the Python read",
        case.get("name").or(case.get("path")).map_or("?", s)
    );
    text
}

/// Hold one case to the contract; `Err` names the first disagreement.
fn compare(case: &Value, rust: &Rust, stricter: bool) -> Result<(), String> {
    let exit = case.get("exit").and_then(Value::as_int).unwrap();
    if stricter {
        return match (exit, rust) {
            (0, Rust::Refused(message)) if !message.is_empty() => Ok(()),
            _ => Err(format!(
                "listed as stricter than the Python, but the Python exited {exit} and the port gave {rust:?}"
            )),
        };
    }
    if rust.exit() != exit {
        return Err(format!(
            "exit status: the Python {exit}, the port {} ({rust:?})",
            rust.exit()
        ));
    }
    let stdout = case.get("stdout").map(|v| s(v).to_owned());
    if let Some(raised) = case.get("raised") {
        // The Python crashed: the port refuses and prints nothing.
        return match rust {
            Rust::Refused(message) if !message.is_empty() => Ok(()),
            _ => Err(format!(
                "the Python raised {}; the port did not refuse: {rust:?}",
                raised.repr()
            )),
        };
    }
    let stderr = s(case.get("stderr").unwrap());
    match rust {
        Rust::Refused(message) => {
            if stderr.is_empty() || stdout.as_deref().is_some_and(|out| !out.is_empty()) {
                return Err(format!(
                    "the port refused ({message:?}) where the Python printed stdout {stdout:?} and stderr {stderr:?}"
                ));
            }
            if format!("{message}\n") != stderr {
                return Err(format!(
                    "refusal message: the Python {stderr:?}, the port {message:?}"
                ));
            }
            Ok(())
        }
        Rust::Report(out, _) => {
            if !stderr.is_empty() {
                return Err(format!(
                    "the Python refused with {stderr:?}; the port reported {out:?}"
                ));
            }
            match (&stdout, case.get("stdout_sha256")) {
                (Some(want), _) if out != want => {
                    Err(format!("stdout: the Python {want:?}, the port {out:?}"))
                }
                (None, Some(sha)) if py::sha256_hex(out.as_bytes()) != s(sha) => Err(format!(
                    "stdout SHA-256: the Python {}, the port {} ({} bytes)",
                    s(sha),
                    py::sha256_hex(out.as_bytes()),
                    out.len()
                )),
                (None, None) => Err("the case records no stdout".to_owned()),
                _ => Ok(()),
            }
        }
    }
}

/// With `JOPE_NEGATIVE_REPORT` naming a file, every replayed case appends one tab-separated line
/// to it: the fixture, the case, the Python's exit status and message (its exception line, the
/// first line it printed on stderr, or the first line of its report) and the port's (its refusal,
/// or the first line of its report). The custody runs keep this table; the test never reads it.
fn record(rel: &str, name: &str, case: &Value, rust: &Rust) {
    let Some(path) = std::env::var_os("JOPE_NEGATIVE_REPORT") else {
        return;
    };
    let first = |text: &str| -> String {
        text.lines()
            .find(|l| !l.is_empty())
            .unwrap_or("")
            .chars()
            .take(160)
            .collect()
    };
    let python = match (case.get("raised"), case.get("stderr").map(s)) {
        (Some(raised), _) => format!("raised {}", s(raised)),
        (None, Some(stderr)) if !stderr.is_empty() => format!("refused {}", first(stderr)),
        _ => format!(
            "report {}",
            case.get("stdout")
                .map_or("(stdout by SHA-256)".to_owned(), |o| first(s(o)))
        ),
    };
    let port = match rust {
        Rust::Refused(message) => format!("refused {}", first(message)),
        Rust::Report(out, _) => format!("report {}", first(out)),
    };
    let line = format!(
        "{rel}\t{name}\t{}\t{python}\t{}\t{port}\n",
        case.get("exit").and_then(Value::as_int).unwrap(),
        rust.exit()
    );
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    file.write_all(line.as_bytes()).unwrap();
}

/// Replay every case of `rel` through `run` (the case and its input text) and require the
/// contract of each; the number of cases is pinned so a fixture cannot shrink unnoticed.
fn replay(rel: &str, base: &str, count: usize, run: &dyn Fn(&Value, &str) -> Rust) {
    let cases = negative(rel);
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let stricter: BTreeSet<&str> = RUST_STRICTER
        .iter()
        .filter(|(fixture, _, _)| *fixture == rel)
        .map(|(_, name, _)| *name)
        .collect();
    let mut failures = vec![];
    for (name, case) in &cases {
        let rust = run(case, &variant(case, base));
        record(rel, name, case, &rust);
        if let Err(why) = compare(case, &rust, stricter.contains(name.as_str())) {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    for name in &stricter {
        assert!(
            cases.iter().any(|(n, _)| n == name),
            "{rel}: RUST_STRICTER names {name}, which is not a case"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn discovered_items() -> Dict {
    json("coverage/discovered.json")
        .get("items")
        .and_then(Value::as_dict)
        .unwrap()
        .clone()
}

#[test]
fn coverage_validate_refuses_every_map_the_python_refused() {
    let discovered = discovered_items();
    replay(
        "coverage/negative-validate.json",
        &text("coverage/mapping.json"),
        55,
        &|_, mapping| {
            let parsed = pyjson::parse_str(mapping)
                .map_err(|m| jope_records::Error::Invalid(format!("coverage map: {m}")));
            Rust::of(parsed.and_then(|map| coverage::validate_text(&map, &discovered)))
        },
    );
}

#[test]
fn coverage_discover_refuses_every_metadata_the_python_refused() {
    let files = text("coverage/files.txt");
    let workflows = json("coverage/workflows.json");
    let read = |name: &str| -> jope_records::Result<String> {
        Ok(s(workflows.get(name).unwrap()).to_owned())
    };
    replay(
        "coverage/negative-discover.json",
        &text("coverage/metadata.json"),
        59,
        &|_, metadata| {
            let result = pyjson::parse_str(metadata)
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| {
                    coverage::discover_from(&meta, files.lines(), Path::new("@ROOT@"), &read)
                });
            Rust::of(result.map(|items| (coverage::discover_document(items), true)))
        },
    );
}

#[test]
fn rust_scope_check_refuses_every_metadata_the_python_refused() {
    let scope = json("scope/scope.json");
    // The base metadata is the recorded workspace, as `cargo metadata` lists it.
    let names: Vec<String> = json("scope/metadata.json")
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p.get("name").unwrap()).to_owned())
        .collect();
    let recorded: Vec<String> = scope
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p).to_owned())
        .collect();
    assert_eq!(
        names, recorded,
        "scope/metadata.json lists scope.json's packages"
    );
    replay(
        "scope/negative.json",
        &text("scope/metadata.json"),
        17,
        &|_, metadata| {
            let result = pyjson::parse_str(metadata)
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| scope_check::workspace_packages(&meta))
                .and_then(|workspace| {
                    scope_check::evaluate(
                        &workspace,
                        s(scope.get("product_stdout").unwrap()),
                        s(scope.get("deferred_stdout").unwrap()),
                        s(scope.get("script").unwrap()),
                    )
                });
            Rust::of(result)
        },
    );
}

#[test]
fn parity_check_refuses_every_metadata_the_python_refused() {
    let doc = json("parity/negative.json");
    let present: BTreeSet<String> = doc
        .get("exists")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|v| s(v).to_owned())
        .collect();
    let matrix = text("parity/negative-matrix.md");
    // The metadata's `@ROOT@` stands for the checked tree; the binaries' paths are made relative
    // to it lexically, so any canonical directory will do.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    replay(
        "parity/negative.json",
        &text("parity/negative-metadata.json"),
        32,
        &|_, metadata| {
            let result = pyjson::parse_str(&metadata.replace("@ROOT@", &root_text))
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| parity::bins_from_metadata(&meta, &root))
                .and_then(|bins| {
                    let tree = parity::Tree {
                        clap: vec![],
                        bins,
                        tools: vec![],
                        jope_py: None,
                        units: vec![],
                    };
                    parity::evaluate(&matrix, &tree, None, &|head: &str| present.contains(head))
                });
            Rust::of(result)
        },
    );
}

#[test]
fn readme_benchmarks_refuses_every_evidence_the_python_refused() {
    replay_readme("readme/negative.json", 85);
}

/// Round two: NaN and infinity formatting, Python 3.12's `sorted`, `max` and `sum` over the
/// same-cut seconds (a NaN among them, compensated float sums), the tie between NaN RMSEs, the
/// short-circuit of `a == b == c`, `repr` of non-printable characters and `$` before a final
/// newline.
#[test]
fn readme_benchmarks_matches_round_two() {
    replay_readme("readme/negative-r2.json", 18);
}

/// Round three: an integer of more than 4300 digits (either sign) in an evidence file, which
/// Python 3.12's `json.loads` refused to convert even in a field nothing reads; 19 digits render.
/// Its addendum: `int()` of a CSV cell of 4300 digits renders and of 4301 refuses, leading zeros
/// counted, whether written plain, with `+` or between spaces (`0_` separators, which Python also
/// read, are refused by the strict number grammar since round four).
#[test]
fn readme_benchmarks_matches_round_three() {
    replay_readme("readme/negative-r3.json", 12);
}

/// Round four (`readme/negative-r4.json`, review 5334300806 of PR #117): `int()` and `float()` of
/// README CSV cells, which the port reads with the strict number grammar. The reviewer's two
/// cells (U+001F before a published `datasets_scored` and before a benchmark-valid
/// `average_rank`), which Python refused although `str.strip()` removes U+001F, are refused;
/// so is every other input Python refused. ASCII whitespace, a sign, leading zeros and exponent
/// forms render as Python rendered them; Unicode whitespace, underscores, other Unicode digits,
/// `nan` and the infinities, which Python read, are refused (`RUST_STRICTER`).
/// Round five (`readme/negative-r5.json`): `sorted(reports.glob("*.bench-report.json"))` lists
/// entries of every kind and names that are not UTF-8 (one such extra report made the Python
/// refuse the count, while the port had skipped it and rendered); a directory or dangling link so
/// named, a report or directory that cannot be read, and an input under a directory that may not
/// be searched all refuse.
#[test]
fn readme_benchmarks_matches_round_five() {
    replay_readme("readme/negative-r5.json", 9);
}

#[test]
fn readme_benchmarks_matches_round_four() {
    replay_readme("readme/negative-r4.json", 37);
}

/// Round eight (`readme/negative-r8.json`, review 5337854418 of PR #117): Python's integers are
/// exact. The per-method correction totals (`get(m, 0) + applied[id]`) pass 64 bits and return
/// (the reviewer's `[2^63-1, 1, -(2^63-1), 45]`, which Python refused as 46 and the port had
/// accepted as 45), leave it below `-2^63`, are printed beyond it in a refusal, and become a float
/// only when a float operand appears; `before - after` at the extremes of `i64` (which a release
/// build had wrapped); and the same-cut seconds as CPython 3.12 sums, divides, sorts and compares
/// them: the C `long` fast path, exact integers and plain float addition after it overflows,
/// Neumaier compensation otherwise (integers added plainly), integer true division rounded once,
/// and `sorted`, `min` and `max` comparing an int with a float exactly.
#[test]
fn readme_benchmarks_matches_round_eight() {
    replay_readme("readme/negative-r8.json", 23);
}

fn replay_readme(rel: &str, count: usize) {
    let doc = json(rel);
    let stale = text("readme/README.stale.md");
    let cases = negative(rel);
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let stricter: BTreeSet<&str> = RUST_STRICTER
        .iter()
        .filter(|(fixture, _, _)| *fixture == rel)
        .map(|(_, name, _)| *name)
        .collect();
    assert_eq!(s(doc.get("root").unwrap()), "readme/root");
    let mut failures = vec![];
    for (name, case) in &cases {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        copy_tree(&fixture().join("readme/root"), &root);
        for edit in case.get("edits").and_then(Value::as_list).unwrap() {
            let path = s(edit.get("path").unwrap());
            let base = text(&format!("readme/root/{path}"));
            fs::write(root.join(path), variant(edit, &base)).unwrap();
        }
        // Copies (round five: also to a name that is not UTF-8), directories and symlinks.
        tree_extras(case, &root);
        fs::write(root.join("README.md"), &stale).unwrap();
        let effective = apply_modes(case, &root, false);
        let rust = Rust::of(readme::run(&root, true));
        apply_modes(case, &root, true);
        if !effective {
            eprintln!("{rel} {name}: skipped, this user may search a directory of that mode");
            continue;
        }
        record(rel, name, case, &rust);
        let is_stricter = stricter.contains(name.as_str());
        let differs = README_MESSAGE_DIFFERS
            .iter()
            .any(|(fixture, n, _)| *fixture == rel && n == name);
        let mut outcome = if differs {
            // Both refuse, in different words: the Python's explicit refusal (nothing on stdout)
            // and the port's.
            let python_refused = case.get("exit").and_then(Value::as_int) == Some(1)
                && case.get("raised").is_none()
                && case.get("stdout").is_some_and(|o| s(o).is_empty())
                && case.get("stderr").is_some_and(|e| !s(e).is_empty());
            match &rust {
                Rust::Refused(message) if python_refused && !message.is_empty() => Ok(()),
                _ => Err(format!(
                    "listed as refused in other words; the Python {:?}, the port {rust:?}",
                    case.get("stderr")
                )),
            }
        } else {
            compare(case, &rust, is_stricter)
        };
        if outcome.is_ok() {
            // What `--write` left in README.md: the Python's bytes where it rendered, the stale
            // README untouched where it (or, for a stricter case, the port) refused.
            let written = fs::read_to_string(root.join("README.md")).unwrap();
            let want = match case.get("readme_sha256") {
                Some(sha) if !is_stricter => s(sha).to_owned(),
                _ => py::sha256_hex(stale.as_bytes()),
            };
            if py::sha256_hex(written.as_bytes()) != want {
                outcome = Err("README.md differs from what the Python left".to_owned());
            }
            // Python 3.12's limit on integer conversion is refused in CPython's words, unless the
            // strict number grammar refused the text before it counted digits.
            let limit = case
                .get("raised")
                .and_then(|raised| s(raised).strip_prefix("ValueError: "))
                .filter(|message| message.starts_with("Exceeds the limit ("));
            let earlier = REFUSED_BEFORE_THE_LIMIT
                .iter()
                .any(|(fixture, n, _)| *fixture == rel && n == name);
            if let (Some(limit), Rust::Refused(message), false) = (limit, &rust, earlier) {
                if !message.contains(limit) {
                    outcome = Err(format!(
                        "refusal message: CPython's {limit:?}, the port {message:?}"
                    ));
                }
            }
        }
        if let Err(why) = outcome {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    for name in &stricter {
        assert!(
            cases.iter().any(|(n, _)| n == name),
            "RUST_STRICTER names {name}"
        );
    }
    for (fixture, name, _) in REFUSED_BEFORE_THE_LIMIT {
        if *fixture == rel {
            let case = cases.iter().find(|(n, _)| n == name).map(|(_, c)| c);
            assert!(
                case.is_some_and(|c| c.get("raised").is_some()),
                "REFUSED_BEFORE_THE_LIMIT names {name}, which is not a case the Python crashed on"
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ------------------------------------------------------------------------------------------------
// Round two: whole-process cases (see `process.json`).

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
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

/// Write a `process.json` case: its tree under `root` (`@ROOT@` in text replaced), and what the
/// `cargo` and `git` shims print under `dir`, encoded as the capture encoded it.
fn materialise(case: &Value, root: &Path, dir: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let root_text = root.to_str().unwrap();
    // Round six: `@LOSSY_REAL_ROOT@` is the real root behind the `root` link, spelled with U+FFFD
    // for each byte that is not UTF-8 (as a lossy conversion spelled it).
    let lossy = case.get("root_link").map(|link| {
        let parent = root.parent().unwrap().to_str().unwrap();
        let name = unescape(s(link));
        format!("{parent}/{}", String::from_utf8_lossy(name.as_bytes()))
    });
    for file in case.get("files").and_then(Value::as_list).unwrap() {
        let path = match file.get("path_hex") {
            Some(hex) => root.join(std::ffi::OsStr::from_bytes(&unhex(s(hex)))),
            None => root.join(s(file.get("path").unwrap())),
        };
        let bytes = match file.get("hex") {
            Some(hex) => unhex(s(hex)),
            None => {
                let text = s(file.get("text").unwrap()).replace("@ROOT@", root_text);
                match &lossy {
                    Some(lossy) => text.replace("@LOSSY_REAL_ROOT@", lossy),
                    None => text,
                }
                .into_bytes()
            }
        };
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
    }
    // Round five: directories, symlinks (a target's `@ROOT@` replaced) and copies of files.
    tree_extras(case, root);
    fs::create_dir_all(dir).unwrap();
    let cargo = case.get("cargo").unwrap();
    let text = s(cargo.get("stdout").unwrap()).replace("@ROOT@", root_text);
    let stdout = match s(cargo.get("encoding").unwrap()) {
        "utf-8" => text.into_bytes(),
        "utf-8-sig" => [&[0xef, 0xbb, 0xbf][..], text.as_bytes()].concat(),
        "utf-16" => [0xff, 0xfe]
            .into_iter()
            .chain(text.encode_utf16().flat_map(u16::to_le_bytes))
            .collect(),
        "hex" => unhex(s(cargo.get("stdout_hex").unwrap())),
        other => panic!("encoding {other}"),
    };
    fs::write(dir.join("cargo.stdout"), stdout).unwrap();
    fs::write(
        dir.join("cargo.stderr"),
        unhex(s(cargo.get("stderr_hex").unwrap())),
    )
    .unwrap();
    fs::write(dir.join("cargo.exit"), s(cargo.get("exit").unwrap())).unwrap();
    let git = match (case.get("git_hex"), case.get("git")) {
        (Some(hex), _) => unhex(s(hex)),
        (None, Some(list)) => list
            .as_list()
            .unwrap()
            .iter()
            .flat_map(|p| format!("{}\0", s(p)).into_bytes())
            .collect(),
        (None, None) => vec![],
    };
    fs::write(dir.join("git.stdout"), git).unwrap();
}

/// A case's `dirs` (created), `symlinks` (`path` -> `target`, `@ROOT@` in the target replaced) and
/// `copies` (`from` -> `to`, or `to_hex` for a name that is not UTF-8), all relative to `root`.
fn tree_extras(case: &Value, root: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let list = |key: &str| {
        case.get(key)
            .and_then(Value::as_list)
            .cloned()
            .unwrap_or_default()
    };
    for dir in list("dirs") {
        fs::create_dir_all(root.join(s(&dir))).unwrap();
    }
    for link in list("symlinks") {
        let path = root.join(s(link.get("path").unwrap()));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let target = s(link.get("target").unwrap()).replace("@ROOT@", root.to_str().unwrap());
        std::os::unix::fs::symlink(target, path).unwrap();
    }
    for copy in list("copies") {
        let to = match (copy.get("to_hex"), copy.get("to")) {
            (Some(hex), _) => root.join(std::ffi::OsStr::from_bytes(&unhex(s(hex)))),
            (None, Some(to)) => root.join(s(to)),
            _ => panic!("a copy without a destination"),
        };
        fs::copy(root.join(s(copy.get("from").unwrap())), to).unwrap();
    }
}

/// A case's `modes` (`path` -> octal mode) applied under `root`, or with `restore` set back to 755.
/// Returns whether each mode took effect for this user: a privileged user may search and list a
/// directory whatever its mode, so a case that relies on a refused search cannot be reproduced.
fn apply_modes(case: &Value, root: &Path, restore: bool) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let mut effective = true;
    for entry in case
        .get("modes")
        .and_then(Value::as_list)
        .cloned()
        .unwrap_or_default()
    {
        let path = root.join(s(entry.get("path").unwrap()));
        let mode = if restore {
            0o755
        } else {
            u32::from_str_radix(s(entry.get("mode").unwrap()), 8).unwrap()
        };
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        if !restore && mode & 0o500 != 0o500 && path.is_dir() {
            // A mode without read and search: would this user still list the directory, or reach
            // an entry of it?
            effective &= fs::read_dir(&path).is_err()
                || fs::read_dir(&path)
                    .unwrap()
                    .flatten()
                    .any(|e| fs::symlink_metadata(e.path()).is_err());
        }
    }
    effective
}

/// The `cargo` and `git` the capture put first on PATH.
fn shims(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(dir).unwrap();
    for (name, text) in [
        (
            "cargo",
            "#!/bin/bash\ncat \"$CASE_DIR/cargo.stderr\" >&2\ncat \"$CASE_DIR/cargo.stdout\"\ne=$(cat \"$CASE_DIR/cargo.exit\")\n[ \"$e\" = kill ] && kill -9 $$\nexit \"$e\"\n",
        ),
        ("git", "#!/bin/bash\ncat \"$CASE_DIR/git.stdout\"\n"),
    ] {
        fs::write(dir.join(name), text).unwrap();
        fs::set_permissions(dir.join(name), fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// Process cases the port refuses although the Python ran to completion (exit status 0): round
/// four's strict number grammar reads a number in ASCII digits only, where Python's `int()` read
/// any Unicode decimal digits; round five's walks refuse a directory they cannot list, which
/// pathlib's globs skipped.
const PROCESS_STRICTER: &[(&str, &str)] = &[
    (
        "parity-tally-arabic-equal",
        "a covering-test tally written in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-tally-4300-arabic-indic-digits",
        "a covering-test tally of 4300 Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-r5-tools-sub-unlistable",
        "a directory under python/src/jope/tools that cannot be listed (pathlib's rglob skipped it)",
    ),
    (
        "parity-r5-fallback-bin-unlistable",
        "a src/bin directory that cannot be listed (pathlib's glob found nothing in it)",
    ),
    (
        "parity-r5-fallback-crate-unlistable",
        "a crate directory that cannot be listed (pathlib's `**` walk skipped it)",
    ),
    (
        "parity-r6-root-not-utf8",
        "a root whose resolved name is not UTF-8 (the Python named every entry point relative to its surrogate-escaped root)",
    ),
];

/// Process cases where both exit 1 but the Python printed a failing report and the port refuses
/// instead:
///
/// - The report names an entry point whose path is not UTF-8: the Python kept it
///   surrogate-escaped, and printed its raw bytes only because the capture's `C.UTF-8` locale gives
///   stdout the `surrogateescape` handler (under another UTF-8 locale it crashed printing them).
///   The port's reports are text, so it refuses, naming the path.
/// - Round four: a number written in Unicode digits other than ASCII, which Python's `int()` read
///   and the strict number grammar refuses before the report is made.
const PROCESS_REFUSES_REPORT: &[(&str, &str)] = &[
    (
        "parity-tool-name-not-utf8-decoy",
        "a Python tool named with a byte that is not UTF-8",
    ),
    (
        "parity-unit-name-not-utf8",
        "a systemd unit named with a byte that is not UTF-8",
    ),
    (
        "parity-pending-wp-arabic-digit",
        "a `pending-WP<n>` status numbered in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-tally-arabic-differs",
        "a covering-test tally written in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
];

/// Hold one process case to the contract: where the Python crashed, the port exits 1 and prints
/// nothing on stdout; otherwise the exit status and stdout are identical, and stderr is identical
/// or (for a refusal) the Python's message after `jope-records: `.
fn judge_process(case: &Value, exit: i32, stdout: &[u8], stderr: &str) -> Result<(), String> {
    let name = s(case.get("name").unwrap());
    let want_exit = case.get("exit").and_then(Value::as_int).unwrap();
    if PROCESS_STRICTER.iter().any(|(n, _)| *n == name) {
        return if want_exit == 0 && exit == 1 && stdout.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "listed as stricter; the Python {want_exit}, the port {exit}"
            ))
        };
    }
    if PROCESS_REFUSES_REPORT.iter().any(|(n, _)| *n == name) {
        return if want_exit == 1
            && case.get("raised").is_none()
            && exit == 1
            && stdout.is_empty()
            && stderr.starts_with("jope-records: ")
        {
            Ok(())
        } else {
            Err(format!(
                "listed as a refusal of a failing report; the Python {want_exit}, the port {exit} {stderr:?}"
            ))
        };
    }
    if let Some(raised) = case.get("raised") {
        return if exit == 1 && stdout.is_empty() && !stderr.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the Python raised {}; the port exited {exit} with stdout {:?} and stderr {stderr:?}",
                s(raised),
                String::from_utf8_lossy(stdout)
            ))
        };
    }
    if i64::from(exit) != want_exit {
        return Err(format!(
            "exit status: the Python {want_exit}, the port {exit} (stderr {stderr:?})"
        ));
    }
    let want_stdout = match case.get("stdout_hex") {
        Some(hex) => unhex(s(hex)),
        None => s(case.get("stdout").unwrap()).as_bytes().to_vec(),
    };
    if stdout != want_stdout {
        return Err(format!(
            "stdout: the Python {:?}, the port {:?}",
            String::from_utf8_lossy(&want_stdout),
            String::from_utf8_lossy(stdout)
        ));
    }
    let want_stderr = s(case.get("stderr").unwrap());
    let refusal =
        exit == 1 && stdout.is_empty() && stderr == format!("jope-records: {want_stderr}");
    if stderr != want_stderr && !refusal {
        return Err(format!(
            "stderr: the Python {want_stderr:?}, the port {stderr:?}"
        ));
    }
    Ok(())
}

/// Round two of the differential review: the guards run end to end, each against a tree and a
/// `cargo` (and `git`) whose output the capture recorded, so what the port does with Cargo's
/// stdout and stderr (decoded as `text=True` decoded them), the files it reads (`read_text`, with
/// and without `errors="replace"`) and the order of its steps is compared with the Python's.
#[test]
fn process_cases_match_the_captured_python() {
    let doc = json("process.json");
    let cases = replay_process("process.json", &doc, 65);
    // Negative control: a changed exit status, stdout or message fails the judgement.
    let base = cases
        .iter()
        .find(|c| c.get("name") == Some(&Value::from("parity-base")))
        .unwrap();
    let base_stdout = s(base.get("stdout").unwrap()).as_bytes();
    assert_eq!(judge_process(base, 0, base_stdout, ""), Ok(()));
    assert!(judge_process(base, 1, base_stdout, "").is_err());
    assert!(judge_process(base, 0, b"x", "").is_err());
    let crash = cases.iter().find(|c| c.get("raised").is_some()).unwrap();
    assert!(judge_process(crash, 0, b"", "").is_err());
    assert!(judge_process(crash, 1, b"a report", "x").is_err());
}

/// Round three, integers (`process-r3-digits.json`, review 5333747872 of PR #117): Python 3.12
/// refused to convert an integer of more than 4300 decimal digits. `json.loads` of Cargo's
/// metadata and of the coverage map, and `int()` of the capability matrix's covering-test tally
/// and of a `pending-WP<n>` status (leading zeros and Unicode digits count), each crashed on 4301
/// digits and read 4300.
#[test]
fn process_cases_match_round_three_digits() {
    replay_process(
        "process-r3-digits.json",
        &json("process-r3-digits.json"),
        15,
    );
}

/// Round three, paths (`process-r3-paths.json`): POSIX's distinct `//` root where the guards
/// compared paths. `Path(p).relative_to(ROOT)` in coverage discovery and in the parity check's
/// `src/bin` fallback refused a path spelled `//...` (and one outside the tree); `///`, inner `//`
/// and `.` spellings normalise; `Path(name).parent` of the listed files is compared normalised; and
/// `os.path.relpath`, which splits on separators, has no distinct `//` root at all.
#[test]
fn process_cases_match_round_three_paths() {
    replay_process("process-r3-paths.json", &json("process-r3-paths.json"), 15);
}

/// Round five (`process-r5-sweep.json`): `parity-check` where Python 3.12's pathlib raised
/// (`main_rs.exists()`, `cli.exists()`, `is_dir()` and `is_file()` in the tool and unit walks, on
/// a directory that may not be listed or searched) and the `src/bin` fallback's globs, which list
/// entries of every kind by name. The port read a permission error as "absent" and skipped what
/// the globs listed; each case now refuses or reports as the Python did, and a directory that
/// cannot be listed refuses (`PROCESS_STRICTER`).
#[test]
fn process_cases_match_round_five() {
    replay_process("process-r5-sweep.json", &json("process-r5-sweep.json"), 28);
}

/// Round six (`process-r6.json`): the guards on a root that is a link to a directory whose name is
/// not UTF-8. The parity check compared entry points with the root converted lossily (U+FFFD for
/// the byte), so a `[[bin]]` path spelled with U+FFFD was found under the root and the check passed
/// where the Python's `relative_to` raised; it now refuses such a root before reading anything
/// (`PROCESS_STRICTER`: the Python ran the check). Coverage discovery refuses as the Python raised;
/// the scope check converts nothing and matches the Python's report.
#[test]
fn process_cases_match_round_six() {
    replay_process("process-r6.json", &json("process-r6.json"), 4);
}

/// Replay every case of a process fixture (the layout of `process.json`) through the binary and
/// hold each to `judge_process`; the cases, for the caller's controls.
fn replay_process<'a>(rel: &str, doc: &'a Value, count: usize) -> &'a Vec<Value> {
    assert!(
        doc.get("source").and_then(Value::as_str).is_some()
            && doc.get("capture").and_then(Value::as_str).is_some(),
        "{rel} names its capture run and command"
    );
    let cases = doc.get("cases").and_then(Value::as_list).unwrap();
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let temp = tempfile::tempdir().unwrap();
    let shim_dir = temp.path().join("shims");
    shims(&shim_dir);
    let path = format!(
        "{}:{}",
        shim_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    for case in cases {
        let name = s(case.get("name").unwrap());
        let base = temp.path().join(name);
        let dir = base.join("shim");
        fs::create_dir_all(&base).unwrap();
        let base = base.canonicalize().unwrap();
        // Round six: `root_link` names the real root (`@Xhh@` for a byte that is not UTF-8), which
        // the case reaches through the link `root`; `@ROOT@` is the link.
        let root = match case.get("root_link") {
            Some(link) => {
                let real = unescape(s(link));
                fs::create_dir(base.join(&real)).unwrap();
                std::os::unix::fs::symlink(&real, base.join("root")).unwrap();
                base.join("root")
            }
            None => {
                fs::create_dir(base.join("root")).unwrap();
                base.join("root")
            }
        };
        materialise(case, &root, &dir);
        let root_text = root.to_str().unwrap();
        let args: Vec<String> = case
            .get("rust")
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .map(|a| s(a).replace("@ROOT@", root_text))
            .collect();
        // Round five: the modes the case ran under, set back after the run.
        let effective = apply_modes(case, &root, false);
        let run = std::process::Command::new(env!("CARGO_BIN_EXE_jope-records"))
            .arg("--repo")
            .arg(&root)
            .args(&args)
            .current_dir(&root)
            .env("PATH", &path)
            .env("CASE_DIR", &dir)
            .output()
            .unwrap();
        apply_modes(case, &root, true);
        if !effective {
            eprintln!("{rel} {name}: skipped, this user may search a directory of that mode");
            continue;
        }
        let replace = |bytes: &[u8]| -> Vec<u8> {
            String::from_utf8(bytes.to_vec())
                .map(|t| t.replace(root_text, "@ROOT@").into_bytes())
                .unwrap_or_else(|_| bytes.to_vec())
        };
        let stdout = replace(&run.stdout);
        let stderr = String::from_utf8_lossy(&replace(&run.stderr)).into_owned();
        let exit = run.status.code().unwrap_or(-1);
        let port = if stdout.is_empty() {
            Rust::Refused(stderr.clone())
        } else {
            Rust::Report(String::from_utf8_lossy(&stdout).into_owned(), exit == 0)
        };
        record(rel, name, case, &port);
        if let Err(why) = judge_process(case, exit, &stdout, &stderr) {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    cases
}

/// The comparison is not vacuous: an altered expectation, a refusal where the Python reported,
/// a report where the Python crashed and a misplaced stricter entry each fail it.
#[test]
fn negative_replay_detects_a_changed_outcome() {
    let cases: BTreeMap<String, Value> = negative("coverage/negative-validate.json")
        .into_iter()
        .collect();
    let discovered = discovered_items();
    let base = text("coverage/mapping.json");
    let run = |name: &str| {
        let case = &cases[name];
        let map = pyjson::parse_str(&variant(case, &base)).unwrap();
        (
            case.clone(),
            Rust::of(coverage::validate_text(&map, &discovered)),
        )
    };
    // An explicit report: identical when replayed, a failure once one byte of it changes.
    let (mut case, rust) = run("lane-null-member");
    assert_eq!(compare(&case, &rust, false), Ok(()));
    let altered = s(case.get("stdout").unwrap()).replacen("None", "none", 1);
    case.as_dict_mut()
        .unwrap()
        .insert("stdout", Value::from(altered));
    assert!(compare(&case, &rust, false).is_err());
    // A crash: the port's refusal passes; a report in its place, or a flipped exit status, fails.
    let (case, rust) = run("lanes-null");
    assert!(matches!(rust, Rust::Refused(_)));
    assert_eq!(compare(&case, &rust, false), Ok(()));
    assert!(compare(&case, &Rust::Report(String::new(), false), false).is_err());
    let mut flipped = case.clone();
    flipped.as_dict_mut().unwrap().insert("exit", Value::Int(0));
    assert!(compare(&flipped, &rust, false).is_err());
    // A success: a refusal fails it, and so does listing it as stricter.
    let (case, rust) = run("lanes-dict-known");
    assert_eq!(compare(&case, &rust, false), Ok(()));
    assert!(compare(&case, &Rust::Refused("x".into()), false).is_err());
    assert!(compare(&case, &rust, true).is_err());
}
