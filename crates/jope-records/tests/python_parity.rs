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
        let (stdout, ok) = parity::evaluate(&matrix, &tree, help, &exists);
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
];

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
                .map(|bins| {
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
    let doc = json("readme/negative.json");
    let stale = text("readme/README.stale.md");
    let cases = negative("readme/negative.json");
    assert_eq!(
        cases.len(),
        85,
        "readme/negative.json: every captured case is present"
    );
    let stricter: BTreeSet<&str> = RUST_STRICTER
        .iter()
        .filter(|(fixture, _, _)| *fixture == "readme/negative.json")
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
        for copy in case.get("copies").and_then(Value::as_list).unwrap() {
            fs::copy(
                root.join(s(copy.get("from").unwrap())),
                root.join(s(copy.get("to").unwrap())),
            )
            .unwrap();
        }
        fs::write(root.join("README.md"), &stale).unwrap();
        let rust = Rust::of(readme::run(&root, true));
        record("readme/negative.json", name, case, &rust);
        let is_stricter = stricter.contains(name.as_str());
        let mut outcome = compare(case, &rust, is_stricter);
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
        }
        if let Err(why) = outcome {
            failures.push(format!("readme/negative.json {name}: {why}"));
        }
    }
    for name in &stricter {
        assert!(
            cases.iter().any(|(n, _)| n == name),
            "RUST_STRICTER names {name}"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
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
