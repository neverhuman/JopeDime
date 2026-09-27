//! The four `jope-records` hygiene guards against the outputs of the Python they replaced.
//!
//! `tests/fixtures/python-parity/python-outputs.json` holds what `deploy/ci/coverage.py`,
//! `rust-test-scope-check.py`, `parity-census.py check` and `readme-benchmarks.py` (archived at
//! 8a55c4fef) printed in the differential run `py-p1-u3-diff-20260927T024147Z`, before they were
//! deleted. The inputs are committed beside them, taken from that run's tree (the U2 commit as
//! first written on `rr2/python-p1`, replayed on landing as 947b1a1aa with a different
//! `deploy/ci/coverage.json`, plus the run's working changes; recovered from an export of it).
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
            coverage::validate_text(&pyjson::parse_str(&mapping).unwrap(), &discovered);
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
