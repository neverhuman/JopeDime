//! `jope-ci test-partition` (port of `deploy/ci/check_test_partition.py`): the fail-closed rust-test
//! shard cases formerly exercised through `deploy/ci/test-aggregate.py`, the receipt discovery and
//! slice checks, the static declaration checks on a synthetic checkout (with and without a base
//! revision) and on this repository, and the command's report, output and exit status.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use jope_ci::test_partition::{self, RUST_TEST_SHARDS};
use jope_records::pyjson;
use serde_json::{json, Value};

const BIN: &str = env!("CARGO_BIN_EXE_jope-ci");
// 2N selected listing cases (+1 ignored); slice M of N owns cases M-1 and M-1+N.
const CASES: usize = 2 * RUST_TEST_SHARDS + 1;

fn selected() -> BTreeSet<String> {
    (0..2 * RUST_TEST_SHARDS).map(|n| n.to_string()).collect()
}

fn mine_of(shard: usize) -> BTreeSet<String> {
    [shard - 1, shard - 1 + RUST_TEST_SHARDS]
        .iter()
        .map(ToString::to_string)
        .collect()
}

fn listing(selected: &BTreeSet<String>, partition: Option<&BTreeSet<String>>) -> String {
    let cases: serde_json::Map<String, Value> = (0..CASES)
        .map(|n| {
            let name = n.to_string();
            let case = if !selected.contains(&name) {
                json!({"ignored": true, "filter-match": {"status": "mismatch", "reason": "ignored"}})
            } else if partition.is_some_and(|p| !p.contains(&name)) {
                json!({"ignored": false, "filter-match": {"status": "mismatch", "reason": "partition"}})
            } else {
                json!({"ignored": false, "filter-match": {"status": "matches"}})
            };
            (name, case)
        })
        .collect();
    json!({"test-count": CASES, "rust-suites": {"fixture": {"status": "listed", "testcases": cases}}})
        .to_string()
}

fn junit<'a>(names: impl IntoIterator<Item = &'a String>) -> String {
    let cases: String = names
        .into_iter()
        .map(|n| format!("<testcase classname=\"fixture\" name=\"{n}\"/>"))
        .collect();
    format!("<testsuites><testsuite>{cases}</testsuite></testsuites>")
}

struct Run {
    _temp: tempfile::TempDir,
    root: PathBuf,
}

impl Run {
    /// One downloaded artifact directory per shard, as the aggregate job lays them out, plus a
    /// non-rust-test lane.
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("ci-download");
        for shard in 1..=RUST_TEST_SHARDS {
            let folder = root.join(format!("ci-1-1-rust-test-{shard}"));
            fs::create_dir_all(&folder).unwrap();
            let receipt = json!({"lane": "rust-test", "environment": {
                "JOPE_RUST_TEST_PARTITION": format!("slice:{shard}/{RUST_TEST_SHARDS}")}});
            fs::write(folder.join("receipt.json"), receipt.to_string()).unwrap();
            let mine = mine_of(shard);
            fs::write(
                folder.join("collection-unsharded.json"),
                listing(&selected(), None),
            )
            .unwrap();
            fs::write(
                folder.join("collection.json"),
                listing(&selected(), Some(&mine)),
            )
            .unwrap();
            fs::write(folder.join("junit.xml"), junit(&mine)).unwrap();
        }
        let other = root.join("ci-1-1-python");
        fs::create_dir_all(&other).unwrap();
        fs::write(other.join("receipt.json"), r#"{"lane": "python"}"#).unwrap();
        Run { _temp: temp, root }
    }

    fn shard(&self, shard: usize) -> PathBuf {
        self.root.join(format!("ci-1-1-rust-test-{shard}"))
    }

    fn write(&self, shard: usize, name: &str, text: &str) {
        fs::write(self.shard(shard).join(name), text).unwrap();
    }

    fn command(&self, args: &[&str]) -> (Output, Value) {
        let output = Command::new(BIN).args(args).output().unwrap();
        let report = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
        (output, report)
    }

    fn receipts(&self) -> (Output, Value) {
        let root = self.root.to_str().unwrap().to_owned();
        self.command(&["test-partition", "rust-shards", "--receipts", root.as_str()])
    }

    fn dirs(&self) -> (Output, Value) {
        let dirs: Vec<String> = (1..=RUST_TEST_SHARDS)
            .map(|shard| self.shard(shard).to_str().unwrap().to_owned())
            .collect();
        let mut args = vec!["test-partition", "rust-shards"];
        for dir in &dirs {
            args.extend(["--shard-dir", dir.as_str()]);
        }
        self.command(&args)
    }
}

/// The report's top-level keys in printed order (serde_json's map here may sort them).
fn keys(output: &Output) -> Vec<String> {
    let report = pyjson::parse_str(std::str::from_utf8(&output.stdout).unwrap()).unwrap();
    report
        .as_dict()
        .unwrap()
        .keys()
        .map(str::to_owned)
        .collect()
}

fn errors(report: &Value) -> Vec<String> {
    report["errors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_str().unwrap().to_owned())
        .collect()
}

fn assert_failed(output: &Output, report: &Value, expected: &str) {
    assert_eq!(Some(1), output.status.code(), "{report}");
    assert_eq!("failed", report["status"], "{report}");
    let errors = errors(report);
    assert!(
        errors.iter().any(|e| e.contains(expected)),
        "{expected:?} not in {errors:?}"
    );
}

#[test]
fn valid_shards_pass_with_the_python_report() {
    let run = Run::new();
    for (output, report) in [run.receipts(), run.dirs()] {
        assert_eq!(Some(0), output.status.code(), "{report}");
        assert_eq!("jope.test-partition/v1", report["schema"]);
        assert_eq!("rust-test", report["kind"]);
        assert_eq!("passed", report["status"]);
        assert_eq!(
            vec![
                "schema",
                "kind",
                "status",
                "unsharded",
                "shards",
                "union",
                "missing",
                "duplicated",
                "extra",
                "errors"
            ],
            keys(&output)
        );
        assert_eq!(CASES, report["unsharded"]["listed"]);
        assert_eq!(selected().len(), report["unsharded"]["selected"]);
        assert_eq!(selected().len(), report["union"]["selected"]);
        assert_eq!(
            report["unsharded"]["selected_sha256"],
            report["union"]["selected_sha256"]
        );
        let shard = &report["shards"]["rust-test-2"];
        assert_eq!(
            json!({"selected": 2, "selected_sha256": shard["selected_sha256"], "executed": 2, "failed": 0}),
            *shard
        );
    }
}

#[test]
fn digest_is_the_sorted_binary_test_lines() {
    let tests = [
        ("fixture".to_owned(), "1".to_owned()),
        ("fixture".to_owned(), "0".to_owned()),
    ];
    // hashlib.sha256(b"fixture\t0\nfixture\t1\n").hexdigest()
    assert_eq!(
        "0bfb4674fdff8e83b1576a10775516b450995d6e6291d35181b054035de68ea0",
        test_partition::digest(&tests)
    );
}

#[test]
fn shards_must_select_and_execute_the_unsharded_set_exactly_once() {
    let everything = selected();
    let without_zero: BTreeSet<String> = everything.iter().filter(|n| *n != "0").cloned().collect();
    let mut overlap = mine_of(2);
    overlap.insert("0".to_owned());
    let one: BTreeSet<String> = ["1".to_owned()].into();
    for (change, expected) in [
        ("overlap", "more than one shard"),
        ("gap", "in no shard"),
        ("unexecuted", "executed 1 tests, selected 2"),
        ("beyond-partition", "beyond the partition"),
        ("unsharded-differs", "differs between shards"),
        ("failed", "1 executed tests did not pass"),
        ("repeated", "repeated ['fixture 1']"),
        ("skipped-is-unexecuted", "unexecuted ['fixture 4']"),
        ("count", "test-count 9 differs from 7 listed tests"),
        ("malformed-junit", "not well-formed"),
        (
            "listed-differently",
            "lists 1 tests the unsharded listing does not",
        ),
    ] {
        let run = Run::new();
        match change {
            "overlap" => {
                run.write(2, "collection.json", &listing(&everything, Some(&overlap)));
                run.write(2, "junit.xml", &junit(&overlap));
            }
            "gap" => {
                run.write(2, "collection.json", &listing(&everything, Some(&one)));
                run.write(2, "junit.xml", &junit(&one));
            }
            "unexecuted" => run.write(2, "junit.xml", &junit(&one)),
            "beyond-partition" => {
                run.write(2, "collection.json", &listing(&without_zero, Some(&mine_of(2))));
            }
            "unsharded-differs" => {
                run.write(2, "collection-unsharded.json", &listing(&without_zero, None));
            }
            "failed" => run.write(
                2,
                "junit.xml",
                "<testsuites><testsuite><testcase classname=\"fixture\" name=\"1\"><failure/></testcase>\
                 <testcase classname=\"fixture\" name=\"4\"/></testsuite></testsuites>",
            ),
            "repeated" => {
                let names = ["1".to_owned(), "4".to_owned(), "1".to_owned()];
                run.write(2, "junit.xml", &junit(&names));
            }
            "skipped-is-unexecuted" => run.write(
                2,
                "junit.xml",
                "<testsuites><testsuite><testcase classname=\"fixture\" name=\"1\"/>\
                 <testcase classname=\"fixture\" name=\"4\"><skipped/></testcase></testsuite></testsuites>",
            ),
            "count" => {
                let text = listing(&everything, Some(&mine_of(2))).replace("\"test-count\":7", "\"test-count\":9");
                run.write(2, "collection.json", &text);
            }
            "malformed-junit" => run.write(2, "junit.xml", "<testsuites><testsuite>"),
            _ => {
                let mut value: Value = serde_json::from_str(&listing(&everything, Some(&mine_of(2)))).unwrap();
                value["test-count"] = json!(CASES + 1);
                value["rust-suites"]["fixture"]["testcases"]["extra"] =
                    json!({"ignored": false, "filter-match": {"status": "mismatch", "reason": "partition"}});
                run.write(2, "collection.json", &value.to_string());
            }
        }
        let (output, report) = run.receipts();
        assert_failed(&output, &report, expected);
        let (output, again) = run.dirs();
        assert_eq!(Some(1), output.status.code());
        assert_eq!(report, again, "{change}");
    }
}

#[test]
fn receipts_need_every_shard_once_with_its_own_slice() {
    for change in ["missing", "wrong-slice", "unrecorded-slice", "extra-slice"] {
        let run = Run::new();
        let receipt = run.shard(2).join("receipt.json");
        match change {
            "missing" => fs::remove_dir_all(run.shard(2)).unwrap(),
            "wrong-slice" => fs::write(&receipt, r#"{"lane": "rust-test", "environment": {"JOPE_RUST_TEST_PARTITION": "slice:4/4"}}"#).unwrap(),
            "unrecorded-slice" => fs::write(&receipt, r#"{"lane": "rust-test", "environment": {}}"#).unwrap(),
            _ => {
                let folder = run.root.join("ci-1-1-rust-test-4");
                fs::create_dir_all(&folder).unwrap();
                fs::write(folder.join("receipt.json"), r#"{"lane": "rust-test", "environment": {"JOPE_RUST_TEST_PARTITION": "slice:4/4"}}"#).unwrap();
            }
        }
        let expected = if change == "unrecorded-slice" {
            "rust-test receipt does not record its slice partition"
        } else {
            "rust-test shard receipts"
        };
        let (output, report) = run.receipts();
        assert_failed(&output, &report, expected);
        assert_eq!(
            json!({"schema": "jope.test-partition/v1", "kind": "rust-shards", "status": "failed",
                          "errors": report["errors"]}),
            report,
            "{change}"
        );
    }
    // Publication retries: the first directory of a unit is its evidence.
    let run = Run::new();
    let retry = run.root.join("ci-1-1-rust-test-2-pub-2");
    fs::create_dir_all(&retry).unwrap();
    fs::copy(
        run.shard(2).join("receipt.json"),
        retry.join("receipt.json"),
    )
    .unwrap();
    assert_eq!(Some(0), run.receipts().0.status.code());
    // The explicit form takes exactly one directory per slice.
    let dir = run.shard(1).to_str().unwrap().to_owned();
    let (output, report) =
        run.command(&["test-partition", "rust-shards", "--shard-dir", dir.as_str()]);
    assert_failed(
        &output,
        &report,
        "1 rust-test shard directories, expected 3",
    );
}

#[test]
fn collections_compare_listings_without_junit() {
    let run = Run::new();
    let unsharded = run.shard(1).join("collection-unsharded.json");
    let shards: Vec<String> = (1..=RUST_TEST_SHARDS)
        .map(|s| {
            run.shard(s)
                .join("collection.json")
                .to_str()
                .unwrap()
                .to_owned()
        })
        .collect();
    let mut args = vec![
        "test-partition",
        "collections",
        "--unsharded",
        unsharded.to_str().unwrap(),
    ];
    for shard in &shards {
        args.extend(["--shard", shard.as_str()]);
    }
    let (output, report) = run.command(&args);
    assert_eq!(Some(0), output.status.code(), "{report}");
    assert_eq!(
        json!({"selected": 2, "selected_sha256": report["shards"][&shards[0]]["selected_sha256"]}),
        report["shards"][&shards[0]]
    );
    let (output, report) = run.command(&args[..args.len() - 2]);
    assert_failed(
        &output,
        &report,
        "2 unsharded tests in no shard: ['fixture 2', 'fixture 5']",
    );
    assert_eq!(json!(["fixture 2", "fixture 5"]), report["missing"]);
}

#[test]
fn usage_errors_and_output_are_argparse_compatible() {
    let run = Run::new();
    let usage: [&[&str]; 3] = [
        &["test-partition", "rust-shards"],
        &["test-partition", "collections", "--unsharded", "a"],
        &["test-partition", "bogus"],
    ];
    for args in usage {
        let (output, _) = run.command(args);
        assert_eq!(Some(2), output.status.code(), "{args:?}");
    }
    let report_path = run.root.join("nested/report.json");
    let root = run.root.to_str().unwrap().to_owned();
    let path = report_path.to_str().unwrap().to_owned();
    let (output, report) = run.command(&[
        "test-partition",
        "rust-shards",
        "--receipts",
        root.as_str(),
        "--output",
        path.as_str(),
    ]);
    assert_eq!(Some(0), output.status.code());
    let written = fs::read(&report_path).unwrap();
    assert_eq!(output.stdout, written);
    assert!(written.starts_with(b"{\n  \"schema\": \"jope.test-partition/v1\",\n"));
    assert_eq!("passed", report["status"]);
    // Exclusive create: an existing report is never overwritten.
    let (output, _) = run.command(&[
        "test-partition",
        "rust-shards",
        "--receipts",
        root.as_str(),
        "--output",
        path.as_str(),
    ]);
    assert_eq!(Some(1), output.status.code());
    assert!(output.stdout.is_empty());
    assert_eq!(written, fs::read(&report_path).unwrap());
}

fn git(root: &Path, args: &[&str]) {
    let status = Command::new("git")
        .args([
            "-c",
            "user.name=fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args)
        .current_dir(root)
        .status()
        .unwrap();
    assert!(status.success(), "git {args:?}");
}

/// A synthetic checkout whose declarations agree: two groups over three families, three slices.
fn checkout(families: &[&str], groups: &str, scope: &str) -> tempfile::TempDir {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path();
    fs::create_dir_all(root.join("deploy/ci")).unwrap();
    fs::create_dir_all(root.join(".github/workflows")).unwrap();
    let lanes: serde_json::Map<String, Value> = families
        .iter()
        .map(|name| ((*name).to_owned(), json!({})))
        .chain([("hygiene".to_owned(), json!({}))])
        .collect();
    fs::write(
        root.join("deploy/ci/coverage.json"),
        json!({"lanes": lanes}).to_string(),
    )
    .unwrap();
    fs::write(
        root.join("deploy/ci/full-main.py"),
        format!(
            "GROUPS = {groups}\n\n\
             def partition_errors(declared, groups):\n\
             \x20   assigned = [name for names in groups.values() for name in names]\n\
             \x20   errors = [f'family {{name}} is in {{assigned.count(name)}} groups' for name in sorted(set(assigned)) if assigned.count(name) > 1]\n\
             \x20   errors += [f'family {{name}} is in no group' for name in sorted(set(declared) - set(assigned))]\n\
             \x20   errors += [f'group family {{name}} is not declared' for name in sorted(set(assigned) - set(declared))]\n\
             \x20   return errors\n"
        ),
    )
    .unwrap();
    fs::write(
        root.join(".github/workflows/full-main.yml"),
        "jobs:\n  cpu:\n    strategy:\n      matrix:\n        group: [cpu-1, cpu-2]\n    steps:\n\
         \x20     - run: python3 deploy/ci/full-main.py run --group \"$GROUP\" --output x\n\
         \x20     - run: python3 deploy/ci/full-main.py summarize --group \"$GROUP\" --output x\n",
    )
    .unwrap();
    fs::write(
        root.join(".github/workflows/ci.yml"),
        "jobs:\n  rust-test:\n    strategy:\n      matrix:\n        shard: [1, 2, 3]\n    env:\n\
         \x20     JOPE_RUST_TEST_PARTITION: slice:${{ matrix.shard }}/3\n",
    )
    .unwrap();
    fs::write(
        root.join("deploy/ci/rust-test-scope.sh"),
        format!("printf -- '-p %s ' {scope}\necho\n"),
    )
    .unwrap();
    temp
}

fn static_check(root: &Path, base: Option<&str>) -> (Output, Value) {
    let mut args = vec!["test-partition", "static", "--root", root.to_str().unwrap()];
    if let Some(base) = base {
        args.extend(["--base", base]);
    }
    let output = Command::new(BIN).args(&args).output().unwrap();
    let report = serde_json::from_slice(&output.stdout).unwrap_or(Value::Null);
    (output, report)
}

const FAMILIES: [&str; 3] = ["full-a", "full-b", "full-c"];
const GROUPS: &str = "{'cpu-1': ('full-a', 'full-c'), 'cpu-2': ('full-b',)}";

#[test]
fn static_declarations_of_a_consistent_checkout_pass() {
    let temp = checkout(&FAMILIES, GROUPS, "one two");
    for base in [None, Some("")] {
        let (output, report) = static_check(temp.path(), base);
        assert_eq!(Some(0), output.status.code(), "{report}");
        assert_eq!(
            json!({"schema": "jope.test-partition/v1", "kind": "static",
                   "full_main": {"declared": ["full-a", "full-b", "full-c"],
                                 "groups": {"cpu-1": ["full-a", "full-c"], "cpu-2": ["full-b"]}},
                   "rust_test": {"shards": 3, "scope": ["one", "two"]},
                   "status": "passed", "errors": []}),
            report
        );
    }
}

#[test]
fn static_refuses_every_inconsistent_declaration() {
    let cases: [(&str, &str, &[&str]); 7] = [
        ("{'cpu-1': ('full-a', 'full-c'), 'cpu-2': ('full-b', 'full-a')}", "", &["family full-a is in 2 groups"]),
        ("{'cpu-1': ('full-a',), 'cpu-2': ('full-b',)}", "", &["family full-c is in no group"]),
        ("{'cpu-1': ('full-a', 'full-c', 'full-d'), 'cpu-2': ('full-b',)}", "", &["group family full-d is not declared"]),
        ("{'cpu-2': ('full-b',), 'cpu-1': ('full-a', 'full-c')}", "", &["full-main.yml cpu matrix ['cpu-1, cpu-2'] does not run exactly the groups ['cpu-2', 'cpu-1']"]),
        (GROUPS, "summarize", &["full-main.yml does not summarize exactly its matrix group"]),
        (GROUPS, "shards", &["ci.yml rust-test matrix ['1, 2'] is not shards ['1', '2', '3']"]),
        (GROUPS, "slice", &["ci.yml rust-test does not partition by slice:<shard>/3"]),
    ];
    for (groups, edit, expected) in cases {
        let temp = checkout(&FAMILIES, groups, "one two");
        let root = temp.path();
        let replace = |file: &str, from: &str, to: &str| {
            let path = root.join(file);
            let text = fs::read_to_string(&path).unwrap();
            assert!(text.contains(from));
            fs::write(&path, text.replace(from, to)).unwrap();
        };
        match edit {
            "summarize" => replace(
                ".github/workflows/full-main.yml",
                "summarize --group",
                "summarize --groups",
            ),
            "shards" => replace(".github/workflows/ci.yml", "[1, 2, 3]", "[1, 2]"),
            "slice" => replace(".github/workflows/ci.yml", "}}/3", "}}/4"),
            _ => {}
        }
        let (output, report) = static_check(root, None);
        assert_eq!(Some(1), output.status.code(), "{report}");
        assert_eq!(*expected, errors(&report), "{groups} {edit}");
    }
    // A full-main.py that cannot load is a failed report, not a pass.
    let temp = checkout(&FAMILIES, "undefined_name", "one two");
    let (output, report) = static_check(temp.path(), None);
    assert_eq!(Some(1), output.status.code());
    assert_eq!("static", report["kind"]);
    assert!(
        errors(&report)[0].contains("returned non-zero exit status 1"),
        "{report}"
    );
}

#[test]
fn static_base_requires_nothing_declared_there_is_missing() {
    let temp = checkout(
        &["full-a", "full-b", "full-c", "full-gone"],
        "{'cpu-1': ('full-a', 'full-b', 'full-c', 'full-gone')}",
        "one two old",
    );
    let root = temp.path();
    git(root, &["init", "-q"]);
    git(root, &["add", "-A"]);
    git(root, &["commit", "-q", "-m", "base"]);
    let fresh = checkout(&FAMILIES, GROUPS, "one two");
    for file in [
        "deploy/ci/coverage.json",
        "deploy/ci/full-main.py",
        "deploy/ci/rust-test-scope.sh",
    ] {
        fs::copy(fresh.path().join(file), root.join(file)).unwrap();
    }
    let (output, report) = static_check(root, Some("HEAD"));
    assert_eq!(Some(1), output.status.code(), "{report}");
    assert_eq!(
        json!({"rev": "HEAD", "full_main_families": ["full-a", "full-b", "full-c", "full-gone"],
               "missing_families": ["full-gone"], "added_families": [],
               "rust_test_scope": ["one", "two", "old"], "missing_scope": ["old"]}),
        report["base"]
    );
    assert_eq!(
        vec![
            "family full-gone ran on HEAD but is in no group",
            "crate old was in the HEAD rust-test scope but is not now"
        ],
        errors(&report)
    );
    assert_eq!(
        vec![
            "schema",
            "kind",
            "full_main",
            "rust_test",
            "base",
            "status",
            "errors"
        ],
        keys(&output)
    );
    // An unknown revision is a failed report.
    let (output, report) = static_check(root, Some("no-such-revision"));
    assert_eq!(Some(1), output.status.code());
    assert!(
        errors(&report)[0].contains("'git', 'show', 'no-such-revision:deploy/ci/coverage.json'"),
        "{report}"
    );
}

#[test]
fn this_repository_declarations_pass() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (output, report) = static_check(&root, None);
    assert_eq!(Some(0), output.status.code(), "{report}");
    assert_eq!(RUST_TEST_SHARDS, report["rust_test"]["shards"]);
    assert!(report["rust_test"]["scope"]
        .as_array()
        .unwrap()
        .contains(&json!("jope-ci")));
}
