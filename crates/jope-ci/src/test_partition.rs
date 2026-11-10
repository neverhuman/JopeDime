//! Prove that split CI jobs run exactly the tests of the single job each one replaced (port of
//! `deploy/ci/check_test_partition.py`, issue #308).
//!
//! rust-test (ci.yml): every `slice:M/N` matrix leg lists the unsharded selection, which is the
//! pre-split rust-test command, as `collection-unsharded.json` next to its own partitioned
//! `collection.json`. The shards' selections must cover the unsharded selection exactly once,
//! differ from it only by the partition, and each shard's JUnit must execute exactly its own
//! selection.
//!
//! full-main-cpu (full-main.yml): `deploy/ci/full-main.py` `GROUPS` must cover every declared
//! `full-*` family exactly once (the families the single pre-split job ran), and the workflow must
//! run every group. Given a base revision, nothing the base declared may be missing. `full-main.py`
//! stays the one owner of `GROUPS` and `partition_errors`; [`static_check`] asks it for both.
//!
//! The report schema (`jope.test-partition/v1`), its key order, the error texts and the exit
//! status (1 unless `passed`) are the Python's. Every failure is a failed report, never a pass.

pub mod junit;

use std::collections::{BTreeMap, BTreeSet};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use jope_records::pyjson::{self, eq_or_identical, str_repr, Dict, Value};
use sha2::{Digest, Sha256};

pub const SCHEMA: &str = "jope.test-partition/v1";
/// The number of `slice:M/N` rust-test legs in ci.yml (`deploy/ci/aggregate.py` agrees).
pub const RUST_TEST_SHARDS: usize = 3;

/// `(binary-id, test)` of a nextest listing, or `(classname, name)` of a JUnit case.
pub type Test = (String, String);
/// `(filter status, mismatch reason, ignored)`; a missing field is `None` (`Value::Null`).
type Row = [Value; 3];

/// A nextest JSON listing in document order, with its rows by test.
pub struct Listing {
    order: Vec<Test>,
    rows: BTreeMap<Test, Row>,
}

fn key_error(key: &str) -> String {
    // `str(KeyError(key))`, the message the Python's report carried.
    str_repr(key)
}

fn read_text(path: &Path) -> Result<String, String> {
    std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))
}

fn read_json(path: &Path) -> Result<Value, String> {
    pyjson::parse_str(&read_text(path)?).map_err(|error| format!("{}: {error}", path.display()))
}

fn dict<'a>(value: &'a Value, what: &str) -> Result<&'a Dict, String> {
    value
        .as_dict()
        .ok_or_else(|| format!("{what} is not an object: {}", value.repr()))
}

/// `{(binary-id, test): (filter status, mismatch reason, ignored)}` of a nextest JSON listing.
pub fn listing(path: &Path) -> Result<Listing, String> {
    let data = read_json(path)?;
    let data = dict(&data, &path.display().to_string())?;
    let suites = data
        .get("rust-suites")
        .ok_or_else(|| key_error("rust-suites"))?;
    let mut order = vec![];
    let mut rows = BTreeMap::new();
    for (binary, suite) in dict(suites, "rust-suites")?.iter() {
        let Some(cases) = dict(suite, binary)?.get("testcases") else {
            continue;
        };
        for (name, case) in dict(cases, binary)?.iter() {
            let case = dict(case, name)?;
            let empty = Value::Dict(Dict::new());
            let filter = dict(case.get("filter-match").unwrap_or(&empty), name)?;
            let field = |d: &Dict, key: &str| d.get(key).cloned().unwrap_or(Value::Null);
            let test = (binary.to_owned(), name.to_owned());
            let row = [
                field(filter, "status"),
                field(filter, "reason"),
                field(case, "ignored"),
            ];
            if rows.insert(test.clone(), row).is_none() {
                order.push(test);
            }
        }
    }
    let count = data
        .get("test-count")
        .ok_or_else(|| key_error("test-count"))?;
    if *count != Value::from(rows.len()) {
        return Err(format!(
            "{}: test-count {} differs from {} listed tests",
            path.display(),
            count.py_str(),
            rows.len()
        ));
    }
    Ok(Listing { order, rows })
}

fn rows_equal(a: &Row, b: &Row) -> bool {
    a.iter().zip(b).all(|(x, y)| eq_or_identical(x, y))
}

fn listings_equal(a: &Listing, b: &Listing) -> bool {
    a.rows.len() == b.rows.len()
        && a.rows
            .iter()
            .all(|(test, row)| b.rows.get(test).is_some_and(|other| rows_equal(row, other)))
}

fn row_repr(row: &Row) -> String {
    format!("({}, {}, {})", row[0].repr(), row[1].repr(), row[2].repr())
}

fn is_str(value: &Value, text: &str) -> bool {
    value.as_str() == Some(text)
}

/// Executed (not skipped) JUnit cases as `[((classname, name), passed)]`.
pub fn executed(path: &Path) -> Result<Vec<(Test, bool)>, String> {
    let root =
        junit::parse(&read_text(path)?).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut cases = vec![];
    root.iter("testcase", &mut cases);
    let mut rows = vec![];
    for case in cases.into_iter().filter(|case| !case.has_child("skipped")) {
        let (Some(classname), Some(name)) = (case.attribute("classname"), case.attribute("name"))
        else {
            return Err(format!(
                "{}: an executed testcase lacks its classname or name",
                path.display()
            ));
        };
        let passed = !case.has_child("failure") && !case.has_child("error");
        rows.push(((classname.to_owned(), name.to_owned()), passed));
    }
    Ok(rows)
}

/// sha256 over the sorted `binary\ttest\n` lines.
pub fn digest<'a>(tests: impl IntoIterator<Item = &'a Test>) -> String {
    let mut sorted: Vec<&Test> = tests.into_iter().collect();
    sorted.sort();
    let mut hasher = Sha256::new();
    for (binary, name) in sorted {
        hasher.update(format!("{binary}\t{name}\n").as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn named(tests: &BTreeSet<Test>, limit: usize) -> Vec<String> {
    tests
        .iter()
        .take(limit)
        .map(|(binary, name)| format!("{binary} {name}"))
        .collect()
}

fn strings(items: impl IntoIterator<Item = impl Into<String>>) -> Value {
    Value::List(
        items
            .into_iter()
            .map(|item| Value::Str(item.into()))
            .collect(),
    )
}

/// Python's `repr` of a list of strings.
fn list_repr(items: &[String]) -> String {
    let quoted: Vec<String> = items.iter().map(|item| str_repr(item)).collect();
    format!("[{}]", quoted.join(", "))
}

fn selection(tests: &BTreeSet<Test>) -> Dict {
    let mut row = Dict::new();
    row.insert("selected", Value::from(tests.len()));
    row.insert("selected_sha256", Value::from(digest(tests)));
    row
}

/// One partitioned run: its name, its nextest listing and, if it ran, its JUnit report.
pub struct Shard {
    pub name: String,
    pub collection: PathBuf,
    pub junit: Option<PathBuf>,
}

/// unsharded: listing paths (one per shard run, all identical); shards: the partitioned runs.
pub fn compare(unsharded: &[PathBuf], shards: &[Shard]) -> Result<Dict, String> {
    let (first, others) = unsharded
        .split_first()
        .ok_or_else(|| "no unsharded listing".to_owned())?;
    let mut errors: Vec<String> = vec![];
    let old = listing(first)?;
    for other in others {
        if !listings_equal(&listing(other)?, &old) {
            errors.push(format!(
                "{}: unsharded listing differs between shards",
                other.display()
            ));
        }
    }
    let selected: BTreeSet<Test> = old
        .rows
        .iter()
        .filter(|(_, row)| is_str(&row[0], "matches"))
        .map(|(test, _)| test.clone())
        .collect();
    let mut union = BTreeSet::new();
    let mut seen: BTreeMap<Test, usize> = BTreeMap::new();
    let mut report = Dict::new();
    for shard in shards {
        let name = &shard.name;
        let current = listing(&shard.collection)?;
        let differing = current
            .rows
            .keys()
            .filter(|test| !old.rows.contains_key(*test))
            .count()
            + old
                .rows
                .keys()
                .filter(|test| !current.rows.contains_key(*test))
                .count();
        if differing != 0 {
            errors.push(format!(
                "{name}: lists {differing} tests the unsharded listing does not"
            ));
        }
        let mut mine = BTreeSet::new();
        for test in &current.order {
            let row = &current.rows[test];
            let Some(before) = old.rows.get(test) else {
                continue;
            };
            let was_selected = is_str(&before[0], "matches");
            if was_selected && is_str(&row[0], "matches") {
                mine.insert(test.clone());
            } else if !(was_selected && is_str(&row[0], "mismatch") && is_str(&row[1], "partition"))
                && !rows_equal(row, before)
            {
                errors.push(format!(
                    "{name}: {} {} selection differs beyond the partition: {} vs {}",
                    test.0,
                    test.1,
                    row_repr(row),
                    row_repr(before)
                ));
            }
        }
        for test in &mine {
            *seen.entry(test.clone()).or_default() += 1;
        }
        union.extend(mine.iter().cloned());
        let mut row = selection(&mine);
        if let Some(junit) = &shard.junit {
            let ran = executed(junit)?;
            let mut counts: BTreeMap<&Test, usize> = BTreeMap::new();
            for (test, _) in &ran {
                *counts.entry(test).or_default() += 1;
            }
            let ids: BTreeSet<Test> = counts.keys().map(|test| (*test).clone()).collect();
            let twice: BTreeSet<Test> = counts
                .iter()
                .filter(|(_, n)| **n > 1)
                .map(|(test, _)| (*test).clone())
                .collect();
            let failed = ran.iter().filter(|(_, passed)| !passed).count();
            row.insert("executed", Value::from(ran.len()));
            row.insert("failed", Value::from(failed));
            if ids != mine || !twice.is_empty() {
                let unexecuted: BTreeSet<Test> = mine.difference(&ids).cloned().collect();
                let unselected: BTreeSet<Test> = ids.difference(&mine).cloned().collect();
                errors.push(format!(
                    "{name}: executed {} tests, selected {}; unexecuted {}, unselected {}, repeated {}",
                    ids.len(),
                    mine.len(),
                    list_repr(&named(&unexecuted, 5)),
                    list_repr(&named(&unselected, 5)),
                    list_repr(&named(&twice, 5))
                ));
            }
            if failed != 0 {
                errors.push(format!("{name}: {failed} executed tests did not pass"));
            }
        }
        report.insert(name.clone(), Value::Dict(row));
    }
    let duplicated: BTreeSet<Test> = seen
        .into_iter()
        .filter(|(_, owners)| *owners > 1)
        .map(|(test, _)| test)
        .collect();
    let missing: BTreeSet<Test> = selected.difference(&union).cloned().collect();
    let extra: BTreeSet<Test> = union.difference(&selected).cloned().collect();
    if !missing.is_empty() {
        errors.push(format!(
            "{} unsharded tests in no shard: {}",
            missing.len(),
            list_repr(&named(&missing, 10))
        ));
    }
    if !duplicated.is_empty() {
        errors.push(format!(
            "{} tests in more than one shard: {}",
            duplicated.len(),
            list_repr(&named(&duplicated, 10))
        ));
    }
    if !extra.is_empty() {
        errors.push(format!(
            "{} shard tests outside the unsharded selection: {}",
            extra.len(),
            list_repr(&named(&extra, 10))
        ));
    }
    let mut unsharded_row = Dict::new();
    unsharded_row.insert("listed", Value::from(old.rows.len()));
    for (key, value) in selection(&selected).iter() {
        unsharded_row.insert(key, value.clone());
    }
    let mut out = Dict::new();
    out.insert("schema", Value::from(SCHEMA));
    out.insert("kind", Value::from("rust-test"));
    out.insert("status", Value::from(status(&errors)));
    out.insert("unsharded", Value::Dict(unsharded_row));
    out.insert("shards", Value::Dict(report));
    out.insert("union", Value::Dict(selection(&union)));
    out.insert("missing", strings(named(&missing, 50)));
    out.insert("duplicated", strings(named(&duplicated, 50)));
    out.insert("extra", strings(named(&extra, 50)));
    out.insert("errors", strings(errors));
    Ok(out)
}

fn status(errors: &[String]) -> &'static str {
    if errors.is_empty() {
        "passed"
    } else {
        "failed"
    }
}

/// The execution unit of a rust-test CI receipt: `rust-test-M` for its `slice:M/N` partition.
pub fn shard_unit(receipt: &Value) -> Result<String, String> {
    let partition = match receipt
        .get("environment")
        .and_then(|environment| environment.get("JOPE_RUST_TEST_PARTITION"))
    {
        Some(value) => value.py_str(),
        None => String::new(),
    };
    slice(&partition)
        .map(|(m, _)| format!("rust-test-{m}"))
        .ok_or_else(|| "rust-test receipt does not record its slice partition".to_owned())
}

/// `slice:([1-9][0-9]*)/([1-9][0-9]*)` in full.
fn slice(text: &str) -> Option<(&str, &str)> {
    let (m, n) = text.strip_prefix("slice:")?.split_once('/')?;
    let number = |s: &str| {
        s.starts_with(|c: char| ('1'..='9').contains(&c)) && s.bytes().all(|b| b.is_ascii_digit())
    };
    (number(m) && number(n)).then_some((m, n))
}

/// The evidence directory of each `rust-test-M` unit under RECEIPTS (one immediate directory per
/// downloaded artifact); publication retries share bytes (`aggregate.py`), so the first wins.
pub fn rust_shard_dirs(receipts: &Path) -> Result<Vec<(String, PathBuf)>, String> {
    let mut artifacts: Vec<PathBuf> = std::fs::read_dir(receipts)
        .and_then(|entries| {
            entries
                .map(|entry| entry.map(|e| e.path()))
                .collect::<std::io::Result<Vec<PathBuf>>>()
        })
        .map_err(|error| format!("{}: {error}", receipts.display()))?;
    artifacts.sort();
    let mut units: BTreeMap<String, PathBuf> = BTreeMap::new();
    for artifact in artifacts {
        let path = artifact.join("receipt.json");
        if artifact.is_dir() && !artifact.is_symlink() && path.is_file() && !path.is_symlink() {
            let receipt = read_json(&path)?;
            if receipt
                .get("lane")
                .is_some_and(|lane| is_str(lane, "rust-test"))
            {
                units.entry(shard_unit(&receipt)?).or_insert(artifact);
            }
        }
    }
    let expected: Vec<String> = (1..=RUST_TEST_SHARDS)
        .map(|shard| format!("rust-test-{shard}"))
        .collect();
    let found: Vec<String> = units.keys().cloned().collect();
    let mut wanted = expected.clone();
    wanted.sort();
    if found != wanted {
        return Err(format!(
            "rust-test shard receipts {}, expected {}",
            list_repr(&found),
            list_repr(&expected)
        ));
    }
    Ok(expected
        .into_iter()
        .map(|unit| {
            let folder = units.remove(&unit).unwrap_or_default();
            (unit, folder)
        })
        .collect())
}

/// Evidence directories in slice order, named `rust-test-1..N` (`aggregate.py` passes its own
/// verified receipts' directories this way).
pub fn named_shard_dirs(dirs: &[PathBuf]) -> Result<Vec<(String, PathBuf)>, String> {
    if dirs.len() != RUST_TEST_SHARDS {
        return Err(format!(
            "{} rust-test shard directories, expected {RUST_TEST_SHARDS}",
            dirs.len()
        ));
    }
    Ok(dirs
        .iter()
        .enumerate()
        .map(|(n, dir)| (format!("rust-test-{}", n + 1), dir.clone()))
        .collect())
}

/// Compare every rust-test shard's evidence directory: `(unit, directory)`.
pub fn verify_shards(dirs: &[(String, PathBuf)]) -> Result<Dict, String> {
    let unsharded: Vec<PathBuf> = dirs
        .iter()
        .map(|(_, folder)| folder.join("collection-unsharded.json"))
        .collect();
    let shards: Vec<Shard> = dirs
        .iter()
        .map(|(unit, folder)| Shard {
            name: unit.clone(),
            collection: folder.join("collection.json"),
            junit: Some(folder.join("junit.xml")),
        })
        .collect();
    compare(&unsharded, &shards)
}

/// The sorted `full-*` lane names a coverage mapping declares.
fn declared_families(text: &str, origin: &str) -> Result<Vec<String>, String> {
    let data = pyjson::parse_str(text).map_err(|error| format!("{origin}: {error}"))?;
    let lanes = data.get("lanes").ok_or_else(|| key_error("lanes"))?;
    let mut names: Vec<String> = dict(lanes, "lanes")?
        .keys()
        .filter(|name| name.starts_with("full-"))
        .map(str::to_owned)
        .collect();
    names.sort();
    Ok(names)
}

/// Resolve a new-push zero-SHA sentinel to the default branch comparison ref.
pub fn static_base(base: &str) -> &str {
    if base.len() == 40 && base.bytes().all(|b| b == b'0') {
        "origin/main"
    } else {
        base
    }
}

/// `subprocess.run(check=True)`: stdout, or the `CalledProcessError` text.
fn run(root: &Path, argv: &[&str], input: Option<&[u8]>) -> Result<String, String> {
    let shown = list_repr(&argv.iter().map(|a| (*a).to_owned()).collect::<Vec<_>>());
    let mut child = Command::new(argv[0])
        .args(&argv[1..])
        .current_dir(root)
        .stdin(if input.is_some() {
            Stdio::piped()
        } else {
            Stdio::inherit()
        })
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| format!("{shown}: {error}"))?;
    if let (Some(input), Some(mut stdin)) = (input, child.stdin.take()) {
        stdin
            .write_all(input)
            .map_err(|error| format!("{shown}: {error}"))?;
    }
    let output = child
        .wait_with_output()
        .map_err(|error| format!("{shown}: {error}"))?;
    if !output.status.success() {
        let code = output
            .status
            .code()
            .map_or_else(|| "a signal".to_owned(), |code| code.to_string());
        return Err(format!(
            "Command '{shown}' returned non-zero exit status {code}."
        ));
    }
    String::from_utf8(output.stdout).map_err(|error| format!("{shown}: {error}"))
}

// `full-main.py` owns GROUPS and partition_errors; ask it for both rather than restate them.
const FULL_MAIN_GROUPS: &str = "\
import importlib.util, json, sys
root = sys.argv[1]
sys.path.insert(0, root + '/deploy/ci')
spec = importlib.util.spec_from_file_location('full_main', root + '/deploy/ci/full-main.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({'groups': {name: list(members) for name, members in module.GROUPS.items()},
                  'errors': module.partition_errors(json.load(sys.stdin), module.GROUPS)}))
";

/// Lines matching `^ +KEY: \[([^\]]*)\]$` (multiline), as their bracketed text.
fn matrix_rows<'a>(text: &'a str, key: &str) -> Vec<&'a str> {
    text.split('\n')
        .filter_map(|line| {
            let body = line.trim_start_matches(' ');
            if body.len() == line.len() {
                return None;
            }
            let inner = body
                .strip_prefix(key)?
                .strip_prefix(": [")?
                .strip_suffix(']')?;
            (!inner.contains(']')).then_some(inner)
        })
        .collect()
}

fn split_trimmed(row: &str) -> Vec<String> {
    row.split(',').map(|item| item.trim().to_owned()).collect()
}

/// The crate names of a `-p NAME -p NAME ...` scope.
fn scope_crates(flags: &str) -> Vec<String> {
    flags
        .split_whitespace()
        .skip(1)
        .step_by(2)
        .map(str::to_owned)
        .collect()
}

/// Workflow, group and scope declarations (hygiene lane) of the checkout at ROOT; with BASE,
/// also require that nothing declared at that revision is missing.
pub fn static_check(root: &Path, base: Option<&str>) -> Result<Dict, String> {
    let mut errors: Vec<String> = vec![];
    // New-branch pushes have no usable before-SHA; compare their declarations to main.
    let base = base.filter(|base| !base.is_empty()).map(static_base);
    let coverage = root.join("deploy/ci/coverage.json");
    let families = declared_families(&read_text(&coverage)?, &coverage.display().to_string())?;
    let families_json = pyjson::dumps(&strings(families.clone()), false, None);
    let root_text = root.to_str().ok_or("the repository root is not UTF-8")?;
    let full = run(
        root,
        &["python3", "-B", "-c", FULL_MAIN_GROUPS, root_text],
        Some(families_json.as_bytes()),
    )?;
    let full = pyjson::parse_str(&full).map_err(|error| format!("full-main.py GROUPS: {error}"))?;
    let mut groups: Vec<(String, Vec<String>)> = vec![];
    for (name, members) in dict(
        full.get("groups").ok_or_else(|| key_error("groups"))?,
        "GROUPS",
    )?
    .iter()
    {
        let members = members
            .as_list()
            .ok_or_else(|| format!("group {name} is not a list"))?
            .iter()
            .map(|member| {
                member
                    .as_str()
                    .map(str::to_owned)
                    .ok_or_else(|| format!("group {name} names a non-string family"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        groups.push((name.to_owned(), members));
    }
    for error in full
        .get("errors")
        .and_then(Value::as_list)
        .ok_or_else(|| key_error("errors"))?
    {
        errors.push(error.py_str());
    }
    let group_names: Vec<String> = groups.iter().map(|(name, _)| name.clone()).collect();
    let workflow = read_text(&root.join(".github/workflows/full-main.yml"))?;
    let matrix = matrix_rows(&workflow, "group");
    if matrix.len() != 1 || split_trimmed(matrix[0]) != group_names {
        errors.push(format!(
            "full-main.yml cpu matrix {} does not run exactly the groups {}",
            list_repr(
                &matrix
                    .iter()
                    .map(|row| (*row).to_owned())
                    .collect::<Vec<_>>()
            ),
            list_repr(&group_names)
        ));
    }
    for step in ["run", "summarize"] {
        if workflow
            .matches(&format!("full-main.py {step} --group \"$GROUP\" "))
            .count()
            != 1
        {
            errors.push(format!(
                "full-main.yml does not {step} exactly its matrix group"
            ));
        }
    }
    let ci = read_text(&root.join(".github/workflows/ci.yml"))?;
    let shards = matrix_rows(&ci, "shard");
    let expected: Vec<String> = (1..=RUST_TEST_SHARDS).map(|n| n.to_string()).collect();
    if shards.len() != 1 || split_trimmed(shards[0]) != expected {
        errors.push(format!(
            "ci.yml rust-test matrix {} is not shards {}",
            list_repr(
                &shards
                    .iter()
                    .map(|row| (*row).to_owned())
                    .collect::<Vec<_>>()
            ),
            list_repr(&expected)
        ));
    }
    if ci
        .matches(&format!(
            "JOPE_RUST_TEST_PARTITION: slice:${{{{ matrix.shard }}}}/{RUST_TEST_SHARDS}\n"
        ))
        .count()
        != 1
    {
        errors.push(format!(
            "ci.yml rust-test does not partition by slice:<shard>/{RUST_TEST_SHARDS}"
        ));
    }
    let scope = scope_crates(&run(root, &["bash", "deploy/ci/rust-test-scope.sh"], None)?);
    let mut report = Dict::new();
    report.insert("schema", Value::from(SCHEMA));
    report.insert("kind", Value::from("static"));
    let mut full_main = Dict::new();
    full_main.insert("declared", strings(families));
    full_main.insert(
        "groups",
        Value::Dict(
            groups
                .iter()
                .map(|(name, members)| (name.clone(), strings(members.clone())))
                .collect(),
        ),
    );
    report.insert("full_main", Value::Dict(full_main));
    let mut rust_test = Dict::new();
    rust_test.insert("shards", Value::from(RUST_TEST_SHARDS));
    rust_test.insert("scope", strings(scope.clone()));
    report.insert("rust_test", Value::Dict(rust_test));
    if let Some(base) = base {
        let spec = format!("{base}:deploy/ci/coverage.json");
        let old = declared_families(&run(root, &["git", "show", &spec], None)?, &spec)?;
        let assigned: BTreeSet<&String> = groups.iter().flat_map(|(_, members)| members).collect();
        let old_set: BTreeSet<&String> = old.iter().collect();
        let spec = format!("{base}:deploy/ci/rust-test-scope.sh");
        let script = run(root, &["git", "show", &spec], None)?;
        let old_scope = scope_crates(&run(root, &["bash", "-s"], Some(script.as_bytes()))?);
        let now: BTreeSet<&String> = scope.iter().collect();
        let missing_families: Vec<String> = old_set
            .difference(&assigned)
            .map(|s| (*s).clone())
            .collect();
        let added_families: Vec<String> = assigned
            .difference(&old_set)
            .map(|s| (*s).clone())
            .collect();
        let missing_scope: Vec<String> = old_scope
            .iter()
            .collect::<BTreeSet<_>>()
            .difference(&now)
            .map(|s| (*s).clone())
            .collect();
        errors.extend(
            missing_families
                .iter()
                .map(|name| format!("family {name} ran on {base} but is in no group")),
        );
        errors.extend(
            missing_scope.iter().map(|name| {
                format!("crate {name} was in the {base} rust-test scope but is not now")
            }),
        );
        let mut row = Dict::new();
        row.insert("rev", Value::from(base));
        row.insert("full_main_families", strings(old));
        row.insert("missing_families", strings(missing_families));
        row.insert("added_families", strings(added_families));
        row.insert("rust_test_scope", strings(old_scope));
        row.insert("missing_scope", strings(missing_scope));
        report.insert("base", Value::Dict(row));
    }
    report.insert("status", Value::from(status(&errors)));
    report.insert("errors", strings(errors));
    Ok(report)
}

/// The report of a check that could not complete (`kind` is the command).
pub fn failure(kind: &str, error: String) -> Dict {
    let mut report = Dict::new();
    report.insert("schema", Value::from(SCHEMA));
    report.insert("kind", Value::from(kind));
    report.insert("status", Value::from("failed"));
    report.insert("errors", strings([error]));
    report
}

/// Print REPORT (indent 2) and, given OUTPUT, first write it there (exclusive create, parents
/// made). The exit status is 0 only for a passed report.
pub fn emit(report: &Dict, output: Option<&Path>) -> Result<u8, String> {
    let text = pyjson::dumps(&Value::Dict(report.clone()), false, Some(2)) + "\n";
    if let Some(output) = output {
        if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("{}: {error}", parent.display()))?;
        }
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)
            .and_then(|mut file| file.write_all(text.as_bytes()))
            .map_err(|error| format!("{}: {error}", output.display()))?;
    }
    print!("{text}");
    Ok(u8::from(!is_str(
        report.get("status").unwrap_or(&Value::Null),
        "passed",
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_push_base_compares_with_main() {
        assert_eq!("origin/main", static_base(&"0".repeat(40)));
        assert_eq!("a".repeat(40), static_base(&"a".repeat(40)));
        assert_eq!("0", static_base("0"));
    }

    #[test]
    fn slice_partition_is_matched_in_full() {
        assert_eq!(Some(("2", "3")), slice("slice:2/3"));
        for bad in [
            "",
            "slice:0/3",
            "slice:2/03",
            "slice:2/3 ",
            "hash:1/3",
            "slice:/3",
            "slice:2",
        ] {
            assert_eq!(None, slice(bad), "{bad}");
        }
    }

    #[test]
    fn matrix_rows_match_indented_bracket_lines() {
        let text = "  group: [cpu-1, cpu-2]\ngroup: [x]\n    group: [a]b]\n  group: [y] \n";
        assert_eq!(vec!["cpu-1, cpu-2"], matrix_rows(text, "group"));
    }

    #[test]
    fn digest_is_order_independent_and_python_compatible() {
        let tests = [
            ("b".to_owned(), "x".to_owned()),
            ("a".to_owned(), "y".to_owned()),
        ];
        // hashlib.sha256(b"a\ty\nb\tx\n").hexdigest()
        let mut hasher = Sha256::new();
        hasher.update(b"a\ty\nb\tx\n");
        let expected: String = hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(expected, digest(&tests));
        assert_eq!(expected, digest(tests.iter().rev()));
    }
}
