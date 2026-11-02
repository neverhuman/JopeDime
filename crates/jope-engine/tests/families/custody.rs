use crate::families_growth_support::*;
use jope_parallel::family_growth::{self, GrowthError};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

// ---- custody that cannot be resolved (review round five of PR #117) --------------------------
//
// `escapes()` took a `canonicalize()` error for "does not escape": a policy or reviewer-key
// directory linked out of the workspace to a directory so deep that the resolved name is longer
// than `PATH_MAX` could not be resolved, yet opened through the root's short path, so the native
// check passed and read custody from outside the root. The retired Python refused it
// (`pinned provenance policy escaped workspace custody`): its non-strict `resolve()` built the
// long name anyway and found it outside the root. `tests/fixtures/family-growth/
// python-custody-r5.jsonl` holds the Python's outcome on each case below (captured by
// `capture_growth_custody_outputs_from_python`, run `p1-sweep5-capture-*`).

const CUSTODY_FIXTURE: &str = "tests/fixtures/family-growth/python-custody-r5.jsonl";
const CUSTODY_CASES: [&str; 7] = [
    "custody-control",
    "policy-link-out-too-long",
    "policy-link-out",
    "key-link-out-too-long",
    "key-link-out",
    "policy-directory-unsearchable",
    "key-directory-unsearchable",
];

/// A directory under `base` whose absolute name is `len` bytes long (components of 200 bytes).
fn deep_directory(base: &Path, len: usize) -> PathBuf {
    let mut dir = base.join("outside");
    while dir.as_os_str().len() + 201 < len {
        dir = dir.join("d".repeat(200));
    }
    let pad = len - dir.as_os_str().len() - 1;
    dir.join("e".repeat(pad))
}

/// Build custody case `name` under `base`: a grown-family assignment produced while custody was
/// intact, then the policy (`manifests`) or reviewer-key (`credentials/provenance-review`)
/// directory moved out of the root behind a symlink (deep enough that the resolved names exceed
/// `PATH_MAX`, or not), or made unsearchable. Returns the root, the assignment and the directory
/// whose mode must be restored after the run.
fn custody_case(name: &str, base: &Path) -> (PathBuf, Value, Option<PathBuf>) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(base.join("root")).unwrap();
    let root = base.join("root").canonicalize().unwrap();
    let base = base.canonicalize().unwrap();
    let fixture = GrowthFixture::new(&root, None, false);
    let assignment = assign(&root, &fixture.inputs(&["new"])).unwrap();
    let (dir, how) = match name.split_once('-') {
        Some(("custody", _)) => return (root, assignment, None),
        Some(("policy", how)) => ("manifests", how),
        Some(("key", how)) => (family_growth::REVIEWER_KEYS, how),
        _ => panic!("unknown custody case {name}"),
    };
    let inside = root.join(dir);
    match how {
        "link-out-too-long" | "link-out" => {
            let outside = if how == "link-out" {
                base.join("outside-short")
            } else {
                deep_directory(&base, 4090)
            };
            fs::create_dir_all(outside.parent().unwrap()).unwrap();
            fs::rename(&inside, &outside).unwrap();
            std::os::unix::fs::symlink(&outside, &inside).unwrap();
            (root, assignment, None)
        }
        "directory-unsearchable" => {
            fs::set_permissions(&inside, fs::Permissions::from_mode(0o000)).unwrap();
            (root, assignment, Some(inside))
        }
        _ => panic!("unknown custody case {name}"),
    }
}

fn restore(dir: Option<PathBuf>) {
    use std::os::unix::fs::PermissionsExt;
    if let Some(dir) = dir {
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// Custody the native check cannot resolve is refused, never taken as custody kept: each case
/// agrees with the retired Python's captured outcome (the escape refusals word for word; a
/// directory that may not be searched as a malformed-input error, where the Python raised
/// `PermissionError`). A long link out of the workspace was accepted before this round.
#[test]
fn custody_that_cannot_be_resolved_is_refused_as_python_refused_it() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(CUSTODY_FIXTURE);
    let records: Vec<Value> = fs::read_to_string(&fixture)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        records
            .iter()
            .map(|r| r["name"].as_str().unwrap())
            .collect::<Vec<_>>(),
        CUSTODY_CASES,
        "every captured custody case is replayed"
    );
    let mut refused = 0;
    for (name, record) in CUSTODY_CASES.iter().zip(&records) {
        let dir = tempfile::tempdir().unwrap();
        let (root, request, locked) = custody_case(name, dir.path());
        // A user who may search anything (root) cannot reproduce an unsearchable directory.
        let privileged = locked.as_ref().is_some_and(|d| fs::read_dir(d).is_ok());
        let outcome = family_growth::verify_assignment(&request);
        restore(locked);
        if privileged {
            eprintln!("{name}: skipped, this user may search a directory of mode 000");
            continue;
        }
        if let Err(why) = agrees(&outcome, &root, record) {
            panic!("{name}: {why}");
        }
        if outcome.is_err() {
            refused += 1;
        }
    }
    assert!(refused >= 4, "refused {refused}");
    // The long link resolves through the root's short path: it is the check, not the read, that
    // refuses it.
    let dir = tempfile::tempdir().unwrap();
    let (root, request, _) = custody_case("policy-link-out-too-long", dir.path());
    assert!(fs::read(root.join(family_growth::POLICY_PATH)).is_ok());
    assert!(root
        .join(family_growth::POLICY_PATH)
        .canonicalize()
        .is_err());
    assert_eq!(
        family_growth::verify_assignment(&request),
        Err(GrowthError::Refused(
            "pinned provenance policy escaped workspace custody".into()
        ))
    );
}

/// Records the retired Python policy's outcome on the custody cases (`python3 -I -B <policy>
/// --verify <request>`) as `<JOPE_GROWTH_CAPTURE_DIR>/custody-r5.jsonl`, the layout of
/// `python-custody-r5.jsonl`, and checks the native verifier agrees with each.
#[test]
#[ignore = "capture: needs JOPE_GROWTH_POLICY_PY (the archived Python policy) and JOPE_GROWTH_CAPTURE_DIR"]
fn capture_growth_custody_outputs_from_python() {
    let policy = PathBuf::from(std::env::var("JOPE_GROWTH_POLICY_PY").unwrap());
    let out = PathBuf::from(std::env::var("JOPE_GROWTH_CAPTURE_DIR").unwrap());
    let (mut lines, mut disagreements) = (vec![], vec![]);
    for name in CUSTODY_CASES {
        let base = out.join("custody").join(name);
        let (root, request, locked) = custody_case(name, &base);
        let request_path = base.join("request.json");
        fs::write(&request_path, format!("{}\n", py_compact(&request))).unwrap();
        let output = std::process::Command::new("/usr/bin/python3")
            .args(["-I", "-B"])
            .arg(&policy)
            .arg("--verify")
            .arg(&request_path)
            .output()
            .unwrap();
        let native = family_growth::verify_assignment(&request);
        restore(locked);
        let stdout = String::from_utf8(output.stdout).unwrap();
        let stderr = String::from_utf8(output.stderr).unwrap();
        fs::write(base.join("python.stdout"), &stdout).unwrap();
        fs::write(base.join("python.stderr"), &stderr).unwrap();
        let code = output.status.code().unwrap();
        let last = stderr
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .unwrap_or("");
        let (error_type, error) = if code == 0 {
            ("", "")
        } else {
            last.split_once(": ").unwrap_or((last, ""))
        };
        let root_text = root.to_str().unwrap();
        let record = json!({
            "name": name, "exit": code,
            "stdout": stdout.trim_end_matches('\n').replace(root_text, "@ROOT@"),
            "error_type": error_type,
            "error": error.replace(root_text, "@ROOT@"),
        });
        if let Err(why) = agrees(&native, &root, &record) {
            disagreements.push(json!({"name": name, "why": why}));
        }
        lines.push(py_compact(&record));
    }
    fs::write(out.join("custody-r5.jsonl"), lines.join("\n") + "\n").unwrap();
    fs::write(
        out.join("custody-r5-disagreements.json"),
        serde_json::to_vec_pretty(&disagreements).unwrap(),
    )
    .unwrap();
    assert!(disagreements.is_empty(), "{disagreements:?}");
}

/// Regenerates `tests/fixtures/family-growth/python-verify-outputs.jsonl` from the archived
/// Python policy and checks both reference forms (relative, the committed one, and absolute,
/// as the Python fixture wrote them). Restore the policy with
/// `git show 8a55c4fef:crates/jope-engine/policy/family_growth_policy.py > <file>`.
#[test]
#[ignore = "capture: needs JOPE_GROWTH_POLICY_PY (the archived Python policy) and JOPE_GROWTH_CAPTURE_DIR"]
fn capture_growth_policy_outputs_from_python() {
    let policy = PathBuf::from(std::env::var("JOPE_GROWTH_POLICY_PY").unwrap());
    let out = PathBuf::from(std::env::var("JOPE_GROWTH_CAPTURE_DIR").unwrap());
    let mut report = Map::new();
    for mode in ["relative", "absolute"] {
        let (mut lines, mut disagreements) = (vec![], vec![]);
        let mut outcomes: BTreeMap<String, usize> = BTreeMap::new();
        for name in scenario_names() {
            let base = out.join(mode).join(name.replace('/', "-"));
            let (root, request) = build_scenario(&name, &base, mode == "absolute");
            // Digest the scenario before the request and the Python outputs join it.
            let digest = scenario_digest(&base, &root, &request);
            let request_path = base.join("request.json");
            fs::write(&request_path, format!("{}\n", py_compact(&request))).unwrap();
            let output = std::process::Command::new("/usr/bin/python3")
                .args(["-I", "-B"])
                .arg(&policy)
                .arg("--verify")
                .arg(&request_path)
                .output()
                .unwrap();
            let stdout = String::from_utf8(output.stdout).unwrap();
            let stderr = String::from_utf8(output.stderr).unwrap();
            fs::write(base.join("python.stdout"), &stdout).unwrap();
            fs::write(base.join("python.stderr"), &stderr).unwrap();
            let code = output.status.code().unwrap();
            let last = stderr
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .unwrap_or("");
            let (error_type, error) = if code == 0 {
                ("", "")
            } else {
                last.split_once(": ").unwrap_or((last, ""))
            };
            let record = json!({
                "name": name, "inputs_sha256": digest,
                "exit": code,
                "stdout": stdout.trim_end_matches('\n').replace(root.to_str().unwrap(), "@ROOT@"),
                "error_type": error_type, "error": error,
            });
            *outcomes
                .entry(if code == 0 {
                    "accepted".into()
                } else {
                    error_type.to_owned()
                })
                .or_default() += 1;
            let native = family_growth::verify_assignment(&request);
            if let Err(why) = agrees(&native, &root, &record) {
                disagreements.push(json!({"name": name, "why": why}));
            }
            lines.push(py_compact(&record));
        }
        fs::write(out.join(format!("{mode}.jsonl")), lines.join("\n") + "\n").unwrap();
        report.insert(
            mode.into(),
            json!({"scenarios": lines.len(), "outcomes": outcomes, "disagreements": disagreements}),
        );
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    for mode in ["relative", "absolute"] {
        assert!(
            report[mode]["disagreements"].as_array().unwrap().is_empty(),
            "{mode}: {}",
            report[mode]["disagreements"]
        );
    }
}
