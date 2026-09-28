//! `jope-ci build-report` against the outputs of the Python it replaced.
//!
//! `tests/fixtures/python-parity/build-report/python-outputs.json` holds what
//! `deploy/maintenance/jope_report.py` (archived at 8a55c4fef) printed in the differential run
//! `py-p1-u5-diff-20260927T025057Z`, before it was deleted. That run scanned a synthetic build
//! tree (the retired `test_report.py` layout); `synthetic_tree` below rebuilds it exactly, with the
//! same fixed epoch for everything old. Re-running the archived Python on the rebuilt tree
//! reproduced every captured output (run `py-p1-fixtures-u5-20260927T033842Z`).
//!
//! Normalised in the fixture and in the Rust output alike: the tree root (`@FIXTURE@`) and its
//! parent (`@RUN@`), `completed_at`, allocated bytes (they depend on the filesystem; checked here
//! to be positive and to add up instead) and any `newest_mtime` other than the fixed epoch
//! (checked here to be the time the test touched it). Receipts drop timestamps, checksums, custody
//! paths and the host-dependent `missing_unit_files`.
//!
//! The only allowed differences are argument-parser wording (argparse vs clap) for the three bad
//! `--stale` values; both exit 2 and print nothing on stdout. The report also takes the approved
//! roots as `--approved-root` arguments (from `deploy/maintenance/jope-build-report.sh`) where the
//! Python held them as a constant.

use serde_json::Value;
use std::fs;
use std::os::unix::fs::{symlink, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn captured() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/python-outputs.json");
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

/// The approved roots the installed wrapper passes (`--approved-root`, twice).
fn approved_roots() -> Vec<String> {
    let wrapper =
        fs::read_to_string(repo().join("deploy/maintenance/jope-build-report.sh")).unwrap();
    let words: Vec<&str> = wrapper.split_whitespace().collect();
    let roots: Vec<String> = words
        .windows(2)
        .filter(|pair| pair[0] == "--approved-root")
        .map(|pair| pair[1].to_owned())
        .collect();
    assert_eq!(roots.len(), 2);
    roots
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

fn touch(args: &[&str], paths: &[PathBuf]) {
    let status = Command::new("touch")
        .args(args)
        .args(paths)
        .status()
        .unwrap();
    assert!(status.success());
}

fn all_entries(path: &Path, out: &mut Vec<PathBuf>) {
    let meta = fs::symlink_metadata(path).unwrap();
    if meta.is_dir() {
        for entry in fs::read_dir(path).unwrap() {
            all_entries(&entry.unwrap().path(), out);
        }
    }
    out.push(path.to_path_buf());
}

/// The differential run's tree under `run/fixture`: stale, fresh and nested markers, an unknown
/// directory, a generic CACHEDIR.TAG, a `.fingerprint` candidate, a marker behind a `.git`, a
/// name with a space, a linked root, a linked marker and a linked `.fingerprint`. Everything is
/// set to `old`; two files are then touched now.
fn synthetic_tree(run: &Path, old: u64) -> PathBuf {
    let tree = run.join("fixture");
    for name in [
        "stale/.rustc_info.json",
        "fresh/.rustc_info.json",
        "target/unknown",
        "generic/CACHEDIR.TAG",
        "stale/debug/receipt.bin",
        "stale/release/proof",
        "stale/tmp/evidence",
        "stale/nested/.rustc_info.json",
        "fingerprint/release/.fingerprint/marker",
        "fresh/debug/recent",
        "recent-nested/.rustc_info.json",
        "recent-nested/nested/.rustc_info.json",
        "odd name/.rustc_info.json",
        "gitguard/.rustc_info.json",
        "gitguard/.git/objects/pack",
    ] {
        let path = tree.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"\x00preserve\xff\n").unwrap();
    }
    let outside = run.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join(".rustc_info.json"), "{}\n").unwrap();
    symlink(&outside, tree.join("linked")).unwrap();
    fs::create_dir(tree.join("marker-link")).unwrap();
    symlink(
        outside.join(".rustc_info.json"),
        tree.join("marker-link/.rustc_info.json"),
    )
    .unwrap();
    fs::create_dir_all(tree.join("relsym/release")).unwrap();
    symlink(&outside, tree.join("relsym/release/.fingerprint")).unwrap();
    let mut entries = vec![];
    all_entries(&tree, &mut entries);
    touch(&["-h", "-d", &format!("@{old}")], &entries);
    touch(
        &[],
        &[
            tree.join("fresh/debug/recent"),
            tree.join("recent-nested/nested/.rustc_info.json"),
        ],
    );
    tree
}

fn jope_ci(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_jope-ci"))
        .arg("build-report")
        .args(args)
        .output()
        .unwrap()
}

/// A scan document as the fixture records it; the replaced values are checked first.
fn normalise_scan(text: &str, old: &str, started: f64) -> String {
    if !text.is_empty() {
        let doc: Value = serde_json::from_str(text).unwrap();
        let candidates = doc["candidates"].as_array().unwrap();
        let mut sum = 0;
        for c in candidates {
            let bytes = c["allocated_bytes"].as_u64().unwrap();
            assert!(bytes > 0, "{c}");
            sum += bytes;
            let mtime = c["newest_mtime"].as_f64().unwrap();
            let fixed: f64 = old.parse().unwrap();
            assert!(
                mtime == fixed || (started - 5.0..=now() + 5.0).contains(&mtime),
                "{c}: newest_mtime is neither the fixed epoch nor the time the test touched it"
            );
        }
        assert_eq!(doc["candidate_allocated_bytes_sum"].as_u64(), Some(sum));
    }
    let mut out = String::new();
    for line in text.split_inclusive('\n') {
        let body = line.trim_start();
        let indent = &line[..line.len() - body.len()];
        let (key, value) = body.split_once(": ").unwrap_or(("", body));
        let comma = if value.trim_end().ends_with(',') {
            ","
        } else {
            ""
        };
        let newline = if line.ends_with('\n') { "\n" } else { "" };
        let bare = value.trim_end().trim_end_matches(',');
        match key {
            "\"completed_at\"" => out.push_str(&format!(
                "{indent}{key}: \"@COMPLETED_AT@\"{comma}{newline}"
            )),
            "\"allocated_bytes\"" | "\"candidate_allocated_bytes_sum\"" => {
                out.push_str(&format!("{indent}{key}: \"@ALLOCATED@\"{comma}{newline}"))
            }
            "\"newest_mtime\"" if bare != old => {
                out.push_str(&format!("{indent}{key}: \"@RECENT@\"{comma}{newline}"))
            }
            _ => out.push_str(line),
        }
    }
    out
}

#[test]
fn scan_matches_the_captured_python_outputs() {
    let captured = captured();
    let old_epoch = captured["old_epoch"].as_f64().unwrap();
    let old = format!("{old_epoch:.1}");
    let temp = tempfile::Builder::new()
        .prefix("build-report-parity-")
        .tempdir()
        .unwrap();
    let run = temp.path().canonicalize().unwrap();
    let started = now();
    let tree = synthetic_tree(&run, old_epoch as u64);
    let (tree_text, run_text) = (
        tree.to_str().unwrap().to_owned(),
        run.to_str().unwrap().to_owned(),
    );
    let guarded = tree.join("gitguard/.git/objects");
    let cases = captured["scan_cases"].as_array().unwrap();
    assert_eq!(cases.len(), 11, "every captured scan case is replayed");
    let mut replayed = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        if name == "scan-descendant-failure" {
            fs::set_permissions(&guarded, fs::Permissions::from_mode(0o000)).unwrap();
            if fs::read_dir(&guarded).is_ok() {
                // A user that ignores permissions (root) cannot reproduce the traversal failure.
                fs::set_permissions(&guarded, fs::Permissions::from_mode(0o755)).unwrap();
                eprintln!("{name}: skipped, permissions are not enforced for this user");
                continue;
            }
        }
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a.as_str()
                    .unwrap()
                    .replace("@FIXTURE@", &tree_text)
                    .replace("@RUN@", &run_text)
            })
            .collect();
        let argv: Vec<&str> = argv.iter().map(String::as_str).collect();
        let output = jope_ci(&argv);
        if name == "scan-descendant-failure" {
            fs::set_permissions(&guarded, fs::Permissions::from_mode(0o755)).unwrap();
        }
        let paths = |bytes: &[u8]| {
            String::from_utf8_lossy(bytes)
                .replace(&tree_text, "@FIXTURE@")
                .replace(&run_text, "@RUN@")
        };
        let stdout = normalise_scan(&String::from_utf8_lossy(&output.stdout), &old, started)
            .replace(&tree_text, "@FIXTURE@")
            .replace(&run_text, "@RUN@");
        let stderr = paths(&output.stderr);
        assert_eq!(
            Some(i64::from(output.status.code().unwrap())),
            case["exit"].as_i64(),
            "{name}: exit status; stderr {stderr}"
        );
        let (want_stdout, want_stderr) = (
            case["stdout"].as_str().unwrap(),
            case["stderr"].as_str().unwrap(),
        );
        // Allowed difference: argument-parser wording for a bad `--stale` value.
        let parser = match name {
            "scan-bad-age" => Some((
                "argument --stale: --stale requires nonnegative whole hours",
                "invalid value 'wat' for '--stale <STALE>': --stale requires nonnegative whole hours",
            )),
            "scan-negative-age" => Some((
                "argument --stale: --stale requires nonnegative whole hours",
                "unexpected argument '-1' found",
            )),
            "scan-no-age" => Some((
                "argument --stale: expected one argument",
                "a value is required for '--stale <STALE>' but none was supplied",
            )),
            _ => None,
        };
        if let Some((python, rust)) = parser {
            assert_eq!((stdout.as_str(), want_stdout), ("", ""), "{name}");
            assert!(
                want_stderr.starts_with("usage: jope_report.py scan ")
                    && want_stderr.contains(&format!("jope_report.py scan: error: {python}\n")),
                "{name}: captured Python stderr {want_stderr}"
            );
            assert!(
                stderr.starts_with(&format!("error: {rust}\n")),
                "{name}: Rust stderr {stderr}"
            );
        } else {
            assert_eq!(stdout, want_stdout, "{name}: stdout");
            assert_eq!(stderr, want_stderr, "{name}: stderr");
        }
        replayed += 1;
    }
    assert!(replayed >= 10);
}

/// Custody under /tmp or inside a Git working tree is refused before anything is created.
#[test]
fn custody_refusals_match_the_captured_python_outputs() {
    let captured = captured();
    let temp = tempfile::Builder::new()
        .prefix("build-report-parity-")
        .tempdir()
        .unwrap();
    let config = temp.path().join("config.json");
    fs::write(
        &config,
        "{\"cargo_roots\": [], \"stale_hours\": 48, \"measurement_roots\": []}\n",
    )
    .unwrap();
    let unique = format!("never-created-python-parity-{}", std::process::id());
    let roots = approved_roots();
    for refusal in captured["custody_refusals"].as_array().unwrap() {
        let name = refusal["name"].as_str().unwrap();
        let custody = match name {
            "report-tmp" => PathBuf::from("/tmp").join(&unique),
            "report-tree" => repo().canonicalize().unwrap().join(&unique),
            other => panic!("unexpected case {other}"),
        };
        let output = jope_ci(&[
            "report",
            "--config",
            config.to_str().unwrap(),
            "--custody",
            custody.to_str().unwrap(),
            "--approved-root",
            &roots[0],
            "--approved-root",
            &roots[1],
        ]);
        assert_eq!(
            Some(i64::from(output.status.code().unwrap())),
            refusal["exit"].as_i64(),
            "{name}"
        );
        assert_eq!(output.stdout, b"", "{name}");
        assert_eq!(
            String::from_utf8_lossy(&output.stderr),
            refusal["stderr"].as_str().unwrap(),
            "{name}"
        );
        assert!(!custody.exists(), "{name}: custody was created");
    }
}

/// Failure receipts for malformed, incomplete and wrong-root configurations. Like the other
/// report fixtures it needs custody outside /tmp and Git working trees:
///
/// ```sh
/// JOPE_MAINTENANCE_SCRATCH=<new dir> cargo test -p jope-ci --test build_report_python_parity -- --ignored
/// ```
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn failure_receipts_match_the_captured_python_outputs() {
    let captured = captured();
    let cases = captured["report_receipts"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    replay_receipts(cases, "python-parity-receipts");
}

/// Malformed configurations the Python refused before any scan (a configuration that is not an
/// object, a wrong `stale_hours` or `cargo_roots`, `measurement_roots` that is not a list):
/// `tests/fixtures/python-parity/build-report/negative.json` holds what
/// `deploy/maintenance/jope_report.py` as it stood at `origin/main` `f46a9048f` (unmodified, run
/// from its git bytes) printed and wrote for each; its `source` and `capture` fields name the
/// custody run and the command. The port writes the same failure receipt, prints the same path and
/// exits 1 for every one of them. Same environment as the test above.
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn config_refusals_match_the_captured_python_outputs() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 12, "every captured case is replayed");
    assert!(cases.iter().all(|c| c["exit"] == 1));
    replay_receipts(cases, "python-parity-config-refusals");
}

/// Round two of the differential review (`build-report/negative-r2.json`): scans of trees holding
/// names that are not UTF-8 (reached and measured as the bytes they are; a candidate whose path
/// is not UTF-8 fails in `du`'s decoded output with the Python's codec message) and roots that
/// cannot be resolved (named as `os.path.realpath(strict=True)` names them). Host-dependent fields
/// are blanked on both sides as the fixture records (`normalisation`).
#[test]
fn scans_match_round_two() {
    replay_scans(&round_two(), 6);
}

/// Round three (`build-report/negative-r3.json`, review 5333747872 of PR #117): scan roots spelled
/// with POSIX's distinct `//` root, which `os.path.abspath` keeps and `resolve(strict=True)` does
/// not, so the Python refused them as roots with symlink components; `///` and `/./` spellings of
/// the same root, which it scanned; and a missing root under `//`, refused while resolving it.
#[test]
fn scans_match_round_three() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r3.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    replay_scans(&doc, 6);
}

/// Replay the scan cases of a round fixture: each tree built under a fresh root, `scan` run on the
/// case's arguments, and the exit status, stdout and stderr compared (host-dependent fields blanked
/// on both sides).
fn replay_scans(doc: &Value, count: usize) {
    use std::os::unix::ffi::OsStrExt;
    let cases: Vec<&Value> = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "scan")
        .collect();
    assert_eq!(cases.len(), count, "every captured scan is replayed");
    let unhex = |text: &str| -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    };
    let temp = tempfile::Builder::new()
        .prefix("build-report-round-")
        .tempdir()
        .unwrap();
    let mut failures = vec![];
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let root = temp.path().join(name);
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        for entry in case["tree"].as_array().unwrap() {
            let relative = match (
                entry.get("dir").or(entry.get("file")),
                entry.get("dir_hex").or(entry.get("file_hex")),
            ) {
                (Some(text), _) => text.as_str().unwrap().as_bytes().to_vec(),
                (None, Some(hex)) => unhex(hex.as_str().unwrap()),
                _ => panic!("{name}: a tree entry without a path"),
            };
            let path = root.join(std::ffi::OsStr::from_bytes(&relative));
            if entry.get("dir").is_some() || entry.get("dir_hex").is_some() {
                fs::create_dir_all(&path).unwrap();
            } else {
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, "x").unwrap();
            }
        }
        let root_text = root.to_str().unwrap();
        let args: Vec<String> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().replace("@ROOT@", root_text))
            .collect();
        let mut argv = vec!["scan"];
        argv.extend(args.iter().map(String::as_str));
        let output = jope_ci(&argv);
        let stdout =
            blank_scan(&String::from_utf8_lossy(&output.stdout).replace(root_text, "@ROOT@"));
        let stderr =
            blank_position(&String::from_utf8_lossy(&output.stderr).replace(root_text, "@ROOT@"));
        let exit = i64::from(output.status.code().unwrap());
        let want = (
            case["exit"].as_i64().unwrap(),
            case["stdout"].as_str().unwrap(),
            case["stderr"].as_str().unwrap(),
        );
        if (exit, stdout.as_str(), stderr.as_str()) != want {
            failures.push(format!(
                "{name}: the Python {want:?}, the port {:?}",
                (exit, stdout, stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(blank_position("in position 81: x"), "in position @N@: x");
}

/// Configurations the Python read as `json.loads(bytes)` (decoded with `surrogatepass`): its
/// codec message for bytes that are not UTF-8, identical in the failure receipt. Same environment
/// as the tests above.
///
/// Two cases hold a lone surrogate, which the Python decoded and then refused as a configuration
/// that does not name the approved roots; the port cannot represent the surrogate and refuses the
/// configuration when it reads it (a failure receipt too, with its own message).
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn config_bytes_match_round_two() {
    const UNREPRESENTABLE: &[&str] = &["config-utf8-encoded-surrogate", "config-surrogate-escape"];
    let doc = round_two();
    let cases: Vec<&Value> = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "report")
        .collect();
    assert_eq!(cases.len(), 4, "every captured configuration is replayed");
    let scratch = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    )
    .join("python-parity-config-bytes-r2");
    fs::create_dir_all(scratch.parent().unwrap()).unwrap();
    fs::create_dir(&scratch).expect("a fresh scratch directory");
    let roots = approved_roots();
    let tools = repo().join("deploy/maintenance");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let hex = case["config_hex"].as_str().unwrap();
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
            .collect();
        let config = scratch.join(format!("{name}.json"));
        fs::write(&config, bytes).unwrap();
        let custody = scratch.join(format!("custody-{name}"));
        let output = jope_ci(&[
            "report",
            "--config",
            config.to_str().unwrap(),
            "--custody",
            custody.to_str().unwrap(),
            "--tools-dir",
            tools.to_str().unwrap(),
            "--approved-root",
            &roots[0],
            "--approved-root",
            &roots[1],
        ]);
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert_eq!(case["exit"], 1, "{name}");
        let runs: Vec<PathBuf> = fs::read_dir(&custody)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(runs.len(), 1, "{name}");
        let failure = fs::read_to_string(runs[0].join("failure.txt")).unwrap();
        if UNREPRESENTABLE.contains(&name) {
            assert!(
                failure.contains("lone surrogate") || failure.contains("Unpaired surrogate"),
                "{name}: {failure}"
            );
        } else {
            assert_eq!(
                failure,
                case["failure_txt"].as_str().unwrap(),
                "{name}: failure.txt"
            );
        }
    }
}

fn round_two() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r2.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    doc
}

/// The host-dependent fields of a scan document blanked as the fixture blanks them.
fn blank_scan(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        let trimmed = line.trim_start();
        let indent = &line[..line.len() - trimmed.len()];
        let mut done = false;
        for (key, value) in [
            ("\"completed_at\": ", "\"@T@\""),
            ("\"allocated_bytes\": ", "0"),
            ("\"newest_mtime\": ", "0"),
            ("\"candidate_allocated_bytes_sum\": ", "0"),
        ] {
            if let Some(rest) = trimmed.strip_prefix(key) {
                let comma = if rest.ends_with(',') { "," } else { "" };
                out.push_str(&format!("{indent}{key}{value}{comma}\n"));
                done = true;
                break;
            }
        }
        if !done {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// A codec position written `@N@`: it counts the host path `du` printed.
fn blank_position(text: &str) -> String {
    let Some(at) = text.find("in position ") else {
        return text.to_owned();
    };
    let start = at + "in position ".len();
    let digits = text[start..].bytes().take_while(u8::is_ascii_digit).count();
    format!("{}@N@{}", &text[..start], &text[start + digits..])
}

/// Run `report` on each case's configuration under a fresh custody directory of the scratch
/// directory `JOPE_MAINTENANCE_SCRATCH` names, and compare the exit status, stdout, stderr, the
/// receipt (normalised as the fixture is) and `failure.txt` with the captured Python outputs.
fn replay_receipts(cases: &[Value], subdirectory: &str) {
    let scratch = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    )
    .join(subdirectory);
    fs::create_dir_all(scratch.parent().unwrap()).unwrap();
    fs::create_dir(&scratch).expect("a fresh scratch directory");
    let scratch = scratch.canonicalize().unwrap();
    let roots = approved_roots();
    let tools = repo().join("deploy/maintenance");
    // A fixture writes the approved roots as `@APPROVED_ROOT_<n>@` (the no-home-paths lint scans
    // the tests); they stand for the roots the wrapper passes.
    let place = |text: &str| {
        text.replace("@APPROVED_ROOT_0@", &roots[0])
            .replace("@APPROVED_ROOT_1@", &roots[1])
    };
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let config = scratch.join(format!("{name}.json"));
        fs::write(&config, place(case["config"].as_str().unwrap())).unwrap();
        let custody = scratch.join(format!("custody-{name}"));
        let output = jope_ci(&[
            "report",
            "--config",
            config.to_str().unwrap(),
            "--custody",
            custody.to_str().unwrap(),
            "--tools-dir",
            tools.to_str().unwrap(),
            "--approved-root",
            &roots[0],
            "--approved-root",
            &roots[1],
        ]);
        assert_eq!(
            Some(i64::from(output.status.code().unwrap())),
            case["exit"].as_i64(),
            "{name}"
        );
        assert_eq!(
            String::from_utf8_lossy(&output.stderr),
            case["stderr"].as_str().unwrap(),
            "{name}"
        );
        let runs: Vec<PathBuf> = fs::read_dir(&custody)
            .unwrap()
            .map(|e| e.unwrap().path())
            .collect();
        assert_eq!(runs.len(), 1, "{name}");
        let run = &runs[0];
        let stamp = run.file_name().unwrap().to_str().unwrap();
        assert_eq!(
            String::from_utf8_lossy(&output.stdout)
                .replace(custody.to_str().unwrap(), "@CUSTODY@")
                .replace(stamp, "@STAMP@"),
            case["stdout"].as_str().unwrap(),
            "{name}"
        );
        let mut receipt: Value =
            serde_json::from_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
        for key in [
            "started_at",
            "completed_at",
            "checksums",
            "custody_argument",
            "custody_resolved",
            "missing_unit_files",
        ] {
            receipt.as_object_mut().unwrap().remove(key);
        }
        let want: Value = serde_json::from_str(&place(&case["receipt"].to_string())).unwrap();
        assert_eq!(receipt, want, "{name}: receipt");
        assert_eq!(
            fs::read_to_string(run.join("failure.txt")).unwrap(),
            case["failure_txt"].as_str().unwrap(),
            "{name}: failure.txt"
        );
    }
}
