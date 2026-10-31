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
//!
//! `negative-r5.json` holds round five (the fail-closed sweep), captured from the same Python:
//! scans of a directory that may be listed but not searched, links that are never entered, a
//! relative root from a deleted working directory, and reports whose unit files under `HOME` cannot
//! be inspected (`scans_match_round_five`, `scan_from_a_deleted_working_directory_matches_round_five`,
//! and the ignored `units_match_round_five`).
//!
//! `negative-r6.json` holds round six (lossy path conversion and `HOME`): scans whose root is not
//! UTF-8 (`scans_match_round_six`), and reports under `HOME` spellings pathlib refused (`~...`) or
//! the port refuses (empty, relative, not UTF-8) and with custody, configuration or tools behind
//! names that are not UTF-8 (the ignored `reports_match_round_six`).
//!
//! `negative-r7.json` holds round seven (the argument grammar): `help` where a command is expected,
//! `--version`, a repeated or abbreviated `--stale`, an empty root and a second root
//! (`grammar_matches_round_seven`).

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
    // The remote test helper runs a source archive without an enclosing `.git` marker. Keep the
    // worktree refusal case self-contained instead of relying on the developer's checkout layout.
    let tree = tempfile::tempdir_in(repo()).unwrap();
    fs::write(tree.path().join(".git"), b"gitdir: /not-needed\n").unwrap();
    let roots = approved_roots();
    for refusal in captured["custody_refusals"].as_array().unwrap() {
        let name = refusal["name"].as_str().unwrap();
        let custody = match name {
            "report-tmp" => PathBuf::from("/tmp").join(&unique),
            "report-tree" => tree.path().join(&unique),
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

fn round_five() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r5.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    assert_eq!(
        doc["cases"].as_array().unwrap().len(),
        16,
        "every captured case is replayed"
    );
    doc
}

/// Round five (`build-report/negative-r5.json`, the fail-closed sweep): a directory that may be
/// listed but not searched. `os.walk` entered it by its `d_type` and raised on its subdirectory;
/// the port took the subdirectory for a file because its `stat` failed and scanned on. Links into
/// a directory that cannot be searched, dangling links and loops are listed, never entered, on
/// both sides.
#[test]
fn scans_match_round_five() {
    replay_scans(&round_five(), 8);
}

/// Round five: `scan` of a relative root from a working directory that no longer exists.
/// `os.path.abspath` raised (`os.getcwd()`); the port took the working directory to be `/` and
/// scanned the root spelled from there. The relative root names the case directory from `/`.
#[test]
fn scan_from_a_deleted_working_directory_matches_round_five() {
    let doc = round_five();
    let case = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["kind"] == "scan-from-deleted-cwd")
        .unwrap();
    let temp = tempfile::Builder::new()
        .prefix("build-report-cwd-")
        .tempdir()
        .unwrap();
    let root = temp.path().join("case").join("x");
    fs::create_dir_all(&root).unwrap();
    let root = temp.path().join("case").canonicalize().unwrap();
    let gone = temp.path().join("gone");
    fs::create_dir(&gone).unwrap();
    let relative = root.to_str().unwrap().trim_start_matches('/').to_owned();
    let output = Command::new("/bin/sh")
        .arg("-c")
        .arg("cd \"$1\" && rmdir \"$1\" && shift && exec \"$@\"")
        .arg("sh")
        .arg(&gone)
        .arg(env!("CARGO_BIN_EXE_jope-ci"))
        .args(["build-report", "scan", &relative, "--stale", "99999999"])
        .output()
        .unwrap();
    let text =
        |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(root.to_str().unwrap(), "@ROOT@");
    assert_eq!(
        (
            i64::from(output.status.code().unwrap()),
            text(&output.stdout),
            text(&output.stderr)
        ),
        (
            case["exit"].as_i64().unwrap(),
            case["stdout"].as_str().unwrap().to_owned(),
            case["stderr"].as_str().unwrap().to_owned()
        )
    );
}

/// Round five's unit cases the port has refused since round six (stricter): an empty `HOME`, which
/// Python 3.12 took for `/` and the port refuses as not absolute (the report fails at the same step,
/// the receipt naming `HOME`).
const R5_UNITS_NOW_STRICTER: &[&str] = &["units-r5-home-empty"];

/// Round five: the unit files under `Path.home()`. `path.is_file()` raised in a unit directory
/// that may not be searched, failing the report; the port recorded the units as missing and went
/// on (to a successful report, given a configuration it accepts: `units-r5-no-search-good-config`).
/// `HOME`'s spelling (a trailing or doubled separator) gives the Python's paths; an empty one is
/// refused since round six (`R5_UNITS_NOW_STRICTER`). Same environment as the other receipt
/// replays; each case runs with the case's own `HOME` and its `a` and `b` as the approved roots.
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn units_match_round_five() {
    let doc = round_five();
    let scratch = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    )
    .join("python-parity-units-r5");
    fs::create_dir_all(scratch.parent().unwrap()).unwrap();
    fs::create_dir(&scratch).expect("a fresh scratch directory");
    let scratch = scratch.canonicalize().unwrap();
    let tools = repo().join("deploy/maintenance");
    let mut failures = vec![];
    let mut replayed = 0;
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "units")
    {
        let name = case["name"].as_str().unwrap();
        let c = scratch.join(name);
        for dir in ["home/.config/systemd/user", "a", "b"] {
            fs::create_dir_all(c.join(dir)).unwrap();
        }
        let c_text = c.to_str().unwrap().to_owned();
        let config = match case["config"].as_str().unwrap() {
            "bad" => doc["bad_config"].as_str().unwrap().to_owned(),
            _ => format!(
                r#"{{"cargo_roots": ["{c_text}/a", "{c_text}/b"], "stale_hours": 48, "measurement_roots": []}}"#
            ),
        };
        fs::write(c.join("cfg.json"), config).unwrap();
        let modes = |restore: bool| {
            for entry in case["modes"].as_array().unwrap() {
                let mode = if restore {
                    0o755
                } else {
                    u32::from_str_radix(entry["mode"].as_str().unwrap(), 8).unwrap()
                };
                fs::set_permissions(
                    c.join(entry["path"].as_str().unwrap()),
                    fs::Permissions::from_mode(mode),
                )
                .unwrap();
            }
        };
        modes(false);
        let searchable = !case["modes"].as_array().unwrap().is_empty()
            && fs::metadata(c.join("home/.config/systemd/user/jope-build-sweep.service"))
                .is_err_and(|e| e.kind() == std::io::ErrorKind::NotFound);
        let output = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .args(["build-report", "report", "--config"])
            .arg(c.join("cfg.json"))
            .arg("--custody")
            .arg(c.join("receipts"))
            .arg("--tools-dir")
            .arg(&tools)
            .arg("--approved-root")
            .arg(c.join("a"))
            .arg("--approved-root")
            .arg(c.join("b"))
            .current_dir(&c)
            .env(
                "HOME",
                case["home"].as_str().unwrap().replace("@C@", &c_text),
            )
            .env_remove("INVOCATION_ID")
            .output()
            .unwrap();
        modes(true);
        if searchable {
            eprintln!("{name}: skipped, this user may search a directory of that mode");
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stamp = stdout
            .rsplit('/')
            .nth(1)
            .filter(|s| s.ends_with('Z') && s.len() == 23)
            .map(str::to_owned);
        let normalise = |text: &str| {
            let text = text.replace(&c_text, "@C@");
            match &stamp {
                Some(stamp) => text.replace(stamp.as_str(), "@STAMP@"),
                None => text,
            }
        };
        let mut why = vec![];
        if i64::from(output.status.code().unwrap()) != case["exit"].as_i64().unwrap() {
            why.push(format!("exit {:?}", output.status.code()));
        }
        if normalise(&stdout) != case["stdout"].as_str().unwrap() {
            why.push(format!("stdout {stdout:?}"));
        }
        let stderr = normalise(&String::from_utf8_lossy(&output.stderr));
        if stderr != case["stderr"].as_str().unwrap() {
            why.push(format!("stderr {stderr:?}"));
        }
        let run = Path::new(stdout.trim_end()).parent().unwrap();
        let text = fs::read_to_string(run.join("receipt.json")).unwrap();
        let receipt: Value = serde_json::from_str(&normalise(&text)).unwrap();
        let kept: Value = ["status", "exit_status", "error", "missing_unit_files"]
            .iter()
            .map(|k| {
                (
                    k.to_string(),
                    receipt.get(*k).cloned().unwrap_or(Value::Null),
                )
            })
            .collect::<serde_json::Map<_, _>>()
            .into();
        if R5_UNITS_NOW_STRICTER.contains(&name) {
            // Round six: an empty `HOME` (which Python took for `/`) refuses at the same step.
            let refused = kept["status"] == "failed"
                && kept["error"]
                    .as_str()
                    .is_some_and(|e| e.starts_with("HOME '' is not an absolute path"))
                && kept["missing_unit_files"] == serde_json::json!([]);
            if !refused || case["receipt"]["status"] != "failed" {
                why.push(format!(
                    "listed as stricter since round six: receipt {kept}"
                ));
            }
        } else if kept != case["receipt"] {
            why.push(format!("receipt {kept}, the Python {}", case["receipt"]));
        }
        if !why.is_empty() {
            failures.push(format!("{name}: {}", why.join("; ")));
        }
        replayed += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(replayed >= 4, "replayed {replayed}");
}

fn round_six() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r6.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    assert_eq!(
        doc["cases"].as_array().unwrap().len(),
        25,
        "every captured case is replayed"
    );
    doc
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

/// A codec position the fixture wrote `@<tag>+k@` (k characters past the capture's directory) at
/// the same place past this one, `chars` characters long.
fn place_positions(text: &str, tag: &str, chars: usize) -> String {
    let marker = format!("@{tag}+");
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find(&marker) {
        let after = &rest[at + marker.len()..];
        let digits = after.bytes().take_while(u8::is_ascii_digit).count();
        assert_eq!(&after[digits..=digits], "@", "{text}");
        out.push_str(&rest[..at]);
        out.push_str(&(chars + after[..digits].parse::<usize>().unwrap()).to_string());
        rest = &after[digits + 1..];
    }
    out.push_str(rest);
    out
}

/// Every `receipt.json` below `dir`, links not followed.
fn receipts_below(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let kind = entry.file_type().unwrap();
        if kind.is_dir() {
            receipts_below(&entry.path(), out);
        } else if entry.file_name() == "receipt.json" {
            out.push(entry.path());
        }
    }
}

/// Round six scans the Python ran to completion and the port refuses (stricter), printing nothing:
/// a relative root from a working directory whose name is not UTF-8 (the Python printed the root
/// surrogate-escaped, `"\udcff"`, in its JSON), and a root argument that is not UTF-8 (refused with
/// the other arguments before anything is read). The refusal each begins with.
const R6_SCAN_STRICTER: &[(&str, &str)] = &[
    (
        "scan-r6-relative-root-from-not-utf8-cwd",
        r"report failed: scan root '@ROOT@/\udcff/r' is not UTF-8",
    ),
    (
        "scan-r6-root-argument-not-utf8",
        r"jope-ci: argument 3 '@ROOT@/\udcff' is not UTF-8",
    ),
];

/// Round six of the review of PR #117 (`negative-r6.json`, `kind: scan-cwd`): a root whose resolved
/// name is not UTF-8. A root with a link component is refused naming it as the Python's stderr
/// printed it (`\udcff`; the port printed U+FFFD); `--delete` is refused first, whatever the other
/// arguments hold (the port panicked on the argument that is not UTF-8 before it looked).
#[test]
fn scans_match_round_six() {
    let doc = round_six();
    let temp = tempfile::Builder::new()
        .prefix("build-report-r6-")
        .tempdir()
        .unwrap();
    let mut failures = vec![];
    let mut replayed = 0;
    for case in doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "scan-cwd")
    {
        let name = case["name"].as_str().unwrap();
        let root = temp.path().join(name);
        fs::create_dir_all(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let root_text = root.to_str().unwrap();
        let sub = |text: &str| unescape(&text.replace("@ROOT@", root_text));
        for entry in case["tree"].as_array().unwrap() {
            if let Some(dir) = entry.get("dir") {
                fs::create_dir_all(root.join(unescape(dir.as_str().unwrap()))).unwrap();
            } else {
                let file = root.join(unescape(entry["file"].as_str().unwrap()));
                fs::create_dir_all(file.parent().unwrap()).unwrap();
                fs::write(file, "x").unwrap();
            }
        }
        for link in case["symlinks"].as_array().into_iter().flatten() {
            symlink(
                unescape(link["target"].as_str().unwrap()),
                root.join(unescape(link["path"].as_str().unwrap())),
            )
            .unwrap();
        }
        let cwd = case
            .get("cwd")
            .and_then(Value::as_str)
            .map_or(root.clone(), |c| PathBuf::from(sub(c)));
        let args: Vec<std::ffi::OsString> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| sub(a.as_str().unwrap()))
            .collect();
        let output = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .args(["build-report", "scan"])
            .args(&args)
            .current_dir(&cwd)
            .output()
            .unwrap();
        let stdout =
            blank_scan(&String::from_utf8_lossy(&output.stdout).replace(root_text, "@ROOT@"));
        let stderr = String::from_utf8_lossy(&output.stderr).replace(root_text, "@ROOT@");
        let exit = i64::from(output.status.code().unwrap());
        let verdict = match R6_SCAN_STRICTER.iter().find(|(n, _)| *n == name) {
            Some((_, refusal)) => {
                case["exit"] == 0
                    && exit != 0
                    && stdout.is_empty()
                    && stderr.starts_with(refusal)
                    && stderr.lines().count() == 1
            }
            None => {
                (exit, stdout.as_str(), stderr.as_str())
                    == (
                        case["exit"].as_i64().unwrap(),
                        blank_scan(case["stdout"].as_str().unwrap()).as_str(),
                        case["stderr"].as_str().unwrap(),
                    )
            }
        };
        if !verdict {
            failures.push(format!(
                "{name}: the Python {} {:?} {:?}, the port {exit} {stdout:?} {stderr:?}",
                case["exit"], case["stdout"], case["stderr"]
            ));
        }
        replayed += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(replayed, 5);
}

/// Round seven (`negative-r7.json`, the argument grammar): what the port's parser refuses (exit
/// status 2, clap's `error: ...` containing the text given, nothing on stdout), and whether the
/// Python accepted it (stricter). The word `help` where a command is expected printed help (exit 0)
/// until this round, where argparse refused the choice; a repeated `--stale` (argparse kept the
/// last), a prefix of it (argparse expanded it) and an empty root (`os.path.abspath("")`, the
/// working directory) the Python scanned.
const R7_USAGE_REFUSED: &[(&str, &str, bool)] = &[
    ("report-r7-help", "unrecognized subcommand 'help'", false),
    (
        "report-r7-help-scan",
        "unrecognized subcommand 'help'",
        false,
    ),
    (
        "report-r7-version",
        "unexpected argument '--version' found",
        false,
    ),
    (
        "report-r7-scan-version",
        "unexpected argument '--version' found",
        false,
    ),
    (
        "report-r7-scan-stale-twice",
        "cannot be used multiple times",
        true,
    ),
    (
        "report-r7-scan-stale-twice-last-bad",
        "cannot be used multiple times",
        false,
    ),
    (
        "report-r7-scan-stale-abbrev",
        "unexpected argument '--st' found",
        true,
    ),
    (
        "report-r7-scan-root-empty",
        "a value is required for '<ROOT>' but none was supplied",
        true,
    ),
    (
        "report-r7-scan-root-twice",
        "unexpected argument '@ROOT@' found",
        false,
    ),
];

/// Round seven of the review of PR #117 (`negative-r7.json`): `jope-ci build-report` given the
/// arguments `jope_report.py` was given, in an empty working directory. Every case the Python
/// refused, the port refuses; `--stale=5` scans as the Python scanned.
#[test]
fn grammar_matches_round_seven() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r7.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 10, "every captured case is replayed");
    for (name, _, _) in R7_USAGE_REFUSED {
        assert!(cases.iter().any(|c| c["name"] == *name), "{name}");
    }
    let temp = tempfile::Builder::new()
        .prefix("build-report-r7-")
        .tempdir()
        .unwrap();
    let mut failures = vec![];
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let root = temp.path().join(name);
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let root_text = root.to_str().unwrap();
        let args: Vec<String> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().replace("@ROOT@", root_text))
            .collect();
        let output = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("build-report")
            .args(&args)
            .current_dir(&root)
            .output()
            .unwrap();
        let stdout =
            blank_scan(&String::from_utf8_lossy(&output.stdout).replace(root_text, "@ROOT@"));
        let stderr = String::from_utf8_lossy(&output.stderr).replace(root_text, "@ROOT@");
        let exit = i64::from(output.status.code().unwrap());
        let python = (
            case["exit"].as_i64().unwrap(),
            blank_scan(case["stdout"].as_str().unwrap()),
            case["stderr"].as_str().unwrap().to_owned(),
        );
        let verdict = match R7_USAGE_REFUSED.iter().find(|(n, _, _)| *n == name) {
            Some((_, text, python_accepted)) => {
                exit == 2
                    && stdout.is_empty()
                    && stderr.starts_with("error: ")
                    && stderr.contains(text)
                    && (python.0 == 0) == *python_accepted
            }
            None => (exit, &stdout, &stderr) == (python.0, &python.1, &python.2),
        };
        if !verdict {
            failures.push(format!(
                "{name}: the Python {python:?}, the port {:?}",
                (exit, stdout, stderr)
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Round six reports the port fails at an earlier step than the Python (stricter): a `HOME` that is
/// empty, relative or not UTF-8 (the Python took `/`, the working directory's relative path, or a
/// surrogate-escaped name, and went on to the configuration), and a checksum key that is not UTF-8
/// (a configuration reached through a directory named with such a byte, and this executable in
/// one), which the Python recorded surrogate-escaped. Each is a failure receipt whose error begins
/// with the text given; the Python's failed later, at the harness's approved roots.
const R6_REPORT_STRICTER: &[(&str, &str)] = &[
    (
        "units-r6-home-relative",
        "HOME 'rel/home' is not an absolute path",
    ),
    ("units-r6-home-empty", "HOME '' is not an absolute path"),
    ("units-r6-home-not-utf8", "HOME is not UTF-8"),
    (
        "custody-r6-config-via-not-utf8-link",
        r"checksum key '@C@/\udcff/cfg.json' is not UTF-8",
    ),
    (
        "custody-r6-tools-in-not-utf8-dir",
        r"checksum key '@C@/\udcff/tools/jope-ci' is not UTF-8",
    ),
];

/// Round six custody whose resolved name is not UTF-8. The Python wrote its receipt, then failed
/// printing the receipt's path (`en_US.UTF-8`: `report failed: 'utf-8' codec can't encode ...`) or
/// printed its raw bytes (`C.UTF-8`); the port refuses with that message before creating anything
/// (stricter in writing nothing, and under `C.UTF-8`). The case whose message is used.
const R6_CUSTODY_NOTHING_WRITTEN: &[(&str, &str)] = &[
    ("custody-r6-link-not-utf8-en", "custody-r6-link-not-utf8-en"),
    ("custody-r6-link-not-utf8-cu", "custody-r6-link-not-utf8-en"),
    (
        "custody-r6-link-not-utf8-good-config-en",
        "custody-r6-link-not-utf8-good-config-en",
    ),
    (
        "custody-r6-relative-from-not-utf8-cwd-en",
        "custody-r6-relative-from-not-utf8-cwd-en",
    ),
];

/// Round six custody arguments that are not UTF-8, refused with the other arguments before
/// anything is read (exit status 2); the Python failed printing a resolved name that is not UTF-8,
/// or resolved `<0xff>/..` to a UTF-8 custody and ran.
const R6_REPORT_ARGV: &[&str] = &[
    "custody-r6-argument-not-utf8-en",
    "custody-r6-argument-not-utf8-dotdot-cu",
];

/// Round six (`negative-r6.json`, kinds `units` and `custody`): `HOME` beginning with `~` fails the
/// receipt with the `RuntimeError` Python 3.12's `Path.home()` raised (`Could not determine home
/// directory.`; the port read it as a relative directory and, given a configuration it accepts,
/// succeeded), `/` and `<dir>/~x` are read as the Python read them; custody, configuration and
/// tools behind names that are not UTF-8 refuse (see the lists above). Same environment as the other
/// receipt replays; a `good` case gets its `a` and `b` as the approved roots, a `bad` one the
/// wrapper's.
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn reports_match_round_six() {
    let doc = round_six();
    let scratch = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    )
    .join("python-parity-r6");
    fs::create_dir_all(scratch.parent().unwrap()).unwrap();
    fs::create_dir(&scratch).expect("a fresh scratch directory");
    let scratch = scratch.canonicalize().unwrap();
    let tools = repo().join("deploy/maintenance");
    let cases: Vec<&Value> = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "units" || c["kind"] == "custody")
        .collect();
    let named = |name: &str| *cases.iter().find(|c| c["name"] == name).unwrap();
    let mut failures = vec![];
    for &case in &cases {
        let name = case["name"].as_str().unwrap();
        let c = scratch.join(name);
        for dir in ["home/.config/systemd/user", "a", "b"] {
            fs::create_dir_all(c.join(dir)).unwrap();
        }
        let c_text = c.to_str().unwrap().to_owned();
        let sub = |text: &str| unescape(&text.replace("@C@", &c_text));
        for dir in case["dirs"].as_array().into_iter().flatten() {
            fs::create_dir_all(c.join(unescape(dir.as_str().unwrap()))).unwrap();
        }
        for link in case["symlinks"].as_array().into_iter().flatten() {
            symlink(
                unescape(link["target"].as_str().unwrap()),
                c.join(link["path"].as_str().unwrap()),
            )
            .unwrap();
        }
        let good = case["config"] == "good";
        let (config, roots) = if good {
            let roots = vec![format!("{c_text}/a"), format!("{c_text}/b")];
            (
                format!(
                    r#"{{"cargo_roots": ["{}", "{}"], "stale_hours": 48, "measurement_roots": []}}"#,
                    roots[0], roots[1]
                ),
                roots,
            )
        } else {
            (
                doc["bad_config"].as_str().unwrap().to_owned(),
                approved_roots(),
            )
        };
        let at = c.join(unescape(
            case.get("config_at")
                .and_then(Value::as_str)
                .unwrap_or("cfg.json"),
        ));
        fs::create_dir_all(at.parent().unwrap()).unwrap();
        fs::write(&at, config).unwrap();
        let (program, tools_args): (PathBuf, Vec<std::ffi::OsString>) = if case["tools"] == "ff" {
            // The executable and its wrappers in a directory named with the byte 0xff: the tools
            // directory is the executable's (no `--tools-dir`).
            let dir = c.join(unescape("@Xff@/tools"));
            fs::create_dir_all(&dir).unwrap();
            fs::copy(env!("CARGO_BIN_EXE_jope-ci"), dir.join("jope-ci")).unwrap();
            for wrapper in ["cargo-build-dirs.sh", "jope-build-report.sh"] {
                fs::copy(tools.join(wrapper), dir.join(wrapper)).unwrap();
            }
            (dir.join("jope-ci"), vec![])
        } else {
            (
                PathBuf::from(env!("CARGO_BIN_EXE_jope-ci")),
                vec!["--tools-dir".into(), tools.clone().into_os_string()],
            )
        };
        let mut command = Command::new(&program);
        command
            .args(["build-report", "report", "--config"])
            .arg(&at)
            .arg("--custody")
            .arg(sub(case["custody"].as_str().unwrap()))
            .args(&tools_args);
        for root in &roots {
            command.arg("--approved-root").arg(root);
        }
        let home = sub(case
            .get("home")
            .and_then(Value::as_str)
            .unwrap_or("@C@/home"));
        let cwd = PathBuf::from(sub(case
            .get("cwd")
            .and_then(Value::as_str)
            .unwrap_or("@C@")));
        command
            .current_dir(&cwd)
            .env("HOME", &home)
            .env_remove("INVOCATION_ID");
        match case.get("locale").and_then(Value::as_str) {
            Some("en") => {
                command
                    .env_remove("LC_ALL")
                    .env_remove("LC_CTYPE")
                    .env_remove("LC_MESSAGES")
                    .env("LANG", "en_US.UTF-8");
            }
            _ => {
                command.env("LC_ALL", "C.UTF-8");
            }
        }
        let output = command.output().unwrap();
        let exit = i64::from(output.status.code().unwrap());
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let mut written = vec![];
        receipts_below(&c, &mut written);
        let stamp = written
            .first()
            .and_then(|r| r.parent())
            .and_then(Path::file_name)
            .and_then(|s| s.to_str())
            .map(str::to_owned);
        let normalise = |text: &str| {
            let text = text.replace(&c_text, "@C@");
            match &stamp {
                Some(stamp) => text.replace(stamp.as_str(), "@STAMP@"),
                None => text,
            }
        };
        let kept = written.first().map(|path| {
            let receipt: Value =
                serde_json::from_str(&normalise(&fs::read_to_string(path).unwrap())).unwrap();
            let kept: Value = ["status", "exit_status", "error", "missing_unit_files"]
                .iter()
                .map(|k| {
                    (
                        k.to_string(),
                        receipt.get(*k).cloned().unwrap_or(Value::Null),
                    )
                })
                .collect::<serde_json::Map<_, _>>()
                .into();
            kept
        });
        let (stdout, stderr) = (normalise(&stdout), normalise(&stderr));
        let receipt_line = "@C@/receipts/@STAMP@/receipt.json\n";
        let why = if let Some((_, refusal)) = R6_REPORT_STRICTER.iter().find(|(n, _)| *n == name) {
            let kept = kept.clone().unwrap_or(Value::Null);
            let refused = exit == 1
                && stdout == receipt_line
                && written.len() == 1
                && kept["status"] == "failed"
                && kept["error"]
                    .as_str()
                    .is_some_and(|e| e.starts_with(refusal))
                && case["exit"] == 1
                && case["receipt"]["error"]
                    == "service requires the two approved Cargo roots and a 48-hour filter";
            (!refused).then(|| format!("listed as stricter: {exit} {stdout:?} {stderr:?} {kept}"))
        } else if let Some((_, message)) =
            R6_CUSTODY_NOTHING_WRITTEN.iter().find(|(n, _)| *n == name)
        {
            let want = place_positions(
                named(message)["stderr"].as_str().unwrap(),
                "C",
                c_text.chars().count(),
            );
            let refused = exit == 1 && stdout.is_empty() && stderr == want && written.is_empty();
            let python_wrote = case["receipts_written"] == 1;
            (!(refused && python_wrote)).then(|| {
                format!(
                    "refusal {exit} {stdout:?} {stderr:?} ({} receipts), want {want:?} and none",
                    written.len()
                )
            })
        } else if R6_REPORT_ARGV.contains(&name) {
            let refused = exit == 2
                && stdout.is_empty()
                && stderr.starts_with("jope-ci: argument ")
                && stderr.contains(" is not UTF-8")
                && written.is_empty();
            (!refused).then(|| format!("argument refusal: {exit} {stdout:?} {stderr:?}"))
        } else {
            let want_stderr = place_positions(
                case["stderr"].as_str().unwrap(),
                "C",
                c_text.chars().count(),
            );
            let same = exit == case["exit"].as_i64().unwrap()
                && Some(stdout.as_str()) == case["stdout"].as_str()
                && stderr == want_stderr
                && kept.clone().unwrap_or(Value::Null) == case["receipt"];
            (!same).then(|| {
                format!(
                    "the Python {} {:?} {want_stderr:?} {}, the port {exit} {stdout:?} {stderr:?} {kept:?}",
                    case["exit"], case["stdout"], case["receipt"]
                )
            })
        };
        if let Some(why) = why {
            failures.push(format!("{name}: {why}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert_eq!(cases.len(), 20);
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
        // Round five: more directories, symlinks (`@ROOT@` in a target replaced), and the modes
        // the case ran under, set back to 755 after the run.
        for dir in case["dirs"].as_array().into_iter().flatten() {
            fs::create_dir_all(root.join(dir.as_str().unwrap())).unwrap();
        }
        for link in case["symlinks"].as_array().into_iter().flatten() {
            let target = link["target"]
                .as_str()
                .unwrap()
                .replace("@ROOT@", root_text);
            symlink(target, root.join(link["path"].as_str().unwrap())).unwrap();
        }
        let modes = |restore: bool| {
            for entry in case["modes"].as_array().into_iter().flatten() {
                let mode = if restore {
                    0o755
                } else {
                    u32::from_str_radix(entry["mode"].as_str().unwrap(), 8).unwrap()
                };
                let path = root.join(entry["path"].as_str().unwrap());
                fs::set_permissions(path, fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        let args: Vec<String> = case["args"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a.as_str().unwrap().replace("@ROOT@", root_text))
            .collect();
        let mut argv = vec!["scan"];
        argv.extend(args.iter().map(String::as_str));
        modes(false);
        // A user who may search anything (root) cannot reproduce a refused search.
        let privileged = case["modes"].as_array().into_iter().flatten().any(|entry| {
            let dir = root.join(entry["path"].as_str().unwrap());
            fs::read_dir(&dir).is_ok_and(|mut entries| {
                entries.all(|e| e.is_ok_and(|e| fs::symlink_metadata(e.path()).is_ok()))
            })
        });
        let output = jope_ci(&argv);
        modes(true);
        if privileged {
            eprintln!("{name}: skipped, this user may search a directory of that mode");
            continue;
        }
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

/// Round four (`build-report/negative-r4.json`, review 5334300806 of PR #117): custody resolved as
/// `Path.resolve()` resolved it, each symlink before a later `..`, then refused under /tmp or a
/// Git working tree before anything is written. The fixture's synthetic layout (links from
/// `outer/` into `guarded/`, whose `.git` is an empty directory, and into `guarded2/`, whose `.git`
/// is a file; a self loop; a dangling link; a directory that may not be searched) is rebuilt for
/// each case, and `report` runs with a configuration that names wrong Cargo roots, so a custody
/// the Python accepted ends in the same failure receipt and nothing is scanned. The exit status,
/// stdout, stderr, the receipt (with `custody_resolved`), `failure.txt` and every path the run
/// created are compared; a symlink loop, which the Python raised as an uncaught `RuntimeError`,
/// is refused with that exception's message. Same environment as the tests above.
#[test]
#[ignore = "needs JOPE_MAINTENANCE_SCRATCH outside /tmp and Git working trees"]
fn custody_matches_round_four() {
    use std::collections::BTreeSet;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/python-parity/build-report/negative-r4.json");
    let doc: Value = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 22, "every captured case is replayed");
    let scratch = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    )
    .join("python-parity-custody-r4");
    fs::create_dir_all(scratch.parent().unwrap()).unwrap();
    fs::create_dir(&scratch).expect("a fresh scratch directory");
    let scratch = scratch.canonicalize().unwrap();
    let roots = approved_roots();
    let tools = repo().join("deploy/maintenance");
    let layout = &doc["layout"];
    let no_search: Vec<&str> = layout["no_search"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_str().unwrap())
        .collect();
    // Every path under `c` as `find . -path ./<no_search> -prune -o -print` lists it.
    fn listing(c: &Path, dir: &Path, skip: &[&str], out: &mut BTreeSet<String>) {
        let rel = dir.strip_prefix(c).unwrap();
        if skip.iter().any(|s| rel == Path::new(s)) {
            return;
        }
        out.insert(
            format!("./{}", rel.display())
                .trim_end_matches('/')
                .to_owned(),
        );
        let meta = fs::symlink_metadata(dir).unwrap();
        if meta.is_dir() {
            for entry in fs::read_dir(dir).unwrap() {
                listing(c, &entry.unwrap().path(), skip, out);
            }
        }
    }
    let mut failures = vec![];
    let mut replayed = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let c = scratch.join(name);
        for dir in layout["dirs"].as_array().unwrap() {
            fs::create_dir_all(c.join(dir.as_str().unwrap())).unwrap();
        }
        for (file, text) in layout["files"].as_object().unwrap() {
            fs::write(c.join(file), text.as_str().unwrap()).unwrap();
        }
        let c_text = c.to_str().unwrap().to_owned();
        for (link, target) in layout["symlinks"].as_object().unwrap() {
            symlink(
                target.as_str().unwrap().replace("@C@", &c_text),
                c.join(link),
            )
            .unwrap();
        }
        for dir in &no_search {
            fs::set_permissions(c.join(dir), fs::Permissions::from_mode(0o000)).unwrap();
        }
        // A user who may search anything (root) cannot reproduce a refused search.
        let searchable = fs::metadata(c.join(no_search[0]).join("x")).is_ok();
        let config = c.join("cfg.json");
        fs::write(&config, doc["config"].as_str().unwrap()).unwrap();
        let mut before = BTreeSet::new();
        listing(&c, &c, &no_search, &mut before);
        let custody = case["custody"].as_str().unwrap().replace("@C@", &c_text);
        let output = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .args(["build-report", "report", "--config"])
            .arg(&config)
            .args(["--custody", &custody, "--tools-dir"])
            .arg(&tools)
            .args(["--approved-root", &roots[0], "--approved-root", &roots[1]])
            .current_dir(c.join(case["cwd"].as_str().unwrap()))
            .env_remove("INVOCATION_ID")
            .output()
            .unwrap();
        for dir in &no_search {
            fs::set_permissions(c.join(dir), fs::Permissions::from_mode(0o755)).unwrap();
        }
        if searchable && name == "custody-permission-denied" {
            eprintln!("{name}: skipped, this user may search a directory of mode 000");
            continue;
        }
        let mut after = BTreeSet::new();
        listing(&c, &c, &no_search, &mut after);
        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stamp = stdout
            .rsplit('/')
            .nth(1)
            .filter(|s| s.ends_with('Z') && s.len() == 23)
            .map(str::to_owned);
        let normalise = |text: &str| {
            let text = text.replace(&c_text, "@C@");
            match &stamp {
                Some(stamp) => text.replace(stamp.as_str(), "@STAMP@"),
                None => text,
            }
        };
        let stderr = normalise(&String::from_utf8_lossy(&output.stderr));
        let mut why = vec![];
        if i64::from(output.status.code().unwrap()) != case["exit"].as_i64().unwrap() {
            why.push(format!("exit {:?}", output.status.code()));
        }
        if normalise(&stdout) != case["stdout"].as_str().unwrap() {
            why.push(format!("stdout {stdout:?}"));
        }
        let want_stderr = match case["raised"].as_str() {
            Some(raised) => format!(
                "report failed: {}\n",
                raised
                    .strip_prefix("RuntimeError: ")
                    .expect("a RuntimeError")
            ),
            None => case["stderr"].as_str().unwrap().to_owned(),
        };
        if stderr != want_stderr {
            why.push(format!("stderr {stderr:?}, the Python {want_stderr:?}"));
        }
        let created: Vec<String> = after.difference(&before).map(|p| normalise(p)).collect();
        let want_created: Vec<&str> = case["created"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_str().unwrap())
            .collect();
        if created != want_created {
            why.push(format!("created {created:?}, the Python {want_created:?}"));
        }
        if !case["receipt"].is_null() {
            let run = Path::new(stdout.trim_end()).parent().unwrap();
            let text = fs::read_to_string(run.join("receipt.json")).unwrap();
            let mut receipt: Value = serde_json::from_str(&normalise(&text)).unwrap();
            for key in [
                "started_at",
                "completed_at",
                "checksums",
                "missing_unit_files",
            ] {
                receipt.as_object_mut().unwrap().remove(key);
            }
            if receipt != case["receipt"] {
                why.push(format!("receipt {receipt}"));
            }
            let failure = fs::read_to_string(run.join("failure.txt")).unwrap();
            if failure != case["failure_txt"].as_str().unwrap() {
                why.push(format!("failure.txt {failure:?}"));
            }
        }
        if !why.is_empty() {
            failures.push(format!("{name}: {}", why.join("; ")));
        }
        replayed += 1;
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(replayed >= 21);
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
