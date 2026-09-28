//! `jope-ci build-report`: fixture runners ported from the retired
//! `deploy/maintenance/test_report.py` and `test_wrapper.py` (archived at 8a55c4fef).
//!
//! Like the Python runners they need an explicit scratch directory outside `/tmp` and outside
//! any Git working tree (report custody refuses both), so they are ignored by default:
//!
//! ```sh
//! JOPE_MAINTENANCE_SCRATCH=<new dir> cargo test -p jope-ci --test build_report -- --ignored
//! ```
//!
//! `JOPE_MAINTENANCE_HELPER` may name an installed `cargo-build-dirs.sh` to test instead of the
//! repository copy wired to this build. The scratch directory keeps the fixtures, their
//! before/after inventories and `results.json` as evidence.

use jope_ci::build_report::{allocated, never_interrupted, report, scan, Failure, Hooks, Outcome};
use jope_records::pyjson::{self, Value};
use std::collections::BTreeMap;
use std::fs;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A fresh scratch directory under `JOPE_MAINTENANCE_SCRATCH`, refusing `/tmp`.
fn scratch(name: &str) -> PathBuf {
    let base = PathBuf::from(
        std::env::var("JOPE_MAINTENANCE_SCRATCH")
            .expect("set JOPE_MAINTENANCE_SCRATCH to a directory outside /tmp and working trees"),
    );
    let base = jope_ci::build_report::abspath(&base).unwrap();
    assert!(!base.starts_with("/tmp"), "fixtures must be outside /tmp");
    let dir = base.join(name);
    fs::create_dir_all(dir.parent().unwrap()).unwrap();
    fs::create_dir(&dir).expect("a fresh scratch directory");
    dir.canonicalize().unwrap()
}

/// A tools directory holding the repository wrappers and this build's `jope-ci`, the installed
/// layout (`~/lane-tools`).
fn tools(scratch: &Path) -> PathBuf {
    let tools = scratch.join("tools");
    fs::create_dir(&tools).unwrap();
    for wrapper in ["cargo-build-dirs.sh", "jope-build-report.sh"] {
        fs::copy(
            repo().join("deploy/maintenance").join(wrapper),
            tools.join(wrapper),
        )
        .unwrap();
    }
    fs::copy(env!("CARGO_BIN_EXE_jope-ci"), tools.join("jope-ci")).unwrap();
    tools
}

/// The approved Cargo roots, read from the repository wrapper that passes them (`--approved-root`,
/// host layout kept out of crates/ by ADR-0003).
fn approved_roots() -> Vec<String> {
    let wrapper =
        fs::read_to_string(repo().join("deploy/maintenance/jope-build-report.sh")).unwrap();
    let words: Vec<&str> = wrapper.split_whitespace().collect();
    let roots: Vec<String> = words
        .windows(2)
        .filter(|pair| pair[0] == "--approved-root")
        .map(|pair| pair[1].to_owned())
        .collect();
    assert_eq!(
        roots.len(),
        2,
        "the wrapper passes exactly two approved roots"
    );
    roots
}

fn touch_old(path: &Path, epoch: u64) {
    let status = Command::new("touch")
        .args(["-h", "-d", &format!("@{epoch}")])
        .arg(path)
        .status()
        .unwrap();
    assert!(status.success());
}

/// Mode, mtime and content (file hash or link target) of everything under `root`.
fn inventory(root: &Path) -> BTreeMap<String, (u32, i64, i64, String)> {
    fn walk(root: &Path, path: &Path, out: &mut BTreeMap<String, (u32, i64, i64, String)>) {
        let meta = fs::symlink_metadata(path).unwrap();
        let content = if meta.file_type().is_symlink() {
            fs::read_link(path).unwrap().to_string_lossy().into_owned()
        } else if meta.is_file() {
            format!("{:x}", md5_like(&fs::read(path).unwrap()))
        } else {
            String::new()
        };
        out.insert(
            path.strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            (meta.mode(), meta.mtime(), meta.mtime_nsec(), content),
        );
        if meta.is_dir() && meta.permissions().mode() & 0o500 == 0o500 {
            let mut entries: Vec<PathBuf> = fs::read_dir(path)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for entry in entries {
                walk(root, &entry, out);
            }
        }
    }
    let mut out = BTreeMap::new();
    walk(root, root, &mut out);
    out
}

/// A content fingerprint (FNV-1a 64) for the preservation inventory.
fn md5_like(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

struct Call {
    code: i32,
    stdout: String,
    stderr: String,
}

fn call(helper: &Path, args: &[&Path]) -> Call {
    let output = Command::new(helper).args(args).output().unwrap();
    Call {
        code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
    }
}

/// Exercise candidate selection, refusals and failure visibility under an explicit custody
/// directory, and prove every fixture byte, link, mode and mtime unchanged.
#[test]
#[ignore = "fixture runner: needs JOPE_MAINTENANCE_SCRATCH outside /tmp and working trees"]
fn scan_preserves_fixtures_and_fails_visibly() {
    let scratch = scratch("scan");
    let tools = tools(&scratch);
    let helper = std::env::var("JOPE_MAINTENANCE_HELPER")
        .map(PathBuf::from)
        .unwrap_or_else(|_| tools.join("cargo-build-dirs.sh"));
    let root = scratch.join("fixtures");
    fs::create_dir(&root).unwrap();
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
        "odd\nname/.rustc_info.json",
        "gitguard/.rustc_info.json",
        "gitguard/.git/objects/pack",
    ] {
        let path = root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"\x00preserve\xff\n").unwrap();
    }
    let outside = scratch.join("outside");
    fs::create_dir(&outside).unwrap();
    fs::write(outside.join(".rustc_info.json"), "{}").unwrap();
    std::os::unix::fs::symlink(&outside, root.join("linked")).unwrap();
    fs::create_dir(root.join("marker-link")).unwrap();
    std::os::unix::fs::symlink(
        outside.join(".rustc_info.json"),
        root.join("marker-link/.rustc_info.json"),
    )
    .unwrap();
    fs::create_dir_all(root.join("relsym/release")).unwrap();
    std::os::unix::fs::symlink(&outside, root.join("relsym/release/.fingerprint")).unwrap();
    let locked = root.join("inaccessible");
    fs::create_dir(&locked).unwrap();
    let old = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
        - 72 * 3600;
    let mut everything = vec![root.clone()];
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).unwrap() {
            let path = entry.unwrap().path();
            everything.push(path.clone());
            if fs::symlink_metadata(&path).unwrap().is_dir() {
                stack.push(path);
            }
        }
    }
    everything.sort_by_key(|p| std::cmp::Reverse(p.components().count()));
    for path in &everything {
        touch_old(path, old);
    }
    for recent in [
        "fresh/debug/recent",
        "recent-nested/nested/.rustc_info.json",
    ] {
        let status = Command::new("touch")
            .arg(root.join(recent))
            .status()
            .unwrap();
        assert!(status.success());
    }
    let before = inventory(&root);
    let mut results: Vec<Value> = vec![];
    let mut record = |label: &str, c: &Call| {
        results.push(Value::Dict(
            [
                ("test", Value::from(label)),
                ("exit", Value::from(i64::from(c.code))),
                ("stdout", Value::from(c.stdout.clone())),
                ("stderr", Value::from(c.stderr.clone())),
            ]
            .into_iter()
            .collect(),
        ));
    };

    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    assert_eq!(c.code, 0, "{}", c.stderr);
    record("stale and marker selection", &c);
    let doc = pyjson::parse_str(&c.stdout).unwrap();
    let names: std::collections::BTreeSet<String> = doc
        .get("candidates")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|r| {
            Path::new(r.get("path").and_then(Value::as_str).unwrap())
                .file_name()
                .unwrap()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    assert_eq!(
        names,
        ["stale", "fingerprint", "odd\nname", "gitguard"]
            .map(String::from)
            .into(),
        "nested Cargo output must not be double-counted"
    );
    let c = call(&helper, &[&root.join("fingerprint/release")]);
    assert_eq!(c.code, 0, "{}", c.stderr);
    record("fingerprint cannot escape supplied root", &c);
    assert_eq!(
        pyjson::parse_str(&c.stdout).unwrap().get("candidates"),
        Some(&Value::List(vec![]))
    );
    let missing = scratch.join("missing");
    let linked = root.join("linked");
    for (label, args, expected) in [
        (
            "delete rejected",
            vec![root.as_path(), Path::new("--delete")],
            2,
        ),
        (
            "delete before root validation",
            vec![missing.as_path(), Path::new("--delete")],
            2,
        ),
        ("missing age", vec![root.as_path(), Path::new("--stale")], 2),
        (
            "invalid age",
            vec![root.as_path(), Path::new("--stale"), Path::new("wat")],
            2,
        ),
        (
            "negative age",
            vec![root.as_path(), Path::new("--stale"), Path::new("-1")],
            2,
        ),
        ("missing root", vec![missing.as_path()], 1),
        ("symlink root", vec![linked.as_path()], 1),
    ] {
        let c = call(&helper, &args);
        assert_eq!(c.code, expected, "{label}: {}", c.stderr);
        assert!(
            !c.stderr.is_empty() && !c.stdout.contains("\"complete\": true"),
            "{label}"
        );
        record(label, &c);
    }
    // Traversal failure: an unreadable directory fails the scan instead of being skipped.
    // Permission changes leave mtimes alone; each mode is restored exactly.
    let mode_of = |p: &Path| fs::symlink_metadata(p).unwrap().permissions().mode() & 0o7777;
    let locked_mode = mode_of(&locked);
    fs::set_permissions(&locked, fs::Permissions::from_mode(0o000)).unwrap();
    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    fs::set_permissions(&locked, fs::Permissions::from_mode(locked_mode)).unwrap();
    assert!(
        c.code != 0 && !c.stdout.contains("\"complete\": true"),
        "traversal failure was suppressed (running as root?)"
    );
    record("traversal failure", &c);
    // A descendant the candidate search never enters (.git) still fails the mtime walk.
    let git_objects = root.join("gitguard/.git/objects");
    let objects_mode = mode_of(&git_objects);
    fs::set_permissions(&git_objects, fs::Permissions::from_mode(0o000)).unwrap();
    let c = call(&helper, &[&root, Path::new("--stale"), Path::new("48")]);
    fs::set_permissions(&git_objects, fs::Permissions::from_mode(objects_mode)).unwrap();
    assert!(
        c.code != 0,
        "candidate mtime traversal failure was suppressed"
    );
    record("descendant mtime traversal failure", &c);
    // Measurement failures (the scan is the library's; the measurement is injected).
    for (label, failure) in [
        (
            "measurement failure",
            Failure::new("Command '['du']' returned non-zero exit status 1."),
        ),
        (
            "measurement timeout",
            Failure::new("Command '['du']' timed out after 600 seconds"),
        ),
    ] {
        let outcome = scan(&root, Some(48), &never_interrupted, &|_, _| {
            Err(failure.clone())
        });
        assert_eq!(outcome.unwrap_err(), failure, "{label}");
    }
    // A real failing du (a vanished path) is an error, never a number.
    assert!(allocated(&scratch.join("absent"), false, &never_interrupted).is_err());
    let after = inventory(&root);
    assert_eq!(
        before, after,
        "fixture contents, membership, links, modes or mtimes changed"
    );
    // The report wrapper writes a failure receipt for a refused configuration.
    let reports = scratch.join("reports");
    fs::create_dir(&reports).unwrap();
    let config = scratch.join("invalid-config.json");
    fs::write(
        &config,
        r#"{"cargo_roots": [], "stale_hours": 48, "measurement_roots": []}"#,
    )
    .unwrap();
    let wrapper = helper.with_file_name("jope-build-report.sh");
    let c = call(
        &wrapper,
        &[
            Path::new("--config"),
            &config,
            Path::new("--custody"),
            &reports,
        ],
    );
    assert_ne!(c.code, 0);
    let run = fs::read_dir(&reports)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let receipt =
        pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
    assert_eq!(receipt.get("status"), Some(&Value::from("failed")));
    assert_eq!(receipt.get("exit_status"), Some(&Value::Int(1)));
    assert!(!fs::read_to_string(run.join("failure.txt"))
        .unwrap()
        .is_empty());
    fs::write(
        scratch.join("results.json"),
        pyjson::dumps(
            &Value::Dict(
                [
                    ("tests", Value::List(results)),
                    ("wrapper_failure", receipt),
                    ("fixtures", Value::from(before.len())),
                    ("preservation", Value::from("exact unchanged file/directory membership, contents, symlinks, modes and mtimes")),
                ]
                .into_iter()
                .collect(),
            ),
            false,
            Some(2),
        ),
    )
    .unwrap();
}

fn fixture_scan_doc(root: &str) -> Value {
    Value::Dict(
        [
            ("root", Value::from(root)),
            ("complete", Value::from(true)),
            ("report_only", Value::from(true)),
            ("candidates", Value::List(vec![])),
        ]
        .into_iter()
        .collect(),
    )
}

/// Configuration, success, overlap, measurement, interruption and failure receipts.
#[test]
#[ignore = "fixture runner: needs JOPE_MAINTENANCE_SCRATCH outside /tmp and working trees"]
fn report_receipts_record_every_outcome() {
    let scratch = scratch("report");
    let tools = tools(&scratch);
    let measured = scratch.join("measured");
    fs::create_dir_all(measured.join("child")).unwrap();
    fs::write(measured.join("evidence.bin"), b"\x00preserve\xff").unwrap();
    let row = |path: &Path| {
        format!(
            r#"{{"group": "fixture", "path": {}}}"#,
            pyjson::dumps(
                &Value::from(path.to_string_lossy().into_owned()),
                false,
                None
            )
        )
    };
    let approved_roots = approved_roots();
    let approved = pyjson::dumps(
        &Value::List(
            approved_roots
                .iter()
                .map(|r| Value::from(r.as_str()))
                .collect(),
        ),
        false,
        None,
    );
    let config = |rows: &[String]| {
        format!(
            r#"{{"cargo_roots": {approved}, "stale_hours": 48, "measurement_roots": [{}]}}"#,
            rows.join(", ")
        )
    };
    let base = config(&[row(&measured)]);
    let mut cases = vec![];
    let mut exercise = |name: &str,
                        value: &str,
                        expected: i32,
                        privilege_denied: bool,
                        scan_failure: bool| {
        let directory = scratch.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        fs::write(&path, value).unwrap();
        let invocations = std::cell::RefCell::new(0usize);
        let scan_hook = |root: &str, _: jope_ci::build_report::Interrupt<'_>| {
            assert!(approved_roots.iter().any(|a| a == root));
            if scan_failure {
                return Err(Failure::new("fixture scan traversal denied"));
            }
            Ok(fixture_scan_doc(root))
        };
        let measure_hook = |p: &Path, privileged: bool, i: jope_ci::build_report::Interrupt<'_>| {
            *invocations.borrow_mut() += 1;
            if privileged && privilege_denied {
                return Err(Failure {
                    message: "Command '['/usr/bin/sudo', '-n', '/usr/bin/du']' returned non-zero exit status 1.".into(),
                    stderr: Some("sudo denied (fixture)".into()),
                });
            }
            // A granted privilege is delegated to the unprivileged measurement.
            allocated(p, false, i)
        };
        let hooks = Hooks {
            scan: &scan_hook,
            measure: &measure_hook,
        };
        let code = report(
            &path,
            &directory.join("reports"),
            &tools,
            &approved_roots,
            &hooks,
            &never_interrupted,
        )
        .unwrap();
        let run = fs::read_dir(directory.join("reports"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt =
            pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
        let exe = std::env::current_exe().unwrap().canonicalize().unwrap();
        assert!(
            receipt
                .get("checksums")
                .and_then(|c| c.get(&exe.to_string_lossy()))
                .is_some(),
            "{name}: executable checksum"
        );
        assert_eq!(code, expected, "{name}");
        assert_eq!(
            receipt.get("exit_status"),
            Some(&Value::Int(i64::from(expected))),
            "{name}"
        );
        assert_eq!(
            receipt.get("status"),
            Some(&Value::from(if expected == 0 {
                "success"
            } else {
                "failed"
            })),
            "{name}"
        );
        if expected != 0 {
            assert!(
                !fs::read_to_string(run.join("failure.txt"))
                    .unwrap()
                    .is_empty(),
                "{name}"
            );
        } else {
            let first = &receipt
                .get("measurements")
                .and_then(Value::as_list)
                .unwrap()[0];
            assert!(
                first
                    .get("allocated_bytes")
                    .and_then(Value::as_int)
                    .unwrap()
                    > 0
            );
            let outputs = receipt
                .get("scan_outputs")
                .and_then(Value::as_list)
                .unwrap();
            assert_eq!(outputs.len(), 2);
            assert!(!fs::read_to_string(run.join("filesystem.txt"))
                .unwrap()
                .is_empty());
            for output in outputs {
                let doc = fs::read_to_string(
                    run.join(output.get("output").and_then(Value::as_str).unwrap()),
                )
                .unwrap();
                assert_eq!(
                    pyjson::parse_str(&doc).unwrap().get("complete"),
                    Some(&Value::from(true))
                );
                assert_eq!(
                    fs::read_to_string(
                        run.join(output.get("stderr").and_then(Value::as_str).unwrap())
                    )
                    .unwrap(),
                    ""
                );
            }
        }
        let count = *invocations.borrow();
        cases.push((name.to_owned(), run.clone(), receipt.clone(), count));
        (run, receipt, count)
    };
    exercise("success", &base, 0, false, false);
    for (name, value) in [
        ("malformed", "{".to_owned()),
        ("missing-key", "{}".to_owned()),
        ("non-object", "[]".to_owned()),
        (
            "wrong-rows",
            format!(
                r#"{{"cargo_roots": {approved}, "stale_hours": 48, "measurement_roots": {{}}}}"#
            ),
        ),
        ("duplicate", config(&[row(&measured), row(&measured)])),
        (
            "nested",
            config(&[row(&measured), row(&measured.join("child"))]),
        ),
    ] {
        exercise(name, &value, 1, false, false);
    }
    for (name, rows) in [
        (
            "missing-measurement-root",
            vec![row(&scratch.join("absent"))],
        ),
        (
            "valid-then-missing-measurement-root",
            vec![row(&measured), row(&scratch.join("absent"))],
        ),
    ] {
        let (run, receipt, invocations) = exercise(name, &config(&rows), 1, false, false);
        // Validate every root before any du; preserve scans even if a later root is absent.
        assert_eq!(
            receipt.get("measurements"),
            Some(&Value::List(vec![])),
            "{name}"
        );
        assert_eq!(invocations, 0, "{name}");
        let outputs = receipt
            .get("scan_outputs")
            .and_then(Value::as_list)
            .unwrap();
        assert_eq!(outputs.len(), 2);
        for output in outputs {
            let doc =
                fs::read_to_string(run.join(output.get("output").and_then(Value::as_str).unwrap()))
                    .unwrap();
            assert_eq!(
                pyjson::parse_str(&doc).unwrap().get("complete"),
                Some(&Value::from(true))
            );
        }
    }
    let privileged = config(&[format!(
        r#"{{"group": "fixture", "path": {}, "privileged": true}}"#,
        pyjson::dumps(
            &Value::from(measured.to_string_lossy().into_owned()),
            false,
            None
        )
    )]);
    exercise("privileged-command", &privileged, 0, false, false);
    exercise("privilege-denied", &privileged, 1, true, false);
    let (run, receipt, _) = exercise("scan-failure", &base, 1, false, true);
    let stderr_name = receipt
        .get("scan_outputs")
        .and_then(Value::as_list)
        .unwrap()[0]
        .get("stderr")
        .and_then(Value::as_str)
        .unwrap()
        .to_owned();
    assert!(fs::read_to_string(run.join(stderr_name))
        .unwrap()
        .contains("traversal denied"));

    // The wrapper's approved list is part of the fence: a single approved root, or a
    // configuration naming only one of the two, is refused before any scan or measurement.
    let lone = format!(
        r#"{{"cargo_roots": {}, "stale_hours": 48, "measurement_roots": [{}]}}"#,
        pyjson::dumps(
            &Value::List(vec![Value::from(approved_roots[0].as_str())]),
            false,
            None
        ),
        row(&measured)
    );
    for (name, list) in [
        ("one-approved-root", approved_roots[..1].to_vec()),
        ("config-names-one-root", approved_roots.clone()),
    ] {
        let directory = scratch.join(name);
        fs::create_dir(&directory).unwrap();
        let path = directory.join("config.json");
        fs::write(&path, &lone).unwrap();
        let scan_hook = |_: &str, _: jope_ci::build_report::Interrupt<'_>| -> Outcome<Value> {
            panic!("{name}: scanned despite the refusal")
        };
        let measure_hook =
            |_: &Path, _: bool, _: jope_ci::build_report::Interrupt<'_>| -> Outcome<u64> {
                panic!("{name}: measured despite the refusal")
            };
        let hooks = Hooks {
            scan: &scan_hook,
            measure: &measure_hook,
        };
        let code = report(
            &path,
            &directory.join("reports"),
            &tools,
            &list,
            &hooks,
            &never_interrupted,
        )
        .unwrap();
        assert_eq!(code, 1, "{name}");
    }

    // A real SIGTERM while the binary scans the approved roots (read-only) fails the receipt.
    if Path::new(&approved_roots[0]).is_dir() {
        let directory = scratch.join("signal");
        fs::create_dir(&directory).unwrap();
        let config_path = directory.join("config.json");
        fs::write(&config_path, &base).unwrap();
        let child = Command::new(tools.join("jope-build-report.sh"))
            .arg("--config")
            .arg(&config_path)
            .arg("--custody")
            .arg(directory.join("reports"))
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let entered = fs::read_dir(directory.join("reports"))
                .ok()
                .and_then(|mut runs| runs.next())
                .is_some_and(|run| run.unwrap().path().join("scan-0.stderr").exists());
            if entered {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "signal fixture did not enter scan"
            );
            std::thread::sleep(Duration::from_millis(20));
        }
        let status = Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .unwrap();
        assert!(status.success());
        let output = child.wait_with_output().unwrap();
        assert_eq!(
            output.status.code(),
            Some(1),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let run = fs::read_dir(directory.join("reports"))
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        let receipt =
            pyjson::parse_str(&fs::read_to_string(run.join("receipt.json")).unwrap()).unwrap();
        assert_eq!(receipt.get("status"), Some(&Value::from("failed")));
        assert!(receipt
            .get("error")
            .and_then(Value::as_str)
            .unwrap()
            .contains("signal 15"));
        cases.push(("real-SIGTERM".into(), run, receipt, 0));
    }
    // Neither refused location is created or modified.
    let config_path = scratch.join("success/config.json");
    for custody in [PathBuf::from("/tmp"), repo()] {
        let hooks = Hooks {
            scan: &|root, _| Ok(fixture_scan_doc(root)),
            measure: &|p, _, i| allocated(p, false, i),
        };
        let error = report(
            &config_path,
            &custody,
            &tools,
            &approved_roots,
            &hooks,
            &never_interrupted,
        )
        .unwrap_err();
        assert!(error.message.contains("outside /tmp and Git working trees"));
    }
    assert_eq!(
        fs::read(measured.join("evidence.bin")).unwrap(),
        b"\x00preserve\xff"
    );
    let summary: Vec<Value> = cases
        .iter()
        .map(|(name, run, receipt, invocations)| {
            Value::Dict(
                [
                    ("name", Value::from(name.clone())),
                    (
                        "receipt",
                        Value::from(run.join("receipt.json").to_string_lossy().into_owned()),
                    ),
                    (
                        "status",
                        receipt.get("status").cloned().unwrap_or(Value::Null),
                    ),
                    ("measure_invocations", Value::from(*invocations)),
                ]
                .into_iter()
                .collect(),
            )
        })
        .collect();
    fs::write(
        scratch.join("results.json"),
        pyjson::dumps(
            &Value::Dict([("cases", Value::List(summary))].into_iter().collect()),
            false,
            Some(2),
        ),
    )
    .unwrap();
}
