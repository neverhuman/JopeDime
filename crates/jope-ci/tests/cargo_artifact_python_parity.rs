//! `jope-ci cargo-artifact` against the outputs of the Python it replaced.
//!
//! `tests/fixtures/python-parity/cargo-artifact/python-outputs.json` holds what
//! `deploy/ci/cargo-artifact.py` (archived at 8a55c4fef) printed in the differential run
//! `py-p1-u2-diff-20260927T023243Z`, before it was deleted, with the lane root replaced by
//! `@ROOT@`. The inputs it ran on are committed beside it: the Cargo message stream of a real
//! `cargo build -p jope-paths --bin jope-units-render --message-format=json-render-diagnostics`,
//! that workspace's `cargo metadata` reduced to the fields `select` reads, and the
//! `.config/nextest.toml` of the time. Re-running the archived Python on exactly these committed
//! inputs reproduced every captured output (run `py-p1-fixtures-u2-20260927T033630Z`).
//!
//! Each case runs the real binary on the same inputs, materialised under a fresh directory that
//! takes the place of `@ROOT@`. Two differences are intended and are the only ones allowed:
//!
//! - `usage`: argument-parser wording (argparse vs clap). Both exit 2 and name the missing
//!   `--metadata` and `--messages`.
//! - `nextest-exists`: the OS error wording for an existing output file (Python's
//!   `[Errno 17] File exists: '<path>'` vs Rust's `<path>: File exists (os error 17)`).
//!
//! # Malformed inputs
//!
//! `tests/fixtures/python-parity/cargo-artifact/negative.json` holds what the same Python, as it
//! stood at `origin/main` `f46a9048f` (unmodified, run from its git bytes in a scratch root), did
//! with malformed Cargo metadata and message streams for `select`, `target-dir`, `nextest-config`
//! and `build` (behind a `cargo` on PATH that prints the case's metadata and messages): the exit
//! status, stdout, and either its `cargo-artifact: ...` message or the last line of its
//! traceback. Its `source` and `capture` fields name the custody run and the command. Every case
//! is held to one contract (`judge`): the port is never more permissive. Where the Python
//! succeeded or printed its own message, the port exits the same way and prints the identical
//! stdout and stderr; where the Python crashed with a traceback, the port refuses (exit status 1,
//! one `cargo-artifact: <field>: <what failed and the offending type>` line, nothing on stdout;
//! the text is not CPython's). The exceptions are listed in `STRICTER` (inputs the Python accepted
//! and the port refuses) and `MESSAGE_DIFFERS` (both refuse; the port's wording differs because
//! it refuses at an earlier step or with Rust's decoder wording).

use serde_json::Value;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/python-parity/cargo-artifact")
}

fn executable(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn cargo_artifact_matches_the_captured_python_outputs() {
    let fixture = fixture();
    let captured: Value =
        serde_json::from_str(&fs::read_to_string(fixture.join("python-outputs.json")).unwrap())
            .unwrap();
    let cases = captured["cases"].as_array().unwrap();
    let names: Vec<&str> = cases.iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "select",
            "select-map",
            "select-badmap",
            "select-unfinished",
            "select-failed",
            "select-wrongpkg",
            "select-example",
            "select-ambiguous",
            "usage",
            "nextest-config",
            "nextest-exists"
        ],
        "every captured case is replayed"
    );

    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-parity-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    let materialise = |name: &str| {
        fs::read_to_string(fixture.join(name))
            .unwrap()
            .replace("@ROOT@", &root_text)
    };
    fs::create_dir_all(root.join("target/debug")).unwrap();
    executable(
        &root.join("target/debug/jope-units-render"),
        "#!/bin/sh\nexit 0\n",
    );
    fs::write(root.join("metadata.json"), materialise("metadata.json")).unwrap();
    let build = materialise("build.jsonl");
    // The message-stream variants of the differential run, derived the same way.
    let unfinished: String = build
        .split_inclusive('\n')
        .filter(|line| !line.contains("\"build-finished\""))
        .collect();
    let failed = build.replace("\"success\":true", "\"success\":false");
    for (name, text) in [
        ("build", build.clone()),
        ("unfinished", unfinished),
        ("failed", failed),
        ("doubled", format!("{build}{build}")),
    ] {
        fs::write(root.join(format!("{name}.jsonl")), text).unwrap();
    }
    // `nextest-config` asks Cargo for the target directory; this Cargo answers as the lane's did.
    let bin = root.join("bin");
    fs::create_dir(&bin).unwrap();
    executable(
        &bin.join("cargo"),
        &format!("#!/bin/sh\nprintf '{{\"target_directory\": \"%s\"}}\\n' '{root_text}/target'\n"),
    );
    let path = format!(
        "{}:{}",
        bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let output_file = root.join(format!("{name}.toml"));
        if name == "nextest-exists" {
            fs::write(&output_file, "").unwrap();
        }
        let output_text = output_file.to_str().unwrap().to_owned();
        let messages = root.join(format!("{}.jsonl", case["messages"].as_str().unwrap()));
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a.as_str()
                    .unwrap()
                    .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                    .replace("@MESSAGES@", messages.to_str().unwrap())
                    .replace("@SOURCE@", fixture.join("nextest.toml").to_str().unwrap())
                    .replace("@OUT@", &output_text)
                    .replace("@ROOT@", &root_text)
            })
            .collect();
        let run = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root)
            .output()
            .unwrap();
        let normalise = |bytes: &[u8]| {
            String::from_utf8_lossy(bytes)
                .replace(&output_text, "@OUT@")
                .replace(&root_text, "@ROOT@")
        };
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let (want_stdout, want_stderr) = (
            case["stdout"].as_str().unwrap(),
            case["stderr"].as_str().unwrap(),
        );
        assert_eq!(
            Some(i64::from(run.status.code().unwrap())),
            case["exit"].as_i64(),
            "{name}: exit status; stderr {stderr}"
        );
        match name {
            "usage" => {
                // Allowed difference: argument-parser wording.
                assert_eq!((stdout.as_str(), want_stdout), ("", ""), "{name}");
                assert!(
                    want_stderr.starts_with("usage: cargo-artifact.py select ")
                        && want_stderr.contains(
                            "error: the following arguments are required: --metadata, --messages"
                        ),
                    "{name}: captured Python stderr {want_stderr}"
                );
                assert!(
                    stderr.contains("the following required arguments were not provided")
                        && stderr.contains("--metadata <METADATA>")
                        && stderr.contains("--messages <MESSAGES>"),
                    "{name}: Rust stderr {stderr}"
                );
            }
            "nextest-exists" => {
                // Allowed difference: the OS error wording for an existing output.
                assert_eq!((stdout.as_str(), want_stdout), ("", ""), "{name}");
                assert_eq!(
                    want_stderr, "cargo-artifact: [Errno 17] File exists: '@OUT@'\n",
                    "{name}: captured Python stderr"
                );
                assert_eq!(
                    stderr, "cargo-artifact: @OUT@: File exists (os error 17)\n",
                    "{name}: Rust stderr"
                );
                assert_eq!(fs::read_to_string(&output_file).unwrap(), "", "{name}");
            }
            _ => {
                assert_eq!(stdout, want_stdout, "{name}: stdout");
                assert_eq!(stderr, want_stderr, "{name}: stderr");
            }
        }
        if name == "nextest-config" {
            let written = fs::read_to_string(&output_file)
                .unwrap()
                .replace(&root_text, "@ROOT@");
            assert_eq!(
                written,
                captured["nextest_output"].as_str().unwrap(),
                "{name}: the relocated nextest config is byte-identical"
            );
        }
    }
}

/// Malformed inputs the Python accepted (exit status 0) and the port refuses: a
/// `target_directory` that is not a string, which the Python printed as `5` or `None`. (Its
/// `nextest-config` crashed on the same values.)
const STRICTER: &[&str] = &["target-dir-int", "target-dir-null"];

/// Both refuse with exit status 1, with different words: integers beyond 64 bits are refused by
/// the port's JSON reader before the package lookup the Python failed in (`found 0`), and bytes
/// that are not UTF-8 are reported in Rust's decoder wording rather than Python's codec wording.
const MESSAGE_DIFFERS: &[&str] = &["ids-beyond-64-bits", "build-non-utf8"];

/// Hold one case to the contract (see the module documentation); `Err` names the disagreement.
fn judge(case: &Value, exit: i64, stdout: &str, stderr: &str) -> Result<(), String> {
    let name = case["name"].as_str().unwrap();
    let want_exit = case["exit"].as_i64().unwrap();
    let refused = exit == 1
        && stdout.is_empty()
        && stderr.starts_with("cargo-artifact: ")
        && stderr.ends_with('\n')
        && stderr.lines().count() == 1;
    if STRICTER.contains(&name) {
        return if want_exit == 0 && refused {
            Ok(())
        } else {
            Err(format!(
                "listed as stricter, but the Python exited {want_exit} and the port {exit} ({stderr:?})"
            ))
        };
    }
    if exit != want_exit {
        return Err(format!(
            "exit status: the Python {want_exit}, the port {exit}; stdout {stdout:?}, stderr {stderr:?}"
        ));
    }
    if let Some(raised) = case.get("raised") {
        return if refused {
            Ok(())
        } else {
            Err(format!(
                "the Python raised {raised}; the port did not refuse: stdout {stdout:?}, stderr {stderr:?}"
            ))
        };
    }
    let want_stdout = case["stdout"].as_str().unwrap();
    if stdout != want_stdout {
        return Err(format!(
            "stdout: the Python {want_stdout:?}, the port {stdout:?}"
        ));
    }
    if MESSAGE_DIFFERS.contains(&name) {
        return if refused {
            Ok(())
        } else {
            Err(format!("the port did not refuse: {stderr:?}"))
        };
    }
    let want_stderr = case["stderr"].as_str().unwrap();
    if stderr != want_stderr {
        return Err(format!(
            "stderr: the Python {want_stderr:?}, the port {stderr:?}"
        ));
    }
    Ok(())
}

/// With `JOPE_NEGATIVE_REPORT` naming a file, every replayed case appends one tab-separated line
/// to it: the fixture, the case, the Python's exit status and message (its exception line, or its
/// stderr or stdout line) and the port's. The custody runs keep this table; the test never reads
/// it.
fn record(case: &Value, exit: i64, stdout: &str, stderr: &str) {
    let Some(path) = std::env::var_os("JOPE_NEGATIVE_REPORT") else {
        return;
    };
    let first = |text: &str| -> String { text.lines().next().unwrap_or("").to_owned() };
    let python = match case.get("raised") {
        Some(raised) => format!("raised {}", raised.as_str().unwrap()),
        None if !case["stderr"].as_str().unwrap().is_empty() => {
            format!("refused {}", first(case["stderr"].as_str().unwrap()))
        }
        None => format!("printed {}", first(case["stdout"].as_str().unwrap())),
    };
    let port = if stderr.is_empty() {
        format!("printed {}", first(stdout))
    } else {
        format!("refused {}", first(stderr))
    };
    let line = format!(
        "cargo-artifact/negative.json\t{}\t{}\t{python}\t{exit}\t{port}\n",
        case["name"].as_str().unwrap(),
        case["exit"]
    );
    use std::io::Write as _;
    fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap()
        .write_all(line.as_bytes())
        .unwrap();
}

#[test]
fn cargo_artifact_refuses_every_input_the_python_refused() {
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join("negative.json")).unwrap())
            .unwrap();
    assert!(
        doc["source"].is_string() && doc["capture"].is_string(),
        "negative.json names its capture run and command"
    );
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 50, "every captured case is replayed");
    for name in STRICTER.iter().chain(MESSAGE_DIFFERS) {
        assert!(
            cases.iter().any(|c| c["name"] == *name),
            "{name} is not a case"
        );
    }

    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-negative-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    fs::create_dir(root.join("bin")).unwrap();
    executable(&root.join("bin/demo"), "#!/bin/sh\nexit 0\n");
    fs::write(
        root.join("nextest.toml"),
        doc["nextest_source"].as_str().unwrap(),
    )
    .unwrap();
    let shims = root.join("shims");
    fs::create_dir(&shims).unwrap();
    executable(
        &shims.join("cargo"),
        "#!/bin/bash\ncase \"$1\" in\n  metadata) cat \"$CA_ROOT/metadata.json\" ;;\n  build) cat \"$CA_ROOT/messages.jsonl\" ;;\nesac\n",
    );
    let path = format!(
        "{}:{}",
        shims.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let output_file = root.join("out.toml");
        let _ = fs::remove_file(&output_file);
        fs::write(
            root.join("metadata.json"),
            case["metadata"]
                .as_str()
                .unwrap()
                .replace("@ROOT@", &root_text),
        )
        .unwrap();
        let mut messages = case["messages"]
            .as_str()
            .unwrap()
            .replace("@ROOT@", &root_text)
            .into_bytes();
        if let Some(bytes) = case.get("append_bytes") {
            messages.extend(
                bytes
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|b| u8::try_from(b.as_u64().unwrap()).unwrap()),
            );
        }
        fs::write(root.join("messages.jsonl"), messages).unwrap();
        let argv: Vec<String> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                a.as_str()
                    .unwrap()
                    .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                    .replace("@MESSAGES@", root.join("messages.jsonl").to_str().unwrap())
                    .replace("@SOURCE@", root.join("nextest.toml").to_str().unwrap())
                    .replace("@OUT@", output_file.to_str().unwrap())
            })
            .collect();
        let run = Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env("CA_ROOT", &root)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root)
            .output()
            .unwrap();
        let normalise = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&root_text, "@ROOT@");
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let exit = i64::from(run.status.code().unwrap());
        record(case, exit, &stdout, &stderr);
        if let Err(why) = judge(case, exit, &stdout, &stderr) {
            failures.push(format!("{name}: {why}"));
        }
        // A refused `nextest-config` leaves no partial output behind.
        if name.starts_with("nextest-config") && exit != 0 && output_file.exists() {
            failures.push(format!("{name}: a refusal left {}", output_file.display()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));

    // Negative control: `judge` fails an altered expectation, a report in place of a refusal and
    // a success in place of a crash.
    let ok = cases.iter().find(|c| c["name"] == "select-ok").unwrap();
    let ok_stdout = ok["stdout"].as_str().unwrap();
    assert_eq!(judge(ok, 0, ok_stdout, ""), Ok(()));
    assert!(judge(ok, 0, &ok_stdout.replace("demo", "dema"), "").is_err());
    assert!(judge(ok, 1, "", "cargo-artifact: x\n").is_err());
    let crash = cases
        .iter()
        .find(|c| c["name"] == "executable-int")
        .unwrap();
    assert_eq!(judge(crash, 1, "", "cargo-artifact: x\n"), Ok(()));
    assert!(judge(crash, 0, ok_stdout, "").is_err());
    assert!(judge(crash, 1, ok_stdout, "cargo-artifact: x\n").is_err());
    let message = cases.iter().find(|c| c["name"] == "success-one").unwrap();
    assert!(judge(message, 1, "", "cargo-artifact: another message\n").is_err());
}
