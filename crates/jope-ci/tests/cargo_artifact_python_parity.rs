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
