//! Artifact protocol regressions; no Rust compilation or external dependencies. Port of
//! `deploy/ci/test-cargo-artifact.py` (archived at 8a55c4fef): the same cases, with the fake
//! `cargo` written in shell instead of Python.

use jope_ci::cargo_artifact::{executable, nextest_config};
use serde_json::{json, Value};
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;

struct Fixture {
    _temp: tempfile::TempDir,
    root: PathBuf,
    binary: PathBuf,
    meta: Value,
    message: Value,
    finished: Value,
}

impl Fixture {
    /// Tests create only disposable fixtures, not a Cargo target directory.
    fn new() -> Self {
        let temp = tempfile::Builder::new()
            .prefix("artifact protocol ")
            .tempdir()
            .unwrap();
        let root = temp.path().canonicalize().unwrap();
        let binary = root.join("external target with spaces").join("program");
        fs::create_dir(binary.parent().unwrap()).unwrap();
        fs::write(&binary, "#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&binary, fs::Permissions::from_mode(0o755)).unwrap();
        let meta = json!({
            "packages": [{"name": "demo", "id": "path+file:///demo#0.1.0"}],
            "workspace_members": ["path+file:///demo#0.1.0"],
            "target_directory": binary.parent().unwrap(),
        });
        let message = json!({
            "reason": "compiler-artifact", "package_id": "path+file:///demo#0.1.0",
            "target": {"kind": ["bin"], "name": "demo"}, "profile": {"test": false},
            "executable": binary, "fresh": true,
        });
        Self {
            _temp: temp,
            root,
            binary,
            meta,
            message,
            finished: json!({"reason": "build-finished", "success": true}),
        }
    }

    fn select(
        &self,
        messages: &[Value],
        path_map: Option<(&Path, &Path)>,
    ) -> Result<PathBuf, String> {
        let text: Vec<String> = messages.iter().map(Value::to_string).collect();
        executable(
            &text.join("\n"),
            &jope_records::pyjson::parse_str(&self.meta.to_string()).unwrap(),
            "demo",
            "bin",
            "demo",
            path_map,
        )
    }

    fn default_select(&self) -> Result<PathBuf, String> {
        self.select(&[self.message.clone(), self.finished.clone()], None)
    }

    /// Run the real `jope-ci` binary with a fake `cargo` first on PATH that records its argv and
    /// Cargo environment, prints the fixture metadata/messages and exits with `FAKE_BUILD_EXIT`.
    fn cli(&self, args: &[&str], exit_status: i32) -> std::process::Output {
        let fake = self.root.join("cargo");
        fs::write(
            &fake,
            r#"#!/bin/bash
root=$(dirname "$0")
{
  printf 'target=%s\n' "${CARGO_TARGET_DIR-<unset>}"
  printf 'incremental=%s\n' "${CARGO_INCREMENTAL-<unset>}"
  printf 'debug=%s\n' "${CARGO_PROFILE_DEV_DEBUG-<unset>}"
  for a in "$@"; do printf 'arg=%s\n' "$a"; done
  echo '--'
} >> "$root/commands.txt"
if [ "$1" = metadata ]; then cat "$root/metadata.json"; else cat "$root/messages.jsonl"; exit "$FAKE_BUILD_EXIT"; fi
"#,
        )
        .unwrap();
        fs::set_permissions(&fake, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(self.root.join("metadata.json"), self.meta.to_string()).unwrap();
        fs::write(
            self.root.join("messages.jsonl"),
            format!("{}\n{}", self.message, self.finished),
        )
        .unwrap();
        let path = format!(
            "{}:{}",
            self.root.display(),
            std::env::var("PATH").unwrap_or_default()
        );
        Command::new(env!("CARGO_BIN_EXE_jope-ci"))
            .arg("cargo-artifact")
            .args(args)
            .env("PATH", path)
            .env("FAKE_BUILD_EXIT", exit_status.to_string())
            .env_remove("CARGO_INCREMENTAL")
            .env_remove("CARGO_PROFILE_DEV_DEBUG")
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&self.root)
            .output()
            .unwrap()
    }

    /// The recorded fake-Cargo invocations: (environment lines, argv).
    fn commands(&self) -> Vec<(Vec<String>, Vec<String>)> {
        let text = fs::read_to_string(self.root.join("commands.txt")).unwrap();
        text.split("--\n")
            .filter(|block| !block.is_empty())
            .map(|block| {
                let (mut env, mut argv) = (vec![], vec![]);
                for line in block.lines() {
                    match line.strip_prefix("arg=") {
                        Some(arg) => argv.push(arg.to_owned()),
                        None => env.push(line.to_owned()),
                    }
                }
                (env, argv)
            })
            .collect()
    }
}

fn with(message: &Value, field: &str, value: Value) -> Value {
    let mut changed = message.clone();
    changed[field] = value;
    changed
}

#[test]
fn cached_and_new_builds_with_spaces() {
    let mut f = Fixture::new();
    for fresh in [true, false] {
        f.message["fresh"] = json!(fresh);
        assert_eq!(f.default_select().unwrap(), f.binary);
    }
}

#[test]
fn rejects_missing_target_and_nonmatching_package_kind_or_name() {
    let f = Fixture::new();
    for (field, value) in [
        ("package_id", json!("dependency")),
        ("target", json!({"kind": ["example"], "name": "demo"})),
        ("target", json!({"kind": ["bin"], "name": "other"})),
        ("profile", json!({"test": true})),
        ("executable", json!(null)),
    ] {
        let error = f
            .select(
                &[with(&f.message, field, value.clone()), f.finished.clone()],
                None,
            )
            .unwrap_err();
        assert!(error.contains("found 0"), "{field}={value}: {error}");
    }
}

#[test]
fn rejects_ambiguous_artifacts() {
    let f = Fixture::new();
    let other = with(&f.message, "executable", json!(f.root.join("other")));
    let error = f
        .select(&[f.message.clone(), other, f.finished.clone()], None)
        .unwrap_err();
    assert!(error.contains("found 2"), "{error}");
}

#[test]
fn rejects_nonexecutables_and_missing_files() {
    let f = Fixture::new();
    fs::set_permissions(&f.binary, fs::Permissions::from_mode(0o644)).unwrap();
    assert!(f.default_select().unwrap_err().contains("not executable"));
    fs::remove_file(&f.binary).unwrap();
    assert!(f.default_select().unwrap_err().contains("missing"));
}

#[test]
fn rejects_failed_or_unfinished_stream() {
    let f = Fixture::new();
    let failed = with(&f.finished, "success", json!(false));
    for messages in [vec![f.message.clone()], vec![f.message.clone(), failed]] {
        let error = f.select(&messages, None).unwrap_err();
        assert!(error.contains("successful completed build"), "{error}");
    }
}

#[test]
fn translates_container_mount() {
    let f = Fixture::new();
    let mounted = with(
        &f.message,
        "executable",
        json!("/work/external target with spaces/program"),
    );
    let messages = [mounted, f.finished.clone()];
    assert_eq!(
        f.select(&messages, Some((Path::new("/work"), &f.root)))
            .unwrap(),
        f.binary
    );
    assert!(f
        .select(&messages, Some((Path::new("/other"), &f.root)))
        .is_err());
}

#[test]
fn late_build_failure_preserves_exit_and_prints_no_executable() {
    let f = Fixture::new();
    let result = f.cli(&["build", "--package", "demo", "--bin", "demo"], 101);
    assert_eq!(
        result.status.code(),
        Some(101),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.is_empty());
}

#[test]
fn cli_returns_only_path_and_passes_profile_features_and_target() {
    let f = Fixture::new();
    let target = f.binary.parent().unwrap().to_str().unwrap().to_owned();
    let result = f.cli(
        &[
            "build",
            "--package",
            "demo",
            "--bin",
            "demo",
            "--profile",
            "release",
            "--features",
            "a,b",
            "--features",
            "c",
            "--target-dir",
            &target,
        ],
        0,
    );
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!("{}\n", f.binary.display())
    );
    let commands = f.commands();
    assert_eq!(commands.len(), 2);
    for (env, _) in &commands {
        assert_eq!(
            env,
            &[
                format!("target={target}"),
                "incremental=0".to_owned(),
                "debug=line-tables-only".to_owned()
            ]
        );
    }
    let argv = &commands[1].1;
    let profile = argv.iter().position(|a| a == "--profile").unwrap();
    assert_eq!(argv[profile + 1], "release");
    assert_eq!(
        argv[argv.len() - 4..],
        ["--features", "a,b", "--features", "c"]
    );
}

#[test]
fn effective_target_directory_comes_from_metadata() {
    let f = Fixture::new();
    let result = f.cli(&["target-dir"], 0);
    assert_eq!(
        result.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        format!("{}\n", f.binary.parent().unwrap().display())
    );
}

#[test]
fn nextest_config_preserves_profiles_and_binds_external_store() {
    let f = Fixture::new();
    let source = f.root.join("nextest.toml");
    let output = f.root.join("generated.toml");
    fs::write(
        &source,
        "[profile.pr]\ndefault-filter = \"all()\"\n[profile.pr.junit]\npath = \"junit.xml\"\n",
    )
    .unwrap();
    let target_path = f.binary.parent().unwrap();
    // `meta["target_directory"]`, as Cargo reports it.
    let target = &jope_records::pyjson::Value::from(target_path.to_str().unwrap());
    nextest_config(&source, &output, target).unwrap();
    let config: toml::Table = fs::read_to_string(&output).unwrap().parse().unwrap();
    assert_eq!(
        config["store"]["dir"].as_str().unwrap(),
        target_path.join("nextest").to_str().unwrap()
    );
    let original: toml::Table = fs::read_to_string(&source).unwrap().parse().unwrap();
    assert_eq!(config["profile"], original["profile"]);
    // The output is created exclusively: an existing file is never overwritten.
    assert!(nextest_config(&source, &output, target).is_err());
    fs::write(&source, "[store]\ndir=\"custom\"\n").unwrap();
    let error = nextest_config(&source, &f.root.join("other.toml"), target).unwrap_err();
    assert!(error.contains("already configures a store"), "{error}");
    assert!(!f.root.join("other.toml").exists());
}

/// Exercise the actual retention command. Stub only Cargo (`jope-ci cargo-artifact target-dir`
/// runs through `cargo run`); its path resolution is tested separately above and with real
/// nextest runs.
#[test]
fn report_retention_requires_execution_and_a_current_report() {
    let script = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../deploy/ci/retain-nextest.sh"),
    )
    .unwrap();
    assert!(
        script.contains("jope-ci -- cargo-artifact target-dir"),
        "retain-nextest.sh must resolve the target through jope-ci"
    );
    let script = format!("cargo() {{ printf '%s\\n' \"$REPORT_TARGET\"; }}\n{script}");
    let f = Fixture::new();
    for (index, (started, report, code)) in [(false, true, 0), (true, false, 1), (true, true, 0)]
        .into_iter()
        .enumerate()
    {
        let case = f.root.join(index.to_string());
        let evidence = case.join("rust-test-evidence");
        fs::create_dir_all(&evidence).unwrap();
        let target = case.join("external target");
        let junit = target.join("nextest/pr/junit.xml");
        fs::create_dir_all(junit.parent().unwrap()).unwrap();
        if started {
            fs::write(evidence.join("nextest-started"), "").unwrap();
        }
        if report {
            fs::write(
                &junit,
                if started {
                    "current report"
                } else {
                    "old report"
                },
            )
            .unwrap();
        }
        let result = Command::new("bash")
            .args(["-c", &script, "retain-nextest"])
            .arg(&evidence)
            .current_dir(&case)
            .env("REPORT_TARGET", &target)
            .output()
            .unwrap();
        assert_eq!(
            result.status.code(),
            Some(code),
            "started={started} report={report}: {}{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(evidence.join("junit.xml").exists(), started && report);
    }
}
