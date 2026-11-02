use super::*;

/// Malformed inputs the Python accepted (exit status 0) and the port refuses: a
/// `target_directory` that is not a string, which the Python printed as `5` or `None`. (Its
/// `nextest-config` crashed on the same values.)
const STRICTER: &[&str] = &["target-dir-int", "target-dir-null"];

/// Hold one case to the contract (see the module documentation); `Err` names the disagreement.
pub(super) fn judge(case: &Value, exit: i64, stdout: &str, stderr: &str) -> Result<(), String> {
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
pub(super) fn record(case: &Value, exit: i64, stdout: &str, stderr: &str) {
    let Some(path) = std::env::var_os("JOPE_NEGATIVE_REPORT") else {
        return;
    };
    let first = |text: &str| -> String { text.lines().next().unwrap_or("").to_owned() };
    // Round six's stdout that is not UTF-8 (the raw bytes the Python printed under C and
    // `C.UTF-8`) is recorded as its hex.
    let python = match case.get("raised") {
        Some(raised) => format!("raised {}", raised.as_str().unwrap()),
        None if !case["stderr"].as_str().unwrap().is_empty() => {
            format!("refused {}", first(case["stderr"].as_str().unwrap()))
        }
        None => match case.get("stdout").and_then(Value::as_str) {
            Some(stdout) => format!("printed {}", first(stdout)),
            None => format!("printed the bytes {}", case["stdout_hex"].as_str().unwrap()),
        },
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
    for name in STRICTER {
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
        let mut case = case.clone();
        if let Some(bytes) = case.get("append_bytes").cloned() {
            // The codec position counts the stream Cargo printed, whose `@ROOT@` was the capture's
            // root: here the appended byte sits after this root, so the Python's message names the
            // position it would have had on these bytes.
            let stderr = case["stderr"].as_str().unwrap();
            let at = stderr.find("in position ").unwrap() + "in position ".len();
            let digits = stderr[at..].bytes().take_while(u8::is_ascii_digit).count();
            let placed = format!(
                "{}{}{}",
                &stderr[..at],
                messages.len(),
                &stderr[at + digits..]
            );
            case["stderr"] = Value::from(placed);
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
        record(&case, exit, &stdout, &stderr);
        if let Err(why) = judge(&case, exit, &stdout, &stderr) {
            failures.push(format!("{name}: {why}"));
        }
        // A refused `nextest-config` leaves the output as the Python did: its `TypeError` came
        // after `open("x")` had created the (still empty) file; every other refusal came first.
        if name.starts_with("nextest-config") && exit != 0 {
            let left = fs::read(&output_file).ok();
            let python_left = case.get("raised").map(|_| Vec::new());
            if left != python_left {
                failures.push(format!(
                    "{name}: the output is {left:?}, the Python left {python_left:?}"
                ));
            }
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
