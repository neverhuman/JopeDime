use super::*;

use super::negative::{judge, record};
use super::round_six::{deepdir, place_positions, round_six_tree, unescape};

/// Replay every case of a round fixture (the layout of `negative-r2.json`); `message_differs` names
/// the cases where both refuse in different words, `stricter` those the Python accepted and the
/// port refuses (writing nothing), `argv_refused` those holding an argument that is not UTF-8,
/// which the port refuses before reading anything, and `usage_refused` those the port's argument
/// parser refuses (exit status 2, clap's `error: ...` containing the text given, nothing on stdout,
/// nothing written), whatever the Python did. A case's `nosearch_mode` is the mode of
/// `@ROOT@/nosearch`, which holds an executable `demo`, during its run. Round six's cases add
/// names that are not UTF-8 (`@Xhh@`, in arguments too), links, a deep directory and the locale
/// the Python ran under (see [`round_six_tree`]); a codec position `@ROOT+k@` counts this root.
/// Returns each case's exit status, stdout and stderr (the root written `@ROOT@`).
pub(super) fn replay_round(
    file: &str,
    count: usize,
    message_differs: &[&str],
    stricter: &[&str],
    argv_refused: &[&str],
    usage_refused: &[(&str, &str)],
) -> std::collections::BTreeMap<String, (i64, String, String)> {
    let doc: Value =
        serde_json::from_str(&fs::read_to_string(fixture().join(file)).unwrap()).unwrap();
    assert!(doc["source"].is_string() && doc["capture"].is_string());
    let cases = doc["cases"].as_array().unwrap();
    assert_eq!(
        cases.len(),
        count,
        "{file}: every captured case is replayed"
    );
    for name in message_differs
        .iter()
        .chain(stricter)
        .chain(argv_refused)
        .chain(usage_refused.iter().map(|(name, _)| name))
    {
        assert!(
            cases.iter().any(|c| c["name"] == *name),
            "{file}: {name} is not a case"
        );
    }
    let unhex = |text: &str| -> Vec<u8> {
        (0..text.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
            .collect()
    };
    let temp = tempfile::Builder::new()
        .prefix("cargo-artifact-round-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    fs::create_dir(root.join("bin")).unwrap();
    executable(&root.join("bin/demo"), "#!/bin/sh\nexit 0\n");
    fs::create_dir(root.join("nosearch")).unwrap();
    executable(&root.join("nosearch/demo"), "#!/bin/sh\nexit 0\n");
    let shims = root.join("shims");
    fs::create_dir(&shims).unwrap();
    executable(
        &shims.join("cargo"),
        "#!/bin/bash\ncase \"$1\" in\n  metadata) cat \"$CA_ROOT/metadata.json\"; e=$(cat \"$CA_ROOT/metadata.exit\") ;;\n  build) cat \"$CA_ROOT/messages.jsonl\"; e=$(cat \"$CA_ROOT/build.exit\") ;;\nesac\n[ \"$e\" = kill ] && kill -9 $$\nexit \"$e\"\n",
    );
    let path = format!(
        "{}:{}",
        shims.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    let mut results = std::collections::BTreeMap::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        for file in ["metadata.json", "messages.jsonl", "source.toml", "out.toml"] {
            let _ = fs::remove_file(root.join(file));
        }
        // Codec positions counted past the capture's root, placed past this one.
        let mut case = case.clone();
        if let Some(stderr) = case.get("stderr").and_then(Value::as_str) {
            case["stderr"] = Value::from(place_positions(stderr, root_text.chars().count()));
        }
        // A deep directory has a fixed absolute length, so the components the Python printed
        // under the capture's root are those `deepdir` gives under this one.
        if let (Some(deep), Some(stdout)) = (
            case.get("deep").cloned(),
            case.get("stdout").and_then(Value::as_str),
        ) {
            let file = deep["file"].as_str().unwrap();
            assert!(
                stdout.starts_with("@ROOT@/deep/") && stdout.ends_with(&format!("/{file}\n")),
                "{name}: {stdout}"
            );
            let here = deepdir(&root_text, deep["length"].as_u64().unwrap() as usize)
                .replacen(&root_text, "@ROOT@", 1);
            case["stdout"] = Value::from(format!("{here}/{file}\n"));
        }
        let case = &case;
        for (key, file) in [
            ("metadata", "metadata.json"),
            ("messages", "messages.jsonl"),
            ("source", "source.toml"),
        ] {
            let bytes = match (case.get(format!("{key}_hex")), case.get(key)) {
                (Some(hex), _) => unhex(hex.as_str().unwrap()),
                (None, Some(text)) => text
                    .as_str()
                    .unwrap()
                    .replace("@ROOT@", &root_text)
                    .into_bytes(),
                (None, None) => continue,
            };
            fs::write(root.join(file), bytes).unwrap();
        }
        for (key, file) in [
            ("metadata_exit", "metadata.exit"),
            ("build_exit", "build.exit"),
        ] {
            let exit = case.get(key).and_then(Value::as_str).unwrap_or("0");
            fs::write(root.join(file), exit).unwrap();
        }
        if case["output_exists"] == true {
            fs::write(root.join("out.toml"), "").unwrap();
        }
        round_six_tree(case, &root);
        let argv: Vec<std::ffi::OsString> = case["argv"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| {
                unescape(
                    &a.as_str()
                        .unwrap()
                        .replace("@METADATA@", root.join("metadata.json").to_str().unwrap())
                        .replace("@MESSAGES@", root.join("messages.jsonl").to_str().unwrap())
                        .replace("@SOURCE@", root.join("source.toml").to_str().unwrap())
                        .replace("@OUT@", root.join("out.toml").to_str().unwrap())
                        .replace("@ROOT@", &root_text),
                )
            })
            .collect();
        let nosearch = case
            .get("nosearch_mode")
            .and_then(Value::as_str)
            .map(|m| u32::from_str_radix(m, 8).unwrap());
        if let Some(mode) = nosearch {
            fs::set_permissions(root.join("nosearch"), fs::Permissions::from_mode(mode)).unwrap();
        }
        // A user who may search anything (root) cannot reproduce a refused search.
        let searchable = nosearch.is_some() && fs::metadata(root.join("nosearch/demo")).is_ok();
        let mut command = Command::new(env!("CARGO_BIN_EXE_jope-ci"));
        command
            .arg("cargo-artifact")
            .args(&argv)
            .env("PATH", &path)
            .env("CA_ROOT", &root)
            .env_remove("CARGO_TARGET_DIR")
            .current_dir(&root);
        // The locale the Python ran under (the port reads none).
        match case.get("locale").and_then(Value::as_str) {
            Some("en") => {
                command
                    .env_remove("LC_ALL")
                    .env_remove("LC_CTYPE")
                    .env_remove("LC_MESSAGES")
                    .env("LANG", "en_US.UTF-8");
            }
            Some("cu") => {
                command.env("LC_ALL", "C.UTF-8");
            }
            Some("c") => {
                command.env("LC_ALL", "C");
            }
            _ => {}
        }
        let run = command.output().unwrap();
        fs::set_permissions(root.join("nosearch"), fs::Permissions::from_mode(0o755)).unwrap();
        if searchable {
            eprintln!("{name}: skipped, this user may search a directory of that mode");
            continue;
        }
        let normalise = |bytes: &[u8]| String::from_utf8_lossy(bytes).replace(&root_text, "@ROOT@");
        let (stdout, stderr) = (normalise(&run.stdout), normalise(&run.stderr));
        let exit = i64::from(run.status.code().unwrap());
        record(case, exit, &stdout, &stderr);
        results.insert(name.to_owned(), (exit, stdout.clone(), stderr.clone()));
        let usage = usage_refused.iter().find(|(n, _)| *n == name).map(|u| u.1);
        let verdict = if let Some(text) = usage {
            // Refused by the argument parser before anything is read or written.
            if exit == 2
                && stdout.is_empty()
                && stderr.starts_with("error: ")
                && stderr.contains(text)
            {
                Ok(())
            } else {
                Err(format!(
                    "listed as a usage refusal ({text:?}), but the port exited {exit} {stdout:?} {stderr:?}"
                ))
            }
        } else if argv_refused.contains(&name) {
            // Refused before anything is read: one usage line, nothing on stdout, no output file.
            let refused = exit == 2
                && stdout.is_empty()
                && stderr.starts_with("jope-ci: argument ")
                && stderr.contains(" is not UTF-8")
                && stderr.lines().count() == 1;
            if refused && (case["exit"] == 0 || case["exit"] == 1) {
                Ok(())
            } else {
                Err(format!(
                    "listed as an argument refusal, but the Python exited {} and the port {exit} {stderr:?}",
                    case["exit"]
                ))
            }
        } else if stricter.contains(&name) {
            // The Python accepted; the port refuses (one `cargo-artifact:` line, nothing on stdout)
            // and leaves no output (checked below). The `toml` crate's parse error of a nextest
            // source spans lines (it quotes the line).
            let refused = exit == 1
                && stdout.is_empty()
                && stderr.starts_with("cargo-artifact: ")
                && (stderr.lines().count() == 1 || case.get("source").is_some());
            if refused && case["exit"] == 0 {
                Ok(())
            } else {
                Err(format!(
                    "listed as stricter, but the Python exited {} and the port {exit} {stderr:?}",
                    case["exit"]
                ))
            }
        } else if message_differs.contains(&name) {
            let refused = exit == 1
                && stdout.is_empty()
                && stderr.starts_with("cargo-artifact: ")
                && case["exit"] == 1
                && case["stdout"] == "";
            if refused {
                Ok(())
            } else {
                Err(format!(
                    "both must refuse: the port {exit} {stdout:?} {stderr:?}"
                ))
            }
        } else {
            judge(case, exit, &stdout, &stderr)
        };
        if let Err(why) = verdict {
            failures.push(format!("{name}: {why}"));
        }
        // The output file as the Python left it (absent, empty, or the written configuration).
        let left = fs::read_to_string(root.join("out.toml"))
            .ok()
            .map(|t| t.replace(&root_text, "@ROOT@"));
        let python_left =
            if stricter.contains(&name) || argv_refused.contains(&name) || usage.is_some() {
                None
            } else {
                case["output_after"].as_str().map(str::to_owned)
            };
        if left != python_left {
            failures.push(format!(
                "{name}: the output is {left:?}, the Python left {python_left:?}"
            ));
        }
    }
    assert!(failures.is_empty(), "{file}:\n{}", failures.join("\n"));
    results
}
