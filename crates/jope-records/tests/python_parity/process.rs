use super::negative::{discovered_items, negative, record};
use super::*;

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
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

/// Write a `process.json` case: its tree under `root` (`@ROOT@` in text replaced), and what the
/// `cargo` and `git` shims print under `dir`, encoded as the capture encoded it.
fn materialise(case: &Value, root: &Path, dir: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let root_text = root.to_str().unwrap();
    // Round six: `@LOSSY_REAL_ROOT@` is the real root behind the `root` link, spelled with U+FFFD
    // for each byte that is not UTF-8 (as a lossy conversion spelled it).
    let lossy = case.get("root_link").map(|link| {
        let parent = root.parent().unwrap().to_str().unwrap();
        let name = unescape(s(link));
        format!("{parent}/{}", String::from_utf8_lossy(name.as_bytes()))
    });
    for file in case.get("files").and_then(Value::as_list).unwrap() {
        let path = match file.get("path_hex") {
            Some(hex) => root.join(std::ffi::OsStr::from_bytes(&unhex(s(hex)))),
            None => root.join(s(file.get("path").unwrap())),
        };
        let bytes = match file.get("hex") {
            Some(hex) => unhex(s(hex)),
            None => {
                let text = s(file.get("text").unwrap()).replace("@ROOT@", root_text);
                match &lossy {
                    Some(lossy) => text.replace("@LOSSY_REAL_ROOT@", lossy),
                    None => text,
                }
                .into_bytes()
            }
        };
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, bytes).unwrap();
    }
    // Round five: directories, symlinks (a target's `@ROOT@` replaced) and copies of files.
    tree_extras(case, root);
    fs::create_dir_all(dir).unwrap();
    let cargo = case.get("cargo").unwrap();
    let text = s(cargo.get("stdout").unwrap()).replace("@ROOT@", root_text);
    let stdout = match s(cargo.get("encoding").unwrap()) {
        "utf-8" => text.into_bytes(),
        "utf-8-sig" => [&[0xef, 0xbb, 0xbf][..], text.as_bytes()].concat(),
        "utf-16" => [0xff, 0xfe]
            .into_iter()
            .chain(text.encode_utf16().flat_map(u16::to_le_bytes))
            .collect(),
        "hex" => unhex(s(cargo.get("stdout_hex").unwrap())),
        other => panic!("encoding {other}"),
    };
    fs::write(dir.join("cargo.stdout"), stdout).unwrap();
    fs::write(
        dir.join("cargo.stderr"),
        unhex(s(cargo.get("stderr_hex").unwrap())),
    )
    .unwrap();
    fs::write(dir.join("cargo.exit"), s(cargo.get("exit").unwrap())).unwrap();
    let git = match (case.get("git_hex"), case.get("git")) {
        (Some(hex), _) => unhex(s(hex)),
        (None, Some(list)) => list
            .as_list()
            .unwrap()
            .iter()
            .flat_map(|p| format!("{}\0", s(p)).into_bytes())
            .collect(),
        (None, None) => vec![],
    };
    fs::write(dir.join("git.stdout"), git).unwrap();
}

/// A case's `dirs` (created), `symlinks` (`path` -> `target`, `@ROOT@` in the target replaced) and
/// `copies` (`from` -> `to`, or `to_hex` for a name that is not UTF-8), all relative to `root`.
pub(super) fn tree_extras(case: &Value, root: &Path) {
    use std::os::unix::ffi::OsStrExt;
    let list = |key: &str| {
        case.get(key)
            .and_then(Value::as_list)
            .cloned()
            .unwrap_or_default()
    };
    for dir in list("dirs") {
        fs::create_dir_all(root.join(s(&dir))).unwrap();
    }
    for link in list("symlinks") {
        let path = root.join(s(link.get("path").unwrap()));
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let target = s(link.get("target").unwrap()).replace("@ROOT@", root.to_str().unwrap());
        std::os::unix::fs::symlink(target, path).unwrap();
    }
    for copy in list("copies") {
        let to = match (copy.get("to_hex"), copy.get("to")) {
            (Some(hex), _) => root.join(std::ffi::OsStr::from_bytes(&unhex(s(hex)))),
            (None, Some(to)) => root.join(s(to)),
            _ => panic!("a copy without a destination"),
        };
        fs::copy(root.join(s(copy.get("from").unwrap())), to).unwrap();
    }
}

/// A case's `modes` (`path` -> octal mode) applied under `root`, or with `restore` set back to 755.
/// Returns whether each mode took effect for this user: a privileged user may search and list a
/// directory whatever its mode, so a case that relies on a refused search cannot be reproduced.
pub(super) fn apply_modes(case: &Value, root: &Path, restore: bool) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let mut effective = true;
    for entry in case
        .get("modes")
        .and_then(Value::as_list)
        .cloned()
        .unwrap_or_default()
    {
        let path = root.join(s(entry.get("path").unwrap()));
        let mode = if restore {
            0o755
        } else {
            u32::from_str_radix(s(entry.get("mode").unwrap()), 8).unwrap()
        };
        fs::set_permissions(&path, fs::Permissions::from_mode(mode)).unwrap();
        if !restore && mode & 0o444 == 0 && path.is_file() && fs::File::open(&path).is_ok() {
            // A privileged test process can read a mode-000 file that the captured Python user
            // could not. This case cannot reproduce the captured PermissionError here.
            effective = false;
        }
        if !restore && mode & 0o500 != 0o500 && path.is_dir() {
            // A mode without read and search: would this user still list the directory, or reach
            // an entry of it?
            effective &= fs::read_dir(&path).is_err()
                || fs::read_dir(&path)
                    .unwrap()
                    .flatten()
                    .any(|e| fs::symlink_metadata(e.path()).is_err());
        }
    }
    effective
}

/// The `cargo` and `git` the capture put first on PATH.
fn shims(dir: &Path) {
    use std::os::unix::fs::PermissionsExt;
    fs::create_dir_all(dir).unwrap();
    for (name, text) in [
        (
            "cargo",
            "#!/bin/bash\ncat \"$CASE_DIR/cargo.stderr\" >&2\ncat \"$CASE_DIR/cargo.stdout\"\ne=$(cat \"$CASE_DIR/cargo.exit\")\n[ \"$e\" = kill ] && kill -9 $$\nexit \"$e\"\n",
        ),
        ("git", "#!/bin/bash\ncat \"$CASE_DIR/git.stdout\"\n"),
    ] {
        fs::write(dir.join(name), text).unwrap();
        fs::set_permissions(dir.join(name), fs::Permissions::from_mode(0o755)).unwrap();
    }
}

/// Process cases the port refuses although the Python ran to completion (exit status 0): round
/// four's strict number grammar reads a number in ASCII digits only, where Python's `int()` read
/// any Unicode decimal digits; round five's walks refuse a directory they cannot list, which
/// pathlib's globs skipped.
const PROCESS_STRICTER: &[(&str, &str)] = &[
    (
        "parity-tally-arabic-equal",
        "a covering-test tally written in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-tally-4300-arabic-indic-digits",
        "a covering-test tally of 4300 Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-r5-tools-sub-unlistable",
        "a directory under python/src/jope/tools that cannot be listed (pathlib's rglob skipped it)",
    ),
    (
        "parity-r5-fallback-bin-unlistable",
        "a src/bin directory that cannot be listed (pathlib's glob found nothing in it)",
    ),
    (
        "parity-r5-fallback-crate-unlistable",
        "a crate directory that cannot be listed (pathlib's `**` walk skipped it)",
    ),
    (
        "parity-r6-root-not-utf8",
        "a root whose resolved name is not UTF-8 (the Python named every entry point relative to its surrogate-escaped root)",
    ),
];

/// Process cases where both exit 1 but the Python printed a failing report and the port refuses
/// instead:
///
/// - The report names an entry point whose path is not UTF-8: the Python kept it
///   surrogate-escaped, and printed its raw bytes only because the capture's `C.UTF-8` locale gives
///   stdout the `surrogateescape` handler (under another UTF-8 locale it crashed printing them).
///   The port's reports are text, so it refuses, naming the path.
/// - Round four: a number written in Unicode digits other than ASCII, which Python's `int()` read
///   and the strict number grammar refuses before the report is made.
const PROCESS_REFUSES_REPORT: &[(&str, &str)] = &[
    (
        "parity-tool-name-not-utf8-decoy",
        "a Python tool named with a byte that is not UTF-8",
    ),
    (
        "parity-unit-name-not-utf8",
        "a systemd unit named with a byte that is not UTF-8",
    ),
    (
        "parity-pending-wp-arabic-digit",
        "a `pending-WP<n>` status numbered in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
    (
        "parity-tally-arabic-differs",
        "a covering-test tally written in Arabic-Indic digits (`int()` of `\\d+`)",
    ),
];

/// Hold one process case to the contract: where the Python crashed, the port exits 1 and prints
/// nothing on stdout; otherwise the exit status and stdout are identical, and stderr is identical
/// or (for a refusal) the Python's message after `jope-records: `.
fn judge_process(case: &Value, exit: i32, stdout: &[u8], stderr: &str) -> Result<(), String> {
    let name = s(case.get("name").unwrap());
    let want_exit = case.get("exit").and_then(Value::as_int).unwrap();
    if PROCESS_STRICTER.iter().any(|(n, _)| *n == name) {
        return if want_exit == 0 && exit == 1 && stdout.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "listed as stricter; the Python {want_exit}, the port {exit}"
            ))
        };
    }
    if PROCESS_REFUSES_REPORT.iter().any(|(n, _)| *n == name) {
        return if want_exit == 1
            && case.get("raised").is_none()
            && exit == 1
            && stdout.is_empty()
            && stderr.starts_with("jope-records: ")
        {
            Ok(())
        } else {
            Err(format!(
                "listed as a refusal of a failing report; the Python {want_exit}, the port {exit} {stderr:?}"
            ))
        };
    }
    if let Some(raised) = case.get("raised") {
        return if exit == 1 && stdout.is_empty() && !stderr.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the Python raised {}; the port exited {exit} with stdout {:?} and stderr {stderr:?}",
                s(raised),
                String::from_utf8_lossy(stdout)
            ))
        };
    }
    if i64::from(exit) != want_exit {
        return Err(format!(
            "exit status: the Python {want_exit}, the port {exit} (stderr {stderr:?})"
        ));
    }
    let want_stdout = match case.get("stdout_hex") {
        Some(hex) => unhex(s(hex)),
        None => s(case.get("stdout").unwrap()).as_bytes().to_vec(),
    };
    if stdout != want_stdout {
        return Err(format!(
            "stdout: the Python {:?}, the port {:?}",
            String::from_utf8_lossy(&want_stdout),
            String::from_utf8_lossy(stdout)
        ));
    }
    let want_stderr = s(case.get("stderr").unwrap());
    let refusal =
        exit == 1 && stdout.is_empty() && stderr == format!("jope-records: {want_stderr}");
    if stderr != want_stderr && !refusal {
        return Err(format!(
            "stderr: the Python {want_stderr:?}, the port {stderr:?}"
        ));
    }
    Ok(())
}

/// Round two of the differential review: the guards run end to end, each against a tree and a
/// `cargo` (and `git`) whose output the capture recorded, so what the port does with Cargo's
/// stdout and stderr (decoded as `text=True` decoded them), the files it reads (`read_text`, with
/// and without `errors="replace"`) and the order of its steps is compared with the Python's.
#[test]
fn process_cases_match_the_captured_python() {
    let doc = json("process.json");
    let cases = replay_process("process.json", &doc, 65);
    // Negative control: a changed exit status, stdout or message fails the judgement.
    let base = cases
        .iter()
        .find(|c| c.get("name") == Some(&Value::from("parity-base")))
        .unwrap();
    let base_stdout = s(base.get("stdout").unwrap()).as_bytes();
    assert_eq!(judge_process(base, 0, base_stdout, ""), Ok(()));
    assert!(judge_process(base, 1, base_stdout, "").is_err());
    assert!(judge_process(base, 0, b"x", "").is_err());
    let crash = cases.iter().find(|c| c.get("raised").is_some()).unwrap();
    assert!(judge_process(crash, 0, b"", "").is_err());
    assert!(judge_process(crash, 1, b"a report", "x").is_err());
}

/// Round three, integers (`process-r3-digits.json`, review 5333747872 of PR #117): Python 3.12
/// refused to convert an integer of more than 4300 decimal digits. `json.loads` of Cargo's
/// metadata and of the coverage map, and `int()` of the capability matrix's covering-test tally
/// and of a `pending-WP<n>` status (leading zeros and Unicode digits count), each crashed on 4301
/// digits and read 4300.
#[test]
fn process_cases_match_round_three_digits() {
    replay_process(
        "process-r3-digits.json",
        &json("process-r3-digits.json"),
        15,
    );
}

/// Round three, paths (`process-r3-paths.json`): POSIX's distinct `//` root where the guards
/// compared paths. `Path(p).relative_to(ROOT)` in coverage discovery and in the parity check's
/// `src/bin` fallback refused a path spelled `//...` (and one outside the tree); `///`, inner `//`
/// and `.` spellings normalise; `Path(name).parent` of the listed files is compared normalised; and
/// `os.path.relpath`, which splits on separators, has no distinct `//` root at all.
#[test]
fn process_cases_match_round_three_paths() {
    replay_process("process-r3-paths.json", &json("process-r3-paths.json"), 15);
}

/// Round five (`process-r5-sweep.json`): `parity-check` where Python 3.12's pathlib raised
/// (`main_rs.exists()`, `cli.exists()`, `is_dir()` and `is_file()` in the tool and unit walks, on
/// a directory that may not be listed or searched) and the `src/bin` fallback's globs, which list
/// entries of every kind by name. The port read a permission error as "absent" and skipped what
/// the globs listed; each case now refuses or reports as the Python did, and a directory that
/// cannot be listed refuses (`PROCESS_STRICTER`).
#[test]
fn process_cases_match_round_five() {
    replay_process("process-r5-sweep.json", &json("process-r5-sweep.json"), 28);
}

/// Round six (`process-r6.json`): the guards on a root that is a link to a directory whose name is
/// not UTF-8. The parity check compared entry points with the root converted lossily (U+FFFD for
/// the byte), so a `[[bin]]` path spelled with U+FFFD was found under the root and the check passed
/// where the Python's `relative_to` raised; it now refuses such a root before reading anything
/// (`PROCESS_STRICTER`: the Python ran the check). Coverage discovery refuses as the Python raised;
/// the scope check converts nothing and matches the Python's report.
#[test]
fn process_cases_match_round_six() {
    replay_process("process-r6.json", &json("process-r6.json"), 4);
}

/// Replay every case of a process fixture (the layout of `process.json`) through the binary and
/// hold each to `judge_process`; the cases, for the caller's controls.
fn replay_process<'a>(rel: &str, doc: &'a Value, count: usize) -> &'a Vec<Value> {
    assert!(
        doc.get("source").and_then(Value::as_str).is_some()
            && doc.get("capture").and_then(Value::as_str).is_some(),
        "{rel} names its capture run and command"
    );
    let cases = doc.get("cases").and_then(Value::as_list).unwrap();
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let temp = tempfile::tempdir().unwrap();
    let shim_dir = temp.path().join("shims");
    shims(&shim_dir);
    let path = format!(
        "{}:{}",
        shim_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let mut failures = vec![];
    for case in cases {
        let name = s(case.get("name").unwrap());
        let base = temp.path().join(name);
        let dir = base.join("shim");
        fs::create_dir_all(&base).unwrap();
        let base = base.canonicalize().unwrap();
        // Round six: `root_link` names the real root (`@Xhh@` for a byte that is not UTF-8), which
        // the case reaches through the link `root`; `@ROOT@` is the link.
        let root = match case.get("root_link") {
            Some(link) => {
                let real = unescape(s(link));
                fs::create_dir(base.join(&real)).unwrap();
                std::os::unix::fs::symlink(&real, base.join("root")).unwrap();
                base.join("root")
            }
            None => {
                fs::create_dir(base.join("root")).unwrap();
                base.join("root")
            }
        };
        materialise(case, &root, &dir);
        let root_text = root.to_str().unwrap();
        let args: Vec<String> = case
            .get("rust")
            .and_then(Value::as_list)
            .unwrap()
            .iter()
            .map(|a| s(a).replace("@ROOT@", root_text))
            .collect();
        // Round five: the modes the case ran under, set back after the run.
        let effective = apply_modes(case, &root, false);
        let run = std::process::Command::new(env!("CARGO_BIN_EXE_jope-records"))
            .arg("--repo")
            .arg(&root)
            .args(&args)
            .current_dir(&root)
            .env("PATH", &path)
            .env("CASE_DIR", &dir)
            .output()
            .unwrap();
        apply_modes(case, &root, true);
        if !effective {
            eprintln!("{rel} {name}: skipped, this user may search a directory of that mode");
            continue;
        }
        let replace = |bytes: &[u8]| -> Vec<u8> {
            String::from_utf8(bytes.to_vec())
                .map(|t| t.replace(root_text, "@ROOT@").into_bytes())
                .unwrap_or_else(|_| bytes.to_vec())
        };
        let stdout = replace(&run.stdout);
        let stderr = String::from_utf8_lossy(&replace(&run.stderr)).into_owned();
        let exit = run.status.code().unwrap_or(-1);
        let port = if stdout.is_empty() {
            Rust::Refused(stderr.clone())
        } else {
            Rust::Report(String::from_utf8_lossy(&stdout).into_owned(), exit == 0)
        };
        record(rel, name, case, &port);
        if let Err(why) = judge_process(case, exit, &stdout, &stderr) {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    cases
}

/// The comparison is not vacuous: an altered expectation, a refusal where the Python reported,
/// a report where the Python crashed and a misplaced stricter entry each fail it.
#[test]
fn negative_replay_detects_a_changed_outcome() {
    let cases: BTreeMap<String, Value> = negative("coverage/negative-validate.json")
        .into_iter()
        .collect();
    let discovered = discovered_items();
    let base = text("coverage/mapping.json");
    let run = |name: &str| {
        let case = &cases[name];
        let map = pyjson::parse_str(&variant(case, &base)).unwrap();
        (
            case.clone(),
            Rust::of(coverage::validate_text(&map, &discovered)),
        )
    };
    // An explicit report: identical when replayed, a failure once one byte of it changes.
    let (mut case, rust) = run("lane-null-member");
    assert_eq!(compare(&case, &rust, false), Ok(()));
    let altered = s(case.get("stdout").unwrap()).replacen("None", "none", 1);
    case.as_dict_mut()
        .unwrap()
        .insert("stdout", Value::from(altered));
    assert!(compare(&case, &rust, false).is_err());
    // A crash: the port's refusal passes; a report in its place, or a flipped exit status, fails.
    let (case, rust) = run("lanes-null");
    assert!(matches!(rust, Rust::Refused(_)));
    assert_eq!(compare(&case, &rust, false), Ok(()));
    assert!(compare(&case, &Rust::Report(String::new(), false), false).is_err());
    let mut flipped = case.clone();
    flipped.as_dict_mut().unwrap().insert("exit", Value::Int(0));
    assert!(compare(&flipped, &rust, false).is_err());
    // A success: a refusal fails it, and so does listing it as stricter.
    let (case, rust) = run("lanes-dict-known");
    assert_eq!(compare(&case, &rust, false), Ok(()));
    assert!(compare(&case, &Rust::Refused("x".into()), false).is_err());
    assert!(compare(&case, &rust, true).is_err());
}
