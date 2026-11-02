use super::*;

/// What the port did with one input.
#[derive(Debug)]
pub(crate) enum Rust {
    /// A report on stdout and whether the command succeeded (exit status 0 or 1).
    Report(String, bool),
    /// A refusal: the error the binary prints after `jope-records: `, exit status 1.
    Refused(String),
}

impl Rust {
    pub(crate) fn of(result: jope_records::Result<(String, bool)>) -> Self {
        match result {
            Ok((stdout, ok)) => Rust::Report(stdout, ok),
            Err(error) => Rust::Refused(error.to_string()),
        }
    }

    fn exit(&self) -> i64 {
        match self {
            Rust::Report(_, true) => 0,
            _ => 1,
        }
    }
}

/// The negative cases of a fixture, by name.
fn negative(rel: &str) -> Vec<(String, Value)> {
    let doc = json(rel);
    assert!(
        doc.get("source").and_then(Value::as_str).is_some()
            && doc.get("capture").and_then(Value::as_str).is_some(),
        "{rel}: names its capture run and command"
    );
    doc.get("cases")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|c| (s(c.get("name").unwrap()).to_owned(), c.clone()))
        .collect()
}

/// A case's input: its `ed` script applied to `base`, checked against the recorded SHA-256 of the
/// file the Python read (the base itself when the case has no script).
pub(crate) fn variant(case: &Value, base: &str) -> String {
    let Some(ed) = case.get("ed") else {
        return base.to_owned();
    };
    let text = apply_ed(base, s(ed));
    assert_eq!(
        py::sha256_hex(text.as_bytes()),
        s(case.get("sha256").unwrap()),
        "{}: the replayed input is not the one the Python read",
        case.get("name").or(case.get("path")).map_or("?", s)
    );
    text
}

/// Hold one case to the contract; `Err` names the first disagreement.
pub(crate) fn compare(case: &Value, rust: &Rust, stricter: bool) -> Result<(), String> {
    let exit = case.get("exit").and_then(Value::as_int).unwrap();
    if stricter {
        return match (exit, rust) {
            (0, Rust::Refused(message)) if !message.is_empty() => Ok(()),
            _ => Err(format!(
                "listed as stricter than the Python, but the Python exited {exit} and the port gave {rust:?}"
            )),
        };
    }
    if rust.exit() != exit {
        return Err(format!(
            "exit status: the Python {exit}, the port {} ({rust:?})",
            rust.exit()
        ));
    }
    let stdout = case.get("stdout").map(|v| s(v).to_owned());
    if let Some(raised) = case.get("raised") {
        // The Python crashed: the port refuses and prints nothing.
        return match rust {
            Rust::Refused(message) if !message.is_empty() => Ok(()),
            _ => Err(format!(
                "the Python raised {}; the port did not refuse: {rust:?}",
                raised.repr()
            )),
        };
    }
    let stderr = s(case.get("stderr").unwrap());
    match rust {
        Rust::Refused(message) => {
            if stderr.is_empty() || stdout.as_deref().is_some_and(|out| !out.is_empty()) {
                return Err(format!(
                    "the port refused ({message:?}) where the Python printed stdout {stdout:?} and stderr {stderr:?}"
                ));
            }
            if format!("{message}\n") != stderr {
                return Err(format!(
                    "refusal message: the Python {stderr:?}, the port {message:?}"
                ));
            }
            Ok(())
        }
        Rust::Report(out, _) => {
            if !stderr.is_empty() {
                return Err(format!(
                    "the Python refused with {stderr:?}; the port reported {out:?}"
                ));
            }
            match (&stdout, case.get("stdout_sha256")) {
                (Some(want), _) if out != want => {
                    Err(format!("stdout: the Python {want:?}, the port {out:?}"))
                }
                (None, Some(sha)) if py::sha256_hex(out.as_bytes()) != s(sha) => Err(format!(
                    "stdout SHA-256: the Python {}, the port {} ({} bytes)",
                    s(sha),
                    py::sha256_hex(out.as_bytes()),
                    out.len()
                )),
                (None, None) => Err("the case records no stdout".to_owned()),
                _ => Ok(()),
            }
        }
    }
}

/// With `JOPE_NEGATIVE_REPORT` naming a file, every replayed case appends one tab-separated line
/// to it: the fixture, the case, the Python's exit status and message (its exception line, the
/// first line it printed on stderr, or the first line of its report) and the port's (its refusal,
/// or the first line of its report). The custody runs keep this table; the test never reads it.
fn record(rel: &str, name: &str, case: &Value, rust: &Rust) {
    let Some(path) = std::env::var_os("JOPE_NEGATIVE_REPORT") else {
        return;
    };
    let first = |text: &str| -> String {
        text.lines()
            .find(|l| !l.is_empty())
            .unwrap_or("")
            .chars()
            .take(160)
            .collect()
    };
    let python = match (case.get("raised"), case.get("stderr").map(s)) {
        (Some(raised), _) => format!("raised {}", s(raised)),
        (None, Some(stderr)) if !stderr.is_empty() => format!("refused {}", first(stderr)),
        _ => format!(
            "report {}",
            case.get("stdout")
                .map_or("(stdout by SHA-256)".to_owned(), |o| first(s(o)))
        ),
    };
    let port = match rust {
        Rust::Refused(message) => format!("refused {}", first(message)),
        Rust::Report(out, _) => format!("report {}", first(out)),
    };
    let line = format!(
        "{rel}\t{name}\t{}\t{python}\t{}\t{port}\n",
        case.get("exit").and_then(Value::as_int).unwrap(),
        rust.exit()
    );
    use std::io::Write as _;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .unwrap();
    file.write_all(line.as_bytes()).unwrap();
}

/// Replay every case of `rel` through `run` (the case and its input text) and require the
/// contract of each; the number of cases is pinned so a fixture cannot shrink unnoticed.
fn replay(rel: &str, base: &str, count: usize, run: &dyn Fn(&Value, &str) -> Rust) {
    let cases = negative(rel);
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let stricter: BTreeSet<&str> = RUST_STRICTER
        .iter()
        .filter(|(fixture, _, _)| *fixture == rel)
        .map(|(_, name, _)| *name)
        .collect();
    let mut failures = vec![];
    for (name, case) in &cases {
        let rust = run(case, &variant(case, base));
        record(rel, name, case, &rust);
        if let Err(why) = compare(case, &rust, stricter.contains(name.as_str())) {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    for name in &stricter {
        assert!(
            cases.iter().any(|(n, _)| n == name),
            "{rel}: RUST_STRICTER names {name}, which is not a case"
        );
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

fn discovered_items() -> Dict {
    json("coverage/discovered.json")
        .get("items")
        .and_then(Value::as_dict)
        .unwrap()
        .clone()
}

#[test]
fn coverage_validate_refuses_every_map_the_python_refused() {
    let discovered = discovered_items();
    replay(
        "coverage/negative-validate.json",
        &text("coverage/mapping.json"),
        55,
        &|_, mapping| {
            let parsed = pyjson::parse_str(mapping)
                .map_err(|m| jope_records::Error::Invalid(format!("coverage map: {m}")));
            Rust::of(parsed.and_then(|map| coverage::validate_text(&map, &discovered)))
        },
    );
}

#[test]
fn coverage_discover_refuses_every_metadata_the_python_refused() {
    let files = text("coverage/files.txt");
    let workflows = json("coverage/workflows.json");
    let read = |name: &str| -> jope_records::Result<String> {
        Ok(s(workflows.get(name).unwrap()).to_owned())
    };
    replay(
        "coverage/negative-discover.json",
        &text("coverage/metadata.json"),
        59,
        &|_, metadata| {
            let result = pyjson::parse_str(metadata)
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| {
                    coverage::discover_from(&meta, files.lines(), Path::new("@ROOT@"), &read)
                });
            Rust::of(result.map(|items| (coverage::discover_document(items), true)))
        },
    );
}

#[test]
fn rust_scope_check_refuses_every_metadata_the_python_refused() {
    let scope = json("scope/scope.json");
    // The base metadata is the recorded workspace, as `cargo metadata` lists it.
    let names: Vec<String> = json("scope/metadata.json")
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p.get("name").unwrap()).to_owned())
        .collect();
    let recorded: Vec<String> = scope
        .get("packages")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|p| s(p).to_owned())
        .collect();
    assert_eq!(
        names, recorded,
        "scope/metadata.json lists scope.json's packages"
    );
    replay(
        "scope/negative.json",
        &text("scope/metadata.json"),
        17,
        &|_, metadata| {
            let result = pyjson::parse_str(metadata)
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| scope_check::workspace_packages(&meta))
                .and_then(|workspace| {
                    scope_check::evaluate(
                        &workspace,
                        s(scope.get("product_stdout").unwrap()),
                        s(scope.get("deferred_stdout").unwrap()),
                        s(scope.get("script").unwrap()),
                    )
                });
            Rust::of(result)
        },
    );
}

#[test]
fn parity_check_refuses_every_metadata_the_python_refused() {
    let doc = json("parity/negative.json");
    let present: BTreeSet<String> = doc
        .get("exists")
        .and_then(Value::as_list)
        .unwrap()
        .iter()
        .map(|v| s(v).to_owned())
        .collect();
    let matrix = text("parity/negative-matrix.md");
    // The metadata's `@ROOT@` stands for the checked tree; the binaries' paths are made relative
    // to it lexically, so any canonical directory will do.
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let root_text = root.to_str().unwrap().to_owned();
    replay(
        "parity/negative.json",
        &text("parity/negative-metadata.json"),
        32,
        &|_, metadata| {
            let result = pyjson::parse_str(&metadata.replace("@ROOT@", &root_text))
                .map_err(|m| jope_records::Error::Invalid(format!("cargo metadata: {m}")))
                .and_then(|meta| parity::bins_from_metadata(&meta, &root))
                .and_then(|bins| {
                    let tree = parity::Tree {
                        clap: vec![],
                        bins,
                        tools: vec![],
                        jope_py: None,
                        units: vec![],
                    };
                    parity::evaluate(&matrix, &tree, None, &|head: &str| present.contains(head))
                });
            Rust::of(result)
        },
    );
}

#[test]
fn readme_benchmarks_refuses_every_evidence_the_python_refused() {
    replay_readme("readme/negative.json", 85);
}

/// Round two: NaN and infinity formatting, Python 3.12's `sorted`, `max` and `sum` over the
/// same-cut seconds (a NaN among them, compensated float sums), the tie between NaN RMSEs, the
/// short-circuit of `a == b == c`, `repr` of non-printable characters and `$` before a final
/// newline.
#[test]
fn readme_benchmarks_matches_round_two() {
    replay_readme("readme/negative-r2.json", 18);
}

/// Round three: an integer of more than 4300 digits (either sign) in an evidence file, which
/// Python 3.12's `json.loads` refused to convert even in a field nothing reads; 19 digits render.
/// Its addendum: `int()` of a CSV cell of 4300 digits renders and of 4301 refuses, leading zeros
/// counted, whether written plain, with `+` or between spaces (`0_` separators, which Python also
/// read, are refused by the strict number grammar since round four).
#[test]
fn readme_benchmarks_matches_round_three() {
    replay_readme("readme/negative-r3.json", 12);
}

/// Round four (`readme/negative-r4.json`, review 5334300806 of PR #117): `int()` and `float()` of
/// README CSV cells, which the port reads with the strict number grammar. The reviewer's two
/// cells (U+001F before a published `datasets_scored` and before a benchmark-valid
/// `average_rank`), which Python refused although `str.strip()` removes U+001F, are refused;
/// so is every other input Python refused. ASCII whitespace, a sign, leading zeros and exponent
/// forms render as Python rendered them; Unicode whitespace, underscores, other Unicode digits,
/// `nan` and the infinities, which Python read, are refused (`RUST_STRICTER`).
/// Round five (`readme/negative-r5.json`): `sorted(reports.glob("*.bench-report.json"))` lists
/// entries of every kind and names that are not UTF-8 (one such extra report made the Python
/// refuse the count, while the port had skipped it and rendered); a directory or dangling link so
/// named, a report or directory that cannot be read, and an input under a directory that may not
/// be searched all refuse.
#[test]
fn readme_benchmarks_matches_round_five() {
    replay_readme("readme/negative-r5.json", 9);
}

#[test]
fn readme_benchmarks_matches_round_four() {
    replay_readme("readme/negative-r4.json", 37);
}

/// Round eight (`readme/negative-r8.json`, review 5337854418 of PR #117): Python's integers are
/// exact. The per-method correction totals (`get(m, 0) + applied[id]`) pass 64 bits and return
/// (the reviewer's `[2^63-1, 1, -(2^63-1), 45]`, which Python refused as 46 and the port had
/// accepted as 45), leave it below `-2^63`, are printed beyond it in a refusal, and become a float
/// only when a float operand appears; `before - after` at the extremes of `i64` (which a release
/// build had wrapped); and the same-cut seconds as CPython 3.12 sums, divides, sorts and compares
/// them: the C `long` fast path, exact integers and plain float addition after it overflows,
/// Neumaier compensation otherwise (integers added plainly), integer true division rounded once,
/// and `sorted`, `min` and `max` comparing an int with a float exactly.
#[test]
fn readme_benchmarks_matches_round_eight() {
    replay_readme("readme/negative-r8.json", 23);
}

fn replay_readme(rel: &str, count: usize) {
    let doc = json(rel);
    let stale = text("readme/README.stale.md");
    let cases = negative(rel);
    assert_eq!(cases.len(), count, "{rel}: every captured case is present");
    let stricter: BTreeSet<&str> = RUST_STRICTER
        .iter()
        .filter(|(fixture, _, _)| *fixture == rel)
        .map(|(_, name, _)| *name)
        .collect();
    assert_eq!(s(doc.get("root").unwrap()), "readme/root");
    let mut failures = vec![];
    for (name, case) in &cases {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("root");
        copy_tree(&fixture().join("readme/root"), &root);
        for edit in case.get("edits").and_then(Value::as_list).unwrap() {
            let path = s(edit.get("path").unwrap());
            let base = text(&format!("readme/root/{path}"));
            fs::write(root.join(path), variant(edit, &base)).unwrap();
        }
        // Copies (round five: also to a name that is not UTF-8), directories and symlinks.
        tree_extras(case, &root);
        fs::write(root.join("README.md"), &stale).unwrap();
        let effective = apply_modes(case, &root, false);
        let rust = Rust::of(readme::run(&root, true));
        apply_modes(case, &root, true);
        if !effective {
            eprintln!("{rel} {name}: skipped, this user may search a directory of that mode");
            continue;
        }
        record(rel, name, case, &rust);
        let is_stricter = stricter.contains(name.as_str());
        let differs = README_MESSAGE_DIFFERS
            .iter()
            .any(|(fixture, n, _)| *fixture == rel && n == name);
        let mut outcome = if differs {
            // Both refuse, in different words: the Python's explicit refusal (nothing on stdout)
            // and the port's.
            let python_refused = case.get("exit").and_then(Value::as_int) == Some(1)
                && case.get("raised").is_none()
                && case.get("stdout").is_some_and(|o| s(o).is_empty())
                && case.get("stderr").is_some_and(|e| !s(e).is_empty());
            match &rust {
                Rust::Refused(message) if python_refused && !message.is_empty() => Ok(()),
                _ => Err(format!(
                    "listed as refused in other words; the Python {:?}, the port {rust:?}",
                    case.get("stderr")
                )),
            }
        } else {
            compare(case, &rust, is_stricter)
        };
        if outcome.is_ok() {
            // What `--write` left in README.md: the Python's bytes where it rendered, the stale
            // README untouched where it (or, for a stricter case, the port) refused.
            let written = fs::read_to_string(root.join("README.md")).unwrap();
            let want = match case.get("readme_sha256") {
                Some(sha) if !is_stricter => s(sha).to_owned(),
                _ => py::sha256_hex(stale.as_bytes()),
            };
            if py::sha256_hex(written.as_bytes()) != want {
                outcome = Err("README.md differs from what the Python left".to_owned());
            }
            // Python 3.12's limit on integer conversion is refused in CPython's words, unless the
            // strict number grammar refused the text before it counted digits.
            let limit = case
                .get("raised")
                .and_then(|raised| s(raised).strip_prefix("ValueError: "))
                .filter(|message| message.starts_with("Exceeds the limit ("));
            let earlier = REFUSED_BEFORE_THE_LIMIT
                .iter()
                .any(|(fixture, n, _)| *fixture == rel && n == name);
            if let (Some(limit), Rust::Refused(message), false) = (limit, &rust, earlier) {
                if !message.contains(limit) {
                    outcome = Err(format!(
                        "refusal message: CPython's {limit:?}, the port {message:?}"
                    ));
                }
            }
        }
        if let Err(why) = outcome {
            failures.push(format!("{rel} {name}: {why}"));
        }
    }
    for name in &stricter {
        assert!(
            cases.iter().any(|(n, _)| n == name),
            "RUST_STRICTER names {name}"
        );
    }
    for (fixture, name, _) in REFUSED_BEFORE_THE_LIMIT {
        if *fixture == rel {
            let case = cases.iter().find(|(n, _)| n == name).map(|(_, c)| c);
            assert!(
                case.is_some_and(|c| c.get("raised").is_some()),
                "REFUSED_BEFORE_THE_LIMIT names {name}, which is not a case the Python crashed on"
            );
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

// ------------------------------------------------------------------------------------------------
// Round two: whole-process cases (see `process.json`).
