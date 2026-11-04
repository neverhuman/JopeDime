use super::*;

#[test]
fn help_for_every_subcommand_passes_shipped_text_lint_and_owner_ban() {
    let mut command = Cli::command();
    command.build();
    // `build` adds clap's own `help` subcommand; its text is still linted below.
    let verbs: Vec<&str> = command
        .get_subcommands()
        .map(|c| c.get_name())
        .filter(|name| *name != "help")
        .collect();
    assert_eq!(verbs, ["mine", "validate"]);
    let mut texts = vec![
        command.render_long_help().to_string(),
        command.render_help().to_string(),
    ];
    for subcommand in command.get_subcommands_mut() {
        texts.push(subcommand.render_long_help().to_string());
        texts.push(subcommand.render_help().to_string());
    }
    assert!(texts.len() >= 6);
    for flag in [
        "--hosts-per-fragment",
        "--state",
        "--out",
        "--min-datasets",
        "--min-families",
        "--max-stages",
        "--families",
    ] {
        assert!(texts.iter().any(|text| text.contains(flag)), "{flag}");
    }
    let table = fixture::labelled_table();
    fixture::assert_carried(&table);
    let sentinels = fixture::sentinels(&table);
    for text in &texts {
        assert!(lint_shipped_text(text).is_empty(), "{text}");
        let lower = text.to_lowercase();
        for word in OWNER_BANNED {
            assert!(!lower.contains(word), "{word} in {text}");
        }
        assert!(!text.contains(crate::NAME_PREFIX));
        for sentinel in &sentinels {
            assert!(!lower.contains(&sentinel.to_lowercase()), "{sentinel}");
        }
    }
}

#[test]
fn json_stdout_passes_lint_and_holds_no_canary() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let table = fixture::labelled_small_table();
    fixture::assert_carried(&table);
    let path = written(directory.path(), "table.json", &table);
    let run = invoke(&["jope-motifs", "validate", "--table", &path, "--json"]);
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert!(run.err.is_empty());
    assert_eq!(run.out.lines().count(), 1);
    let value: Value = serde_json::from_str(&run.out).expect("one JSON document");
    assert_eq!(value["schema"], VALIDATE_SCHEMA);
    assert_eq!(value["ok"], Value::Bool(true));
    assert_eq!(value["fragments"], 1);
    assert_eq!(value["promotion"], "required_not_run");
    assert_eq!(value["table_sha256"], table.table_sha256.as_str());
    assert_clean(&run.out, &table);
}

#[test]
fn human_output_passes_lint_and_holds_no_canary() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let table = fixture::labelled_small_table();
    fixture::assert_carried(&table);
    let path = written(directory.path(), "table.json", &table);
    let run = invoke(&["jope-motifs", "validate", "--table", &path]);
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert_eq!(
        run.out,
        "Checked 1 fragments: 1 passed, 0 failed. Promotion evidence: not run.\n"
    );
    assert_clean(&run.out, &table);
    assert!(!run.out.contains(&table.table_sha256));
}

#[test]
fn error_codes_are_lint_clean_and_hold_no_reserved_prefix() {
    let codes = ERROR_CODES
        .iter()
        .chain(CHECK_CODES.iter().map(|(_, code)| code));
    for code in codes {
        assert!(lint_shipped_text(code).is_empty(), "{code}");
        assert!(!code.contains(crate::NAME_PREFIX), "{code}");
        assert!(!code.contains("teaching"), "{code}");
        assert!(!code.contains("motif"), "{code}");
        for word in OWNER_BANNED {
            assert!(!code.contains(word), "{code}");
        }
        assert!(
            code.bytes().all(|b| b.is_ascii_lowercase() || b == b'_'),
            "{code}"
        );
    }
    let distinct: BTreeSet<&str> = ERROR_CODES.iter().copied().collect();
    assert_eq!(distinct.len(), ERROR_CODES.len());
    let mut pairs = CHECK_CODES.to_vec();
    pairs.sort_unstable();
    pairs.dedup();
    assert_eq!(pairs.len(), CHECK_CODES.len());
    for (check, _) in CHECK_CODES {
        assert!(lint_shipped_text(check).is_empty(), "{check}");
        let unexercised = if check == "context_compatibility" {
            String::from("context_not_exercised")
        } else {
            format!("{check}_not_exercised")
        };
        assert!(
            CHECK_CODES.contains(&(check, unexercised.as_str())),
            "{check}"
        );
    }
    let table = fixture::labelled_table();
    fixture::assert_carried(&table);
    let sentinels = fixture::sentinels(&table);
    let texts = ERROR_CODES
        .iter()
        .chain(CHECK_CODES.iter().map(|(check, _)| check))
        .chain(CHECK_CODES.iter().map(|(_, code)| code));
    for text in texts {
        for sentinel in &sentinels {
            assert!(!text.contains(sentinel.as_str()), "{sentinel} in {text}");
        }
    }
}

#[test]
fn validate_exit_codes_are_0_1_2() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let passing = written(directory.path(), "passing.json", &fixture::small_table());
    let failing = written(directory.path(), "failing.json", &fixture::refused_table());
    let missing = directory.path().join("missing.json");
    let missing = missing.to_str().expect("utf-8 path");

    let run = invoke(&["jope-motifs", "validate", "--table", &passing]);
    assert_eq!(run.status, 0);
    let run = invoke(&["jope-motifs", "validate", "--table", &failing, "--json"]);
    assert_eq!(run.status, 1);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["ok"], Value::Bool(false));
    assert_eq!(value["failed"], 1);
    let failures = value["failures"].as_array().expect("failures");
    assert!(failures
        .iter()
        .any(|line| line["code"] == "context_not_fitted"
            && line["check"] == "context_compatibility"
            && line["fragment"] == 0));
    // Nothing fitted, so determinism is reported as not exercised, never as passed.
    assert!(failures
        .iter()
        .any(|line| line["code"] == "determinism_not_exercised" && line["check"] == "determinism"));
    assert_eq!(value["checks"]["determinism"]["passed"], 0);
    for line in failures {
        assert!(
            CHECK_CODES
                .iter()
                .any(|(check, code)| line["check"] == *check && line["code"] == *code),
            "{line}"
        );
    }
    let run = invoke(&["jope-motifs", "validate", "--table", missing]);
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: table_unreadable\n");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &passing,
        "--hosts-per-fragment",
        "0",
    ]);
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: invalid_parameters\n");
}

#[test]
fn errors_print_fixed_codes_never_program_text() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let mut tampered = fixture::small_table();
    let key = tampered.body.hosts.keys().next().cloned().expect("a host");
    tampered
        .body
        .hosts
        .insert(key, fixture::canonical("(pipe X (standardize) (ridge))"));
    tampered.table_sha256 = tampered.body.digest().expect("digest");
    let path = written(directory.path(), "tampered.json", &tampered);
    let log = directory.path().join("log.txt");
    let log_path = log.to_str().expect("utf-8 path");

    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &path,
        "--private-log",
        log_path,
    ]);
    assert_eq!(run.status, 2);
    assert!(run.out.is_empty());
    assert_eq!(run.err, "error: provenance_inconsistent\n");
    assert_clean(&run.err, &tampered);
    let chain = std::fs::read_to_string(&log).expect("private log written");
    assert!(chain.contains("provenance_inconsistent"));

    // A log that cannot be written never changes the code, but the operator is told.
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &path,
        "--private-log",
        log_path,
    ]);
    assert_eq!(run.status, 2);
    assert_eq!(
        run.err,
        "error: output_path_refused\nwarning: private log not written\n"
    );
    assert_clean(&run.err, &tampered);
    assert_eq!(std::fs::read_to_string(&log).expect("log kept"), chain);

    let garbage = directory.path().join("garbage.json");
    std::fs::write(&garbage, b"{\"schema\":\"(pipe X (ridge))\"}").expect("write");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        garbage.to_str().expect("utf-8 path"),
        "--json",
    ]);
    assert_eq!(run.status, 2);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["ok"], Value::Bool(false));
    assert_eq!(value["error"], "table_schema");
    assert!(!run.out.contains("pipe"));
    assert!(ERROR_CODES.contains(&"table_schema"));
}

#[test]
fn public_out_is_written_only_after_all_checks_pass() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let failing = written(directory.path(), "failing.json", &fixture::refused_table());
    let refused_summary = directory.path().join("refused-summary.json");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &failing,
        "--public-out",
        refused_summary.to_str().expect("utf-8 path"),
    ]);
    assert_eq!(run.status, 1);
    assert!(!refused_summary.exists());
    assert!(!run.out.contains("Wrote the shareable summary."));

    // No emitter produced an artifact for a multiclass-only fragment, so name absence was
    // not exercised: the run fails and no summary is written.
    let unexercised = written(
        directory.path(),
        "multiclass.json",
        &fixture::multiclass_table(),
    );
    let unexercised_summary = directory.path().join("multiclass-summary.json");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &unexercised,
        "--public-out",
        utf8(&unexercised_summary),
        "--json",
    ]);
    assert_eq!(run.status, 1, "{}{}", run.out, run.err);
    assert!(!unexercised_summary.exists());
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["public_written"], Value::Bool(false));
    assert_eq!(value["checks"]["name_absence"]["passed"], 0);
    let failures = value["failures"].as_array().expect("failures");
    assert!(failures.iter().any(
        |line| line["check"] == "name_absence" && line["code"] == "name_absence_not_exercised"
    ));

    let table = fixture::labelled_small_table();
    fixture::assert_carried(&table);
    let passing = written(directory.path(), "passing.json", &table);
    let summary = directory.path().join("summary.json");
    let report = directory.path().join("report.json");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        &passing,
        "--public-out",
        summary.to_str().expect("utf-8 path"),
        "--report",
        report.to_str().expect("utf-8 path"),
    ]);
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert!(run.out.ends_with("Wrote the shareable summary.\n"));
    let text = std::fs::read_to_string(&summary).expect("summary written");
    assert_clean(&text, &table);
    assert!(!text.contains(&table.table_sha256[..16]));
    assert!(export::scan(&text, &table).is_ok());
    let private: Validation =
        serde_json::from_slice(&std::fs::read(&report).expect("report")).expect("json");
    assert!(private.passed());
    assert_eq!(private.table_sha256, table.table_sha256);
}

#[test]
fn refused_output_paths_exit_2_before_any_work_and_write_nothing() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let table = written(directory.path(), "table.json", &fixture::small_table());
    let refused = |env: &Env, flag: &str, target: &Path| {
        let run = invoke_in(
            env,
            &[
                "jope-motifs",
                "validate",
                "--table",
                &table,
                flag,
                utf8(target),
            ],
        );
        assert_eq!(run.status, 2, "{flag}");
        assert!(run.out.is_empty(), "{flag}");
        // A refused log path is also the path the error chain would have gone to.
        let expected = if flag == "--private-log" {
            "error: output_path_refused\nwarning: private log not written\n"
        } else {
            "error: output_path_refused\n"
        };
        assert_eq!(run.err, expected, "{flag}");
        assert!(
            target.symlink_metadata().is_err(),
            "{flag} wrote {target:?}"
        );
    };

    // The standard policy: this scratch directory lies under the process temporary
    // directory (and, when that is inside a checkout, under a version-controlled path).
    // Whether it also refuses the resolved default state root is not shown here: that
    // would mean changing the process environment. `mine_refuses_output_under_state_root`
    // shows that every configured root is refused.
    let standard = Env::standard();
    refused(&standard, "--report", &directory.path().join("report.json"));
    refused(
        &standard,
        "--public-out",
        &directory.path().join("summary.json"),
    );

    // A version-controlled checkout, for private outputs.
    let checkout = directory.path().join("checkout");
    std::fs::create_dir_all(checkout.join(".git")).expect("checkout marker");
    let versioned = env_with(Policy::for_test(Vec::new(), true), unopened());
    refused(&versioned, "--report", &checkout.join("report.json"));
    refused(&versioned, "--private-log", &checkout.join("log.txt"));

    // The state root, for private and shareable outputs alike.
    let state = directory.path().join("state");
    std::fs::create_dir(&state).expect("state directory");
    let stateful = env_with(
        Policy::for_test(Vec::new(), false).with_state_root(state.clone()),
        unopened(),
    );
    refused(&stateful, "--report", &state.join("report.json"));
    refused(&stateful, "--public-out", &state.join("summary.json"));
}
