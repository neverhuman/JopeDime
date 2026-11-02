use super::*;

#[test]
fn mine_writes_the_private_table_and_every_input_identity() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let lines = fixture::lines();
    let state = FakeState::of(&lines);
    let out = directory.path().join("mined");
    let run = invoke_in(
        &state_env(state.clone()),
        &["jope-motifs", "mine", "--out", utf8(&out)],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert!(run.err.is_empty());

    let table = read_table(&out);
    assert!(table.verify().is_ok());
    let set = StateTeaching::new(&state).qualified().expect("reads");
    assert_eq!(
        table,
        mine(&set, None, MineParameters::default()).expect("mines")
    );
    assert_eq!(table.body.motifs, fixture::table().body.motifs);
    let inputs = read_inputs(&out);
    assert_eq!(inputs, input_manifest(&set, &table).expect("inputs"));
    assert_eq!(inputs.records.len(), lines.len());

    assert_eq!(
        run.out,
        format!(
            "Wrote a private table with {} fragments (all quarantined).\n",
            table.body.motifs.len()
        )
    );
    assert_clean(&run.out, &table);
    let mut names: Vec<String> = std::fs::read_dir(&out)
        .expect("listing")
        .map(|entry| {
            entry
                .expect("entry")
                .file_name()
                .to_string_lossy()
                .into_owned()
        })
        .collect();
    names.sort();
    assert_eq!(names, [INPUTS_FILE, TABLE_FILE]);
}

#[cfg(unix)]
#[test]
fn mine_output_is_a_new_0700_directory_of_0600_files() {
    use std::os::unix::fs::PermissionsExt;
    let directory = tempfile::tempdir().expect("temporary directory");
    let out = directory.path().join("mined");
    let env = state_env(FakeState::of(&fixture::small_lines()));
    let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    let mode = |path: &Path| {
        std::fs::metadata(path)
            .expect("metadata")
            .permissions()
            .mode()
            & 0o777
    };
    assert_eq!(mode(&out), 0o700);
    assert_eq!(mode(&out.join(TABLE_FILE)), 0o600);
    assert_eq!(mode(&out.join(INPUTS_FILE)), 0o600);
    let before = std::fs::read(out.join(TABLE_FILE)).expect("table");

    // An existing directory is never reused or overwritten.
    let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: output_path_refused\n");
    assert_eq!(std::fs::read(out.join(TABLE_FILE)).expect("table"), before);
}

#[test]
fn mine_json_stdout_passes_lint_and_holds_no_canary() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let families = directory.path().join("families.json");
    std::fs::write(&families, fixture::family_map_bytes()).expect("write family map");
    let out = directory.path().join("mined");
    let mut state = FakeState::of(&fixture::labelled_small_lines());
    state.read.excluded = 2;
    let run = invoke_in(
        &state_env(state),
        &[
            "jope-motifs",
            "mine",
            "--out",
            utf8(&out),
            "--families",
            utf8(&families),
            "--min-families",
            "2",
            "--json",
        ],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert!(run.err.is_empty());
    assert_eq!(run.out.lines().count(), 1);

    let table = read_table(&out);
    fixture::assert_carried(&table);
    assert_eq!(
        table.body.motifs,
        fixture::labelled_small_table().body.motifs
    );
    let value: Value = serde_json::from_str(&run.out).expect("one JSON document");
    assert_eq!(value["schema"], MINE_SCHEMA);
    assert_eq!(value["ok"], Value::Bool(true));
    assert_eq!(value["table_sha256"], table.table_sha256.as_str());
    assert_eq!(value["fragments"], 1);
    assert_eq!(value["quarantined"], 1);
    assert_eq!(value["stage_counts"]["s2"], 1);
    assert_eq!(value["parameters"]["min_datasets"], 2);
    assert_eq!(value["parameters"]["min_families"], 2);
    assert_eq!(value["parameters"]["max_stages"], 6);
    assert_eq!(value["parameters"]["family_map"], Value::Bool(true));
    assert_eq!(value["inputs"]["qualified"], 2);
    assert_eq!(value["inputs"]["used"], 2);
    assert_eq!(value["inputs"]["excluded_at_source"], 2);
    assert_eq!(value["inputs"]["excluded_adaptive"], 0);
    assert_clean(&run.out, &table);
}

#[test]
fn mine_refuses_output_under_state_root() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let resolved = directory.path().join("resolved");
    let explicit = directory.path().join("explicit");
    std::fs::create_dir(&resolved).expect("resolved state directory");
    std::fs::create_dir(&explicit).expect("explicit state directory");
    // `resolved` stands for the product's resolved default root.
    let env = env_with(
        Policy::for_test(Vec::new(), false).with_state_root(resolved.clone()),
        unopened(),
    );
    for (state, target) in [
        (None, resolved.join("mined")),
        (Some(&explicit), explicit.join("mined")),
        (Some(&explicit), resolved.join("mined")),
    ] {
        let mut arguments = vec!["jope-motifs", "mine", "--out", utf8(&target)];
        if let Some(state) = state {
            arguments.extend(["--state", utf8(state)]);
        }
        let run = invoke_in(&env, &arguments);
        assert_eq!(run.status, 2, "{arguments:?}");
        assert!(run.out.is_empty(), "{arguments:?}");
        assert_eq!(run.err, "error: output_path_refused\n", "{arguments:?}");
        assert!(target.symlink_metadata().is_err(), "{arguments:?}");
    }

    // The private log is refused under an explicit state root as well.
    let log = explicit.join("log.txt");
    let out = directory.path().join("mined");
    let run = invoke_in(
        &env,
        &[
            "jope-motifs",
            "mine",
            "--state",
            utf8(&explicit),
            "--out",
            utf8(&out),
            "--private-log",
            utf8(&log),
        ],
    );
    assert_eq!(run.status, 2);
    assert_eq!(
        run.err,
        "error: output_path_refused\nwarning: private log not written\n"
    );
    assert!(log.symlink_metadata().is_err());
    assert!(out.symlink_metadata().is_err());
}

#[test]
fn mine_with_zero_admitted_programs_writes_nothing() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let out = directory.path().join("mined");
    let mut empty = FakeState::of(&[]);
    empty.read.excluded = 4;
    let env = state_env(empty);
    let run = invoke_in(&env, &["jope-motifs", "mine", "--out", utf8(&out)]);
    assert_eq!(run.status, 2);
    assert!(run.out.is_empty());
    assert_eq!(run.err, "error: no_admitted_programs\n");
    assert!(out.symlink_metadata().is_err());

    let run = invoke_in(
        &env,
        &["jope-motifs", "mine", "--out", utf8(&out), "--json"],
    );
    assert_eq!(run.status, 2);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["schema"], MINE_SCHEMA);
    assert_eq!(value["ok"], Value::Bool(false));
    assert_eq!(value["error"], "no_admitted_programs");
    assert!(out.symlink_metadata().is_err());
}

#[test]
fn mine_without_a_reader_exits_2_state_unavailable() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let out = directory.path().join("mined");
    // No reader at all.
    let run = invoke(&["jope-motifs", "mine", "--out", utf8(&out)]);
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: state_unavailable\n");
    assert!(out.symlink_metadata().is_err());

    let mut broken = FakeState::of(&fixture::small_lines());
    broken.broken = true;
    let run = invoke_in(
        &state_env(broken),
        &["jope-motifs", "mine", "--out", utf8(&out)],
    );
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: state_unavailable\n");
    assert!(out.symlink_metadata().is_err());
}

/// A root that holds no State (here an existing directory without the HWM chain tip, as a
/// mistyped `--state` would be) and a missing root are both `state_unavailable` for both
/// verbs, never an empty State, and nothing is written. The standard opener is the real
/// learning adapter.
#[test]
fn verbs_over_a_root_without_a_state_write_nothing() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let root = directory.path().join("state");
    std::fs::create_dir(&root).expect("state root");
    let out = directory.path().join("mined");
    let env = env_with(
        Policy::for_test(Vec::new(), false),
        Env::standard().open_state,
    );
    let run = invoke_in(
        &env,
        &[
            "jope-motifs",
            "mine",
            "--state",
            utf8(&root),
            "--out",
            utf8(&out),
        ],
    );
    assert_eq!(run.status, 2, "{}{}", run.out, run.err);
    assert_eq!(run.err, "error: state_unavailable\n");
    assert!(out.symlink_metadata().is_err());

    let table = written(directory.path(), "table.json", &fixture::small_table());
    let run = invoke_in(
        &env,
        &[
            "jope-motifs",
            "validate",
            "--table",
            &table,
            "--state",
            utf8(&root),
        ],
    );
    assert_eq!(run.status, 2, "{}{}", run.out, run.err);
    assert!(run.out.is_empty());
    assert_eq!(run.err, "error: state_unavailable\n");

    let absent = directory.path().join("absent");
    let run = invoke_in(
        &env,
        &[
            "jope-motifs",
            "mine",
            "--state",
            utf8(&absent),
            "--out",
            utf8(&out),
        ],
    );
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: state_unavailable\n");
    assert!(absent.symlink_metadata().is_err());
    assert!(out.symlink_metadata().is_err());
}

#[test]
fn mine_refuses_unqualified_unburned_or_unauthenticated_records() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let out = directory.path().join("mined");
    let mut unburned = FakeState::of(&fixture::small_lines());
    unburned.read.entries[0].original_label_role = String::from("test");
    let mut unqualified = FakeState::of(&fixture::small_lines());
    unqualified.read.entries[1].improved_folds = 3;
    let mut unauthenticated = FakeState::of(&fixture::small_lines());
    unauthenticated.pointers[1].sealed = true;
    for (state, code) in [
        (unburned, "label_burn_missing"),
        (unqualified, "record_unqualified"),
        (unauthenticated, "record_unauthenticated"),
    ] {
        let run = invoke_in(
            &state_env(state),
            &["jope-motifs", "mine", "--out", utf8(&out)],
        );
        assert_eq!(run.status, 2, "{code}");
        assert_eq!(run.err, format!("error: {code}\n"));
        assert!(out.symlink_metadata().is_err(), "{code}");
    }

    // The full chain goes only to the private log.
    let log = directory.path().join("log.txt");
    let mut burned = FakeState::of(&fixture::small_lines());
    burned.read.entries[0].label_burn_sha256 = Some(fixture::digest("burn/unexpected"));
    let run = invoke_in(
        &state_env(burned),
        &[
            "jope-motifs",
            "mine",
            "--out",
            utf8(&out),
            "--private-log",
            utf8(&log),
        ],
    );
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: label_origin_refused\n");
    let chain = std::fs::read_to_string(&log).expect("private log written");
    assert!(chain.contains("label_origin_refused"));
    assert!(out.symlink_metadata().is_err());
}

#[test]
fn verbs_refuse_missing_required_arguments_and_bad_limits() {
    for arguments in [
        vec!["jope-motifs"],
        vec!["jope-motifs", "mine"],
        vec!["jope-motifs", "mine", "--state", "/state"],
        vec![
            "jope-motifs",
            "mine",
            "--out",
            "/mined",
            "--max-stages",
            "six",
        ],
        vec!["jope-motifs", "validate"],
        vec!["jope-motifs", "validate", "--state", "/state"],
    ] {
        assert!(
            Cli::try_parse_from(arguments.iter().copied()).is_err(),
            "{arguments:?}"
        );
    }

    let directory = tempfile::tempdir().expect("temporary directory");
    let out = directory.path().join("mined");
    let bad_map = directory.path().join("families.json");
    std::fs::write(&bad_map, b"{\"abc\":\"canaryfamily01\"}").expect("write");
    let missing_map = directory.path().join("absent.json");
    let env = env_with(Policy::for_test(Vec::new(), false), unopened());
    for (extra, code) in [
        (vec!["--max-stages", "7"], "invalid_parameters"),
        (vec!["--max-stages", "1"], "invalid_parameters"),
        (vec!["--min-datasets", "0"], "invalid_parameters"),
        (vec!["--min-families", "2"], "invalid_parameters"),
        (vec!["--families", utf8(&bad_map)], "invalid_family_map"),
        (vec!["--families", utf8(&missing_map)], "invalid_family_map"),
    ] {
        let mut arguments = vec!["jope-motifs", "mine", "--out", utf8(&out)];
        arguments.extend(extra);
        let run = invoke_in(&env, &arguments);
        assert_eq!(run.status, 2, "{arguments:?}");
        assert_eq!(run.err, format!("error: {code}\n"), "{arguments:?}");
        assert!(out.symlink_metadata().is_err(), "{arguments:?}");
    }
}

#[test]
fn validate_with_state_reports_source_verification() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let lines = fixture::labelled_small_lines();
    let state = FakeState::of(&lines);
    let families = directory.path().join("families.json");
    std::fs::write(&families, fixture::family_map_bytes()).expect("write family map");
    let mined = directory.path().join("mined");
    let run = invoke_in(
        &state_env(state.clone()),
        &[
            "jope-motifs",
            "mine",
            "--out",
            utf8(&mined),
            "--families",
            utf8(&families),
        ],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    let table = read_table(&mined);
    fixture::assert_carried(&table);
    let table_path = mined.join(TABLE_FILE);
    // The fake opener serves its State for any root; the root only has to be named.
    let root = directory.path().join("state");

    // Current: every occurrence is confirmed, and the shareable summary says so.
    let summary = directory.path().join("summary.json");
    let run = invoke_in(
        &state_env(state),
        &[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--state",
            utf8(&root),
            "--public-out",
            utf8(&summary),
            "--json",
        ],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["ok"], Value::Bool(true));
    assert_eq!(value["source_verified"], Value::Bool(true));
    assert_eq!(value["public_written"], Value::Bool(true));
    assert!(value["failures"].as_array().expect("failures").is_empty());
    assert_clean(&run.out, &table);
    let text = std::fs::read_to_string(&summary).expect("summary written");
    let shared: Value = serde_json::from_str(&text).expect("json");
    assert_eq!(shared["source_verified"], Value::Bool(true));
    assert_clean(&text, &table);
    assert!(export::scan(&text, &table).is_ok());

    // Stale: one recorded occurrence is no longer a current qualified record. The checks
    // themselves still pass, but the run fails and nothing is shared.
    let stale = FakeState::of(&[
        lines[0].clone(),
        fixture::qualified(lines[1].program(), TaskLabel::Regression, "d3", "s03-other"),
    ]);
    let stale_summary = directory.path().join("stale-summary.json");
    let run = invoke_in(
        &state_env(stale.clone()),
        &[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--state",
            utf8(&root),
            "--public-out",
            utf8(&stale_summary),
            "--json",
        ],
    );
    assert_eq!(run.status, 1, "{}{}", run.out, run.err);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["ok"], Value::Bool(false));
    assert_eq!(value["source_verified"], Value::Bool(false));
    assert_eq!(value["failed"], 0);
    assert_eq!(value["public_written"], Value::Bool(false));
    assert!(!stale_summary.exists());
    // No fragment failed, so the failed run names its reason in a table-wide line.
    let failures = value["failures"].as_array().expect("failures");
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0]["check"], "source");
    assert_eq!(failures[0]["code"], "source_not_confirmed");
    assert!(failures[0].get("fragment").is_none());
    assert!(CHECK_CODES.contains(&("source", "source_not_confirmed")));
    assert_clean(&run.out, &table);
    let report = directory.path().join("report.json");
    let run = invoke_in(
        &state_env(stale),
        &[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--state",
            utf8(&root),
            "--report",
            utf8(&report),
        ],
    );
    assert_eq!(run.status, 1);
    assert!(
        run.out.ends_with("Source records: not confirmed.\n"),
        "{}",
        run.out
    );
    let private: Validation =
        serde_json::from_slice(&std::fs::read(&report).expect("report")).expect("json");
    assert!(!private.source_verified);
    // Equal digests prove nothing alone: the fake State keeps its digest.
    assert_eq!(private.source_digest_equal, Some(true));

    // A State that cannot be read is an error, never an unconfirmed pass.
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        utf8(&table_path),
        "--state",
        utf8(&root),
    ]);
    assert_eq!(run.status, 2);
    assert_eq!(run.err, "error: state_unavailable\n");
}

#[test]
fn empty_table_fails_validation_and_shares_nothing() {
    let directory = tempfile::tempdir().expect("temporary directory");
    let mined = directory.path().join("mined");
    let run = invoke_in(
        &state_env(FakeState::of(&fixture::small_lines())),
        &[
            "jope-motifs",
            "mine",
            "--out",
            utf8(&mined),
            "--min-datasets",
            "3",
            "--json",
        ],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["fragments"], 0);
    assert_eq!(value["inputs"]["qualified"], 2);
    // No fragment was kept, but every examined input identity is preserved.
    assert_eq!(read_inputs(&mined).records.len(), 2);

    let table_path = mined.join(TABLE_FILE);
    let summary = directory.path().join("summary.json");
    let run = invoke(&[
        "jope-motifs",
        "validate",
        "--table",
        utf8(&table_path),
        "--public-out",
        utf8(&summary),
        "--json",
    ]);
    assert_eq!(run.status, 1, "{}{}", run.out, run.err);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["ok"], Value::Bool(false));
    assert_eq!(value["fragments"], 0);
    assert_eq!(value["public_written"], Value::Bool(false));
    assert!(!summary.exists());

    let run = invoke(&["jope-motifs", "validate", "--table", utf8(&table_path)]);
    assert_eq!(run.status, 1);
    assert_eq!(
        run.out,
        "Checked 0 fragments: nothing was examined, so the table does not pass. \
         Promotion evidence: not run.\n"
    );
    assert!(lint_shipped_text(&run.out).is_empty());

    // With --state, the source check examined no occurrence and says so.
    let root = directory.path().join("state");
    let run = invoke_in(
        &state_env(FakeState::of(&fixture::small_lines())),
        &[
            "jope-motifs",
            "validate",
            "--table",
            utf8(&table_path),
            "--state",
            utf8(&root),
            "--json",
        ],
    );
    assert_eq!(run.status, 1, "{}{}", run.out, run.err);
    let value: Value = serde_json::from_str(&run.out).expect("json");
    assert_eq!(value["source_verified"], Value::Bool(false));
    let failures = value["failures"].as_array().expect("failures");
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert_eq!(failures[0]["check"], "source");
    assert_eq!(failures[0]["code"], "source_not_exercised");
    assert!(failures[0].get("fragment").is_none());

    // The one-line summary of an empty table claims nothing about quarantine.
    let human = directory.path().join("mined-human");
    let run = invoke_in(
        &state_env(FakeState::of(&fixture::small_lines())),
        &[
            "jope-motifs",
            "mine",
            "--out",
            utf8(&human),
            "--min-datasets",
            "3",
        ],
    );
    assert_eq!(run.status, 0, "{}{}", run.out, run.err);
    assert_eq!(
        run.out,
        "Wrote a private table with no fragments; every examined input is listed.\n"
    );
    assert!(!run.out.contains("quarantined"));
    assert!(lint_shipped_text(&run.out).is_empty());
    let lower = run.out.to_lowercase();
    for word in OWNER_BANNED {
        assert!(!lower.contains(word), "{word}");
    }
    assert_eq!(read_inputs(&human).records.len(), 2);
}
