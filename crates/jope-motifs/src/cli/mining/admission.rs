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
