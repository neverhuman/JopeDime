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
