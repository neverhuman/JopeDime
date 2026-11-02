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
