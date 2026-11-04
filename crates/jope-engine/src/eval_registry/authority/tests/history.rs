use super::*;

#[test]
fn history_batches_resume_without_activating_and_require_every_occurrence() {
    let fixture = Fixture::new(false);
    let inventory = staged_history_fixture(&fixture);
    let mut authority = fixture.open();
    let stage = authority.history_stage(&inventory).unwrap();
    let sequence = authority.checkpoint.seq;
    assert_eq!(authority.history_stage(&inventory).unwrap(), stage);
    assert_eq!(authority.checkpoint.seq, sequence);
    assert!(state(authority.claim(&fit(), &request()).unwrap()).contains("coverage"));
    assert!(authority.history_commit_coverage(&stage).is_err());
    let first = HistoryBatch {
        stage_sha256: stage.clone(),
        start: 0,
        end: 1,
    };
    let batch = authority.history_import_batch(&first).unwrap();
    let sequence = authority.checkpoint.seq;
    assert_eq!(authority.history_import_batch(&first).unwrap(), batch);
    assert_eq!(authority.checkpoint.seq, sequence);
    assert!(authority
        .history_import_batch(&HistoryBatch {
            stage_sha256: stage.clone(),
            start: 0,
            end: 2
        })
        .is_err());
    assert!(authority.history_commit_coverage(&stage).is_err());
    drop(authority);
    let mut authority = fixture.open();
    authority
        .history_import_batch(&HistoryBatch {
            stage_sha256: stage.clone(),
            start: 1,
            end: 3,
        })
        .unwrap();
    assert_eq!(authority.history_commit_coverage(&stage).unwrap(), stage);
    let sequence = authority.checkpoint.seq;
    authority.history_commit_coverage(&stage).unwrap();
    assert_eq!(authority.checkpoint.seq, sequence);
    assert!(state(authority.claim(&fit(), &request()).unwrap()).contains("inactive"));
    for capability in [
        "synthetic_qualification",
        "regression_collection",
        "regression_training",
        "regression_promotion",
    ] {
        assert!(authority
            .namespace_activate("synthetic", capability)
            .unwrap_err()
            .to_string()
            .contains("unavailable"));
    }
    assert_eq!(
        authority.checkpoint.seq, sequence,
        "activation refusal must not journal"
    );
    assert_eq!(
        authority
            .db
            .query_row("SELECT count(*) FROM entries", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn history_batch_shared_evidence_is_read_once_and_rechecked_on_retry() {
    use std::io::Read;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;

    // Reserve the watcher before the fixture and its database consume descriptors.
    let mut events = match unsafe { libc::inotify_init1(libc::IN_NONBLOCK | libc::IN_CLOEXEC) } {
        fd if fd >= 0 => Some(unsafe { std::fs::File::from_raw_fd(fd) }),
        _ => {
            let error = std::io::Error::last_os_error();
            if !matches!(
                error.raw_os_error(),
                Some(libc::EMFILE | libc::ENFILE | libc::ENOMEM)
            ) {
                panic!("inotify_init1 failed: {error}");
            }
            eprintln!("skipping physical open-count observation: {error}");
            None
        }
    };
    let fixture = Fixture::new(false);
    let mut inventory = staged_history_fixture(&fixture);
    let original = inventory.occurrences[0].clone();
    let other_evidence = inventory.occurrences[1..].to_vec();
    inventory.occurrences = (0..64)
        .map(|i| HistoricalExclusion {
            reason: format!("separate original occurrence {i}"),
            ..original.clone()
        })
        .collect();
    inventory.occurrences.extend(other_evidence);
    let mut authority = fixture.open();
    let stage = authority.history_stage(&inventory).unwrap();
    let evidence = fixture.root.join("objects").join(&original.evidence_sha256);
    // Observe real opens, not a timing threshold or a replacement verifier.
    // OPEN and CLOSE events alternate, so inotify cannot coalesce repeated opens.
    let path = std::ffi::CString::new(evidence.as_os_str().as_bytes()).unwrap();
    if let Some(events) = events.as_ref() {
        assert!(
            unsafe {
                libc::inotify_add_watch(
                    events.as_raw_fd(),
                    path.as_ptr(),
                    libc::IN_OPEN | libc::IN_CLOSE_NOWRITE,
                )
            } >= 0
        );
    }
    let batch = HistoryBatch {
        stage_sha256: stage,
        start: 0,
        end: inventory.occurrences.len(),
    };
    let result = authority.history_import_batch(&batch).unwrap();
    if let Some(events) = events.as_mut() {
        let mut bytes = [0u8; 8192];
        let n = events.read(&mut bytes).unwrap();
        let mut offset = 0;
        let mut opens = 0;
        while offset < n {
            let mask = u32::from_ne_bytes(bytes[offset + 4..offset + 8].try_into().unwrap());
            let length = u32::from_ne_bytes(bytes[offset + 12..offset + 16].try_into().unwrap());
            assert_eq!(mask & libc::IN_Q_OVERFLOW, 0);
            opens += usize::from(mask & libc::IN_OPEN != 0);
            offset += 16 + length as usize;
        }
        assert_eq!(
            opens, 1,
            "one physical verification per distinct batch evidence"
        );
    }
    assert_eq!(
        authority
            .db
            .query_row("SELECT count(*) FROM exclusions", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        66
    );
    let sequence = authority.checkpoint.seq;
    assert_eq!(authority.history_import_batch(&batch).unwrap(), result);
    assert_eq!(authority.checkpoint.seq, sequence);
    fs::set_permissions(&evidence, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&evidence, b"corrupted original shared evidence").unwrap();
    fs::set_permissions(&evidence, fs::Permissions::from_mode(0o444)).unwrap();
    assert!(authority.history_import_batch(&batch).is_err());
    assert_eq!(authority.checkpoint.seq, sequence);
    assert_eq!(
        authority
            .db
            .query_row("SELECT count(*) FROM history_batches", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        authority
            .db
            .query_row("SELECT count(*) FROM entries", [], |r| r.get::<_, u64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn historical_global_lookup_index_preserves_absent_present_and_null_guard_semantics() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch("CREATE TABLE exclusions(id TEXT PRIMARY KEY,scope TEXT NOT NULL,program TEXT,training TEXT,evidence_sha TEXT NOT NULL,reason TEXT NOT NULL) WITHOUT ROWID; CREATE INDEX exclusions_match ON exclusions(scope,program,training); CREATE TABLE scope_training(scope TEXT,training TEXT); CREATE INDEX scope_training_hash ON scope_training(training); INSERT INTO scope_training VALUES('selected','selected-training');").unwrap();
    for (id, scope, program, training) in [
        ("program", "elsewhere", Some("blocked"), None),
        ("training", "elsewhere", None, Some("blocked-training")),
        ("wild", "selected", None, None),
        ("joint", "elsewhere", Some("joint"), Some("joint-training")),
    ] {
        db.execute(
            "INSERT INTO exclusions VALUES(?,?,?,?,?,?)",
            params![id, scope, program, training, ZERO, id],
        )
        .unwrap();
    }
    for i in 0..10_000 {
        db.execute(
            "INSERT INTO exclusions VALUES(?,?,?,?,?,?)",
            params![
                format!("filler-{i}"),
                "elsewhere",
                format!("other-program-{i}"),
                Option::<String>::None,
                ZERO,
                "retained"
            ],
        )
        .unwrap();
    }
    let cases = [
        ("absent", "other", false),
        ("blocked", "other", true),
        ("absent", "blocked-training", true),
        ("absent", "selected-training", true),
        ("joint", "joint-training", true),
        ("joint", "other", false),
    ];
    let all = HISTORY_LOOKUP_SQL.replace(" LIMIT 1", "");
    let lookup = |sql: &str, program: &str, training: &str| {
        db.prepare(sql)
            .unwrap()
            .query_map(params![program, training, training], |r| {
                r.get::<_, String>(0)
            })
            .unwrap()
            .collect::<std::result::Result<BTreeSet<_>, _>>()
            .unwrap()
    };
    let before: Vec<_> = cases.iter().map(|(p, t, _)| lookup(&all, p, t)).collect();
    Authority::create_history_lookup_index(&db).unwrap();
    for ((program, training, blocked), original) in cases.iter().zip(before) {
        assert_eq!(lookup(&all, program, training), original);
        assert_eq!(
            !lookup(HISTORY_LOOKUP_SQL, program, training).is_empty(),
            *blocked
        );
        let plans: Vec<String> = db
            .prepare(&format!("EXPLAIN QUERY PLAN {HISTORY_LOOKUP_SQL}"))
            .unwrap()
            .query_map(params![program, training, training], |r| r.get(3))
            .unwrap()
            .collect::<std::result::Result<_, _>>()
            .unwrap();
        assert!(
            plans.iter().any(|p| p.contains("exclusions_global_match")),
            "{plans:?}"
        );
        assert!(
            !plans.iter().any(|p| p.contains("SCAN exclusions")),
            "{plans:?}"
        );
    }
}

#[test]
fn history_lookup_maintenance_is_explicit_idempotent_and_preserves_original_journal() {
    let fixture = Fixture::new(false);
    let authority = fixture.open();
    authority
        .db
        .execute_batch("DROP INDEX exclusions_global_match")
        .unwrap();
    drop(authority);
    let mut authority = fixture.open();
    assert!(
        !authority
            .db
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='exclusions_global_match')",
                [],
                |r| r.get::<_, bool>(0)
            )
            .unwrap(),
        "ordinary open must not change original schema"
    );
    let journal = fs::read(fixture.root.join("dispatch.jsonl")).unwrap();
    let witness = fs::read(&fixture.witness).unwrap();
    assert_eq!(
        authority.prepare_history_lookup_index().unwrap()["already_present"],
        false
    );
    assert_eq!(
        authority.prepare_history_lookup_index().unwrap()["already_present"],
        true
    );
    assert_eq!(
        fs::read(fixture.root.join("dispatch.jsonl")).unwrap(),
        journal
    );
    assert_eq!(fs::read(&fixture.witness).unwrap(), witness);
    authority.db.execute_batch("DROP INDEX exclusions_global_match; CREATE INDEX exclusions_global_match ON exclusions(scope)").unwrap();
    assert!(
        authority.prepare_history_lookup_index().is_err(),
        "conflicting index cannot be silently adopted"
    );
}

#[test]
fn external_index_tamper_is_detected_even_when_another_valid_event_advanced() {
    let fixture = Fixture::new(false);
    let inventory = staged_history_fixture(&fixture);
    let mut authority = fixture.open();
    authority.import_history(&inventory).unwrap();
    let mut next = inventory.clone();
    next.occurrences[0].reason.push_str(" next retained stage");
    fixture.open().history_stage(&next).unwrap();
    let db = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    db.execute(
        "UPDATE exclusions SET reason='unrelated external tamper'",
        [],
    )
    .unwrap();
    assert!(authority.verify_tail().is_err());
}
