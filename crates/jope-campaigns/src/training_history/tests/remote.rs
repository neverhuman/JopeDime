use super::*;

#[test]
fn a_live_tail_is_not_abandoned_but_a_quiescent_one_is() {
    let now = 10_000_000u64;
    assert!(
        !tail_is_abandoned(now - 60_000, now),
        "1 min old: mid-write"
    );
    assert!(
        !tail_is_abandoned(0, now),
        "unknown mtime must never unblock pruning"
    );
    assert!(
        !tail_is_abandoned(now + 3_600_000, now),
        "an mtime in the future is a broken clock, not an old file"
    );
    assert!(tail_is_abandoned(
        now - (TAIL_QUIESCENT_SECS + 60) * 1000,
        now
    ));
}

#[test]
fn remote_quiescence_is_measured_on_the_local_clock_only() {
    let now = 10_000_000u64;
    let stable = SegmentState {
        source_fingerprint: "1383:1".into(),
        // Seen unchanged here for longer than the threshold.
        fingerprint_since_ms: now - (TAIL_QUIESCENT_SECS + 60) * 1000,
        ..Default::default()
    };
    assert!(remote_tail_is_abandoned(
        Some(&stable),
        "1383:1",
        now,
        TAIL_QUIESCENT_SECS
    ));
    // The remote mtime says the file is ancient, but we have only just
    // started watching it: not abandoned. A wrong remote clock cannot make
    // us write off a row that is still being written.
    let fresh = SegmentState {
        source_fingerprint: "1383:1".into(),
        fingerprint_since_ms: now - 5_000,
        ..Default::default()
    };
    assert!(!remote_tail_is_abandoned(
        Some(&fresh),
        "1383:1",
        now,
        TAIL_QUIESCENT_SECS
    ));
    // The file changed since we last looked: definitively live.
    assert!(!remote_tail_is_abandoned(
        Some(&stable),
        "1500:2",
        now,
        TAIL_QUIESCENT_SECS
    ));
    // Never seen before: nothing to claim.
    assert!(!remote_tail_is_abandoned(
        None,
        "1383:1",
        now,
        TAIL_QUIESCENT_SECS
    ));
}

#[test]
fn recording_a_truncation_keeps_the_fragment_and_is_idempotent() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    let complete = format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4));
    let fragment = b"{\"epoch\":3,\"train_lo";
    fs::write(
        &source,
        format!("{complete}{}", String::from_utf8_lossy(fragment)),
    )?;

    let outcome = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(outcome.appended_rows, 2, "every COMPLETE row is archived");
    assert_eq!(outcome.residual_bytes, fragment.len() as u64);
    assert!(!outcome.covered(), "strict coverage is still false");

    let mtime = now_ms() - (TAIL_QUIESCENT_SECS + 60) * 1000;
    assert!(record_truncation(
        &archive,
        &key(),
        "xbabe2",
        "src",
        fragment,
        outcome.source_bytes,
        mtime
    )?);
    // Re-recording the same hole must not grow the ledger.
    assert!(!record_truncation(
        &archive,
        &key(),
        "xbabe2",
        "src",
        fragment,
        outcome.source_bytes,
        mtime
    )?);

    let seg = segment_dir(&archive, &key());
    let state = SegmentState::load(&seg)
        .expect("state could be read")
        .expect("state");
    assert_eq!(state.truncation_count, 1);
    assert_eq!(state.truncated_tail_bytes, fragment.len() as u64);
    assert_eq!(state.archived_rows, 2, "complete rows untouched");

    // The ledger exists per-segment and archive-wide, one line each.
    let seg_ledger = fs::read_to_string(seg.join(TRUNCATION_FILE))?;
    let all_ledger = fs::read_to_string(archive.join(TRUNCATIONS_FILE))?;
    assert_eq!(seg_ledger.lines().count(), 1);
    assert_eq!(all_ledger.lines().count(), 1);
    let record: Value = serde_json::from_str(all_ledger.lines().next().unwrap())?;
    assert_eq!(record["schema"], TRUNCATION_SCHEMA);
    assert_eq!(record["residual_bytes"], fragment.len());
    assert_eq!(record["archived_rows"], 2);
    assert_eq!(record["last_archived_epoch"], 2);

    // The fragment itself is kept verbatim: it is not counted as a row, but
    // it is not thrown away either.
    let evidence = record["evidence_file"].as_str().expect("evidence");
    assert_eq!(fs::read(seg.join(evidence))?, fragment.to_vec());

    // ...and the index says the record has a hole.
    let index = rebuild_index(temp.path(), &archive)?;
    assert_eq!(index["truncated_segments"], 1);
    assert_eq!(index["truncated_tail_bytes"], fragment.len());
    assert_eq!(index["segments"][0]["truncation_count"], 1);
    Ok(())
}

#[test]
fn ssh_argv_never_targets_the_local_host_and_knows_xbabe4s_port() {
    assert!(
        ssh_argv(LOCAL_NODE, 30).is_err(),
        "ssh xbabe2 is a Mac Studio"
    );
    assert!(ssh_argv("nope", 30).is_err());
    let one = ssh_argv("xbabe1", 30).unwrap();
    assert_eq!(one[0], "timeout");
    assert_eq!(one.last().unwrap(), "xbabe1");
    assert!(one.iter().any(|a| a == "BatchMode=yes"));
    let four = ssh_argv("xbabe4", 30).unwrap();
    let port = four.iter().position(|a| a == "-p").expect("-p");
    assert_eq!(four[port + 1], "2224");
    assert!(REMOTE_NODES.iter().all(|node| *node != LOCAL_NODE));
}

#[test]
fn remote_paths_that_could_reach_a_shell_are_refused() {
    let source = RemoteSource {
        node: "xbabe1".into(),
        runs_root: jope_paths::to_string(&crate::paths::scq_runs_root()),
        relative: "run/../../etc/shadow".into(),
        len: 10,
        deadline_secs: 5,
    };
    assert!(source.read_range(0, 10).is_err());
    let source = RemoteSource {
        node: "xbabe1".into(),
        runs_root: "/runs; rm -rf /".into(),
        relative: "a/b".into(),
        len: 10,
        deadline_secs: 5,
    };
    assert!(source.read_range(0, 10).is_err());
    assert!(remote_scan("xbabe1", "/runs; rm -rf /", 5).is_err());
}

#[test]
fn node_status_round_trips() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    fs::create_dir_all(&archive)?;
    let mut nodes = BTreeMap::new();
    nodes.insert(
        "xbabe1".to_string(),
        NodeStatus {
            node: "xbabe1".into(),
            last_attempt_ms: 5,
            last_success_ms: 4,
            consecutive_failures: 2,
            last_error: "boom".into(),
            segments_seen: 3,
            reachable: false,
        },
    );
    save_nodes(&archive, &nodes)?;
    let back = load_nodes(&archive);
    assert_eq!(back["xbabe1"].consecutive_failures, 2);
    assert_eq!(back["xbabe1"].last_error, "boom");
    assert!(!back["xbabe1"].reachable);
    Ok(())
}

#[test]
fn lock_is_exclusive_and_released_on_drop() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let held = CaptureLock::acquire(&archive, 1, 0)?;
    assert!(held.is_some());
    // A second open file description cannot take it while the first holds.
    assert!(CaptureLock::acquire(&archive, 1, 0)?.is_none());
    drop(held);
    assert!(CaptureLock::acquire(&archive, 1, 0)?.is_some());
    Ok(())
}
