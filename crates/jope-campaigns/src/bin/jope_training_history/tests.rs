use super::*;
use std::time::SystemTime;
use tempfile::TempDir;

/// Local-only capture, i.e. the pre-existing behaviour, for tests that are
/// about the local path. Remote coverage has its own tests.
fn local(root: &Path, runs: &Path) -> Cfg {
    Cfg {
        root: root.to_path_buf(),
        runs_root: runs.to_path_buf(),
        node: "xbabe2".into(),
        remote_nodes: Vec::new(),
        remote_runs_root: runs.to_string_lossy().into_owned(),
        ssh_deadline_secs: 5,
        quiescent_secs: th::TAIL_QUIESCENT_SECS,
    }
}

fn write_run(runs: &Path, run_id: &str, attempt: &str, body: &str) -> PathBuf {
    let outputs = runs
        .join(run_id)
        .join("attempts")
        .join(attempt)
        .join("outputs/training/watermark-001805437");
    std::fs::create_dir_all(&outputs).unwrap();
    let path = outputs.join(th::METRICS_FILE);
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn capture_is_incremental_and_survives_the_run_dir_disappearing() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g9";
    let source = write_run(
        &runs,
        run_id,
        "1",
        "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n",
    );

    let first = capture(&local(&root, &runs))?;
    assert_eq!(first["appended_rows"], 1);
    assert_eq!(first["archive_rows"], 1);

    // Second cycle, one more epoch: only the delta is read and appended.
    std::fs::write(
        &source,
        "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n{\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}\n",
    )?;
    let second = capture(&local(&root, &runs))?;
    assert_eq!(second["appended_rows"], 1);
    assert_eq!(second["archive_rows"], 2);

    // Idempotent: nothing new appended when nothing changed.
    let third = capture(&local(&root, &runs))?;
    assert_eq!(third["appended_rows"], 0);
    assert_eq!(third["archive_rows"], 2);
    assert!(uncovered(&root, &runs)?.is_empty());

    // The collector reclaims the 12 GB run directory. The rows stay.
    std::fs::remove_dir_all(runs.join(run_id))?;
    let after = capture(&local(&root, &runs))?;
    assert_eq!(after["archive_rows"], 2, "archive keeps the epochs");
    assert_eq!(after["sealed_now"], 1);
    let index: Value = serde_json::from_str(&std::fs::read_to_string(
        th::archive_root(&root).join(th::INDEX_FILE),
    )?)?;
    assert_eq!(index["total_rows"], 2);
    assert_eq!(index["segments"][0]["source_present"], false);
    assert_eq!(index["segments"][0]["last_epoch"], 2);
    Ok(())
}

#[test]
fn verify_reports_a_run_that_is_ahead_of_the_archive() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-c";
    write_run(&runs, run_id, "0", "{\"epoch\":1}\n");
    // Nothing captured yet, so the run is uncovered.
    assert_eq!(uncovered(&root, &runs)?.len(), 1);
    capture(&local(&root, &runs))?;
    assert!(uncovered(&root, &runs)?.is_empty());
    Ok(())
}

#[test]
fn multiple_attempts_are_archived_separately() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-c-r000001";
    write_run(&runs, run_id, "0", "{\"epoch\":1}\n");
    write_run(&runs, run_id, "1", "{\"epoch\":1}\n{\"epoch\":2}\n");
    let record = capture(&local(&root, &runs))?;
    assert_eq!(record["segments_live"], 2);
    assert_eq!(record["archive_rows"], 3);
    Ok(())
}

#[test]
fn idle_cycles_do_not_grow_the_receipts_log() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    write_run(
        &runs,
        "jope-selfplay-train-001805437-c",
        "0",
        "{\"epoch\":1}\n",
    );
    let events = th::archive_root(&root).join(th::EVENTS_FILE);

    capture(&local(&root, &runs))?; // wrote a row -> receipted
    let after_work = std::fs::read_to_string(&events)?.lines().count();
    assert_eq!(after_work, 1);

    // Three cycles that change nothing must add nothing...
    for _ in 0..3 {
        capture(&local(&root, &runs))?;
    }
    assert_eq!(std::fs::read_to_string(&events)?.lines().count(), 1);

    // ...but the index is still refreshed every cycle, which is what the
    // health alarm reads for freshness.
    assert!(th::archive_root(&root).join(th::INDEX_FILE).exists());

    // Once the log goes stale a heartbeat lands.
    let old = SystemTime::now() - std::time::Duration::from_secs(IDLE_RECEIPT_SECS + 60);
    std::fs::File::options()
        .write(true)
        .open(&events)?
        .set_times(std::fs::FileTimes::new().set_modified(old))?;
    capture(&local(&root, &runs))?;
    assert_eq!(std::fs::read_to_string(&events)?.lines().count(), 2);
    Ok(())
}

/// Backdate a file so its trailing fragment reads as abandoned.
fn age_file(path: &Path, secs: u64) {
    let when = SystemTime::now() - std::time::Duration::from_secs(secs);
    std::fs::File::options()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(when))
        .unwrap();
}

#[test]
fn an_abandoned_partial_row_is_written_off_but_every_complete_row_survives() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g9";
    // Two finished epochs plus the row the trainer was killed mid-write on.
    let source = write_run(
        &runs,
        run_id,
        "1",
        "{\"epoch\":1,\"train_loss\":3.0,\"val_loss\":3.4}\n\
         {\"epoch\":2,\"train_loss\":2.9,\"val_loss\":3.3}\n\
         {\"epoch\":3,\"train_lo",
    );

    // While the file is still fresh the fragment is treated as mid-write:
    // nothing is written off and the segment is honestly reported behind.
    let live = capture(&local(&root, &runs))?;
    assert_eq!(live["appended_rows"], 2);
    assert_eq!(live["truncations"].as_array().unwrap().len(), 0);
    assert_eq!(uncovered(&root, &runs)?.len(), 1);
    assert_eq!(uncovered(&root, &runs)?[0]["residual"], "partial_row");

    // Once the writer has been gone long enough, the hole is recorded.
    age_file(&source, th::TAIL_QUIESCENT_SECS + 120);
    let after = capture(&local(&root, &runs))?;
    let truncations = after["truncations"].as_array().unwrap();
    assert_eq!(truncations.len(), 1);
    assert_eq!(truncations[0]["residual_bytes"], 20);
    assert_eq!(after["archive_rows"], 2, "complete rows kept");
    // ...and the run directory is no longer reported as holding rows we lack.
    assert!(uncovered(&root, &runs)?.is_empty());

    // Recording is idempotent: the next tick adds nothing.
    let again = capture(&local(&root, &runs))?;
    assert_eq!(again["truncations"].as_array().unwrap().len(), 0);
    let ledger = std::fs::read_to_string(th::archive_root(&root).join(th::TRUNCATIONS_FILE))?;
    assert_eq!(ledger.lines().count(), 1);

    // A shortfall that spans COMPLETE rows is never written off, even when
    // the file is equally stale.
    let run_two = "jope-selfplay-train-001805437-c";
    let second = write_run(
        &runs,
        run_two,
        "0",
        "{\"epoch\":1,\"train_loss\":1.0}\n{\"epoch\":2,\"train_loss\":0.9}\n",
    );
    age_file(&second, th::TAIL_QUIESCENT_SECS + 120);
    // Archive only the first row, then make the archive look behind.
    capture(&local(&root, &runs))?;
    let key = th::SegmentKey {
        run_id: run_two.into(),
        attempt: "0".into(),
        watermark: "watermark-001805437".into(),
    };
    let archived = th::SegmentState::metrics_path(&th::segment_dir(&th::archive_root(&root), &key));
    let body = std::fs::read(&archived)?;
    let keep = body.iter().position(|b| *b == b'\n').unwrap() as u64 + 1;
    std::fs::OpenOptions::new()
        .write(true)
        .open(&archived)?
        .set_len(keep)?;
    let reader = th::FileSource(second.clone());
    let residual = th::classify_residual(&reader, keep, std::fs::metadata(&second)?.len())?;
    assert!(
        matches!(residual, th::Residual::CompleteRowsMissing { .. }),
        "a missing complete row must never look like a truncated tail: {residual:?}"
    );
    Ok(())
}

#[test]
fn an_unreachable_node_is_reported_and_its_segments_are_not_sealed() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let archive = th::archive_root(&root);
    // A segment that was captured from xbabe1 on some earlier pass.
    let key = th::SegmentKey {
        run_id: "jope-selfplay-train-001805437-b".into(),
        attempt: "0".into(),
        watermark: "watermark-001805437".into(),
    };
    th::ingest_source(
        &archive,
        &key,
        &th::BytesSource(b"{\"epoch\":1,\"train_loss\":2.0}\n".to_vec()),
        "xbabe1:/runs/x",
        "xbabe1",
    )?;

    // An unusable remote runs root fails enumeration without any ssh, which
    // is exactly the shape of an unreachable node.
    let cfg = Cfg {
        root: root.clone(),
        runs_root: runs.clone(),
        node: "xbabe2".into(),
        remote_nodes: vec!["xbabe1".into()],
        remote_runs_root: "/not a valid root".into(),
        ssh_deadline_secs: 5,
        quiescent_secs: th::TAIL_QUIESCENT_SECS,
    };
    let record = capture(&cfg)?;
    assert_eq!(
        record["coverage_complete"], false,
        "must not read as success"
    );
    assert_eq!(record["nodes_skipped"].as_array().unwrap().len(), 1);
    assert_eq!(record["nodes_skipped"][0]["node"], "xbabe1");
    assert_eq!(record["sealed_now"], 0);
    assert_eq!(record["seal_deferred"], 1);
    let state = th::SegmentState::load(&th::segment_dir(&archive, &key))?
        .expect("state exists after capture");
    assert!(
        state.source_present,
        "an unreachable node says nothing about whether its runs still exist"
    );

    // The failure is durable and visible without re-running ssh.
    let nodes = th::load_nodes(&archive);
    assert_eq!(nodes["xbabe1"].consecutive_failures, 1);
    assert!(!nodes["xbabe1"].reachable);
    assert!(!nodes["xbabe1"].last_error.is_empty());
    Ok(())
}

#[test]
fn the_local_host_is_never_an_ssh_target() {
    let (nodes, rejected) = parse_nodes("xbabe1,xbabe2,xbabe3,xbabe4,xbabe1");
    assert_eq!(nodes, vec!["xbabe1", "xbabe3", "xbabe4"]);
    assert_eq!(rejected.len(), 1);
    assert_eq!(rejected[0]["node"], "xbabe2");
    let (nodes, _) = parse_nodes(DEFAULT_REMOTE_NODES);
    assert!(!nodes.iter().any(|node| node == LOCAL_NODE));
}

#[test]
fn a_concurrent_holder_makes_the_tick_skip_not_corrupt() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    write_run(
        &runs,
        "jope-selfplay-train-001805437-c",
        "0",
        "{\"epoch\":1}\n",
    );
    let archive = th::archive_root(&root);
    let held = th::CaptureLock::acquire(&archive, 1, 0)?;
    assert!(held.is_some());
    let record = capture(&local(&root, &runs))?;
    assert_eq!(record["event"], "skipped_locked");
    drop(held);
    assert_eq!(capture(&local(&root, &runs))?["appended_rows"], 1);
    Ok(())
}
