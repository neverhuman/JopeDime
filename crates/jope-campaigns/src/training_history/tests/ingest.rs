use super::*;

#[test]
fn incremental_append_is_idempotent_and_whole_row_only() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);

    fs::write(
        &source,
        format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4)),
    )?;
    let first = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(first.appended_rows, 2);
    assert_eq!(first.last_epoch, Some(2));
    assert!(first.covered());

    // Re-running with no new bytes must append nothing.
    let again = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(again.appended_rows, 0);
    assert_eq!(again.archived_rows, 2);

    // A row still being written (no trailing newline) is not archived, and
    // coverage correctly reports the source as ahead.
    let partial = format!(
        "{}\n{}\n{}",
        row(1, 3.0, 3.5),
        row(2, 2.9, 3.4),
        "{\"epoch\":3,\"train_lo"
    );
    fs::write(&source, &partial)?;
    let mid = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(mid.appended_rows, 0);
    assert!(!mid.covered());

    // Once the row completes it lands, and the archive is byte-identical.
    fs::write(
        &source,
        format!(
            "{}\n{}\n{}\n",
            row(1, 3.0, 3.5),
            row(2, 2.9, 3.4),
            row(3, 2.8, 3.3)
        ),
    )?;
    let done = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(done.appended_rows, 1);
    assert_eq!(done.last_epoch, Some(3));
    assert!(done.covered());
    let archived = fs::read(SegmentState::metrics_path(&segment_dir(&archive, &key())))?;
    assert_eq!(archived, fs::read(&source)?);
    Ok(())
}

#[test]
fn a_replaced_source_rotates_instead_of_truncating() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    fs::write(
        &source,
        format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4)),
    )?;
    ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;

    // A rerun restarts the file with different content.
    fs::write(&source, format!("{}\n", row(1, 9.9, 9.9)))?;
    let after = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert!(after.rotated);
    assert_eq!(after.archived_rows, 1);

    let seg = segment_dir(&archive, &key());
    let rotated: Vec<_> = fs::read_dir(seg.join("rewrites"))?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".jsonl"))
        .collect();
    assert_eq!(rotated.len(), 1, "old stream must be kept, not truncated");
    let preserved = fs::read_to_string(seg.join("rewrites").join(&rotated[0]))?;
    assert!(preserved.contains("\"epoch\":2"), "old rows survive");
    Ok(())
}

#[test]
fn shrinking_source_also_rotates() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    fs::write(
        &source,
        format!(
            "{}\n{}\n{}\n",
            row(1, 3.0, 3.5),
            row(2, 2.9, 3.4),
            row(3, 2.8, 3.3)
        ),
    )?;
    ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    fs::write(&source, "")?;
    let after = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert!(after.rotated);
    assert_eq!(after.archived_rows, 0);
    // Coverage is trivially true against an empty source; the point is that
    // the three original rows are still on disk under rewrites/.
    assert!(after.covered());
    assert_eq!(
        fs::read_dir(segment_dir(&archive, &key()).join("rewrites"))?
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().ends_with(".jsonl"))
            .count(),
        1
    );
    Ok(())
}

#[test]
fn a_lost_archive_file_is_re_ingested_without_double_counting() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    fs::write(
        &source,
        format!(
            "{}\n{}\n{}\n",
            row(1, 3.0, 3.5),
            row(2, 2.9, 3.4),
            row(3, 2.8, 3.3)
        ),
    )?;
    ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;

    // Simulate a crash that lost the archive body but kept the state file:
    // the offset must fall back to the file, and the counters with it.
    let seg = segment_dir(&archive, &key());
    fs::remove_file(SegmentState::metrics_path(&seg))?;
    let stale = SegmentState::load(&seg)
        .expect("state could be read")
        .expect("state survived");
    assert_eq!(stale.archived_rows, 3);

    let after = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(after.archived_rows, 3, "rows recounted, not doubled");
    assert_eq!(after.last_epoch, Some(3));
    assert!(after.covered());
    assert_eq!(
        fs::read(SegmentState::metrics_path(&seg))?,
        fs::read(&source)?
    );
    Ok(())
}

#[test]
fn a_partially_lost_archive_file_recounts_from_what_survived() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    let body = format!(
        "{}\n{}\n{}\n",
        row(1, 3.0, 3.5),
        row(2, 2.9, 3.4),
        row(3, 2.8, 3.3)
    );
    fs::write(&source, &body)?;
    ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;

    // Truncate the archive to the first two rows behind our back.
    let seg = segment_dir(&archive, &key());
    let keep = body.find(&row(3, 2.8, 3.3)).unwrap() as u64;
    let file = fs::OpenOptions::new()
        .write(true)
        .open(SegmentState::metrics_path(&seg))?;
    file.set_len(keep)?;
    drop(file);

    let after = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert_eq!(after.archived_rows, 3);
    assert_eq!(after.appended_rows, 1);
    assert_eq!(fs::read(SegmentState::metrics_path(&seg))?, body.as_bytes());
    Ok(())
}

#[test]
fn coverage_tracks_the_source_length() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    fs::write(&source, format!("{}\n", row(1, 3.0, 3.5)))?;
    let out = ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    assert!(out.covered());
    assert_eq!(
        archived_bytes(&archive, &key()),
        fs::metadata(&source)?.len()
    );
    Ok(())
}

#[test]
fn unsafe_keys_are_refused() {
    let temp = TempDir::new().unwrap();
    let archive = temp.path().join("history");
    let bad = SegmentKey {
        run_id: "../../etc".into(),
        attempt: "0".into(),
        watermark: "w".into(),
    };
    assert!(ingest_source(&archive, &bad, &BytesSource(b"{}\n".to_vec()), "s", "n").is_err());
    assert!(!archive.join("runs").exists());
}

#[test]
fn key_from_path_extracts_attempt_and_watermark() {
    let path = Path::new(
            "/runs/jope-selfplay-train-001805437-a-g9/attempts/1/outputs/training/watermark-001805437/train_metrics.jsonl",
        );
    let key = key_from_path("jope-selfplay-train-001805437-a-g9", path).unwrap();
    assert_eq!(key.attempt, "1");
    assert_eq!(key.watermark, "watermark-001805437");
}

#[test]
fn scan_and_index_round_trip() -> Result<()> {
    let temp = TempDir::new()?;
    let root = temp.path().join("gen0");
    let runs = temp.path().join("runs");
    let run_id = "jope-selfplay-train-001805437-a-g9";
    let outputs = runs
        .join(run_id)
        .join("attempts/1/outputs/training/watermark-001805437");
    fs::create_dir_all(&outputs)?;
    fs::write(
        outputs.join(METRICS_FILE),
        format!("{}\n", row(7, 2.5, 3.6)),
    )?;
    fs::create_dir_all(root.join("specs"))?;
    // Shaped like a real spec: the warm checkpoint is a placeholder whose
    // identity only appears in the matching [[input]] block.
    fs::write(
            root.join("specs").join(format!("{run_id}.toml")),
            format!(
                concat!(
                    "args = [\"--recipe\", \"c\", \"--warm\", \"{{input:checkpoint}}\"]\n",
                    "\n[[input]]\nname = \"train\"\n",
                    "source = \"{flywheel}/training/snapshot-001805437-full_consolidation-eq32v32c128/corpus.train.jsonl\"\n",
                    "sha256 = \"d124ed7b\"\n",
                    "\n[[input]]\nname = \"checkpoint\"\n",
                    "source = \"/x/attempts/0/outputs/training/w/d36274de.model.pt\"\n",
                    "sha256 = \"d36274de99ddf3b12dffd534145b42bfcace1bbd6230bd65cc37288efe3a3bf3\"\n",
                ),
                flywheel = crate::paths::flywheel_root().display()
            ),
        )?;

    let found = scan_run_dir(&runs, run_id);
    assert_eq!(found.len(), 1);
    let archive = archive_root(&root);
    ingest_source(
        &archive,
        &found[0].key,
        &FileSource(found[0].path.clone()),
        &found[0].path.to_string_lossy(),
        "xbabe2",
    )?;
    let index = rebuild_index(&root, &archive)?;
    assert_eq!(index["schema"], INDEX_SCHEMA);
    assert_eq!(index["total_rows"], 1);
    let segment = &index["segments"][0];
    assert_eq!(segment["run_id"], run_id);
    assert_eq!(segment["last_epoch"], 7);
    assert_eq!(segment["recipe"], "c");
    assert_eq!(
        segment["warm_start"],
        "d36274de99ddf3b12dffd534145b42bfcace1bbd6230bd65cc37288efe3a3bf3"
    );
    assert_eq!(
        segment["corpus_snapshot"],
        "snapshot-001805437-full_consolidation-eq32v32c128"
    );
    assert_eq!(segment["tail"]["val_loss"], 3.6);
    Ok(())
}

#[test]
fn discover_only_matches_trainer_dirs() -> Result<()> {
    let temp = TempDir::new()?;
    fs::create_dir_all(temp.path().join("jope-selfplay-train-001805437-c"))?;
    fs::create_dir_all(temp.path().join("jte-abc"))?;
    fs::create_dir_all(temp.path().join("jope-testonly-train-000090066"))?;
    let names = discover_trainer_runs(temp.path())?;
    assert_eq!(names.len(), 2);
    assert!(names.iter().all(|n| n.contains(TRAIN_MARKER)));
    Ok(())
}

#[test]
fn sealing_marks_the_source_gone_without_touching_rows() -> Result<()> {
    let temp = TempDir::new()?;
    let archive = temp.path().join("history");
    let source = temp.path().join(METRICS_FILE);
    fs::write(&source, format!("{}\n", row(1, 3.0, 3.5)))?;
    ingest_source(
        &archive,
        &key(),
        &FileSource(source.clone()),
        "src",
        "xbabe2",
    )?;
    let before = archived_bytes(&archive, &key());
    seal_segment(&archive, &key())?;
    let state = SegmentState::load(&segment_dir(&archive, &key()))?.unwrap();
    assert!(!state.source_present);
    assert_eq!(archived_bytes(&archive, &key()), before);
    Ok(())
}

#[test]
fn residual_tells_a_partial_row_from_missing_complete_rows() -> Result<()> {
    // Ends on a row boundary and fully archived: nothing outstanding.
    let full = format!("{}\n{}\n", row(1, 3.0, 3.5), row(2, 2.9, 3.4));
    let source = BytesSource(full.clone().into_bytes());
    assert_eq!(
        classify_residual(&source, full.len() as u64, full.len() as u64)?,
        Residual::None
    );

    // One unterminated fragment past the archive head: a row that was never
    // finished. This is the only shape the policy writes off.
    let partial = format!("{full}{{\"epoch\":3,\"train_lo");
    let source = BytesSource(partial.clone().into_bytes());
    let verdict = classify_residual(&source, full.len() as u64, partial.len() as u64)?;
    assert_eq!(
        verdict,
        Residual::PartialRow(b"{\"epoch\":3,\"train_lo".to_vec())
    );

    // A shortfall spanning a newline is COMPLETE rows that are not archived.
    // Writing this off would lose data, so it must be refused.
    let verdict = classify_residual(&source, 0, partial.len() as u64)?;
    assert!(matches!(verdict, Residual::CompleteRowsMissing { .. }));

    // Unterminated but far too big to be one row: also refused.
    let huge = vec![b'x'; MAX_TRUNCATED_TAIL_BYTES as usize + 10];
    let source = BytesSource(huge.clone());
    assert!(matches!(
        classify_residual(&source, 0, huge.len() as u64)?,
        Residual::OversizedFragment { .. }
    ));
    Ok(())
}
