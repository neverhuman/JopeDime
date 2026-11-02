use super::*;

#[test]
fn full_state_rejects_matching_journal_and_index_tamper_with_unchanged_watermarks() {
    let fixture = Fixture::new(false);
    let inventory = staged_history_fixture(&fixture);
    let mut authority = fixture.open();
    authority.import_history(&inventory).unwrap();
    let path = fixture.root.join("dispatch.jsonl");
    let original = fs::read(&path).unwrap();
    let mut changed = Vec::new();
    let mut victim = None;
    for line in original.split_inclusive(|b| *b == b'\n') {
        let mut event: Event = serde_json::from_slice(line).unwrap();
        if victim.is_none() {
            if let Some(row) = event.effects.iter_mut().find(|r| r.table == "exclusions") {
                let reason = row.row["reason"].as_str().unwrap();
                let mut replacement = reason.to_owned();
                replacement.replace_range(replacement.len() - 1.., "X");
                row.row["reason"] = json!(replacement);
                victim = Some((row.key.clone(), replacement));
                let bytes = encoded(&event).unwrap();
                assert_eq!(bytes.len() + 1, line.len());
                changed.extend(bytes);
                changed.push(b'\n');
                continue;
            }
        }
        changed.extend_from_slice(line);
    }
    let (key, reason) = victim.unwrap();
    assert_eq!(changed.len(), original.len());
    fs::write(&path, changed).unwrap();
    let db = Connection::open(fixture.root.join("registry.sqlite")).unwrap();
    db.execute(
        "UPDATE exclusions SET reason=? WHERE id=?",
        params![reason, key],
    )
    .unwrap();
    assert!(
        authority.verify_tail().is_err(),
        "matching altered postimages cannot repair the original hash chain"
    );
}

#[test]
fn state_digest_map_preserves_latest_rows_and_rejects_missing_extra_and_tampered_rows() {
    let index = StateDigests::new().unwrap();
    let mut row = StateRow {
        table: "exclusions".into(),
        key: "original-key".into(),
        row: json!({"value":"old","nullable":null}),
    };
    index.insert(&row).unwrap();
    row.row["value"] = json!("latest");
    index.insert(&row).unwrap();
    assert!(
        index.ensure_empty().is_err(),
        "missing indexed row must fail"
    );
    assert!(index.consume("exclusions", "extra-key", &row.row).is_err());
    assert!(index
        .consume(
            "exclusions",
            &row.key,
            &json!({"value":"old","nullable":null})
        )
        .is_err());
    assert!(
        index
            .consume("exclusions", &row.key, &json!({"value":"latest"}))
            .is_err(),
        "complete postimage includes null fields"
    );
    index.consume("exclusions", &row.key, &row.row).unwrap();
    index.ensure_empty().unwrap();
    assert!(index.consume("exclusions", &row.key, &row.row).is_err());
    row.table = "unreviewed_table".into();
    assert!(index.insert(&row).is_err());
}

#[test]
fn state_digest_map_exhaustion_fails_without_disk_fallback() {
    let index = StateDigests::with_page_limit(2).unwrap();
    let row = StateRow {
        table: "exclusions".into(),
        key: "x".repeat(32 * 1024),
        row: json!({"original":true}),
    };
    let file: String = index
        .db
        .query_row(
            "SELECT file FROM pragma_database_list WHERE name='main'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert!(file.is_empty());
    assert_eq!(
        index
            .db
            .pragma_query_value(None, "max_page_count", |r| r.get::<_, u64>(0))
            .unwrap(),
        2
    );
    assert!(index.insert(&row).is_err());
    // SQLITE_FULL may invalidate this disposable journal-OFF database. The
    // verifier propagates the error and discards it; never reuse it afterward.
}

#[test]
fn recovery_locator_retains_original_watermark_and_latest_postimage() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("original.jsonl");
    let first = Event {
        instance: "fixture".into(),
        seq: 1,
        previous: ZERO.into(),
        kind: "fixture".into(),
        payload: json!({}),
        effects: vec![StateRow {
            table: "exclusions".into(),
            key: "a".into(),
            row: json!({"version":"old"}),
        }],
    };
    let second = Event {
        instance: "fixture".into(),
        seq: 2,
        previous: hash(&encoded(&first).unwrap()),
        kind: "fixture".into(),
        payload: json!({}),
        effects: vec![
            StateRow {
                table: "exclusions".into(),
                key: "a".into(),
                row: json!({"version":"new"}),
            },
            StateRow {
                table: "exclusions".into(),
                key: "b".into(),
                row: json!({"version":"only"}),
            },
        ],
    };
    let mut bytes = encoded(&first).unwrap();
    bytes.push(b'\n');
    let offset = bytes.len() as u64;
    bytes.extend(encoded(&second).unwrap());
    bytes.push(b'\n');
    fs::write(&path, &bytes).unwrap();
    let mut rows = RecoveryRows::new(path.clone()).unwrap();
    rows.insert(&first.effects[0], 0, 0, true).unwrap();
    rows.insert(&second.effects[0], offset, 0, false).unwrap();
    rows.insert(&second.effects[1], offset, 1, false).unwrap();
    rows.length = bytes.len() as u64;
    rows.journal_sha256 = hash(&bytes);
    assert_eq!(
        rows.get("exclusions", "a").unwrap(),
        Some(json!({"version":"new"}))
    );
    let old: Vec<u8> = rows
        .db
        .query_row(
            "SELECT db_digest FROM latest WHERE state_key='a'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        old,
        Sha256::digest(encoded(&first.effects[0].row).unwrap()).as_slice()
    );
    let yielded = rows.iter().unwrap().collect::<Result<Vec<_>>>().unwrap();
    assert_eq!(yielded.len(), 2);
    assert_eq!(yielded[0].1, json!({"version":"new"}));
    assert!(rows.get("exclusions", "absent").unwrap().is_none());
    let original = bytes.clone();
    let walk = rows.iter().unwrap();
    let mut appended = original.clone();
    appended.extend_from_slice(b"{}\n");
    fs::write(&path, appended).unwrap();
    assert!(
        walk.collect::<Result<Vec<_>>>().is_err(),
        "append after iterator construction must be refused"
    );
    fs::write(&path, &original).unwrap();
    let superseded = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("old", "bad");
    fs::write(&path, superseded).unwrap();
    assert!(
        rows.get("exclusions", "a").is_err(),
        "unchanged latest row cannot conceal superseded journal mutation"
    );
    assert!(rows.iter().unwrap().collect::<Result<Vec<_>>>().is_err());
    fs::write(&path, original).unwrap();
    let changed = String::from_utf8(bytes).unwrap().replace("new", "bad");
    fs::write(&path, changed).unwrap();
    assert!(rows.get("exclusions", "a").is_err());
    assert!(rows.iter().unwrap().collect::<Result<Vec<_>>>().is_err());
}

#[test]
fn recovery_report_bounds_display_without_losing_count_or_digest() {
    let mut rows = RecoveryReportRows::new();
    let mut expected = Sha256::new();
    for i in 0..1100 {
        let row = json!({"original":i});
        let bytes = encoded(&row).unwrap();
        expected.update((bytes.len() as u64).to_le_bytes());
        expected.update(bytes);
        rows.push(row).unwrap();
    }
    assert_eq!(rows.rows.len(), 1024);
    assert_eq!(rows.summary()["total"], 1100);
    assert_eq!(rows.summary()["truncated"], true);
    assert_eq!(
        rows.summary()["ordered_records_sha256"],
        hex::encode(expected.finalize())
    );
}

#[test]
fn retained_history_corruption_blocks_batch_and_coverage() {
    let fixture = Fixture::new(false);
    let inventory = staged_history_fixture(&fixture);
    let mut authority = fixture.open();
    let stage = authority.history_stage(&inventory).unwrap();
    authority
        .history_import_batch(&HistoryBatch {
            stage_sha256: stage.clone(),
            start: 0,
            end: 3,
        })
        .unwrap();
    let evidence = &inventory.occurrences[1].evidence_sha256;
    fs::remove_file(fixture.root.join("objects").join(evidence)).unwrap();
    assert!(authority
        .history_import_batch(&HistoryBatch {
            stage_sha256: stage.clone(),
            start: 0,
            end: 3
        })
        .is_err());
    assert!(authority.history_commit_coverage(&stage).is_err());
    assert!(state(authority.claim(&fit(), &request()).unwrap()).contains("coverage"));
}
