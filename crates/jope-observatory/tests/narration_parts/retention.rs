use super::*;

#[test]
fn retained_pages_use_exclusive_run_scoped_decimal_cursors() {
    let dir = fixture_dir();
    let (path, events) = fixture(dir.path(), 3);
    let mut index = index(&path);
    let first = index.page(&zero(), 2, MAX_PAGE_BYTES).unwrap();
    assert_eq!(first.events, events[..2]);
    assert_eq!(first.next.sequence, "2");
    assert!(first.more);
    let second = index.page(&first.next, 2, MAX_PAGE_BYTES).unwrap();
    assert_eq!(second.events, events[2..]);
    assert!(!second.more);
    assert!(index
        .page(&second.next, 2, MAX_PAGE_BYTES)
        .unwrap()
        .events
        .is_empty());
    assert!(index
        .page(
            &Cursor {
                run_id: "other".into(),
                sequence: "0".into()
            },
            2,
            MAX_PAGE_BYTES
        )
        .is_err());
    assert!(index
        .page(
            &Cursor {
                run_id: "run-a".into(),
                sequence: "4".into()
            },
            2,
            MAX_PAGE_BYTES
        )
        .is_err());
    for s in ["01", "-1", "1e2", "+1", "18446744073709551616", ""] {
        assert!(narration::sequence(s).is_err(), "{s}");
    }
    assert!(index.page(&zero(), 0, MAX_PAGE_BYTES).is_err());
    assert!(index.page(&zero(), 257, MAX_PAGE_BYTES).is_err());
    assert!(index.page(&zero(), 1, 1024).is_err());
    assert!(index.page(&zero(), 1, MAX_PAGE_BYTES + 1).is_err());
}

#[test]
fn partial_live_tail_waits_then_yields_same_original_event() {
    let dir = fixture_dir();
    let events = records("run-a", 2);
    let path = dir.path().join("journal.jsonl");
    let first = bytes(&events[..1]);
    let second = bytes(&events[1..]);
    let mut initial = first.clone();
    initial.extend_from_slice(&second[..second.len() / 2]);
    fs::write(&path, &initial).unwrap();
    commit_fixture(&path, &events[..1]);
    let mut index = index(&path);
    let page = index.page(&zero(), 10, MAX_PAGE_BYTES).unwrap();
    assert_eq!(page.events, events[..1]);
    assert!(page.partial_tail);
    let mut writer = OpenOptions::new().append(true).open(&path).unwrap();
    writer.write_all(&second[second.len() / 2..]).unwrap();
    writer.sync_all().unwrap();
    let waiting = index.page(&page.next, 10, MAX_PAGE_BYTES).unwrap();
    assert!(
        waiting.events.is_empty(),
        "a complete newline is not a durability acknowledgement"
    );
    commit_fixture(&path, &events);
    let next = index.page(&page.next, 10, MAX_PAGE_BYTES).unwrap();
    assert_eq!(next.events, events[1..]);
    assert!(!next.partial_tail);
    assert_eq!(fs::read(&path).unwrap(), bytes(&events));
}

#[test]
fn complete_corruption_wrong_run_and_oversize_tail_fail_closed() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 2);
    assert!(Index::open(&path, "other", Limits::default()).is_err());
    let mut raw = fs::read(&path).unwrap();
    let at = raw.iter().position(|b| *b == b'R').unwrap();
    raw[at] = b'X';
    fs::write(&path, raw).unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
    fs::write(&path, b"{}\n").unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
    fs::write(&path, vec![b'x'; narration::MAX_RECORD_BYTES + 2]).unwrap();
    assert!(Index::open(&path, "run-a", Limits::default()).is_err());
}

#[test]
fn truncation_replacement_and_record_caps_are_explicit_errors() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 2);
    assert!(Index::open(
        &path,
        "run-a",
        Limits {
            max_records: 1,
            ..Limits::default()
        }
    )
    .is_err());
    assert!(Index::open(
        &path,
        "run-a",
        Limits {
            max_journal_bytes: 1,
            ..Limits::default()
        }
    )
    .is_err());
    let mut index = index(&path);
    fs::write(&path, b"").unwrap();
    assert!(index.status().is_err());
    fs::write(&path, bytes(&records("run-a", 2))).unwrap();
    assert!(
        index.status().is_err(),
        "integrity failure must remain sticky"
    );
    let mut index = Index::open(&path, "run-a", Limits::default()).unwrap();
    let replacement = dir.path().join("new.jsonl");
    fs::write(&replacement, bytes(&records("run-a", 2))).unwrap();
    fs::rename(replacement, &path).unwrap();
    assert!(index.status().is_err());
}

#[cfg(unix)]
#[test]
fn symlink_journals_are_refused() {
    let dir = fixture_dir();
    let (path, _) = fixture(dir.path(), 1);
    let link = dir.path().join("link");
    std::os::unix::fs::symlink(path, &link).unwrap();
    assert!(Index::open(&link, "run-a", Limits::default()).is_err());
}

#[test]
fn real_durable_journal_is_read_without_rerendering() {
    let dir = fixture_dir();
    let path = dir.path().join("real.jsonl");
    let (tap, journal) =
        FactJournal::create(&path, "run-a", jope_narrator::Limits::default()).unwrap();
    let _ = tap.emit(LineageFact::RunStopped {
        status: "completed".into(),
    });
    journal.finish().unwrap();
    let original = fs::read(&path).unwrap();
    let mut index = index(&path);
    let mut output = Vec::new();
    let page = narration::print_page(&mut index, &zero(), 64, false, &mut output).unwrap();
    assert!(!page.events.is_empty());
    assert_eq!(output, original);
    assert_eq!(
        fs::read(path).unwrap(),
        original,
        "read-only delivery must not touch journal bytes"
    );
}
