use super::*;

#[test]
fn a_full_outbox_pauses_new_work_and_never_drops_or_grows() {
    let e = env();
    let remote = Mirror::open("memory:///").unwrap();
    let r = run("run-w8-full", 3, 3);
    let limits = Limits {
        pause_entries: 4,
        pause_bytes: 1 << 20,
        max_entries: 6,
        max_bytes: 1 << 20,
    };
    let mut o = Outbox::open(&e.state, limits).unwrap();
    let mut peak = 0;
    let mut track = |o: &Outbox| {
        assert!(o.pending_entries() <= limits.max_entries);
        peak = peak.max(o.pending_entries());
    };
    let f1 = r.failed(1, 1, FailureClass::Timeout, true);
    for item in [
        Item::Object(r.manifest.encoded().unwrap()),
        obj(r.scheduled(0)),
        obj(r.scheduled(1)),
        obj(r.started(0, 1)),
    ] {
        item.append(&mut o).unwrap();
        track(&o);
    }
    assert_eq!(
        o.admission(),
        Admission::Paused {
            pending_entries: 4,
            pending_bytes: o.pending_bytes()
        }
    );

    // Paused: new work is refused at this committed boundary, and nothing is written.
    let dir = Outbox::dir(&e.state);
    let before = snapshot(&dir);
    assert!(matches!(
        refusal(obj(r.scheduled(2)).append(&mut o)),
        Refused::Paused {
            pending_entries: 4,
            ..
        }
    ));
    assert_eq!(snapshot(&dir), before);

    // Outcomes of work already in flight are still recorded: failure accounting is preserved.
    obj(f1.clone()).append(&mut o).unwrap();
    track(&o);
    Item::Blob(Run::result(0, 1)).append(&mut o).unwrap();
    track(&o);
    assert_eq!(o.pending_entries(), limits.max_entries);

    // Full: every append is refused and nothing is written or dropped.
    let before = snapshot(&dir);
    assert_eq!(
        refusal(obj(r.completed(0, 1)).append(&mut o)),
        Refused::Full {
            pending_entries: 6,
            pending_bytes: o.pending_bytes(),
            item_bytes: r.completed(0, 1).encoded().unwrap().bytes.len() as u64,
        }
    );
    assert!(matches!(
        refusal(obj(r.retry(1, 2, &f1)).append(&mut o)),
        Refused::Paused { .. }
    ));
    assert!(matches!(
        refusal(Item::Blob(Run::result(1, 2)).append(&mut o)),
        Refused::Full { .. }
    ));
    assert_eq!(snapshot(&dir), before);
    assert_eq!(o.records(), 6);

    // Delivery drains it; the refused items are appended again by their producers.
    o.deliver(&remote, Some(2)).unwrap();
    assert!(matches!(o.admission(), Admission::Paused { .. }));
    obj(r.completed(0, 1)).append(&mut o).unwrap();
    track(&o);
    assert!(matches!(
        refusal(obj(r.scheduled(2)).append(&mut o)),
        Refused::Paused { .. }
    ));
    o.deliver(&remote, Some(3)).unwrap();
    assert_eq!(o.admission(), Admission::Open);
    for item in [
        obj(r.scheduled(2)),
        obj(r.retry(1, 2, &f1)),
        Item::Blob(Run::result(1, 2)),
        obj(r.completed(1, 2)),
    ] {
        item.append(&mut o).unwrap();
        track(&o);
    }
    let d = o.deliver(&remote, None).unwrap();
    assert_eq!((d.pending, d.failure.is_none()), (0, true));
    assert_eq!(
        peak, limits.max_entries,
        "the bound was reached and never exceeded"
    );

    // Nothing was dropped: every produced item is on the remote exactly once, and the census
    // rebuilt from it still counts the failure recorded while paused.
    assert_eq!(o.records(), 11);
    assert_eq!(remote_keys(&remote).len(), 11);
    let events: Vec<Decoded<Event>> = remote
        .list("progress/objects/event/")
        .unwrap()
        .iter()
        .map(|k| decode::<Event>(&remote.get(k).unwrap().unwrap()).unwrap())
        .collect();
    let c = reconcile(&r.manifest, &events, &[]);
    assert!(c.violations.is_empty(), "{:?}", c.violations);
    assert_eq!(c.counts.failures["timeout"], 1);
    assert_eq!((c.counts.completed, c.counts.retries), (2, 1));
    assert_eq!(
        c.unit(&r.work[2]).unwrap().disposition,
        Disposition::InFlight { attempt: 1 }
    );

    // The byte bound holds the same way.
    let e = env();
    let mut o = Outbox::open(
        &e.state,
        Limits {
            pause_entries: 10,
            pause_bytes: 64,
            max_entries: 10,
            max_bytes: 100,
        },
    )
    .unwrap();
    assert!(matches!(
        refusal(o.append_blob(&[7u8; 101])),
        Refused::Full {
            item_bytes: 101,
            ..
        }
    ));
    o.append_blob(&[7u8; 64]).unwrap();
    assert!(matches!(o.admission(), Admission::Paused { .. }));
    assert!(matches!(
        refusal(o.append_blob(&[8u8; 40])),
        Refused::Full { .. }
    ));
    o.append_blob(&[8u8; 36]).unwrap();
    assert_eq!(o.pending_bytes(), 100);
    assert!(Limits {
        pause_entries: 11,
        ..o.limits()
    }
    .check()
    .is_err());
}
