//! W8-T02: the durable outbox. A crash after a local append but before acknowledgement or
//! delivery recovers exactly once logically, a full outbox pauses instead of dropping or growing,
//! and only verified content-addressed items enter it.

use super::fixture::{
    env, obj, producer, remote_keys, run, snapshot, Fault, Faulty, Item, Run, BIG,
};
use crate::mirror::Mirror;
use crate::progress::census::{reconcile, Disposition};
use crate::progress::inventory::*;
use crate::progress::outbox::{Admission, Appended, Boundary, Limits, Outbox, Outcome, Refused};
use crate::state::State;
use anyhow::Result;
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

fn refusal(result: Result<Appended>) -> Refused {
    result
        .expect_err("append was admitted")
        .downcast::<Refused>()
        .expect("refused for another reason")
}

#[test]
fn crash_after_local_append_before_delivery_recovers_exactly_once() {
    let r = run("run-w8-t02", 2, 3);
    let items = producer(&r);
    let n = items.len() as u64;
    let expected: BTreeSet<String> = items.iter().map(Item::key).collect();
    assert_eq!(expected.len() as u64, n);

    // Append side: crash at each append boundary, at an object, a blob and a retry.
    for boundary in [Boundary::Payload, Boundary::Record] {
        for k in [0usize, 3, 7] {
            let e = env();
            {
                let mut o = Outbox::open(&e.state, BIG).unwrap();
                for item in &items[..k] {
                    item.append(&mut o).unwrap();
                }
                o.crash_at = Some(boundary);
                let crash = items[k].append(&mut o).unwrap_err();
                assert!(crash.to_string().contains("simulated crash"), "{crash:#}");
                // The process is gone here: nothing held in memory survives.
            }
            // An interrupted atomic write that never reached its rename.
            let log = Outbox::dir(&e.state).join("log");
            fs::write(log.join(".00000000000000000099.json.tmp-4242"), b"{\"sch").unwrap();

            let mut o = Outbox::open(&e.state, BIG).unwrap();
            let rec = o.recovery().clone();
            let committed = k as u64 + u64::from(boundary == Boundary::Record);
            assert_eq!(
                (rec.records, rec.pending, rec.acknowledged),
                (committed, committed, 0)
            );
            assert_eq!(
                rec.uncommitted,
                vec!["log/.00000000000000000099.json.tmp-4242".to_string()]
            );
            let orphan = boundary == Boundary::Payload && matches!(items[k], Item::Object(_));
            assert_eq!(rec.orphan_payloads.len(), usize::from(orphan), "{rec:?}");

            // The producer does not know what landed and replays everything it produced.
            for (i, item) in items.iter().enumerate() {
                let appended = item.append(&mut o).unwrap();
                assert_eq!(appended.seq(), i as u64 + 1, "sequence order changed");
                assert_eq!(
                    matches!(appended, Appended::Existing { .. }),
                    (i as u64) < committed,
                    "item {i}: {appended:?}"
                );
            }
            assert_eq!(o.records(), n);

            let d = o.deliver(&e.remote, None).unwrap();
            assert_eq!(
                (d.attempted, d.created, d.existed, d.pending),
                (n, n, 0, 0),
                "{boundary:?} at {k}: {d:?}"
            );
            assert!(d.failure.is_none());
            assert_eq!(
                remote_keys(&e.remote),
                expected,
                "exactly the produced items"
            );
            assert_eq!(o.deliver(&e.remote, None).unwrap().attempted, 0);
            drop(o);
            let o = Outbox::open(&e.state, BIG).unwrap();
            assert_eq!(
                (o.recovery().acknowledged, o.recovery().pending),
                (n, 0),
                "every item acknowledged once"
            );
            assert!(o.recovery().orphan_payloads.is_empty());
        }
    }

    // Delivery side: crash after the remote accepted an object but before its ack, and after the
    // ack. Neither loses an item or creates one twice.
    for (boundary, created_after, existed_after) in
        [(Boundary::RemotePut, n - 5, 1), (Boundary::Ack, n - 5, 0)]
    {
        let e = env();
        {
            let mut o = Outbox::open(&e.state, BIG).unwrap();
            for item in &items {
                item.append(&mut o).unwrap();
            }
            assert_eq!(o.deliver(&e.remote, Some(4)).unwrap().created, 4);
            o.crash_at = Some(boundary);
            let crash = o.deliver(&e.remote, None).unwrap_err();
            assert!(crash.to_string().contains("simulated crash"), "{crash:#}");
        }
        assert_eq!(
            remote_keys(&e.remote).len(),
            5,
            "the fifth object reached the remote"
        );
        let mut o = Outbox::open(&e.state, BIG).unwrap();
        let acked = 4 + u64::from(boundary == Boundary::Ack);
        assert_eq!(o.recovery().acknowledged, acked);
        assert_eq!(o.recovery().pending, n - acked);
        let d = o.deliver(&e.remote, None).unwrap();
        assert_eq!(
            (d.created, d.existed, d.pending),
            (created_after, existed_after, 0),
            "{boundary:?}: {d:?}"
        );
        assert_eq!(remote_keys(&e.remote), expected);
        let fifth = o.ack(5).unwrap();
        assert_eq!(
            fifth.outcome,
            if boundary == Boundary::RemotePut {
                Outcome::Existed
            } else {
                Outcome::Created
            }
        );
        // Every acknowledgement binds the exact bytes the remote holds.
        for seq in 1..=n {
            let ack = o.ack(seq).unwrap();
            let held = e.remote.get(&ack.key).unwrap().unwrap();
            assert_eq!(crate::canon::sha256_hex(&held), ack.bytes_sha256);
            assert_eq!(ack.bytes_sha256, o.record(seq).unwrap().bytes_sha256);
        }
    }
}

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

#[test]
fn a_failed_delivery_stays_pending_and_is_never_acknowledged() {
    let e = env();
    let r = run("run-w8-deliver", 2, 3);
    let items = producer(&r);
    let n = items.len() as u64;
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    for item in &items {
        item.append(&mut o).unwrap();
    }

    // The request never reaches the store: nothing after it is attempted or acknowledged.
    let faulty = Faulty::new(&e.remote).fault_put(2, Fault::Disconnect);
    let d = o.deliver(&faulty, None).unwrap();
    assert_eq!((d.attempted, d.created, d.pending), (3, 2, n - 2));
    let failure = d.failure.unwrap();
    assert_eq!((failure.seq, failure.key.clone()), (3, items[2].key()));
    assert!(o.ack(3).is_none());
    assert!(!remote_keys(&e.remote).contains(&items[2].key()));

    // The store applies the write but the response is lost: still not acknowledged.
    let faulty = Faulty::new(&e.remote).fault_put(0, Fault::ResponseLost);
    let d = o.deliver(&faulty, None).unwrap();
    assert_eq!((d.attempted, d.created, d.pending), (1, 0, n - 2));
    assert_eq!(d.failure.unwrap().seq, 3);
    assert!(o.ack(3).is_none());
    assert!(remote_keys(&e.remote).contains(&items[2].key()));

    // After a restart the entry is still pending and is acknowledged once, as already present.
    drop(o);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    assert_eq!(o.recovery().pending, n - 2);
    let d = o.deliver(&e.remote, None).unwrap();
    assert_eq!((d.created, d.existed, d.pending), (n - 3, 1, 0));
    assert_eq!(o.ack(3).unwrap().outcome, Outcome::Existed);
    assert_eq!(remote_keys(&e.remote).len() as u64, n);

    // A remote object under the same key with other bytes is never acknowledged or overwritten.
    let e = env();
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    items[0].append(&mut o).unwrap();
    e.remote
        .put_immutable(&items[0].key(), b"not the manifest")
        .unwrap();
    let d = o.deliver(&e.remote, None).unwrap();
    assert!(d.failure.unwrap().error.contains("different bytes"));
    assert!(o.ack(1).is_none());
    assert_eq!(
        e.remote.get(&items[0].key()).unwrap().unwrap(),
        b"not the manifest"
    );
}

#[test]
fn dependencies_are_recorded_and_delivered_first() {
    let e = env();
    let r = run("run-w8-deps", 2, 3);
    let items = producer(&r);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let dir = Outbox::dir(&e.state);

    let before = snapshot(&dir);
    for early in [1usize, 4, 7] {
        let err = items[early].append(&mut o).unwrap_err();
        assert!(
            format!("{err:#}").contains("has not recorded"),
            "item {early}: {err:#}"
        );
    }
    assert_eq!(snapshot(&dir), before, "a refused append wrote something");
    assert_eq!(o.records(), 0);

    for item in &items {
        item.append(&mut o).unwrap();
    }
    for (i, item) in items.iter().enumerate() {
        let record = o.record(i as u64 + 1).unwrap();
        assert_eq!(
            record
                .dependencies
                .iter()
                .map(Dependency::key)
                .collect::<BTreeSet<_>>(),
            item.deps().into_iter().collect::<BTreeSet<_>>()
        );
    }
    let faulty = Faulty::new(&e.remote);
    o.deliver(&faulty, None).unwrap();
    let order = faulty.keys("put");
    assert_eq!(order.len(), items.len());
    for item in &items {
        let at = order.iter().position(|k| *k == item.key()).unwrap();
        for dep in item.deps() {
            let dep_at = order.iter().position(|k| *k == dep).unwrap();
            assert!(dep_at < at, "{dep} delivered after {}", item.key());
        }
    }
}

#[test]
fn only_verified_content_addressed_items_enter() {
    let e = env();
    let r = run("run-w8-verified", 1, 3);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let manifest = r.manifest.encoded().unwrap();
    o.append_object(&manifest).unwrap();
    let dir = Outbox::dir(&e.state);
    let before = snapshot(&dir);
    let blobs_before = snapshot(&e.state.root().join("blobs"));

    // A worker result that is not the bytes its completion will reference.
    let err = o
        .append_result(b"forged result", &BlobRef::of(&Run::result(0, 1)))
        .unwrap_err();
    assert!(err.to_string().contains("not the referenced"), "{err:#}");

    let good = r.scheduled(0).encoded().unwrap();
    let mut tampered = good.clone();
    let last = tampered.bytes.len() - 3;
    tampered.bytes[last] ^= 0x01;
    let mut other_kind = good.clone();
    other_kind.kind = Kind::CandidateEvidence;
    other_kind.digest = domain_digest(Kind::CandidateEvidence.schema(), &good.bytes);
    let mut no_deps = good.clone();
    no_deps.dependencies.clear();
    let pretty = serde_json::to_vec_pretty(r.scheduled(0).value()).unwrap();
    let noncanonical = Encoded {
        digest: domain_digest(Kind::Event.schema(), &pretty),
        bytes: pretty,
        ..good.clone()
    };
    // A run pointer offered as if it were an object: pointers never enter the outbox.
    let head = Head {
        schema: HEAD_SCHEMA.into(),
        run_id: r.id().into(),
        commit: good.digest.clone(),
        height: 0,
    }
    .encode()
    .unwrap();
    let pointer = Encoded {
        kind: Kind::Commit,
        run_id: r.id().into(),
        digest: domain_digest(Kind::Commit.schema(), &head),
        bytes: head,
        dependencies: Vec::new(),
    };
    for (label, forged) in [
        ("tampered", tampered),
        ("other kind", other_kind),
        ("dependencies hidden", no_deps),
        ("noncanonical", noncanonical),
        ("run pointer", pointer),
    ] {
        assert!(o.append_object(&forged).is_err(), "{label} was admitted");
    }
    assert_eq!(snapshot(&dir), before);
    assert_eq!(snapshot(&e.state.root().join("blobs")), blobs_before);
    assert_eq!(o.records(), 1);
    // Positive control.
    assert_eq!(o.append_object(&good).unwrap(), Appended::Added { seq: 2 });
}

fn copy_tree(from: &Path, to: &Path) {
    for entry in walkdir::WalkDir::new(from) {
        let entry = entry.unwrap();
        let dest = to.join(entry.path().strip_prefix(from).unwrap());
        if entry.file_type().is_dir() {
            fs::create_dir_all(&dest).unwrap();
        } else {
            fs::copy(entry.path(), &dest).unwrap();
        }
    }
}

#[test]
fn committed_state_is_verified_on_reopen() {
    let e = env();
    let r = run("run-w8-reopen", 2, 3);
    let items = producer(&r);
    {
        let mut o = Outbox::open(&e.state, BIG).unwrap();
        for item in &items {
            item.append(&mut o).unwrap();
        }
        o.deliver(&e.remote, Some(2)).unwrap();
        let second = Outbox::open(&e.state, BIG)
            .err()
            .expect("second writer admitted");
        assert!(
            second.to_string().contains("held by another writer"),
            "{second:#}"
        );
    }
    let manifest_payload = format!(
        "progress/outbox/objects/run-manifest/{}.json",
        r.manifest.digest()
    );
    let result_blob = State::blob_rel(&BlobRef::of(&Run::result(0, 1)).sha256);
    type Tamper = Box<dyn Fn(&Path)>;
    let tampers: Vec<(&str, Tamper)> = vec![
        (
            "a record removed",
            Box::new(|s: &Path| {
                fs::remove_file(s.join("progress/outbox/log/00000000000000000005.json")).unwrap()
            }),
        ),
        (
            "a record rewritten",
            Box::new(|s: &Path| {
                let p = s.join("progress/outbox/log/00000000000000000004.json");
                let text = fs::read_to_string(&p).unwrap();
                fs::write(&p, text.replacen("\"seq\":4", "\"seq\":40", 1)).unwrap();
            }),
        ),
        (
            "a record replaced by another item's",
            Box::new(|s: &Path| {
                let log = s.join("progress/outbox/log");
                fs::copy(
                    log.join("00000000000000000002.json"),
                    log.join("00000000000000000003.json"),
                )
                .unwrap();
            }),
        ),
        (
            "an object payload altered",
            Box::new(move |s: &Path| {
                let p = s.join(&manifest_payload);
                let mut bytes = fs::read(&p).unwrap();
                let last = bytes.len() - 3;
                bytes[last] ^= 0x01;
                fs::write(&p, bytes).unwrap();
            }),
        ),
        (
            "a blob payload lost",
            Box::new(move |s: &Path| fs::remove_file(s.join(&result_blob)).unwrap()),
        ),
        (
            "an ack for another record",
            Box::new(|s: &Path| {
                let acks = s.join("progress/outbox/acks");
                fs::copy(
                    acks.join("00000000000000000001.json"),
                    acks.join("00000000000000000006.json"),
                )
                .unwrap();
            }),
        ),
        (
            "an ack for other bytes",
            Box::new(|s: &Path| {
                let p = s.join("progress/outbox/acks/00000000000000000002.json");
                let text = fs::read_to_string(&p).unwrap();
                let at = text.find("\"bytes_sha256\":\"").unwrap() + 16;
                let flipped = if &text[at..=at] == "0" { "1" } else { "0" };
                fs::write(&p, format!("{}{flipped}{}", &text[..at], &text[at + 1..])).unwrap();
            }),
        ),
        (
            "an ack for no record",
            Box::new(|s: &Path| {
                let acks = s.join("progress/outbox/acks");
                fs::copy(
                    acks.join("00000000000000000001.json"),
                    acks.join("00000000000000000099.json"),
                )
                .unwrap();
            }),
        ),
        (
            "an unknown file",
            Box::new(|s: &Path| {
                fs::write(s.join("progress/outbox/log/notes.txt"), b"hello").unwrap()
            }),
        ),
    ];
    for (label, tamper) in &tampers {
        let copy = tempfile::tempdir().unwrap();
        copy_tree(e.state.root(), copy.path());
        tamper(copy.path());
        let before = snapshot(copy.path());
        let err = Outbox::open(&State::at(copy.path()), BIG)
            .err()
            .unwrap_or_else(|| panic!("{label}: opened"));
        assert_eq!(
            snapshot(copy.path()),
            before,
            "{label}: open wrote something"
        );
        eprintln!("{label}: {err:#}");
    }
    // Positive control: an untouched copy opens with the same state.
    let copy = tempfile::tempdir().unwrap();
    copy_tree(e.state.root(), copy.path());
    let o = Outbox::open(&State::at(copy.path()), BIG).unwrap();
    assert_eq!(
        (
            o.recovery().records,
            o.recovery().acknowledged,
            o.recovery().pending
        ),
        (items.len() as u64, 2, items.len() as u64 - 2)
    );
}
