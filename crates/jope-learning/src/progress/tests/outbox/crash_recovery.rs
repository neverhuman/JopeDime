use super::*;

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
