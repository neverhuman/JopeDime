use super::*;

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
