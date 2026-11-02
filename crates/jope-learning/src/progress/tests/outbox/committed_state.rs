use super::*;

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
