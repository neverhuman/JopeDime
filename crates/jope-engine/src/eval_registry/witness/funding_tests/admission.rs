use super::*;

#[test]
fn original_start_and_rank_are_witnessed_retained_and_single_use() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let server = fixture.server(2, false);
    let start = authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns,
        )
        .unwrap();
    fixture.verify_proof(&start, None);
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let rank = authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns,
        )
        .unwrap();
    fixture.verify_proof(&rank, Some(0));
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(1),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .unwrap_err()
        .to_string()
        .contains("participant process already consumed"));
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(2),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let originals = server.join().unwrap();
    assert_eq!(originals.len(), 2);
    // Even the real authority signing key cannot ask the independent witness
    // to attest a funding postimage that differs from the signed original.
    let client = WitnessClient::new(fixture.client.clone()).unwrap();
    let mut changed: Value = serde_json::from_slice(&originals[0].body.event).unwrap();
    changed["effects"][0]["row"]["state"] = json!("reserved");
    assert!(client
        .request(&serde_json::to_vec(&changed).unwrap())
        .is_err());
    let mut changed: Value = serde_json::from_slice(&originals[1].body.event).unwrap();
    changed["effects"][0]["row"]["slot"] = json!("1");
    assert!(client
        .request(&serde_json::to_vec(&changed).unwrap())
        .is_err());
    drop(authority);
    let db = Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM funding_witness", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        2
    );
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM funding_participants", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        1
    );
    drop(db);
    let mut reopened = fixture.open();
    assert!(reopened
        .funding_preflight(
            &fixture.authorization,
            &fixture.reservation,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    assert!(reopened
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let report = Authority::recovery_inspect(&fixture.authority, &fixture.watermark).unwrap();
    assert_eq!(report["retained_artifact_errors"], json!([]));
}

#[test]
fn lost_ack_preserves_original_witness_and_blocks_replay_or_rank_release() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    let server = fixture.server(1, true);
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let original = server.join().unwrap().pop().unwrap();
    let store =
        WitnessStore::open_read_only(&fixture.witness, &fixture.policy, &fixture.ack_key).unwrap();
    let ack = store
        .get_ack(&original)
        .unwrap()
        .expect("original ACK survives lost response");
    ack.verify(&fixture.policy, &original).unwrap();
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    drop(authority);
    assert!(Authority::open(&fixture.authority, &fixture.watermark).is_err());
    let db = Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM funding_participants", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        0
    );
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM funding_reservations", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        1
    );
    assert!(Authority::recovery_replay(&fixture.authority, &fixture.watermark).is_err());
}

#[test]
fn short_admission_deadline_keeps_late_original_ack_ambiguous() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    // A deployed witness opens and qualifies its executable before accepting
    // requests. Keep that startup work outside the original two-second window.
    let mut witness_store =
        WitnessStore::open(&fixture.witness, &fixture.policy, &fixture.ack_key).unwrap();
    let signed_deadline = fixture.reservation.body.demand.deadline_monotonic_ns;
    let admission_deadline = f::FundingClock::current().unwrap().monotonic_ns + 2_000_000_000;
    assert!(admission_deadline < signed_deadline);
    let listener = UnixListener::bind(&fixture.socket).unwrap();
    listener.set_nonblocking(true).unwrap();
    let server = std::thread::spawn(move || {
        let stop = std::time::Instant::now() + Duration::from_secs(5);
        let mut stream = loop {
            match listener.accept() {
                Ok((stream, _)) => break stream,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < stop,
                        "bounded original funding request absent"
                    );
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(error) => panic!("private witness accept: {error}"),
            }
        };
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let request: SignedRequest = read_frame(&mut stream).unwrap();
        let ack = witness_store.record_authorization(&request).unwrap();
        assert!(
            f::FundingClock::current().unwrap().monotonic_ns < admission_deadline,
            "fixture did not retain ACK inside original admission window"
        );
        let now = f::FundingClock::current().unwrap().monotonic_ns;
        std::thread::sleep(Duration::from_nanos(
            admission_deadline.saturating_sub(now) + 100_000_000,
        ));
        let _ = write_frame(&mut stream, &ack);
        (request, ack)
    });
    let started = std::time::Instant::now();
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            admission_deadline
        )
        .is_err());
    assert!(
        started.elapsed() < Duration::from_secs(10),
        "short admission inherited long task deadline"
    );
    let (original, ack) = server.join().unwrap();
    let event: Value = serde_json::from_slice(&original.body.event).unwrap();
    assert_eq!(
        event["payload"]["admission_deadline_monotonic_ns"],
        admission_deadline
    );
    assert_eq!(
        event["payload"]["original"]["demand"]["deadline_monotonic_ns"],
        signed_deadline
    );
    assert!(f::FundingClock::current().unwrap().monotonic_ns < signed_deadline);
    let store =
        WitnessStore::open_read_only(&fixture.witness, &fixture.policy, &fixture.ack_key).unwrap();
    assert_eq!(store.get_ack(&original).unwrap(), Some(ack));
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            &fixture.descriptor,
            signed_deadline
        )
        .is_err());
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            Some(0),
            &fixture.descriptor,
            signed_deadline
        )
        .is_err());
    drop(authority);
    assert!(Authority::open(&fixture.authority, &fixture.watermark).is_err());
    assert!(Authority::recovery_replay(&fixture.authority, &fixture.watermark).is_err());
}
