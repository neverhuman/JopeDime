use super::*;

#[test]
fn funding_lock_contention_expires_before_consumption_or_journal() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    let before = fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let blocker = OpenOptions::new()
        .read(true)
        .write(true)
        .open(fixture.authority.join("authority.lock"))
        .unwrap();
    blocker.try_lock().unwrap();
    let deadline = f::FundingClock::current().unwrap().monotonic_ns + 500_000_000;
    let started = std::time::Instant::now();
    let result = authority.funding_admit(
        &fixture.authorization,
        &fixture.reservation,
        &fixture.execution,
        None,
        &fixture.descriptor,
        deadline,
    );
    blocker.unlock().unwrap();
    drop(blocker);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "short admission used default10-second lock wait"
    );
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("lock admission deadline elapsed"));
    assert_eq!(
        fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
    authority
        .funding_preflight(
            &fixture.authorization,
            &fixture.reservation,
            fixture.reservation.body.demand.deadline_monotonic_ns,
        )
        .unwrap();
    let db = Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(
        db.query_row("SELECT COUNT(*) FROM funding_witness", [], |r| r
            .get::<_, u64>(0))
            .unwrap(),
        0
    );
}

#[test]
fn original_deadline_descriptor_and_installed_trust_fail_before_start_journal() {
    let mut fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    let before = fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &fixture.execution,
            None,
            b"{}",
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    let mut foreign = fixture.execution.clone();
    foreign.process.birth_ticks += 1;
    // Authority-internal inputs are trusted process observations; request
    // transport derives them itself. Here test a clear original runtime clash.
    foreign.runtime_sha256 = "8".repeat(64);
    assert!(authority
        .funding_admit(
            &fixture.authorization,
            &fixture.reservation,
            &foreign,
            None,
            &fixture.descriptor,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    fixture.reservation.body.demand.deadline_unix_ms =
        f::FundingClock::current().unwrap().unix_ms - 1;
    fixture.reservation =
        f::sign_reservation(fixture.reservation.body.clone(), &fixture.delegate).unwrap();
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
    let mut changed: Value =
        serde_json::from_slice(&fs::read(&fixture.installation).unwrap()).unwrap();
    changed["trust"]["authorization_sha256"] = json!("9".repeat(64));
    fs::write(&fixture.installation, serde_json::to_vec(&changed).unwrap()).unwrap();
    assert!(authority
        .funding_register(
            &fixture.authorization,
            fixture.reservation.body.demand.deadline_monotonic_ns
        )
        .is_err());
    assert_eq!(
        fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
}

#[test]
fn retained_v4_store_opens_without_migration_and_cannot_attach_funding() {
    let fixture = Fixture::new();
    let path = fixture.authority.join("authority.json");
    let mut manifest: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    manifest["schema"] = json!("parallel.fit_authority.v4");
    // Keep the sealed original v5 bytes. Construct a separate private v4
    // fixture inode; production authority immutability remains unchanged.
    fs::rename(&path, path.with_extension("v5-fixture.json")).unwrap();
    fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let db = Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    db.execute_batch("DROP TABLE funding_authorizations; DROP TABLE funding_reservations; DROP TABLE funding_participants; DROP TABLE funding_witness; PRAGMA user_version=4;").unwrap();
    drop(db);
    let before = fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let mut authority = Authority::open(&fixture.authority, &fixture.watermark).unwrap();
    assert!(authority
        .attach_funding(InstalledFunding::for_test(&fixture.installation).unwrap())
        .unwrap_err()
        .to_string()
        .contains("v5 authority"));
    drop(authority);
    assert_eq!(
        fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
    let db = Connection::open(fixture.authority.join("registry.sqlite")).unwrap();
    assert_eq!(
        db.query_row("PRAGMA user_version", [], |r| r.get::<_, u32>(0))
            .unwrap(),
        4
    );
    assert_eq!(
        db.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'funding_%'",
            [],
            |r| r.get::<_, u64>(0)
        )
        .unwrap(),
        0
    );
}

#[test]
fn metadata_funding_operations_obey_transport_deadline_without_journaling() {
    let fixture = Fixture::new();
    let mut authority = fixture.open();
    fixture.reserve(&mut authority);
    let before = fs::read(fixture.authority.join("dispatch.jsonl")).unwrap();
    let blocker = OpenOptions::new()
        .read(true)
        .write(true)
        .open(fixture.authority.join("authority.lock"))
        .unwrap();
    blocker.try_lock().unwrap();
    for operation in 0..3 {
        let deadline = f::FundingClock::current().unwrap().monotonic_ns + 100_000_000;
        let started = std::time::Instant::now();
        let result = match operation {
            0 => authority
                .funding_register(&fixture.authorization, deadline)
                .map(|_| ()),
            1 => authority
                .funding_reserve(&fixture.authorization, &fixture.reservation, deadline)
                .map(|_| ()),
            _ => {
                authority.funding_preflight(&fixture.authorization, &fixture.reservation, deadline)
            }
        };
        assert!(
            started.elapsed() < Duration::from_secs(2),
            "metadata operation exceeded transport lock window"
        );
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("lock admission deadline elapsed"));
        assert_eq!(
            fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),
            before
        );
    }
    blocker.unlock().unwrap();
    drop(blocker);
    // Expired uncontended requests must fail too; releasing contention must not
    // revive the old request. A distinguishable new request may still preflight.
    let expired = f::FundingClock::current().unwrap().monotonic_ns - 1;
    assert!(authority
        .funding_register(&fixture.authorization, expired)
        .is_err());
    assert!(authority
        .funding_reserve(&fixture.authorization, &fixture.reservation, expired)
        .is_err());
    assert!(authority
        .funding_preflight(&fixture.authorization, &fixture.reservation, expired)
        .is_err());
    assert_eq!(
        fs::read(fixture.authority.join("dispatch.jsonl")).unwrap(),
        before
    );
    authority
        .funding_preflight(
            &fixture.authorization,
            &fixture.reservation,
            fixture.reservation.body.demand.deadline_monotonic_ns,
        )
        .unwrap();
}
