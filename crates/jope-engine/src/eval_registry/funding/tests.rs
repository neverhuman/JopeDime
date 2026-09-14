//! Private cryptographic/accounting fixtures. These never install trust, launch
//! work, or stand in for the authority service's independent witness tests.
use super::*;

struct Fixture {
    authorizer: SigningKey,
    delegate: SigningKey,
    trust: FundingTrust,
    authorization: SignedAuthorization,
    reservation: SignedReservation,
    clock: FundingClock,
    execution: ExecutionClaim,
}

impl Fixture {
    fn new() -> Self {
        let authorizer = SigningKey::from_bytes(&[11; 32]);
        let delegate = SigningKey::from_bytes(&[29; 32]);
        let public = hex::encode(delegate.verifying_key().to_bytes());
        let body = FundingAuthorization {
            schema: "parallel.funding_authorization.v1".into(), domain_id: "private-fixture-stage".into(),
            scope: "private cryptographic fixture; not a production grant".into(),
            period_start_unix_ms: 100, period_end_unix_ms: 10_000,
            authority_evidence_sha256: "a".repeat(64), historical_reconciliation_sha256: "b".repeat(64),
            release_prerequisites_sha256: "a1".repeat(32),
            approved_task_sha256: BTreeSet::from(["e".repeat(64), "4".repeat(64), "9".repeat(64)]),
            historical_core_ms: 100, historical_node_ms: 50, core_ms: 100_000, node_ms: 10_000,
            max_concurrent_cpu_searches: 16,
            allocation: FundingAllocation { cpu_cores: 16, gpu_devices: (0..8).collect(),
                host_memory_bytes: 16_000, gpu_memory_bytes_per_device: 8_000 },
            capabilities: BTreeSet::from(["regression_training".into()]), reservation_public_key: public.clone(),
        };
        let authorization = SignedAuthorization { signature: hex::encode(authorizer.sign(&signed_bytes(AUTH_TAG, &body).unwrap()).to_bytes()), body };
        let trust = FundingTrust {
            schema: "parallel.funding_trust.v1".into(), domain_id: authorization.body.domain_id.clone(),
            authority_instance: "private-authority".into(), authority_epoch: "private-epoch".into(),
            authority_manifest_sha256: "c".repeat(64), authority_executable_sha256: "d".repeat(64),
            authorization_public_key: hex::encode(authorizer.verifying_key().to_bytes()),
            authorization_sha256: digest(&authorization.body).unwrap(), reservation_public_key: public,
            host_id: "private-host".into(), boot_id: "private-boot".into(),
        };
        let demand = TaskDemand {
            task_sha256: "e".repeat(64), capability: "regression_training".into(),
            host_id: trust.host_id.clone(), boot_id: trust.boot_id.clone(),
            executable_sha256: "f".repeat(64), runtime_sha256: "1".repeat(64),
            deadline_unix_ms: 500, deadline_monotonic_ns: 500_000_000,
            resources: ResourceCaps { cpu_cores: 2, gpus: 2, world: 2, host_memory_bytes: 4_000,
                gpu_memory_bytes_per_rank: 2_000, cpu_searches: 0, gpu_devices: vec![0, 1] },
        };
        let reservation = sign_reservation(TaskReservation {
            schema: "parallel.task_reservation.v1".into(), reservation_id: "2".repeat(64),
            authorization_sha256: trust.authorization_sha256.clone(), authority_instance: trust.authority_instance.clone(),
            authority_epoch: trust.authority_epoch.clone(), authority_manifest_sha256: trust.authority_manifest_sha256.clone(),
            authority_executable_sha256: trust.authority_executable_sha256.clone(), domain_id: trust.domain_id.clone(),
            not_before_unix_ms: 100, demand: demand.clone(), core_ms: 800, node_ms: 400, participant_slots: vec![0, 1],
        }, &delegate).unwrap();
        let execution = ExecutionClaim { process: ProcessIdentity { boot_id: trust.boot_id.clone(), pid: 71, birth_ticks: 81 },
            executable_sha256: demand.executable_sha256, runtime_sha256: demand.runtime_sha256,
            request_sha256: demand.task_sha256 };
        Self { authorizer, delegate, trust, authorization, reservation,
            clock: FundingClock { unix_ms: 200, monotonic_ns: 200_000_000 }, execution }
    }
    fn verified(&self) -> Result<VerifiedReservation> {
        verify_reservation(&self.trust, &self.authorization, &self.reservation, &self.reservation.body.demand, self.clock)
    }
    fn resign(&mut self) {
        self.reservation = sign_reservation(self.reservation.body.clone(), &self.delegate).unwrap();
    }
    fn reauthorize(&mut self) {
        self.authorization.signature = hex::encode(self.authorizer.sign(&signed_bytes(AUTH_TAG, &self.authorization.body).unwrap()).to_bytes());
        self.trust.authorization_sha256 = digest(&self.authorization.body).unwrap();
        self.reservation.body.authorization_sha256 = self.trust.authorization_sha256.clone();
        self.resign();
    }
    fn database(&self) -> Connection {
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(SQL_SCHEMA).unwrap();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        register_authorization(&db, &self.trust, &self.authorization, self.clock).unwrap();
        reserve(&db, &self.verified().unwrap()).unwrap();
        db.execute_batch("COMMIT").unwrap();
        db
    }
}

#[test]
fn real_signatures_and_pinned_decision_are_required() {
    let mut fixture = Fixture::new();
    fixture.verified().unwrap();
    fixture.authorization.signature = "00".repeat(64);
    assert!(fixture.verified().is_err());
    let mut fixture = Fixture::new();
    fixture.reservation.signature = "00".repeat(64);
    assert!(fixture.verified().is_err());
    let mut fixture = Fixture::new();
    fixture.authorization.body.core_ms += 1;
    fixture.authorization.signature = hex::encode(fixture.authorizer.sign(
        &signed_bytes(AUTH_TAG, &fixture.authorization.body).unwrap()).to_bytes());
    assert!(fixture.verified().unwrap_err().to_string().contains("pinned decision"));
    let fixture = Fixture::new();
    let mut raw = serde_json::to_value(&fixture.authorization).unwrap();
    raw["eligible"] = true.into();
    assert!(serde_json::from_value::<SignedAuthorization>(raw).is_err());
}

#[test]
fn oversized_original_authorization_is_rejected_before_registration() {
    for too_many_tasks in [true, false] {
        let mut fixture = Fixture::new();
        if too_many_tasks {
            fixture.authorization.body.approved_task_sha256 = (1..=MAX_APPROVED_TASKS + 1)
                .map(|id| format!("{id:064x}")).collect();
        } else {
            fixture.authorization.body.capabilities = (0..600)
                .map(|id| format!("{id:04}{}", "x".repeat(250))).collect();
        }
        fixture.reauthorize();
        let db = Connection::open_in_memory().unwrap();
        db.execute_batch(SQL_SCHEMA).unwrap();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        assert!(register_authorization(&db, &fixture.trust, &fixture.authorization, fixture.clock)
            .unwrap_err().to_string().contains("bounded witness input limits"));
        assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_authorizations", [],
            |row| row.get::<_, u64>(0)).unwrap(), 0);
        db.execute_batch("ROLLBACK").unwrap();
    }
}

#[test]
fn actual_request_identity_deadline_runtime_and_caps_cannot_be_substituted() {
    let fixture = Fixture::new();
    let original = fixture.reservation.body.demand.clone();
    let mut mutations = Vec::new();
    let mut changed = original.clone(); changed.task_sha256 = "3".repeat(64); mutations.push(changed);
    let mut changed = original.clone(); changed.host_id.push('x'); mutations.push(changed);
    let mut changed = original.clone(); changed.boot_id.push('x'); mutations.push(changed);
    let mut changed = original.clone(); changed.executable_sha256 = "4".repeat(64); mutations.push(changed);
    let mut changed = original.clone(); changed.runtime_sha256 = "5".repeat(64); mutations.push(changed);
    let mut changed = original.clone(); changed.deadline_unix_ms += 1; mutations.push(changed);
    let mut changed = original.clone(); changed.deadline_monotonic_ns += 1; mutations.push(changed);
    let mut changed = original.clone(); changed.resources.cpu_cores += 1; mutations.push(changed);
    let mut changed = original.clone(); changed.resources.host_memory_bytes += 1; mutations.push(changed);
    let mut changed = original; changed.resources.gpu_devices.reverse(); mutations.push(changed);
    for demand in mutations {
        assert!(verify_reservation(&fixture.trust, &fixture.authorization, &fixture.reservation, &demand, fixture.clock).is_err());
    }
    let db = fixture.database();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    let mut execution = fixture.execution.clone(); execution.request_sha256 = "6".repeat(64);
    assert!(consume_start(&db, &fixture.verified().unwrap(), &execution, fixture.clock).is_err());
    db.execute_batch("ROLLBACK").unwrap();
    preflight(&db, &fixture.verified().unwrap()).unwrap();
}

#[test]
fn signed_envelope_still_requires_valid_original_window_world_and_allocation() {
    for change in 0..9 {
        let mut fixture = Fixture::new();
        match change {
            0 => fixture.reservation.body.demand.resources.world = 3,
            1 => fixture.reservation.body.participant_slots.push(2),
            2 => fixture.reservation.body.demand.resources.gpu_devices = vec![0, 0],
            3 => fixture.reservation.body.demand.resources.cpu_searches = 17,
            4 => fixture.reservation.body.core_ms = 799,
            5 => fixture.reservation.body.node_ms = 399,
            6 => fixture.reservation.body.demand.deadline_monotonic_ns = u64::MAX,
            7 => fixture.reservation.body.demand.resources.host_memory_bytes = 16_001,
            8 => fixture.reservation.body.demand.task_sha256 = "6".repeat(64),
            _ => unreachable!(),
        }
        fixture.resign();
        assert!(fixture.verified().is_err(), "invalid signed fixture variant {change}");
    }
    let mut fixture = Fixture::new(); fixture.clock.unix_ms = 10_000;
    assert!(fixture.verified().is_err());
}

#[test]
fn transaction_and_domain_history_cannot_be_reset() {
    let fixture = Fixture::new();
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(SQL_SCHEMA).unwrap();
    assert!(register_authorization(&db, &fixture.trust, &fixture.authorization, fixture.clock).is_err());
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    register_authorization(&db, &fixture.trust, &fixture.authorization, fixture.clock).unwrap();
    assert!(register_authorization(&db, &fixture.trust, &fixture.authorization, fixture.clock).is_err());
    db.execute_batch("COMMIT").unwrap();
    assert!(reserve(&db, &fixture.verified().unwrap()).is_err());
}

#[test]
fn repeated_task_or_reservation_never_creates_new_funding() {
    let mut fixture = Fixture::new();
    let db = fixture.database();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    assert!(reserve(&db, &fixture.verified().unwrap()).is_err());
    fixture.reservation.body.reservation_id = "7".repeat(64);
    fixture.resign();
    assert!(reserve(&db, &fixture.verified().unwrap()).is_err());
    db.execute_batch("ROLLBACK").unwrap();
    assert_eq!(db.query_row("SELECT COUNT(*) FROM funding_reservations", [], |row| row.get::<_, u64>(0)).unwrap(), 1);
}

#[test]
fn historical_and_reserved_ceilings_are_charged_without_refunds_or_overflow() {
    let mut fixture = Fixture::new();
    fixture.authorization.body.core_ms = 900;
    fixture.authorization.body.node_ms = 450;
    fixture.reauthorize();
    let db = fixture.database();
    fixture.reservation.body.reservation_id = "8".repeat(64);
    fixture.reservation.body.demand.task_sha256 = "9".repeat(64);
    fixture.resign();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    assert!(reserve(&db, &fixture.verified().unwrap()).unwrap_err().to_string().contains("exhausted"));
    db.execute_batch("ROLLBACK").unwrap();
    let mut fixture = Fixture::new();
    fixture.authorization.body.historical_core_ms = u64::MAX;
    fixture.authorization.body.core_ms = u64::MAX;
    fixture.reauthorize();
    let db = Connection::open_in_memory().unwrap(); db.execute_batch(SQL_SCHEMA).unwrap();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    register_authorization(&db, &fixture.trust, &fixture.authorization, fixture.clock).unwrap();
    assert!(reserve(&db, &fixture.verified().unwrap()).is_err());
}

#[test]
fn preflight_does_not_consume_and_one_task_start_is_durable() {
    let fixture = Fixture::new(); let db = fixture.database(); let verified = fixture.verified().unwrap();
    preflight(&db, &verified).unwrap(); preflight(&db, &verified).unwrap();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    let event = consume_start(&db, &verified, &fixture.execution, fixture.clock).unwrap();
    assert_eq!(event.demand.task_sha256, fixture.execution.request_sha256);
    db.execute_batch("COMMIT").unwrap();
    assert!(preflight(&db, &verified).is_err());
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    assert!(consume_start(&db, &verified, &fixture.execution, fixture.clock).is_err());
    assert!(claim_participant(&db, &verified, 0, &fixture.execution, fixture.clock)
        .unwrap_err().to_string().contains("witness ACK"));
    db.execute_batch("ROLLBACK").unwrap();
}

#[test]
fn overlapping_gpu_and_aggregate_host_allocations_are_rejected_atomically() {
    for mode in 0..3 {
        let mut fixture = Fixture::new();
        if mode == 1 { fixture.authorization.body.allocation.cpu_cores = 3; }
        if mode == 2 { fixture.authorization.body.allocation.host_memory_bytes = 7_000; }
        fixture.reauthorize();
        let db = fixture.database();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        consume_start(&db, &fixture.verified().unwrap(), &fixture.execution, fixture.clock).unwrap();
        db.execute_batch("COMMIT").unwrap();
        fixture.reservation.body.reservation_id = "3".repeat(64);
        fixture.reservation.body.demand.task_sha256 = "4".repeat(64);
        fixture.execution.request_sha256 = "4".repeat(64);
        if mode > 0 { fixture.reservation.body.demand.resources.gpu_devices = vec![2, 3]; }
        fixture.resign();
        db.execute_batch("BEGIN IMMEDIATE").unwrap();
        reserve(&db, &fixture.verified().unwrap()).unwrap();
        assert!(consume_start(&db, &fixture.verified().unwrap(), &fixture.execution, fixture.clock).is_err());
        db.execute_batch("ROLLBACK").unwrap();
    }
}

#[test]
fn consumed_window_and_foreign_process_cannot_be_reused() {
    let fixture = Fixture::new(); let verified = fixture.verified().unwrap(); let db = fixture.database();
    db.execute_batch("BEGIN IMMEDIATE").unwrap();
    let late = FundingClock { unix_ms: 500, monotonic_ns: 500_000_000 };
    assert!(consume_start(&db, &verified, &fixture.execution, late).is_err());
    let mut foreign = fixture.execution.clone(); foreign.process.boot_id = "another-boot".into();
    assert!(consume_start(&db, &verified, &foreign, fixture.clock).is_err());
    let mut foreign = fixture.execution.clone(); foreign.process.birth_ticks = 0;
    assert!(consume_start(&db, &verified, &foreign, fixture.clock).is_err());
    db.execute_batch("ROLLBACK").unwrap();
    preflight(&db, &verified).unwrap();
}

#[test]
fn original_witness_event_can_be_verified_after_expiry_without_authorizing_a_new_start() {
    let fixture = Fixture::new(); let verified = fixture.verified().unwrap();
    let value = event(&verified, &fixture.execution, None, fixture.clock).unwrap();
    let (key, deadline) = validate_witness_event(&fixture.trust, &value,
        &fixture.trust.authority_instance, &fixture.trust.authority_epoch).unwrap();
    assert_eq!(key, value.event_key); assert_eq!(deadline, value.demand.deadline_monotonic_ns);
    assert!(verify_event(&fixture.trust, &value, FundingClock { unix_ms: 501, monotonic_ns: 501_000_000 }).is_err());
    let mut changed = value.clone(); changed.claimed_unix_ms = 501;
    assert!(validate_witness_event(&fixture.trust, &changed,
        &fixture.trust.authority_instance, &fixture.trust.authority_epoch).is_err());
    let mut changed = value; changed.participant_slot = Some(0);
    assert!(validate_witness_event(&fixture.trust, &changed,
        &fixture.trust.authority_instance, &fixture.trust.authority_epoch).is_err());
}
