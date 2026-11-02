use super::*;

#[test]
fn native_candidate_ledger_reopens_without_renewing_deadline_or_budget() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().join("authority");
    let witness = directory.path().join("witness");
    Authority::initialize_native(
        &root,
        &witness,
        vec![ScopePolicy {
            scope: "ledger-only-no-fit".into(),
            training_sha256: vec!["1".repeat(64)],
            numerical_contract_sha256: vec!["2".repeat(64)],
            fit_seeds: vec![0],
        }],
    )
    .unwrap();
    let task = json!({"budget":1,"task_manifest_sha256":hash(b"task"),"deadline_unix_ms":1000});
    let slot = json!({"task_key":"task","candidate_index":0,"fit_key":hash(b"fit"),"descriptor_sha256":hash(b"descriptor"),"boot_id":"original-boot","deadline_monotonic_ns":500});
    let mut authority = Authority::open(&root, &witness).unwrap();
    assert_eq!(
        authority
            .reserve_native_candidate("task", "slot", &task, &slot)
            .unwrap(),
        slot
    );
    drop(authority);
    let mut authority = Authority::open(&root, &witness).unwrap();
    let mut later = slot.clone();
    later["deadline_monotonic_ns"] = json!(99999);
    assert_eq!(
        authority
            .reserve_native_candidate("task", "slot", &task, &later)
            .unwrap(),
        slot
    );
    let mut conflict = slot.clone();
    conflict["descriptor_sha256"] = json!(hash(b"different descriptor"));
    assert!(authority
        .reserve_native_candidate("task", "slot", &task, &conflict)
        .is_err());
    let mut reboot = slot.clone();
    reboot["boot_id"] = json!("new-boot");
    assert!(authority
        .reserve_native_candidate("task", "slot", &task, &reboot)
        .is_err());
    let mut second = slot.clone();
    second["candidate_index"] = json!(1);
    assert!(authority
        .reserve_native_candidate("task", "slot-2", &task, &second)
        .is_err());
    assert!(authority
        .reserve_native_candidate("task", "another-slot-for-same-candidate", &task, &slot)
        .is_err());
    let mut changed = task.clone();
    changed["budget"] = json!(2);
    assert!(authority
        .reserve_native_candidate("task", "slot", &changed, &slot)
        .is_err());
    assert_eq!(
        authority
            .db
            .query_row("SELECT COUNT(*) FROM native_slots", [], |r| r
                .get::<_, u64>(0))
            .unwrap(),
        1
    );
    drop(authority);
    assert!(Authority::open(&root, &witness).is_ok());
}
