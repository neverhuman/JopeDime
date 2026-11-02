use super::*;

#[test]
fn logical_work_identity_is_separate_from_attempt_and_delivery_identity() {
    let r = run("run-w8-ids", 3, 3);
    // The logical identity is a function of the run and the specification only.
    assert_eq!(spec(1).work_id("run-w8-ids").unwrap(), r.work[1]);
    assert_ne!(spec(1).work_id("run-w8-other").unwrap(), r.work[1]);
    let mut changed = spec(1);
    changed.inputs = vec![sha256_hex(b"another input")];
    assert_ne!(changed.work_id("run-w8-ids").unwrap(), r.work[1]);
    let mut changed = spec(1);
    changed.stage = "teacher-admission".into();
    assert_ne!(changed.work_id("run-w8-ids").unwrap(), r.work[1]);

    // Two attempts at one unit: same logical identity, different attempt and delivery identity.
    let f1 = r.failed(1, 1, FailureClass::Timeout, true);
    let a1 = r.completed(1, 1);
    let a2 = r.completed(1, 2);
    assert_eq!(a1.value().work_id, a2.value().work_id);
    assert_ne!(a1.value().attempt, a2.value().attempt);
    assert_ne!(a1.digest(), a2.digest());
    // The same bytes are the same delivery.
    assert_eq!(r.completed(1, 2).digest(), a2.digest());
    // A retry depends on the exact failure it retries, and on the manifest.
    let retry = r.retry(1, 2, &f1).encoded().unwrap();
    assert!(retry.dependencies.contains(&Dependency::Object(ObjectRef {
        kind: Kind::Event,
        digest: f1.digest().into()
    })));
    assert!(retry.dependencies.contains(&Dependency::Object(ObjectRef {
        kind: Kind::RunManifest,
        digest: r.manifest.digest().into()
    })));

    // The same logical work cannot be planned twice, and a unit must carry its own work id.
    let dup = RunManifest::new(
        "run-w8-ids",
        PUBLISHER,
        SOURCE_COMMIT,
        3,
        vec![spec(1), spec(1)],
    )
    .unwrap();
    assert!(encode(&dup).is_err());
    let mut forged = r.manifest.value().clone();
    forged.units[0].work_id = "0".repeat(64);
    assert!(
        encode(&forged).is_err(),
        "a work id its specification does not produce"
    );
    let mut unsorted = r.manifest.value().clone();
    unsorted.units.reverse();
    assert!(encode(&unsorted).is_err());
}
