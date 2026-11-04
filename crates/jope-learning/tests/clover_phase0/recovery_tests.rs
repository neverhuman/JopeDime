use super::*;

fn recovery(id: &ExperimentIdentity, generation: u64) -> Vec<u8> {
    canonical_json(
        &json!({"schema":"clover.recovery/v1","experiment_sha256":id.sha256().unwrap(),
        "hold_generation":generation,"guard_sha256":id.guards["hold_recovery"]}),
    )
    .unwrap()
}

#[test]
fn hold_persists_and_explicit_recovery_is_generation_and_identity_bound() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let id = identity();
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let run = RunStore::create(&path, "gp", id.clone()).unwrap();
    run.hold("quantile guard").unwrap();
    drop(run);
    let mut run = RunStore::open(&path, "gp", id.clone()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("a").is_err());
    assert_eq!(run.recover().unwrap(), 0);
    assert!(run.require_ready().is_err());
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.hold("another violation").unwrap();
    assert!(run.release_hold(generation, &manifest, &evidence).is_err());
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.release_hold(generation, &manifest, &evidence).unwrap();
    run.require_ready().unwrap();
    drop(run);
    RunStore::open(&path, "gp", id)
        .unwrap()
        .require_ready()
        .unwrap();
}

#[test]
fn interrupted_attempts_terminalize_unknown_without_reward_or_budget_refund() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let id = identity();
    let mut run = RunStore::create(&path, "gp", id.clone()).unwrap();
    run.start_attempt("lost-worker").unwrap();
    drop(run);
    let mut run = RunStore::open(&path, "gp", id.clone()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("before-reconcile").is_err());
    assert_eq!(run.recover().unwrap(), 1);
    assert_eq!(run.recover().unwrap(), 0);
    assert!(run.require_ready().is_err());
    assert!(run
        .finish_attempt("lost-worker", AttemptOutcome::Qualified, Some(1.))
        .is_err());
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.release_hold(generation, &manifest, &evidence).unwrap();
    run.start_attempt("second").unwrap();
    assert!(run
        .finish_attempt("second", AttemptOutcome::Unknown, Some(0.))
        .is_err());
    run.finish_attempt("second", AttemptOutcome::Unknown, None)
        .unwrap();
    assert!(run.start_attempt("third").is_err());
    let db = rusqlite::Connection::open(path).unwrap();
    let unknown: u64 = db
        .query_row(
            "SELECT count(*) FROM attempts WHERE outcome='\"unknown\"' AND reward IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unknown, 2);
}

#[test]
fn every_publication_boundary_is_manifest_last_and_retry_is_immutable() {
    let id = identity();
    let payloads = vec![b"attempt failed".to_vec(), b"attempt unknown".to_vec()];
    for boundary in 0..4 {
        let tmp = scratch();
        let store = EvidenceStore::open(tmp.path()).unwrap();
        let mut writes = Vec::new();
        assert!(store
            .publish(&id, &payloads, |i, h| {
                writes.push(h.to_owned());
                if i == boundary {
                    bail!("injected crash/lost response");
                }
                Ok(())
            })
            .is_err());
        let before = writes
            .iter()
            .map(|h| (h.clone(), store.read(h).unwrap()))
            .collect::<Vec<_>>();
        let mut retry = Vec::new();
        let manifest = store
            .publish(&id, &payloads, |_, h| {
                retry.push(h.to_owned());
                Ok(())
            })
            .unwrap();
        assert_eq!(retry.last(), Some(&manifest));
        for (h, bytes) in before {
            assert_eq!(store.read(&h).unwrap(), bytes);
        }
        store.verify_manifest(&manifest, &id).unwrap();
        if boundary < 3 {
            assert!(!writes.contains(&manifest));
        }
    }
}

#[test]
fn publication_fault_persists_hold_across_controller_restart() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let run = RunStore::create(&path, "gp", identity()).unwrap();
    assert!(run
        .publish_evidence(&evidence, &[b"result".to_vec()], |_, _| bail!(
            "lost object response"
        ))
        .is_err());
    drop(run);
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("after-publish-fault").is_err());
    assert!(run
        .publish_evidence(&evidence, &[b"result".to_vec()], |_, _| Ok(()))
        .is_err());
}

#[test]
fn corrupt_missing_or_foreign_evidence_never_releases_hold() {
    let tmp = scratch();
    let id = identity();
    let store = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let mut run = RunStore::create(&tmp.path().join("state.sqlite"), "gp", id.clone()).unwrap();
    run.hold("fault").unwrap();
    let generation = run.hold_generation().unwrap();
    let manifest = store
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    let mut foreign = id.clone();
    foreign.seed = 1;
    assert!(store.verify_manifest(&manifest, &foreign).is_err());
    assert!(run
        .release_hold(generation, &digest("missing"), &store)
        .is_err());
    let object = sha256_hex(&recovery(&id, generation));
    fs::write(
        tmp.path().join("evidence/objects").join(&object),
        b"tampered",
    )
    .unwrap();
    assert!(store.put(&recovery(&id, generation)).is_err());
    assert!(run.release_hold(generation, &manifest, &store).is_err());
    assert!(run.require_ready().is_err());
}

#[test]
fn failed_recovery_transaction_keeps_unknown_work_fenced() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("interrupted").unwrap();
    drop(run);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("CREATE TRIGGER injected_failure BEFORE UPDATE ON attempts BEGIN SELECT RAISE(ABORT,'injected storage fault'); END;").unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.recover().is_err());
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("new").is_err());
    drop(run);
    db.execute_batch("DROP TRIGGER injected_failure;").unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert_eq!(run.recover().unwrap(), 1);
    assert!(run.require_ready().is_err());
}

#[test]
fn missing_payload_and_corrupted_earlier_readback_block_manifest_publication() {
    let tmp = scratch();
    let store = EvidenceStore::open(tmp.path()).unwrap();
    let mut written = Vec::new();
    assert!(store
        .publish(&identity(), &[b"attempt".to_vec()], |index, hash| {
            written.push(hash.to_owned());
            if index == 1 {
                fs::write(
                    tmp.path().join("objects").join(&written[0]),
                    b"corrupt earlier readback",
                )?;
            }
            Ok(())
        })
        .is_err());
    assert_eq!(
        written.len(),
        2,
        "manifest was published after failed closure read-back"
    );
    let tmp = scratch();
    let store = EvidenceStore::open(tmp.path()).unwrap();
    let missing = canonical_json(
        &json!({"schema":"clover.evidence/v1", "experiment_sha256": identity().sha256().unwrap(),
        "objects":[identity().sha256().unwrap(),digest("absent payload")]}),
    )
    .unwrap();
    let hash = store.put(&missing).unwrap();
    assert!(store.verify_manifest(&hash, &identity()).is_err());
}

#[test]
fn unrecognized_persisted_state_is_terminal_unknown_and_charged() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("alien-state").unwrap();
    drop(run);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "UPDATE attempts SET outcome='unrecognized-after-partial-recovery'",
        [],
    )
    .unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.require_ready().is_err());
    assert_eq!(run.recover().unwrap(), 1);
    assert!(run.require_ready().is_err());
    let (outcome, reward): (String, Option<f64>) = db
        .query_row("SELECT outcome,reward FROM attempts", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(outcome, "\"unknown\"");
    assert_eq!(reward, None);
}

#[test]
fn over_budget_publication_writes_nothing_and_latches_hold() {
    let tmp = scratch();
    let store = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let mut id = identity();
    id.budget.daily_bytes = 1;
    let run = RunStore::create(&tmp.path().join("state.sqlite"), "gp", id).unwrap();
    assert!(run
        .publish_evidence(&store, &[b"result".to_vec()], |_, _| Ok(()))
        .is_err());
    assert!(run.require_ready().is_err());
    assert_eq!(
        fs::read_dir(tmp.path().join("evidence/objects"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn missing_or_partial_registry_never_resets_hold_or_attempt_accounting() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    assert!(!path.exists());
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("charged").unwrap();
    run.hold("guard").unwrap();
    drop(run);
    assert!(RunStore::create(&path, "gp", identity()).is_err());
    let retained = tmp.path().join("retained.sqlite");
    fs::rename(&path, &retained).unwrap();
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    assert!(!path.exists());
    fs::rename(retained, &path).unwrap();
    RunStore::open(&path, "gp", identity())
        .unwrap()
        .require_ready()
        .unwrap_err();
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute("DELETE FROM holds", []).unwrap();
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    let count: u64 = db
        .query_row("SELECT count(*) FROM holds", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
