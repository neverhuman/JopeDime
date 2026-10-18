//! W8-T03: dependency-first conditional publication. Two writers racing the same predecessor get
//! one winner and one explicit conflict; a missing or corrupt dependency prevents the pointer
//! from moving and prevents reuse; the pointer only ever moves from its expected predecessor.

use super::fixture::{
    append, commit, env, head, head_bytes, producer, published, run, snapshot, training, Fault,
    Faulty, Run, BIG, PUBLISHER,
};
use crate::hwm::Transition;
use crate::mirror::Mirror;
use crate::progress::inventory::*;
use crate::progress::outbox::Outbox;
use crate::progress::publish::{advance, prepare, publish, reusable, Plan, Prepared, Publication};
use crate::progress::remote::Remote;
use crate::state::State;
use std::fs;

fn ready(p: Prepared) -> Plan {
    match p {
        Prepared::Ready(plan) => plan,
        Prepared::Done(done) => panic!("expected a plan, got {done:?}"),
    }
}

fn race(remote: &dyn Remote) {
    let dir = tempfile::tempdir().unwrap();
    let r = run("run-w8-race", 3, 3);
    let m = r.manifest.encoded().unwrap();
    let mut a = Outbox::open(&State::at(&dir.path().join("host-a")), BIG).unwrap();
    let mut b = Outbox::open(&State::at(&dir.path().join("host-b")), BIG).unwrap();
    let ea = r.scheduled(0).encoded().unwrap();
    let eb = r.scheduled(1).encoded().unwrap();
    for (o, e) in [(&mut a, &ea), (&mut b, &eb)] {
        o.append_object(&m).unwrap();
        o.append_object(e).unwrap();
    }
    let ca = commit(&r, None, &[&ea]);
    let cb = commit(&r, None, &[&eb]);

    // Both writers finish every step before the pointer against the same (absent) predecessor.
    let pa = ready(prepare(&mut a, remote, PUBLISHER, &r.manifest, &ca).unwrap());
    let pb = ready(prepare(&mut b, remote, PUBLISHER, &r.manifest, &cb).unwrap());
    assert_eq!(
        advance(remote, pa).unwrap(),
        Publication::Advanced {
            commit: ca.digest.clone(),
            height: 0,
            from: None
        }
    );
    assert_eq!(
        advance(remote, pb).unwrap(),
        Publication::Conflict {
            commit: cb.digest.clone(),
            expected: None,
            found: Some(ca.digest.clone())
        }
    );
    assert_eq!(head(remote, &r).unwrap().commit, ca.digest);
    // The losing commit is preserved for reconciliation, not overwritten or deleted.
    assert_eq!(remote.get(&cb.key()).unwrap().unwrap(), cb.bytes);

    // The same race one step later, between two commits on the same parent.
    let e2 = r.started(0, 1).encoded().unwrap();
    let e3 = r.scheduled(2).encoded().unwrap();
    a.append_object(&e2).unwrap();
    a.append_object(&e3).unwrap();
    let x = commit(&r, Some(&ca), &[&e2]);
    let y = commit(&r, Some(&ca), &[&e3]);
    let px = ready(prepare(&mut a, remote, PUBLISHER, &r.manifest, &x).unwrap());
    let py = ready(prepare(&mut a, remote, PUBLISHER, &r.manifest, &y).unwrap());
    assert!(matches!(
        advance(remote, py).unwrap(),
        Publication::Advanced { height: 1, .. }
    ));
    assert_eq!(
        advance(remote, px).unwrap(),
        Publication::Conflict {
            commit: x.digest.clone(),
            expected: Some(ca.digest.clone()),
            found: Some(y.digest.clone())
        }
    );

    // A conflict does not wedge the winner, and a retry of a published commit is idempotent.
    let z = commit(&r, Some(&y), &[&e2]);
    assert!(matches!(
        publish(&mut a, remote, PUBLISHER, &r.manifest, &z).unwrap(),
        Publication::Advanced { height: 2, .. }
    ));
    assert_eq!(
        publish(&mut a, remote, PUBLISHER, &r.manifest, &z).unwrap(),
        Publication::AlreadyPublished {
            commit: z.digest.clone()
        }
    );
    // The stale loser retried later is still a conflict, and the pointer stays put.
    assert_eq!(
        publish(&mut b, remote, PUBLISHER, &r.manifest, &cb).unwrap(),
        Publication::Conflict {
            commit: cb.digest.clone(),
            expected: None,
            found: Some(z.digest.clone())
        }
    );
    assert_eq!(head(remote, &r).unwrap().commit, z.digest);
    assert_eq!(head(remote, &r).unwrap().height, 2);
}

#[test]
fn racing_publishers_at_one_predecessor_get_one_winner_and_one_conflict() {
    // The in-memory backend exercises the server-side version precondition.
    race(&Mirror::open("memory:///").unwrap());
    // The file backend exercises the local lock-and-revalidate path.
    let dir = tempfile::tempdir().unwrap();
    race(&Mirror::open(dir.path().to_str().unwrap()).unwrap());
}

fn blocked(p: Publication) -> String {
    match p {
        Publication::Blocked { reason, .. } => reason,
        other => panic!("expected Blocked, got {other:?}"),
    }
}

#[test]
fn a_missing_or_corrupt_dependency_blocks_the_pointer() {
    let r = run("run-w8-deps", 3, 3);
    let items = producer(&r);

    // Another writer's partial upload already sits under a dependency's immutable key.
    let e = env();
    let result_key = items[3].key();
    let partial_path = e.remote_dir.join(&result_key);
    fs::create_dir_all(partial_path.parent().unwrap()).unwrap();
    fs::write(&partial_path, b"partial").unwrap();
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let objects = append(&mut o, &items[..5]);
    let c0 = commit(&r, None, &objects[1..].iter().collect::<Vec<_>>());
    let reason = blocked(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c0).unwrap());
    assert!(reason.contains("different bytes"), "{reason}");
    assert_eq!(head_bytes(&e.remote, &r), None, "the pointer moved");
    assert_eq!(e.remote.get(&result_key).unwrap().unwrap(), b"partial");
    assert!(
        e.remote.get(&c0.key()).unwrap().is_none(),
        "a dependent went up first"
    );

    // Positive control on a clean remote.
    let e = env();
    let (mut o, _, c0) = published(&e, &r);
    let before = head_bytes(&e.remote, &r).unwrap();
    items[5].append(&mut o).unwrap();
    let e5 = items[5].encoded().clone();
    let c1 = commit(&r, Some(&c0), &[&e5]);

    // Lost remotely after delivery: re-uploaded create-only from the verified local payload.
    o.deliver(&e.remote, None).unwrap();
    fs::remove_file(e.remote_dir.join(e5.key())).unwrap();
    let plan = ready(prepare(&mut o, &e.remote, PUBLISHER, &r.manifest, &c1).unwrap());
    assert_eq!(plan.reuploaded, vec![e5.key()]);
    assert_eq!(
        head_bytes(&e.remote, &r).unwrap(),
        before,
        "prepare wrote the pointer"
    );
    drop(plan);

    // Corrupted remotely: never overwritten, and the pointer does not move.
    fs::write(e.remote_dir.join(e5.key()), b"{}").unwrap();
    let reason = blocked(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c1).unwrap());
    assert!(reason.contains("Corrupt"), "{reason}");
    assert_eq!(head_bytes(&e.remote, &r).unwrap(), before);
    assert_eq!(fs::read(e.remote_dir.join(e5.key())).unwrap(), b"{}");

    // Lost remotely and locally: nothing can back it, so the pointer does not move.
    let e = env();
    let (mut o, objects, c0) = published(&e, &r);
    let before = head_bytes(&e.remote, &r).unwrap();
    let blob = BlobRef::of(&Run::result(0, 1));
    fs::remove_file(e.remote_dir.join(blob_key(&blob.sha256))).unwrap();
    fs::remove_file(e.state.blob_path(&blob.sha256)).unwrap();
    let c1 = commit(&r, Some(&c0), &[&objects[1]]);
    let reason = blocked(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c1).unwrap());
    assert!(reason.contains("Missing"), "{reason}");
    assert_eq!(head_bytes(&e.remote, &r).unwrap(), before);
}

#[test]
fn graph_rules_are_checked_before_the_pointer_moves() {
    let r = run("run-w8-rules", 3, 3);
    let e = env();
    let (mut o, objects, c0) = published(&e, &r);
    let before = head_bytes(&e.remote, &r).unwrap();

    // A height that does not follow its parent.
    let mut skip = decode::<Commit>(&commit(&r, Some(&c0), &[&objects[1]]).bytes)
        .unwrap()
        .into_value();
    skip.height = 5;
    let skip = encode(&skip).unwrap();

    // A decision drawn from an attempt that did not complete.
    let failed = r.failed(1, 1, FailureClass::Timeout, true);
    let scheduled1 = r.scheduled(1).encoded().unwrap();
    o.append_object(&scheduled1).unwrap();
    o.append_object(&failed.encoded().unwrap()).unwrap();
    let bad_decision = r.decision(&failed, Transition::Improved, "01K62AAAAAAAAAAAAAAAAAAAA9");
    o.append_blob(
        format!(
            "record {} {}",
            failed.value().work_id,
            failed.value().attempt
        )
        .as_bytes(),
    )
    .unwrap();
    let bad_decision = bad_decision.encoded().unwrap();
    o.append_object(&bad_decision).unwrap();
    let decided = commit(&r, Some(&c0), &[&bad_decision]);

    // A checkpoint that skips an update.
    let (corpus, exposure, _) = training(&mut o, &r, 0, None, 3);
    let mut gap = Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: r.id().into(),
        update_index: 1,
        parent: None,
        exposure: exposure.digest.clone(),
        corpus: corpus.digest.clone(),
        state: vec![BlobRef::of(b"state-gap")],
    };
    o.append_blob(b"state-gap").unwrap();
    let gap_checkpoint = encode(&gap).unwrap();
    o.append_object(&gap_checkpoint).unwrap();
    let gapped = commit(&r, Some(&c0), &[&gap_checkpoint]);
    gap.update_index = 0;

    for (label, c) in [
        ("height", skip),
        ("decision", decided),
        ("checkpoint", gapped),
    ] {
        let reason = blocked(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c).unwrap());
        assert!(reason.contains("Invalid"), "{label}: {reason}");
        assert_eq!(head_bytes(&e.remote, &r).unwrap(), before, "{label}");
    }

    // A commit naming something the outbox never recorded is refused before any remote write.
    let stranger = r.scheduled(2).encoded().unwrap();
    let c = commit(&r, Some(&c0), &[&stranger]);
    let remote_before = snapshot(&e.remote_dir);
    assert!(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c).is_err());
    assert_eq!(snapshot(&e.remote_dir), remote_before);

    // Positive control: a contiguous checkpoint in a valid commit.
    o.append_blob(b"state-0").unwrap();
    let good = encode(&Checkpoint {
        state: vec![BlobRef::of(b"state-0")],
        ..gap
    })
    .unwrap();
    o.append_object(&good).unwrap();
    let c = commit(&r, Some(&c0), &[&good]);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c).unwrap(),
        Publication::Advanced { height: 1, .. }
    ));
}

#[test]
fn the_pointer_moves_only_from_its_expected_predecessor() {
    let r = run("run-w8-pointer", 3, 3);
    let e = env();
    let (mut o, objects, c0) = published(&e, &r);
    let key = head_key(r.id()).unwrap();
    let head_path = e.remote_dir.join(&key);

    // Not this run's publisher, or not this run's commit: refused before anything is written.
    let c1 = commit(&r, Some(&c0), &[&objects[1]]);
    let local = snapshot(o.root());
    let remote_before = snapshot(&e.remote_dir);
    assert!(publish(&mut o, &e.remote, "xbabe3", &r.manifest, &c1).is_err());
    let other = run("run-w8-other", 3, 3);
    assert!(publish(&mut o, &e.remote, PUBLISHER, &other.manifest, &c1).is_err());
    assert_eq!(snapshot(o.root()), local);
    assert_eq!(snapshot(&e.remote_dir), remote_before);

    // A pointer write that fails outright: an error, one attempt, no fallback, nothing moved.
    let before = head_bytes(&e.remote, &r).unwrap();
    let faulty = Faulty::new(&e.remote).fault_pointer(0, Fault::Disconnect);
    let err = publish(&mut o, &faulty, PUBLISHER, &r.manifest, &c1).unwrap_err();
    assert!(
        format!("{err:#}").contains("pointer is unchanged"),
        "{err:#}"
    );
    assert_eq!(faulty.keys("pointer"), vec![key.clone()]);
    assert!(
        !faulty.keys("put").contains(&key),
        "an unconditional write reached the pointer"
    );
    assert_eq!(head_bytes(&e.remote, &r).unwrap(), before);

    // A pointer write whose response is lost: the pointer itself says it moved.
    let faulty = Faulty::new(&e.remote).fault_pointer(0, Fault::ResponseLost);
    assert!(matches!(
        publish(&mut o, &faulty, PUBLISHER, &r.manifest, &c1).unwrap(),
        Publication::Advanced { height: 1, .. }
    ));
    assert_eq!(faulty.keys("pointer").len(), 1);
    assert_eq!(head(&e.remote, &r).unwrap().commit, c1.digest);

    // A stale parent: the pointer has moved on, so this is a conflict and nothing is written.
    let stale = commit(&r, Some(&c0), &[&objects[2]]);
    let before = head_bytes(&e.remote, &r).unwrap();
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &stale).unwrap(),
        Publication::Conflict { .. }
    ));
    assert_eq!(head_bytes(&e.remote, &r).unwrap(), before);

    // A pointer that is garbage, belongs to another run or misstates its height is left alone.
    let c2 = commit(&r, Some(&c1), &[&objects[2]]);
    let good_head = fs::read(&head_path).unwrap();
    let mut wrong_height = Head::decode(&good_head).unwrap();
    wrong_height.height = 7;
    let mut wrong_run = Head::decode(&good_head).unwrap();
    wrong_run.run_id = "run-w8-other".into();
    for (label, bytes) in [
        ("garbage", b"not a pointer".to_vec()),
        ("other run", wrong_run.encode().unwrap()),
        ("wrong height", wrong_height.encode().unwrap()),
    ] {
        fs::write(&head_path, &bytes).unwrap();
        blocked(publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c2).unwrap());
        assert_eq!(
            fs::read(&head_path).unwrap(),
            bytes,
            "{label} was overwritten"
        );
    }
    fs::write(&head_path, &good_head).unwrap();
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c2).unwrap(),
        Publication::Advanced { height: 2, .. }
    ));
}

#[test]
fn only_a_verified_closure_is_reusable() {
    let r = run("run-w8-reuse", 3, 3);
    let e = env();
    assert!(reusable(&e.remote, r.id()).unwrap().is_none());
    let (mut o, _, c0) = published(&e, &r);

    let (corpus, exposure0, members) = training(&mut o, &r, 0, None, 3);
    o.append_blob(b"parameters-0").unwrap();
    let ck0 = encode(&Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: r.id().into(),
        update_index: 0,
        parent: None,
        exposure: exposure0.digest.clone(),
        corpus: corpus.digest.clone(),
        state: vec![BlobRef::of(b"parameters-0")],
    })
    .unwrap();
    o.append_object(&ck0).unwrap();
    // Consuming the same corpus again is a second exposure, not a deduplicated one.
    let exposure1 = encode(&Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: r.id().into(),
        update_index: 1,
        corpus: corpus.digest.clone(),
        checkpoint_before: Some(ck0.digest.clone()),
        consumed_rows: 3,
    })
    .unwrap();
    o.append_object(&exposure1).unwrap();
    o.append_blob(b"parameters-1").unwrap();
    let ck1 = encode(&Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: r.id().into(),
        update_index: 1,
        parent: Some(ck0.digest.clone()),
        exposure: exposure1.digest.clone(),
        corpus: corpus.digest.clone(),
        state: vec![BlobRef::of(b"parameters-1")],
    })
    .unwrap();
    o.append_object(&ck1).unwrap();
    let c1 = commit(&r, Some(&c0), &[&corpus, &exposure0, &ck0]);
    let c2 = commit(&r, Some(&c1), &[&exposure1, &ck1]);
    for c in [&c1, &c2] {
        assert!(matches!(
            publish(&mut o, &e.remote, PUBLISHER, &r.manifest, c).unwrap(),
            Publication::Advanced { .. }
        ));
    }
    let state = reusable(&e.remote, r.id()).unwrap().unwrap();
    assert_eq!(
        (state.head.commit.clone(), state.head.height),
        (c2.digest.clone(), 2)
    );
    assert_eq!(state.graph.corpora(), vec![corpus.digest.clone()]);
    let mut checkpoints = vec![ck0.digest.clone(), ck1.digest.clone()];
    checkpoints.sort();
    assert_eq!(state.graph.checkpoints(), checkpoints);
    assert_eq!(
        state.graph.all::<Exposure>().len(),
        2,
        "both exposures stay visible"
    );
    assert!(state.graph.reads > 0);

    // Anything missing or corrupt anywhere in the closure: nothing is reusable.
    let damage: Vec<(&str, String, Option<Vec<u8>>)> = vec![
        (
            "checkpoint state corrupt",
            blob_key(&BlobRef::of(b"parameters-1").sha256),
            Some(b"parameters-X".to_vec()),
        ),
        (
            "corpus member lost",
            blob_key(&BlobRef::of(&members[0]).sha256),
            None,
        ),
        ("parent checkpoint lost", ck0.key(), None),
        ("first commit lost", c0.key(), None),
    ];
    for (label, key, replacement) in damage {
        let path = e.remote_dir.join(&key);
        let original = fs::read(&path).unwrap();
        match &replacement {
            Some(bytes) => fs::write(&path, bytes).unwrap(),
            None => fs::remove_file(&path).unwrap(),
        }
        assert!(
            reusable(&e.remote, r.id()).is_err(),
            "{label} still reusable"
        );
        fs::write(&path, original).unwrap();
        assert!(reusable(&e.remote, r.id()).is_ok(), "{label}: restore");
    }

    // A pointer naming a commit the remote does not hold is stale, not reusable.
    let path = e.remote_dir.join(head_key(r.id()).unwrap());
    let good = fs::read(&path).unwrap();
    let mut dangling = Head::decode(&good).unwrap();
    dangling.commit = "0".repeat(64);
    fs::write(&path, dangling.encode().unwrap()).unwrap();
    assert!(reusable(&e.remote, r.id()).is_err());
    fs::write(&path, good).unwrap();
}
