//! W8-T04: derived views rebuild from committed records. Deleting a disposable derived index,
//! or its summary object, and rebuilding gives the same view. Stale, corrupt or dangling state
//! is a visible failure. Retries and lost responses never duplicate committed events. No source
//! evidence is deleted by any of this.

use super::fixture::{
    append, commit, env, producer, published, run, snapshot, training, Env, Fault, Faulty, Run,
    PUBLISHER,
};
use crate::progress::inventory::*;
use crate::progress::outbox::Outbox;
use crate::progress::publish::{publish, reusable, Publication};
use crate::progress::rebuild::{
    index_key, publish_summary, read_summary, reconstruct, summarize, Freshness,
};
use crate::progress::remote::Remote;
use std::collections::BTreeMap;
use std::fs;

/// Two published commits:
/// - c0: unit 0 completes;
/// - c1: unit 1 times out, is retried and completes, with compact candidate evidence. It also
///   holds a corpus of 3 teachers consumed twice (updates 0 and 1) and the two checkpoints those
///   updates committed.
fn rich(e: &Env, r: &Run, remote: &dyn Remote) -> (Outbox, Encoded, Encoded) {
    let (mut o, _, c0) = published(e, r);
    let (c1, ck1) = rich_commit(&mut o, r, &c0);
    let p = publish(&mut o, remote, PUBLISHER, &r.manifest, &c1).unwrap();
    assert!(
        matches!(p, Publication::Advanced { height: 1, .. }),
        "{p:?}"
    );
    (o, c1, ck1)
}

/// Append everything c1 adds on top of `c0`; returns (c1, the latest checkpoint). Unpublished.
fn rich_commit(o: &mut Outbox, r: &Run, c0: &Encoded) -> (Encoded, Encoded) {
    let items = producer(r);
    let mut objects = append(o, &items[5..]);
    let completion = items[9].encoded().clone();
    o.append_blob(b"record of unit 1").unwrap();
    let evidence = encode(&CandidateEvidence {
        schema: Kind::CandidateEvidence.schema().into(),
        run_id: r.id().into(),
        work_id: r.work[1].clone(),
        attempt: 2,
        completion: completion.digest.clone(),
        record: BlobRef::of(b"record of unit 1"),
        retention: Retention::Compact,
        artifacts: vec![],
    })
    .unwrap();
    o.append_object(&evidence).unwrap();
    let (corpus, exposure0, _) = training(o, r, 0, None, 3);
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
    objects.extend([evidence, corpus, exposure0, ck0, exposure1, ck1.clone()]);
    let c1 = commit(r, Some(c0), &objects.iter().collect::<Vec<_>>());
    (c1, ck1)
}

/// Everything under the remote except the derived index and summary objects: the evidence the
/// rebuild must never touch.
fn evidence(e: &Env) -> BTreeMap<String, String> {
    snapshot(&e.remote_dir)
        .into_iter()
        .filter(|(k, _)| {
            !k.starts_with("progress/objects/summary/") && !k.ends_with("summary.json")
        })
        .collect()
}

#[test]
fn a_deleted_derived_index_rebuilds_to_the_same_view() {
    let e = env();
    let r = run("run-w8-rebuild", 3, 3);
    let (_o, c1, ck1) = rich(&e, &r, &e.remote);
    let source = evidence(&e);

    let summary = publish_summary(&e.remote, r.id()).unwrap().unwrap();
    assert_eq!(
        (summary.derived_from.clone(), summary.height),
        (c1.digest.clone(), 1)
    );
    let k = &summary.counts;
    assert_eq!(
        (k.planned, k.completed, k.in_flight, k.unscheduled),
        (3, 2, 0, 1)
    );
    assert_eq!((k.attempts, k.retries, k.failures["timeout"]), (3, 1, 1));
    assert_eq!(
        (k.coverage.numerator, k.coverage.denominator),
        (2, 3),
        "coverage reports the unscheduled unit rather than hiding it"
    );
    assert_eq!(
        (
            summary.candidate_yield.numerator,
            summary.candidate_yield.denominator
        ),
        (1, 2)
    );
    assert_eq!(
        (summary.exposures, summary.consumed_rows, summary.corpora),
        (2, 6, 1),
        "repeated consumption of one corpus is two exposures"
    );
    assert_eq!((summary.unique_teachers, summary.checkpoints), (3, 2));
    assert_eq!(
        summary.latest_checkpoint.as_deref(),
        Some(ck1.digest.as_str())
    );
    let index = index_key(r.id()).unwrap();
    let index_bytes = e.remote.get(&index).unwrap().unwrap();
    let object = encode(&summary).unwrap().key();
    let object_bytes = e.remote.get(&object).unwrap().unwrap();
    assert_eq!(
        read_summary(&e.remote, r.id()).unwrap(),
        (Freshness::Fresh, Some(summary.clone()))
    );

    // Delete only the disposable index: the view is missing, then rebuilds identically.
    fs::remove_file(e.remote_dir.join(&index)).unwrap();
    assert_eq!(
        read_summary(&e.remote, r.id()).unwrap(),
        (Freshness::Missing, None)
    );
    assert_eq!(
        publish_summary(&e.remote, r.id()).unwrap().unwrap(),
        summary
    );
    assert_eq!(e.remote.get(&index).unwrap().unwrap(), index_bytes);

    // Delete the summary object as well: rebuilt to the same bytes.
    fs::remove_file(e.remote_dir.join(&index)).unwrap();
    fs::remove_file(e.remote_dir.join(&object)).unwrap();
    assert_eq!(
        publish_summary(&e.remote, r.id()).unwrap().unwrap(),
        summary
    );
    assert_eq!(e.remote.get(&object).unwrap().unwrap(), object_bytes);

    // An index whose summary object is gone is a visible failure, not a view, until rebuilt.
    fs::remove_file(e.remote_dir.join(&object)).unwrap();
    assert!(read_summary(&e.remote, r.id()).is_err());
    publish_summary(&e.remote, r.id()).unwrap();
    assert_eq!(read_summary(&e.remote, r.id()).unwrap().0, Freshness::Fresh);

    // The summary is a pure function of the committed closure.
    let (again, _) = summarize(&reusable(&e.remote, r.id()).unwrap().unwrap()).unwrap();
    assert_eq!(again, summary);
    assert_eq!(evidence(&e), source, "a rebuild touched source evidence");
}

#[test]
fn stale_corrupt_or_dangling_state_is_a_visible_failure() {
    let e = env();
    let r = run("run-w8-stale", 3, 3);
    let (mut o, c1, _) = rich(&e, &r, &e.remote);
    let first = publish_summary(&e.remote, r.id()).unwrap().unwrap();

    // The pointer moves on: the old view reports itself stale until rebuilt.
    let scheduled = r.scheduled(2).encoded().unwrap();
    o.append_object(&scheduled).unwrap();
    let c2 = commit(&r, Some(&c1), &[&scheduled]);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c2).unwrap(),
        Publication::Advanced { height: 2, .. }
    ));
    assert_eq!(
        read_summary(&e.remote, r.id()).unwrap(),
        (
            Freshness::Stale {
                derived_from: c1.digest.clone(),
                derived_height: 1,
                head: Some(c2.digest.clone()),
                head_height: Some(2),
            },
            Some(first.clone())
        )
    );
    let second = publish_summary(&e.remote, r.id()).unwrap().unwrap();
    assert_eq!(second.counts.in_flight, 1, "unit 2 is now scheduled");
    assert_eq!(read_summary(&e.remote, r.id()).unwrap().0, Freshness::Fresh);

    // A corrupt or inconsistent index is refused, not shown.
    let index = e.remote_dir.join(index_key(r.id()).unwrap());
    let good = fs::read(&index).unwrap();
    let mut other = serde_json::from_slice::<serde_json::Value>(&good).unwrap();
    other["height"] = 7.into();
    for (label, bytes) in [
        ("garbage", b"not an index".to_vec()),
        ("inconsistent", serde_json::to_vec(&other).unwrap()),
        (
            "pretty",
            serde_json::to_vec_pretty(&serde_json::from_slice::<serde_json::Value>(&good).unwrap())
                .unwrap(),
        ),
    ] {
        fs::write(&index, &bytes).unwrap();
        assert!(
            read_summary(&e.remote, r.id()).is_err(),
            "{label} index was read"
        );
    }
    // Rebuilding replaces the derived index conditionally; nothing else changes.
    let source = evidence(&e);
    publish_summary(&e.remote, r.id()).unwrap();
    assert_eq!(fs::read(&index).unwrap(), good);
    assert_eq!(evidence(&e), source);

    // A dangling pointer: nothing can be rebuilt or reconstructed, and the view says stale.
    let head_path = e.remote_dir.join(head_key(r.id()).unwrap());
    let head = fs::read(&head_path).unwrap();
    let mut dangling = Head::decode(&head).unwrap();
    dangling.commit = "0".repeat(64);
    fs::write(&head_path, dangling.encode().unwrap()).unwrap();
    assert!(publish_summary(&e.remote, r.id()).is_err());
    assert!(reconstruct(&e.remote, r.id(), None).is_err());
    assert_eq!(
        fs::read(&index).unwrap(),
        good,
        "a failed rebuild moved the index"
    );
    assert!(matches!(
        read_summary(&e.remote, r.id()).unwrap().0,
        Freshness::Stale { .. }
    ));
    fs::write(&head_path, &head).unwrap();

    // A divergent branch is visible in the reconstruction and absent from the view.
    let loser = commit(&r, Some(&c1), &[&r.started(2, 1).encoded().unwrap()]);
    o.append_object(&r.started(2, 1).encoded().unwrap())
        .unwrap();
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &loser).unwrap(),
        Publication::Conflict { .. }
    ));
    let rec = reconstruct(&e.remote, r.id(), Some(&o)).unwrap();
    assert!(rec.divergent.contains(&loser.digest));
    assert_eq!(publish_summary(&e.remote, r.id()).unwrap().unwrap(), second);
}

#[test]
fn retries_and_lost_responses_never_duplicate_committed_events() {
    // A clean run and a run where every third write loses its response, and the pointer write
    // loses its response too. Retrying until published gives byte-identical committed state.
    let clean = env();
    let r = run("run-w8-retry", 3, 3);
    rich(&clean, &r, &clean.remote);
    let clean_summary = publish_summary(&clean.remote, r.id()).unwrap().unwrap();

    let e = env();
    let (mut o, _, c0) = published(&e, &r);
    let (c1, _) = rich_commit(&mut o, &r, &c0);
    let mut flaky = Faulty::new(&e.remote);
    for call in (0..60).step_by(3) {
        flaky = flaky.fault_put(call, Fault::ResponseLost);
    }
    let flaky = flaky.fault_pointer(0, Fault::ResponseLost);
    // The same c1 as the clean run, published through the flaky store by plain retries.
    let mut attempts = 0;
    loop {
        attempts += 1;
        assert!(attempts < 40, "publication never converged");
        match publish(&mut o, &flaky, PUBLISHER, &r.manifest, &c1) {
            Ok(Publication::Advanced { .. } | Publication::AlreadyPublished { .. }) => break,
            Ok(Publication::Blocked { .. }) | Err(_) => continue,
            Ok(other) => panic!("{other:?}"),
        }
    }
    assert!(attempts > 1, "no fault was exercised");
    let summary = publish_summary(&e.remote, r.id()).unwrap().unwrap();
    assert_eq!(summary, clean_summary, "retries changed the committed view");
    assert_eq!(summary.counts.duplicate_deliveries, 0);
    assert_eq!(
        snapshot(&e.remote_dir)
            .into_keys()
            .filter(|k| k.starts_with("progress/objects/event/"))
            .count(),
        snapshot(&clean.remote_dir)
            .into_keys()
            .filter(|k| k.starts_with("progress/objects/event/"))
            .count(),
        "a retried delivery created a second event object"
    );
    let lost = flaky
        .log
        .borrow()
        .iter()
        .filter(|(_, _, outcome)| outcome == "applied-response-lost")
        .count();
    assert!(lost >= 2, "{lost} lost responses exercised");
    eprintln!("converged after {attempts} publication attempts and {lost} lost responses");
}
