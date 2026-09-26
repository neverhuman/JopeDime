//! What the publication receiving path refuses before the run pointer moves, beyond the closure
//! checks of W8-T03:
//!
//! * a committed graph whose optimizer updates are not one consecutive chain: two exposures or
//!   two checkpoints for the same update index (a forked or repeated update);
//! * an HWM decision the publishing host's own HWM ledger does not hold.
//!
//! Every refusal leaves the pointer byte-identical; the positive controls advance it.

use super::fixture::{
    append, commit, env, head_bytes, hwm_record, producer, published, run, training, Env, Run, BIG,
    PUBLISHER,
};
use crate::hwm::{self, Candidate, Transition};
use crate::progress::inventory::*;
use crate::progress::outbox::Outbox;
use crate::progress::publish::{publish, Publication};
use crate::progress::rebuild::restore;
use crate::state::State;
use serde_json::json;

fn checkpoint(
    o: &mut Outbox,
    r: &Run,
    update: u64,
    parent: Option<&Encoded>,
    exposure: &Encoded,
    corpus: &Encoded,
    state: &[u8],
) -> Encoded {
    o.append_blob(state).unwrap();
    let ck = encode(&Checkpoint {
        schema: Kind::Checkpoint.schema().into(),
        run_id: r.id().into(),
        update_index: update,
        parent: parent.map(|p| p.digest.clone()),
        exposure: exposure.digest.clone(),
        corpus: corpus.digest.clone(),
        state: vec![BlobRef::of(state)],
    })
    .unwrap();
    o.append_object(&ck).unwrap();
    ck
}

fn exposure(
    o: &mut Outbox,
    r: &Run,
    update: u64,
    corpus: &Encoded,
    before: Option<&Encoded>,
    rows: u64,
) -> Encoded {
    let e = encode(&Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: r.id().into(),
        update_index: update,
        corpus: corpus.digest.clone(),
        checkpoint_before: before.map(|b| b.digest.clone()),
        consumed_rows: rows,
    })
    .unwrap();
    o.append_object(&e).unwrap();
    e
}

/// A run whose committed tip (height 1) holds update 0: a corpus, its exposure and checkpoint.
struct Trained {
    e: Env,
    r: Run,
    o: Outbox,
    c1: Encoded,
    corpus: Encoded,
    exposure0: Encoded,
    ck0: Encoded,
}

fn trained() -> Trained {
    let r = run("run-w8-updates", 3, 3);
    let e = env();
    let (mut o, _, c0) = published(&e, &r);
    let (corpus, exposure0, _) = training(&mut o, &r, 0, None, 3);
    let ck0 = checkpoint(&mut o, &r, 0, None, &exposure0, &corpus, b"parameters-0");
    let c1 = commit(&r, Some(&c0), &[&corpus, &exposure0, &ck0]);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c1).unwrap(),
        Publication::Advanced { height: 1, .. }
    ));
    Trained {
        e,
        r,
        o,
        c1,
        corpus,
        exposure0,
        ck0,
    }
}

#[test]
fn a_forked_or_repeated_update_index_never_reaches_the_pointer() {
    // Each case starts from its own committed update 0, so every defect is observed on its own.
    type Case = (&'static str, fn(&mut Trained) -> Encoded);
    let cases: [Case; 4] = [
        (
            "a second update 0 in a later commit (a fork from nothing)",
            |t| {
                let (corpus_b, exposure_b, _) = training(&mut t.o, &t.r, 0, None, 2);
                let ck = checkpoint(&mut t.o, &t.r, 0, None, &exposure_b, &corpus_b, b"fork-0");
                commit(&t.r, Some(&t.c1), &[&corpus_b, &exposure_b, &ck])
            },
        ),
        (
            "two updates 1 from the same checkpoint in one commit",
            |t| {
                let (corpus, ck0) = (t.corpus.clone(), t.ck0.clone());
                let ea = exposure(&mut t.o, &t.r, 1, &corpus, Some(&ck0), 3);
                let ka = checkpoint(&mut t.o, &t.r, 1, Some(&ck0), &ea, &corpus, b"one-a");
                let eb = exposure(&mut t.o, &t.r, 1, &corpus, Some(&ck0), 4);
                let kb = checkpoint(&mut t.o, &t.r, 1, Some(&ck0), &eb, &corpus, b"one-b");
                commit(&t.r, Some(&t.c1), &[&ea, &ka, &eb, &kb])
            },
        ),
        ("an update 1 repeated after update 1 was committed", |t| {
            let (corpus, ck0) = (t.corpus.clone(), t.ck0.clone());
            let ea = exposure(&mut t.o, &t.r, 1, &corpus, Some(&ck0), 3);
            let ka = checkpoint(&mut t.o, &t.r, 1, Some(&ck0), &ea, &corpus, b"one-a");
            let c2 = commit(&t.r, Some(&t.c1), &[&ea, &ka]);
            assert!(matches!(
                publish(&mut t.o, &t.e.remote, PUBLISHER, &t.r.manifest, &c2).unwrap(),
                Publication::Advanced { height: 2, .. }
            ));
            t.c1 = c2.clone();
            let eb = exposure(&mut t.o, &t.r, 1, &corpus, Some(&ck0), 5);
            commit(&t.r, Some(&c2), &[&eb])
        }),
        (
            "a second checkpoint of update 0 committing the same exposure",
            |t| {
                let (corpus, exposure0) = (t.corpus.clone(), t.exposure0.clone());
                let again = checkpoint(&mut t.o, &t.r, 0, None, &exposure0, &corpus, b"again-0");
                commit(&t.r, Some(&t.c1), &[&again])
            },
        ),
    ];
    let mut accepted = Vec::new();
    for (label, build) in cases {
        let mut t = trained();
        let c = build(&mut t);
        let before = head_bytes(&t.e.remote, &t.r).unwrap();
        match publish(&mut t.o, &t.e.remote, PUBLISHER, &t.r.manifest, &c).unwrap() {
            Publication::Blocked { reason, .. } => {
                assert!(
                    reason.contains("Invalid") && reason.contains("update"),
                    "{label}: {reason}"
                );
                assert_eq!(head_bytes(&t.e.remote, &t.r).unwrap(), before, "{label}");
            }
            other => accepted.push(format!("{label}: {other:?}")),
        }
    }
    assert!(
        accepted.is_empty(),
        "forked or repeated updates reached the run pointer:\n{}",
        accepted.join("\n")
    );

    // Positive control: updates 1 and 2, each from the checkpoint before it, advance.
    let mut t = trained();
    let (corpus, ck0) = (t.corpus.clone(), t.ck0.clone());
    let e1 = exposure(&mut t.o, &t.r, 1, &corpus, Some(&ck0), 3);
    let k1 = checkpoint(&mut t.o, &t.r, 1, Some(&ck0), &e1, &corpus, b"parameters-1");
    let e2 = exposure(&mut t.o, &t.r, 2, &corpus, Some(&k1), 3);
    let k2 = checkpoint(&mut t.o, &t.r, 2, Some(&k1), &e2, &corpus, b"parameters-2");
    let c2 = commit(&t.r, Some(&t.c1), &[&e1, &k1]);
    let c3 = commit(&t.r, Some(&c2), &[&e2, &k2]);
    for (c, height) in [(&c2, 2), (&c3, 3)] {
        assert_eq!(
            publish(&mut t.o, &t.e.remote, PUBLISHER, &t.r.manifest, c).unwrap(),
            Publication::Advanced {
                commit: c.digest.clone(),
                height,
                from: Some(if height == 2 {
                    t.c1.digest.clone()
                } else {
                    c2.digest.clone()
                }),
            }
        );
    }
}

/// An HWM decision link for `completion`, built from a real `hwm::ingest` batch of `state`.
fn ingest_and_link(
    state: &State,
    r: &Run,
    completion: &Decoded<Event>,
    cut: &str,
    value: f64,
    transition: Transition,
) -> (HwmDecision, Vec<u8>) {
    let (record, blob) = hwm_record(cut, value);
    let report = hwm::ingest(
        state,
        None,
        vec![Candidate {
            record: record.clone(),
            blob: blob.clone(),
            chain_from_source: false,
        }],
        json!({"test": "w8-receiving"}),
    )
    .unwrap();
    let batch = report.batch.unwrap();
    let decision = HwmDecision {
        schema: Kind::HwmDecision.schema().into(),
        run_id: r.id().into(),
        work_id: completion.value().work_id.clone(),
        attempt: completion.value().attempt,
        completion: completion.digest().into(),
        record: BlobRef::of(&blob),
        identity_sha256: record.identity_sha256.clone(),
        ledger_batch: batch.batch,
        ledger_path: batch.path,
        ledger_batch_sha256: batch.sha256,
        seq: 0,
        transition,
        qualification_sha256: None,
    };
    (decision, blob)
}

fn append_decision(o: &mut Outbox, decision: &HwmDecision, blob: &[u8]) -> Encoded {
    o.append_blob(blob).unwrap();
    let encoded = encode(decision).unwrap();
    o.append_object(&encoded).unwrap();
    encoded
}

#[test]
fn hwm_decisions_are_checked_against_the_local_hwm_ledger_at_publication() {
    let r = run("run-w8-decisions", 3, 3);
    let completed = r.completed(0, 1);

    // Forged, misdescribed or foreign decisions: none of them may move the pointer.
    type Case = (
        &'static str,
        fn(&Env, &Run, &Decoded<Event>) -> (HwmDecision, Vec<u8>),
    );
    let cases: [Case; 3] = [
        (
            "a decision naming a batch this host never wrote",
            |_, r, c| {
                let d = r.decision(c, Transition::Improved, "01K62AAAAAAAAAAAAAAAAAAAA7");
                let blob = format!("record {} {}", c.value().work_id, c.value().attempt);
                (d.into_value(), blob.into_bytes())
            },
        ),
        ("an initialization claimed as an improvement", |e, r, c| {
            let (d, blob) = ingest_and_link(&e.state, r, c, "cut-a", 0.5, Transition::Initialized);
            (
                HwmDecision {
                    transition: Transition::Improved,
                    ..d
                },
                blob,
            )
        }),
        ("a decision another host's ledger holds", |_, r, c| {
            let elsewhere = tempfile::tempdir().unwrap();
            ingest_and_link(
                &State::at(elsewhere.path()),
                r,
                c,
                "cut-b",
                0.5,
                Transition::Initialized,
            )
        }),
    ];
    let mut accepted = Vec::new();
    for (label, build) in cases {
        let e = env();
        let (mut o, _, c0) = published(&e, &r);
        let (decision, blob) = build(&e, &r, &completed);
        let d = append_decision(&mut o, &decision, &blob);
        let c = commit(&r, Some(&c0), &[&d]);
        let before = head_bytes(&e.remote, &r).unwrap();
        match publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c).unwrap() {
            Publication::Blocked { reason, .. } => {
                assert!(reason.contains("HWM ledger"), "{label}: {reason}");
                assert_eq!(head_bytes(&e.remote, &r).unwrap(), before, "{label}");
            }
            other => accepted.push(format!("{label}: {other:?}")),
        }
    }
    assert!(
        accepted.is_empty(),
        "decisions the local HWM ledger does not hold reached the run pointer:\n{}",
        accepted.join("\n")
    );

    // Positive control: the decision this host's ledger actually recorded advances.
    let e = env();
    let (mut o, _, c0) = published(&e, &r);
    let (decision, blob) = ingest_and_link(
        &e.state,
        &r,
        &completed,
        "cut-a",
        0.5,
        Transition::Initialized,
    );
    let d = append_decision(&mut o, &decision, &blob);
    let c1 = commit(&r, Some(&c0), &[&d]);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c1).unwrap(),
        Publication::Advanced { height: 1, .. }
    ));

    // A fresh host that restored the committed state publishes on top of it without re-deciding
    // the committed decision; a new decision it cannot back against its own ledger is refused.
    let fresh = tempfile::tempdir().unwrap();
    let mut b = Outbox::open(&State::at(fresh.path()), BIG).unwrap();
    restore(&e.remote, r.id(), &mut b).unwrap().unwrap();
    let more: Vec<_> = producer(&r)[5..8].to_vec();
    let objects = append(&mut b, &more);
    let c2 = commit(&r, Some(&c1), &objects.iter().collect::<Vec<_>>());
    assert!(matches!(
        publish(&mut b, &e.remote, PUBLISHER, &r.manifest, &c2).unwrap(),
        Publication::Advanced { height: 2, .. }
    ));
    let (foreign, blob) = ingest_and_link(
        &e.state,
        &r,
        &completed,
        "cut-c",
        0.25,
        Transition::Initialized,
    );
    let d = append_decision(&mut b, &foreign, &blob);
    let c3 = commit(&r, Some(&c2), &[&d]);
    let before = head_bytes(&e.remote, &r).unwrap();
    match publish(&mut b, &e.remote, PUBLISHER, &r.manifest, &c3).unwrap() {
        Publication::Blocked { reason, .. } => assert!(reason.contains("HWM ledger"), "{reason}"),
        other => panic!("a decision the publishing host cannot back advanced: {other:?}"),
    }
    assert_eq!(head_bytes(&e.remote, &r).unwrap(), before);
}
