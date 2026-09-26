//! W8-T04: reconciling a publication that lost a race, as a product procedure. The plan is
//! read-only; apply adopts the winner and commits the loser's own work on top of it, through the
//! ordinary receiving checks, and never displaces the winner or commits anything twice. A rebase
//! the receiving checks would refuse is refused before anything is written.

use super::fixture::{commit, env, run, snapshot, training, Env, Run, BIG, PUBLISHER};
use crate::progress::inventory::*;
use crate::progress::outbox::Outbox;
use crate::progress::product::{Config, Driver};
use crate::progress::publish::{publish, reusable, Publication};
use crate::progress::rebuild::reconstruct;
use crate::progress::reconcile::{apply, plan};
use crate::state::State;

fn config() -> Config {
    Config {
        limits: BIG,
        ..Config::adopted()
    }
}

/// Host A publishes `a_work` at the empty pointer; host B, with the same manifest, then fails to
/// publish `b_work` on the same (absent) predecessor.
fn race(r: &Run, a_work: &[Encoded], b_work: &[Encoded]) -> (Env, State, Encoded) {
    let e = env();
    let m = r.manifest.encoded().unwrap();
    let mut a = Outbox::open(&e.state, BIG).unwrap();
    a.append_object(&m).unwrap();
    for w in a_work {
        a.append_object(w).unwrap();
    }
    let ca = commit(r, None, &a_work.iter().collect::<Vec<_>>());
    assert!(matches!(
        publish(&mut a, &e.remote, PUBLISHER, &r.manifest, &ca).unwrap(),
        Publication::Advanced { height: 0, .. }
    ));
    drop(a);
    let b_dir = e.state.root().parent().unwrap().join("host-b");
    let b = State::at(&b_dir);
    let mut o = Outbox::open(&b, BIG).unwrap();
    o.append_object(&m).unwrap();
    for w in b_work {
        o.append_object(w).unwrap();
    }
    let cb = commit(r, None, &b_work.iter().collect::<Vec<_>>());
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &cb).unwrap(),
        Publication::Conflict { .. }
    ));
    (e, b, ca)
}

#[test]
fn a_lost_race_is_rebased_onto_the_winner_only_on_apply() {
    let r = run("run-w8-reconcile", 3, 3);
    let (a_event, b_event) = (
        r.scheduled(0).encoded().unwrap(),
        r.scheduled(1).encoded().unwrap(),
    );
    let (e, b, winner) = race(&r, &[a_event.clone()], &[b_event.clone()]);

    // Without reconciliation the loser is stuck: its driver's commit names a winner it never
    // recorded.
    let mut d = Driver::open(&b, &e.remote, config(), r.id(), PUBLISHER, 0).unwrap();
    assert!(d.commit().is_err());
    drop(d);

    // The plan writes nothing.
    let (remote_before, local_before) = (snapshot(&e.remote_dir), snapshot(b.root()));
    let p = plan(&b, &e.remote, r.id(), PUBLISHER, BIG).unwrap();
    assert!(p.admissible, "{:?}", p.refusal);
    assert_eq!(p.head.as_ref().unwrap().commit, winner.digest);
    assert_eq!(p.recommit, vec![b_event.key()]);
    assert_eq!(p.divergent.len(), 1);
    assert_eq!(p.adopt, vec![a_event.key(), winner.key()]);
    assert_eq!(snapshot(&e.remote_dir), remote_before);
    assert_eq!(snapshot(b.root()), local_before);

    // Apply: the winner is the parent, and each event is committed exactly once.
    let applied = apply(&b, &e.remote, config(), r.id(), PUBLISHER, 0).unwrap();
    match applied.applied.unwrap().publication.unwrap() {
        Publication::Advanced { height, from, .. } => {
            assert_eq!((height, from), (1, Some(winner.digest.clone())))
        }
        other => panic!("expected the rebase to advance, got {other:?}"),
    }
    let state = reusable(&e.remote, r.id()).unwrap().unwrap();
    let events: Vec<String> = state
        .graph
        .all::<Event>()
        .iter()
        .map(|e| e.digest().to_string())
        .collect();
    let mut expected = vec![a_event.digest.clone(), b_event.digest.clone()];
    expected.sort();
    assert_eq!(events, expected);
    // The losing commit stays on the remote, unreferenced, as divergent history.
    let rec = reconstruct(&e.remote, r.id(), None).unwrap();
    assert_eq!(rec.divergent.len(), 1);
    // Nothing further to do, and a second apply changes nothing.
    let p = plan(&b, &e.remote, r.id(), PUBLISHER, BIG).unwrap();
    assert!(!p.admissible && p.recommit.is_empty());
    let before = snapshot(&e.remote_dir);
    assert!(apply(&b, &e.remote, config(), r.id(), PUBLISHER, 0)
        .unwrap()
        .applied
        .is_none());
    assert_eq!(snapshot(&e.remote_dir), before);
}

#[test]
fn a_rebase_the_receiving_checks_would_refuse_writes_nothing() {
    let r = run("run-w8-reconcile-fork", 3, 3);
    let m = r.manifest.encoded().unwrap();
    // Both hosts trained update 0, on different corpora: rebasing B's update 0 onto A's would
    // fork the update chain.
    let e = env();
    let mut a = Outbox::open(&e.state, BIG).unwrap();
    a.append_object(&m).unwrap();
    let (ca, ea, _) = training(&mut a, &r, 0, None, 3);
    let c = commit(&r, None, &[&ca, &ea]);
    assert!(matches!(
        publish(&mut a, &e.remote, PUBLISHER, &r.manifest, &c).unwrap(),
        Publication::Advanced { height: 0, .. }
    ));
    drop(a);
    let b = State::at(&e.state.root().parent().unwrap().join("host-b"));
    let mut o = Outbox::open(&b, BIG).unwrap();
    o.append_object(&m).unwrap();
    let (cb, eb, _) = training(&mut o, &r, 0, None, 2);
    let c = commit(&r, None, &[&cb, &eb]);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c).unwrap(),
        Publication::Conflict { .. }
    ));
    drop(o);

    let p = plan(&b, &e.remote, r.id(), PUBLISHER, BIG).unwrap();
    assert!(!p.admissible);
    assert!(
        p.refusal.as_deref().unwrap_or("").contains("update 0"),
        "{:?}",
        p.refusal
    );
    let (remote_before, local_before) = (snapshot(&e.remote_dir), snapshot(b.root()));
    let applied = apply(&b, &e.remote, config(), r.id(), PUBLISHER, 0).unwrap();
    assert!(applied.applied.is_none());
    assert_eq!(snapshot(&e.remote_dir), remote_before);
    assert_eq!(snapshot(b.root()), local_before);
}
