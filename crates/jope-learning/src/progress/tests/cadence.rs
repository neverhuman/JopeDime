//! W8-T05: every flush and checkpoint trigger, and outage behaviour, under a deterministic
//! clock. A failed flush stays pending and never advances the durable watermark. Non-winner
//! outcomes keep compact evidence; nothing is dropped or deleted.

use super::fixture::{env, producer, run, snapshot, Fault, Faulty, Item, BIG};
use crate::progress::cadence::*;
use crate::progress::census::reconcile;
use crate::progress::inventory::*;
use crate::progress::outbox::Outbox;
use std::cell::Cell;
use std::collections::BTreeMap;

struct TestClock(Cell<u64>);

impl TestClock {
    fn advance(&self, ms: u64) {
        self.0.set(self.0.get() + ms);
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.get()
    }
}

fn clock() -> TestClock {
    TestClock(Cell::new(1_000_000))
}

/// Every file of `before` is still present with the same bytes in `after`.
fn nothing_deleted(before: &BTreeMap<String, String>, after: &BTreeMap<String, String>) {
    for (path, sha) in before {
        assert_eq!(after.get(path), Some(sha), "{path} was deleted or changed");
    }
}

#[test]
fn each_flush_trigger_fires_exactly_when_due() {
    let e = env();
    let r = run("run-w8-cadence", 3, 3);
    let clock = clock();
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let mut cadence = Cadence::start(Policy::PROPOSED, clock.now_ms(), &o);
    assert!(
        cadence.pending_urgent().is_empty(),
        "a fresh outbox owes nothing"
    );
    producer(&r)[0].append(&mut o).unwrap();

    // Nothing is due before the interval, with little pending and no urgency.
    clock.advance(59_999);
    let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
    assert_eq!((t.due, t.flushed, t.durable_through), (None, false, 0));
    // The interval: 60 s after the last flush.
    clock.advance(1);
    let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
    assert_eq!(t.due, Some(FlushDue::Interval { waited_ms: 60_000 }));
    assert!(t.flushed);
    assert_eq!(t.durable_through, 1);

    // Size: 1 MiB pending, well inside the interval.
    clock.advance(1_000);
    o.append_blob(&vec![7u8; (1 << 20) - 1]).unwrap();
    assert_eq!(
        cadence.due(clock.now_ms(), &o),
        None,
        "one byte short of 1 MiB"
    );
    o.append_blob(&[8u8]).unwrap();
    let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
    assert_eq!(
        t.due,
        Some(FlushDue::Size {
            pending_bytes: 1 << 20
        })
    );
    assert!(t.flushed);
    // The interval restarted at that flush.
    clock.advance(59_999);
    o.append_blob(b"after the size flush").unwrap();
    assert_eq!(cadence.due(clock.now_ms(), &o), None);

    // Each urgent event flushes immediately.
    for why in [
        Urgent::HwmChange,
        Urgent::CorpusFreeze,
        Urgent::CheckpointCommit,
        Urgent::TerminalFailure,
        Urgent::Shutdown,
    ] {
        clock.advance(10);
        o.append_blob(format!("progress under {why:?}").as_bytes())
            .unwrap();
        cadence.request(why);
        let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
        assert!(
            matches!(&t.due, Some(FlushDue::Urgent { requests }) if requests.contains(&why)),
            "{why:?}: {:?}",
            t.due
        );
        assert!(t.flushed && cadence.pending_urgent().is_empty());
    }
    assert_eq!(durable_through(&o), o.records());

    // An urgent request with nothing pending is satisfied without a flush.
    cadence.request(Urgent::HwmChange);
    let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
    assert_eq!((t.due, t.delivery), (None, None));
    assert!(cadence.pending_urgent().is_empty());
}

#[test]
fn a_failed_flush_stays_pending_and_never_claims_durability() {
    let e = env();
    let r = run("run-w8-outage", 3, 3);
    let clock = clock();
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let mut cadence = Cadence::start(Policy::PROPOSED, clock.now_ms(), &o);
    for item in &producer(&r)[..4] {
        item.append(&mut o).unwrap();
    }
    cadence.request(Urgent::TerminalFailure);
    let before = snapshot(o.root());

    // Outage: every request fails.
    let mut down = Faulty::new(&e.remote);
    for call in 0..10 {
        down = down.fault_put(call, Fault::Disconnect);
    }
    let t = cadence.tick(&clock, &mut o, &down).unwrap();
    assert!(!t.flushed && t.delivery.as_ref().unwrap().failure.is_some());
    assert_eq!(
        t.durable_through, 0,
        "an outage advanced the durable watermark"
    );
    assert_eq!(o.pending_entries(), 4);
    assert_eq!(
        cadence.pending_urgent(),
        vec![Urgent::TerminalFailure],
        "urgency was dropped"
    );
    nothing_deleted(&before, &snapshot(o.root()));

    // A partial flush: two writes succeed, then the store fails again.
    let partial = Faulty::new(&e.remote).fault_put(2, Fault::Disconnect);
    clock.advance(5);
    let t = cadence.tick(&clock, &mut o, &partial).unwrap();
    assert!(!t.flushed);
    assert_eq!(t.durable_through, 2, "exactly the acknowledged prefix");
    assert_eq!(o.pending_entries(), 2);
    assert!(!cadence.pending_urgent().is_empty());

    // A lost response is not durability either.
    let lost = Faulty::new(&e.remote).fault_put(0, Fault::ResponseLost);
    clock.advance(5);
    let t = cadence.tick(&clock, &mut o, &lost).unwrap();
    assert_eq!(t.durable_through, 2);

    // The process dies here. The restarted cadence finds owed entries and flushes at once,
    // reporting the same watermark as before the crash.
    drop(o);
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    assert_eq!(durable_through(&o), 2);
    let mut cadence = Cadence::start(Policy::PROPOSED, clock.now_ms(), &o);
    assert_eq!(cadence.pending_urgent(), vec![Urgent::Restart]);
    let t = cadence.tick(&clock, &mut o, &e.remote).unwrap();
    assert!(t.flushed);
    assert_eq!((t.durable_through, o.pending_entries()), (4, 0));
    nothing_deleted(&before, &snapshot(o.root()));
}

#[test]
fn each_checkpoint_trigger_fires_at_committed_updates_and_shutdown() {
    let clock = clock();
    let mut k = Checkpoints::new(Policy::PROPOSED, clock.now_ms());
    assert_eq!(k.shutdown(), None, "nothing to checkpoint at shutdown");

    // 100 committed updates.
    for _ in 0..99 {
        clock.advance(1_000);
        assert_eq!(k.committed_update(clock.now_ms()), None);
    }
    clock.advance(1_000);
    assert_eq!(
        k.committed_update(clock.now_ms()),
        Some(CheckpointDue::Updates {
            uncheckpointed: 100
        })
    );
    k.checkpointed(clock.now_ms());
    assert_eq!(k.uncheckpointed(), 0);

    // Five minutes, measured at the committed update that crosses them.
    clock.advance(299_999);
    assert_eq!(k.committed_update(clock.now_ms()), None);
    clock.advance(1);
    assert_eq!(
        k.committed_update(clock.now_ms()),
        Some(CheckpointDue::Interval { waited_ms: 300_000 })
    );

    // Controlled shutdown checkpoints whatever is uncheckpointed.
    assert_eq!(
        k.shutdown(),
        Some(CheckpointDue::Shutdown { uncheckpointed: 2 })
    );
    k.checkpointed(clock.now_ms());
    assert_eq!(k.shutdown(), None);

    // A checkpoint is never due without a committed update to checkpoint.
    clock.advance(10 * 300_000);
    assert_eq!(k.shutdown(), None);
    assert_eq!(
        k.committed_update(clock.now_ms()),
        Some(CheckpointDue::Interval {
            waited_ms: 3_000_000
        })
    );
}

#[test]
fn non_winner_outcomes_keep_compact_evidence_never_nothing() {
    assert_eq!(retention(Role::Winner), Retention::Full);
    assert_eq!(retention(Role::AdmittedTeacher), Retention::Full);
    assert_eq!(retention(Role::DiagnosticSample), Retention::Full);
    assert_eq!(retention(Role::Other), Retention::Compact);

    // Every completed attempt gets candidate evidence of its class, and everything is
    // delivered: the compact non-winner is recorded, not dropped.
    let r = run("run-w8-retention", 2, 3);
    let e = env();
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let items = producer(&r);
    for item in &items {
        item.append(&mut o).unwrap();
    }
    let mut evidence = Vec::new();
    for (unit, attempt, role) in [(0usize, 1u32, Role::Winner), (1, 2, Role::Other)] {
        let completion = r.completed(unit, attempt);
        let record = format!("record {unit}").into_bytes();
        o.append_blob(&record).unwrap();
        let artifacts = if retention(role) == Retention::Full {
            let fitted = format!("fitted {unit}").into_bytes();
            o.append_blob(&fitted).unwrap();
            vec![BlobRef::of(&fitted)]
        } else {
            vec![]
        };
        let ev = encode(&CandidateEvidence {
            schema: Kind::CandidateEvidence.schema().into(),
            run_id: r.id().into(),
            work_id: r.work[unit].clone(),
            attempt,
            completion: completion.digest().into(),
            record: BlobRef::of(&record),
            retention: retention(role),
            artifacts,
        })
        .unwrap();
        o.append_object(&ev).unwrap();
        evidence.push(ev);
    }
    let d = o.deliver(&e.remote, None).unwrap();
    assert_eq!((d.pending, d.failure), (0, None));
    for ev in &evidence {
        assert!(
            e.remote.get(&ev.key()).unwrap().is_some(),
            "{} not delivered",
            ev.key()
        );
    }
    let compact = decode::<CandidateEvidence>(&evidence[1].bytes).unwrap();
    assert_eq!(compact.value().retention, Retention::Compact);
    assert!(compact.value().artifacts.is_empty());

    // The census still counts the non-winner's failure and completion.
    let events: Vec<Decoded<Event>> = items
        .iter()
        .filter_map(|i| match i {
            Item::Object(e) if e.kind == Kind::Event => Some(decode::<Event>(&e.bytes).unwrap()),
            _ => None,
        })
        .collect();
    let c = reconcile(&r.manifest, &events, &[]);
    assert_eq!(c.counts.completed, 2);
    assert_eq!(c.counts.failures["timeout"], 1);
}
