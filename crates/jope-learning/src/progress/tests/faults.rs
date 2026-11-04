//! W8-T06: the offline fault matrix.
//!
//! It covers crashes on both sides of every durable boundary and remote faults: disconnect,
//! partial upload, duplicate delivery, conditional conflict, missing dependency, corrupt blob,
//! stale and dangling pointers, and response loss. After each, the test reconstructs fresh
//! local state from the remote committed graph alone.

use super::fixture::{
    append, commit, env, hwm_record, producer, published, run, Env, Fault, Faulty, Item, Run, BIG,
    PUBLISHER,
};
use crate::hwm::{self, Candidate};
use crate::ledger::Ledger;
use crate::progress::inventory::*;
use crate::progress::outbox::{Boundary, Outbox};
use crate::progress::publish::{prepare, publish, reusable, Prepared, Publication};
use crate::progress::rebuild::{reconstruct, restore, Reconstruction, Restored};
use crate::progress::remote::Remote;
use crate::state::{PointerKey, State};
use anyhow::Result;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

mod cases;
use cases::cases;

fn candidate(value: f64) -> Candidate {
    let (record, blob) = hwm_record("cut-a", value);
    Candidate {
        record,
        blob,
        chain_from_source: false,
    }
}

/// The HWM ledger boundary: `hwm::ingest` appends a batch (batch rename, then `HEAD.json`
/// rename), then moves pointers. A crash between the two renames used to wedge every later
/// ingest on `require_intact`'s `head-mismatch`.
#[test]
fn a_crash_between_an_hwm_batch_and_its_head_does_not_wedge_ingest() {
    let dir = tempfile::tempdir().unwrap();
    let state = State::at(dir.path());
    let ledger = Ledger::open(state.root(), "hwm");
    let key = PointerKey::of(&candidate(0.5).record);
    hwm::ingest(&state, None, vec![candidate(0.5)], json!({"k": 1})).unwrap();
    let head_1 = fs::read(ledger.root().join("HEAD.json")).unwrap();
    let pointer_1 = fs::read(state.pointer_path(&key).unwrap()).unwrap();
    let improved = hwm::ingest(&state, None, vec![candidate(0.4)], json!({"k": 2})).unwrap();
    assert_eq!(improved.accepted, 1);
    // The process died after batch 2's rename: neither its HEAD.json nor its pointer moved.
    fs::write(ledger.root().join("HEAD.json"), &head_1).unwrap();
    fs::write(state.pointer_path(&key).unwrap(), &pointer_1).unwrap();

    let report = hwm::ingest(&state, None, vec![candidate(0.45)], json!({"k": 3}))
        .expect("a crash-lagged HEAD.json wedged every later ingest");
    // Batch 2 is durable and readers already see it, so 0.4 is the HWM: 0.45 does not improve it.
    assert_eq!((report.accepted, report.pointers_recovered), (0, 1));
    assert_eq!(
        state.read_pointer(&key).unwrap().unwrap().score.value,
        0.4,
        "the pointer did not roll forward to the durable acceptance"
    );
    let v = hwm::verify(&state, &[]).unwrap();
    assert!(v.ok(), "{:?}", v.breaks);
    assert_eq!((v.events, v.accepted), (3, 2));
}

// ------------------------------------------------------------------------------------------
// The matrix
// ------------------------------------------------------------------------------------------

/// Commit c0 (unit 0 completed) is published. The outbox also holds unit 1's timeout, retry
/// and completion, and commit c1 over them, which is not yet published.
struct World {
    e: Env,
    r: Run,
    o: Option<Outbox>,
    c0: Encoded,
    c1: Encoded,
    second: Vec<Item>,
    stash: BTreeMap<&'static str, Vec<u8>>,
}

fn world() -> World {
    let e = env();
    let r = run("run-w8-faults", 3, 3);
    let (mut o, _, c0) = published(&e, &r);
    let second: Vec<Item> = producer(&r)[5..].to_vec();
    let objects = append(&mut o, &second);
    let c1 = commit(&r, Some(&c0), &objects.iter().collect::<Vec<_>>());
    World {
        e,
        r,
        o: Some(o),
        c0,
        c1,
        second,
        stash: BTreeMap::new(),
    }
}

impl World {
    /// The outbox, reopened (and therefore recovered and verified) after a crash.
    fn outbox(&mut self) -> &mut Outbox {
        if self.o.is_none() {
            self.o = Some(Outbox::open(&self.e.state, BIG).unwrap());
        }
        self.o.as_mut().unwrap()
    }

    fn crash(&mut self) {
        self.o = None;
    }

    fn remote_path(&self, key: &str) -> std::path::PathBuf {
        self.e.remote_dir.join(key)
    }

    fn head(&self) -> Option<Head> {
        self.e
            .remote
            .get(&head_key(self.r.id()).unwrap())
            .unwrap()
            .map(|b| Head::decode(&b).unwrap())
    }
}

fn publish_with(w: &mut World, remote: Option<&dyn Remote>, c: &Encoded) -> Result<Publication> {
    w.outbox();
    let remote = remote.unwrap_or(&w.e.remote);
    publish(w.o.as_mut().unwrap(), remote, PUBLISHER, &w.r.manifest, c)
}

fn publish_c1(w: &mut World, remote: Option<&dyn Remote>) -> Result<Publication> {
    let c1 = w.c1.clone();
    publish_with(w, remote, &c1)
}

/// Publish c1 through the remote with the faults `inject` adds.
fn publish_c1_faulty(
    w: &mut World,
    inject: impl for<'a> FnOnce(Faulty<'a>) -> Faulty<'a>,
) -> Result<Publication> {
    w.outbox();
    let faulty = inject(Faulty::new(&w.e.remote));
    publish(
        w.o.as_mut().unwrap(),
        &faulty,
        PUBLISHER,
        &w.r.manifest,
        &w.c1,
    )
}

fn height_after(p: Publication, w: &World) -> u64 {
    match p {
        Publication::Advanced { height, .. } => height,
        Publication::AlreadyPublished { .. } => w.head().unwrap().height,
        other => panic!("expected publication, got {other:?}"),
    }
}

fn resume_c1(w: &mut World) -> u64 {
    let p = publish_c1(w, None).unwrap();
    height_after(p, w)
}

fn nothing(_: &mut World) {}

fn keys(o: &Outbox) -> Vec<String> {
    (1..=o.records())
        .map(|s| o.record(s).unwrap().item.key())
        .collect()
}

/// A fresh host that restores the run's committed state from the remote alone.
fn fresh_restore(
    remote: &dyn Remote,
    run_id: &str,
) -> Result<(tempfile::TempDir, Outbox, Restored)> {
    let dir = tempfile::tempdir()?;
    let mut o = Outbox::open(&State::at(dir.path()), BIG)?;
    let restored = restore(remote, run_id, &mut o)?.expect("no committed state to restore");
    Ok((dir, o, restored))
}

/// Every outbox entry is accounted for: an unacknowledged entry is either local-only (lost if
/// this host is lost) or already on the remote. An acknowledged entry is on the remote unless
/// it is reported as acknowledged-missing. Local-only progress is never committed.
fn account(o: &Outbox, rec: &Reconstruction) {
    let mut pending = 0;
    for seq in 1..=o.records() {
        let key = o.record(seq).unwrap().item.key();
        if o.ack(seq).is_none() {
            pending += 1;
            assert!(
                rec.local_only.contains(&key) ^ rec.unacknowledged.contains(&key),
                "pending {key} is unaccounted for"
            );
        } else {
            assert!(!rec.local_only.contains(&key) && !rec.unacknowledged.contains(&key));
        }
    }
    assert_eq!(pending, rec.local_only.len() + rec.unacknowledged.len());
    assert!(rec.local_only.is_disjoint(&rec.committed));
    assert!(rec.acknowledged_missing.is_disjoint(&rec.committed));
}

/// Unit 2's work, published from `o` on top of the remote head: the resumed computation.
fn resume_computation(w: &World, o: &mut Outbox) -> u64 {
    let head = w.head().unwrap();
    let parent = Encoded::from_bytes(
        Kind::Commit,
        &head.commit,
        &w.e.remote
            .get(&object_key(Kind::Commit, &head.commit))
            .unwrap()
            .unwrap(),
    )
    .unwrap();
    let items = [
        Item::Object(w.r.scheduled(2).encoded().unwrap()),
        Item::Object(w.r.started(2, 1).encoded().unwrap()),
        Item::Blob(Run::result(2, 1)),
        Item::Object(w.r.completed(2, 1).encoded().unwrap()),
    ];
    let objects = append(o, &items);
    let next = commit(&w.r, Some(&parent), &objects.iter().collect::<Vec<_>>());
    let p = publish(o, &w.e.remote, PUBLISHER, &w.r.manifest, &next).unwrap();
    match p {
        Publication::Advanced { height, .. } => height,
        other => panic!("the restored host could not resume: {other:?}"),
    }
}

struct Case {
    name: &'static str,
    /// Inject the fault and attempt the publication of c1; returns what happened.
    inject: fn(&mut World) -> String,
    /// The committed height afterwards, or `None` when the committed state must fail visibly.
    committed: Option<u64>,
    /// Store or operator recovery.
    clear: fn(&mut World),
    /// Publish again; returns the committed height.
    resume: fn(&mut World) -> u64,
}

fn blocked(p: Publication) -> String {
    match p {
        Publication::Blocked { reason, .. } => reason,
        other => panic!("expected Blocked, got {other:?}"),
    }
}

fn result_blob_key() -> String {
    blob_key(&BlobRef::of(&Run::result(0, 1)).sha256)
}

fn crash_at(w: &mut World, boundary: Boundary) -> String {
    w.outbox().crash_at = Some(boundary);
    let err = publish_c1(w, None).unwrap_err();
    assert!(err.to_string().contains("simulated crash"), "{err:#}");
    w.crash();
    let recovery = w.outbox().recovery().clone();
    format!(
        "crashed at {boundary:?}; reopened with {} pending, {} orphan payload(s)",
        recovery.pending,
        recovery.orphan_payloads.len()
    )
}

// Each case owns all state, remote storage and fresh restored hosts. Overlap the
// mandatory fsync waits without removing writes or expanding the one test's
// original 240-second PR watchdog. Four case workers stay within the local CPU cap.
const CASE_WORKERS: usize = 4;

fn exercise_case(case: &Case) -> String {
    let mut w = world();
    let what = (case.inject)(&mut w);
    w.outbox();
    let run_id = w.r.id().to_string();
    let found = reconstruct(&w.e.remote, &run_id, w.o.as_ref());
    let line = match case.committed {
        None => {
            let err = found.expect_err("a damaged committed state reconstructed");
            assert!(reusable(&w.e.remote, &run_id).is_err(), "{}", case.name);
            assert!(
                fresh_restore(&w.e.remote, &run_id).is_err(),
                "{}: restored from a damaged graph",
                case.name
            );
            let err = format!("{err:#}");
            let defect = ["Missing", "Corrupt", "Invalid"]
                .into_iter()
                .find(|d| err.contains(d))
                .unwrap_or("error");
            format!("committed state refused ({defect}); reusable and restore refuse too")
        }
        Some(height) => {
            let rec = found.unwrap_or_else(|e| panic!("{}: {e:#}", case.name));
            assert_eq!(
                rec.head.as_ref().map(|h| h.height),
                Some(height),
                "{}",
                case.name
            );
            account(w.o.as_ref().unwrap(), &rec);
            // A fresh host restores exactly the committed closure, deterministically, and
            // nothing that exists only on the surviving host.
            let (_a, fresh, restored) = fresh_restore(&w.e.remote, &run_id).unwrap();
            let (_b, again, _) = fresh_restore(&w.e.remote, &run_id).unwrap();
            assert_eq!(restored.head.height, height);
            let restored_keys = keys(&fresh);
            assert_eq!(
                restored_keys,
                keys(&again),
                "{}: nondeterministic",
                case.name
            );
            assert_eq!(
                restored_keys.iter().cloned().collect::<BTreeSet<_>>(),
                rec.committed,
                "{}",
                case.name
            );
            assert!(rec.local_only.iter().all(|k| !restored_keys.contains(k)));
            let mut fresh = fresh;
            let twice = restore(&w.e.remote, &run_id, &mut fresh).unwrap().unwrap();
            assert_eq!(
                (twice.appended, twice.present),
                (0, rec.committed.len() as u64)
            );
            format!(
                "committed h{height}: {} committed, {} uncommitted, {} divergent, {} \
                 quarantined, {} local-only, {} unacknowledged, {} acknowledged-missing; \
                 restored {} items",
                rec.committed.len(),
                rec.uncommitted.len(),
                rec.divergent.len(),
                rec.quarantined.len(),
                rec.local_only.len(),
                rec.unacknowledged.len(),
                rec.acknowledged_missing.len(),
                restored.appended
            )
        }
    };
    (case.clear)(&mut w);
    let height = (case.resume)(&mut w);
    let rec = reconstruct(&w.e.remote, &run_id, w.o.as_ref())
        .unwrap_or_else(|e| panic!("{}: after recovery: {e:#}", case.name));
    assert_eq!(rec.head.as_ref().map(|h| h.height), Some(height));
    assert!(
        rec.local_only.is_empty()
            && rec.unacknowledged.is_empty()
            && rec.acknowledged_missing.is_empty()
            && rec.quarantined.is_empty(),
        "{}: {rec:?}",
        case.name
    );
    account(w.o.as_ref().unwrap(), &rec);
    // A fresh host restored from the remote alone resumes the computation on the tip.
    let (_dir, mut fresh, _) = fresh_restore(&w.e.remote, &run_id).unwrap();
    let resumed = resume_computation(&w, &mut fresh);
    assert_eq!(resumed, height + 1, "{}", case.name);
    format!(
        "fault {:<36} | {} | {line} | recovered to h{height}, restored host resumed to \
         h{resumed}",
        case.name,
        what.chars().take(90).collect::<String>().replace('\n', " ")
    )
}

#[test]
fn every_fault_leaves_committed_state_restorable_and_every_delivery_accounted_for() {
    let cases = cases();
    let names: BTreeSet<&str> = cases.iter().map(|c| c.name).collect();
    assert_eq!(names.len(), cases.len(), "case names must be unique");
    assert_eq!(cases.len(), 18, "the complete fault matrix must execute");
    let mut completed = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..CASE_WORKERS)
            .map(|worker| {
                let cases = &cases;
                scope.spawn(move || {
                    cases
                        .iter()
                        .enumerate()
                        .skip(worker)
                        .step_by(CASE_WORKERS)
                        .map(|(index, case)| (index, exercise_case(case)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("a fault case failed"))
            .collect::<Vec<_>>()
    });
    completed.sort_by_key(|(index, _)| *index);
    assert_eq!(completed.len(), cases.len(), "no case may be omitted");
    for (expected, (index, line)) in completed.into_iter().enumerate() {
        assert_eq!(index, expected, "each case executes exactly once");
        eprintln!("{line}");
    }
}
