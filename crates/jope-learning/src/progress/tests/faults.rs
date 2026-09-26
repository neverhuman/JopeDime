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

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "disconnect-before-upload",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(0, Fault::Disconnect)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.local_only.len(), w.second.len() + 1, "{rec:?}");
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "disconnect-mid-delivery",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(2, Fault::Disconnect)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.local_only.len(), w.second.len() + 1 - 2);
                assert_eq!(rec.uncommitted.len(), 2);
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "response-lost-mid-delivery",
            inject: |w| {
                let reason =
                    blocked(publish_c1_faulty(w, |f| f.fault_put(2, Fault::ResponseLost)).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.unacknowledged.len(), 1, "{rec:?}");
                reason
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "partial-upload",
            inject: |w| {
                let key = w.second[0].key();
                let full = w.second[0].encoded().bytes.clone();
                let path = w.remote_path(&key);
                fs::create_dir_all(path.parent().unwrap()).unwrap();
                fs::write(&path, &full[..full.len() / 2]).unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.quarantined.contains(&key), "{rec:?}");
                assert!(
                    rec.local_only.contains(&key),
                    "an item shadowed by a partial upload is local-only, not delivered"
                );
                assert_eq!(fs::read(&path).unwrap(), &full[..full.len() / 2]);
                reason
            },
            committed: Some(0),
            clear: |w| {
                let key = w.second[0].key();
                fs::remove_file(w.remote_path(&key)).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "duplicate-delivery",
            inject: |w| {
                let before = super::fixture::remote_keys(&w.e.remote);
                let (_dir, mut other, restored) = fresh_restore(&w.e.remote, w.r.id()).unwrap();
                let again = other.deliver(&w.e.remote, None).unwrap();
                for seq in 1..=w.o.as_ref().unwrap().records() {
                    let key = w.o.as_ref().unwrap().record(seq).unwrap().item.key();
                    if w.o.as_ref().unwrap().ack(seq).is_some() {
                        w.o.as_ref().unwrap().redeliver(&w.e.remote, &key).unwrap();
                    }
                }
                assert_eq!(super::fixture::remote_keys(&w.e.remote), before);
                format!(
                    "a second host restored {} items (all already present) and redelivered; \
                     {} attempted again",
                    restored.appended, again.attempted
                )
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "conditional-conflict",
            inject: |w| {
                let (_dir, mut rival, _) = fresh_restore(&w.e.remote, w.r.id()).unwrap();
                let e = w.r.scheduled(2).encoded().unwrap();
                rival.append_object(&e).unwrap();
                let rc = commit(&w.r, Some(&w.c0), &[&e]);
                let p = publish(&mut rival, &w.e.remote, PUBLISHER, &w.r.manifest, &rc).unwrap();
                assert!(matches!(p, Publication::Advanced { height: 1, .. }));
                let p = publish_c1(w, None).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.divergent.contains(&w.c1.digest), "{rec:?}");
                format!("{p:?}")
            },
            committed: Some(1),
            clear: nothing,
            resume: |w| {
                // Reconcile: adopt the winner's committed closure into this host's outbox, then
                // commit the same content-addressed work on top of it.
                w.outbox();
                let run_id = w.r.id().to_string();
                let adopted = restore(&w.e.remote, &run_id, w.o.as_mut().unwrap())
                    .unwrap()
                    .unwrap();
                assert!(
                    adopted.appended >= 2,
                    "the winner's commit and event were adopted"
                );
                let head = w.head().unwrap();
                let winner = Encoded::from_bytes(
                    Kind::Commit,
                    &head.commit,
                    &w.e.remote
                        .get(&object_key(Kind::Commit, &head.commit))
                        .unwrap()
                        .unwrap(),
                )
                .unwrap();
                let objects: Vec<Encoded> = w
                    .second
                    .iter()
                    .filter_map(|i| match i {
                        Item::Object(e) => Some(e.clone()),
                        Item::Blob(_) => None,
                    })
                    .collect();
                let rebased = commit(&w.r, Some(&winner), &objects.iter().collect::<Vec<_>>());
                let p = publish_with(w, None, &rebased).unwrap();
                height_after(p, w)
            },
        },
        Case {
            name: "missing-dependency-remote-only",
            inject: |w| {
                fs::remove_file(w.remote_path(&result_blob_key())).unwrap();
                let err = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap_err();
                format!("{err:#}")
            },
            committed: None,
            clear: nothing,
            // The surviving host backs the lost blob and re-uploads it during publication.
            resume: resume_c1,
        },
        Case {
            name: "missing-dependency-everywhere",
            inject: |w| {
                let key = result_blob_key();
                w.stash
                    .insert("blob", fs::read(w.remote_path(&key)).unwrap());
                fs::remove_file(w.remote_path(&key)).unwrap();
                let sha = BlobRef::of(&Run::result(0, 1)).sha256;
                fs::remove_file(w.e.state.blob_path(&sha)).unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                assert!(reason.contains("Missing"), "{reason}");
                reason
            },
            committed: None,
            clear: |w| {
                let bytes = w.stash["blob"].clone();
                w.e.remote
                    .put_immutable(&result_blob_key(), &bytes)
                    .unwrap();
                w.e.state.put_blob(&bytes).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "corrupt-blob",
            inject: |w| {
                let path = w.remote_path(&result_blob_key());
                fs::write(&path, b"not the result").unwrap();
                let reason = blocked(publish_c1(w, None).unwrap());
                assert!(reason.contains("Corrupt"), "{reason}");
                assert_eq!(fs::read(&path).unwrap(), b"not the result", "overwritten");
                reason
            },
            committed: None,
            // The operator quarantines the corrupt object; the surviving host re-uploads it.
            clear: |w| fs::remove_file(w.remote_path(&result_blob_key())).unwrap(),
            resume: resume_c1,
        },
        Case {
            name: "stale-pointer",
            inject: |w| {
                let key = head_key(w.r.id()).unwrap();
                let at_c0 = fs::read(w.remote_path(&key)).unwrap();
                assert!(matches!(
                    publish_c1(w, None).unwrap(),
                    Publication::Advanced { height: 1, .. }
                ));
                w.stash
                    .insert("head", fs::read(w.remote_path(&key)).unwrap());
                // Someone rolls the pointer back to c0.
                fs::write(w.remote_path(&key), &at_c0).unwrap();
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.divergent.contains(&w.c1.digest), "{rec:?}");
                let e = w.r.scheduled(2).encoded().unwrap();
                w.outbox().append_object(&e).unwrap();
                let c2 = commit(&w.r, Some(&w.c1), &[&e]);
                let p = publish_with(w, None, &c2).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                format!("{p:?}")
            },
            committed: Some(0),
            clear: |w| {
                let key = head_key(w.r.id()).unwrap();
                fs::write(w.remote_path(&key), &w.stash["head"]).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "dangling-pointer",
            inject: |w| {
                let key = head_key(w.r.id()).unwrap();
                w.stash
                    .insert("head", fs::read(w.remote_path(&key)).unwrap());
                let mut h = w.head().unwrap();
                h.commit = "0".repeat(64);
                fs::write(w.remote_path(&key), h.encode().unwrap()).unwrap();
                let p = publish_c1(w, None).unwrap();
                assert!(matches!(p, Publication::Conflict { .. }), "{p:?}");
                format!("{p:?}")
            },
            committed: None,
            clear: |w| {
                let key = head_key(w.r.id()).unwrap();
                fs::write(w.remote_path(&key), &w.stash["head"]).unwrap();
            },
            resume: resume_c1,
        },
        Case {
            name: "crash-after-commit-payload",
            inject: |w| crash_at(w, Boundary::Payload),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-commit-record",
            inject: |w| crash_at(w, Boundary::Record),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-remote-put-before-ack",
            inject: |w| {
                let what = crash_at(w, Boundary::RemotePut);
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert_eq!(rec.unacknowledged.len(), 1, "{rec:?}");
                what
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-after-ack",
            inject: |w| crash_at(w, Boundary::Ack),
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "crash-before-pointer-write",
            inject: |w| {
                w.outbox();
                let c1 = w.c1.clone();
                let p = prepare(
                    w.o.as_mut().unwrap(),
                    &w.e.remote,
                    PUBLISHER,
                    &w.r.manifest,
                    &c1,
                )
                .unwrap();
                assert!(matches!(p, Prepared::Ready(_)));
                drop(p);
                w.crash();
                w.outbox();
                let rec = reconstruct(&w.e.remote, w.r.id(), w.o.as_ref()).unwrap();
                assert!(rec.uncommitted.contains(&w.c1.key()) && rec.local_only.is_empty());
                "everything uploaded and verified; the pointer was never written".into()
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "pointer-disconnect",
            inject: |w| {
                let err =
                    publish_c1_faulty(w, |f| f.fault_pointer(0, Fault::Disconnect)).unwrap_err();
                format!("{err:#}")
            },
            committed: Some(0),
            clear: nothing,
            resume: resume_c1,
        },
        Case {
            name: "pointer-response-lost",
            inject: |w| {
                let p = publish_c1_faulty(w, |f| f.fault_pointer(0, Fault::ResponseLost)).unwrap();
                assert!(
                    matches!(p, Publication::Advanced { height: 1, .. }),
                    "{p:?}"
                );
                format!("{p:?}")
            },
            committed: Some(1),
            clear: nothing,
            resume: resume_c1,
        },
    ]
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

#[test]
fn every_fault_leaves_committed_state_restorable_and_every_delivery_accounted_for() {
    let cases = cases();
    let names: BTreeSet<&str> = cases.iter().map(|c| c.name).collect();
    assert_eq!(names.len(), cases.len(), "case names must be unique");
    for case in cases {
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
        eprintln!(
            "fault {:<36} | {} | {line} | recovered to h{height}, restored host resumed to \
             h{resumed}",
            case.name,
            what.chars().take(90).collect::<String>().replace('\n', " ")
        );
    }
}
