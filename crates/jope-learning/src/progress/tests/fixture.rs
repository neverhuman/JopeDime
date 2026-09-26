//! Shared builders for the W8 progress tests.

use crate::canon::sha256_hex;
use crate::hwm::Transition;
use crate::metric::Metric;
use crate::mirror::Mirror;
use crate::progress::inventory::*;
use crate::progress::outbox::{Appended, Limits, Outbox};
use crate::progress::publish::{publish, Publication};
use crate::progress::remote::Remote;
use crate::record::*;
use crate::state::State;
use anyhow::{bail, Result};
use jope_datasets::store::{ObservedObject, PutOutcome};
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

pub const SOURCE_COMMIT: &str = "c4f60b4c7a2e27a31fc41f875650c89900dc32ed";
pub const PUBLISHER: &str = "xbabe2";

/// One thing a producer appends to the outbox.
#[derive(Clone)]
pub enum Item {
    Object(Encoded),
    Blob(Vec<u8>),
}

impl Item {
    pub fn key(&self) -> String {
        match self {
            Item::Object(e) => e.key(),
            Item::Blob(b) => blob_key(&BlobRef::of(b).sha256),
        }
    }

    pub fn deps(&self) -> Vec<String> {
        match self {
            Item::Object(e) => e.dependencies.iter().map(Dependency::key).collect(),
            Item::Blob(_) => Vec::new(),
        }
    }

    pub fn append(&self, o: &mut Outbox) -> Result<Appended> {
        match self {
            Item::Object(e) => o.append_object(e),
            Item::Blob(b) => o.append_result(b, &BlobRef::of(b)),
        }
    }

    pub fn encoded(&self) -> &Encoded {
        match self {
            Item::Object(e) => e,
            Item::Blob(_) => panic!("a blob has no encoding"),
        }
    }
}

pub fn obj(e: Decoded<Event>) -> Item {
    Item::Object(e.encoded().unwrap())
}

/// What a producer appends for a two-unit run, in the order it produces it: unit 0 completes on
/// its first attempt; unit 1 times out, is retried and completes on attempt 2.
pub fn producer(r: &Run) -> Vec<Item> {
    let f = r.failed(1, 1, FailureClass::Timeout, true);
    vec![
        Item::Object(r.manifest.encoded().unwrap()),
        obj(r.scheduled(0)),
        obj(r.started(0, 1)),
        Item::Blob(Run::result(0, 1)),
        obj(r.completed(0, 1)),
        obj(r.scheduled(1)),
        obj(f.clone()),
        obj(r.retry(1, 2, &f)),
        Item::Blob(Run::result(1, 2)),
        obj(r.completed(1, 2)),
    ]
}

pub const BIG: Limits = Limits {
    pause_entries: 1000,
    pause_bytes: 1 << 30,
    max_entries: 1000,
    max_bytes: 1 << 30,
};

/// A local state tree and a `file://` remote in one temporary directory.
pub struct Env {
    /// Owns the temporary directory; dropped last.
    pub _dir: tempfile::TempDir,
    pub state: State,
    pub remote: Mirror,
    pub remote_dir: PathBuf,
}

pub fn env() -> Env {
    let dir = tempfile::tempdir().unwrap();
    let remote_dir = dir.path().join("remote");
    std::fs::create_dir_all(&remote_dir).unwrap();
    Env {
        state: State::at(&dir.path().join("state")),
        remote: Mirror::open(remote_dir.to_str().unwrap()).unwrap(),
        remote_dir,
        _dir: dir,
    }
}

/// Every key the remote holds under the W8 prefixes.
pub fn remote_keys(remote: &dyn Remote) -> BTreeSet<String> {
    let mut keys: BTreeSet<String> = remote.list("progress/").unwrap().into_iter().collect();
    keys.extend(remote.list("blobs/").unwrap());
    keys
}

/// Every regular file under `root`, relative path to sha256: "nothing was written" is
/// `snapshot(before) == snapshot(after)`.
pub fn snapshot(root: &Path) -> BTreeMap<String, String> {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| {
            (
                e.path()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
                sha256_hex(&std::fs::read(e.path()).unwrap()),
            )
        })
        .collect()
}

/// An injected remote fault.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The request never reaches the store.
    Disconnect,
    /// The store applies the request, then the response is lost.
    ResponseLost,
}

/// A [`Remote`] that injects faults by call number and records every request, so tests can
/// assert order and count requests (failed ones included).
pub struct Faulty<'a> {
    pub inner: &'a dyn Remote,
    /// `put_immutable` call index (0-based) to fault.
    pub put_faults: RefCell<BTreeMap<usize, Fault>>,
    /// `put_pointer` call index (0-based) to fault.
    pub pointer_faults: RefCell<BTreeMap<usize, Fault>>,
    /// `(operation, key, outcome)` for every request.
    pub log: RefCell<Vec<(String, String, String)>>,
    puts: RefCell<usize>,
    pointers: RefCell<usize>,
}

impl<'a> Faulty<'a> {
    pub fn new(inner: &'a dyn Remote) -> Faulty<'a> {
        Faulty {
            inner,
            put_faults: RefCell::default(),
            pointer_faults: RefCell::default(),
            log: RefCell::default(),
            puts: RefCell::new(0),
            pointers: RefCell::new(0),
        }
    }

    pub fn fault_put(self, call: usize, fault: Fault) -> Self {
        self.put_faults.borrow_mut().insert(call, fault);
        self
    }

    pub fn fault_pointer(self, call: usize, fault: Fault) -> Self {
        self.pointer_faults.borrow_mut().insert(call, fault);
        self
    }

    /// Keys of successful or failed requests of `op`, in order.
    pub fn keys(&self, op: &str) -> Vec<String> {
        self.log
            .borrow()
            .iter()
            .filter(|(o, _, _)| o == op)
            .map(|(_, k, _)| k.clone())
            .collect()
    }

    fn note(&self, op: &str, key: &str, outcome: &str) {
        self.log
            .borrow_mut()
            .push((op.into(), key.into(), outcome.into()));
    }

    fn faulted<T>(
        &self,
        op: &str,
        key: &str,
        fault: Option<Fault>,
        call: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        match fault {
            Some(Fault::Disconnect) => {
                self.note(op, key, "disconnect");
                bail!("injected disconnect: {op} {key}")
            }
            Some(Fault::ResponseLost) => {
                let applied = call();
                self.note(
                    op,
                    key,
                    if applied.is_ok() {
                        "applied-response-lost"
                    } else {
                        "failed-response-lost"
                    },
                );
                bail!("injected response loss: {op} {key}")
            }
            None => {
                let result = call();
                self.note(op, key, if result.is_ok() { "ok" } else { "error" });
                result
            }
        }
    }
}

impl Remote for Faulty<'_> {
    fn url(&self) -> &str {
        self.inner.url()
    }

    fn put_immutable(&self, key: &str, bytes: &[u8]) -> Result<PutOutcome> {
        let n = {
            let mut n = self.puts.borrow_mut();
            *n += 1;
            *n - 1
        };
        let fault = self.put_faults.borrow_mut().remove(&n);
        self.faulted("put", key, fault, || self.inner.put_immutable(key, bytes))
    }

    fn get(&self, key: &str) -> Result<Option<Vec<u8>>> {
        let result = self.inner.get(key);
        self.note("get", key, if result.is_ok() { "ok" } else { "error" });
        result
    }

    fn get_observed(&self, key: &str) -> Result<Option<ObservedObject>> {
        let result = self.inner.get_observed(key);
        self.note("get", key, if result.is_ok() { "ok" } else { "error" });
        result
    }

    fn put_pointer(
        &self,
        key: &str,
        bytes: &[u8],
        observed: Option<&ObservedObject>,
    ) -> Result<()> {
        let n = {
            let mut n = self.pointers.borrow_mut();
            *n += 1;
            *n - 1
        };
        let fault = self.pointer_faults.borrow_mut().remove(&n);
        self.faulted("pointer", key, fault, || {
            self.inner.put_pointer(key, bytes, observed)
        })
    }

    fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let result = self.inner.list(prefix);
        self.note("list", prefix, if result.is_ok() { "ok" } else { "error" });
        result
    }
}

pub fn spec(i: usize) -> WorkSpec {
    WorkSpec {
        stage: "gp-fit".into(),
        subject: format!("dataset-{i:03}"),
        inputs: vec![sha256_hex(format!("input-{i}").as_bytes())],
    }
}

/// A run manifest plus the work id of each `spec(i)` (manifest order is by work id, not `i`).
pub struct Run {
    pub manifest: Decoded<RunManifest>,
    pub work: Vec<String>,
}

pub fn run(id: &str, units: usize, max_attempts: u32) -> Run {
    let specs: Vec<WorkSpec> = (0..units).map(spec).collect();
    let work = specs.iter().map(|s| s.work_id(id).unwrap()).collect();
    let manifest = RunManifest::new(id, PUBLISHER, SOURCE_COMMIT, max_attempts, specs).unwrap();
    Run {
        manifest: Decoded::new(manifest).unwrap(),
        work,
    }
}

impl Run {
    pub fn id(&self) -> &str {
        &self.manifest.value().run_id
    }

    pub fn event(&self, unit: usize, attempt: u32, body: EventBody) -> Decoded<Event> {
        Decoded::new(Event::new(&self.manifest, &self.work[unit], attempt, body)).unwrap()
    }

    pub fn scheduled(&self, unit: usize) -> Decoded<Event> {
        self.event(unit, 1, EventBody::Scheduled)
    }

    pub fn started(&self, unit: usize, attempt: u32) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Started {
                worker: format!("worker-{unit}-{attempt}"),
            },
        )
    }

    pub fn failed(
        &self,
        unit: usize,
        attempt: u32,
        class: FailureClass,
        retryable: bool,
    ) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Failed {
                class,
                retryable,
                detail: format!("{} on attempt {attempt}", class.as_str()),
            },
        )
    }

    pub fn retry(&self, unit: usize, attempt: u32, failure: &Decoded<Event>) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Retry {
                failed_event: failure.digest().into(),
            },
        )
    }

    pub fn result(unit: usize, attempt: u32) -> Vec<u8> {
        format!("result of unit {unit} attempt {attempt}").into_bytes()
    }

    pub fn completed(&self, unit: usize, attempt: u32) -> Decoded<Event> {
        self.event(
            unit,
            attempt,
            EventBody::Completed {
                result: BlobRef::of(&Run::result(unit, attempt)),
            },
        )
    }

    pub fn excluded(&self, unit: usize, reason: &str) -> Decoded<Event> {
        self.event(
            unit,
            0,
            EventBody::Excluded {
                reason: reason.into(),
            },
        )
    }

    /// An HWM decision link for `completion` (not bound to a real ledger).
    pub fn decision(
        &self,
        completion: &Decoded<Event>,
        transition: Transition,
        batch: &str,
    ) -> Decoded<HwmDecision> {
        let e = completion.value();
        Decoded::new(HwmDecision {
            schema: Kind::HwmDecision.schema().into(),
            run_id: self.id().into(),
            work_id: e.work_id.clone(),
            attempt: e.attempt,
            completion: completion.digest().into(),
            record: BlobRef::of(format!("record {} {}", e.work_id, e.attempt).as_bytes()),
            identity_sha256: sha256_hex(e.work_id.as_bytes()),
            ledger_batch: batch.into(),
            ledger_path: format!("2026/09/{batch}.jsonl"),
            ledger_batch_sha256: sha256_hex(batch.as_bytes()),
            seq: 0,
            transition,
            qualification_sha256: None,
        })
        .unwrap()
    }
}

/// A labelled event list, so a test can drop exactly one delivery by name.
pub struct Scenario {
    pub events: Vec<(&'static str, Decoded<Event>)>,
    pub decisions: Vec<(&'static str, Decoded<HwmDecision>)>,
}

impl Scenario {
    pub fn events(&self) -> Vec<Decoded<Event>> {
        self.events.iter().map(|(_, e)| e.clone()).collect()
    }

    pub fn decisions(&self) -> Vec<Decoded<HwmDecision>> {
        self.decisions.iter().map(|(_, d)| d.clone()).collect()
    }

    pub fn event(&self, label: &str) -> &Decoded<Event> {
        &self
            .events
            .iter()
            .find(|(l, _)| *l == label)
            .unwrap_or_else(|| panic!("no event {label}"))
            .1
    }

    pub fn without(&self, labels: &[&str]) -> Vec<Decoded<Event>> {
        for l in labels {
            self.event(l);
        }
        self.events
            .iter()
            .filter(|(l, _)| !labels.contains(l))
            .map(|(_, e)| e.clone())
            .collect()
    }
}

/// Five planned units, a budget of three attempts, and one of each terminal shape:
///
/// * unit 0 completes on attempt 1 and initializes its HWM;
/// * unit 1 times out, is retried, completes on attempt 2 and improves its HWM;
/// * unit 2 is refused by preflight (not retryable): terminal failure;
/// * unit 3 is excluded before scheduling (missing paired arm);
/// * unit 4 fails three retryable attempts and exhausts its budget.
pub fn standard() -> (Run, Scenario) {
    let r = run("run-w8-t01", 5, 3);
    let u1_f1 = r.failed(1, 1, FailureClass::Timeout, true);
    let u4_f1 = r.failed(4, 1, FailureClass::Timeout, true);
    let u4_f2 = r.failed(4, 2, FailureClass::WorkerLost, true);
    let u0_c1 = r.completed(0, 1);
    let u1_c2 = r.completed(1, 2);
    let events = vec![
        ("u0-scheduled", r.scheduled(0)),
        ("u0-started-1", r.started(0, 1)),
        ("u0-completed-1", u0_c1.clone()),
        ("u1-scheduled", r.scheduled(1)),
        ("u1-failed-1", u1_f1.clone()),
        ("u1-retry-2", r.retry(1, 2, &u1_f1)),
        ("u1-completed-2", u1_c2.clone()),
        ("u2-scheduled", r.scheduled(2)),
        (
            "u2-failed-1",
            r.failed(2, 1, FailureClass::PreflightRefusal, false),
        ),
        ("u3-excluded", r.excluded(3, "missing paired arm")),
        ("u4-scheduled", r.scheduled(4)),
        ("u4-failed-1", u4_f1.clone()),
        ("u4-retry-2", r.retry(4, 2, &u4_f1)),
        ("u4-failed-2", u4_f2.clone()),
        ("u4-retry-3", r.retry(4, 3, &u4_f2)),
        (
            "u4-failed-3",
            r.failed(4, 3, FailureClass::ParseRejection, true),
        ),
    ];
    let decisions = vec![
        (
            "u0-initialized",
            r.decision(
                &u0_c1,
                Transition::Initialized,
                "01K62AAAAAAAAAAAAAAAAAAAA0",
            ),
        ),
        (
            "u1-improved",
            r.decision(&u1_c2, Transition::Improved, "01K62AAAAAAAAAAAAAAAAAAAA1"),
        ),
    ];
    (r, Scenario { events, decisions })
}

/// An `hwm::ingest`-able record (legacy diagnostic pointer space: no comparison context).
pub fn hwm_record(cut: &str, value: f64) -> (HwmRecord, Vec<u8>) {
    let identity = HwmIdentity {
        comparison: None,
        task: "regression".into(),
        dataset_hash: "dataset".into(),
        cut_id: sha256_hex(cut.as_bytes()),
        evaluator_sha256: UNBOUND.into(),
        vocabulary_sha256: UNBOUND.into(),
        protocol: "test".into(),
    };
    let dsl = format!("(pipe X (ridge alpha={value}))");
    let blob = format!("{cut}/{value}").into_bytes();
    let record = HwmRecord {
        schema: HWM_SCHEMA.into(),
        identity_sha256: identity.sha256().unwrap(),
        identity,
        legacy_identity_sha256: None,
        genome: Genome {
            program_sha256: sha256_hex(dsl.as_bytes()),
            dsl,
            semantic_sha256: None,
        },
        score: Score {
            metric: Metric::Rmse,
            value,
            direction: Metric::Rmse.direction(),
            units: "original_target_units".into(),
        },
        cost: Cost::default(),
        evaluator: Evaluator {
            sha256: UNBOUND.into(),
            version: "test".into(),
        },
        benchmark_valid: false,
        sealed: false,
        replayable: false,
        provenance: Provenance {
            source: "test".into(),
            run_id: "run".into(),
            git_sha: UNBOUND.into(),
            bundle_id: UNBOUND.into(),
            record_sha256: sha256_hex(&blob),
        },
        prev_event_sha256: None,
        improvement_number: 0,
        ts_unix: 1,
    };
    (record, blob)
}

/// A commit of `r` on `parent` adding `objects`.
pub fn commit(r: &Run, parent: Option<&Encoded>, objects: &[&Encoded]) -> Encoded {
    let (parent, height) = match parent {
        Some(p) => (
            Some(p.digest.clone()),
            decode::<Commit>(&p.bytes).unwrap().value().height + 1,
        ),
        None => (None, 0),
    };
    let mut objects: Vec<ObjectRef> = objects.iter().map(|e| e.reference()).collect();
    objects.sort();
    objects.dedup();
    encode(&Commit {
        schema: Kind::Commit.schema().into(),
        run_id: r.id().into(),
        run_manifest: r.manifest.digest().into(),
        parent,
        height,
        objects,
    })
    .unwrap()
}

pub fn head(remote: &dyn Remote, r: &Run) -> Option<Head> {
    remote
        .get(&head_key(r.id()).unwrap())
        .unwrap()
        .map(|b| Head::decode(&b).unwrap())
}

pub fn head_bytes(remote: &dyn Remote, r: &Run) -> Option<Vec<u8>> {
    remote.get(&head_key(r.id()).unwrap()).unwrap()
}

/// Append `items` and return the encoded objects among them, in order.
pub fn append(o: &mut Outbox, items: &[Item]) -> Vec<Encoded> {
    let mut objects = Vec::new();
    for item in items {
        item.append(o).unwrap();
        if let Item::Object(e) = item {
            objects.push(e.clone());
        }
    }
    objects
}

/// A published first commit for `producer(r)`, plus the objects it holds.
pub fn published(e: &Env, r: &Run) -> (Outbox, Vec<Encoded>, Encoded) {
    let mut o = Outbox::open(&e.state, BIG).unwrap();
    let objects = append(&mut o, &producer(r)[..5]);
    let refs: Vec<&Encoded> = objects[1..].iter().collect();
    let c0 = commit(r, None, &refs);
    assert!(matches!(
        publish(&mut o, &e.remote, PUBLISHER, &r.manifest, &c0).unwrap(),
        Publication::Advanced { .. }
    ));
    (o, objects, c0)
}

/// Append a corpus of `rows` members plus the exposure for `update` from `before`; returns
/// (corpus, exposure, member bytes).
pub fn training(
    o: &mut Outbox,
    r: &Run,
    update: u64,
    before: Option<&Encoded>,
    rows: usize,
) -> (Encoded, Encoded, Vec<Vec<u8>>) {
    let members: Vec<Vec<u8>> = (0..rows)
        .map(|i| format!("teacher {update} {i}").into_bytes())
        .collect();
    for m in &members {
        o.append_blob(m).unwrap();
    }
    let mut refs: Vec<BlobRef> = members.iter().map(|m| BlobRef::of(m)).collect();
    refs.sort();
    let corpus = encode(&CorpusManifest {
        schema: Kind::CorpusManifest.schema().into(),
        run_id: r.id().into(),
        corpus_id: format!("corpus-{update}"),
        parent: None,
        members: refs,
        decisions: vec![],
    })
    .unwrap();
    o.append_object(&corpus).unwrap();
    let exposure = encode(&Exposure {
        schema: Kind::Exposure.schema().into(),
        run_id: r.id().into(),
        update_index: update,
        corpus: corpus.digest.clone(),
        checkpoint_before: before.map(|b| b.digest.clone()),
        consumed_rows: rows as u64,
    })
    .unwrap();
    o.append_object(&exposure).unwrap();
    (corpus, exposure, members)
}
