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

mod fault_injection;
pub use fault_injection::{Fault, Faulty};

pub fn spec(i: usize) -> WorkSpec {
    WorkSpec {
        stage: "gp-fit".into(),
        subject: format!("dataset-{i:03}"),
        inputs: vec![sha256_hex(format!("input-{i}").as_bytes())],
    }
}

mod run;
pub use run::{run, Run};

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
