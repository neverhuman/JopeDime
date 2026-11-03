//! W8 product integration: the engine's durable events reach the committed graph through the
//! product driver, and only when they verify.
//!
//! The engine tree is written in the engine's own formats, and the engine's own code checks it:
//! `jope_parallel::corpus_publication::verify` accepts the fixture corpus, and
//! `jope_parallel::checkpoint_recovery::reconcile` validates the published checkpoint and writes
//! the real recovery receipt the driver maps.

use super::fixture::{snapshot, Fault, Faulty, PUBLISHER, SOURCE_COMMIT};
use crate::hwm::{self, Candidate};
use crate::progress::cadence::{CheckpointDue, Clock, FlushDue, Policy, Urgent};
use crate::progress::census::Disposition;
use crate::progress::engine::{self, CHECKPOINT_SCHEMA, CORPUS_PUBLICATION_SCHEMA, CORPUS_SCHEMA};
use crate::progress::inventory::*;
use crate::progress::outbox::Refused;
use crate::progress::product::{
    CheckpointStep, Config, Driver, Step, APPROVAL, CONFIG_SCHEMA, OVERRIDDEN,
};
use crate::progress::publish::{reusable, Publication};
use crate::progress::rebuild::summarize;
use jope_learning::canon::sha256_hex;
use jope_learning::mirror::Mirror;
use jope_learning::state::State;
use serde_json::json;
use std::cell::Cell;
use std::fs;
use std::path::{Path, PathBuf};

const RUN: &str = "run-w8-product";
const GLOBAL_BATCH: u64 = 8;

struct TestClock(Cell<u64>);

impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.0.get()
    }
}

impl TestClock {
    fn at(ms: u64) -> TestClock {
        TestClock(Cell::new(ms))
    }

    fn set(&self, ms: u64) {
        self.0.set(ms);
    }
}

fn pretty(v: &serde_json::Value) -> Vec<u8> {
    serde_json::to_vec_pretty(v).unwrap()
}

/// The identities a published corpus binds.
struct CorpusIds {
    contexts: String,
    vocabulary: String,
    index: String,
}

/// Write a published corpus directory exactly as `corpus_publication::build` leaves one.
fn write_corpus(dir: &Path, qualification: bool, seed: &str) -> CorpusIds {
    fs::create_dir_all(dir).unwrap();
    let records = seed.as_bytes().repeat(12)[..12].to_vec();
    let vocabulary = pretty(&json!({"tokens": ["pipe", "ridge"], "seed": seed}));
    let labels = format!("{{\"label\":\"{seed}-1\"}}\n{{\"label\":\"{seed}-2\"}}\n");
    let contexts = sha256_hex(format!("contexts {seed}").as_bytes());
    let index = pretty(&json!({
        "schema": CORPUS_SCHEMA, "qualification": qualification,
        "vocabulary_sha256": sha256_hex(&vocabulary), "context_index_sha256": contexts,
        "records_sha256": sha256_hex(&records), "record_bytes": 4, "records": 3,
        "legal_sets": [], "contexts": [], "groups": []
    }));
    let provenance = pretty(&json!({
        "index_sha256": sha256_hex(&index), "labels_sha256": sha256_hex(labels.as_bytes())
    }));
    let publication = pretty(&json!({
        "schema": CORPUS_PUBLICATION_SCHEMA, "contexts_sha256": contexts,
        "vocabulary_sha256": sha256_hex(&vocabulary), "observations": [],
        "qualification": qualification
    }));
    for (name, bytes) in [
        ("records.bin", records.clone()),
        ("vocabulary.json", vocabulary.clone()),
        ("labels.jsonl", labels.into_bytes()),
        ("index.json", index.clone()),
        ("provenance.json", provenance),
        ("publication.json", publication),
    ] {
        fs::write(dir.join(name), bytes).unwrap();
    }
    CorpusIds {
        contexts,
        vocabulary: sha256_hex(&vocabulary),
        index: sha256_hex(&index),
    }
}

/// An engine tree: one published corpus and one training run on it.
struct Engine {
    dir: tempfile::TempDir,
    ids: CorpusIds,
}

impl Engine {
    fn new() -> Engine {
        let dir = tempfile::tempdir().unwrap();
        let ids = write_corpus(&dir.path().join("corpora/teaching-a"), true, "a");
        fs::create_dir_all(dir.path().join("runs/train-a")).unwrap();
        Engine { dir, ids }
    }

    fn root(&self) -> &Path {
        self.dir.path()
    }

    fn corpus(&self) -> PathBuf {
        self.root().join("corpora/teaching-a")
    }

    fn run(&self) -> PathBuf {
        PathBuf::from("runs/train-a")
    }

    /// Write checkpoint `step` as `training::checkpoint_commit::commit` does, without
    /// publishing it.
    fn write_checkpoint(&self, step: u64, corpus_index: &str) -> PathBuf {
        let dir = self
            .root()
            .join(format!("runs/train-a/checkpoints/step-{step}"));
        fs::create_dir_all(&dir).unwrap();
        let model = format!("model after step {step}").into_bytes();
        let optimizer = format!("optimizer after step {step}").into_bytes();
        let optimizer_meta = pretty(&json!({"step": step, "sha256": sha256_hex(&optimizer),
            "lr": 0.001, "decay": 0.0, "backend": "cpu"}));
        let config = pretty(&json!({"layers": 2}));
        let state = pretty(&json!({
            "schema": CHECKPOINT_SCHEMA, "step": step, "world": 1, "global_batch": GLOBAL_BATCH,
            "corpus_sha256": corpus_index, "vocabulary_sha256": self.ids.vocabulary,
            "model_sha256": sha256_hex(&model), "optimizer_sha256": sha256_hex(&optimizer),
            "optimizer_metadata_sha256": sha256_hex(&optimizer_meta),
            "config_sha256": sha256_hex(&config)
        }));
        for (name, bytes) in [
            ("model.ot", model),
            ("optimizer.ot", optimizer),
            ("optimizer.json", optimizer_meta),
            ("config.json", config),
            ("state.json", state),
        ] {
            fs::write(dir.join(name), bytes).unwrap();
        }
        dir
    }

    /// Publish checkpoint `step` through `latest.json`.
    fn publish(&self, step: u64) {
        let dir = self.write_checkpoint(step, &self.ids.index);
        let state = fs::read(dir.join("state.json")).unwrap();
        fs::write(
            self.root().join("runs/train-a/latest.json"),
            pretty(
                &json!({"checkpoint": format!("runs/train-a/checkpoints/step-{step}"),
                "state_sha256": sha256_hex(&state), "step": step}),
            ),
        )
        .unwrap();
    }
}

/// A publisher host: a state tree and a `file://` remote.
struct Host {
    _dir: tempfile::TempDir,
    state: State,
    remote: Mirror,
}

fn host() -> Host {
    let dir = tempfile::tempdir().unwrap();
    let remote_dir = dir.path().join("remote");
    fs::create_dir_all(&remote_dir).unwrap();
    Host {
        state: State::at(&dir.path().join("state")),
        remote: Mirror::open(remote_dir.to_str().unwrap()).unwrap(),
        _dir: dir,
    }
}

fn spec(stage: &str, subject: &str) -> WorkSpec {
    WorkSpec {
        stage: stage.into(),
        subject: subject.into(),
        inputs: vec![],
    }
}

fn manifest() -> RunManifest {
    RunManifest::new(
        RUN,
        PUBLISHER,
        SOURCE_COMMIT,
        3,
        vec![
            spec("corpus-publication", "teaching-a"),
            spec("update", "train-a"),
        ],
    )
    .unwrap()
}

fn update_unit() -> String {
    spec("update", "train-a").work_id(RUN).unwrap()
}

fn mapped(step: &Step) -> (u64, u64, u64, CheckpointDue) {
    match step.checkpoint.clone() {
        Some(CheckpointStep::Mapped {
            step,
            update_index,
            consumed_rows,
            due,
        }) => (step, update_index, consumed_rows, due),
        other => panic!("expected a mapped checkpoint, got {other:?}"),
    }
}

fn advanced(step: &Step) -> u64 {
    match &step.publication {
        Some(Publication::Advanced { height, .. }) => *height,
        other => panic!("expected an advanced pointer, got {other:?}"),
    }
}

include!("product/verified_events.rs");

/// A driver whose run has committed the corpus and update 0 (step 100). Every published engine
/// checkpoint is due (`checkpoint_updates: 1`).
fn committed_update_zero<'h>(h: &'h Host, engine: &Engine, clock: &TestClock) -> Driver<'h> {
    let config = Config {
        cadence: Policy {
            checkpoint_updates: 1,
            ..Policy::ADOPTED
        },
        ..Config::adopted()
    };
    let mut d = Driver::init(
        &h.state,
        &h.remote,
        config,
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    d.corpus(&engine.corpus(), clock).unwrap();
    engine.publish(100);
    let s = d
        .checkpoint(engine.root(), &engine.run(), &engine.corpus(), 0, clock)
        .unwrap();
    assert_eq!(mapped(&s).1, 0);
    d
}

/// Everything durable: the local state tree and the remote.
fn durable(h: &Host) -> (std::collections::BTreeMap<String, String>, Vec<String>) {
    let mut remote: Vec<String> = h.remote.list("").unwrap();
    remote.sort();
    let remote = remote
        .into_iter()
        .map(|k| format!("{k}={}", sha256_hex(&h.remote.get(&k).unwrap().unwrap())))
        .collect();
    (snapshot(h.state.root()), remote)
}

include!("product/refusal_receipts.rs");

include!("product/outbox_hwm.rs");
