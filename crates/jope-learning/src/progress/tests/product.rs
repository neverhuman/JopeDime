//! W8 product integration: the engine's durable events reach the committed graph through the
//! product driver, and only when they verify.
//!
//! The engine tree is written in the engine's own formats, and the engine's own code checks it:
//! `jope_parallel::corpus_publication::verify` accepts the fixture corpus, and
//! `jope_parallel::checkpoint_recovery::reconcile` validates the published checkpoint and writes
//! the real recovery receipt the driver maps.

use super::fixture::{snapshot, Fault, Faulty, PUBLISHER, SOURCE_COMMIT};
use crate::canon::sha256_hex;
use crate::hwm::{self, Candidate};
use crate::mirror::Mirror;
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
use crate::state::State;
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

#[test]
fn engine_events_reach_the_committed_graph_through_the_product_driver() {
    let engine = Engine::new();
    // The engine's own verifier accepts the fixture as a published corpus.
    jope_parallel::corpus_publication::verify(
        &engine.corpus(),
        &engine.ids.contexts,
        &engine.ids.vocabulary,
    )
    .unwrap();
    let h = host();
    let clock = TestClock::at(1_000_000);
    let update = update_unit();
    let mut d = Driver::init(
        &h.state,
        &h.remote,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    assert!(d.head().is_none());

    // Attempt 1 of the update is scheduled. Nothing is due yet: the flush interval is 60 s.
    let s = d.schedule(&update).unwrap();
    assert_eq!((s.appended.len(), s.staged), (1, 1));
    let t = d.tick(&clock).unwrap();
    assert!(t.due.is_none() && t.publication.is_none(), "{t:?}");

    // The corpus freeze flushes at once: the corpus and the scheduled event commit at height 0.
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    assert_eq!(
        s.appended.len(),
        7,
        "six payload files and the corpus manifest"
    );
    assert!(matches!(
        &s.due,
        Some(FlushDue::Urgent { requests }) if requests == &[Urgent::CorpusFreeze]
    ));
    assert_eq!(advanced(&s), 0);
    assert!(s.flushed && s.staged == 0);

    // Engine checkpoints at steps 40 and 80 are counted, not published as W8 updates.
    let root = engine.root().to_path_buf();
    let run = engine.run();
    let corpus = engine.corpus();
    for (step, at) in [(40, 1_010_000), (80, 1_020_000)] {
        engine.publish(step);
        clock.set(at);
        let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
        assert_eq!(
            s.checkpoint,
            Some(CheckpointStep::Deferred {
                step,
                uncheckpointed: step
            })
        );
        assert!(s.appended.is_empty() && s.publication.is_none());
    }
    // Step 100: the update trigger fires, and update 0 covers the 100 steps.
    engine.publish(100);
    clock.set(1_030_000);
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            100,
            0,
            100 * GLOBAL_BATCH,
            CheckpointDue::Updates {
                uncheckpointed: 100
            }
        )
    );
    assert_eq!(advanced(&s), 1);
    assert!(s.flushed);
    let checkpointed_at = clock.now_ms();

    // The ranks die after step 100: an unpublished checkpoint and a staging directory are left.
    // The engine's own recovery quarantines both and writes its receipt.
    engine.write_checkpoint(130, &engine.ids.index);
    let staging = root.join("runs/train-a/checkpoints/.step-140-4242");
    fs::create_dir_all(&staging).unwrap();
    fs::write(staging.join("model.ot"), b"partial").unwrap();
    jope_parallel::checkpoint_recovery::reconcile(&root, &root.join(&run)).unwrap();
    let receipt = fs::read_dir(root.join("runs/train-a/checkpoint-recovery"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path()
        .join("receipt.json");
    let s = d.recovery(&update, 1, &receipt).unwrap();
    assert_eq!(s.appended.len(), 1);
    let s = d.retry(&update, 2).unwrap();
    assert_eq!((s.appended.len(), s.staged), (1, 2));
    // The interval flush commits the failure and its retry: not at 59.999 s, at 60 s.
    clock.set(checkpointed_at + 59_999);
    assert!(d.tick(&clock).unwrap().publication.is_none());
    clock.set(checkpointed_at + 60_000);
    let t = d.tick(&clock).unwrap();
    assert!(matches!(t.due, Some(FlushDue::Interval { .. })), "{t:?}");
    assert_eq!(advanced(&t), 2);

    // Step 150, five minutes after update 0 was committed: the interval trigger fires.
    engine.publish(150);
    clock.set(checkpointed_at + 300_000);
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            150,
            1,
            50 * GLOBAL_BATCH,
            CheckpointDue::Interval { waited_ms: 300_000 }
        )
    );
    assert_eq!(advanced(&s), 3);

    // The retried attempt's result is a no-replace engine publication.
    let result = b"{\"schema\":\"parallel.training_complete.v1\",\"steps\":180}".to_vec();
    fs::write(root.join("runs/train-a/complete-attempt-2.json"), &result).unwrap();
    let s = d
        .result(
            &update,
            2,
            &root,
            Path::new("runs/train-a/complete-attempt-2.json"),
            &sha256_hex(&result),
        )
        .unwrap();
    assert_eq!(s.appended.len(), 2, "the result blob and its completion");

    // Controlled shutdown at step 180: the remaining 30 steps become update 2, and everything
    // pending, the completion included, commits.
    engine.publish(180);
    clock.set(checkpointed_at + 310_000);
    let s = d.shutdown(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(
        mapped(&s),
        (
            180,
            2,
            30 * GLOBAL_BATCH,
            CheckpointDue::Shutdown { uncheckpointed: 30 }
        )
    );
    assert_eq!(advanced(&s), 4);
    assert!(s.flushed && s.staged == 0);
    assert_eq!(d.outbox().pending_entries(), 0);

    // The committed graph: three consecutive updates on one corpus, and the census.
    let state = reusable(&h.remote, RUN).unwrap().unwrap();
    let mut checkpoints: Vec<_> = state.graph.all::<Checkpoint>();
    checkpoints.sort_by_key(|k| k.value().update_index);
    assert_eq!(
        checkpoints
            .iter()
            .map(|k| k.value().update_index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
    for pair in checkpoints.windows(2) {
        assert_eq!(pair[1].value().parent.as_deref(), Some(pair[0].digest()));
    }
    let mut rows: Vec<(u64, u64)> = state
        .graph
        .all::<Exposure>()
        .iter()
        .map(|e| (e.value().update_index, e.value().consumed_rows))
        .collect();
    rows.sort();
    assert_eq!(rows, [(0, 800), (1, 400), (2, 240)]);
    let (summary, census) = summarize(&state).unwrap();
    assert_eq!(
        (summary.corpora, summary.checkpoints, summary.consumed_rows),
        (1, 3, 180 * GLOBAL_BATCH)
    );
    assert_eq!((census.counts.completed, census.counts.unscheduled), (1, 1));
    assert_eq!(census.counts.retries, 1);
    assert_eq!(census.counts.failures.get("worker-lost"), Some(&1));
    let unit = census.units.iter().find(|u| u.work_id == update).unwrap();
    assert!(matches!(
        unit.disposition,
        Disposition::Completed { attempt: 2, .. }
    ));
    // The quarantined step-130 checkpoint never became progress.
    let quarantined = fs::read(receipt.parent().unwrap().join("step-130/state.json")).unwrap();
    assert!(h
        .remote
        .get(&blob_key(&sha256_hex(&quarantined)))
        .unwrap()
        .is_none());

    // Restart: a reopened driver re-derives its state, and the same engine state maps to
    // nothing new.
    drop(d);
    let mut d = Driver::open(
        &h.state,
        &h.remote,
        Config::adopted(),
        RUN,
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    assert_eq!(d.previous().unwrap().step, 180);
    assert_eq!(d.previous().unwrap().checkpoint.value().update_index, 2);
    assert_eq!((d.staged(), d.head().unwrap().height), (0, 4));
    let s = d.checkpoint(&root, &run, &corpus, 0, &clock).unwrap();
    assert_eq!(s.checkpoint, Some(CheckpointStep::Nothing));
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    assert!(s.appended.is_empty() && s.existing.len() == 7 && s.publication.is_none());
}

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

#[test]
fn engine_events_that_do_not_verify_append_nothing_and_move_nothing() {
    type Case = (
        &'static str,
        fn(&Engine, &mut Driver<'_>, &TestClock) -> anyhow::Result<Step>,
    );
    let cases: Vec<Case> = vec![
        ("a corpus still in staging", |e, d, c| {
            let to = e.root().join("corpora/.teaching-b.staging");
            write_corpus(&to, true, "b");
            d.corpus(&to, c)
        }),
        ("a retired corpus build", |e, d, c| {
            let to = e.root().join("corpora/teaching-b.retired-1-2-0");
            write_corpus(&to, true, "b");
            d.corpus(&to, c)
        }),
        ("a symlinked corpus directory", |e, d, c| {
            let to = e.root().join("corpora/teaching-link");
            std::os::unix::fs::symlink(e.corpus(), &to).unwrap();
            d.corpus(&to, c)
        }),
        ("a diagnostic corpus", |e, d, c| {
            let to = e.root().join("corpora/teaching-b");
            write_corpus(&to, false, "b");
            d.corpus(&to, c)
        }),
        (
            "a corpus whose records changed after publication",
            |e, d, c| {
                let to = e.root().join("corpora/teaching-b");
                write_corpus(&to, true, "b");
                fs::write(to.join("records.bin"), b"bbbbbbbbbbbX").unwrap();
                d.corpus(&to, c)
            },
        ),
        ("a corpus with truncated records", |e, d, c| {
            let to = e.root().join("corpora/teaching-b");
            write_corpus(&to, true, "b");
            // Re-bind index and receipt to four records the payload does not hold.
            let mut index: serde_json::Value =
                serde_json::from_slice(&fs::read(to.join("index.json")).unwrap()).unwrap();
            index["records"] = 4.into();
            let index = pretty(&index);
            fs::write(to.join("index.json"), &index).unwrap();
            let mut prov: serde_json::Value =
                serde_json::from_slice(&fs::read(to.join("provenance.json")).unwrap()).unwrap();
            prov["index_sha256"] = sha256_hex(&index).into();
            fs::write(to.join("provenance.json"), pretty(&prov)).unwrap();
            d.corpus(&to, c)
        }),
        (
            "a corpus whose publication request names other contexts",
            |e, d, c| {
                let to = e.root().join("corpora/teaching-b");
                write_corpus(&to, true, "b");
                let mut p: serde_json::Value =
                    serde_json::from_slice(&fs::read(to.join("publication.json")).unwrap())
                        .unwrap();
                p["contexts_sha256"] = sha256_hex(b"other contexts").into();
                fs::write(to.join("publication.json"), pretty(&p)).unwrap();
                d.corpus(&to, c)
            },
        ),
        ("latest.json naming another run's checkpoint", |e, d, c| {
            e.publish(200);
            let latest = e.root().join("runs/train-a/latest.json");
            let mut v: serde_json::Value =
                serde_json::from_slice(&fs::read(&latest).unwrap()).unwrap();
            v["checkpoint"] = "runs/other/checkpoints/step-200".into();
            fs::write(&latest, pretty(&v)).unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        (
            "a checkpoint object changed after publication",
            |e, d, c| {
                e.publish(200);
                fs::write(
                    e.root().join("runs/train-a/checkpoints/step-200/model.ot"),
                    b"x",
                )
                .unwrap();
                d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
            },
        ),
        (
            "a checkpoint manifest latest.json did not publish",
            |e, d, c| {
                e.publish(200);
                let state = e
                    .root()
                    .join("runs/train-a/checkpoints/step-200/state.json");
                let mut v: serde_json::Value =
                    serde_json::from_slice(&fs::read(&state).unwrap()).unwrap();
                v["global_batch"] = 4096.into();
                fs::write(&state, pretty(&v)).unwrap();
                d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
            },
        ),
        ("a checkpoint trained on another corpus", |e, d, c| {
            let dir = e.write_checkpoint(200, &sha256_hex(b"another corpus index"));
            let state = fs::read(dir.join("state.json")).unwrap();
            fs::write(
                e.root().join("runs/train-a/latest.json"),
                pretty(&json!({"checkpoint": "runs/train-a/checkpoints/step-200",
                    "state_sha256": sha256_hex(&state), "step": 200})),
            )
            .unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        ("a symlinked checkpoint directory", |e, d, c| {
            let real = e.write_checkpoint(200, &e.ids.index);
            let moved = e.root().join("elsewhere-step-200");
            fs::rename(&real, &moved).unwrap();
            std::os::unix::fs::symlink(&moved, &real).unwrap();
            let state = fs::read(moved.join("state.json")).unwrap();
            fs::write(
                e.root().join("runs/train-a/latest.json"),
                pretty(&json!({"checkpoint": "runs/train-a/checkpoints/step-200",
                    "state_sha256": sha256_hex(&state), "step": 200})),
            )
            .unwrap();
            d.checkpoint(e.root(), &e.run(), &e.corpus(), 0, c)
        }),
        ("an incomplete recovery receipt", |e, d, _| {
            let path = e.root().join("receipt.json");
            fs::write(
                &path,
                pretty(
                    &json!({"schema": engine::RECOVERY_SCHEMA, "completed": false,
                    "published_step": 100, "moves": [{"name": "step-130", "moved": true}]}),
                ),
            )
            .unwrap();
            d.recovery(&update_unit(), 1, &path)
        }),
        ("a recovery for work the run never planned", |e, d, _| {
            let path = e.root().join("receipt.json");
            fs::write(
                &path,
                pretty(
                    &json!({"schema": engine::RECOVERY_SCHEMA, "completed": true,
                    "published_step": 100, "moves": [{"name": "step-130", "moved": true}]}),
                ),
            )
            .unwrap();
            d.recovery(&"0".repeat(64), 1, &path)
        }),
        (
            "a result whose bytes are not the published digest",
            |e, d, _| {
                fs::write(e.root().join("runs/train-a/result.json"), b"result").unwrap();
                d.result(
                    &update_unit(),
                    1,
                    e.root(),
                    Path::new("runs/train-a/result.json"),
                    &sha256_hex(b"another result"),
                )
            },
        ),
        ("a result path that leaves the engine root", |e, d, _| {
            fs::write(e.root().join("outside.json"), b"result").unwrap();
            d.result(
                &update_unit(),
                1,
                e.root(),
                Path::new("runs/../outside.json"),
                &sha256_hex(b"result"),
            )
        }),
        ("a symlinked result", |e, d, _| {
            fs::write(e.root().join("outside.json"), b"result").unwrap();
            std::os::unix::fs::symlink(
                e.root().join("outside.json"),
                e.root().join("runs/train-a/result.json"),
            )
            .unwrap();
            d.result(
                &update_unit(),
                1,
                e.root(),
                Path::new("runs/train-a/result.json"),
                &sha256_hex(b"result"),
            )
        }),
    ];
    for (label, run_case) in cases {
        let engine = Engine::new();
        let h = host();
        let clock = TestClock::at(5_000_000);
        let mut d = committed_update_zero(&h, &engine, &clock);
        let before = durable(&h);
        let head = d.head().cloned();
        let result = run_case(&engine, &mut d, &clock);
        assert!(result.is_err(), "{label} was accepted: {:?}", result.ok());
        assert_eq!(durable(&h), before, "{label}: something durable changed");
        assert_eq!(d.head().cloned(), head, "{label}");
        assert_eq!(d.staged(), 0, "{label}");
        drop(d);
    }

    // An unpublished checkpoint directory is never mapped, however new: only latest.json names
    // progress.
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(5_000_000);
    let mut d = committed_update_zero(&h, &engine, &clock);
    engine.write_checkpoint(300, &engine.ids.index);
    let before = durable(&h);
    let s = d
        .checkpoint(engine.root(), &engine.run(), &engine.corpus(), 0, &clock)
        .unwrap();
    assert_eq!(s.checkpoint, Some(CheckpointStep::Nothing));
    assert_eq!(durable(&h), before);

    // Positive control: the next published checkpoint maps as update 1 and advances.
    engine.publish(200);
    let s = d
        .checkpoint(engine.root(), &engine.run(), &engine.corpus(), 0, &clock)
        .unwrap();
    assert_eq!(mapped(&s).1, 1);
    assert_eq!(advanced(&s), 2);
    // A checkpoint that does not advance past the last update is refused by the mapper itself.
    let corpus = engine::corpus(RUN, &engine.corpus(), None, 1 << 30).unwrap();
    assert!(engine::checkpoint(
        engine.root(),
        &engine.run(),
        &corpus,
        d.previous(),
        0,
        1 << 30
    )
    .is_err());
}

#[test]
fn a_failed_flush_keeps_progress_owed_and_claims_nothing() {
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let faulty = Faulty::new(&h.remote).fault_put(2, Fault::Disconnect);
    let mut d = Driver::init(
        &h.state,
        &faulty,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    d.schedule(&update_unit()).unwrap();
    let s = d.corpus(&engine.corpus(), &clock).unwrap();
    match &s.publication {
        Some(Publication::Blocked { reason, .. }) => {
            assert!(reason.contains("delivery"), "{reason}")
        }
        other => panic!("expected a blocked publication, got {other:?}"),
    }
    assert!(!s.flushed && s.staged == 2 && d.head().is_none());
    assert!(h.remote.get(&head_key(RUN).unwrap()).unwrap().is_none());
    assert!(d.outbox().pending_entries() > 0);
    // The urgency is kept: the next tick retries and commits without any new event.
    let t = d.tick(&clock).unwrap();
    assert!(matches!(
        &t.due,
        Some(FlushDue::Urgent { requests }) if requests.contains(&Urgent::CorpusFreeze)
    ));
    assert_eq!(advanced(&t), 0);
    assert!(t.flushed && d.outbox().pending_entries() == 0);
    assert_eq!(t.tick.unwrap().durable_through, d.outbox().records());

    // A restart with appended but uncommitted progress commits it at once.
    let s = d.schedule(
        &spec("corpus-publication", "teaching-a")
            .work_id(RUN)
            .unwrap(),
    );
    s.unwrap();
    drop(d);
    let mut d = Driver::open(
        &h.state,
        &h.remote,
        Config::adopted(),
        RUN,
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    assert_eq!(d.staged(), 1);
    assert_eq!(advanced(&d.tick(&clock).unwrap()), 1);
}

#[test]
fn outbox_limits_and_cadence_are_explicit_adopted_configuration() {
    let adopted = Config::adopted();
    assert_eq!(
        (
            adopted.limits.pause_entries,
            adopted.limits.pause_bytes,
            adopted.limits.max_entries,
            adopted.limits.max_bytes
        ),
        (7_500, 768 << 20, 10_000, 1 << 30)
    );
    assert_eq!(
        (
            adopted.cadence.flush_interval_ms,
            adopted.cadence.flush_bytes,
            adopted.cadence.checkpoint_updates,
            adopted.cadence.checkpoint_interval_ms
        ),
        (60_000, 1 << 20, 100, 300_000)
    );
    let report = adopted.report();
    assert!(report.adopted_defaults);
    assert_eq!(
        (report.approval, APPROVAL),
        (
            "adopted-by-owner-delegation-2026-09-26",
            "adopted-by-owner-delegation-2026-09-26"
        )
    );

    // An explicit file names every value; other values are reported as the operator's, and
    // nothing in the file can mark them adopted.
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("config.json");
    let explicit = json!({"schema": CONFIG_SCHEMA,
        "limits": {"pause_entries": 2, "pause_bytes": 1 << 20, "max_entries": 3, "max_bytes": 1 << 20},
        "cadence": {"flush_interval_ms": 1000, "flush_bytes": 4096, "checkpoint_updates": 10,
            "checkpoint_interval_ms": 5000}});
    fs::write(&file, explicit.to_string()).unwrap();
    let config = Config::load(&file).unwrap();
    let report = config.report();
    assert!(!report.adopted_defaults);
    assert_eq!(report.approval, OVERRIDDEN);
    for (label, mutate) in [
        (
            "an approval field",
            (|v: &mut serde_json::Value| v["approval"] = "approved".into())
                as fn(&mut serde_json::Value),
        ),
        ("a missing cadence", |v| {
            v.as_object_mut().unwrap().remove("cadence");
        }),
        ("a missing limit", |v| {
            v["limits"].as_object_mut().unwrap().remove("max_bytes");
        }),
        ("a zero flush interval", |v| {
            v["cadence"]["flush_interval_ms"] = 0.into()
        }),
        ("a pause above the maximum", |v| {
            v["limits"]["pause_entries"] = 4.into()
        }),
        ("another schema", |v| {
            v["schema"] = "jope.progress.config/v0".into()
        }),
    ] {
        let mut v = explicit.clone();
        mutate(&mut v);
        fs::write(&file, v.to_string()).unwrap();
        assert!(Config::load(&file).is_err(), "{label} was accepted");
    }

    // The configured limits govern the product outbox: at the pause, new work is refused with
    // nothing written, while the outcome of work in flight is still admitted.
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let mut d = Driver::init(&h.state, &h.remote, config, manifest(), PUBLISHER, 1_000).unwrap();
    let update = update_unit();
    d.schedule(&update).unwrap();
    assert_eq!(d.outbox().pending_entries(), 2);
    let before = snapshot(h.state.root());
    let err = d
        .schedule(
            &spec("corpus-publication", "teaching-a")
                .work_id(RUN)
                .unwrap(),
        )
        .unwrap_err();
    assert!(
        matches!(err.downcast_ref::<Refused>(), Some(Refused::Paused { .. })),
        "{err:#}"
    );
    assert_eq!(snapshot(h.state.root()), before);
    let result = b"result".to_vec();
    fs::write(engine.root().join("runs/train-a/r.json"), &result).unwrap();
    let err = d
        .result(
            &update,
            1,
            engine.root(),
            Path::new("runs/train-a/r.json"),
            &sha256_hex(&result),
        )
        .unwrap_err();
    // Two more entries would pass the maximum of three: refused whole, nothing written.
    assert!(
        matches!(err.downcast_ref::<Refused>(), Some(Refused::Full { .. })),
        "{err:#}"
    );
    assert_eq!(snapshot(h.state.root()), before);
    // A flush drains the outbox, and the refused work is then admitted.
    clock.set(1_000 + 1_000);
    let t = d.tick(&clock).unwrap();
    assert!(t.flushed, "{t:?}");
    d.result(
        &update,
        1,
        engine.root(),
        Path::new("runs/train-a/r.json"),
        &sha256_hex(&result),
    )
    .unwrap();
}

#[test]
fn hwm_decisions_link_through_this_hosts_ledger_and_flush_at_once() {
    let engine = Engine::new();
    let h = host();
    let clock = TestClock::at(1_000);
    let update = update_unit();
    let mut d = Driver::init(
        &h.state,
        &h.remote,
        Config::adopted(),
        manifest(),
        PUBLISHER,
        clock.now_ms(),
    )
    .unwrap();
    d.schedule(&update).unwrap();
    let result = b"fitted candidate".to_vec();
    fs::write(engine.root().join("runs/train-a/candidate.json"), &result).unwrap();
    d.result(
        &update,
        1,
        engine.root(),
        Path::new("runs/train-a/candidate.json"),
        &sha256_hex(&result),
    )
    .unwrap();
    let (record, blob) = super::fixture::hwm_record("cut-a", 0.5);
    let report = hwm::ingest(
        &h.state,
        None,
        vec![Candidate {
            record,
            blob,
            chain_from_source: false,
        }],
        json!({"test": "w8-product"}),
    )
    .unwrap();
    let batch = report.batch.unwrap();

    // Refusals write nothing.
    let before = durable(&h);
    for (label, attempt, path, seq) in [
        (
            "a batch this ledger does not hold",
            1,
            "2026/09/01K62ZZZZZZZZZZZZZZZZZZZZZ.jsonl",
            0,
        ),
        (
            "an event the batch does not hold",
            1,
            batch.path.as_str(),
            5,
        ),
        ("an attempt that never completed", 2, batch.path.as_str(), 0),
    ] {
        assert!(
            d.decision(&update, attempt, path, seq, &clock).is_err(),
            "{label} was linked"
        );
        assert_eq!(durable(&h), before, "{label}");
    }

    // The initialization flushes at once and the committed decision verifies against the ledger.
    let s = d.decision(&update, 1, &batch.path, 0, &clock).unwrap();
    assert!(matches!(
        &s.due,
        Some(FlushDue::Urgent { requests }) if requests == &[Urgent::HwmChange]
    ));
    assert_eq!(advanced(&s), 0);
    let state = reusable(&h.remote, RUN).unwrap().unwrap();
    let decisions = state.graph.all::<HwmDecision>();
    assert_eq!(decisions.len(), 1);
    decisions[0]
        .value()
        .verify_against(&crate::ledger::Ledger::open(h.state.root(), "hwm"))
        .unwrap();
    assert_eq!(
        decisions[0].value().transition,
        crate::hwm::Transition::Initialized
    );
}
