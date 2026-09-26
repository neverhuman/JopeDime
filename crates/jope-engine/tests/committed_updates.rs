#![cfg(feature = "neural")]
//! W6-T04: actual consumption and committed updates (CPU only).
//!
//! The trainer loop is simulated with the real sampler, the real CPU reference
//! AdamW, the real consumption journal and a tiny linear model, so that
//! interruption, rejection and recovery can be exercised without CUDA. This
//! proves the accounting and the determinism of the CPU reference path, not
//! GPU training.

use jope_parallel::{
    corpus::{Group, Index},
    manifest::{file_hash, read_json, write_json},
    neural::AdamW,
    training::{
        consumption::{self, Header, Journal},
        objective::{self, RowLabels},
        sample,
    },
};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};
use tch::{nn, Device, Kind, Tensor};

const MICRO: usize = 16;
const GLOBAL: usize = 32;
const SEED: u64 = 20260926;
const TARGET: u64 = 6;
const CHECKPOINT_EVERY: u64 = 2;
const DIGEST: &str = "1111111111111111111111111111111111111111111111111111111111111111";

// ---------------------------------------------------------------------------
// Reference optimizer: a rejected update changes nothing
// ---------------------------------------------------------------------------

fn snapshot(vs: &nn::VarStore) -> BTreeMap<String, Vec<f32>> {
    vs.variables()
        .into_iter()
        .map(|(name, p)| {
            (
                name,
                Vec::<f32>::try_from(p.detach().flatten(0, -1)).unwrap(),
            )
        })
        .collect()
}

fn moments(optimizer: &AdamW) -> BTreeMap<String, (Vec<f32>, Vec<f32>)> {
    optimizer
        .moments
        .iter()
        .map(|(name, (m, v))| {
            (
                name.clone(),
                (
                    Vec::<f32>::try_from(m.flatten(0, -1)).unwrap(),
                    Vec::<f32>::try_from(v.flatten(0, -1)).unwrap(),
                ),
            )
        })
        .collect()
}

#[test]
fn reference_optimizer_rejects_a_nonfinite_update_without_partial_change() {
    tch::set_num_threads(1);
    let vs = nn::VarStore::new(Device::Cpu);
    // BTreeMap order: "a" is updated before "b" is reached.
    let a = vs.root().var("a", &[4], nn::Init::Const(0.5));
    let b = vs.root().var("b", &[4], nn::Init::Const(-0.25));
    let mut optimizer = AdamW::new(1e-2, 0.01);
    (a.square().sum(Kind::Float) + b.square().sum(Kind::Float)).backward();
    optimizer.update_reference(&vs).unwrap();
    assert_eq!(optimizer.step, 1);
    let parameters = snapshot(&vs);
    let state = moments(&optimizer);
    // Finite gradient on "a", NaN on "b".
    (a.sum(Kind::Float) + (&b * f64::NAN).sum(Kind::Float)).backward();
    let error = optimizer.update_reference(&vs).unwrap_err();
    assert!(error.to_string().contains("nonfinite"), "{error}");
    assert_eq!(optimizer.step, 1, "a rejected update advanced the step");
    assert_eq!(
        snapshot(&vs),
        parameters,
        "a rejected update changed parameters"
    );
    assert_eq!(
        moments(&optimizer),
        state,
        "a rejected update changed moments"
    );
    // Recovery: clear the rejected gradients; the next valid update is step 2.
    for (_, mut p) in vs.variables() {
        p.zero_grad();
    }
    (a.square().sum(Kind::Float) + b.square().sum(Kind::Float)).backward();
    optimizer.update_reference(&vs).unwrap();
    assert_eq!(optimizer.step, 2);
    // A rejected first update leaves a fresh optimizer fresh.
    let fresh = nn::VarStore::new(Device::Cpu);
    let c = fresh.root().var("c", &[2], nn::Init::Const(1.0));
    let mut first = AdamW::new(1e-2, 0.01);
    (&c * f64::INFINITY).sum(Kind::Float).backward();
    assert!(first.update_reference(&fresh).is_err());
    assert_eq!(first.step, 0);
    assert!(first.moments.is_empty());
}

// ---------------------------------------------------------------------------
// Supervision summary
// ---------------------------------------------------------------------------

#[test]
fn supervision_summary_matches_an_independent_count() {
    // Two comparisons; one tie in the first, none in the second.
    let values = [
        0.9f32, 0.7, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.8, 0.75, 0.6, 0.55, 0.5, 0.3, 0.2, 0.1,
    ];
    let rows = (0..16)
        .map(|i| RowLabels {
            record: 100 + i,
            context: 3 + i / 8,
            value: values[i],
            percentile: 1.0 - (i % 8) as f32 / 7.0,
            runtime: 0.1,
        })
        .collect::<Vec<_>>();
    let lengths = (0..16).map(|i| 5 + (i * 7) % 20).collect::<Vec<_>>();
    let s = objective::supervision(&rows, &lengths).unwrap();
    let longest = *lengths.iter().max().unwrap();
    assert_eq!(s.rows, 16);
    assert_eq!(s.width, (longest + 7) / 8 * 8);
    assert_eq!(
        s.supervised_positions,
        lengths.iter().map(|l| l - 1).sum::<usize>()
    );
    let weights = rows
        .iter()
        .map(|r| (0.05f32 + r.percentile.powi(4)) as f64)
        .sum::<f64>();
    assert_eq!(s.row_weight_sum, weights);
    assert_eq!((s.ranking_pairs, s.tied_pairs), (55, 1));
    assert_eq!(s.listwise_groups, 2);
    // A one-token row supervises nothing and is refused.
    let mut short = lengths.clone();
    short[4] = 1;
    assert!(objective::supervision(&rows, &short).is_err());
}

// ---------------------------------------------------------------------------
// A deterministic CPU training loop around the real journal
// ---------------------------------------------------------------------------

/// Four training groups of sixteen records in two families. Record `16g + k`
/// has context `g`; `k = 0` is the group champion.
fn index() -> (Index, BTreeMap<String, Vec<usize>>) {
    let groups = (0..4)
        .map(|g| Group {
            context: g,
            family: format!("family{}", g % 2),
            partition: "meta_train".into(),
            records: (g * 16..g * 16 + 16).collect(),
            max_length: 12,
        })
        .collect::<Vec<_>>();
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    for (g, group) in groups.iter().enumerate() {
        families.entry(group.family.clone()).or_default().push(g);
    }
    let index = Index {
        schema: "parallel.teaching_corpus.v1".into(),
        qualification: true,
        vocabulary_sha256: "v".into(),
        context_index_sha256: "c".into(),
        records_sha256: "r".into(),
        record_bytes: 532,
        records: 64,
        legal_sets: vec![],
        contexts: vec![],
        labels_sha256: None,
        groups,
    };
    (index, families)
}

fn labels(record: usize) -> (RowLabels, usize) {
    let k = record % 16;
    (
        RowLabels {
            record,
            context: record / 16,
            value: 1.0 / (1.0 + k as f32 / 4.0),
            percentile: 1.0 - k as f32 / 15.0,
            runtime: 0.2f32.ln_1p(),
        },
        4 + record % 7,
    )
}

fn features(record: usize) -> [f32; 4] {
    let r = record as f32;
    [
        (r * 0.37).sin(),
        (r * 0.11).cos(),
        (record % 5) as f32 / 5.0,
        1.0,
    ]
}

fn target(record: usize) -> f32 {
    ((record * 3) % 7) as f32 / 7.0
}

struct Model {
    vs: nn::VarStore,
    w: Tensor,
    optimizer: AdamW,
}

fn fresh_model() -> Model {
    let vs = nn::VarStore::new(Device::Cpu);
    let w = vs.root().var("w", &[4], nn::Init::Const(0.1));
    Model {
        vs,
        w,
        optimizer: AdamW::new(5e-2, 0.01),
    }
}

fn load_model(checkpoint: &Path) -> Model {
    let mut model = fresh_model();
    model.vs.load(checkpoint.join("model.ot")).unwrap();
    model.optimizer = AdamW::load(&checkpoint.join("optimizer.ot"), Device::Cpu).unwrap();
    model
}

#[derive(Clone, Copy)]
enum Stop {
    /// Stop after the update of this optimizer step (0-based) was applied,
    /// with no further event: a hard interruption.
    AfterApplied(u64),
    /// Poison the gradient of this step: the update is rejected.
    RejectAt(u64),
}

struct Attempt {
    journal: PathBuf,
    applied_through: u64,
}

/// Run one attempt from the latest durable checkpoint (or fresh) up to
/// `TARGET`, committing every `CHECKPOINT_EVERY` updates.
fn attempt(run: &Path, number: usize, stop: Option<Stop>) -> Attempt {
    tch::set_num_threads(1);
    let (index, families) = index();
    let latest = run.join("latest.json");
    let (mut model, resume) = if latest.exists() {
        let latest: Value = read_json(&latest).unwrap();
        let checkpoint = run.join(latest["checkpoint"].as_str().unwrap());
        (
            load_model(&checkpoint),
            Some(file_hash(&checkpoint.join("state.json")).unwrap()),
        )
    } else {
        (fresh_model(), None)
    };
    let launch = run.join(format!("attempt-{number}"));
    fs::create_dir_all(&launch).unwrap();
    let header = Header {
        schema: consumption::SCHEMA.into(),
        rank: 0,
        world: 1,
        microbatch: MICRO,
        global_batch: GLOBAL,
        accumulation: GLOBAL / MICRO,
        seed: SEED,
        sampler_id: jope_parallel::training::SAMPLER_ID.into(),
        rng: consumption::RNG_ID.into(),
        sampling_offset: 0,
        initial_optimizer_step: model.optimizer.step,
        target_optimizer_step: TARGET,
        corpus_sha256: DIGEST.into(),
        replay_policy_sha256: None,
        resume_state_sha256: resume,
    };
    let path = launch.join("consumption-rank-0.jsonl");
    let mut journal = Journal::create(&path, header.clone()).unwrap();
    for step in model.optimizer.step..TARGET {
        for mb in 0..header.accumulation {
            tch::manual_seed(header.dropout_seed(step, mb).unwrap());
            let serial = header.serial(step, mb).unwrap();
            let ids = sample(&index, &families, serial as usize, MICRO, SEED);
            let x = Tensor::from_slice(&ids.iter().flat_map(|&r| features(r)).collect::<Vec<_>>())
                .reshape([MICRO as i64, 4]);
            let y = Tensor::from_slice(&ids.iter().map(|&r| target(r)).collect::<Vec<_>>());
            let loss =
                (x.matmul(&model.w) - y).square().mean(Kind::Float) / header.accumulation as f64;
            loss.backward();
            let (rows, lengths): (Vec<_>, Vec<_>) = ids.iter().map(|&r| labels(r)).unzip();
            journal
                .microbatch(
                    step,
                    mb,
                    serial,
                    header.dropout_seed(step, mb).unwrap(),
                    &ids,
                    objective::supervision(&rows, &lengths).unwrap(),
                )
                .unwrap();
        }
        if matches!(stop, Some(Stop::RejectAt(s)) if s == step) {
            tch::no_grad(|| {
                let _ = model.w.grad().fill_(f64::NAN);
            });
        }
        let norm = model.w.grad().norm().double_value(&[]);
        let before = model.optimizer.step;
        if let Err(error) = model.optimizer.update_reference(&model.vs) {
            assert_eq!(model.optimizer.step, before);
            journal.rejected(step, &format!("{error:#}")).unwrap();
            return Attempt {
                journal: path,
                applied_through: journal.applied_through(),
            };
        }
        journal.applied(step, norm).unwrap();
        let applied = model.optimizer.step;
        assert_eq!(applied, step + 1);
        if applied % CHECKPOINT_EVERY == 0 || applied == TARGET {
            let binding = journal.sync().unwrap();
            let checkpoint = run.join(format!("checkpoints/step-{applied}"));
            fs::create_dir_all(&checkpoint).unwrap();
            model.vs.save(checkpoint.join("model.ot")).unwrap();
            model
                .optimizer
                .save(&checkpoint.join("optimizer.ot"))
                .unwrap();
            write_json(
                &checkpoint.join("state.json"),
                &json!({"step":applied,"consumption_journal":binding,
                    "journal":path.strip_prefix(run).unwrap()}),
            )
            .unwrap();
            let state_sha = file_hash(&checkpoint.join("state.json")).unwrap();
            write_json(
                &run.join("latest.json"),
                &json!({"checkpoint":format!("checkpoints/step-{applied}"),
                    "step":applied,"state_sha256":state_sha}),
            )
            .unwrap();
            journal.commit(applied, &state_sha).unwrap();
        }
        if matches!(stop, Some(Stop::AfterApplied(s)) if s == step) {
            break;
        }
    }
    Attempt {
        journal: path,
        applied_through: journal.applied_through(),
    }
}

fn durable(run: &Path) -> Option<u64> {
    let latest = run.join("latest.json");
    latest.exists().then(|| {
        read_json::<Value>(&latest).unwrap()["step"]
            .as_u64()
            .unwrap()
    })
}

fn microbatches(journal: &Path) -> Vec<(u64, usize, Vec<u64>)> {
    fs::read_to_string(journal)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["event"] == "microbatch")
        .map(|event| {
            (
                event["step"].as_u64().unwrap(),
                event["microbatch"].as_u64().unwrap() as usize,
                serde_json::from_value(event["records"].clone()).unwrap(),
            )
        })
        .collect()
}

fn parameters(run: &Path) -> Vec<f32> {
    let latest: Value = read_json(&run.join("latest.json")).unwrap();
    let model = load_model(&run.join(latest["checkpoint"].as_str().unwrap()));
    Vec::try_from(model.w.detach()).unwrap()
}

fn sampler(serial: usize, count: usize) -> Vec<usize> {
    let (index, families) = index();
    sample(&index, &families, serial, count, SEED)
}

#[test]
fn interrupted_and_uninterrupted_runs_consume_the_same_records_and_commit_the_same_state() {
    let base = tempfile::tempdir().unwrap();
    // Uninterrupted reference run.
    let straight = base.path().join("straight");
    let reference = attempt(&straight, 0, None);
    let reference_bytes = fs::read(&reference.journal).unwrap();
    let whole = consumption::reconcile(&reference_bytes, durable(&straight)).unwrap();
    assert_eq!(whole.committed_updates, TARGET);
    assert_eq!(whole.applied_updates, TARGET);
    assert_eq!(whole.uncommitted_applied_updates, 0);
    assert_eq!(whole.commit_events, TARGET / CHECKPOINT_EVERY);
    assert_eq!(whole.computed.microbatches, 2 * TARGET);
    assert_eq!(whole.committed.presentations, TARGET * GLOBAL as u64);
    assert_eq!(
        consumption::verify_records(&reference_bytes, sampler).unwrap(),
        2 * TARGET
    );

    // Interrupted run: step 4's update is applied but never committed.
    let broken = base.path().join("broken");
    let first = attempt(&broken, 1, Some(Stop::AfterApplied(4)));
    assert_eq!(first.applied_through, 5);
    let first_bytes = fs::read(&first.journal).unwrap();
    let one = consumption::reconcile(&first_bytes, durable(&broken)).unwrap();
    assert_eq!(
        (
            one.applied_updates,
            one.committed_updates,
            one.uncommitted_applied_updates
        ),
        (5, 4, 1)
    );
    assert_eq!(one.committed_through, 4);
    // The receipt alone reconstructs the next sample sequence.
    assert_eq!(one.next.optimizer_step, 4);
    assert_eq!(one.next.serial, 4 * GLOBAL as u64);
    let next = sampler(one.next.serial as usize, MICRO);

    // Recovery from the durable checkpoint.
    let second = attempt(&broken, 2, None);
    let second_bytes = fs::read(&second.journal).unwrap();
    let two = consumption::reconcile(&second_bytes, durable(&broken)).unwrap();
    let resumed = microbatches(&second.journal);
    assert_eq!(resumed[0].0, 4);
    assert_eq!(
        resumed[0].2,
        next.iter().map(|&r| r as u64).collect::<Vec<_>>(),
        "recovery did not consume the reconstructed next records"
    );
    let reference_mbs = microbatches(&reference.journal);
    let interrupted_mbs = microbatches(&first.journal)
        .into_iter()
        .filter(|(step, _, _)| *step < 4)
        .chain(resumed)
        .collect::<Vec<_>>();
    assert_eq!(
        interrupted_mbs, reference_mbs,
        "committed consumption differs from the uninterrupted run"
    );
    assert_eq!(
        parameters(&broken),
        parameters(&straight),
        "interrupted run reached a different committed state"
    );

    // Run totals keep computation, applied and committed updates apart.
    let run = consumption::reconcile_run(&[one.clone(), two.clone()]).unwrap();
    assert_eq!(run.committed_updates, TARGET);
    assert_eq!(run.committed_through, TARGET);
    assert_eq!(run.applied_updates, TARGET + 1);
    assert_eq!(run.lost_applied_updates, 1);
    assert_eq!(run.computed.microbatches, 2 * (TARGET + 1));
    assert_eq!(run.committed.microbatches, 2 * TARGET);
    assert_eq!(run.committed, whole.committed);
    assert_eq!(run.unique_committed_records, whole.unique_committed_records);
    assert!(run.unique_committed_records < run.committed.presentations);
    // Attempts must chain at the durable commit.
    assert!(consumption::reconcile_run(&[two.clone(), one.clone()]).is_err());
    let mut fork = two.clone();
    fork.header.seed ^= 1;
    assert!(consumption::reconcile_run(&[one, fork]).is_err());

    // Every checkpoint binds a byte-exact journal prefix that still verifies.
    for step in [2u64, 4] {
        let state: Value =
            read_json(&broken.join(format!("checkpoints/step-{step}/state.json"))).unwrap();
        let binding: consumption::Binding =
            serde_json::from_value(state["consumption_journal"].clone()).unwrap();
        consumption::verify_binding(&first_bytes, &binding).unwrap();
        let mut altered = first_bytes.clone();
        altered[40] ^= 1;
        assert!(consumption::verify_binding(&altered, &binding).is_err());
    }
}

#[test]
fn a_rejected_update_commits_nothing_and_ends_the_attempt() {
    let base = tempfile::tempdir().unwrap();
    let run = base.path().join("run");
    let rejected = attempt(&run, 0, Some(Stop::RejectAt(3)));
    assert_eq!(rejected.applied_through, 3);
    let bytes = fs::read(&rejected.journal).unwrap();
    let r = consumption::reconcile(&bytes, durable(&run)).unwrap();
    assert!(r.ended_by_rejection);
    assert_eq!(
        (r.attempted_updates, r.applied_updates, r.rejected_updates),
        (4, 3, 1)
    );
    assert_eq!((r.committed_through, r.committed_updates), (2, 2));
    assert_eq!(r.uncommitted_applied_updates, 1);
    // The rejected step's microbatches were computed but never applied.
    assert_eq!(r.computed.microbatches, 8);
    assert_eq!(r.committed.microbatches, 4);
    assert_eq!(r.next.optimizer_step, 2);
    // The durable checkpoint still holds step 2; nothing claims step 4.
    assert_eq!(durable(&run), Some(2));
    // A reader cannot be persuaded that the rejected step was applied.
    let mut forged = bytes.clone();
    forged.extend_from_slice(
        b"{\"event\":\"update\",\"step\":3,\"outcome\":\"applied\",\"microbatches\":2,\
          \"gradient_norm\":0.5,\"clipped\":false,\"reason\":null}\n",
    );
    assert!(consumption::reconcile(&forged, None).is_err());
}

#[test]
fn the_writer_refuses_out_of_order_or_unsupported_accounting() {
    let base = tempfile::tempdir().unwrap();
    let header = |initial: u64| Header {
        schema: consumption::SCHEMA.into(),
        rank: 0,
        world: 1,
        microbatch: MICRO,
        global_batch: GLOBAL,
        accumulation: 2,
        seed: SEED,
        sampler_id: "fixture".into(),
        rng: consumption::RNG_ID.into(),
        sampling_offset: 0,
        initial_optimizer_step: initial,
        target_optimizer_step: 4,
        corpus_sha256: DIGEST.into(),
        replay_policy_sha256: None,
        resume_state_sha256: None,
    };
    let ids = sampler(0, MICRO);
    let supervision = {
        let (rows, lengths): (Vec<_>, Vec<_>) = ids.iter().map(|&r| labels(r)).unzip();
        objective::supervision(&rows, &lengths).unwrap()
    };
    let path = base.path().join("journal.jsonl");
    let mut journal = Journal::create(&path, header(0)).unwrap();
    // Wrong cursor, wrong RNG identity, wrong microbatch index.
    let seed0 = consumption::dropout_seed(SEED, 0, 0, 0);
    assert!(journal
        .microbatch(0, 0, 8, seed0, &ids, supervision.clone())
        .is_err());
    assert!(journal
        .microbatch(0, 0, 0, seed0 ^ 1, &ids, supervision.clone())
        .is_err());
    assert!(journal
        .microbatch(0, 1, 16, seed0, &ids, supervision.clone())
        .is_err());
    journal
        .microbatch(0, 0, 0, seed0, &ids, supervision.clone())
        .unwrap();
    // An update before every accumulation microbatch, a commit of an
    // unapplied update, and a commit inside an unfinished update.
    assert!(journal.applied(0, 0.5).is_err());
    assert!(journal.commit(1, DIGEST).is_err());
    let ids1 = sampler(16, MICRO);
    journal
        .microbatch(
            0,
            1,
            16,
            consumption::dropout_seed(SEED, 0, 0, 1),
            &ids1,
            supervision.clone(),
        )
        .unwrap();
    assert!(journal.applied(0, f64::NAN).is_err());
    journal.applied(0, 1.5).unwrap();
    assert!(journal.commit(1, "not-a-digest").is_err());
    journal.commit(1, DIGEST).unwrap();
    // Commits must advance.
    assert!(journal.commit(1, DIGEST).is_err());
    // A journal cannot be created twice for one attempt.
    assert!(Journal::create(&path, header(0)).is_err());
    // Header validation: initial beyond target, partition mismatch.
    let mut bad = header(5);
    assert!(Journal::create(&base.path().join("bad-1.jsonl"), bad.clone()).is_err());
    bad = header(0);
    bad.accumulation = 3;
    assert!(Journal::create(&base.path().join("bad-2.jsonl"), bad).is_err());
}

#[test]
fn journal_ingestion_is_idempotent_and_refuses_inconsistent_history() {
    let base = tempfile::tempdir().unwrap();
    let run = base.path().join("run");
    let first = attempt(&run, 1, Some(Stop::AfterApplied(4)));
    let bytes = fs::read(&first.journal).unwrap();
    let text = String::from_utf8(bytes.clone()).unwrap();
    let lines = text.lines().collect::<Vec<_>>();
    let baseline = consumption::reconcile(&bytes, Some(4)).unwrap();

    // Exact duplicate ingestion leaves every total unchanged.
    let mut duplicated = bytes.clone();
    for line in [lines[1], lines[3], lines[lines.len() - 1]] {
        duplicated.extend_from_slice(line.as_bytes());
        duplicated.push(b'\n');
    }
    let again = consumption::reconcile(&duplicated, Some(4)).unwrap();
    assert_eq!(again.duplicate_events_ignored, 3);
    assert_eq!(
        (
            again.computed,
            again.committed,
            again.applied_updates,
            again.committed_records.clone()
        ),
        (
            baseline.computed,
            baseline.committed,
            baseline.applied_updates,
            baseline.committed_records.clone()
        )
    );
    // A conflicting duplicate is refused.
    let mut conflicting = bytes.clone();
    let mut event: Value = serde_json::from_str(lines[1]).unwrap();
    event["records"][0] = json!(event["records"][1].clone());
    conflicting.extend_from_slice(serde_json::to_string(&event).unwrap().as_bytes());
    conflicting.push(b'\n');
    assert!(consumption::reconcile(&conflicting, Some(4)).is_err());

    // A torn final line (hard kill mid-write) is reported and not counted.
    let mut torn = bytes.clone();
    torn.extend_from_slice(&lines[1].as_bytes()[..30]);
    let t = consumption::reconcile(&torn, Some(4)).unwrap();
    assert!(t.torn_tail);
    assert_eq!(t.computed, baseline.computed);

    // An interrupted step's microbatches are unfinished work, not an update.
    let cut = lines[..lines.len() - 1].join("\n") + "\n";
    let partial = consumption::reconcile(cut.as_bytes(), Some(4)).unwrap();
    assert_eq!(partial.unfinished_step, Some(4));
    assert_eq!(partial.unfinished.microbatches, 2);
    assert_eq!(partial.applied_updates, 4);

    // Tampered cursor, RNG identity or records are refused.
    for (field, value) in [
        ("serial", json!(1)),
        ("dropout_seed", json!(7)),
        ("records", json!(vec![0; MICRO])),
    ] {
        let mut altered = lines
            .clone()
            .into_iter()
            .map(String::from)
            .collect::<Vec<_>>();
        let mut event: Value = serde_json::from_str(&altered[1]).unwrap();
        event[field] = value;
        altered[1] = serde_json::to_string(&event).unwrap();
        let joined = altered.join("\n") + "\n";
        let refused = consumption::reconcile(joined.as_bytes(), Some(4)).is_err()
            || consumption::verify_records(joined.as_bytes(), sampler).is_err();
        assert!(refused, "{field} tampering was accepted");
    }
    // Unknown fields, a missing header and an unknown schema are refused.
    let mut unknown: Value = serde_json::from_str(lines[1]).unwrap();
    unknown["weight"] = json!(2.0);
    let mut altered = lines.iter().map(|l| l.to_string()).collect::<Vec<_>>();
    altered[1] = unknown.to_string();
    assert!(consumption::reconcile((altered.join("\n") + "\n").as_bytes(), None).is_err());
    let headless = lines[1..].join("\n") + "\n";
    assert!(consumption::reconcile(headless.as_bytes(), None).is_err());

    // Durable state versus commit events.
    assert!(
        consumption::reconcile(&bytes, Some(2)).is_err(),
        "journal commits beyond the durable checkpoint were accepted"
    );
    assert!(
        consumption::reconcile(&bytes, Some(6)).is_err(),
        "a durable step beyond applied updates was accepted"
    );
    // Crash between checkpoint publication and the journal commit event.
    let without_last_commit = lines
        .iter()
        .filter(|line| !line.contains("\"event\":\"commit\",\"step\":4"))
        .copied()
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let missing = consumption::reconcile(without_last_commit.as_bytes(), Some(4)).unwrap();
    assert!(missing.commit_event_missing);
    assert_eq!(missing.committed_through, 4);
    let unknown_durable = consumption::reconcile(without_last_commit.as_bytes(), None).unwrap();
    assert_eq!(unknown_durable.committed_through, 2);
    assert_eq!(unknown_durable.uncommitted_applied_updates, 3);
    let _: BTreeSet<u64> = missing.committed_records;
}
