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
use serde_json::Value;
use std::{
    collections::BTreeMap,
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

// ---------------------------------------------------------------------------
// Supervision summary
// ---------------------------------------------------------------------------

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

mod optimizer;
mod supervision;
mod training_tests;
