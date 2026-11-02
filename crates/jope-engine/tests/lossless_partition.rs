#![cfg(feature = "neural")]
//! W7-T06: the lossless multi-rank partition (CPU only, in-process ranks).
//!
//! Each simulated rank gets its own copy of the policy outputs for the whole
//! global batch and backpropagates only its own microbatches, so its output
//! gradients are its rank gradient; summing (or averaging) them across ranks
//! is what the all-reduce delivers. Parameter gradients are linear in these
//! output gradients, so equality here is equality of the parameter update up
//! to the same rounding.
//!
//! The declared single-process reference is the T02-proven v1 composite over
//! the whole global batch as one microbatch. No device, NCCL, dropout or
//! training run.

use jope_parallel::{
    corpus::{Group, Index},
    manifest::hash,
    neural::Config,
    task_contract::training_partition,
    training::{
        consumption,
        continuation::{check_parent, check_parent_partition, Current, Mode},
        objective::{self, composite, Outputs, RowLabels},
        partition::{
            self, denominators, empty_rank_objective, gradient_within_policy, loss_within_policy,
            CommittedUpdate, Denominators, Plan, State, Transport,
        },
        sample,
    },
};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use tch::{Kind, Tensor};

const T: i64 = 6;
const V: i64 = 7;
const SEED: u64 = 7;

/// `groups` complete comparisons of varying length (3..=6 tokens, PAD beyond)
/// with legal sets that vary by row and position. Values tie within pairs of
/// comparators and fully in the first `tied_groups` comparisons; percentile
/// 1.0 on every champion makes boosted pairs.
struct Batch {
    rows: Vec<RowLabels>,
    tokens: Tensor,
    legal: Tensor,
    logits: Vec<f64>,
    rewards: Vec<f64>,
    runtime: Vec<f64>,
    stability: Vec<f64>,
    ablation: Vec<f64>,
}

fn batch(groups: usize, tied_groups: usize) -> Batch {
    let n = groups * 8;
    let mut tokens = vec![0i64; n * T as usize];
    let mut legal = vec![0u8; n * (T as usize - 1) * V as usize];
    for i in 0..n {
        let length = 3 + i % 4;
        tokens[i * T as usize] = 1;
        for j in 1..length - 1 {
            tokens[i * T as usize + j] = 3 + ((i + j) % 4) as i64;
        }
        tokens[i * T as usize + length - 1] = 2;
        for j in 0..T as usize - 1 {
            let x = tokens[i * T as usize + j + 1] as usize;
            let set: Vec<usize> = match x {
                0 => vec![0],
                2 => vec![2],
                _ => (3..(4 + (i + j) % 4)).chain([x]).collect(),
            };
            for k in set {
                legal[(i * (T as usize - 1) + j) * V as usize + k] = 1;
            }
        }
    }
    let rows = (0..n)
        .map(|i| {
            let k = i % 8;
            RowLabels {
                record: i,
                context: i / 8,
                value: if i / 8 < tied_groups {
                    0.5
                } else {
                    1.0 / (1.0 + (k / 2) as f32 + (i / 8) as f32 * 0.1)
                },
                percentile: 1.0 - k as f32 / 7.0,
                runtime: ((i % 5) as f32 * 0.3).ln_1p(),
            }
        })
        .collect();
    let wave = |len: usize, a: f64, b: f64| {
        (0..len)
            .map(|n| ((n as f64 * a).sin() + (n as f64 * b).cos()) * 0.5)
            .collect::<Vec<_>>()
    };
    Batch {
        rows,
        tokens: Tensor::from_slice(&tokens).reshape([n as i64, T]),
        legal: Tensor::from_slice(&legal)
            .reshape([n as i64, T - 1, V])
            .to_kind(Kind::Bool),
        logits: wave(n * (T as usize + 1) * V as usize, 0.37, 0.11),
        rewards: wave(n * 4, 0.53, 0.29),
        runtime: (0..n).map(|i| 0.1 + (i % 7) as f64 * 0.4).collect(),
        stability: wave(n, 0.7, 0.3),
        ablation: wave(n, 0.41, 0.17),
    }
}

struct Leaves {
    logits: Tensor,
    rewards: Tensor,
    runtime: Tensor,
    stability: Tensor,
    ablation: Tensor,
}

impl Leaves {
    fn all(&self) -> [&Tensor; 5] {
        [
            &self.logits,
            &self.rewards,
            &self.runtime,
            &self.stability,
            &self.ablation,
        ]
    }
}

const OUTPUTS: [&str; 5] = ["logits", "rewards", "runtime", "stability", "ablation"];

fn leaf(values: &[f64], shape: &[i64]) -> Tensor {
    Tensor::from_slice(&values.iter().map(|&x| x as f32).collect::<Vec<_>>())
        .reshape(shape)
        .set_requires_grad(true)
}

fn leaves(b: &Batch) -> Leaves {
    let n = b.rows.len() as i64;
    Leaves {
        logits: leaf(&b.logits, &[n, T + 1, V]),
        rewards: leaf(&b.rewards, &[n, 4]),
        runtime: leaf(&b.runtime, &[n]),
        stability: leaf(&b.stability, &[n]),
        ablation: leaf(&b.ablation, &[n]),
    }
}

fn host(t: &Tensor) -> Vec<f64> {
    Vec::<f64>::try_from(t.to_kind(Kind::Double).flatten(0, -1)).unwrap()
}

fn grads(l: &Leaves) -> Vec<Vec<f64>> {
    l.all()
        .iter()
        .map(|t| {
            let g = t.grad();
            assert!(g.defined(), "every output must have a defined gradient");
            host(&g)
        })
        .collect()
}

/// The v1 composite (`None`) or the v2 composite (`Some(global)`) over rows
/// `first_row..first_row + rows`.
fn objective_of(
    b: &Batch,
    l: &Leaves,
    first_row: i64,
    rows: i64,
    global: Option<&Denominators>,
) -> anyhow::Result<(Tensor, [f64; 5])> {
    let slice = |t: &Tensor| t.narrow(0, first_row, rows);
    let (logits, rewards, runtime, stability, ablation) = (
        slice(&l.logits),
        slice(&l.rewards),
        slice(&l.runtime),
        slice(&l.stability),
        slice(&l.ablation),
    );
    let outputs = Outputs {
        logits: &logits,
        rewards: &rewards,
        runtime: &runtime,
        stability: &stability,
        ablation: &ablation,
    };
    let (tokens, legal) = (slice(&b.tokens), slice(&b.legal));
    let labels = &b.rows[first_row as usize..(first_row + rows) as usize];
    match global {
        None => composite(
            &outputs,
            std::iter::empty(),
            &tokens,
            &tokens.ne(0),
            &legal,
            labels,
        ),
        Some(global) => partition::composite(
            &outputs,
            std::iter::empty(),
            &tokens,
            &tokens.ne(0),
            &legal,
            labels,
            global,
        ),
    }
}

fn v1(b: &Batch, l: &Leaves, first_row: i64, rows: i64) -> (Tensor, [f64; 5]) {
    objective_of(b, l, first_row, rows, None).unwrap()
}

struct Run {
    loss: f64,
    terms: [f64; 5],
    grads: Vec<Vec<f64>>,
}

/// The declared single-process reference.
fn reference(b: &Batch) -> Run {
    let l = leaves(b);
    let (loss, terms) = v1(b, &l, 0, b.rows.len() as i64);
    loss.backward();
    Run {
        loss: loss.double_value(&[]),
        terms,
        grads: grads(&l),
    }
}

/// One simulated rank at local step 0: its objective scaled by `scale`, its
/// term contributions and its own output gradients.
fn run_rank(b: &Batch, plan: &Plan, rank: usize, scale: f64) -> Run {
    let l = leaves(b);
    let global = denominators(&b.rows).unwrap();
    let slots = plan.slots(rank, SEED, 0).unwrap();
    let (mut loss, mut terms) = (0.0, [0.0; 5]);
    if slots.is_empty() {
        empty_rank_objective(l.all()).unwrap().backward();
    }
    for slot in &slots {
        assert_eq!(slot.serial, (slot.first_comparison * 8) as u64);
        let (objective, contributions) =
            objective_of(b, &l, slot.serial as i64, slot.rows as i64, Some(&global)).unwrap();
        let objective = objective * scale;
        loss += objective.double_value(&[]);
        for i in 0..5 {
            terms[i] += contributions[i] * scale;
        }
        objective.backward();
    }
    Run {
        loss,
        terms,
        grads: grads(&l),
    }
}

/// Every rank of `plan`, reduced by `transport`.
fn run_plan(b: &Batch, plan: &Plan, transport: Transport, scale: f64) -> Run {
    let runs = (0..plan.world())
        .map(|rank| run_rank(b, plan, rank, scale))
        .collect::<Vec<_>>();
    let divide = match transport {
        Transport::Sum => 1.0,
        Transport::Mean => plan.world() as f64,
    };
    let mut total = Run {
        loss: 0.0,
        terms: [0.0; 5],
        grads: runs[0].grads.iter().map(|g| vec![0.0; g.len()]).collect(),
    };
    for run in &runs {
        total.loss += run.loss / divide;
        for i in 0..5 {
            total.terms[i] += run.terms[i] / divide;
        }
        for (sum, g) in total.grads.iter_mut().zip(&run.grads) {
            for (s, x) in sum.iter_mut().zip(g) {
                *s += x / divide;
            }
        }
    }
    total
}

fn max_relative_gradient_error(actual: &[f64], reference: &[f64]) -> f64 {
    let scale = reference.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let error = actual
        .iter()
        .zip(reference)
        .fold(0.0f64, |m, (a, r)| m.max((a - r).abs()));
    if scale == 0.0 {
        error
    } else {
        error / scale
    }
}

fn within_policy(label: &str, run: &Run, reference: &Run) -> bool {
    let mut ok = loss_within_policy(run.loss, reference.loss);
    let mut worst = 0.0f64;
    for i in 0..5 {
        ok &= gradient_within_policy(&run.grads[i], &reference.grads[i]);
        ok &= loss_within_policy(run.terms[i], reference.terms[i]);
        worst = worst.max(max_relative_gradient_error(
            &run.grads[i],
            &reference.grads[i],
        ));
    }
    println!(
        "{label}: loss {:.9} vs {:.9} (|Δ| {:.2e}), worst output-gradient error {:.2e} of max |g|, within policy {ok}",
        run.loss,
        reference.loss,
        (run.loss - reference.loss).abs(),
        worst
    );
    ok
}

fn assert_within_policy(label: &str, run: &Run, reference: &Run) {
    assert!(
        within_policy(label, run, reference),
        "{label} is outside the frozen numerical policy"
    );
    for (i, name) in OUTPUTS.iter().enumerate() {
        assert!(
            gradient_within_policy(&run.grads[i], &reference.grads[i]),
            "{label}: {name} gradient outside policy"
        );
    }
}

fn index() -> (Index, BTreeMap<String, Vec<usize>>) {
    let groups = (0..6)
        .map(|i| Group {
            context: i,
            family: format!("family{}", i / 2),
            partition: "meta_train".into(),
            records: (i * 12..i * 12 + 12).collect(),
            max_length: 8,
        })
        .collect::<Vec<_>>();
    let mut families = BTreeMap::<String, Vec<usize>>::new();
    for (i, g) in groups.iter().enumerate() {
        families.entry(g.family.clone()).or_default().push(i);
    }
    (
        Index {
            schema: "parallel.teaching_corpus.v1".into(),
            qualification: true,
            vocabulary_sha256: "v".into(),
            context_index_sha256: "c".into(),
            records_sha256: "r".into(),
            record_bytes: 532,
            records: 72,
            legal_sets: vec![],
            contexts: vec![],
            labels_sha256: None,
            groups,
        },
        families,
    )
}

fn update(plan: &Plan, b: &Batch, step: u64, local_step: u64) -> anyhow::Result<CommittedUpdate> {
    let run = run_plan(b, plan, Transport::Sum, 1.0);
    CommittedUpdate::new(
        plan,
        step,
        local_step,
        &denominators(&b.rows)?,
        run.terms,
        &Tensor::from_slice(&[0.5f32, -0.25, 0.125]),
        0.57,
        &"b".repeat(64),
    )
}

fn config() -> Value {
    serde_json::to_value(Config::regression(401)).unwrap()
}

fn parent(execution: Value, identity: Value) -> Value {
    let world = execution["world"].clone();
    let digest = objective::identity_digest(&identity);
    json!({"schema":"parallel.checkpoint.v1","step":3000,"world":world,"global_batch":2048,
        "corpus_sha256":CORPUS,"vocabulary_sha256":VOCABULARY,"seed":SEED,"sampler_id":SAMPLER,
        "ranking_backward":"dense_pairwise_v2","runtime_objective":"huber_log1p_seconds_v2",
        "execution_partition":execution,"objective_sha256":digest,"objective":identity,
        "initialization":{"kind":"fresh","continuation_mode":"cold_initialization"}})
}

fn check(
    mode: Mode,
    parent: &Value,
    world: usize,
    partition: &Value,
    policy: Option<&Value>,
) -> anyhow::Result<Value> {
    let config = config();
    let digest = partition::objective_sha256();
    check_parent_partition(
        mode,
        parent,
        &config,
        &Current {
            corpus_sha256: CORPUS,
            vocabulary_sha256: VOCABULARY,
            world,
            microbatch: 256,
            global_batch: 2048,
            seed: SEED,
            sampler_id: SAMPLER,
            config: &config,
            objective_sha256: &digest,
        },
        partition,
        policy,
    )
}

#[path = "lossless_partition/objective.rs"]
mod objective;
#[path = "lossless_partition/partition.rs"]
mod partition;
#[path = "lossless_partition/rank_draw.rs"]
mod rank_draw;
#[path = "lossless_partition/resume.rs"]
mod resume;
#[path = "lossless_partition/transition.rs"]
mod transition;
