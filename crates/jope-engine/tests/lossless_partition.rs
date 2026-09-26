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

#[test]
fn partition_v1_cannot_form_three_ranks_and_averages_unequal_local_means() {
    // v1 requires global_batch % (world × microbatch) == 0 and a world in
    // {1, 2, 4, 8}: no three-rank partition of the 2048-row batch exists.
    for micro in [64, 128, 256] {
        assert!(training_partition(3, micro, 2048).is_err());
    }
    // Its normalization averages microbatch-local means. With unequal shards
    // (7 comparisons over three ranks: 3, 2, 2) that is a different objective
    // from the single-process one.
    let b = batch(7, 0);
    let reference = v1(&b, &leaves(&b), 0, 56).0.double_value(&[]);
    let l = leaves(&b);
    let averaged = [(0, 24), (24, 16), (40, 16)]
        .iter()
        .map(|&(first, rows)| v1(&b, &l, first, rows).0.double_value(&[]))
        .sum::<f64>()
        / 3.0;
    assert!(
        (averaged - reference).abs() > 1e-3,
        "unequal local means happened to agree: {averaged} vs {reference}"
    );
}

#[test]
fn assignment_is_lossless_deterministic_and_keeps_comparisons_whole() {
    for groups in 0..=40 {
        for world in 1..=9 {
            for micro in 1..=5 {
                let shards = partition::assign(groups, world, micro).unwrap();
                assert_eq!(shards, partition::assign(groups, world, micro).unwrap());
                assert_eq!(shards.len(), world);
                let mut next = 0;
                for (rank, shard) in shards.iter().enumerate() {
                    assert_eq!(shard.rank, rank);
                    assert_eq!(shard.first_group, next, "contiguous, in rank order");
                    assert_eq!(
                        shard.groups,
                        groups / world + usize::from(rank < groups % world)
                    );
                    assert_eq!(shard.microbatches.iter().sum::<usize>(), shard.groups);
                    assert!(shard.microbatches.iter().all(|&m| (1..=micro).contains(&m)));
                    next += shard.groups;
                }
                assert_eq!(next, groups, "no comparison dropped or repeated");
            }
        }
    }
    // The three-rank partition of the 2048-row batch with 256-row microbatches.
    let plan = Plan::new(3, 2048, 256).unwrap();
    let shards = plan.shards();
    assert_eq!(
        shards.iter().map(|s| s.groups).collect::<Vec<_>>(),
        [86, 85, 85]
    );
    assert_eq!(shards[0].microbatches, [32, 32, 22]);
    assert_eq!(shards[1].microbatches, [32, 32, 21]);
    assert_eq!(shards[2].first_group, 171);
    let record = plan.record();
    assert_eq!(record["schema"], partition::SCHEMA);
    assert_eq!(record["world"], 3);
    assert_eq!(record["normalization"], partition::NORMALIZATION);
    assert_eq!(
        record["numerical_policy"]["loss_rtol"],
        partition::LOSS_RTOL
    );
    assert_eq!(
        record["numerical_policy"]["gradient_rtol_of_max_abs"],
        partition::GRADIENT_RTOL
    );
    assert_eq!(record, Plan::new(3, 2048, 256).unwrap().record());
    // Malformed plans are refused.
    for (world, global, micro) in [
        (0, 2048, 256),
        (3, 0, 256),
        (3, 2044, 256),
        (3, 2048, 0),
        (3, 2048, 12),
    ] {
        assert!(
            Plan::new(world, global, micro).is_err(),
            "{world} {global} {micro}"
        );
    }
}

#[test]
fn every_partition_matches_the_single_process_reference() {
    // Uneven groups, nondivisible batches, masks, ties, boosted pairs and a
    // rank whose first microbatches are all ties (zero local pair weight).
    for (label, b) in [
        ("7 comparisons, 2 tied", batch(7, 2)),
        ("11 comparisons", batch(11, 0)),
    ] {
        let reference = reference(&b);
        let groups = b.rows.len() / 8;
        for (world, micro) in [
            (1, 1),
            (1, 2),
            (1, 64),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 3),
            (4, 2),
            (5, 1),
            (7, 1),
            (8, 1),
            (9, 2),
            (12, 1),
        ] {
            let plan = Plan::new(world, groups * 8, micro * 8).unwrap();
            let run = run_plan(&b, &plan, Transport::Sum, 1.0);
            assert_within_policy(
                &format!("{label}: world {world}, microbatch {micro} comparisons"),
                &run,
                &reference,
            );
        }
    }
}

#[test]
fn a_mean_transport_needs_the_world_scale() {
    let b = batch(11, 0);
    let reference = reference(&b);
    let plan = Plan::new(3, 88, 16).unwrap();
    assert_eq!(plan.local_scale(Transport::Sum), 1.0);
    assert_eq!(plan.local_scale(Transport::Mean), 3.0);
    let scaled = run_plan(
        &b,
        &plan,
        Transport::Mean,
        plan.local_scale(Transport::Mean),
    );
    assert_within_policy("mean transport, scaled by world", &scaled, &reference);
    // Negative control: without the scale a mean transport trains on L / W.
    let unscaled = run_plan(&b, &plan, Transport::Mean, 1.0);
    assert!(!within_policy(
        "mean transport, unscaled",
        &unscaled,
        &reference
    ));
    assert!((unscaled.loss * 3.0 - reference.loss).abs() < 1e-4);
}

#[test]
fn globally_empty_terms_are_zero_and_empty_ranks_contribute_zero_gradients() {
    // Every pair ties: both Bradley–Terry terms are globally empty.
    let b = batch(5, 5);
    let global = denominators(&b.rows).unwrap();
    assert_eq!(global.pair_weight, 0.0);
    assert_eq!(global.empty_terms(), ["ranking_bt", "contextfree_bt"]);
    let reference = reference(&b);
    assert_eq!(reference.terms[1], 0.0);
    assert_eq!(reference.terms[4], 0.0);
    assert!(reference.grads[4].iter().all(|&g| g == 0.0));
    for (world, micro) in [(1, 5), (3, 1), (5, 1), (7, 1)] {
        let plan = Plan::new(world, 40, micro * 8).unwrap();
        let run = run_plan(&b, &plan, Transport::Sum, 1.0);
        assert_eq!(run.terms[1], 0.0, "a globally empty term has value 0");
        assert_eq!(run.terms[4], 0.0);
        assert!(run.grads[4].iter().all(|&g| g == 0.0), "and gradient 0");
        assert_within_policy(&format!("all ties, world {world}"), &run, &reference);
    }
    // More ranks than comparisons: ranks 7 and 8 are empty, explicitly.
    let b = batch(7, 2);
    let plan = Plan::new(9, 56, 8).unwrap();
    assert_eq!(
        plan.record()["shards"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["groups"].as_u64().unwrap())
            .collect::<Vec<_>>(),
        [1, 1, 1, 1, 1, 1, 1, 0, 0]
    );
    for rank in [7, 8] {
        assert!(plan.slots(rank, SEED, 0).unwrap().is_empty());
        let run = run_rank(&b, &plan, rank, 1.0);
        assert_eq!(run.loss, 0.0);
        assert_eq!(run.terms, [0.0; 5]);
        for (name, g) in OUTPUTS.iter().zip(&run.grads) {
            assert!(
                g.iter().all(|&x| x == 0.0),
                "empty rank {rank} has a nonzero {name} gradient"
            );
        }
    }
    assert!(empty_rank_objective(std::iter::empty()).is_err());
}

#[test]
fn term_denominators_are_global_and_must_cover_the_microbatch() {
    let b = batch(7, 2);
    let global = denominators(&b.rows).unwrap();
    assert_eq!(global.listwise_groups, 7);
    assert_eq!(global.runtime_rows, 56);
    // Tied comparisons 0 and 1 carry row weight but no pair weight.
    let tied = denominators(&b.rows[..16]).unwrap();
    assert_eq!(tied.pair_weight, 0.0);
    let fractions = tied.fractions_of(&global).unwrap();
    assert!(fractions[0] > 0.0 && fractions[2] == 2.0 / 7.0 && fractions[3] == 16.0 / 56.0);
    assert_eq!((fractions[1], fractions[4]), (0.0, 0.0));
    // Term-specific shares: every comparison has the same percentiles, so its
    // row-weight share is 1/7, but its pair-weight share is not 1/5 of the
    // five untied comparisons, because |Δv| shrinks with the comparison index.
    let untied = denominators(&b.rows[16..24]).unwrap();
    let f = untied.fractions_of(&global).unwrap();
    assert!((f[0] - 1.0 / 7.0).abs() < 1e-12, "{}", f[0]);
    assert!(
        f[1] - 1.0 / 5.0 > 1e-2,
        "pair share {} is a weight share",
        f[1]
    );
    assert_eq!(f[1], f[4], "the ablation term shares the pair denominator");
    // A microbatch that the global denominators do not cover is refused.
    let smaller = denominators(&b.rows[..24]).unwrap();
    let whole = denominators(&b.rows).unwrap();
    assert!(whole.fractions_of(&smaller).is_err());
    let l = leaves(&b);
    let error = objective_of(&b, &l, 24, 32, Some(&smaller)).unwrap_err();
    assert!(error.to_string().contains("exceeds the global"), "{error}");
    // Pair weight where the global batch has none is refused.
    let all_tied = denominators(&batch(7, 7).rows).unwrap();
    assert!(untied.fractions_of(&all_tied).is_err());
    // Incomplete comparisons never reach a denominator or an objective.
    assert!(denominators(&b.rows[..7]).is_err());
    assert!(objective_of(&b, &l, 0, 12, Some(&global)).is_err());
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
            groups,
        },
        families,
    )
}

#[test]
fn every_rank_count_draws_the_same_global_batch_with_rank_local_rng() {
    let (index, families) = index();
    for local_step in 0..3u64 {
        let plans = [(1, 16), (3, 16), (4, 24), (5, 8), (13, 8)]
            .map(|(world, micro)| Plan::new(world, 88, micro).unwrap());
        let expected = sample(
            &index,
            &families,
            plans[0].global_serial(local_step).unwrap() as usize,
            88,
            SEED,
        );
        for plan in &plans {
            let mut ids = Vec::new();
            let mut seeds = std::collections::BTreeSet::new();
            for rank in 0..plan.world() {
                for slot in plan.slots(rank, SEED, local_step).unwrap() {
                    assert_eq!(slot.rows, slot.comparisons * 8);
                    assert_eq!(
                        slot.dropout_seed,
                        consumption::dropout_seed(SEED, local_step, rank, slot.microbatch)
                    );
                    assert!(seeds.insert(slot.dropout_seed), "rank-local RNG streams");
                    ids.extend(sample(
                        &index,
                        &families,
                        slot.serial as usize,
                        slot.rows,
                        SEED,
                    ));
                }
            }
            assert_eq!(ids, expected, "world {} changed the data", plan.world());
        }
    }
    // The v1 cursor cannot express an uneven three-rank shard.
    assert!(consumption::sampler_serial(0, 88, 3, 1, 0, 16).is_err());
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

#[test]
fn persisted_rank_state_and_committed_update_resume_exactly() {
    let b = batch(11, 0);
    let plan = Plan::new(3, 88, 16).unwrap();
    let state = plan
        .state(SEED, 0, update(&plan, &b, 5, 4).unwrap())
        .unwrap();
    let value = serde_json::to_value(&state).unwrap();
    let reloaded: State = serde_json::from_value(value.clone()).unwrap();
    assert_eq!(reloaded, state);
    let next = reloaded.resume(&plan, SEED).unwrap();
    for (rank, slots) in next.iter().enumerate() {
        assert_eq!(*slots, plan.slots(rank, SEED, 5).unwrap());
        assert_eq!(
            state.ranks[rank].sampler_cursor,
            Some(5 * 88 + 8 * plan.shards()[rank].first_group as u64)
        );
    }
    assert_eq!(state.committed_update.empty_terms, Vec::<String>::new());
    assert!(!state.committed_update.clipped);
    // The committed update is consistent with the step and sampling origin.
    assert!(plan
        .state(SEED, 2, update(&plan, &b, 5, 4).unwrap())
        .is_err());
    assert!(plan
        .state(SEED, 2, update(&plan, &b, 5, 2).unwrap())
        .is_ok());
    // Any altered persisted field is refused on reload.
    type Edit = Box<dyn Fn(&mut Value)>;
    let edits: Vec<(&str, Edit)> = vec![
        (
            "sampler cursor",
            Box::new(|s| s["ranks"][1]["sampler_cursor"] = json!(5 * 88 + 8 * 4 + 8)),
        ),
        (
            "dropout seed",
            Box::new(|s| s["ranks"][2]["next_slots"][0]["dropout_seed"] = json!(1)),
        ),
        (
            "next local step",
            Box::new(|s| s["ranks"][0]["next_local_step"] = json!(6)),
        ),
        (
            "missing rank",
            Box::new(|s| drop(s["ranks"].as_array_mut().unwrap().pop())),
        ),
        (
            "partition",
            Box::new(|s| s["partition"]["world"] = json!(4)),
        ),
        (
            "partition digest",
            Box::new(|s| s["partition_sha256"] = json!("c".repeat(64))),
        ),
        (
            "committed step",
            Box::new(|s| s["committed_step"] = json!(6)),
        ),
        (
            "loss",
            Box::new(|s| {
                let loss = s["committed_update"]["loss"].as_f64().unwrap();
                s["committed_update"]["loss"] = json!(loss * 1.01);
            }),
        ),
        (
            "term",
            Box::new(|s| s["committed_update"]["term_values"][1] = json!(9.0)),
        ),
        (
            "clipping",
            Box::new(|s| s["committed_update"]["clipped"] = json!(true)),
        ),
        (
            "empty terms",
            Box::new(|s| s["committed_update"]["empty_terms"] = json!(["ranking_bt"])),
        ),
        (
            "coverage",
            Box::new(|s| s["committed_update"]["comparisons"] = json!(10)),
        ),
        (
            "digest",
            Box::new(|s| s["committed_update"]["gradient_sha256"] = json!("x")),
        ),
        (
            "schema",
            Box::new(|s| s["schema"] = json!("parallel.partition_state.v0")),
        ),
        (
            "unknown field",
            Box::new(|s| s["ranks"][0]["extra"] = json!(1)),
        ),
    ];
    for (what, edit) in edits {
        let mut altered = value.clone();
        edit(&mut altered);
        let refused = serde_json::from_value::<State>(altered)
            .map_err(anyhow::Error::from)
            .and_then(|s| s.resume(&plan, SEED));
        assert!(refused.is_err(), "altered {what} was accepted");
    }
    // A different rank count or microbatch, or seed, is not an exact resume.
    let error = state
        .resume(&Plan::new(4, 88, 16).unwrap(), SEED)
        .unwrap_err();
    assert!(error.to_string().contains("explicit transition"), "{error}");
    assert!(state.resume(&Plan::new(3, 88, 24).unwrap(), SEED).is_err());
    assert!(state.resume(&plan, SEED + 1).is_err());
    // An empty rank persists an explicit empty sampler state.
    let wide = Plan::new(12, 88, 8).unwrap();
    let state = wide
        .state(SEED, 0, update(&wide, &b, 1, 0).unwrap())
        .unwrap();
    assert_eq!(state.ranks[11].sampler_cursor, None);
    assert!(state.ranks[11].next_slots.is_empty());
    assert!(state.resume(&wide, SEED).is_ok());
    // A globally empty term is recorded, and a value for it is refused.
    let tied = batch(11, 11);
    let good = update(&plan, &tied, 5, 4).unwrap();
    assert_eq!(good.empty_terms, ["ranking_bt", "contextfree_bt"]);
    let mut values = good.term_values;
    values[1] = 0.25;
    assert!(CommittedUpdate::new(
        &plan,
        5,
        4,
        &denominators(&tied.rows).unwrap(),
        values,
        &Tensor::from_slice(&[0.0f32]),
        0.0,
        &"b".repeat(64),
    )
    .is_err());
}

const CORPUS: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const VOCABULARY: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SAMPLER: &str = "family_uniform_champion_plus_seven_quantiles_v2";

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

#[test]
fn a_rank_count_change_is_an_explicit_transition() {
    let eight = Plan::new(8, 2048, 256).unwrap().record();
    let three = Plan::new(3, 2048, 256).unwrap().record();
    let from_eight = parent(eight.clone(), partition::objective_identity());
    // Exact resume refuses a rank-count change.
    let error = check(Mode::ExactResume, &from_eight, 3, &three, None).unwrap_err();
    assert!(
        error.to_string().contains("exact resume requires"),
        "{error}"
    );
    // Optimizer carry must declare it, exactly.
    let error = check(Mode::NewCorpusTransition, &from_eight, 3, &three, None).unwrap_err();
    assert!(error.to_string().contains("undeclared"), "{error}");
    let declared = json!({"optimizer":"carry","declared_changes":{"execution_partition":three}});
    let record = check(
        Mode::NewCorpusTransition,
        &from_eight,
        3,
        &three,
        Some(&declared),
    )
    .unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "changed");
    assert_eq!(record["execution_partition"]["declared"], three);
    assert_eq!(record["objective"]["continuity"], "identical");
    assert_eq!(record["exact_resume_claim"], false);
    let four = Plan::new(4, 2048, 256).unwrap().record();
    let wrong = json!({"optimizer":"carry","declared_changes":{"execution_partition":four}});
    assert!(check(
        Mode::NewCorpusTransition,
        &from_eight,
        3,
        &three,
        Some(&wrong)
    )
    .is_err());
    // A weight warm start carries nothing and declares nothing.
    let record = check(Mode::WeightWarmStart, &from_eight, 3, &three, None).unwrap();
    assert_eq!(record["optimizer"], "reset");
    assert!(check(
        Mode::WeightWarmStart,
        &from_eight,
        3,
        &three,
        Some(&declared)
    )
    .is_err());
    // The same three-rank partition resumes exactly.
    let from_three = parent(three.clone(), partition::objective_identity());
    let record = check(Mode::ExactResume, &from_three, 3, &three, None).unwrap();
    assert_eq!(record["execution_partition"]["continuity"], "identical");
    // A partition record for another topology than the request is refused.
    assert!(check(Mode::ExactResume, &from_three, 3, &four, None).is_err());
    // v1 → v2 changes the objective's reductions: never an optimizer carry.
    let v1_parent = parent(
        training_partition(8, 256, 2048).unwrap(),
        objective::identity(),
    );
    let error = check(
        Mode::NewCorpusTransition,
        &v1_parent,
        3,
        &three,
        Some(&declared),
    )
    .unwrap_err();
    assert!(error.to_string().contains("same objective"), "{error}");
    let record = check(Mode::WeightWarmStart, &v1_parent, 3, &three, None).unwrap();
    assert_eq!(record["objective"]["continuity"], "changed");
    // The v1 check cannot even describe three ranks.
    let config = config();
    let v1_digest = objective::identity_sha256();
    assert!(check_parent(
        Mode::WeightWarmStart,
        &v1_parent,
        &config,
        &Current {
            corpus_sha256: CORPUS,
            vocabulary_sha256: VOCABULARY,
            world: 3,
            microbatch: 256,
            global_batch: 2048,
            seed: SEED,
            sampler_id: SAMPLER,
            config: &config,
            objective_sha256: &v1_digest,
        },
        None,
    )
    .is_err());
}

#[test]
fn the_v2_objective_differs_from_v1_only_in_its_reductions() {
    let (v1, v2) = (objective::identity(), partition::objective_identity());
    assert_eq!(
        v1["normalization"],
        "microbatch_weighted_means_rank_mean_v1"
    );
    assert_eq!(v2["normalization"], partition::NORMALIZATION);
    assert_ne!(objective::identity_sha256(), partition::objective_sha256());
    assert_eq!(
        partition::objective_sha256(),
        hash(&serde_json::to_vec(&v2).unwrap())
    );
    let mut changed = Vec::new();
    for (section, value) in v1.as_object().unwrap() {
        match value.as_object() {
            Some(fields) => {
                for (field, x) in fields {
                    if v2[section][field] != *x {
                        changed.push(format!("{section}.{field}"));
                    }
                }
                assert_eq!(
                    fields.len(),
                    v2[section].as_object().unwrap().len(),
                    "{section} gained a field"
                );
            }
            None if v2[section] != *value => changed.push(section.clone()),
            None => {}
        }
    }
    changed.sort();
    assert_eq!(
        changed,
        [
            "normalization",
            "ranking.empty_support",
            "ranking.listwise_reduction",
            "ranking.pair_reduction",
            "runtime.reduction",
            "sequence.batch_reduction",
        ]
    );
    assert_eq!(v1.as_object().unwrap().len(), v2.as_object().unwrap().len());
}
