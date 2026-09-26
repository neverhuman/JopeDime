//! W4-T05: what the reward/runtime readout depends on (CPU/Candle, synthetic weights).
//!
//! `PolicyModel::reward_runtime` pools a program as the PAD-masked *mean* of its
//! `reward_token_emb` rows, with no position embedding or attention, and every reward head and
//! the runtime head read the program only through that mean. Programs with equal token
//! multisets therefore score identically whatever their order: the readout is order-blind. It
//! is not program-blind: nonsaturated single-token edits move every reward head and the runtime
//! estimate. The autoregressive decoder, by contrast, is order-aware. Both are structural
//! properties of the executed computation, shown here on deterministic synthetic weights (reward
//! path of magnitude <= 0.2, so its activations stay in their responsive range); nothing here
//! changes the accepted architecture, and an order-aware readout is a W10 challenger, not a
//! repair. No accepted-weight magnitude is claimed.
//!
//! Perturbing disposable copies of each stored tensor group maps it to the outputs that respond:
//! reward heads to predicted quality (hence ranking order), the runtime head to the reported
//! runtime only, the context/decoder tensors to proposal logits only, and the stored
//! `stability_head`, `reward_head_ablation` and `loss_log_vars` to nothing at inference.
use candle_core::{Device, Tensor};
use jope_policy::config::PolicyConfig;
use jope_policy::context::PolicyContext;
use jope_policy::grammar::{bind_vocabulary, Vocab, EOS, PAD};
use jope_policy::model::{expected_inventory, PolicyModel};
use jope_policy::ranking::{score_pool, Bindings, Options};
use jope_policy::{context_sha256, sha256_hex, CHECKPOINT_VOCABULARY_SHA256};
use std::path::Path;

fn vocab() -> Vocab {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar/featmath.json");
    bind_vocabulary(&std::fs::read(path).unwrap(), CHECKPOINT_VOCABULARY_SHA256)
        .unwrap()
        .vocab()
        .clone()
}

fn context() -> PolicyContext {
    PolicyContext::from_frozen_values((0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect())
        .unwrap()
}

fn config() -> PolicyConfig {
    let mut config = PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config
}

/// Deterministic synthetic weights: magnitude 1 for the token/position embeddings, decoder
/// layers and output norm (sharp enough attention to make order visible), <= 0.2 elsewhere. `shift(name)` scales
/// a perturbation added to tensor `name`; it varies across entries, so LayerNorm cannot cancel
/// it as a constant offset.
fn model(shift: impl Fn(&str) -> f32) -> PolicyModel {
    let config = config();
    let tensors = expected_inventory(&config)
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel = shape.iter().product::<usize>();
            let delta = shift(&name);
            let scale = if ["token_emb.", "pos_emb.", "decoder.", "out_norm."]
                .iter()
                .any(|p| name.starts_with(p))
            {
                1.0
            } else {
                0.2
            };
            let values = (0..numel)
                .map(|j| {
                    ((i * 131 + j) as f32 * 0.37).sin() * scale
                        + delta * ((1 + j * 37 % 11) as f32 / 11.0)
                })
                .collect::<Vec<f32>>();
            let tensor = if shape.is_empty() {
                Tensor::new(values[0], &Device::Cpu)
            } else {
                Tensor::from_vec(values, shape, &Device::Cpu)
            }
            .unwrap();
            (name, tensor)
        })
        .collect::<Vec<_>>();
    PolicyModel::from_tensors(config, tensors, Device::Cpu).unwrap()
}

/// Rows right-padded with PAD to a shared length, as `score_pool` batches them.
fn padded(rows: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let length = rows.iter().map(Vec::len).max().unwrap();
    rows.iter()
        .map(|row| {
            let mut row = row.clone();
            row.resize(length, PAD);
            row
        })
        .collect()
}

struct Readout {
    heads: Vec<[f32; 4]>,
    runtime: Vec<f32>,
    /// Decoder next-token logits at each row's history before its EOS.
    logits: Vec<Vec<f32>>,
}

fn readout(model: &PolicyModel, ctx: &PolicyContext, rows: &[Vec<usize>]) -> Readout {
    let (heads, runtime) = model.reward_runtime(ctx, &padded(rows)).unwrap();
    let logits = rows
        .iter()
        .map(|row| {
            // The logits that choose the final EOS: the history before it.
            let history = &row[..row.len() - 1];
            model
                .next_token_logits(ctx, &[history.to_vec()])
                .unwrap()
                .remove(0)
        })
        .collect();
    Readout {
        heads,
        runtime,
        logits,
    }
}

fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            assert!(x.is_finite() && y.is_finite());
            (x - y).abs()
        })
        .fold(0.0, f32::max)
}

fn sorted(tokens: &[usize]) -> Vec<usize> {
    let mut tokens = tokens.to_vec();
    tokens.sort_unstable();
    tokens
}

/// Distinct valid programs whose token sequences are permutations of each other. The last pair
/// swaps the operands of a subtraction: different functions, same multiset.
const COLLISIONS: [(&str, &str); 3] = [
    (
        "(pipe X (blom) (rank) (ridge))",
        "(pipe X (rank) (blom) (ridge))",
    ),
    (
        "(pipe X (standardize) (sigmoid) (robust_scale) (stack))",
        "(pipe X (robust_scale) (standardize) (sigmoid) (stack))",
    ),
    (
        "(pipe X (derive expr=\"(row_mean - row_std)\") (ridge))",
        "(pipe X (derive expr=\"(row_std - row_mean)\") (ridge))",
    ),
];

/// Single-token edits of the first collision programs (head, transform and operator changes).
const EDITS: [(&str, &str); 3] = [
    (
        "(pipe X (blom) (rank) (ridge))",
        "(pipe X (blom) (rank) (stack))",
    ),
    (
        "(pipe X (blom) (rank) (ridge))",
        "(pipe X (blom) (sigmoid) (ridge))",
    ),
    (
        "(pipe X (derive expr=\"(row_mean - row_std)\") (ridge))",
        "(pipe X (derive expr=\"(row_mean + row_std)\") (ridge))",
    ),
];

#[test]
fn reward_readout_is_order_blind_but_program_sensitive() {
    let vocab = vocab();
    let ctx = context();
    let model = model(|_| 0.0);
    let mut collision_noise = 0f32;
    for (a, b) in COLLISIONS {
        let (ta, tb) = (vocab.encode(a).unwrap(), vocab.encode(b).unwrap());
        assert_ne!(a, b);
        assert_ne!(ta, tb);
        assert_eq!(
            sorted(&ta),
            sorted(&tb),
            "fixture pair must share a token multiset"
        );
        let r = readout(&model, &ctx, &[ta, tb]);
        let noise =
            max_difference(&r.heads[0], &r.heads[1]).max((r.runtime[0] - r.runtime[1]).abs());
        // Equal up to the floating-point summation order of the mean.
        assert!(noise <= 1e-6, "{a} vs {b}: {noise}");
        collision_noise = collision_noise.max(noise);
        // The decoder is order-aware: the same multiset reached in another order predicts a
        // different next-token distribution (far above float32 rounding of O(1) logits).
        let order = max_difference(&r.logits[0], &r.logits[1]);
        eprintln!("{a} vs {b}: readout {noise:e}, decoder logits {order:e}");
        assert!(order > 1e-4, "{a} vs {b}: decoder logits moved {order}");
    }
    let mut smallest_edit = f32::INFINITY;
    for (a, b) in EDITS {
        let (ta, tb) = (vocab.encode(a).unwrap(), vocab.encode(b).unwrap());
        assert_eq!(ta.len(), tb.len());
        assert_eq!(ta.iter().zip(&tb).filter(|(x, y)| x != y).count(), 1);
        let r = readout(&model, &ctx, &[ta, tb]);
        for head in 0..4 {
            let moved = (r.heads[0][head] - r.heads[1][head]).abs();
            assert!(moved > 1e-4, "{a} -> {b}: reward head {head} moved {moved}");
            smallest_edit = smallest_edit.min(moved);
        }
        assert!(
            (r.runtime[0] - r.runtime[1]).abs() > 1e-5,
            "{a} -> {b}: runtime"
        );
    }
    eprintln!(
        "largest collision difference {collision_noise:e}; smallest single-token edit {smallest_edit:e}"
    );
    // Every edit moves every head by far more than float32 rounding of O(1) outputs (and than
    // any permutation did); a single equal pair is never the evidence either way.
    assert!(smallest_edit > 100.0 * collision_noise.max(f32::EPSILON));
    // Through the production ranking: the permuted programs receive the same predicted quality,
    // so their relative order comes from the documented program-hash tie break alone (or from
    // summation noise below 1e-6), never from their order.
    let programs = COLLISIONS
        .iter()
        .flat_map(|(a, b)| [a.to_string(), b.to_string()])
        .collect::<Vec<_>>();
    let bindings = Bindings::new(
        &sha256_hex(b"synthetic scorer fixture"),
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    let ranking = score_pool(
        &model,
        &vocab,
        &ctx,
        &programs,
        bindings,
        &Options {
            batch_size: 64,
            keep: programs.len(),
        },
    )
    .unwrap();
    let quality = |program: &str| {
        ranking
            .ranked()
            .iter()
            .find(|p| p.program == program)
            .unwrap()
            .predicted_quality
    };
    for (a, b) in COLLISIONS {
        assert!((quality(a) - quality(b)).abs() <= 1e-6, "{a} vs {b}");
    }
}

#[test]
fn padding_is_excluded_from_the_program_mean() {
    let vocab = vocab();
    let ctx = context();
    let model = model(|_| 0.0);
    let program = vocab.encode(EDITS[0].0).unwrap();
    let mut longer = program.clone();
    longer.extend([PAD; 7]);
    let (plain, plain_runtime) = model.reward_runtime(&ctx, &[program]).unwrap();
    let (padded, padded_runtime) = model.reward_runtime(&ctx, &[longer]).unwrap();
    assert!(max_difference(&plain[0], &padded[0]) <= 1e-6);
    assert!((plain_runtime[0] - padded_runtime[0]).abs() <= 1e-6);
}

/// Stored tensor groups and the outputs each may move at inference.
#[derive(Debug)]
struct Group {
    name: &'static str,
    prefixes: &'static [&'static str],
    rewards: bool,
    runtime: bool,
    logits: bool,
}

const GROUPS: [Group; 9] = [
    Group {
        name: "reward heads",
        prefixes: &["reward_heads."],
        rewards: true,
        runtime: false,
        logits: false,
    },
    Group {
        name: "runtime head",
        prefixes: &["runtime_head."],
        rewards: false,
        runtime: true,
        logits: false,
    },
    Group {
        name: "program pooling",
        prefixes: &["reward_token_emb.", "reward_norm."],
        rewards: true,
        runtime: true,
        logits: false,
    },
    Group {
        name: "reward context",
        prefixes: &["reward_encoder.", "z_gamma.", "z_proj."],
        rewards: true,
        runtime: true,
        logits: false,
    },
    Group {
        name: "decoder context",
        prefixes: &["encoder.", "ctx_proj."],
        rewards: false,
        runtime: false,
        logits: true,
    },
    Group {
        name: "decoder",
        prefixes: &["token_emb.", "pos_emb.", "decoder.", "out_norm."],
        rewards: false,
        runtime: false,
        logits: true,
    },
    Group {
        name: "stability head (stored, inactive)",
        prefixes: &["stability_head."],
        rewards: false,
        runtime: false,
        logits: false,
    },
    Group {
        name: "ablation head (stored, inactive)",
        prefixes: &["reward_head_ablation."],
        rewards: false,
        runtime: false,
        logits: false,
    },
    Group {
        name: "loss weights (stored, inactive)",
        prefixes: &["loss_log_vars."],
        rewards: false,
        runtime: false,
        logits: false,
    },
];

#[test]
fn perturbing_each_stored_head_moves_only_its_declared_consumers() {
    let vocab = vocab();
    let ctx = context();
    let rows = EDITS
        .iter()
        .map(|(a, _)| vocab.encode(a).unwrap())
        .collect::<Vec<_>>();
    let base = readout(&model(|_| 0.0), &ctx, &rows);
    // Every stored tensor belongs to exactly one group.
    for (name, _) in expected_inventory(&config()) {
        let owners = GROUPS
            .iter()
            .filter(|g| g.prefixes.iter().any(|p| name.starts_with(p)))
            .count();
        assert_eq!(owners, 1, "{name}");
    }
    for group in &GROUPS {
        let copy = model(|name| {
            if group.prefixes.iter().any(|p| name.starts_with(p)) {
                0.5
            } else {
                0.0
            }
        });
        let r = readout(&copy, &ctx, &rows);
        let rewards = (0..rows.len())
            .map(|i| max_difference(&base.heads[i], &r.heads[i]))
            .fold(0.0, f32::max);
        let runtime = max_difference(&base.runtime, &r.runtime);
        let logits = (0..rows.len())
            .map(|i| max_difference(&base.logits[i], &r.logits[i]))
            .fold(0.0, f32::max);
        eprintln!(
            "{:<34} rewards {rewards:.3e} runtime {runtime:.3e} logits {logits:.3e}",
            group.name
        );
        // Declared consumers respond; every other output is bit-identical.
        for (declared, moved, output) in [
            (group.rewards, rewards, "rewards"),
            (group.runtime, runtime, "runtime"),
            (group.logits, logits, "logits"),
        ] {
            if declared {
                assert!(moved > 1e-4, "{}: {output} did not respond", group.name);
            } else {
                assert_eq!(moved, 0.0, "{}: {output} moved", group.name);
            }
        }
    }
    assert!(rows.iter().all(|r| r.last() == Some(&EOS)));
}
