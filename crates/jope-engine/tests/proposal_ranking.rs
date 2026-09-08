#![cfg(feature = "neural")]

use jope_parallel::{
    grammar::Vocab,
    manifest::{file_hash, hash, read_json},
    neural::{Config, Policy},
    proposal_ranking::{score_pool, Bindings, Options},
};
use std::path::Path;
use tch::{nn, Device, Kind, Tensor};

fn fixture(max_seq: i64) -> (nn::VarStore, Policy, Vocab, Vec<f32>, Bindings, Vec<String>) {
    tch::set_num_threads(2);
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json");
    let vocab: Vocab = read_json(&path).unwrap();
    let mut config = Config::regression(vocab.tokens.len() as i64);
    config.width = 32;
    config.ffn = 128;
    config.layers = 2;
    config.max_seq = max_seq;
    config.dropout = 0.25; // train=false must keep inference deterministic.
    let vs = nn::VarStore::new(Device::Cpu);
    let policy = Policy::new(vs.root(), config).unwrap();
    let values = (0..2048)
        .map(|i| ((i * 7 % 29) as f32 - 14.) / 14.)
        .collect::<Vec<_>>();
    let bindings = Bindings {
        checkpoint_sha256: "a".repeat(64),
        conditioning_context_sha256: hash(
            &values
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        vocabulary_sha256: file_hash(&path).unwrap(),
    };
    let programs = [
        "(pipe X (standardize) (ridge))",
        "(pipe X (ridge))",
        "(pipe X (rff_cos n=16 gamma=0.5 seed=7 append=true) (ridge))",
        "(pipe X (map expr=\"tanh(x*x)\") (ridge))",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    (vs, policy, vocab, values, bindings, programs)
}

#[test]
fn batched_quality_and_runtime_match_direct_policy_outputs() {
    let (vs, policy, vocab, values, bindings, programs) = fixture(128);
    let rank = score_pool(
        &policy,
        &vocab,
        &values,
        &programs,
        bindings.clone(),
        Options {
            device: Device::Cpu,
            batch_size: 2,
            keep: 2,
        },
    )
    .unwrap();
    assert_eq!(rank.scoring_batches, 2);
    assert_eq!(rank.selected().len(), 2);
    assert!(!rank.gp_scores_read && !rank.heldout_labels_read && rank.training_context_only);
    let rows = programs
        .iter()
        .map(|s| vocab.encode(s).unwrap())
        .collect::<Vec<_>>();
    // Direct reference follows training's eight-token padding buckets, which
    // differ from the scorer's exact-length batches. Padding cannot change heads.
    let length = rows.iter().map(|v| v.len()).max().unwrap().div_ceil(8) * 8;
    let mut token_values = vec![0i64; programs.len() * length];
    for (i, row) in rows.iter().enumerate() {
        for (j, token) in row.iter().enumerate() {
            token_values[i * length + j] = *token as i64;
        }
    }
    let tokens = Tensor::from_slice(&token_values).view([programs.len() as i64, length as i64]);
    let contexts = Tensor::from_slice(&values)
        .view([1, 2048])
        .repeat([programs.len() as i64, 1]);
    let (_, heads, runtime, _, _) = tch::no_grad(|| {
        policy.forward(
            &contexts.narrow(1, 0, 1536),
            &contexts.narrow(1, 1536, 512),
            &tokens,
            &tokens.ne(0),
            false,
        )
    })
    .unwrap();
    let expected_quality = heads.mean_dim(&[1i64][..], false, Kind::Float);
    let mut maximum_error = 0f64;
    for (i, program) in programs.iter().enumerate() {
        let scored = rank.ranked.iter().find(|r| &r.program == program).unwrap();
        maximum_error = maximum_error
            .max((scored.predicted_quality - expected_quality.double_value(&[i as i64])).abs());
        maximum_error = maximum_error
            .max((scored.predicted_runtime_seconds - runtime.double_value(&[i as i64])).abs());
        for head in 0..4 {
            maximum_error = maximum_error.max(
                (scored.reward_heads[head] - heads.double_value(&[i as i64, head as i64])).abs(),
            );
        }
    }
    assert!(
        maximum_error < 2e-6,
        "scoring/reference mismatch {maximum_error}"
    );
    assert!(rank
        .ranked
        .windows(2)
        .all(|rows| rows[0].predicted_quality >= rows[1].predicted_quality));
    assert!(vs
        .variables()
        .values()
        .all(|tensor| !tensor.grad().defined()));
    let reversed = programs.into_iter().rev().collect::<Vec<_>>();
    let reordered = score_pool(
        &policy,
        &vocab,
        &values,
        &reversed,
        bindings,
        Options {
            device: Device::Cpu,
            batch_size: 2,
            keep: 2,
        },
    )
    .unwrap();
    assert_eq!(
        serde_json::to_value(&rank).unwrap(),
        serde_json::to_value(&reordered).unwrap()
    );
    eprintln!("proposal quality/runtime scoring: direct Policy.forward reference max error {maximum_error:e}, tolerance2e-6; input permutation identical");
}

#[test]
fn rejects_wrong_context_duplicate_unrepresentable_and_overlong_programs() {
    let (_, policy, vocab, values, bindings, programs) = fixture(128);
    let options = || Options {
        device: Device::Cpu,
        batch_size: 2,
        keep: 1,
    };
    let mut changed = values.clone();
    changed[0] += 0.5;
    assert!(score_pool(
        &policy,
        &vocab,
        &changed,
        &programs,
        bindings.clone(),
        options()
    )
    .unwrap_err()
    .to_string()
    .contains("context bytes differ"));
    changed[0] = f32::NAN;
    assert!(score_pool(
        &policy,
        &vocab,
        &changed,
        &programs,
        bindings.clone(),
        options()
    )
    .is_err());
    let repeated = vec![programs[0].clone(), programs[0].clone()];
    assert!(score_pool(
        &policy,
        &vocab,
        &values,
        &repeated,
        bindings.clone(),
        options()
    )
    .unwrap_err()
    .to_string()
    .contains("duplicate generated"));
    let unrepresentable = vec!["(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))".into()];
    assert!(score_pool(
        &policy,
        &vocab,
        &values,
        &unrepresentable,
        bindings.clone(),
        options()
    )
    .is_err());
    let (_, short, _, _, _, _) = fixture(4);
    assert!(
        score_pool(&short, &vocab, &values, &programs, bindings, options())
            .unwrap_err()
            .to_string()
            .contains("sequence limit")
    );
}

#[test]
fn reward_only_is_exact_and_independent_of_decoder_parameters() {
    let (vs, policy, vocab, values, _, programs) = fixture(128);
    let encoded = programs
        .iter()
        .map(|p| vocab.encode(p).unwrap())
        .collect::<Vec<_>>();
    let length = encoded.iter().map(Vec::len).max().unwrap();
    let mut ids = vec![0i64; programs.len() * length];
    for (i, row) in encoded.iter().enumerate() {
        for (j, &id) in row.iter().enumerate() {
            ids[i * length + j] = id as i64;
        }
    }
    let tokens = Tensor::from_slice(&ids).view([programs.len() as i64, length as i64]);
    let contexts = Tensor::from_slice(&values)
        .view([1, 2048])
        .repeat([programs.len() as i64, 1]);
    let dataset = contexts.narrow(1, 0, 1536);
    let target = contexts.narrow(1, 1536, 512);
    let pad = tokens.ne(0);
    let (_, full_reward, full_runtime, _, _) =
        tch::no_grad(|| policy.forward(&dataset, &target, &tokens, &pad, false)).unwrap();
    let (reward, runtime) =
        tch::no_grad(|| policy.reward_runtime(&dataset, &target, &tokens, &pad)).unwrap();
    assert_eq!((&full_reward - &reward).abs().max().double_value(&[]), 0.0);
    assert_eq!(
        (&full_runtime - &runtime).abs().max().double_value(&[]),
        0.0
    );
    let mut damaged = 0;
    tch::no_grad(|| {
        for (name, mut tensor) in vs.variables() {
            if name.starts_with("decoder.")
                || name.starts_with("encoder.")
                || name.starts_with("token_emb.")
            {
                let _ = tensor.fill_(f64::NAN);
                damaged += 1;
            }
        }
    });
    assert!(damaged > 0);
    let (after_reward, after_runtime) =
        tch::no_grad(|| policy.reward_runtime(&dataset, &target, &tokens, &pad)).unwrap();
    assert_eq!((&reward - after_reward).abs().max().double_value(&[]), 0.0);
    assert_eq!(
        (&runtime - after_runtime).abs().max().double_value(&[]),
        0.0
    );
}
