//! jope-dime step-60000 policy port (`feature = "testing"`), CPU only.
//!
//! Default tests use the committed vocabularies and a tiny deterministic random-weight model.
//! Ignored tests read external evidence in place:
//! - `JOPE_DIME_POLICY_SAFETENSORS`: the exported `policy.safetensors` (receipt from
//!   `JOPE_DIME_EXPORT_RECEIPT`, default `<dir>/../step-60000.receipt.json`);
//! - `JOPE_DIME_GENERATION_000011`: a jope-dime `generation-000011` results directory;
//! - `JOPE_DIME_CONTEXTS`: a `contexts-assembly-v2` directory (`<dataset>/cut-N.f32`).
//!
//! Run the ignored tests with an optimized profile, e.g.
//! `cargo test --release -p jope-policy --features testing --test jope_dime -- --include-ignored`.
#![cfg(feature = "testing")]

use jope_policy::config::{CheckpointBinding, PolicyConfig};
use jope_policy::decode::{self, SamplingOptions};
use jope_policy::grammar::{bind_vocabulary, verify_restriction, VocabularyIdentity};
use jope_policy::grammar::{Vocab, BOS, EOS, PAD};
use jope_policy::model::{expected_inventory, PolicyModel};
use jope_policy::ranking::{
    compare_predictions, pool_sha256, score_pool, sorted_predictions, Bindings, Options,
    Prediction, Ranking, SELECTION_ID,
};
use jope_policy::seeds::{
    self, propose, validate_for_search, write_population, ProposalSeeds, SeedBinding,
    SeedCandidate, ORIGIN_NEURAL, ORIGIN_NEURAL_FALLBACK, ORIGIN_RANDOM,
};
use jope_policy::testing::{
    assemble_for_testing, propose_unbound, pure_pool_unbound, ranking_for_testing,
    UNBOUND_TEST_STATE_SHA256,
};
use jope_policy::{
    context_sha256, sha256_hex, ACCEPTED_CONFIG_SHA256, ACCEPTED_PARAMETER_SHA256,
    ACCEPTED_STATE_SHA256, CHECKPOINT_VOCABULARY_SHA256, PRODUCT_VOCABULARY_SHA256,
    REJECTED_MODEL_SHA256, REJECTED_STATE_SHA256,
};
use serde_json::{json, Value};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

fn grammar_dir() -> PathBuf {
    // The committed vocabularies live with jope-engine (byte-identical to feat-core's
    // `artifacts/jope-dime/grammar`); the crate does not carry a second copy.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../jope-engine/assets/grammar")
}

fn base_bytes() -> Vec<u8> {
    std::fs::read(grammar_dir().join("featmath.json")).unwrap()
}

fn product_bytes() -> Vec<u8> {
    std::fs::read(grammar_dir().join("featmath-jain-v1.json")).unwrap()
}

fn base_vocab() -> Vocab {
    bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256)
        .unwrap()
        .vocab()
        .clone()
}

fn tiny_config() -> PolicyConfig {
    let mut config = PolicyConfig::regression(401);
    config.width = 16;
    config.heads = 4;
    config.ffn = 32;
    config.layers = 1;
    config.dropout = 0.0;
    config
}

fn tiny_model(seed: u64) -> PolicyModel {
    PolicyModel::random_weights(tiny_config(), seed).unwrap()
}

/// Deterministic tiny-config tensors; `pad_row_shift` is added to the PAD row (row 0) of
/// `token_emb.weight` only.
fn tiny_tensors(pad_row_shift: f32) -> std::collections::HashMap<String, candle_core::Tensor> {
    let config = tiny_config();
    expected_inventory(&config)
        .into_iter()
        .enumerate()
        .map(|(i, (name, shape))| {
            let numel = shape.iter().product::<usize>();
            let pad_row = if name == "token_emb.weight" {
                config.width
            } else {
                0
            };
            let values = (0..numel)
                .map(|j| {
                    let v = ((i * 131 + j) as f32 * 0.37).sin() * 0.2;
                    if j < pad_row {
                        v + pad_row_shift * (1.0 + j as f32 * 0.1)
                    } else {
                        v
                    }
                })
                .collect::<Vec<f32>>();
            let t = candle_core::Tensor::from_vec(values, shape, &cpu()).unwrap();
            (name, t)
        })
        .collect()
}

fn context() -> Vec<f32> {
    (0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect()
}

fn digest(tag: &str) -> String {
    sha256_hex(tag.as_bytes())
}

/// Programs taken from jope-dime generation-000011 neural seed files (step-60000 samples).
const GENERATION_11_PROGRAMS: &[&str] = &[
    "(pipe X (ridge))",
    "(pipe X (target_bin_encode bins=4 prior=10) (quantile_bins bins=32) (target_bin_encode append=true bins=4) (robust_scale append=true) (robust_scale) (elasticnet alpha=0.001 l1_ratio=0.2))",
    "(pipe X (target_bin_encode append=true bins=4) (map append=true expr=\"(((0.05 / 0.05) / 0.05) / 0.05)\") (quantile_bins append=true bins=32) (elasticnet alpha=0.0005 l1_ratio=0.2))",
    "(pipe X (blom append=true) (cheb append=false degree=2) (quantile_bins append=false bins=4) (sigmoid append=false) (derive append=true expr=\"col(6)\") (blom append=true) (krr m=1000))",
    "(pipe X (blom append=true) (map append=true expr=\"((5 / 5) / 5)\") (sigmoid append=false) (sigmoid append=false) (rff_cos append=true gamma=2.0 n=32 seed=23) (krr m=200))",
    "(pipe X (quantile_bins append=true bins=8) (derive expr=\"(((row_std / row_std) / (-1 * -1)) / -1)\") (quantile_bins append=true bins=8) (svd_project append=true iters=8 k=32) (standardize append=false) (robust_scale) (gpr kernel=rq m=300 steps=40))",
];

// ---------------------------------------------------------------- configuration and receipt

#[test]
fn config_parse_rejects_unknown_fields() {
    let good = serde_json::to_vec(&PolicyConfig::regression(401)).unwrap();
    let parsed = PolicyConfig::from_json(&good).unwrap();
    assert_eq!(parsed.embedding_rows, 657);
    assert_eq!(parsed.max_seq, 128);
    let mut extra: Value = serde_json::from_slice(&good).unwrap();
    extra["kv_cache"] = json!(true);
    let err = PolicyConfig::from_json(&serde_json::to_vec(&extra).unwrap()).unwrap_err();
    assert!(format!("{err:#}").contains("unknown field"), "{err:#}");
    let mut missing: Value = serde_json::from_slice(&good).unwrap();
    missing.as_object_mut().unwrap().remove("heads");
    assert!(PolicyConfig::from_json(&serde_json::to_vec(&missing).unwrap()).is_err());
    let mut invalid = PolicyConfig::regression(401);
    invalid.heads = 3;
    assert!(invalid.validate().is_err());
}

#[test]
fn step_60000_inventory_has_136_tensors() {
    let inventory = expected_inventory(&PolicyConfig::regression(401));
    assert_eq!(inventory.len(), 136);
    let names = inventory
        .iter()
        .map(|e| e.0.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), 136);
    for name in [
        "reward_encoder.0.weight",
        "reward_heads.3.3.bias",
        "loss_log_vars.routing_ce",
        "decoder.layers.5.self_attn.in_proj_weight",
    ] {
        assert!(names.contains(name), "{name}");
    }
    let find = |n: &str| inventory.iter().find(|e| e.0 == n).unwrap().1.clone();
    assert_eq!(find("token_emb.weight"), vec![657, 512]);
    assert_eq!(find("pos_emb.weight"), vec![129, 512]);
    assert_eq!(find("encoder.0.weight"), vec![512, 2048]);
    assert_eq!(find("runtime_head.0.weight"), vec![512, 1024]);
    assert_eq!(find("loss_log_vars.seq_nll"), Vec::<usize>::new());
}

fn synthetic_receipt(state: &str, model: &str, step: u64) -> Value {
    let config = PolicyConfig::regression(401);
    let tensors = expected_inventory(&config)
        .into_iter()
        .map(|(name, shape)| {
            let numel = shape.iter().product::<usize>();
            json!({"name": name, "dtype": "F32", "shape": shape, "numel": numel,
                   "bytes": numel * 4, "sha256": digest(&name)})
        })
        .collect::<Vec<_>>();
    json!({
        "schema": "jope.checkpoint_export.v1",
        "checkpoint": {
            "accepted_identity": true,
            "carried_bindings": {
                "corpus_sha256": digest("corpus"),
                "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
            },
            "config": config,
            "config_sha256": ACCEPTED_CONFIG_SHA256,
            "expected_state_sha256": state,
            "model_sha256": model,
            "parameter_sha256": digest("parameters"),
            "parameter_sha256_recomputed": digest("parameters"),
            "state_sha256": state,
            "step": step,
            "verified_bindings": ["model_sha256", "config_sha256", "autocast_cache_lifetime", "parameter_sha256"],
        },
        "policy": {
            "output": {"sha256": digest("policy")},
            "tensor_count": tensors.len(),
            "tensors": tensors,
        },
    })
}

fn parse_receipt(receipt: &Value) -> anyhow::Result<CheckpointBinding> {
    CheckpointBinding::from_receipt_bytes(&serde_json::to_vec(receipt).unwrap())
}

#[test]
fn receipt_binds_the_accepted_checkpoint() {
    let receipt = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    let binding = parse_receipt(&receipt).unwrap();
    assert_eq!(binding.tensors().len(), 136);
    assert_eq!(binding.vocabulary_sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(*binding.config(), PolicyConfig::regression(401));
}

#[test]
fn rejected_checkpoint_state_is_refused() {
    for receipt in [
        synthetic_receipt(REJECTED_STATE_SHA256, &digest("model"), 60000),
        synthetic_receipt(ACCEPTED_STATE_SHA256, REJECTED_MODEL_SHA256, 60000),
        synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 69000),
    ] {
        let err = parse_receipt(&receipt).unwrap_err();
        assert!(format!("{err:#}").contains("rejected"), "{err:#}");
    }
    // Any other identity, an unverified parameter hash, or an unknown config field is refused.
    let other = synthetic_receipt(&digest("other state"), &digest("model"), 60000);
    assert!(parse_receipt(&other).is_err());
    let mut unverified = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    unverified["checkpoint"]["parameter_sha256_recomputed"] = json!(digest("different"));
    assert!(parse_receipt(&unverified).is_err());
    let mut unknown = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    unknown["checkpoint"]["config"]["rope"] = json!(true);
    assert!(parse_receipt(&unknown).is_err());
    // Scoring bindings and the seed assembler refuse the rejected state too.
    assert!(Bindings::new(REJECTED_STATE_SHA256, &digest("context"), &digest("vocab")).is_err());
    let binding =
        SeedBinding::new(&digest("prepared"), 0, &digest("train"), &digest("context")).unwrap();
    let ranking = fake_ranking(&digest("other state"), &binding, &[]);
    assert!(assemble_for_testing(&binding, REJECTED_STATE_SHA256, &ranking, &[], &[]).is_err());
}

// ---------------------------------------------------------------- vocabulary and grammar

#[test]
fn committed_vocabularies_are_the_documented_bytes() {
    assert_eq!(sha256_hex(&base_bytes()), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(sha256_hex(&product_bytes()), PRODUCT_VOCABULARY_SHA256);
    let vocab = base_vocab();
    assert_eq!(vocab.tokens.len(), 401);
    assert_eq!(vocab.engine_ids["featmath"], 3);
    assert_eq!(vocab.task_ids["regression"], 397);
    assert_eq!(vocab.append_ids, [50, 51, 52]);
    assert_eq!(
        vocab.proposal_prefix("regression").unwrap(),
        vec![1, 3, 397]
    );
    assert!(vocab.task_id("binary").is_err());
}

#[test]
fn product_restriction_is_token_compatible_and_bound() {
    let bound = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    assert_eq!(
        *bound.identity(),
        VocabularyIdentity::ProductRestriction {
            base_sha256: CHECKPOINT_VOCABULARY_SHA256.into()
        }
    );
    verify_restriction(&base_vocab(), bound.vocab()).unwrap();
    // Removed heads are no longer encodable; kept programs encode to identical tokens.
    let base = base_vocab();
    assert!(base.encode("(pipe X (ridge_gcv))").is_ok());
    assert!(bound.vocab().encode("(pipe X (ridge_gcv))").is_err());
    for program in GENERATION_11_PROGRAMS {
        assert_eq!(
            base.encode(program).unwrap(),
            bound.vocab().encode(program).unwrap()
        );
    }
    // A restriction whose tables differ from the base fails the full proof.
    let mut tampered = bound.vocab().clone();
    tampered.tokens[60] = "OP=featmath:other".into();
    assert!(verify_restriction(&base, &tampered).is_err());
    // Extra engine table keys are refused with an error, not a panic.
    let mut extra = bound.vocab().clone();
    extra.engine_transforms.insert("phantom".into(), Vec::new());
    assert!(verify_restriction(&base, &extra).is_err());
    let mut extra = bound.vocab().clone();
    extra.engine_heads.insert("phantom".into(), Vec::new());
    assert!(verify_restriction(&base, &extra).is_err());
    let mut missing = bound.vocab().clone();
    missing.engine_heads.clear();
    assert!(verify_restriction(&base, &missing).is_err());
}

#[test]
fn wrong_vocabulary_is_refused() {
    // Bound to a different checkpoint vocabulary.
    let other = digest("another checkpoint vocabulary");
    assert!(bind_vocabulary(&base_bytes(), &other).is_err());
    assert!(bind_vocabulary(&product_bytes(), &other).is_err());
    // One changed byte (still valid JSON) is a different, undocumented vocabulary.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let err = bind_vocabulary(&edited, CHECKPOINT_VOCABULARY_SHA256).unwrap_err();
    assert!(format!("{err:#}").contains("refused"), "{err:#}");
    let mut edited = product_bytes();
    edited.push(b'\n');
    assert!(bind_vocabulary(&edited, CHECKPOINT_VOCABULARY_SHA256).is_err());
    // The product file is not itself a checkpoint vocabulary.
    assert!(bind_vocabulary(&product_bytes(), PRODUCT_VOCABULARY_SHA256).is_err());
}

fn assert_refused<T>(result: anyhow::Result<T>, needle: &str) {
    match result {
        Ok(_) => panic!("expected a refusal containing {needle:?}"),
        Err(err) => assert!(format!("{err:#}").contains(needle), "{err:#}"),
    }
}

#[test]
fn accepted_config_is_pinned() {
    // The config.json layout of PolicyConfig::regression(401) is the file state.json binds.
    assert_eq!(
        PolicyConfig::regression(401).sha256().unwrap(),
        ACCEPTED_CONFIG_SHA256
    );
    let receipt = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    parse_receipt(&receipt)
        .unwrap()
        .require_accepted_identity()
        .unwrap();
    // A receipt claiming another config hash is refused, even around the accepted config.
    let mut claimed = receipt.clone();
    claimed["checkpoint"]["config_sha256"] = json!(digest("config"));
    assert_refused(
        parse_receipt(&claimed).unwrap().require_accepted_identity(),
        "config_sha256",
    );
    // A forged config under the accepted claim is refused by the recomputed hash, whether the
    // change keeps every tensor shape (rms, dropout) or not (heads).
    for (field, value) in [
        ("dataset_rms", json!(2.0)),
        ("target_rms", json!(0.5)),
        ("dropout", json!(0.0)),
        ("heads", json!(8)),
    ] {
        let mut forged = receipt.clone();
        forged["checkpoint"]["config"][field] = value;
        let binding = parse_receipt(&forged).unwrap();
        assert_eq!(binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
        assert_refused(binding.require_accepted_identity(), "config hashes to");
    }
    // Another carried vocabulary label is refused.
    let mut relabelled = receipt;
    relabelled["checkpoint"]["carried_bindings"]["vocabulary_sha256"] =
        json!(PRODUCT_VOCABULARY_SHA256);
    assert_refused(
        parse_receipt(&relabelled)
            .unwrap()
            .require_accepted_identity(),
        "vocabulary",
    );
}

#[test]
fn vocabulary_is_judged_by_its_bytes_not_its_label() {
    let bound = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    assert_eq!(bound.sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(*bound.identity(), VocabularyIdentity::Checkpoint);
    // Different bytes bound against their own hash: bind_vocabulary accepts the pair, but the
    // recorded hash is the real one, and proposing with it is refused.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let edited_sha = sha256_hex(&edited);
    let forged = bind_vocabulary(&edited, &edited_sha).unwrap();
    assert_eq!(forged.sha256(), edited_sha);
    assert_eq!(forged.vocab(), bound.vocab());
    let model = tiny_model(8);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(1);
    // Under the step-60000 checkpoint label the bytes are refused.
    assert_refused(
        propose_unbound(
            &model,
            &tiny_checkpoint(),
            &forged,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options,
        ),
        "neither the step-60000 vocabulary",
    );
    // A checkpoint that carries the forged label is refused before the vocabulary is used.
    let relabelled = CheckpointBinding::unbound_for_testing(tiny_config(), &edited_sha);
    assert_refused(
        propose_unbound(
            &model,
            &relabelled,
            &forged,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options,
        ),
        "is not the step-60000 vocabulary",
    );
    // So is a checkpoint carrying the product hash as its own vocabulary, with either file.
    let product_labelled =
        CheckpointBinding::unbound_for_testing(tiny_config(), PRODUCT_VOCABULARY_SHA256);
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    for vocabulary in [&bound, &product] {
        assert_refused(
            propose_unbound(
                &model,
                &product_labelled,
                vocabulary,
                &ctx,
                &run_binding(&ctx),
                &seeds,
                &options,
            ),
            "is not the step-60000 vocabulary",
        );
    }
    // The genuine bytes pass the same path.
    assert!(propose_unbound(
        &model,
        &tiny_checkpoint(),
        &bound,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_ok());
}

#[test]
fn grammar_round_trips_programs() {
    let vocab = base_vocab();
    for program in GENERATION_11_PROGRAMS {
        let tokens = vocab.encode(program).unwrap();
        assert_eq!(tokens[..3], [BOS, 3, 397]);
        assert_eq!(tokens.last(), Some(&EOS));
        assert_eq!(&vocab.decode(&tokens).unwrap(), program);
        let legal = vocab.replay(&tokens).unwrap();
        for (set, token) in legal.iter().zip(&tokens[1..]) {
            assert!(set.contains(token));
        }
    }
    // Grammar-uniform programs: decode(encode(p)) == p and encode(decode(t)) == t.
    let mut checked = 0;
    for seed in 0..300u64 {
        let mut rng = <rand_chacha::ChaCha8Rng as rand::SeedableRng>::seed_from_u64(seed);
        if let Some(program) =
            decode::grammar_uniform_program(&vocab, "regression", &mut rng).unwrap()
        {
            let tokens = vocab.encode(&program).unwrap();
            assert_eq!(vocab.decode(&tokens).unwrap(), program);
            feat_math::GenomeIr::parse(&program).unwrap();
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} uniform programs terminated");
    // Programs outside the frozen vocabulary are refused rather than snapped.
    assert!(vocab.encode("(pipe X (ridge alpha=0.123456))").is_err());
    assert!(vocab.encode("(pipe X (not_an_op))").is_err());
    assert!(vocab
        .project_proposal("(pipe X (ridge alpha=0.123456))")
        .is_ok());
    assert!(vocab.decode(&[BOS, 3, 397, EOS]).is_err());
}

// ---------------------------------------------------------------- model and decode

#[test]
fn model_refuses_a_changed_tensor_set() {
    let config = tiny_config();
    let model = tiny_model(1);
    let inventory = model.tensor_inventory().unwrap();
    // One decoder layer: 136 - 5 * 12 tensors.
    assert_eq!(inventory.len(), 76);
    assert_eq!(inventory.len(), expected_inventory(&config).len());
    let tensors = expected_inventory(&config)
        .into_iter()
        .map(|(name, shape)| {
            let t = candle_zeros(&shape);
            (name, t)
        })
        .collect::<Vec<_>>();
    assert!(PolicyModel::from_tensors(config.clone(), tensors.clone(), cpu()).is_ok());
    let mut missing = tensors.clone();
    missing.retain(|(n, _)| n != "reward_encoder.4.bias");
    assert!(PolicyModel::from_tensors(config.clone(), missing, cpu()).is_err());
    let mut extra = tensors.clone();
    extra.push(("runtime_head.4.bias".into(), candle_zeros(&[1])));
    assert!(PolicyModel::from_tensors(config.clone(), extra, cpu()).is_err());
    let mut reshaped = tensors;
    for (name, t) in reshaped.iter_mut() {
        if name == "z_proj.bias" {
            *t = candle_zeros(&[15]);
        }
    }
    assert!(PolicyModel::from_tensors(config, reshaped, cpu()).is_err());
}

fn cpu() -> candle_core::Device {
    candle_core::Device::Cpu
}

fn candle_zeros(shape: &[usize]) -> candle_core::Tensor {
    candle_core::Tensor::zeros(shape, candle_core::DType::F32, &cpu()).unwrap()
}

#[test]
fn decoder_masks_padding_and_future_tokens() {
    let model = tiny_model(2);
    let ctx = context();
    let program = base_vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    // Logits at position t depend only on tokens before t (PAD keys: `decoder_ignores_pad_keys`).
    let full = model
        .logits(&ctx, std::slice::from_ref(&program))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let prefix = program[..6].to_vec();
    let short = model
        .logits(&ctx, std::slice::from_ref(&prefix))
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    for t in 0..=prefix.len() {
        for (a, b) in full[0][t].iter().zip(&short[0][t]) {
            assert!((a - b).abs() < 1e-4, "position {t}: {a} vs {b}");
        }
    }
    let mut padded = prefix.clone();
    padded.extend([PAD; 3]);
    let with_pad = model.next_token_logits(&ctx, &[padded]).unwrap();
    assert_eq!(with_pad[0].len(), 401);
    let rewards_plain = model
        .reward_runtime(&ctx, std::slice::from_ref(&program))
        .unwrap();
    let mut longer = program.clone();
    longer.extend([PAD; 5]);
    let rewards_padded = model.reward_runtime(&ctx, &[longer]).unwrap();
    for (a, b) in rewards_plain.0[0].iter().zip(&rewards_padded.0[0]) {
        assert!((a - b).abs() < 1e-5);
    }
    assert!(rewards_plain.1[0] >= 0.0);
}

/// PAD keys are invisible: two models that differ only in the PAD embedding row give the same
/// logits at every position whose query is not PAD, including positions after an interior PAD
/// and positions of right-padded rows (the PAD logit column itself is excluded, because the
/// output projection is tied to `token_emb`).
#[test]
fn decoder_ignores_pad_keys() {
    let plain = PolicyModel::from_tensors(tiny_config(), tiny_tensors(0.0), cpu()).unwrap();
    let shifted = PolicyModel::from_tensors(tiny_config(), tiny_tensors(3.0), cpu()).unwrap();
    let ctx = context();
    let program = base_vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    assert!(program.len() > 8);
    let mut interior = program.clone();
    interior[4] = PAD;
    let mut padded = program[..6].to_vec();
    padded.extend([PAD; 3]);
    // Pad the interior row to the padded row's length (or vice versa) so both share a batch.
    let length = interior.len().max(padded.len());
    interior.resize(length, PAD);
    padded.resize(length, PAD);
    let rows = vec![interior, padded];
    let a = plain.logits(&ctx, &rows).unwrap().to_vec3::<f32>().unwrap();
    let b = shifted
        .logits(&ctx, &rows)
        .unwrap()
        .to_vec3::<f32>()
        .unwrap();
    let mut pad_query_moved = false;
    for (r, row) in rows.iter().enumerate() {
        // Position t + 1 is the query for token t; position 0 is the context query.
        for t in 0..=row.len() {
            let pad_query = t > 0 && row[t - 1] == PAD;
            let moved = a[r][t][1..]
                .iter()
                .zip(&b[r][t][1..])
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            if pad_query {
                pad_query_moved |= moved > 1e-4;
            } else {
                assert!(moved < 1e-6, "row {r} position {t} moved by {moved}");
            }
        }
    }
    // The shift is large enough to matter wherever PAD is actually read.
    assert!(pad_query_moved);
}

#[test]
fn pure_pool_counts_only_sampled_tokens() {
    let vocab = base_vocab();
    let model = tiny_model(3);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 24,
        temperature: 25.0,
    };
    let (pool, _) = pure_pool_unbound(&model, &vocab, &ctx, "regression", 7, 42, &options).unwrap();
    // The first round replays from the same ChaCha20 seed.
    let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(7);
    let rows =
        decode::sample_batch(&model, &vocab, &ctx, "regression", 24, 25.0, &mut rng).unwrap();
    let drawn = rows
        .iter()
        .map(|row| {
            let end = row.program.as_ref().map_or(row.tokens.len(), Vec::len);
            assert!(row.tokens[3..end].iter().all(|&t| t != PAD));
            assert!(row.tokens[end..].iter().all(|&t| t == PAD));
            end - 3
        })
        .sum::<usize>();
    let padded = rows.iter().map(|row| row.tokens.len() - 3).sum::<usize>();
    assert_eq!(pool.sampled_tokens, drawn);
    // At least one row finished early, so padded positions would have been over-counted.
    assert!(drawn < padded, "{drawn} vs {padded}");
}

#[test]
fn every_sampled_token_is_legal() {
    let vocab = base_vocab();
    let model = tiny_model(3);
    let ctx = context();
    let mut rng = <rand_chacha::ChaCha20Rng as rand::SeedableRng>::seed_from_u64(7);
    let rows = decode::sample_batch(&model, &vocab, &ctx, "regression", 24, 1.0, &mut rng).unwrap();
    assert_eq!(rows.len(), 24);
    let length = rows[0].tokens.len();
    let mut finished = 0;
    for row in &rows {
        assert_eq!(row.tokens.len(), length);
        assert_eq!(row.tokens[..3], [BOS, 3, 397]);
        let mut state = vocab.prefix_state("regression").unwrap();
        let mut ended = false;
        for &token in &row.tokens[3..] {
            if ended {
                assert_eq!(token, PAD, "finished rows are padded");
                continue;
            }
            // Sampling-mode legal set: generate=false operators must never appear.
            assert!(state.legal(&vocab).contains(&token), "illegal {token}");
            state.advance(&vocab, token).unwrap();
            ended = state.done();
        }
        assert_eq!(ended, row.program.is_some());
        if let Some(program) = &row.program {
            finished += 1;
            vocab.replay(program).unwrap();
            if let Some(genome) = decode::valid_genome(&vocab, program) {
                assert_eq!(&vocab.encode(&genome).unwrap(), program);
            }
        }
    }
    assert!(finished > 0);
    // Illegal tokens with overwhelming logits are still never drawn.
    let mut logits = vec![1e30f32; 401];
    let legal = [60usize, 61, 62];
    for i in legal {
        logits[i] = -5.0;
    }
    for _ in 0..200 {
        let token = decode::sample_legal(&logits, &legal, 1.0, &mut rng).unwrap();
        assert!(legal.contains(&token));
    }
    assert!(decode::sample_legal(&logits, &[], 1.0, &mut rng).is_err());
    logits[60] = f32::NAN;
    assert!(decode::sample_legal(&logits, &legal, 1.0, &mut rng).is_err());
}

#[test]
fn pure_pool_is_deterministic_and_honest_about_origins() {
    let vocab = base_vocab();
    let model = tiny_model(4);
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 16,
        temperature: 1.0,
    };
    let run = || pure_pool_unbound(&model, &vocab, &ctx, "regression", 11, 12, &options).unwrap();
    let (pool, seen) = run();
    let (again, _) = run();
    assert_eq!(pool.candidates(), again.candidates());
    assert_eq!(pool.candidates().len(), 64);
    assert_eq!(pool.attempted, 16);
    assert!(pool.unique_neural <= pool.valid && pool.valid <= 16);
    let neural = pool
        .candidates()
        .iter()
        .filter(|c| c.origin() == ORIGIN_NEURAL)
        .count();
    assert_eq!(neural, pool.unique_neural);
    assert!(pool.candidates()[neural..]
        .iter()
        .all(|c| c.origin() == ORIGIN_NEURAL_FALLBACK));
    assert!(pool.candidates().iter().all(|c| seen.contains(c.genome())));
    // The production pool refuses untrained weights: its samples would be labelled neural.
    let err = decode::pure_pool(&model, &vocab, &ctx, "regression", 11, 12, &options).unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
}

// ---------------------------------------------------------------- ranking

fn prediction(program: &str, quality: f64) -> Prediction {
    Prediction {
        program: program.into(),
        program_sha256: sha256_hex(program.as_bytes()),
        token_count: 5,
        predicted_quality: quality,
        predicted_runtime_seconds: 2.0,
        reward_heads: [quality; 4],
    }
}

#[test]
fn ranking_ties_break_by_program_hash_then_text() {
    let mut better = prediction("better", 2.0);
    better.predicted_runtime_seconds = 999.0;
    let a = prediction("a", 1.0);
    let b = prediction("b", 1.0);
    let negative_zero = prediction("nz", -0.0);
    let zero = prediction("z", 0.0);
    let first = sorted_predictions(vec![
        a.clone(),
        better.clone(),
        b.clone(),
        zero.clone(),
        negative_zero.clone(),
    ])
    .unwrap();
    let second = sorted_predictions(vec![negative_zero, b, zero, a, better]).unwrap();
    assert_eq!(first, second);
    assert_eq!(first[0].program, "better");
    assert!(first[1].program_sha256 < first[2].program_sha256);
    assert!(first[3].program_sha256 < first[4].program_sha256);
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(sorted_predictions(vec![prediction("bad", value)]).is_err());
        let mut row = prediction("bad runtime", 1.0);
        row.predicted_runtime_seconds = value;
        assert!(sorted_predictions(vec![row]).is_err());
    }
    let mut row = prediction("negative runtime", 1.0);
    row.predicted_runtime_seconds = -1.0;
    assert!(sorted_predictions(vec![row]).is_err());
}

fn uniform_pool(vocab: &Vocab, n: usize, seed: u64) -> Vec<String> {
    decode::uniform_programs(vocab, "regression", seed, n, &mut BTreeSet::new()).unwrap()
}

/// Scoring bindings for untrained weights (the unbound test state).
fn bindings_for(ctx: &[f32]) -> Bindings {
    Bindings::new(
        UNBOUND_TEST_STATE_SHA256,
        &context_sha256(ctx),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap()
}

#[test]
fn score_pool_order_is_deterministic_and_documented() {
    let vocab = base_vocab();
    let model = tiny_model(5);
    let ctx = context();
    let mut programs = uniform_pool(&vocab, 64, 900);
    programs.extend(GENERATION_11_PROGRAMS[1..].iter().map(|p| p.to_string()));
    let options = Options {
        batch_size: 16,
        keep: 64,
    };
    let ranking = score_pool(
        &model,
        &vocab,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &options,
    )
    .unwrap();
    assert_eq!(ranking.selection_id(), SELECTION_ID);
    assert_eq!(ranking.pool_size(), programs.len());
    assert_eq!(ranking.ranked().len(), programs.len());
    assert_eq!(ranking.selected().len(), 64);
    assert_eq!(ranking.scoring_batches(), programs.len().div_ceil(16));
    for pair in ranking.ranked().windows(2) {
        assert_ne!(compare_predictions(&pair[0], &pair[1]), Ordering::Greater);
        if pair[0].predicted_quality == pair[1].predicted_quality {
            assert!(pair[0].program_sha256 < pair[1].program_sha256);
        }
    }
    for p in ranking.ranked() {
        let mean = p.reward_heads.iter().map(|v| *v as f32).sum::<f32>() / 4.0;
        assert_eq!(p.predicted_quality, mean as f64);
        assert_eq!(p.token_count, vocab.encode(&p.program).unwrap().len());
    }
    // Pool order and scoring batch size do not change the result.
    let mut reversed = programs.clone();
    reversed.reverse();
    let again = score_pool(
        &model,
        &vocab,
        &ctx,
        &reversed,
        bindings_for(&ctx),
        &options,
    )
    .unwrap();
    assert_eq!(
        serde_json::to_vec(&ranking).unwrap(),
        serde_json::to_vec(&again).unwrap()
    );
    assert_eq!(ranking.pool_sha256(), pool_sha256(&reversed).unwrap());
    let single = score_pool(
        &model,
        &vocab,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &Options {
            batch_size: 1,
            keep: 64,
        },
    )
    .unwrap();
    let order = |r: &Ranking| {
        r.ranked()
            .iter()
            .map(|p| p.program.clone())
            .collect::<Vec<_>>()
    };
    assert_eq!(order(&ranking), order(&single));
    // Wrong context identity, duplicates and unencodable programs are refused.
    let wrong = Bindings::new(
        UNBOUND_TEST_STATE_SHA256,
        &digest("other context"),
        CHECKPOINT_VOCABULARY_SHA256,
    )
    .unwrap();
    assert!(score_pool(&model, &vocab, &ctx, &programs, wrong, &options).is_err());
    // Untrained weights are never stamped with the accepted state: the constructor refuses it,
    // and bindings that claim it (only deserializable under test support) are refused at scoring.
    assert!(Bindings::new(
        ACCEPTED_STATE_SHA256,
        &context_sha256(&ctx),
        CHECKPOINT_VOCABULARY_SHA256
    )
    .is_err());
    let err = Bindings::for_accepted_policy(
        &model,
        &context_sha256(&ctx),
        &bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap(),
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    let err = score_pool(&model, &vocab, &ctx, &programs, claimed, &options).unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    let mut duplicated = programs.clone();
    duplicated.push(programs[0].clone());
    assert!(score_pool(
        &model,
        &vocab,
        &ctx,
        &duplicated,
        bindings_for(&ctx),
        &options
    )
    .is_err());
    let snapped = vec!["(pipe X (ridge alpha=0.123456))".to_string()];
    let one = Options {
        batch_size: 1,
        keep: 1,
    };
    assert!(score_pool(&model, &vocab, &ctx, &snapped, bindings_for(&ctx), &one).is_err());
}

// ---------------------------------------------------------------- seed writer

fn fake_ranking(state: &str, binding: &SeedBinding, programs: &[String]) -> Ranking {
    ranking_for_testing(
        Bindings::new(
            state,
            binding.context_sha256(),
            CHECKPOINT_VOCABULARY_SHA256,
        )
        .unwrap(),
        programs
            .iter()
            .enumerate()
            .map(|(i, p)| prediction(p, -(i as f64)))
            .collect(),
    )
}

/// The test binding for untrained weights (state `UNBOUND_TEST_STATE_SHA256`).
fn tiny_checkpoint() -> CheckpointBinding {
    CheckpointBinding::unbound_for_testing(tiny_config(), CHECKPOINT_VOCABULARY_SHA256)
}

fn run_binding(ctx: &[f32]) -> SeedBinding {
    SeedBinding::new(
        &digest("prepared.json"),
        0,
        &digest("cut-0/train.csv"),
        &context_sha256(ctx),
    )
    .unwrap()
}

/// The candidate and population shape jope-dime `gp.rs:551-574` deserializes.
fn assert_gp_seed_shape(value: &Value) {
    let object = value.as_object().unwrap();
    assert_eq!(
        object.keys().cloned().collect::<BTreeSet<_>>(),
        [
            "schema",
            "prepared_sha256",
            "cut",
            "training_sha256",
            "context_sha256",
            "checkpoint_sha256",
            "candidates"
        ]
        .into_iter()
        .map(String::from)
        .collect()
    );
    assert_eq!(value["schema"], "parallel.seed_population.v1");
    assert!(value["cut"].is_u64());
    for key in [
        "prepared_sha256",
        "training_sha256",
        "context_sha256",
        "checkpoint_sha256",
    ] {
        let s = value[key].as_str().unwrap();
        assert!(s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit()));
    }
    let candidates = value["candidates"].as_array().unwrap();
    assert_eq!(candidates.len(), 80);
    for c in candidates {
        let keys = c
            .as_object()
            .unwrap()
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>();
        assert!(keys.contains("genome") && keys.contains("origin"));
        assert!(keys
            .iter()
            .all(|k| ["genome", "origin", "rank_log_weight"].contains(&k.as_str())));
        assert!(c["genome"].is_string() && c["origin"].is_string());
    }
}

#[test]
fn seed_writer_produces_the_population_search_accepts() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let model = tiny_model(6);
    let ctx = context();
    let checkpoint = tiny_checkpoint();
    let binding = run_binding(&ctx);
    let options = SamplingOptions {
        rounds: 2,
        batch: 16,
        temperature: 1.0,
    };
    let proposal = propose_unbound(
        &model,
        &checkpoint,
        &vocab,
        &ctx,
        &binding,
        &ProposalSeeds::derive(20260917),
        &options,
    )
    .unwrap();
    let population = proposal.population();
    // Untrained weights are stamped with the test state, never a checkpoint state.
    assert_eq!(population.checkpoint_sha256(), UNBOUND_TEST_STATE_SHA256);
    assert_eq!(
        proposal.ranking().bindings().checkpoint_sha256(),
        UNBOUND_TEST_STATE_SHA256
    );
    let origins = population.origins();
    assert_eq!(
        origins.get(ORIGIN_NEURAL).copied().unwrap_or(0)
            + origins.get(ORIGIN_NEURAL_FALLBACK).copied().unwrap_or(0),
        64
    );
    assert_eq!(
        origins.get(ORIGIN_NEURAL).copied().unwrap_or(0),
        proposal.pool().unique_neural
    );
    assert_eq!(origins[ORIGIN_RANDOM], 16);
    assert!(population.candidates()[..64]
        .iter()
        .all(|c| c.origin() != ORIGIN_RANDOM && c.rank_log_weight().is_some()));
    assert!(population.candidates()[64..]
        .iter()
        .all(|c| c.origin() == ORIGIN_RANDOM && c.rank_log_weight().is_none()));
    // Seed order is ranking order; weights are a normalized log-softmax of predicted quality.
    for (candidate, p) in population
        .candidates()
        .iter()
        .zip(proposal.ranking().selected())
    {
        assert_eq!(candidate.genome(), p.program);
    }
    let total = population.candidates()[..64]
        .iter()
        .map(|c| c.rank_log_weight().unwrap().exp())
        .sum::<f64>();
    assert!((total - 1.0).abs() < 1e-9);

    let dir = std::env::temp_dir().join(format!("jope-dime-seeds-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("seeds.json");
    let sha = write_population(&path, &proposal).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(sha, sha256_hex(&bytes));
    assert_eq!(bytes, serde_json::to_vec_pretty(population).unwrap());
    assert_eq!(bytes, seeds::population_bytes(&proposal).unwrap());
    assert_gp_seed_shape(&serde_json::from_slice(&bytes).unwrap());

    let admitted = validate_for_search(
        &bytes,
        binding.prepared_sha256(),
        binding.cut(),
        binding.training_sha256(),
    )
    .unwrap();
    assert_eq!(&admitted, population);
    assert!(admitted.claims_neural() == (proposal.pool().unique_neural > 0));
    // Wrong run identity is the same refusal jope-dime's search gives.
    for (prepared, cut, training) in [
        (
            digest("other prepared.json"),
            0,
            binding.training_sha256().to_string(),
        ),
        (
            binding.prepared_sha256().to_string(),
            1,
            binding.training_sha256().to_string(),
        ),
        (
            binding.prepared_sha256().to_string(),
            0,
            digest("other train"),
        ),
    ] {
        let err = validate_for_search(&bytes, &prepared, cut, &training).unwrap_err();
        assert!(format!("{err:#}").contains("identity mismatch"), "{err:#}");
    }
    // Legacy `policy_logprob` spelling is read; a file with both spellings is refused.
    let mut legacy: Value = serde_json::from_slice(&bytes).unwrap();
    let weight = legacy["candidates"][0]["rank_log_weight"].take();
    legacy["candidates"][0]
        .as_object_mut()
        .unwrap()
        .remove("rank_log_weight");
    legacy["candidates"][0]["policy_logprob"] = weight.clone();
    let legacy_bytes = serde_json::to_vec(&legacy).unwrap();
    let admit = |bytes: &[u8]| {
        validate_for_search(
            bytes,
            binding.prepared_sha256(),
            0,
            binding.training_sha256(),
        )
    };
    assert!(admit(&legacy_bytes).is_ok());
    legacy["candidates"][0]["rank_log_weight"] = weight;
    let both = serde_json::to_vec(&legacy).unwrap();
    assert!(admit(&both).is_err());
    // Count, duplicates and the rejected checkpoint are refused.
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    let mut short = original.clone();
    short["candidates"].as_array_mut().unwrap().pop();
    assert!(admit(&serde_json::to_vec(&short).unwrap()).is_err());
    let mut duplicate = original.clone();
    duplicate["candidates"][79]["genome"] = original["candidates"][0]["genome"].clone();
    assert!(admit(&serde_json::to_vec(&duplicate).unwrap()).is_err());
    let mut rejected = original.clone();
    rejected["checkpoint_sha256"] = json!(REJECTED_STATE_SHA256);
    assert!(admit(&serde_json::to_vec(&rejected).unwrap()).is_err());
    assert!(admit(&serde_json::to_vec(&original).unwrap()).is_ok());
}

#[test]
fn propose_refuses_unbound_inputs() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let model = tiny_model(8);
    let ctx = context();
    let checkpoint = tiny_checkpoint();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(1);
    // Malformed run identity, wrong context, other vocabulary binding.
    assert!(SeedBinding::new("not-a-digest", 0, &digest("t"), &context_sha256(&ctx)).is_err());
    let bad = SeedBinding::new(&digest("p"), 0, &digest("t"), &digest("other context")).unwrap();
    assert!(propose_unbound(&model, &checkpoint, &vocab, &ctx, &bad, &seeds, &options).is_err());
    let other_vocab =
        CheckpointBinding::unbound_for_testing(tiny_config(), &digest("other vocabulary"));
    for vocabulary in [&vocab, &product] {
        assert!(propose_unbound(
            &model,
            &other_vocab,
            vocabulary,
            &ctx,
            &run_binding(&ctx),
            &seeds,
            &options
        )
        .is_err());
    }
    let other_config = CheckpointBinding::unbound_for_testing(
        PolicyConfig::regression(401),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    assert!(propose_unbound(
        &model,
        &other_config,
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options
    )
    .is_err());
    // The production path refuses untrained weights outright, even with every other input valid.
    let err = propose(
        &model,
        &checkpoint,
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    // The documented product restriction is accepted.
    let proposal = propose_unbound(
        &model,
        &checkpoint,
        &product,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap();
    assert_eq!(proposal.population().candidates().len(), 80);
    assert_eq!(
        proposal.ranking().bindings().vocabulary_sha256(),
        PRODUCT_VOCABULARY_SHA256
    );
    // Assembling with a pool that differs from the ranking is refused.
    let binding = run_binding(&ctx);
    let state = UNBOUND_TEST_STATE_SHA256;
    let pool = proposal.pool().candidates();
    let programs = pool
        .iter()
        .map(|c| c.genome().to_string())
        .collect::<Vec<_>>();
    let mut seen = programs.iter().cloned().collect::<BTreeSet<_>>();
    let tail = decode::random_exploration(&base_vocab(), "regression", 77, &mut seen).unwrap();
    let ranking = fake_ranking(state, &binding, &programs);
    assert!(assemble_for_testing(&binding, state, &ranking, pool, &tail).is_ok());
    let mut other = programs.clone();
    other[3] = tail[0].genome().to_string();
    let other = fake_ranking(state, &binding, &other);
    assert!(assemble_for_testing(&binding, state, &other, pool, &tail).is_err());
    // A tail holding neural-arm candidates, or a ranking for another state, is refused.
    assert!(assemble_for_testing(&binding, state, &ranking, pool, &pool[..16]).is_err());
    let elsewhere = fake_ranking(&digest("other state"), &binding, &programs);
    assert!(assemble_for_testing(&binding, state, &elsewhere, pool, &tail).is_err());
    // Even test support never stamps the accepted state, whatever the ranking claims.
    assert!(assemble_for_testing(&binding, ACCEPTED_STATE_SHA256, &ranking, pool, &tail).is_err());
    let mut claimed = serde_json::to_value(&ranking).unwrap();
    claimed["bindings"]["checkpoint_sha256"] = json!(ACCEPTED_STATE_SHA256);
    let claimed: Ranking = serde_json::from_value(claimed).unwrap();
    for checkpoint in [ACCEPTED_STATE_SHA256, state] {
        let err = assemble_for_testing(&binding, checkpoint, &claimed, pool, &tail).unwrap_err();
        assert!(format!("{err:#}").contains("accepted state"), "{err:#}");
    }
}

/// Run identity for a production caller; the accepted state comes from the checkpoint binding.
fn accepted_run_binding(ctx: &[f32]) -> SeedBinding {
    run_binding(ctx)
}

/// Write deterministic tiny weights to `dir/policy.safetensors` and return a receipt that is
/// internally consistent with them (inventory, parameter hash, file hash) and names the
/// accepted state: everything a forged export could attest to.
fn self_attested_tiny_export(dir: &Path) -> (PathBuf, Value) {
    let config = tiny_config();
    let tensors = tiny_tensors(0.0);
    let path = dir.join("policy.safetensors");
    candle_core::safetensors::save(&tensors, &path).unwrap();
    let loaded = PolicyModel::load_safetensors(&path, config.clone(), cpu()).unwrap();
    assert!(loaded.bound_identity().is_none());
    let parameters = loaded.parameter_sha256().unwrap();
    let inventory = serde_json::to_value(loaded.tensor_inventory().unwrap()).unwrap();
    let mut receipt = synthetic_receipt(ACCEPTED_STATE_SHA256, &digest("model"), 60000);
    receipt["checkpoint"]["config"] = serde_json::to_value(&config).unwrap();
    receipt["checkpoint"]["parameter_sha256"] = json!(parameters);
    receipt["checkpoint"]["parameter_sha256_recomputed"] = json!(parameters);
    receipt["policy"]["tensor_count"] = json!(inventory.as_array().unwrap().len());
    receipt["policy"]["tensors"] = inventory;
    receipt["policy"]["output"]["sha256"] = json!(sha256_hex(&std::fs::read(&path).unwrap()));
    (path, receipt)
}

#[test]
fn propose_requires_the_accepted_weights_loaded_through_load_bound() {
    let vocab = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let ctx = context();
    let options = SamplingOptions {
        rounds: 1,
        batch: 4,
        temperature: 1.0,
    };
    let seeds = ProposalSeeds::derive(3);
    // A random-weight model with a real (parsed, accepted-state) binding is refused.
    let accepted = parse_receipt(&synthetic_receipt(
        ACCEPTED_STATE_SHA256,
        &digest("model"),
        60000,
    ))
    .unwrap();
    let random = PolicyModel::random_weights(PolicyConfig::regression(401), 11).unwrap();
    assert!(random.bound_identity().is_none());
    let err = propose(
        &random,
        &accepted,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("load_bound"), "{err:#}");
    // The unbound test path refuses the accepted binding, so it never stamps the accepted state.
    assert!(propose_unbound(
        &random,
        &accepted,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_err());

    // A self-consistent receipt around other weights passes load_bound, but the recomputed
    // parameter hash is not the accepted one, so propose still refuses.
    let dir = std::env::temp_dir().join(format!("jope-dime-forged-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (path, receipt) = self_attested_tiny_export(&dir);
    let forged = parse_receipt(&receipt).unwrap();
    let bound = PolicyModel::load_bound(&path, &forged, cpu()).unwrap();
    let other_file =
        CheckpointBinding::unbound_for_testing(tiny_config(), CHECKPOINT_VOCABULARY_SHA256);
    assert!(PolicyModel::load_bound(&path, &other_file, cpu()).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
    let identity = bound.bound_identity().unwrap();
    assert_eq!(identity.state_sha256(), ACCEPTED_STATE_SHA256);
    assert_eq!(identity.parameter_sha256(), forged.parameter_sha256());
    assert_ne!(identity.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    let err = propose(
        &bound,
        &forged,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &seeds,
        &options,
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("accepted step-60000 parameters"),
        "{err:#}"
    );
    // A bound model is never admitted by the unbound test path either.
    assert!(propose_unbound(
        &bound,
        &tiny_checkpoint(),
        &vocab,
        &ctx,
        &run_binding(&ctx),
        &seeds,
        &options,
    )
    .is_err());
    // The same forged-but-bound model cannot label a pool as neural or stamp a ranking with the
    // accepted state, and is never scored under another state either.
    let parameters_err = |result: anyhow::Result<()>| {
        let err = result.unwrap_err();
        assert!(
            format!("{err:#}").contains("accepted step-60000 parameters"),
            "{err:#}"
        );
    };
    let base = vocab.vocab();
    parameters_err(decode::pure_pool(&bound, base, &ctx, "regression", 1, 2, &options).map(drop));
    parameters_err(Bindings::for_accepted_policy(&bound, &context_sha256(&ctx), &vocab).map(drop));
    let programs = uniform_pool(base, 4, 9);
    let four = Options {
        batch_size: 4,
        keep: 4,
    };
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    parameters_err(score_pool(&bound, base, &ctx, &programs, claimed, &four).map(drop));
    let err = score_pool(&bound, base, &ctx, &programs, bindings_for(&ctx), &four).unwrap_err();
    assert!(format!("{err:#}").contains("scored under"), "{err:#}");
    // The untrained twin of that model scores fine under the test state.
    assert!(score_pool(
        &tiny_model(9),
        base,
        &ctx,
        &programs,
        bindings_for(&ctx),
        &four
    )
    .is_ok());
}

// ---------------------------------------------------------------- external evidence (ignored)

fn env_path(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var_os(name)
            .unwrap_or_else(|| panic!("set {name} to run this ignored evidence test")),
    )
}

#[test]
#[ignore = "needs JOPE_DIME_POLICY_SAFETENSORS (exported step-60000 policy)"]
fn exported_policy_matches_its_export_receipt() {
    let policy = env_path("JOPE_DIME_POLICY_SAFETENSORS");
    let receipt_path = std::env::var_os("JOPE_DIME_EXPORT_RECEIPT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            policy
                .parent()
                .and_then(Path::parent)
                .unwrap()
                .join("step-60000.receipt.json")
        });
    let binding =
        CheckpointBinding::from_receipt_bytes(&std::fs::read(&receipt_path).unwrap()).unwrap();
    assert_eq!(binding.state_sha256(), ACCEPTED_STATE_SHA256);
    assert_eq!(binding.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    assert_eq!(binding.vocabulary_sha256(), CHECKPOINT_VOCABULARY_SHA256);
    assert_eq!(binding.tensors().len(), 136);
    // Name/shape/dtype set of the receipt equals the port's inventory.
    let receipt_set = binding
        .tensors()
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone(), t.dtype.clone()))
        .collect::<BTreeSet<_>>();
    let port_set = expected_inventory(binding.config())
        .into_iter()
        .map(|(n, s)| (n, s, "F32".to_string()))
        .collect::<BTreeSet<_>>();
    assert_eq!(receipt_set, port_set);
    // Loading proves file hash, per-tensor bytes and the recomputed parameter hash.
    let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
    assert_eq!(
        model.parameter_sha256().unwrap(),
        binding.parameter_sha256()
    );
    let identity = model.bound_identity().unwrap();
    assert_eq!(identity.parameter_sha256(), ACCEPTED_PARAMETER_SHA256);
    assert_eq!(
        identity.policy_safetensors_sha256(),
        binding.policy_safetensors_sha256()
    );
    let vocab = bind_vocabulary(&product_bytes(), binding.vocabulary_sha256()).unwrap();
    let ctx = vec![0.0f32; 2048];
    let logits = model.next_token_logits(&ctx, &[vec![BOS, 3, 397]]).unwrap();
    assert_eq!(logits[0].len(), 401);
    assert!(logits[0].iter().all(|v| v.is_finite()));
    let program = vocab.vocab().encode(GENERATION_11_PROGRAMS[1]).unwrap();
    let (heads, runtime) = model.reward_runtime(&ctx, &[program]).unwrap();
    assert!(heads[0].iter().all(|v| v.is_finite()) && runtime[0].is_finite());
    // A receipt whose file hash differs is refused.
    let wrong = CheckpointBinding::unbound_for_testing(
        binding.config().clone(),
        CHECKPOINT_VOCABULARY_SHA256,
    );
    assert!(PolicyModel::load_bound(&policy, &wrong, cpu()).is_err());
    // The bound accepted policy is the one model the production proposal path admits.
    let options = SamplingOptions {
        rounds: 1,
        batch: 8,
        temperature: 1.0,
    };
    let proposal = propose(
        &model,
        &binding,
        &vocab,
        &ctx,
        &accepted_run_binding(&ctx),
        &ProposalSeeds::derive(60000),
        &options,
    )
    .unwrap();
    assert_eq!(
        proposal.population().checkpoint_sha256(),
        ACCEPTED_STATE_SHA256
    );
    assert_eq!(proposal.population().candidates().len(), 80);
    assert_eq!(
        proposal.ranking().bindings().vocabulary_sha256(),
        PRODUCT_VOCABULARY_SHA256
    );

    // The accepted config: claimed, recomputed, and (when the checkpoint directory the receipt
    // names is present) the config.json file itself.
    binding.require_accepted_identity().unwrap();
    assert_eq!(binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
    assert_eq!(binding.config().sha256().unwrap(), ACCEPTED_CONFIG_SHA256);
    assert_eq!(
        model.bound_identity().unwrap().config_sha256(),
        ACCEPTED_CONFIG_SHA256
    );
    let receipt_bytes = std::fs::read(&receipt_path).unwrap();
    let receipt: Value = serde_json::from_slice(&receipt_bytes).unwrap();
    let config_file =
        Path::new(receipt["checkpoint"]["directory"].as_str().unwrap()).join("config.json");
    match std::fs::read(&config_file) {
        Ok(bytes) => {
            assert_eq!(sha256_hex(&bytes), ACCEPTED_CONFIG_SHA256);
            assert_eq!(
                PolicyConfig::from_json(&bytes).unwrap().sha256().unwrap(),
                ACCEPTED_CONFIG_SHA256
            );
            println!(
                "config.json {} matches ACCEPTED_CONFIG_SHA256",
                config_file.display()
            );
        }
        Err(err) => println!("config.json not checked ({}: {err})", config_file.display()),
    }

    // Forged receipts around the real weights: load_bound still succeeds (the tensors and the
    // parameter hash are real), but nothing is stamped with the accepted state.
    let run = accepted_run_binding(&ctx);
    let seeds = ProposalSeeds::derive(60000);
    let base = bind_vocabulary(&base_bytes(), CHECKPOINT_VOCABULARY_SHA256).unwrap();
    let forge = |edit: &dyn Fn(&mut Value)| {
        let mut forged = receipt.clone();
        edit(&mut forged);
        let binding =
            CheckpointBinding::from_receipt_bytes(&serde_json::to_vec(&forged).unwrap()).unwrap();
        let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
        (binding, model)
    };
    // Forged config (same tensor shapes) under the accepted claim.
    let (forged_binding, forged_model) =
        forge(&|r| r["checkpoint"]["config"]["dataset_rms"] = json!(2.0));
    assert_eq!(forged_binding.config_sha256(), ACCEPTED_CONFIG_SHA256);
    assert_refused(
        propose(
            &forged_model,
            &forged_binding,
            &vocab,
            &ctx,
            &run,
            &seeds,
            &options,
        ),
        "config hashes to",
    );
    assert_refused(
        Bindings::for_accepted_policy(&forged_model, &context_sha256(&ctx), &base),
        "config hashes to",
    );
    assert_refused(
        decode::pure_pool(
            &forged_model,
            base.vocab(),
            &ctx,
            "regression",
            1,
            2,
            &options,
        ),
        "config hashes to",
    );
    // Forged config_sha256 claim around the real config.
    let (claimed_binding, claimed_model) =
        forge(&|r| r["checkpoint"]["config_sha256"] = json!(digest("config")));
    assert_refused(
        propose(
            &claimed_model,
            &claimed_binding,
            &vocab,
            &ctx,
            &run,
            &seeds,
            &options,
        ),
        "config_sha256",
    );
    // Checkpoint vocabulary label mismatch.
    for label in [
        PRODUCT_VOCABULARY_SHA256.to_string(),
        digest("other vocabulary"),
    ] {
        let (relabelled, relabelled_model) =
            forge(&|r| r["checkpoint"]["carried_bindings"]["vocabulary_sha256"] = json!(label));
        assert_refused(
            propose(
                &relabelled_model,
                &relabelled,
                &vocab,
                &ctx,
                &run,
                &seeds,
                &options,
            ),
            "vocabulary",
        );
    }
    // Wrong vocabulary bytes with the real receipt, under their own hash.
    let mut edited = base_bytes();
    edited.push(b'\n');
    let wrong = bind_vocabulary(&edited, &sha256_hex(&edited)).unwrap();
    assert_refused(
        propose(&model, &binding, &wrong, &ctx, &run, &seeds, &options),
        "neither the step-60000 vocabulary",
    );
    assert_refused(
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &wrong),
        "neither the step-60000 vocabulary",
    );
    // Bindings made for one vocabulary table cannot score with another.
    let programs = uniform_pool(base.vocab(), 4, 9);
    let four = Options {
        batch_size: 4,
        keep: 4,
    };
    let product_bindings =
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &vocab).unwrap();
    assert_refused(
        score_pool(
            &model,
            base.vocab(),
            &ctx,
            &programs,
            product_bindings,
            &four,
        ),
        "differs from the bound vocabulary",
    );
    let base_bindings =
        Bindings::for_accepted_policy(&model, &context_sha256(&ctx), &base).unwrap();
    let ranking = score_pool(&model, base.vocab(), &ctx, &programs, base_bindings, &four).unwrap();
    assert_eq!(
        ranking.bindings().checkpoint_sha256(),
        ACCEPTED_STATE_SHA256
    );
    // Deserialized accepted bindings carry no vocabulary table and are refused.
    let claimed: Bindings = serde_json::from_value(json!({
        "checkpoint_sha256": ACCEPTED_STATE_SHA256,
        "conditioning_context_sha256": context_sha256(&ctx),
        "vocabulary_sha256": CHECKPOINT_VOCABULARY_SHA256,
    }))
    .unwrap();
    assert_refused(
        score_pool(&model, base.vocab(), &ctx, &programs, claimed, &four),
        "bound step-60000 vocabulary",
    );
}

#[test]
#[ignore = "needs JOPE_DIME_GENERATION_000011 (jope-dime results directory)"]
fn generation_000011_programs_round_trip_and_rankings_follow_the_tie_break() {
    let root = env_path("JOPE_DIME_GENERATION_000011");
    let vocab = base_vocab();
    let product = bind_vocabulary(&product_bytes(), CHECKPOINT_VOCABULARY_SHA256)
        .unwrap()
        .vocab()
        .clone();
    let mut files = Vec::new();
    for dataset in std::fs::read_dir(root.join("neural")).unwrap() {
        let dataset = dataset.unwrap().path();
        if !dataset.is_dir() {
            continue;
        }
        for cut in std::fs::read_dir(&dataset).unwrap() {
            let cut = cut.unwrap().path();
            if cut.join("seeds.json").is_file() {
                files.push(cut);
            }
        }
    }
    files.sort();
    assert_eq!(
        files.len(),
        140,
        "generation-000011 has 140 neural seed files"
    );
    let (mut neural, mut product_encodable, mut others, mut others_encodable) = (0, 0, 0, 0);
    for dir in &files {
        let seeds: Value =
            serde_json::from_slice(&std::fs::read(dir.join("seeds.json")).unwrap()).unwrap();
        assert_eq!(seeds["schema"], "parallel.seed_population.v1");
        assert_eq!(seeds["checkpoint_sha256"], ACCEPTED_STATE_SHA256);
        let candidates: Vec<SeedCandidate> =
            serde_json::from_value(seeds["candidates"].clone()).unwrap();
        assert_eq!(candidates.len(), 80);
        for c in &candidates {
            if c.origin() == ORIGIN_NEURAL {
                neural += 1;
                let tokens = vocab
                    .encode(c.genome())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", c.genome()));
                assert_eq!(vocab.decode(&tokens).unwrap(), c.genome());
                feat_math::GenomeIr::parse(c.genome()).unwrap();
                if product.encode(c.genome()).is_ok() {
                    product_encodable += 1;
                }
            } else {
                others += 1;
                if let Ok(tokens) = vocab.encode(c.genome()) {
                    assert_eq!(vocab.decode(&tokens).unwrap(), c.genome());
                    others_encodable += 1;
                }
            }
        }
        let ranking: Ranking =
            serde_json::from_slice(&std::fs::read(dir.join("ranking.json")).unwrap()).unwrap();
        assert_eq!(ranking.selection_id(), SELECTION_ID);
        assert_eq!(
            ranking.bindings().checkpoint_sha256(),
            ACCEPTED_STATE_SHA256
        );
        assert_eq!(
            ranking.bindings().vocabulary_sha256(),
            CHECKPOINT_VOCABULARY_SHA256
        );
        assert_eq!(
            ranking.bindings().conditioning_context_sha256(),
            seeds["context_sha256"]
        );
        let programs = ranking
            .ranked()
            .iter()
            .map(|p| p.program.clone())
            .collect::<Vec<_>>();
        assert_eq!(ranking.pool_sha256(), pool_sha256(&programs).unwrap());
        assert_eq!(
            sorted_predictions(ranking.ranked().to_vec()).unwrap(),
            ranking.ranked()
        );
        for p in ranking.ranked() {
            assert_eq!(p.program_sha256, sha256_hex(p.program.as_bytes()));
            assert_eq!(p.token_count, vocab.encode(&p.program).unwrap().len());
        }
        let seeded = candidates[..64]
            .iter()
            .map(|c| c.genome().to_string())
            .collect::<Vec<_>>();
        assert_eq!(seeded, programs[..64]);
    }
    assert_eq!(neural, 140 * 64);
    println!(
        "generation-000011: {} files, {neural} neural programs round-tripped ({product_encodable} also in featmath-jain-v1), {others_encodable}/{others} other seeds encodable",
        files.len()
    );
}

/// Cross-precision diagnostic, not parity: re-score each generation-000011 pool with the
/// exported weights on CPU float32 and compare with the reward heads jope-dime recorded under
/// CUDA BF16 autocast. BF16 rounding bounds how close these can be, so the bound is loose; the
/// atol 3e-4 parity gate still needs float32 reference fixtures (plan v2 W4 PR-B).
#[test]
#[ignore = "needs JOPE_DIME_POLICY_SAFETENSORS, JOPE_DIME_GENERATION_000011 and JOPE_DIME_CONTEXTS"]
fn generation_000011_rewards_agree_with_bf16_records() {
    let policy = env_path("JOPE_DIME_POLICY_SAFETENSORS");
    let receipt = policy
        .parent()
        .and_then(Path::parent)
        .unwrap()
        .join("step-60000.receipt.json");
    let receipt = std::env::var_os("JOPE_DIME_EXPORT_RECEIPT")
        .map(PathBuf::from)
        .unwrap_or(receipt);
    let binding = CheckpointBinding::from_receipt_bytes(&std::fs::read(receipt).unwrap()).unwrap();
    let model = PolicyModel::load_bound(&policy, &binding, cpu()).unwrap();
    let bound_vocab = bind_vocabulary(&base_bytes(), binding.vocabulary_sha256()).unwrap();
    let vocab = bound_vocab.vocab().clone();
    let root = env_path("JOPE_DIME_GENERATION_000011");
    let contexts = env_path("JOPE_DIME_CONTEXTS");
    let (mut pools, mut max_head, mut max_quality, mut same_top, mut kendall_sum) =
        (0usize, 0f64, 0f64, 0usize, 0f64);
    let uniform = uniform_pool(&vocab, 64, 4242);
    let (mut decoder_pools, mut neural_lp, mut uniform_lp) = (0usize, 0f64, 0f64);
    for dataset in std::fs::read_dir(root.join("neural")).unwrap() {
        let dataset = dataset.unwrap().path();
        if !dataset.is_dir() {
            continue;
        }
        for cut in std::fs::read_dir(&dataset).unwrap() {
            let cut = cut.unwrap().path();
            let Ok(bytes) = std::fs::read(cut.join("ranking.json")) else {
                continue;
            };
            let recorded: Ranking = serde_json::from_slice(&bytes).unwrap();
            let context_file = contexts.join(dataset.file_name().unwrap()).join(format!(
                "{}.f32",
                cut.file_name().unwrap().to_string_lossy()
            ));
            let Ok(context_bytes) = std::fs::read(&context_file) else {
                continue;
            };
            let ctx = jope_policy::context_from_bytes(&context_bytes).unwrap();
            assert_eq!(
                context_sha256(&ctx),
                recorded.bindings().conditioning_context_sha256()
            );
            let programs = recorded
                .ranked()
                .iter()
                .map(|p| p.program.clone())
                .collect::<Vec<_>>();
            let ours = score_pool(
                &model,
                &vocab,
                &ctx,
                &programs,
                {
                    // Deserialized bindings carry no vocabulary table, so re-derive them from the
                    // bound model and vocabulary and check they name what was recorded.
                    let ours = Bindings::for_accepted_policy(
                        &model,
                        recorded.bindings().conditioning_context_sha256(),
                        &bound_vocab,
                    )
                    .unwrap();
                    assert_eq!(
                        serde_json::to_value(&ours).unwrap(),
                        serde_json::to_value(recorded.bindings()).unwrap()
                    );
                    ours
                },
                &Options::default(),
            )
            .unwrap();
            assert_eq!(ours.pool_sha256(), recorded.pool_sha256());
            let by_program = ours
                .ranked()
                .iter()
                .map(|p| (p.program.as_str(), p))
                .collect::<std::collections::BTreeMap<_, _>>();
            for r in recorded.ranked() {
                let o = by_program[r.program.as_str()];
                for (a, b) in o.reward_heads.iter().zip(&r.reward_heads) {
                    max_head = max_head.max((a - b).abs());
                }
                max_quality = max_quality.max((o.predicted_quality - r.predicted_quality).abs());
            }
            let rank = |p: &str| ours.ranked().iter().position(|x| x.program == p).unwrap();
            let order = recorded
                .ranked()
                .iter()
                .map(|p| rank(&p.program))
                .collect::<Vec<_>>();
            let (mut concordant, mut total) = (0usize, 0usize);
            for i in 0..order.len() {
                for j in i + 1..order.len() {
                    total += 1;
                    if order[i] < order[j] {
                        concordant += 1;
                    }
                }
            }
            kendall_sum += 2.0 * concordant as f64 / total as f64 - 1.0;
            if decoder_pools < 6 {
                neural_lp += mean_token_logprob(&model, &vocab, &ctx, &programs);
                uniform_lp += mean_token_logprob(&model, &vocab, &ctx, &uniform);
                decoder_pools += 1;
            }
            same_top += usize::from(order[0] == 0);
            pools += 1;
        }
    }
    assert!(
        pools > 0,
        "no generation-000011 pool had its context under JOPE_DIME_CONTEXTS"
    );
    println!(
        "bf16-vs-f32 diagnostic: {pools} pools, max |head diff| {max_head:.5}, max |quality diff| {max_quality:.5}, mean kendall tau {:.4}, same top-1 {same_top}/{pools}",
        kendall_sum / pools as f64
    );
    let (neural_lp, uniform_lp) = (
        neural_lp / decoder_pools as f64,
        uniform_lp / decoder_pools as f64,
    );
    println!(
        "decoder diagnostic: {decoder_pools} contexts, mean legal-token log-prob of recorded step-60000 samples {neural_lp:.4} vs grammar-uniform programs {uniform_lp:.4}"
    );
    assert!(
        neural_lp > uniform_lp + 0.5,
        "decoder does not prefer the checkpoint's own samples"
    );
    assert!(
        max_quality < 0.25,
        "CPU float32 rewards diverge from the BF16 records"
    );
}

/// Mean grammar-masked log-probability per generated token (after the 3-token prefix).
fn mean_token_logprob(model: &PolicyModel, vocab: &Vocab, ctx: &[f32], programs: &[String]) -> f64 {
    let encoded = programs
        .iter()
        .map(|p| vocab.encode(p).unwrap())
        .collect::<Vec<_>>();
    let length = encoded.iter().map(Vec::len).max().unwrap();
    let rows = encoded
        .iter()
        .map(|t| {
            let mut row = t.clone();
            row.resize(length, PAD);
            row
        })
        .collect::<Vec<_>>();
    let logits = model.logits(ctx, &rows).unwrap().to_vec3::<f32>().unwrap();
    let (mut total, mut count) = (0f64, 0usize);
    for (row, tokens) in encoded.iter().enumerate() {
        let legal = vocab.replay(tokens).unwrap();
        // logits[t] predicts tokens[t]; legal[t - 1] is the set offered before tokens[t].
        for t in 3..tokens.len() {
            let set = &legal[t - 1];
            let values = set
                .iter()
                .map(|&i| logits[row][t][i] as f64)
                .collect::<Vec<_>>();
            let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let log_z = max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln();
            total += logits[row][t][tokens[t]] as f64 - log_z;
            count += 1;
        }
    }
    total / count as f64
}
