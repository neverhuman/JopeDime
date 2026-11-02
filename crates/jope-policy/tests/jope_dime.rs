//! jope-dime step-60000 policy port (`feature = "testing"`), CPU only.
//!
//! Default tests use the committed vocabularies and a tiny deterministic random-weight model.
//! Ignored tests read external evidence in place:
//! - `JOPE_DIME_POLICY_SAFETENSORS`: the exported `policy.safetensors` (receipt from
//!   `JOPE_DIME_EXPORT_RECEIPT`, default `<dir>/../step-60000.receipt.json`);
//! - `JOPE_DIME_EXPORT_RECEIPT` alone, optionally with `JOPE_DIME_REEXPORT_RECEIPT` (a
//!   `jope-parallel export-policy` re-export of the accepted checkpoint): the pinned receipt
//!   field set (W2 G-08);
//! - `JOPE_DIME_GENERATION_000011`: a jope-dime `generation-000011` results directory;
//! - `JOPE_DIME_CONTEXTS`: a `contexts-assembly-v2` directory (`<dataset>/cut-N.f32`);
//! - `JOPE_DIME_CHALLENGER_SAFETENSORS`, `JOPE_DIME_CHALLENGER_RECEIPT` and
//!   `JOPE_DIME_CHALLENGER_STATE`: a native challenger export, its export receipt and the
//!   challenger checkpoint's `state.json`.
//!
//! Run the ignored tests with an optimized profile, e.g.
//! `cargo test --release -p jope-policy --features testing --test jope_dime -- --include-ignored`.
#![cfg(feature = "testing")]

use jope_policy::config::{ChallengerIdentity, CheckpointBinding, PolicyConfig};
use jope_policy::context::PolicyContext;
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
    context_sha256, sha256_hex, ACCEPTED_CONFIG_SHA256, ACCEPTED_MODEL_SHA256,
    ACCEPTED_PARAMETER_SHA256, ACCEPTED_STATE_SHA256, ACCEPTED_STEP, CHECKPOINT_VOCABULARY_SHA256,
    PRODUCT_VOCABULARY_SHA256, REJECTED_MODEL_SHA256, REJECTED_STATE_SHA256,
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

fn context() -> PolicyContext {
    PolicyContext::from_frozen_values((0..2048).map(|i| (i % 13) as f32 / 13.0 - 0.4).collect())
        .unwrap()
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

mod config_receipt;
mod decode;
mod external_generation;
mod external_policy;
mod pinned_receipts;
mod ranking;
mod seed_writer;
mod vocabulary;
pub(crate) use config_receipt::*;
pub(crate) use decode::*;
pub(crate) use external_generation::*;
pub(crate) use external_policy::*;
pub(crate) use pinned_receipts::*;
pub(crate) use ranking::*;
pub(crate) use seed_writer::*;
pub(crate) use vocabulary::*;
