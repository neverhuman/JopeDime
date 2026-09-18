//! Gen31 JOPE policy parity against the frozen PyTorch oracle.

use anyhow::{bail, Result};
use candle_core::Device;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use starforge::bundle::LoadedModelBundle;
use starforge::jope_policy::JopePolicyModel;
use starforge::JOPE_GEN31_PARITY_FIXTURE_JSON;
use std::path::{Path, PathBuf};

const MAX_ABS_TOLERANCE: f32 = 1e-5;
const FIXTURE_SHA256: &str = "9f4d3d4ee12828eb8ba0f6312615e1e807f622894123ba2f7bd52efc8c11f619";
const TRACE_SHA256: &str = "02cae5c33190f552ddabda1dd104b68e55b9491d7f8ff35d8abcc3bfe9e086ba";

#[derive(Debug, Deserialize)]
struct Fixture {
    hash: String,
    embedding_1536: Vec<f32>,
    target_512: Vec<f32>,
    z_512: Vec<f32>,
    greedy_featmath_tokens: Vec<usize>,
    greedy_featmath_genome: String,
    greedy_featmath_reward_mean: f32,
    greedy_featmath_reward_std: f32,
    greedy_featmath_p_bad: f32,
    greedy_newmath_tokens: Vec<usize>,
    greedy_newmath_genome: String,
    greedy_newmath_reward_mean: f32,
    greedy_newmath_reward_std: f32,
    greedy_newmath_p_bad: f32,
    fixed_tokens: Vec<usize>,
    next_token_logits_full: Vec<f32>,
    propose_top6: Vec<Proposal>,
    meta: Metadata,
}

#[derive(Debug, Deserialize)]
struct Metadata {
    gen: usize,
    seed: u64,
    temperature: f32,
    python: String,
    torch: String,
}

#[derive(Debug, Deserialize)]
struct Proposal {
    rank: Option<usize>,
    engine: String,
    canonical_genome: String,
    reward_mean: f32,
    reward_std: f32,
    decoder: String,
}

#[derive(Debug, Deserialize)]
struct DecoderTrace {
    schema: String,
    fixed_tokens: Vec<usize>,
    layers: Vec<Vec<f32>>,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn max_abs(actual: &[f32], expected: &[f32]) -> Result<(f32, usize)> {
    if actual.len() != expected.len() {
        bail!(
            "parity vector lengths differ: {} != {}",
            actual.len(),
            expected.len()
        );
    }
    actual.iter().zip(expected).enumerate().try_fold(
        (0.0f32, 0usize),
        |maximum, (index, (&actual, &expected))| {
            if !actual.is_finite() || !expected.is_finite() {
                bail!("JOPE parity values must be finite");
            }
            let error = (actual - expected).abs();
            Ok(if error > maximum.0 {
                (error, index)
            } else {
                maximum
            })
        },
    )
}

fn assert_scalar_close(label: &str, actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    assert!(
        error <= MAX_ABS_TOLERANCE,
        "{label} error {error:.9e} exceeds {MAX_ABS_TOLERANCE:.1e}: {actual} != {expected}"
    );
}

#[test]
fn gen31_policy_matches_frozen_full_oracle() -> Result<()> {
    assert_eq!(
        format!("{:x}", Sha256::digest(JOPE_GEN31_PARITY_FIXTURE_JSON)),
        FIXTURE_SHA256
    );
    let fixture: Fixture = serde_json::from_slice(JOPE_GEN31_PARITY_FIXTURE_JSON)?;
    assert_eq!(fixture.hash, "014b93f43dbaa1a4");
    assert_eq!((fixture.meta.gen, fixture.meta.seed), (31, 17));
    assert_eq!(fixture.meta.temperature, 0.7);
    assert_eq!(
        (fixture.meta.python.as_str(), fixture.meta.torch.as_str()),
        ("3.12.3", "2.7.0")
    );
    assert_eq!(fixture.fixed_tokens, [1, 3, 1101, 298, 51, 76]);
    assert_eq!(
        (
            fixture.embedding_1536.len(),
            fixture.target_512.len(),
            fixture.z_512.len(),
            fixture.next_token_logits_full.len()
        ),
        (1536, 512, 512, 1103)
    );

    let root = repo_root();
    let bundle = LoadedModelBundle::load(&root)?;
    assert_eq!(bundle.manifest.bundle_id, "jope-prime-gen31-v2");
    assert_eq!(
        bundle.manifest.source_checkpoint_sha256,
        "6a20443443d4d7edc75aa18ef2825c3f240ea92b02f09573c6f53c41e76f60d0"
    );
    let model = JopePolicyModel::load_bundle(&bundle, Device::Cpu)?;
    let context = model.context(&fixture.embedding_1536, &fixture.target_512)?;
    let context_values = model.context_values(&context)?;
    let (context_error, context_index) = max_abs(&context_values, &fixture.z_512)?;
    eprintln!("Gen31 JOPE context max_abs={context_error:.9e} index={context_index}");
    assert!(
        context_error <= MAX_ABS_TOLERANCE,
        "context error {context_error:.9e} exceeds {MAX_ABS_TOLERANCE:.1e}"
    );

    let oracle_context = model.context_from_values(&fixture.z_512)?;
    let trace_bytes =
        std::fs::read(root.join(
            "crates/starforge/tests/fixtures/jope_gen31_fastpath_trace_014b93f43dbaa1a4.json",
        ))?;
    assert_eq!(format!("{:x}", Sha256::digest(&trace_bytes)), TRACE_SHA256);
    let trace_fixture: DecoderTrace = serde_json::from_slice(&trace_bytes)?;
    assert_eq!(trace_fixture.schema, "jope-gen31-fastpath-trace/v1");
    assert_eq!(trace_fixture.fixed_tokens, fixture.fixed_tokens);
    let actual_trace = model.parity_decoder_trace(&oracle_context, &fixture.fixed_tokens)?;
    assert_eq!(actual_trace.len(), trace_fixture.layers.len());
    for (layer, (actual, expected)) in actual_trace.iter().zip(&trace_fixture.layers).enumerate() {
        let (error, index) = max_abs(actual, expected)?;
        eprintln!("Gen31 JOPE decoder trace layer={layer} max_abs={error:.9e} index={index}");
    }
    let oracle_norm = model
        .parity_output_norm_from_hidden(&trace_fixture.layers[trace_fixture.layers.len() - 2])?;
    let (oracle_norm_error, oracle_norm_index) = max_abs(
        &oracle_norm,
        trace_fixture.layers.last().expect("output norm trace"),
    )?;
    eprintln!(
        "Gen31 JOPE oracle-hidden output norm max_abs={oracle_norm_error:.9e} index={oracle_norm_index}"
    );
    assert!(oracle_norm_error <= MAX_ABS_TOLERANCE);
    let oracle_hidden_logits =
        model.parity_logits_from_hidden(trace_fixture.layers.last().expect("output norm trace"))?;
    let (oracle_hidden_error, oracle_hidden_index) =
        max_abs(&oracle_hidden_logits, &fixture.next_token_logits_full)?;
    eprintln!(
        "Gen31 JOPE oracle-hidden projection max_abs={oracle_hidden_error:.9e} index={oracle_hidden_index}"
    );
    assert!(oracle_hidden_error <= MAX_ABS_TOLERANCE);
    let logits = model.next_token_logits(&oracle_context, &fixture.fixed_tokens)?;
    let (logits_error, logits_index) = max_abs(&logits, &fixture.next_token_logits_full)?;
    eprintln!(
        "Gen31 JOPE fixed-prefix logits max_abs={logits_error:.9e} index={logits_index} actual={:.9e} expected={:.9e}",
        logits[logits_index], fixture.next_token_logits_full[logits_index]
    );
    assert!(
        logits_error <= MAX_ABS_TOLERANCE,
        "fixed-prefix logits error {logits_error:.9e} exceeds {MAX_ABS_TOLERANCE:.1e}"
    );

    let (featmath_mean, featmath_std) =
        model.reward_stats(&context, &fixture.greedy_featmath_tokens)?;
    assert_scalar_close(
        "featmath reward mean",
        featmath_mean,
        fixture.greedy_featmath_reward_mean,
    );
    assert_scalar_close(
        "featmath reward std",
        featmath_std,
        fixture.greedy_featmath_reward_std,
    );
    assert_scalar_close(
        "featmath p_bad",
        model.p_bad(&context, &fixture.greedy_featmath_tokens)?,
        fixture.greedy_featmath_p_bad,
    );
    let (newmath_mean, newmath_std) =
        model.reward_stats(&context, &fixture.greedy_newmath_tokens)?;
    assert_scalar_close(
        "newmath reward mean",
        newmath_mean,
        fixture.greedy_newmath_reward_mean,
    );
    assert_scalar_close(
        "newmath reward std",
        newmath_std,
        fixture.greedy_newmath_reward_std,
    );
    assert_scalar_close(
        "newmath p_bad",
        model.p_bad(&context, &fixture.greedy_newmath_tokens)?,
        fixture.greedy_newmath_p_bad,
    );

    assert_eq!(
        fixture.greedy_featmath_genome,
        "(pipe X (rank) (cheb degree=9) (cheb degree=9) (standardize) (rank) (cheb degree=6) (histgbm2 bags=3 leaves=31 lr=0.08 rounds=1000))"
    );
    assert_eq!(
        fixture.greedy_newmath_genome,
        "(pipe X (cheb degree=3) (gbm depth=4 lr=0.05 rounds=80))"
    );
    assert_eq!(fixture.propose_top6.len(), 6);
    assert_eq!(fixture.propose_top6[0].rank, None);
    assert_eq!(fixture.propose_top6[0].engine, "newmath");
    assert_eq!(fixture.propose_top6[0].decoder, "greedy");
    assert_eq!(
        fixture.propose_top6[0].canonical_genome,
        fixture.greedy_newmath_genome
    );
    assert_scalar_close(
        "top proposal reward mean",
        fixture.propose_top6[0].reward_mean,
        fixture.greedy_newmath_reward_mean,
    );
    assert_scalar_close(
        "top proposal reward std",
        fixture.propose_top6[0].reward_std,
        fixture.greedy_newmath_reward_std,
    );
    assert!(fixture.propose_top6.iter().all(|proposal| {
        matches!(proposal.engine.as_str(), "featmath" | "newmath")
            && matches!(proposal.decoder.as_str(), "greedy" | "sample")
            && proposal.canonical_genome.starts_with("(pipe X ")
            && proposal.canonical_genome.ends_with(')')
            && proposal.reward_mean.is_finite()
            && proposal.reward_std.is_finite()
    }));
    Ok(())
}

#[test]
fn gen31_oracle_provenance_binds_governed_bytes() -> Result<()> {
    let provenance_path = repo_root().join(
        "crates/starforge/tests/fixtures/jope_gen31_context_014b93f43dbaa1a4.provenance.json",
    );
    let provenance: serde_json::Value = serde_json::from_slice(&std::fs::read(provenance_path)?)?;
    assert_eq!(
        provenance["schema"],
        "jain.jope-policy-parity-provenance/v1"
    );
    assert_eq!(provenance["fixture_sha256"], FIXTURE_SHA256);
    assert_eq!(provenance["fastpath_trace_sha256"], TRACE_SHA256);
    assert_eq!(
        provenance["model_bundle_manifest_sha256"],
        "5e7884ec1c0e0b955a2381d5b69c54beddad9ec66a1980c94ad279d974486d05"
    );
    assert_eq!(
        provenance["policy_safetensors_sha256"],
        "2c4c5531c5f1d834653fb1fef8b2b51cfd9d90044ff15cc98b67cd1b5aaae871"
    );
    assert_eq!(
        provenance["frozen_generator_sha256"],
        "1a3c2bf4ccbe1173a663ca9c3bbaeee7b97bff5f968c461a29880c60f51dcc7d"
    );
    assert_eq!(
        provenance["gen31_generator_sha256"],
        "198366aa4d80951e57cd94661a75405b30b4049b2b40b2ee6e89c136f54de629"
    );
    Ok(())
}
