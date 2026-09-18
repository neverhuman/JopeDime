//! Mandatory component parity for JOPE's frozen Starlight recipe.
//!
//! This proves the raw-input foundation encoder and its intermediate tensor seam.
//! It does not prove end-to-end Prime or Lime proposal/prediction parity.

use anyhow::{bail, Context, Result};

#[path = "support/models_root.rs"]
mod models_root;
use candle_core::{DType, Device};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use starforge::bundle::LoadedModelBundle;
use starforge::foundation::{
    read_jope_train_csv, FoundationEncodeOptions, StarlightFoundationEncoder,
    JOPE_FOUNDATION_CONTEXT_ROWS, JOPE_FOUNDATION_HIDDEN_DIM, JOPE_FOUNDATION_QUERY_ROWS,
    JOPE_FOUNDATION_SEED, JOPE_FOUNDATION_VECTOR_DIM,
};
use std::io::Read;
use std::path::{Path, PathBuf};

const MAX_ABS_TOLERANCE: f32 = 1e-5;
const MAX_REL_TOLERANCE: f32 = 1e-5;
const WEIGHTS_SHA256: &str = "f663ae49cc0f1504f07800a69dd6c31ca22fc473f26bbc0684eac25129a04128";

#[derive(Debug, Deserialize)]
struct FoundationFixture {
    schema: String,
    dataset_hash: String,
    encoder_version: String,
    seed: u64,
    context_rows: usize,
    query_rows: usize,
    context_indices: Vec<usize>,
    query_indices: Vec<usize>,
    source_csv_sha256: String,
    oracle_npy_sha256: String,
    oracle_payload_sha256: String,
    encode_log_sha256: String,
    generator_sha256: String,
    fusion_experts_sha256: String,
    model_io_sha256: String,
    starlight_config_sha256: String,
    weights_sha256: String,
    oracle_f32_le_bits: String,
}

#[derive(Debug, Deserialize)]
struct JopeContextFixture {
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
    meta: JopeContextMeta,
}

#[derive(Debug, Deserialize)]
struct JopeContextMeta {
    gen: usize,
    seed: u64,
    python: String,
    torch: String,
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut file = std::fs::File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn decode_oracle(bits: &str) -> Result<Vec<f32>> {
    if bits.len() != 2048 * 8 || !bits.is_ascii() {
        bail!("foundation oracle must contain 2048 eight-digit f32 bit patterns");
    }
    bits.as_bytes()
        .chunks_exact(8)
        .map(|chunk| {
            let text = std::str::from_utf8(chunk)?;
            Ok(f32::from_bits(u32::from_str_radix(text, 16)?))
        })
        .collect()
}

fn parity_metrics(actual: &[f32], expected: &[f32]) -> Result<(f32, f32)> {
    if actual.len() != expected.len() {
        bail!("parity vector lengths differ");
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    for (&actual, &expected) in actual.iter().zip(expected) {
        if !actual.is_finite() || !expected.is_finite() {
            bail!("foundation parity vectors must be finite");
        }
        let delta = (actual - expected).abs();
        max_abs = max_abs.max(delta);
        // A unit floor gives a stable mixed absolute/relative metric near zero.
        max_rel = max_rel.max(delta / actual.abs().max(expected.abs()).max(1.0));
    }
    Ok((max_abs, max_rel))
}

#[test]
fn foundation_2048_matches_starlight_reference() -> Result<()> {
    let fixture_path = fixtures().join(concat!("foundation_", "tab", "pfn_v1_00709.json"));
    let source_path = fixtures().join(concat!("foundation_", "tab", "pfn_v1_00709_train.csv.txt"));
    let fixture: FoundationFixture = serde_json::from_slice(&std::fs::read(&fixture_path)?)?;

    assert_eq!(fixture.schema, "jain.foundation-parity/v1");
    assert_eq!(fixture.dataset_hash, "00709d315efacf0c");
    assert_eq!(
        fixture.encoder_version,
        concat!("foundation_enc.v1.", "tab", "pfn")
    );
    assert_eq!(fixture.seed, JOPE_FOUNDATION_SEED);
    assert_eq!(fixture.context_rows, 30);
    assert_eq!(fixture.query_rows, 10);
    assert_eq!(
        fixture.oracle_npy_sha256,
        "ab5906d12762ea1c82e66b1d637ea85af7feca0f1b938c2ce3d0bac617d1c26a"
    );
    assert_eq!(
        fixture.encode_log_sha256,
        "aa60ff08fad5b1a0405cc14a0f00914dd2c538d0edb6487704d8c8bb952963fd"
    );
    assert_eq!(
        fixture.generator_sha256,
        "b00f2a120f9d0e078cf44de2d49132479fbf9c5ae45d79f34093c01a2cf361be"
    );
    assert_eq!(
        fixture.fusion_experts_sha256,
        "1edcdb276a55c618f202a837108c77a86122ba2d617c6d0444d004ae60c10bbc"
    );
    assert_eq!(
        fixture.model_io_sha256,
        "5e142384c871f4df8e59271e5998524ea1806e8a8ea2f240ea9a8190d8e28805"
    );
    assert_eq!(
        fixture.starlight_config_sha256,
        "4acacec8c148702e07147452e3e13dad586bd40f90878a3bbac8355753978acb"
    );
    assert_eq!(fixture.weights_sha256, WEIGHTS_SHA256);
    assert_eq!(sha256_path(&source_path)?, fixture.source_csv_sha256);
    let Some(root) = models_root::bundle_root("foundation_2048_matches_starlight_reference") else {
        return Ok(());
    };
    let weights_path = root.join("artifacts/starforge/starlight_core_regressor.safetensors");
    assert_eq!(sha256_path(&weights_path)?, fixture.weights_sha256);

    let expected = decode_oracle(&fixture.oracle_f32_le_bits)?;
    let payload = expected
        .iter()
        .flat_map(|value| value.to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        format!("{:x}", Sha256::digest(&payload)),
        fixture.oracle_payload_sha256
    );
    assert!(expected.iter().all(|value| value.is_finite()));

    let (input, context_indices, query_indices) = read_jope_train_csv(&source_path)?;
    assert_eq!(context_indices, fixture.context_indices);
    assert_eq!(query_indices, fixture.query_indices);
    assert_eq!(input.context_x.len(), fixture.context_rows);
    assert_eq!(input.context_y.len(), fixture.context_rows);
    assert_eq!(input.query_x.len(), fixture.query_rows);
    assert!(input
        .context_x
        .iter()
        .chain(&input.query_x)
        .flatten()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));

    let bundle = LoadedModelBundle::load(&root).context("load governed model bundle")?;
    let encoder = StarlightFoundationEncoder::load_bundle(&bundle, Device::Cpu, DType::F32)
        .context("load verified f663 Starlight buffer")?;
    let actual = encoder.encode_with_intermediates(
        &input,
        FoundationEncodeOptions {
            max_context_rows: JOPE_FOUNDATION_CONTEXT_ROWS,
            max_query_rows: JOPE_FOUNDATION_QUERY_ROWS,
            seed: JOPE_FOUNDATION_SEED,
        },
    )?;
    assert_eq!(actual.context_rows, fixture.context_rows);
    assert_eq!(actual.query_rows, fixture.query_rows);
    assert_eq!(actual.hidden_dim, JOPE_FOUNDATION_HIDDEN_DIM);
    assert_eq!(
        actual.query_hidden.len(),
        fixture.query_rows * JOPE_FOUNDATION_HIDDEN_DIM
    );
    assert!(actual.query_hidden.iter().all(|value| value.is_finite()));
    assert_eq!(actual.vector.len(), JOPE_FOUNDATION_VECTOR_DIM);
    let reconstructed = actual
        .mean
        .iter()
        .chain(&actual.std)
        .chain(&actual.p10)
        .chain(&actual.p90)
        .copied()
        .collect::<Vec<_>>();
    assert_eq!(actual.vector, reconstructed);
    eprintln!(
        "Starlight first values actual={:?} expected={:?}",
        &actual.vector[..4],
        &expected[..4]
    );
    for (name, block, range) in [
        ("mean", actual.mean.as_slice(), 0..512),
        ("std", actual.std.as_slice(), 512..1024),
        ("p10", actual.p10.as_slice(), 1024..1536),
        ("p90", actual.p90.as_slice(), 1536..2048),
    ] {
        let (block_abs, block_rel) = parity_metrics(block, &expected[range])?;
        eprintln!("Starlight {name} max_abs={block_abs:.9e} max_rel={block_rel:.9e}");
    }
    let (max_abs, max_rel) = parity_metrics(&actual.vector, &expected)?;
    eprintln!("Starlight foundation parity max_abs={max_abs:.9e} max_rel={max_rel:.9e}");
    assert!(
        max_abs <= MAX_ABS_TOLERANCE,
        "max_abs {max_abs:.9e} exceeds {MAX_ABS_TOLERANCE:.1e}"
    );
    assert!(
        max_rel <= MAX_REL_TOLERANCE,
        "max_rel {max_rel:.9e} exceeds {MAX_REL_TOLERANCE:.1e}"
    );
    Ok(())
}

#[test]
fn committed_gen18_full_context_is_exact_and_complete() -> Result<()> {
    let path = fixtures().join("jope_gen18_context_0bfe8ef7abefb869.json");
    let bytes = std::fs::read(path)?;
    let source_bytes = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
    assert_eq!(
        format!("{:x}", Sha256::digest(source_bytes)),
        "77e23d521bf8e5bd9a873d6fb4580133a0e2fb1c94161693ae3d8c74e205a1e4"
    );
    let fixture: JopeContextFixture = serde_json::from_slice(&bytes)?;
    assert_eq!(fixture.hash, "0bfe8ef7abefb869");
    assert_eq!((fixture.meta.gen, fixture.meta.seed), (18, 17));
    assert_eq!(
        (fixture.meta.python.as_str(), fixture.meta.torch.as_str()),
        ("3.12.3", "2.7.0")
    );
    assert_eq!(fixture.embedding_1536.len(), 1536);
    assert_eq!(fixture.target_512.len(), 512);
    assert_eq!(fixture.z_512.len(), 512);
    assert_eq!(fixture.next_token_logits_full.len(), 1103);
    assert_eq!(fixture.fixed_tokens, [1, 3, 1101, 600, 606, 610]);
    assert_eq!(fixture.greedy_featmath_tokens.first(), Some(&1));
    assert_eq!(fixture.greedy_featmath_tokens.last(), Some(&2));
    assert_eq!(fixture.greedy_newmath_tokens.first(), Some(&1));
    assert_eq!(fixture.greedy_newmath_tokens.last(), Some(&2));
    assert_eq!(
        fixture.greedy_featmath_genome,
        "(pipe X (gpr kernel=rq+linear m=300 steps=40))"
    );
    assert_eq!(fixture.greedy_newmath_genome, "(pipe X (rank) (ridge))");
    assert!(fixture
        .embedding_1536
        .iter()
        .chain(&fixture.target_512)
        .chain(&fixture.z_512)
        .chain(&fixture.next_token_logits_full)
        .copied()
        .chain([
            fixture.greedy_featmath_reward_mean,
            fixture.greedy_featmath_reward_std,
            fixture.greedy_featmath_p_bad,
            fixture.greedy_newmath_reward_mean,
            fixture.greedy_newmath_reward_std,
            fixture.greedy_newmath_p_bad,
        ])
        .all(f32::is_finite));
    Ok(())
}
