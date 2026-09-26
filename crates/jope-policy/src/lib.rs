//! Candle port of the jope-dime step-60000 proposal policy (plan v2 W4 PR-C).
//!
//! Reference implementation: hosted `veox/jope-dime` main `07304d6f`, files
//! `parallel/src/neural.rs` (`Config::regression`, `Policy` with `reward_encoder` and heads),
//! `parallel/src/grammar.rs` (vocabulary, automaton, legal masks),
//! `parallel/src/proposals.rs` (pure pool sampling loop),
//! `parallel/src/proposal_population.rs` (pure dedup and fill),
//! `parallel/src/proposal_ranking.rs` (`score_pool`, [`ranking::SELECTION_ID`]) and
//! `parallel/src/gp.rs` (`parallel.seed_population.v1`, `attach_policy_logprobs`).
//!
//! This crate is separate from feat-core's older `jope` module (the `policy_v3.2` candle policy
//! with a 1103-token vocabulary); it neither reads nor changes that module.
//!
//! Pieces:
//! - [`config`]: the policy configuration (`deny_unknown_fields`) and the export-receipt binding
//!   (accepted step-60000 identity, or exactly a pinned native challenger; the rejected
//!   step-69000 state is refused).
//! - [`grammar`]: the 401-token vocabulary file format, the masking automaton, `encode`,
//!   `decode`, `project_proposal`, plus the vocabulary binding (checkpoint vocabulary or the
//!   documented token-compatible product restriction `featmath-jain-v1`).
//! - [`model`]: the policy forward pass on candle (tensor names and shapes equal the
//!   `jope.checkpoint_export.v1` inventory: 136 tensors).
//! - [`decode`]: batched grammar-masked sampling and the pure 64-program pool.
//! - [`ranking`]: `score_pool` with `mean_reward_heads_desc_program_hash_v1`.
//! - [`seeds`]: the 80-candidate `parallel.seed_population.v1` writer and the checks jope-dime's
//!   `search --seeds` applies.
//! - [`proposal_ledger`]: every draw's terminal outcome, each handed-off candidate's original
//!   tokens, program, origin, scores, parser identity and adaptation edges, and the reconciled
//!   denominators of one proposal.
//!
//! # Provenance boundary
//!
//! In a product build (no `testing` feature) the only ways to obtain a seed population are
//! [`seeds::propose`], which requires a model from `PolicyModel::load_bound` whose recomputed
//! parameter hash is the accepted step-60000 hash, a receipt whose claimed and recomputed config
//! hash is [`ACCEPTED_CONFIG_SHA256`] and whose vocabulary is [`CHECKPOINT_VOCABULARY_SHA256`],
//! and a [`grammar::BoundVocabulary`] (private fields; only `bind_vocabulary` builds it from the
//! bytes it hashes) that is that vocabulary or [`PRODUCT_VOCABULARY_SHA256`], and
//! [`seeds::propose_challenger`], which requires a model from `PolicyModel::load_bound` whose
//! receipt names exactly a pinned [`config::ChallengerIdentity`] (never the accepted state,
//! parameters or weights) and stamps that challenger's state.
//! [`seeds::Proposal`], [`seeds::SeedPopulation`], [`seeds::SeedBinding`], [`ranking::Ranking`]
//! and [`ranking::Bindings`] have private fields;
//! the population assembler is private and the writer takes a [`seeds::Proposal`], so no code
//! outside this crate can put the accepted state or a `neural_policy` origin into a seed file;
//! a `neural_policy` origin comes only from a bound checkpoint, identified by the file's
//! `checkpoint_sha256`.
//! [`ranking::score_pool`] and [`decode::pure_pool`] refuse to label output with the accepted
//! state (or as neural samples) unless the policy is that bound model. Untrained weights and the
//! unbound helpers exist only behind the `testing` feature, and even those refuse the
//! accepted state.
//!
//! ```compile_fail,E0451
//! use jope_policy::seeds::SeedBinding;
//! // Private fields: a caller cannot write a binding (nor name a checkpoint in it).
//! let _ = SeedBinding { prepared_sha256: String::new(), cut: 0, training_sha256: String::new(), context_sha256: String::new() };
//! ```
//!
//! ```compile_fail,E0451
//! use jope_policy::ranking::Bindings;
//! let _ = Bindings { checkpoint_sha256: jope_policy::ACCEPTED_STATE_SHA256.into(), conditioning_context_sha256: String::new(), vocabulary_sha256: String::new(), vocabulary: None };
//! ```
//!
//! ```compile_fail,E0603
//! // The assembler is private to the crate.
//! use jope_policy::seeds::assemble_population;
//! ```
//!
//! ```compile_fail,E0308
//! // The writer only takes a `Proposal`, which only `propose` or `propose_challenger` constructs.
//! fn write(population: &jope_policy::seeds::SeedPopulation) {
//!     let _ = jope_policy::seeds::write_population(std::path::Path::new("s.json"), population);
//! }
//! ```
//!
//! ```compile_fail,E0277
//! // Seed populations and rankings cannot be deserialized into existence.
//! let _: jope_policy::seeds::SeedPopulation = serde_json::from_slice(b"{}").unwrap();
//! ```
//!
//! A binding built through its constructor carries no checkpoint, and a scoring binding cannot
//! name the accepted state:
//!
//! ```
//! use jope_policy::{ranking::Bindings, sha256_hex, ACCEPTED_STATE_SHA256};
//! let d = |t: &str| sha256_hex(t.as_bytes());
//! assert!(Bindings::new(ACCEPTED_STATE_SHA256, &d("c"), &d("v")).is_err());
//! assert!(Bindings::new(&d("other state"), &d("c"), &d("v")).is_ok());
//! ```
//!
//! # What is and is not proven here
//!
//! - **No numeric parity with the tch forward pass is claimed.** The arithmetic follows
//!   `neural.rs` (pre-norm decoder, LayerNorm eps 1e-5, erf GELU, boolean attention mask as an
//!   additive `-inf` mask, tied output embedding), but equality within atol 3e-4 can only be
//!   shown against reference fixtures produced by jope-dime itself (plan v2 W4 PR-B, owned by
//!   Codex root). Until those fixtures exist, logits and rewards from this port are unverified.
//!   The ignored test `generation_000011_rewards_agree_with_bf16_records` is only a
//!   cross-precision diagnostic: it compares CPU float32 reward heads with the BF16 values
//!   jope-dime recorded in generation-000011 and checks that the decoder prefers the
//!   checkpoint's recorded samples over grammar-uniform programs. It is not a parity gate.
//! - **CPU float32 only.** jope-dime samples and ranks under CUDA BF16 autocast; this port runs
//!   float32. The `cuda` feature is declared but build-unverified: it has never been
//!   compiled; the plan v2 W7 native GPU gate must build and test it on a CUDA builder.
//! - **Sampling streams differ by construction.** jope-dime draws with `torch.multinomial` after
//!   `tch::manual_seed`; this port draws from a seeded ChaCha20 stream. The same checkpoint and
//!   context therefore give a different (equally distributed, if the logits agree) pool.
//! - **Fallback and exploration programs differ.** jope-dime fills with
//!   `feat_math::evolve::random_canonical_pipeline` from its vendored feat-math; the pinned
//!   jain-math revision has no such function, so this port draws uniformly over the legal tokens
//!   of the same grammar. Those candidates carry their own origin labels
//!   (`neural_rejection_fallback`, `random_exploration`) and are never labelled `neural_policy`.

#[cfg(test)]
mod challenger_tests;
pub mod config;
pub mod context;
#[cfg(feature = "context-producer")]
pub mod context_producer;
pub mod decode;
pub mod grammar;
mod hashing;
pub mod jope_embed;
pub mod model;
pub mod producer;
pub mod proposal_ledger;
pub mod ranking;
pub mod seeds;
pub mod semantic;

/// Test support (`testing`, never enabled by a product build): untrained weights and
/// unbound helpers. Every helper here refuses the accepted state, so even a test build cannot
/// stamp it without the bound accepted model.
#[cfg(any(test, feature = "testing"))]
pub mod testing {
    /// sha256 of `jope-dime unbound test state`; never a checkpoint state.
    pub const UNBOUND_TEST_STATE_SHA256: &str =
        "1e89f0c12accb00068baf89a8f5dae4e962068679f0420c7da51aa5ef5f98a2f";
    pub use super::decode::pure_pool_unbound;
    pub use super::ranking::ranking_for_testing;
    pub use super::seeds::{assemble_for_testing, propose_unbound};
}

use sha2::{Digest, Sha256};

/// `state.json` sha256 of the accepted step-60000 checkpoint (cycle-000018).
pub const ACCEPTED_STATE_SHA256: &str =
    "8ac80c269cc47c253b6f53fa3bc1eb75bb2fcda0bdace3ff8554be0175c400d6";
/// `model.ot` sha256 of the accepted step-60000 checkpoint.
pub const ACCEPTED_MODEL_SHA256: &str =
    "053df4f53b84d2ebe7c14fbf907a149760af3d11de4a57b5ff89be8c2d3e8129";
/// Training step of the accepted checkpoint.
pub const ACCEPTED_STEP: u64 = 60000;
/// jope-dime parameter hash (sorted names + little-endian f32 values) of the accepted
/// step-60000 weights, as recomputed by `tools/ot-export` (`step-60000.receipt.json`,
/// `checkpoint.parameter_sha256_recomputed`). `seeds::propose` requires the loaded weights to
/// hash to this value, so a self-consistent receipt around other weights is not enough.
pub const ACCEPTED_PARAMETER_SHA256: &str =
    "9c7057e7f6a9e348f030da73f03522977b8c08915d3a221d7d5ab2e4213fcd92";
/// `config.json` sha256 of the accepted step-60000 checkpoint, as bound by its `state.json`
/// (`config_sha256`) and recorded by `tools/ot-export` (`checkpoint.config_sha256`). It is the
/// hash of `PolicyConfig::regression(401)` in the `config.json` layout
/// ([`config::PolicyConfig::sha256`]).
pub const ACCEPTED_CONFIG_SHA256: &str =
    "2508fc8b8268ea000141c9b119ba3da4e777c506c366168912a88d03482939bf";
/// `state.json` sha256 of the rejected step-69000 checkpoint (cycle-000021). Always refused.
pub const REJECTED_STATE_SHA256: &str =
    "58c4a1f62ca46611913ed75f8cb04cf5ffced7175ab1816af8e38ff9e6329c1c";
/// `model.ot` sha256 of the rejected step-69000 checkpoint. Always refused.
pub const REJECTED_MODEL_SHA256: &str =
    "f3694654b3d847b0a2f99c98ba4d47a6b44c7b3b32d1f423bcee0a2106fab582";
/// Training step of the rejected checkpoint.
pub const REJECTED_STEP: u64 = 69000;
/// sha256 of jope-dime `parallel/assets/grammar/featmath.json` (the step-60000 vocabulary).
pub const CHECKPOINT_VOCABULARY_SHA256: &str =
    "8d55fee4366acd2c175cf55bfd769cebc904770f5227a6fd50a613723dc85787";
/// sha256 of `featmath-jain-v1.json`: `featmath.json` restricted to heads jain-math 9fa594e2
/// can execute, with the 401-token table unchanged (jope-dime branch
/// `jepsont/featmath-jain-v1-vocabulary-20260917`).
pub const PRODUCT_VOCABULARY_SHA256: &str =
    "570b2e5e8d859a73b888f5d0bee27f59da0e2376536babce24c241604f818e81";

/// Documented token-compatible product restrictions: `(restricted sha256, base sha256)`.
pub const PRODUCT_RESTRICTIONS: &[(&str, &str)] =
    &[(PRODUCT_VOCABULARY_SHA256, CHECKPOINT_VOCABULARY_SHA256)];

/// Lowercase hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// True for a 64-character lowercase-or-uppercase hex digest.
pub fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit())
}

/// The identity of a policy that may label output with the accepted state: loaded through
/// `PolicyModel::load_bound`, bound to [`ACCEPTED_STATE_SHA256`], with the recomputed parameter
/// hash equal to [`ACCEPTED_PARAMETER_SHA256`], the loaded file equal to the released
/// `policy.safetensors`, and with the receipt's `config_sha256` and the loaded configuration's
/// recomputed hash both equal to [`ACCEPTED_CONFIG_SHA256`].
pub(crate) fn require_accepted_policy(
    model: &model::PolicyModel,
) -> anyhow::Result<&model::BoundIdentity> {
    use anyhow::Context;
    let bound = model
        .bound_identity()
        .context("refused: policy was not loaded through PolicyModel::load_bound")?;
    anyhow::ensure!(
        bound.state_sha256() == ACCEPTED_STATE_SHA256,
        "refused: loaded policy is bound to {}, not the accepted step-60000 state",
        bound.state_sha256()
    );
    anyhow::ensure!(
        bound.parameter_sha256() == ACCEPTED_PARAMETER_SHA256,
        "refused: loaded weights are not the accepted step-60000 parameters"
    );
    anyhow::ensure!(
        bound.policy_safetensors_sha256() == config::ACCEPTED_WEIGHTS_SHA256,
        "refused: loaded file {} is not the released step-60000 policy.safetensors",
        bound.policy_safetensors_sha256()
    );
    anyhow::ensure!(
        bound.config_sha256() == ACCEPTED_CONFIG_SHA256,
        "refused: loaded policy's receipt config_sha256 {} is not the accepted step-60000 config",
        bound.config_sha256()
    );
    let recomputed = model.config().sha256()?;
    anyhow::ensure!(
        recomputed == ACCEPTED_CONFIG_SHA256,
        "refused: loaded policy config hashes to {recomputed}, not the accepted step-60000 config"
    );
    // W2-T02: the release decision is the accepted semantic manifest; the reviewed constants
    // above are its values (`semantic::tests::every_compiled_identity_is_the_manifests`).
    let release = semantic::accepted()?.policy;
    anyhow::ensure!(
        bound.state_sha256() == release.state_sha256
            && bound.parameter_sha256() == release.parameter_sha256
            && bound.policy_safetensors_sha256() == release.weights_sha256
            && bound.config_sha256() == release.config_sha256
            && recomputed == release.config_sha256,
        "refused: loaded policy is not the accepted semantic release {}",
        semantic::ACCEPTED_SEMANTIC_MANIFEST_SHA256
    );
    Ok(bound)
}

/// A vocabulary that may be stamped next to the accepted state: the step-60000 vocabulary
/// ([`CHECKPOINT_VOCABULARY_SHA256`]) or its documented token-compatible product restriction
/// ([`PRODUCT_VOCABULARY_SHA256`]), judged by the sha256 [`grammar::bind_vocabulary`] computed
/// from the parsed bytes, never by the label it was bound against.
pub(crate) fn require_accepted_vocabulary(
    vocabulary: &grammar::BoundVocabulary,
) -> anyhow::Result<()> {
    // W2-T02: judged against the accepted semantic manifest's vocabulary meanings.
    let release = semantic::accepted()?.vocabulary;
    let accepted = match vocabulary.identity() {
        grammar::VocabularyIdentity::Checkpoint => {
            vocabulary.sha256() == CHECKPOINT_VOCABULARY_SHA256
                && vocabulary.sha256() == release.checkpoint_sha256
        }
        grammar::VocabularyIdentity::ProductRestriction { base_sha256 } => {
            vocabulary.sha256() == PRODUCT_VOCABULARY_SHA256
                && base_sha256 == CHECKPOINT_VOCABULARY_SHA256
                && vocabulary.sha256() == release.product_restriction.sha256
                && *base_sha256 == release.product_restriction.base_sha256
        }
    };
    anyhow::ensure!(
        accepted,
        "refused: vocabulary {} is neither the step-60000 vocabulary nor its documented product restriction",
        vocabulary.sha256()
    );
    Ok(())
}

/// True when `state_sha256` names the accepted state (any hex case).
pub(crate) fn is_accepted_state(state_sha256: &str) -> bool {
    state_sha256.eq_ignore_ascii_case(ACCEPTED_STATE_SHA256)
}

/// Refuse the rejected step-69000 checkpoint by state or model identity.
pub fn refuse_rejected_checkpoint(state_sha256: &str, model_sha256: &str) -> anyhow::Result<()> {
    anyhow::ensure!(
        !state_sha256.eq_ignore_ascii_case(REJECTED_STATE_SHA256)
            && !model_sha256.eq_ignore_ascii_case(REJECTED_MODEL_SHA256),
        "refused: rejected step-{REJECTED_STEP} checkpoint {state_sha256}"
    );
    Ok(())
}

/// The policy context of an 8192-byte little-endian context file (W2-T03: the frozen context
/// ABI, [`context::PolicyContext::from_le_bytes`]).
pub fn context_from_bytes(bytes: &[u8]) -> anyhow::Result<context::PolicyContext> {
    context::PolicyContext::from_le_bytes(bytes)
}

/// sha256 of a context's little-endian float32 bytes (jope-dime `conditioning_context_sha256`).
pub fn context_sha256(values: &[f32]) -> String {
    let bytes = values
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    sha256_hex(&bytes)
}
