//! The versioned semantic manifest of the accepted assets (W2-T02, gap G-01).
//!
//! Until this manifest, the accepted release was a set of compile-time constants spread over
//! `lib.rs`, `config.rs`, `context_producer.rs`, `worker.rs`, `admission.rs` and duplicated as
//! literals in other crates; no receiver consumed one release document. The manifest binds, in
//! one canonical versioned document, the policy identity (state, model, step, parameters,
//! configuration layout, weights file, tensor and parameter totals, active head roles and the
//! ordering rule), the rejected checkpoint, the vocabulary and its product restriction, the
//! context ABI, the encoder closure (Hyperion, component bundle, Foundation, Target, frozen
//! transforms, assembly, recipe, seeds and label normalizations) and the native qualification
//! policy.
//!
//! Its identity is [`ACCEPTED_SEMANTIC_MANIFEST_SHA256`], the release decision: the manifest is
//! embedded at build time and [`accepted`] refuses any other bytes, unknown fields, an unknown
//! schema version or a non-canonical encoding. The accepted receivers
//! (`require_accepted_policy`, `require_accepted_vocabulary`,
//! `CheckpointBinding::require_accepted_identity` and, with `context-producer`,
//! `EncoderIdentity::require_accepted`) compare with the manifest's values; the reviewed
//! constants must equal them (`tests::every_compiled_identity_is_the_manifests`). A new release
//! is a new manifest with a new identity, never an edit of this one.
//!
//! Canonical encoding: JSON object keys in byte order, no insignificant whitespace, integers and
//! strings only (tolerances are decimal strings), one trailing newline.

use anyhow::{ensure, Context, Result};
use serde::{Deserialize, Serialize};

/// The manifest schema version this build understands.
pub const SEMANTIC_SCHEMA: &str = "jope.policy-semantic-manifest/v1";
/// Identity (sha256 of the exact committed bytes) of the accepted step-60000 semantic manifest.
pub const ACCEPTED_SEMANTIC_MANIFEST_SHA256: &str =
    "60f61cd8b58a1eaa72220a03e91af9be074793b60cd1a05df9c6cac1ac225935";
const ACCEPTED_SEMANTIC_MANIFEST: &[u8] =
    include_bytes!("../release/policy-step-60000.semantic-v1.json");

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SemanticManifest {
    pub schema: String,
    pub release: String,
    pub policy: PolicySemantics,
    pub rejected: RejectedCheckpoint,
    pub vocabulary: VocabularySemantics,
    pub context_abi: ContextAbi,
    pub encoders: EncoderSemantics,
    pub qualification: QualificationSemantics,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicySemantics {
    pub state_sha256: String,
    pub model_sha256: String,
    pub step: u64,
    pub parameter_sha256: String,
    pub config_sha256: String,
    pub config_layout: String,
    pub weights_sha256: String,
    pub tensors: u64,
    pub parameters: u64,
    pub heads: HeadRoles,
    pub ordering: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadRoles {
    pub decode: Vec<String>,
    pub rank: Vec<String>,
    pub training_only: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RejectedCheckpoint {
    pub state_sha256: String,
    pub model_sha256: String,
    pub step: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VocabularySemantics {
    pub checkpoint_sha256: String,
    pub tokens: u64,
    pub product_restriction: ProductRestriction,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProductRestriction {
    pub name: String,
    pub sha256: String,
    pub base_sha256: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextAbi {
    pub encoding: String,
    pub values: u64,
    pub bytes: u64,
    pub clip: u64,
    pub dataset_dim: u64,
    pub target_dim: u64,
    pub blocks: Vec<ContextBlock>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ContextBlock {
    pub name: String,
    pub values: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderSemantics {
    pub hyperion_weights_sha256: String,
    pub hyperion_manifest_sha256: String,
    pub hyperion_seed: u64,
    pub component_bundle_manifest_sha256: String,
    pub component_bundle_identity: String,
    pub foundation_weights_sha256: String,
    pub foundation_seed: u64,
    pub target_weights_sha256: String,
    pub target_window_seed: u64,
    pub frozen_transforms_sha256: String,
    pub assembly_sha256: String,
    pub recipe: String,
    pub label_normalizations: LabelNormalizations,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelNormalizations {
    pub foundation: String,
    pub target: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationSemantics {
    pub reference_registry_sha256: String,
    pub reference_index_sha256: String,
    pub max_abs: String,
    pub verification_policy: String,
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// Canonical bytes: sorted keys, compact, trailing newline.
pub fn canonical(manifest: &SemanticManifest) -> Result<Vec<u8>> {
    // `serde_json::Value` orders object keys (no `preserve_order` in this workspace).
    let mut bytes = serde_json::to_vec(&serde_json::to_value(manifest)?)?;
    bytes.push(b'\n');
    Ok(bytes)
}

impl SemanticManifest {
    /// Strict decode: exact schema version, no unknown fields, canonical bytes and well-formed
    /// identities. Opening proves structure, not acceptance; see [`accepted`].
    pub fn open(bytes: &[u8]) -> Result<Self> {
        let value: serde_json::Value =
            serde_json::from_slice(bytes).context("parse semantic manifest")?;
        ensure!(
            value["schema"] == SEMANTIC_SCHEMA,
            "unsupported semantic manifest schema {}",
            value["schema"]
        );
        let manifest: Self = serde_json::from_value(value).context("semantic manifest fields")?;
        ensure!(
            canonical(&manifest)? == bytes,
            "semantic manifest is not canonically encoded"
        );
        manifest.validate()?;
        Ok(manifest)
    }

    fn validate(&self) -> Result<()> {
        let p = &self.policy;
        let e = &self.encoders;
        for (name, digest) in [
            ("policy state", &p.state_sha256),
            ("policy model", &p.model_sha256),
            ("policy parameters", &p.parameter_sha256),
            ("policy config", &p.config_sha256),
            ("policy weights", &p.weights_sha256),
            ("rejected state", &self.rejected.state_sha256),
            ("rejected model", &self.rejected.model_sha256),
            ("vocabulary", &self.vocabulary.checkpoint_sha256),
            (
                "product vocabulary",
                &self.vocabulary.product_restriction.sha256,
            ),
            (
                "product vocabulary base",
                &self.vocabulary.product_restriction.base_sha256,
            ),
            ("hyperion weights", &e.hyperion_weights_sha256),
            ("hyperion manifest", &e.hyperion_manifest_sha256),
            (
                "component bundle manifest",
                &e.component_bundle_manifest_sha256,
            ),
            ("component bundle", &e.component_bundle_identity),
            ("foundation weights", &e.foundation_weights_sha256),
            ("target weights", &e.target_weights_sha256),
            ("frozen transforms", &e.frozen_transforms_sha256),
            ("assembly", &e.assembly_sha256),
            (
                "reference registry",
                &self.qualification.reference_registry_sha256,
            ),
            (
                "reference index",
                &self.qualification.reference_index_sha256,
            ),
        ] {
            ensure!(
                is_sha256(digest),
                "semantic manifest {name} digest malformed"
            );
        }
        ensure!(
            self.rejected.state_sha256 != p.state_sha256
                && self.rejected.model_sha256 != p.model_sha256,
            "semantic manifest names the rejected checkpoint as accepted"
        );
        ensure!(
            self.vocabulary.product_restriction.base_sha256 == self.vocabulary.checkpoint_sha256,
            "product vocabulary restriction is not of the checkpoint vocabulary"
        );
        let abi = &self.context_abi;
        ensure!(
            abi.bytes == abi.values * 4
                && abi.dataset_dim + abi.target_dim == abi.values
                && abi.blocks.iter().map(|b| b.values).sum::<u64>() == abi.values,
            "semantic manifest context ABI is inconsistent"
        );
        let heads = &p.heads;
        let mut roots: Vec<&String> = heads
            .decode
            .iter()
            .chain(&heads.rank)
            .chain(&heads.training_only)
            .collect();
        let count = roots.len();
        roots.sort();
        roots.dedup();
        ensure!(
            roots.len() == count && !heads.decode.is_empty() && !heads.rank.is_empty(),
            "semantic manifest head roles overlap or are empty"
        );
        Ok(())
    }
}

/// The accepted semantic manifest: the embedded document, required to be exactly the release
/// decision's bytes ([`ACCEPTED_SEMANTIC_MANIFEST_SHA256`]).
pub fn accepted() -> Result<SemanticManifest> {
    open_accepted(ACCEPTED_SEMANTIC_MANIFEST)
}

/// Open `bytes` as the accepted semantic manifest: its identity must be the release decision.
/// A candidate's own manifest is never accepted on its word.
pub fn open_accepted(bytes: &[u8]) -> Result<SemanticManifest> {
    ensure!(
        crate::sha256_hex(bytes) == ACCEPTED_SEMANTIC_MANIFEST_SHA256,
        "semantic manifest {} is not the accepted release decision",
        crate::sha256_hex(bytes)
    );
    SemanticManifest::open(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    fn committed() -> SemanticManifest {
        accepted().unwrap()
    }

    fn reencode(value: serde_json::Value) -> Vec<u8> {
        let mut bytes = serde_json::to_vec(&value).unwrap();
        bytes.push(b'\n');
        bytes
    }

    /// Every reviewed constant a receiver compares with is the manifest's value, including the
    /// jope-policy duplicates the dependency table listed (G-01).
    #[test]
    fn every_compiled_identity_is_the_manifests() {
        let m = committed();
        assert_eq!(m.policy.state_sha256, ACCEPTED_STATE_SHA256);
        assert_eq!(m.policy.model_sha256, ACCEPTED_MODEL_SHA256);
        assert_eq!(m.policy.step, ACCEPTED_STEP);
        assert_eq!(m.policy.parameter_sha256, ACCEPTED_PARAMETER_SHA256);
        assert_eq!(m.policy.config_sha256, ACCEPTED_CONFIG_SHA256);
        assert_eq!(m.policy.weights_sha256, config::ACCEPTED_WEIGHTS_SHA256);
        assert_eq!(
            config::PolicyConfig::regression(m.vocabulary.tokens as usize)
                .sha256()
                .unwrap(),
            m.policy.config_sha256
        );
        assert_eq!(m.policy.ordering, ranking::SELECTION_ID);
        assert_eq!(m.rejected.state_sha256, REJECTED_STATE_SHA256);
        assert_eq!(m.rejected.model_sha256, REJECTED_MODEL_SHA256);
        assert_eq!(m.rejected.step, REJECTED_STEP);
        assert_eq!(m.vocabulary.checkpoint_sha256, CHECKPOINT_VOCABULARY_SHA256);
        assert_eq!(
            m.vocabulary.product_restriction.sha256,
            PRODUCT_VOCABULARY_SHA256
        );
        assert_eq!(
            PRODUCT_RESTRICTIONS,
            &[(
                m.vocabulary.product_restriction.sha256.as_str(),
                m.vocabulary.product_restriction.base_sha256.as_str()
            )]
        );
        // The committed step-60000 vocabulary is the manifest's, with its token count.
        let vocabulary = include_bytes!("../../jope-engine/assets/grammar/featmath.json");
        assert_eq!(sha256_hex(vocabulary), m.vocabulary.checkpoint_sha256);
        let tokens = grammar::Vocab::from_json(vocabulary).unwrap().tokens.len();
        assert_eq!(tokens as u64, m.vocabulary.tokens);
        let config = config::PolicyConfig::regression(tokens);
        assert_eq!(
            (config.dataset_dim + config.target_dim) as u64,
            m.context_abi.values
        );
        assert_eq!(config.dataset_dim as u64, m.context_abi.dataset_dim);
        assert_eq!(config.target_dim as u64, m.context_abi.target_dim);
        // The recorded tensor inventory of the released weights has the manifest's totals.
        let inventory: serde_json::Value = serde_json::from_slice(include_bytes!(
            "../../../docs/internal/core-ip/data/tensors.json"
        ))
        .unwrap();
        assert_eq!(inventory["sha256"], m.policy.weights_sha256.as_str());
        assert_eq!(inventory["tensor_count"], m.policy.tensors);
        assert_eq!(inventory["parameters"], m.policy.parameters);
        assert_eq!(
            model::expected_inventory(&config).len() as u64,
            m.policy.tensors
        );
    }

    #[cfg(feature = "context-producer")]
    #[test]
    fn the_encoder_release_identity_is_the_manifests() {
        use crate::context_producer::{self as cp, self as admission, worker};
        let e = committed().encoders;
        let accepted = cp::EncoderIdentity::accepted();
        assert_eq!(accepted.hyperion_weights, e.hyperion_weights_sha256);
        assert_eq!(accepted.hyperion_manifest, e.hyperion_manifest_sha256);
        assert_eq!(accepted.component_bundle, e.component_bundle_identity);
        assert_eq!(accepted.foundation_weights, e.foundation_weights_sha256);
        assert_eq!(accepted.target_weights, e.target_weights_sha256);
        assert_eq!(accepted.frozen_transforms, e.frozen_transforms_sha256);
        assert_eq!(accepted.assembly, e.assembly_sha256);
        assert_eq!(accepted.recipe, e.recipe);
        assert_eq!(cp::TRANSFORMS_SHA256, e.frozen_transforms_sha256);
        assert_eq!(cp::ASSEMBLY_SHA256, e.assembly_sha256);
        assert_eq!(
            worker::COMPONENT_BUNDLE_MANIFEST_SHA256,
            e.component_bundle_manifest_sha256
        );
        let q = committed().qualification;
        assert_eq!(
            admission::REFERENCE_REGISTRY_SHA256,
            q.reference_registry_sha256
        );
        assert_eq!(admission::REFERENCE_INDEX_SHA256, q.reference_index_sha256);
        assert_eq!(admission::VERIFICATION_POLICY, q.verification_policy);
    }

    /// The embedded manifest reopens to its exact bytes; any other document, unknown field,
    /// unknown version or non-canonical encoding is refused.
    #[test]
    fn only_the_accepted_manifest_bytes_are_the_release_decision() {
        let m = committed();
        assert_eq!(canonical(&m).unwrap(), ACCEPTED_SEMANTIC_MANIFEST);
        let value = serde_json::to_value(&m).unwrap();
        // A single substituted dependency: structurally valid, not the release decision.
        let mut substituted = value.clone();
        substituted["encoders"]["foundation_weights_sha256"] = serde_json::json!("0".repeat(64));
        let bytes = reencode(substituted);
        assert!(SemanticManifest::open(&bytes).is_ok());
        let refused = open_accepted(&bytes).unwrap_err().to_string();
        assert!(
            refused.contains("not the accepted release decision"),
            "{refused}"
        );
        // Unknown field, unknown version, non-canonical encoding.
        let mut unknown = value.clone();
        unknown["policy"]["cache_manifest"] = serde_json::json!("x");
        assert!(SemanticManifest::open(&reencode(unknown)).is_err());
        let mut version = value.clone();
        version["schema"] = serde_json::json!("jope.policy-semantic-manifest/v2");
        let refused = SemanticManifest::open(&reencode(version))
            .unwrap_err()
            .to_string();
        assert!(refused.contains("unsupported"), "{refused}");
        let pretty = serde_json::to_vec_pretty(&value).unwrap();
        let refused = SemanticManifest::open(&pretty).unwrap_err().to_string();
        assert!(refused.contains("canonical"), "{refused}");
        // Internally inconsistent documents are refused at open.
        let mut rejected = value.clone();
        rejected["rejected"]["state_sha256"] = value["policy"]["state_sha256"].clone();
        assert!(SemanticManifest::open(&reencode(rejected)).is_err());
        let mut abi = value;
        abi["context_abi"]["values"] = serde_json::json!(2047);
        assert!(SemanticManifest::open(&reencode(abi)).is_err());
    }
}
