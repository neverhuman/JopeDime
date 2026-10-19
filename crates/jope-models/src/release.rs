//! Immutable release manifests pinned by the reviewed source that built this binary.
//!
//! This is metadata verification, not weight loading, runtime qualification, training
//! admission or serving promotion. The compiled registry is deliberately empty until an
//! independently reviewed publication supplies real compatible artifacts. Caller checkpoint
//! claims and arbitrary manifest files cannot add registry pins.

use crate::manifest::{safe_relative, valid_sha256, BundleManifest};
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::hash::sha256_hex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const SCHEMA_ID: &str = "jope.model-release/v1";
pub const INDEX_SCHEMA_ID: &str = "jope.model-release-index/v1";
pub const SCHEMA_JSON: &str = include_str!("../../../schemas/jope.model-release.v1.schema.json");
const COMPILED_INDEX: &str = include_str!("../../../models/releases/INDEX.json");
const MAX_MANIFEST_BYTES: usize = 1024 * 1024;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ComponentRole {
    Encoder,
    Policy,
    ModelConfig,
    Tokenizer,
    Preprocessing,
    Runtime,
}

/// Declared channel only. Neither value grants execution or serving permission.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredChannel {
    Research,
    ServingCandidate,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Compatibility {
    pub model_family: String,
    pub vocabulary_sha256: String,
    pub vocabulary_size: u32,
    pub context_dim: u32,
    pub dsl_sha256: String,
    pub evaluator_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReleaseComponent {
    pub id: String,
    pub role: ComponentRole,
    pub bundle_id: String,
    pub bundle_manifest_sha256: String,
    pub bundle_manifest_version_id: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
    pub object_version_id: String,
}

/// An untrusted document until matched to a compiled release pin.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ReleaseManifest {
    pub schema: String,
    pub release_id: String,
    pub source_commit: String,
    pub channel: DeclaredChannel,
    pub parameter_sha256: String,
    pub training_config_sha256: String,
    pub dataset_snapshot_sha256: String,
    pub protocol_sha256: String,
    pub seed: u64,
    pub compatibility: Compatibility,
    pub components: Vec<ReleaseComponent>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleasePin {
    release_id: String,
    manifest_sha256: String,
    source_commit: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReleaseIndex {
    schema: String,
    releases: Vec<ReleasePin>,
}

/// A manifest checked against an immutable pin in this binary's reviewed source.
///
/// Fields are private and this type cannot be deserialized. It does NOT assert that
/// artifacts are downloaded, hash-verified, licensed for a requested use, executable,
/// compatible with a real runtime, scientifically qualified, or admitted for serving.
#[derive(Clone, Debug)]
pub struct VerifiedReleaseManifest {
    manifest: ReleaseManifest,
    manifest_sha256: String,
}

fn valid_id(s: &str) -> bool {
    (2..=64).contains(&s.len())
        && s.as_bytes()[0].is_ascii_lowercase()
        && s.bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'.')
        && !s.ends_with('.')
}

fn valid_commit(s: &str) -> bool {
    s.len() == 40
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn valid_version(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 1024
        && s != "null"
        && s != "UNPINNED"
        && !s.chars().any(|c| c.is_whitespace() || c.is_control())
}

impl ReleaseManifest {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        ensure!(
            bytes.len() <= MAX_MANIFEST_BYTES,
            "release manifest exceeds byte limit"
        );
        // Deserialize the typed document first: duplicate struct fields, including nested
        // component/compatibility fields, must not disappear through a JSON Value map.
        let manifest: Self = serde_json::from_slice(bytes).context("strict release manifest")?;
        let schema: serde_json::Value = serde_json::from_str(SCHEMA_JSON)?;
        let value: serde_json::Value = serde_json::from_slice(bytes)?;
        let errors = jope_datasets::schema::validate(&schema, &value);
        ensure!(errors.is_empty(), "release schema: {}", errors.join("; "));
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == SCHEMA_ID && valid_id(&self.release_id) && self.release_id != "latest",
            "invalid release schema/id"
        );
        ensure!(
            valid_commit(&self.source_commit),
            "invalid producer source commit"
        );
        for digest in [
            &self.parameter_sha256,
            &self.training_config_sha256,
            &self.dataset_snapshot_sha256,
            &self.protocol_sha256,
            &self.compatibility.vocabulary_sha256,
            &self.compatibility.dsl_sha256,
            &self.compatibility.evaluator_sha256,
        ] {
            ensure!(valid_sha256(digest), "invalid release identity digest");
        }
        ensure!(
            !self.compatibility.model_family.trim().is_empty(),
            "missing model family"
        );
        ensure!(
            self.compatibility.vocabulary_size > 0 && self.compatibility.context_dim > 0,
            "invalid vocabulary/context dimensions"
        );
        ensure!(
            (6..=32).contains(&self.components.len()),
            "release needs all six component roles, at most 32 components"
        );
        let mut ids = BTreeSet::new();
        let mut roles = BTreeMap::<ComponentRole, usize>::new();
        let mut bundles = BTreeMap::<&str, (&str, &str)>::new();
        for c in &self.components {
            ensure!(
                valid_id(&c.id) && ids.insert(&c.id),
                "duplicate/invalid component id"
            );
            ensure!(
                valid_id(&c.bundle_id)
                    && safe_relative(&c.path)
                    && !c.path.chars().any(char::is_control),
                "unsafe component bundle/path"
            );
            ensure!(
                valid_sha256(&c.sha256) && valid_sha256(&c.bundle_manifest_sha256) && c.bytes > 0,
                "invalid component identity"
            );
            ensure!(
                valid_version(&c.object_version_id) && valid_version(&c.bundle_manifest_version_id),
                "unversioned component"
            );
            *roles.entry(c.role).or_default() += 1;
            let binding = (
                c.bundle_manifest_sha256.as_str(),
                c.bundle_manifest_version_id.as_str(),
            );
            if let Some(previous) = bundles.insert(&c.bundle_id, binding) {
                ensure!(
                    previous == binding,
                    "mixed manifest identities for one bundle"
                );
            }
        }
        ensure!(
            roles.get(&ComponentRole::Encoder).copied().unwrap_or(0) >= 1,
            "missing encoder"
        );
        for role in [
            ComponentRole::Policy,
            ComponentRole::ModelConfig,
            ComponentRole::Tokenizer,
            ComponentRole::Preprocessing,
            ComponentRole::Runtime,
        ] {
            ensure!(
                roles.get(&role) == Some(&1),
                "missing or duplicate singleton role {role:?}"
            );
        }
        Ok(())
    }
}

fn compiled_index() -> Result<ReleaseIndex> {
    let index: ReleaseIndex =
        serde_json::from_str(COMPILED_INDEX).context("compiled release registry")?;
    ensure!(
        index.schema == INDEX_SCHEMA_ID,
        "invalid release registry schema"
    );
    let mut ids = BTreeSet::new();
    for pin in &index.releases {
        ensure!(
            valid_id(&pin.release_id) && pin.release_id != "latest" && ids.insert(&pin.release_id),
            "invalid/duplicate compiled release id"
        );
        ensure!(
            valid_sha256(&pin.manifest_sha256) && valid_commit(&pin.source_commit),
            "invalid compiled release pin"
        );
    }
    Ok(index)
}

/// Release IDs compiled into this build; no mutable `latest` or operator-supplied registry.
pub fn pinned_release_ids() -> Result<Vec<String>> {
    Ok(compiled_index()?
        .releases
        .into_iter()
        .map(|p| p.release_id)
        .collect())
}

/// Verify metadata from an arbitrary file against the independently compiled registry.
/// Unknown versions refuse; accepting a caller-provided hash is deliberately not an API.
pub fn verify_manifest(release_id: &str, bytes: &[u8]) -> Result<VerifiedReleaseManifest> {
    let index = compiled_index()?;
    let Some(pin) = index.releases.iter().find(|p| p.release_id == release_id) else {
        bail!("release {release_id:?} is not pinned in this build");
    };
    verify_against_pin(pin, bytes)
}

fn verify_against_pin(pin: &ReleasePin, bytes: &[u8]) -> Result<VerifiedReleaseManifest> {
    ensure!(
        bytes.len() <= MAX_MANIFEST_BYTES,
        "release manifest exceeds byte limit"
    );
    let digest = sha256_hex(bytes);
    ensure!(
        digest == pin.manifest_sha256,
        "release manifest differs from compiled pin"
    );
    let manifest = ReleaseManifest::parse(bytes)?;
    ensure!(
        manifest.release_id == pin.release_id && manifest.source_commit == pin.source_commit,
        "release id/source differs from compiled pin"
    );
    Ok(VerifiedReleaseManifest {
        manifest,
        manifest_sha256: digest,
    })
}

impl VerifiedReleaseManifest {
    pub fn manifest(&self) -> &ReleaseManifest {
        &self.manifest
    }
    pub fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }

    /// Declaration equality only; the runtime owner must separately verify the real loader.
    pub fn require_compatibility(&self, expected: &Compatibility) -> Result<()> {
        ensure!(
            &self.manifest.compatibility == expected,
            "incompatible release declaration"
        );
        Ok(())
    }

    /// Check a component's bundle metadata against the pinned release. Does not verify the
    /// component object's bytes/VersionId or establish a license/admission/runtime capability.
    pub fn verify_component_manifest(
        &self,
        component_id: &str,
        bytes: &[u8],
    ) -> Result<ReleaseComponent> {
        let c = self
            .manifest
            .components
            .iter()
            .find(|c| c.id == component_id)
            .context("unknown release component")?;
        ensure!(
            bytes.len() <= MAX_MANIFEST_BYTES && sha256_hex(bytes) == c.bundle_manifest_sha256,
            "bundle manifest differs from release pin"
        );
        let bundle = BundleManifest::parse(bytes)
            .map_err(|e| anyhow::anyhow!("bundle: {}", e.join("; ")))?;
        ensure!(bundle.bundle_id == c.bundle_id, "wrong component bundle");
        let file = bundle
            .file(&c.path)
            .context("component absent from bundle")?;
        ensure!(
            file.sha256 == c.sha256 && file.bytes == c.bytes,
            "component differs from bundle declaration"
        );
        Ok(c.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> ReleaseManifest {
        let roles = [
            ComponentRole::Encoder,
            ComponentRole::Policy,
            ComponentRole::ModelConfig,
            ComponentRole::Tokenizer,
            ComponentRole::Preprocessing,
            ComponentRole::Runtime,
        ];
        ReleaseManifest {
            schema: SCHEMA_ID.into(),
            release_id: "fixture-v1".into(),
            source_commit: "a".repeat(40),
            channel: DeclaredChannel::Research,
            parameter_sha256: "1".repeat(64),
            training_config_sha256: "2".repeat(64),
            dataset_snapshot_sha256: "3".repeat(64),
            protocol_sha256: "4".repeat(64),
            seed: 42,
            compatibility: Compatibility {
                model_family: "fixture-policy".into(),
                vocabulary_sha256: "5".repeat(64),
                vocabulary_size: 282,
                context_dim: 2051,
                dsl_sha256: "6".repeat(64),
                evaluator_sha256: "7".repeat(64),
            },
            components: roles
                .iter()
                .enumerate()
                .map(|(i, role)| ReleaseComponent {
                    id: format!("component-{i}"),
                    role: *role,
                    bundle_id: format!("bundle-{i}"),
                    bundle_manifest_sha256: "8".repeat(64),
                    bundle_manifest_version_id: "manifest-version".into(),
                    path: "model.bin".into(),
                    sha256: "9".repeat(64),
                    bytes: 1,
                    object_version_id: "object-version".into(),
                })
                .collect(),
        }
    }

    fn pinned(m: &ReleaseManifest) -> (ReleasePin, Vec<u8>) {
        let bytes = serde_json::to_vec(m).unwrap();
        (
            ReleasePin {
                release_id: m.release_id.clone(),
                manifest_sha256: sha256_hex(&bytes),
                source_commit: m.source_commit.clone(),
            },
            bytes,
        )
    }

    #[test]
    fn valid_metadata_requires_an_independent_pin_and_exact_source() {
        let m = fixture();
        let (mut pin, bytes) = pinned(&m);
        let verified = verify_against_pin(&pin, &bytes).unwrap();
        assert_eq!(verified.manifest(), &m);
        assert_eq!(verified.manifest_sha256(), sha256_hex(&bytes));
        assert!(verify_manifest("fixture-v1", &bytes).is_err());
        pin.source_commit = "b".repeat(40);
        assert!(verify_against_pin(&pin, &bytes).is_err());
        pin.source_commit = m.source_commit.clone();
        let mut changed = bytes.clone();
        changed.push(b' ');
        assert!(verify_against_pin(&pin, &changed).is_err());
    }

    #[test]
    fn missing_or_duplicate_components_and_mixed_bundles_refuse() {
        let mut m = fixture();
        m.components.pop();
        assert!(m.validate().is_err());
        let mut m = fixture();
        m.components[1].role = ComponentRole::Encoder;
        assert!(m.validate().is_err());
        let mut m = fixture();
        m.components[1].id = m.components[0].id.clone();
        assert!(m.validate().is_err());
        let mut m = fixture();
        m.components[1].bundle_id = m.components[0].bundle_id.clone();
        m.components[1].bundle_manifest_sha256 = "a".repeat(64);
        assert!(m.validate().is_err());
    }

    #[test]
    fn unsafe_paths_unversioned_objects_and_unknown_fields_refuse() {
        let mut alias = fixture();
        alias.release_id = "latest".into();
        assert!(alias.validate().is_err());
        for path in ["../outside", "/outside", "a//b", "a\\b"] {
            let mut m = fixture();
            m.components[0].path = path.into();
            assert!(m.validate().is_err());
        }
        let mut m = fixture();
        m.components[0].object_version_id = "null".into();
        assert!(m.validate().is_err());
        let mut value = serde_json::to_value(fixture()).unwrap();
        value["unreviewed"] = json!(true);
        assert!(ReleaseManifest::parse(&serde_json::to_vec(&value).unwrap()).is_err());
        let (pin, bytes) = pinned(&fixture());
        let text = String::from_utf8(bytes).unwrap().replacen(
            "{",
            "{\"schema\":\"jope.model-release/v1\",",
            1,
        );
        assert!(verify_against_pin(
            &ReleasePin {
                manifest_sha256: sha256_hex(text.as_bytes()),
                ..pin
            },
            text.as_bytes()
        )
        .is_err());
    }

    #[test]
    fn family_vocabulary_and_context_do_not_silently_coerce() {
        let m = fixture();
        let (pin, bytes) = pinned(&m);
        let v = verify_against_pin(&pin, &bytes).unwrap();
        v.require_compatibility(&m.compatibility).unwrap();
        let mut x = m.compatibility.clone();
        x.vocabulary_size = 401;
        assert!(v.require_compatibility(&x).is_err());
        let mut x = m.compatibility.clone();
        x.context_dim = 128;
        assert!(v.require_compatibility(&x).is_err());
        let mut x = m.compatibility.clone();
        x.model_family = "other-policy".into();
        assert!(v.require_compatibility(&x).is_err());
    }

    #[test]
    fn bundle_manifest_and_component_identity_are_both_bound() {
        let mut m = fixture();
        let bundle = json!({"schema":"jope.model-bundle/v2","bundle_id":"bundle-0","bucket":"fixture","published_at":"2026-10-19T00:00:00Z","notices":[],"files":[{"path":"model.bin","sha256":"9".repeat(64),"bytes":1,"blob":format!("blobs/sha256/{}","9".repeat(64)),"kind":"model","license":null}]});
        let b = serde_json::to_vec(&bundle).unwrap();
        m.components[0].bundle_manifest_sha256 = sha256_hex(&b);
        let (pin, bytes) = pinned(&m);
        let v = verify_against_pin(&pin, &bytes).unwrap();
        v.verify_component_manifest("component-0", &b).unwrap();
        assert!(v.verify_component_manifest("absent", &b).is_err());
        let mut changed = b.clone();
        changed.push(b' ');
        assert!(v
            .verify_component_manifest("component-0", &changed)
            .is_err());
        let mut wrong = m;
        wrong.components[0].bytes = 2;
        let (pin, bytes) = pinned(&wrong);
        let v = verify_against_pin(&pin, &bytes).unwrap();
        assert!(v.verify_component_manifest("component-0", &b).is_err());
    }

    #[test]
    fn serving_candidate_declaration_cannot_add_a_compiled_pin() {
        pinned_release_ids().unwrap();
        let mut m = fixture();
        m.channel = DeclaredChannel::ServingCandidate;
        let (_, bytes) = pinned(&m);
        assert!(verify_manifest(&m.release_id, &bytes).is_err());
    }
}
