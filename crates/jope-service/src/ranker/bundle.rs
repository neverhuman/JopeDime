//! Bounded inputs for fused-ranker verification (§5.8).
//!
//! File hashes and the declared qualification matrix do not authenticate a
//! bundle. The loader must verify the release signature before using either.

use jope_api::error::ErrorCode;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt, io::Read};

/// Every failure at this boundary has the API/CLI integrity disposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleIntegrityError {
    detail: &'static str,
}

impl BundleIntegrityError {
    pub const fn code(&self) -> ErrorCode {
        ErrorCode::ModelIntegrityFailed
    }

    pub fn exit_code(&self) -> u8 {
        self.code()
            .exit_code()
            .expect("integrity failures have a CLI exit code")
    }
}

impl fmt::Display for BundleIntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code(), self.detail)
    }
}

impl std::error::Error for BundleIntegrityError {}

type Result<T> = std::result::Result<T, BundleIntegrityError>;

fn invalid(detail: &'static str) -> BundleIntegrityError {
    BundleIntegrityError { detail }
}

/// Identity declared by the authenticated bundle manifest, not a file SHA or
/// the frozen DOPE parameter/buffer composite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BundleDigest([u8; 32]);

impl BundleDigest {
    pub fn parse(value: &str) -> Result<Self> {
        let value = value
            .strip_prefix("sha256:")
            .ok_or_else(|| invalid("invalid bundle digest"))?;
        Ok(Self(parse_sha256(value)?))
    }
}

impl fmt::Display for BundleDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("sha256:")?;
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

fn parse_sha256(value: &str) -> Result<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid SHA-256 encoding"));
    }
    let mut digest = [0; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        let digit = |b: u8| if b <= b'9' { b - b'0' } else { b - b'a' + 10 };
        digest[index] = digit(pair[0]) * 16 + digit(pair[1]);
    }
    Ok(digest)
}

/// Allocation bounds supplied by the embedding service, independently of any
/// untrusted file's length or tensor header.
#[derive(Clone, Copy, Debug)]
pub struct BundleLimits {
    pub manifest_bytes: u64,
    pub tensor_header_bytes: u64,
    pub file_bytes: u64,
    pub total_bytes: u64,
}

impl Default for BundleLimits {
    fn default() -> Self {
        Self {
            manifest_bytes: 1 << 20,
            tensor_header_bytes: 8 << 20,
            file_bytes: 2 << 30,
            total_bytes: 4 << 30,
        }
    }
}

/// Decoded file identity from a trusted release catalog. This is an internal
/// loader input, not an extension of the frozen fused-manifest schema.
#[derive(Clone, Copy, Debug)]
pub struct AssetIdentity {
    pub bytes: u64,
    pub sha256: [u8; 32],
}

/// Check the entire planned allocation before retaining any asset bytes.
pub fn check_byte_budget(assets: &[AssetIdentity], limits: BundleLimits) -> Result<()> {
    let mut total = 0u64;
    for asset in assets {
        if asset.bytes > limits.file_bytes
            || asset.bytes > isize::MAX as u64
            || usize::try_from(asset.bytes).is_err()
        {
            return Err(invalid("asset exceeds the file byte cap"));
        }
        total = total
            .checked_add(asset.bytes)
            .filter(|total| *total <= limits.total_bytes)
            .ok_or_else(|| invalid("bundle exceeds the total byte cap"))?;
    }
    Ok(())
}

/// Read exactly the authenticated decoded size, with a one-byte growth probe.
/// A hash match here is not a signature or permission to use a ranker.
pub fn read_hashed_asset(
    mut reader: impl Read,
    identity: AssetIdentity,
    limits: BundleLimits,
) -> Result<Vec<u8>> {
    check_byte_budget(&[identity], limits)?;
    let len = usize::try_from(identity.bytes).map_err(|_| invalid("asset length overflows"))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len)
        .map_err(|_| invalid("asset allocation refused"))?;
    bytes.resize(len, 0);
    reader
        .read_exact(&mut bytes)
        .map_err(|_| invalid("asset read or length mismatch"))?;
    let mut extra = [0];
    if reader
        .read(&mut extra)
        .map_err(|_| invalid("asset read failed"))?
        != 0
    {
        return Err(invalid("asset grew beyond its authenticated length"));
    }
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != identity.sha256 {
        return Err(invalid("asset SHA-256 mismatch"));
    }
    Ok(bytes)
}

/// Bound SafeTensors' untrusted little-endian header length before a parser
/// allocates its metadata. Tensor validation is a separate mandatory step.
pub fn check_tensor_header(bytes: &[u8], limits: BundleLimits) -> Result<()> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("truncated SafeTensors header"))?
        .try_into()
        .map_err(|_| invalid("truncated SafeTensors header"))?;
    let len = u64::from_le_bytes(prefix);
    if len == 0 || len > limits.tensor_header_bytes {
        return Err(invalid("SafeTensors header exceeds its byte cap"));
    }
    if len
        .checked_add(8)
        .is_none_or(|end| end > bytes.len() as u64)
    {
        return Err(invalid("SafeTensors header is outside the file"));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QualifiedTask {
    Regression,
    Binary,
    Multiclass,
}

/// An authenticated declaration only; empirical qualification is a separate gate.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Qualification {
    pub task: QualifiedTask,
    pub landmarks: String,
    pub backend: String,
}

/// Parsed declarations, with no claim of authentication or qualification.
#[derive(Clone, Debug)]
pub struct UnverifiedManifest {
    pub digest: BundleDigest,
    pub dope_core_rev: String,
    pub dope_weights_sha256: [u8; 32],
    pub jope_weights_sha256: [u8; 32],
    pub reference_source_sha256: BTreeMap<String, [u8; 32]>,
    pub surrogate_recipe: String,
    pub trained_objectives: Vec<String>,
    pub landmark_recipe: String,
    pub landmark_sample_rows: i64,
    pub qualified: Vec<Qualification>,
    pub asset_policy: String,
}

/// Inspect bounded JSON against the existing frozen schema and the one known
/// architecture. This does not load a model or verify a signature.
pub fn inspect_manifest(bytes: &[u8], limits: BundleLimits) -> Result<UnverifiedManifest> {
    if bytes.len() as u64 > limits.manifest_bytes {
        return Err(invalid("manifest exceeds its byte cap"));
    }
    let manifest: FusedManifest =
        serde_json::from_slice(bytes).map_err(|_| invalid("invalid fused manifest"))?;
    if manifest.schema != "jope.fused-bundle/v1" {
        return Err(invalid("unknown fused manifest schema"));
    }
    if manifest.architecture != "dope-fused-ranker/h40b" {
        return Err(invalid("unknown fused ranker architecture"));
    }
    let c = &manifest.constants;
    if c.scale != 0.4208
        || c.ctx_feat_dim != 584
        || c.revealed != 4
        || c.proposals != 8
        || c.proposals_explore != 16
        || c.admit != 4
        || c.elite_pool != 48
    {
        return Err(invalid("unsupported fused ranker constants"));
    }
    if manifest.dope_core_rev.len() != 40
        || !manifest
            .dope_core_rev
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    {
        return Err(invalid("invalid DOPE source revision"));
    }
    Ok(UnverifiedManifest {
        digest: BundleDigest::parse(&manifest.bundle_digest)?,
        dope_core_rev: manifest.dope_core_rev,
        dope_weights_sha256: parse_sha256(&manifest.dope_weights_sha256)?,
        jope_weights_sha256: parse_sha256(&manifest.jope_weights_sha256)?,
        reference_source_sha256: manifest
            .reference_source_sha256
            .into_iter()
            .map(|(name, hash)| parse_sha256(&hash).map(|hash| (name, hash)))
            .collect::<Result<_>>()?,
        surrogate_recipe: manifest.surrogate_recipe,
        trained_objectives: manifest.trained_objectives,
        landmark_recipe: manifest.landmark_recipe,
        landmark_sample_rows: manifest.landmark_sample_rows,
        qualified: manifest.qualified,
        asset_policy: manifest.asset_policy,
    })
}

/// Exact fields of jope.fused-bundle/v1. Deserialization rejects extensions;
/// parsing this structure alone does not establish authenticity.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FusedManifest {
    schema: String,
    architecture: String,
    bundle_digest: String,
    dope_core_rev: String,
    dope_weights_sha256: String,
    jope_weights_sha256: String,
    reference_source_sha256: BTreeMap<String, String>,
    constants: FusedConstants,
    surrogate_recipe: String,
    trained_objectives: Vec<String>,
    landmark_recipe: String,
    landmark_sample_rows: i64,
    qualified: Vec<Qualification>,
    asset_policy: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FusedConstants {
    scale: f64,
    ctx_feat_dim: i64,
    revealed: i64,
    proposals: i64,
    proposals_explore: i64,
    admit: i64,
    elite_pool: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{cell::Cell, io::Cursor};

    fn identity(bytes: &[u8]) -> AssetIdentity {
        AssetIdentity {
            bytes: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        }
    }

    fn limits() -> BundleLimits {
        BundleLimits {
            manifest_bytes: 1024,
            tensor_header_bytes: 64,
            file_bytes: 128,
            total_bytes: 256,
        }
    }

    fn refusal<T: fmt::Debug>(result: Result<T>) {
        let error = result.unwrap_err();
        assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
        assert_eq!(error.exit_code(), 3);
        assert!(error.to_string().starts_with("MODEL_INTEGRITY_FAILED: "));
    }

    #[test]
    fn exact_hashed_read_accepts_and_refuses_corruption_truncation_and_growth() {
        let original = b"signed catalog file";
        let expected = identity(original);
        assert_eq!(
            read_hashed_asset(Cursor::new(original), expected, limits()).unwrap(),
            original
        );
        refusal(read_hashed_asset(
            Cursor::new(b"corrupt catalog fil"),
            expected,
            limits(),
        ));
        refusal(read_hashed_asset(
            Cursor::new(&original[..original.len() - 1]),
            expected,
            limits(),
        ));
        refusal(read_hashed_asset(
            Cursor::new([original.as_slice(), b"!"].concat()),
            expected,
            limits(),
        ));
    }

    #[test]
    fn caps_refuse_before_read_and_checked_total_never_wraps() {
        struct CountReads<'a>(&'a Cell<usize>);
        impl Read for CountReads<'_> {
            fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
                self.0.set(self.0.get() + 1);
                panic!("oversized asset must be refused before reading")
            }
        }
        let count = Cell::new(0);
        let too_big = AssetIdentity {
            bytes: 129,
            sha256: [0; 32],
        };
        refusal(read_hashed_asset(CountReads(&count), too_big, limits()));
        assert_eq!(count.get(), 0);
        let asset = AssetIdentity {
            bytes: 100,
            sha256: [0; 32],
        };
        assert!(check_byte_budget(&[asset, asset], limits()).is_ok());
        refusal(check_byte_budget(&[asset, asset, asset], limits()));
        let huge = AssetIdentity {
            bytes: u64::MAX,
            sha256: [0; 32],
        };
        refusal(check_byte_budget(
            &[huge, huge],
            BundleLimits {
                file_bytes: u64::MAX,
                total_bytes: u64::MAX,
                ..limits()
            },
        ));
    }

    #[test]
    fn header_length_is_bounded_without_parsing_or_overflow() {
        let mut valid = 2u64.to_le_bytes().to_vec();
        valid.extend_from_slice(b"{}");
        assert!(check_tensor_header(&valid, limits()).is_ok());
        refusal(check_tensor_header(&valid[..7], limits()));
        refusal(check_tensor_header(&[0; 8], limits()));
        refusal(check_tensor_header(&65u64.to_le_bytes(), limits()));
        refusal(check_tensor_header(&u64::MAX.to_le_bytes(), limits()));
        refusal(check_tensor_header(&64u64.to_le_bytes(), limits()));
    }

    #[test]
    fn bundle_digest_requires_exact_lowercase_prefixed_identity() {
        let text = format!("sha256:{}", "ab".repeat(32));
        assert_eq!(BundleDigest::parse(&text).unwrap().to_string(), text);
        for bad in [
            "ab".repeat(32),
            format!("sha256:{}", "AB".repeat(32)),
            format!("sha256:{}", "ab".repeat(31)),
        ] {
            refusal(BundleDigest::parse(&bad));
        }
    }

    fn manifest() -> serde_json::Value {
        serde_json::json!({
            "schema": "jope.fused-bundle/v1", "architecture": "dope-fused-ranker/h40b",
            "bundle_digest": format!("sha256:{}", "ab".repeat(32)),
            "dope_core_rev": "1".repeat(40),
            "dope_weights_sha256": "2".repeat(64), "jope_weights_sha256": "3".repeat(64),
            "reference_source_sha256": {"fixture.rs": "4".repeat(64)},
            "constants": {"scale": 0.4208, "ctx_feat_dim": 584, "revealed": 4,
                "proposals": 8, "proposals_explore": 16, "admit": 4, "elite_pool": 48},
            "surrogate_recipe": "fixture", "trained_objectives": ["regression/r2"],
            "landmark_recipe": "fixture", "landmark_sample_rows": 256,
            "qualified": [{"task": "regression", "landmarks": "fixture", "backend": "cpu"}],
            "asset_policy": "fixture, not a release"
        })
    }

    #[test]
    fn frozen_manifest_is_strict_and_unknown_architectures_are_refused() {
        let bytes = serde_json::to_vec(&manifest()).unwrap();
        let parsed = inspect_manifest(&bytes, limits()).unwrap();
        assert_eq!(parsed.dope_weights_sha256, [0x22; 32]);
        assert_eq!(parsed.qualified[0].task, QualifiedTask::Regression);
        refusal(inspect_manifest(
            &bytes,
            BundleLimits {
                manifest_bytes: 10,
                ..limits()
            },
        ));
        for (field, value) in [
            ("architecture", serde_json::json!("future-architecture")),
            ("schema", serde_json::json!("jope.fused-bundle/v2")),
            ("bundle_digest", serde_json::json!("sha256:invalid")),
            ("dope_weights_sha256", serde_json::json!("invalid")),
            ("dope_core_rev", serde_json::json!("invalid")),
            ("extra", serde_json::json!(true)),
        ] {
            let mut bad = manifest();
            bad[field] = value;
            refusal(inspect_manifest(
                &serde_json::to_vec(&bad).unwrap(),
                limits(),
            ));
        }
        for pointer in ["/constants/extra", "/qualified/0/extra"] {
            let mut bad = manifest();
            let (parent, key) = pointer.rsplit_once('/').unwrap();
            bad.pointer_mut(parent).unwrap()[key] = true.into();
            refusal(inspect_manifest(
                &serde_json::to_vec(&bad).unwrap(),
                limits(),
            ));
        }
        let mut bad = manifest();
        bad["constants"]["ctx_feat_dim"] = 585.into();
        refusal(inspect_manifest(
            &serde_json::to_vec(&bad).unwrap(),
            limits(),
        ));
    }
}
