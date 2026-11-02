//! Bounded inputs for fused-ranker verification (§5.8).
//!
//! File hashes and the declared qualification matrix do not authenticate a
//! bundle. The loader must verify the release signature before using either.

use jope_api::error::ErrorCode;
use p256::{
    ecdsa::{signature::Verifier, Signature, VerifyingKey},
    pkcs8::{DecodePublicKey, EncodePublicKey},
};
pub use safetensors::Dtype;
use safetensors::SafeTensors;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt, io::Read};
use std::{collections::BTreeSet, fs::File, path::Path};

const SIGNING_DOMAIN: &[u8] = b"jope.singlefile.manifest-signature.v1\0";
const RELEASE_KEY: &str = include_str!("../../../../docs/release/signing.pub");
/// Logical parameter/buffer identity. This is deliberately not a weight-file SHA.
pub const FROZEN_DOPE_COMPOSITE: &str =
    "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814";

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

/// Existing jope.singlefile envelope proof, supplied by the package adapter.
/// The exact signed catalog authenticates decoded file sizes and hashes. The
/// loader does not introduce a second signature format or decompress assets.
pub struct SignedAssetContext<'a> {
    pub manifest: &'a [u8],
    pub footer: &'a [u8; 128],
    pub signature: &'a [u8; 64],
}

/// Exact tensor expectation from trusted release/custody metadata, including
/// buffers. Never construct this catalog by inspecting the file being loaded.
#[derive(Clone, Debug)]
pub struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: Dtype,
}

#[derive(Clone, Debug)]
pub struct ReleaseAsset {
    /// Entry id in the existing signed appliance catalog.
    pub signed_asset_id: String,
    /// Mandatory for every .safetensors asset; None for other assets.
    pub tensors: Option<BTreeMap<String, TensorSpec>>,
}

/// Trusted host configuration, not a new bundle wire format. Custody supplies
/// the frozen file-to-logical-DOPE binding and the complete tensor inventories.
/// BundleDigest is pinned here because RES-06 owns its canonical computation.
#[derive(Clone, Debug)]
pub struct ReleaseCatalog {
    pub bundle_digest: BundleDigest,
    pub dope_logical_composite: [u8; 32],
    pub dope_weights_sha256: [u8; 32],
    /// Relative bundle path -> signed asset entry and trusted tensor inventory.
    pub assets: BTreeMap<String, ReleaseAsset>,
}

pub struct EmbeddedAsset<'a> {
    pub path: &'a str,
    pub bytes: &'a [u8],
}

/// Directory selection is for the development CLI adapter. Release startup
/// supplies decoded embedded bytes through the same verification boundary.
pub enum BundleInput<'a> {
    Embedded(&'a [EmbeddedAsset<'a>]),
    DevelopmentDirectory(&'a Path),
}

/// Immutable, all-or-nothing result. Construction is private to the loader.
pub struct VerifiedBundle {
    manifest: UnverifiedManifest,
    assets: BTreeMap<String, Vec<u8>>,
}

impl VerifiedBundle {
    pub fn digest(&self) -> BundleDigest {
        self.manifest.digest
    }

    /// Authenticated declarations, not empirical M2/M3/Q1 qualification.
    pub fn declared_qualifications(&self) -> &[Qualification] {
        &self.manifest.qualified
    }

    pub fn asset(&self, path: &str) -> Option<&[u8]> {
        self.assets.get(path).map(Vec::as_slice)
    }
}

#[cfg(test)]
pub(crate) fn test_verified_bundle() -> VerifiedBundle {
    integrity_tests::test_verified_bundle()
}

impl fmt::Debug for VerifiedBundle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VerifiedBundle")
            .field("digest", &self.digest())
            .field("assets", &self.assets.keys())
            .finish()
    }
}

/// Verify the existing release signature, all selected file identities and the
/// complete trusted tensor catalogs before publishing any loaded bytes.
/// Production trusts only the repository's established release public key.
pub fn load_bundle(
    input: BundleInput<'_>,
    context: SignedAssetContext<'_>,
    release: &ReleaseCatalog,
    limits: BundleLimits,
) -> Result<VerifiedBundle> {
    let key = VerifyingKey::from_public_key_pem(RELEASE_KEY.trim())
        .map_err(|_| invalid("invalid compiled release key"))?;
    load_with_keys(input, context, release, limits, &[key])
}

// Only the internal fixture path can select the existing development key.
fn load_with_keys(
    input: BundleInput<'_>,
    context: SignedAssetContext<'_>,
    release: &ReleaseCatalog,
    limits: BundleLimits,
    keys: &[VerifyingKey],
) -> Result<VerifiedBundle> {
    let signed = verify_asset_context(context, limits, keys)?;
    if release.dope_logical_composite != parse_sha256(FROZEN_DOPE_COMPOSITE)? {
        return Err(invalid("release catalog does not bind frozen DOPE"));
    }
    for required in [
        "manifest.json",
        "dope.safetensors",
        "jope.safetensors",
        "featurizer.json",
        "bank.jsonl.zst",
    ] {
        if !release.assets.contains_key(required) {
            return Err(invalid("release catalog is missing a required asset"));
        }
    }
    if !release
        .assets
        .keys()
        .any(|p| p.starts_with("golden/") && p.ends_with(".safetensors"))
    {
        return Err(invalid("release catalog is missing golden tensors"));
    }
    let mut identities = BTreeMap::new();
    let mut selected_ids = BTreeSet::new();
    for (path, spec) in &release.assets {
        if !safe_path(path) || !selected_ids.insert(&spec.signed_asset_id) {
            return Err(invalid("invalid or aliased release asset path"));
        }
        match (&spec.tensors, path.ends_with(".safetensors")) {
            (Some(tensors), true) if !tensors.is_empty() => {}
            (None, false) => {}
            _ => return Err(invalid("missing or inappropriate trusted tensor catalog")),
        }
        let identity = signed
            .get(&spec.signed_asset_id)
            .ok_or_else(|| invalid("asset missing from the signed catalog"))?;
        identities.insert(path.as_str(), *identity);
    }
    check_byte_budget(&identities.values().copied().collect::<Vec<_>>(), limits)?;
    if identities["manifest.json"].bytes > limits.manifest_bytes {
        return Err(invalid("fused manifest exceeds its byte cap"));
    }
    if identities["dope.safetensors"].sha256 != release.dope_weights_sha256 {
        return Err(invalid("signed DOPE file differs from the custody pin"));
    }
    let source = AssetReader::new(input, &release.assets)?;
    let manifest_bytes = source.read("manifest.json", identities["manifest.json"], limits)?;
    let manifest = inspect_manifest(&manifest_bytes, limits)?;
    if manifest.digest != release.bundle_digest
        || manifest.dope_weights_sha256 != identities["dope.safetensors"].sha256
        || manifest.jope_weights_sha256 != identities["jope.safetensors"].sha256
    {
        return Err(invalid(
            "fused manifest differs from authenticated release identities",
        ));
    }
    let mut assets = BTreeMap::from([("manifest.json".into(), manifest_bytes)]);
    for (path, spec) in &release.assets {
        if path == "manifest.json" {
            continue;
        }
        let bytes = source.read(path, identities[path.as_str()], limits)?;
        if let Some(expected) = &spec.tensors {
            validate_tensors(&bytes, expected, limits)?;
        }
        assets.insert(path.clone(), bytes);
    }
    Ok(VerifiedBundle { manifest, assets })
}

// This is an authenticated projection of the existing appliance catalog, not
// another serializer. Other package fields/frames remain the package adapter's
// responsibility. No canonicalization or reserialization precedes verification.
#[derive(Deserialize)]
struct ApplianceCatalog {
    schema: String,
    format_version: u16,
    product: String,
    entries: Vec<ApplianceEntry>,
}
#[derive(Deserialize)]
struct ApplianceEntry {
    id: String,
    sha256: String,
    bytes: u64,
}

fn verify_asset_context(
    context: SignedAssetContext<'_>,
    limits: BundleLimits,
    keys: &[VerifyingKey],
) -> Result<BTreeMap<String, AssetIdentity>> {
    let footer = context.footer;
    let u16_at = |n| u16::from_le_bytes(footer[n..n + 2].try_into().expect("fixed footer"));
    let u32_at = |n| u32::from_le_bytes(footer[n..n + 4].try_into().expect("fixed footer"));
    let u64_at = |n| u64::from_le_bytes(footer[n..n + 8].try_into().expect("fixed footer"));
    if &footer[..8] != b"JOPEEXE1"
        || u16_at(8) != 1
        || u16_at(10) != 0
        || u32_at(12) != 128
        || u32_at(56) != 64
        || u32_at(60) != 1
    {
        return Err(invalid("invalid signed appliance footer"));
    }
    let base = u64_at(24);
    let offset = u64_at(32);
    let len = u64_at(40);
    if base == 0
        || base
            .checked_next_multiple_of(4096)
            .is_none_or(|aligned| offset < aligned)
        || len == 0
        || len > limits.manifest_bytes
        || len != context.manifest.len() as u64
    {
        return Err(invalid(
            "signed catalog exceeds its bound or has invalid layout",
        ));
    }
    let end = offset
        .checked_add(len)
        .filter(|n| n.checked_add(192).is_some())
        .ok_or_else(|| invalid("signed catalog layout overflows"))?;
    if u64_at(16) != end || u64_at(48) != end {
        return Err(invalid("signed catalog layout disagrees with footer"));
    }
    if Sha256::digest(context.manifest).as_slice() != &footer[96..128] {
        return Err(invalid("signed catalog SHA-256 mismatch"));
    }
    let key = keys
        .iter()
        .find(|key| {
            key.to_public_key_der()
                .is_ok_and(|der| Sha256::digest(der.as_bytes()).as_slice() == &footer[64..96])
        })
        .ok_or_else(|| invalid("untrusted package signer"))?;
    let signature = Signature::from_slice(context.signature)
        .map_err(|_| invalid("invalid package signature encoding"))?;
    if signature.normalize_s() != signature {
        return Err(invalid("noncanonical package signature"));
    }
    let mut message = Vec::with_capacity(SIGNING_DOMAIN.len() + 128);
    message.extend_from_slice(SIGNING_DOMAIN);
    message.extend_from_slice(footer);
    key.verify(&message, &signature)
        .map_err(|_| invalid("package signature verification failed"))?;
    let catalog: ApplianceCatalog = serde_json::from_slice(context.manifest)
        .map_err(|_| invalid("invalid signed appliance catalog"))?;
    if catalog.schema != "jope.appliance-manifest/v1"
        || catalog.format_version != 1
        || catalog.product != "jopedime"
        || catalog.entries.len() > 4096
    {
        return Err(invalid("unsupported signed appliance catalog"));
    }
    let mut assets = BTreeMap::new();
    let mut previous = None;
    for entry in catalog.entries {
        if previous.as_ref().is_some_and(|p| p >= &entry.id) || !relative_name(&entry.id, true) {
            return Err(invalid("signed assets are not sorted unique names"));
        }
        previous = Some(entry.id.clone());
        assets.insert(
            entry.id,
            AssetIdentity {
                bytes: entry.bytes,
                sha256: parse_sha256(&entry.sha256)?,
            },
        );
    }
    Ok(assets)
}

fn safe_path(path: &str) -> bool {
    relative_name(path, false)
}

fn relative_name(path: &str, signed_id: bool) -> bool {
    !path.is_empty()
        && path.len() <= 256
        && path.bytes().all(|b| {
            b.is_ascii_alphanumeric()
                || matches!(b, b'/' | b'.' | b'_' | b'-')
                || (signed_id && matches!(b, b'+' | b':'))
        })
        && path
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != "..")
}

fn validate_tensors(
    bytes: &[u8],
    expected: &BTreeMap<String, TensorSpec>,
    limits: BundleLimits,
) -> Result<()> {
    check_tensor_header(bytes, limits)?;
    let header_len = u64::from_le_bytes(bytes[..8].try_into().expect("checked prefix")) as usize;
    serde_json::from_slice::<UniqueTensorNames>(&bytes[8..8 + header_len])
        .map_err(|_| invalid("invalid or duplicate tensor header names"))?;
    let tensors = SafeTensors::deserialize(bytes)
        .map_err(|_| invalid("invalid SafeTensors header or offsets"))?;
    if tensors.len() != expected.len() {
        return Err(invalid("tensor inventory length mismatch"));
    }
    for (name, tensor) in tensors.iter() {
        let spec = expected
            .get(name)
            .ok_or_else(|| invalid("unexpected tensor name"))?;
        if tensor.shape() != spec.shape || tensor.dtype() != spec.dtype {
            return Err(invalid("tensor shape or dtype mismatch"));
        }
        check_finite(tensor.data(), spec.dtype)?;
    }
    Ok(())
}

// SafeTensors' map deserializer collapses repeated top-level tensor names.
// Refuse ambiguity before its validated offset/dtype parser sees the header.
struct UniqueTensorNames;
impl<'de> Deserialize<'de> for UniqueTensorNames {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Names;
        impl<'de> serde::de::Visitor<'de> for Names {
            type Value = UniqueTensorNames;
            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a tensor header with unique names")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut names = BTreeSet::new();
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name) {
                        return Err(serde::de::Error::custom("duplicate tensor name"));
                    }
                    map.next_value::<serde::de::IgnoredAny>()?;
                }
                Ok(UniqueTensorNames)
            }
        }
        deserializer.deserialize_map(Names)
    }
}

fn check_finite(bytes: &[u8], dtype: Dtype) -> Result<()> {
    let (width, exponent) = match dtype {
        Dtype::F64 => (8, 0x7ff0_0000_0000_0000u64),
        Dtype::F32 => (4, 0x7f80_0000),
        Dtype::F16 => (2, 0x7c00),
        Dtype::BF16 => (2, 0x7f80),
        Dtype::BOOL
        | Dtype::U8
        | Dtype::I8
        | Dtype::I16
        | Dtype::U16
        | Dtype::I32
        | Dtype::U32
        | Dtype::I64
        | Dtype::U64 => return Ok(()),
        _ => return Err(invalid("unsupported tensor dtype")),
    };
    for value in bytes.chunks_exact(width) {
        let mut word = [0u8; 8];
        word[..width].copy_from_slice(value);
        if u64::from_le_bytes(word) & exponent == exponent {
            return Err(invalid("nonfinite tensor parameter or buffer"));
        }
    }
    Ok(())
}

enum AssetReader<'a> {
    Embedded(&'a [EmbeddedAsset<'a>]),
    Directory(DirectoryReader),
}

impl<'a> AssetReader<'a> {
    fn new(input: BundleInput<'a>, expected: &BTreeMap<String, ReleaseAsset>) -> Result<Self> {
        match input {
            BundleInput::Embedded(assets) => {
                let paths: BTreeSet<_> = assets.iter().map(|a| a.path).collect();
                if paths.len() != assets.len()
                    || paths.len() != expected.len()
                    || paths.iter().any(|p| !expected.contains_key(*p))
                {
                    return Err(invalid(
                        "embedded asset inventory differs from release catalog",
                    ));
                }
                Ok(Self::Embedded(assets))
            }
            BundleInput::DevelopmentDirectory(root) => {
                Ok(Self::Directory(DirectoryReader::open(root)?))
            }
        }
    }

    fn read(&self, path: &str, identity: AssetIdentity, limits: BundleLimits) -> Result<Vec<u8>> {
        match self {
            Self::Embedded(assets) => {
                let asset = assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .ok_or_else(|| invalid("missing embedded asset"))?;
                if asset.bytes.len() as u64 != identity.bytes {
                    return Err(invalid("embedded asset size mismatch"));
                }
                read_hashed_asset(asset.bytes, identity, limits)
            }
            Self::Directory(directory) => {
                let file = directory.open_asset(path)?;
                let metadata = file
                    .metadata()
                    .map_err(|_| invalid("cannot stat opened asset"))?;
                if !metadata.is_file() || metadata.len() != identity.bytes {
                    return Err(invalid("asset is nonregular or has an unexpected size"));
                }
                read_hashed_asset(file, identity, limits)
            }
        }
    }
}

#[cfg(unix)]
struct DirectoryReader {
    root: File,
}

#[cfg(unix)]
impl DirectoryReader {
    fn open(path: &Path) -> Result<Self> {
        use std::os::unix::fs::OpenOptionsExt;
        let root = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC)
            .open(path)
            .map_err(|_| invalid("cannot open bundle directory"))?;
        Ok(Self { root })
    }

    #[allow(unsafe_code)] // nix 0.29 returns RawFd; only the owned conversion below is unsafe.
    fn open_asset(&self, path: &str) -> Result<File> {
        use nix::{
            fcntl::{openat, OFlag},
            sys::stat::Mode,
        };
        use std::os::fd::{AsRawFd, FromRawFd};
        let mut parent = self
            .root
            .try_clone()
            .map_err(|_| invalid("cannot retain bundle directory"))?;
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            let mut flags =
                OFlag::O_RDONLY | OFlag::O_NOFOLLOW | OFlag::O_NONBLOCK | OFlag::O_CLOEXEC;
            if parts.peek().is_some() {
                flags |= OFlag::O_DIRECTORY;
            }
            let fd = openat(Some(parent.as_raw_fd()), part, flags, Mode::empty())
                .map_err(|_| invalid("cannot open regular bundle asset without links"))?;
            // SAFETY: openat returned a new owned descriptor on success. It is
            // transferred exactly once to File; parent stays live through openat.
            parent = unsafe { File::from_raw_fd(fd) };
        }
        Ok(parent)
    }
}

#[cfg(windows)]
struct DirectoryReader {
    root: std::path::PathBuf,
    pins: Vec<File>,
}

#[cfg(windows)]
impl DirectoryReader {
    fn open(path: &Path) -> Result<Self> {
        // Canonical root is caller-selected; pin every ancestor without delete
        // sharing before walking asset paths, so directories cannot be swapped.
        let selected = Self::open_path(path)?;
        if !selected
            .metadata()
            .map_err(|_| invalid("cannot stat bundle directory"))?
            .is_dir()
        {
            return Err(invalid("bundle root is not a directory"));
        }
        let root =
            std::fs::canonicalize(path).map_err(|_| invalid("cannot resolve bundle directory"))?;
        let mut current = std::path::PathBuf::new();
        let mut pins = vec![selected];
        for part in root.components() {
            current.push(part);
            if matches!(part, std::path::Component::Prefix(_)) {
                continue;
            }
            let file = Self::open_path(&current)?;
            if !file
                .metadata()
                .map_err(|_| invalid("cannot stat directory"))?
                .is_dir()
            {
                return Err(invalid("bundle ancestor is not a directory"));
            }
            pins.push(file);
        }
        Ok(Self { root, pins })
    }

    fn open_path(path: &Path) -> Result<File> {
        use std::os::windows::fs::{MetadataExt, OpenOptionsExt};
        // BACKUP_SEMANTICS | OPEN_REPARSE_POINT; FILE_SHARE_READ | WRITE only.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .share_mode(3)
            .custom_flags(0x0200_0000 | 0x0020_0000)
            .open(path)
            .map_err(|_| invalid("cannot open bundle asset"))?;
        if file
            .metadata()
            .map_err(|_| invalid("cannot stat opened asset"))?
            .file_attributes()
            & 0x400
            != 0
        {
            return Err(invalid("bundle reparse points are refused"));
        }
        Ok(file)
    }

    fn open_asset(&self, path: &str) -> Result<File> {
        let _root_pins = &self.pins;
        let mut current = self.root.clone();
        let mut pins = Vec::new();
        let mut parts = path.split('/').peekable();
        while let Some(part) = parts.next() {
            current.push(part);
            let file = Self::open_path(&current)?;
            if parts.peek().is_none() {
                return Ok(file);
            }
            if !file
                .metadata()
                .map_err(|_| invalid("cannot stat asset parent"))?
                .is_dir()
            {
                return Err(invalid("asset parent is not a directory"));
            }
            pins.push(file);
        }
        Err(invalid("empty bundle asset path"))
    }
}

#[cfg(not(any(unix, windows)))]
struct DirectoryReader;
#[cfg(not(any(unix, windows)))]
impl DirectoryReader {
    fn open(_: &Path) -> Result<Self> {
        Err(invalid(
            "development directories unsupported on this platform",
        ))
    }
    fn open_asset(&self, _: &str) -> Result<File> {
        Err(invalid(
            "development directories unsupported on this platform",
        ))
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod integrity_tests;
