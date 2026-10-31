//! `jope.appliance-manifest/v1`: what a package holds and how to find, authenticate and decode
//! each asset. Schema: `schemas/jope.appliance-manifest.v1.schema.json`.
//!
//! The manifest is serialized with `serde_json::to_vec` from these structs (fields in declaration
//! order, no whitespace, integers and strings only), entries sorted by id and methods sorted.
//! Those exact bytes are what the footer hashes; [`Manifest::parse`] refuses any other encoding of
//! the same values, so one package has one manifest byte string.

use crate::{limits, AssetError, Codec, ALIGN, FORMAT_VERSION, MANIFEST_SCHEMA};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Canonical signed inventory of the executable and its packaged assets.
pub struct Manifest {
    /// Versioned schema identifier.
    pub schema: String,
    /// Envelope format version supported by this package.
    pub format_version: u32,
    /// `jopedime`.
    pub product: String,
    /// Product release version recorded by the packager.
    pub product_version: String,
    /// Rust target triple of the executable region.
    pub target: String,
    /// `deploy/appliance/profiles/<profile>.toml`.
    pub profile: String,
    /// Maximum redistribution audience for all package assets.
    pub scope: Scope,
    /// Length and digest of the executable region before the payload.
    pub code: CodeIdentity,
    /// Source commit and lock/profile identities used for the package.
    pub source: SourceIdentity,
    /// Method ids the profile promises, sorted; every one must be executable in this build.
    pub methods: Vec<String>,
    /// Assets included in the package, sorted by logical id.
    pub entries: Vec<Entry>,
}

/// The audience a package may reach. A narrower scope never carries a wider-rights asset
/// (`jope-dist` enforces the redistribution classes when it resolves the lock).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Scope {
    /// CI fixture packages: generated assets, development key only.
    Development,
    /// `redistribution: permitted` assets only.
    Public,
    /// Adds `internal`; never published outside the organization.
    Internal,
    /// Adds `prohibited`, behind a recorded owner acknowledgement; never published.
    InternalRestricted,
}

impl Scope {
    /// Return the canonical manifest spelling of this audience.
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::Development => "development",
            Scope::Public => "public",
            Scope::Internal => "internal",
            Scope::InternalRestricted => "internal-restricted",
        }
    }

    /// Parse one canonical audience spelling.
    pub fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "development" => Scope::Development,
            "public" => Scope::Public,
            "internal" => Scope::Internal,
            "internal-restricted" => Scope::InternalRestricted,
            _ => return None,
        })
    }
}

/// The executable region in front of the payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CodeIdentity {
    /// Executable region length in bytes.
    pub len: u64,
    /// Lowercase SHA-256 digest of the executable region.
    pub sha256: String,
}

/// The inputs the package was built from.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct SourceIdentity {
    /// Full source commit.
    pub sha40: String,
    /// SHA-256 digest of the source tree's `Cargo.lock` bytes.
    pub cargo_lock_sha256: String,
    /// SHA-256 digest of the selected appliance profile bytes.
    pub profile_sha256: String,
    /// SHA-256 digest of the model lock bytes.
    pub models_lock_sha256: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// One logical asset and the frames that reconstruct its decoded bytes.
pub struct Entry {
    /// Logical id, `<role>/<component>/<path>`; unique, the sort key.
    pub id: String,
    /// `model`, `config`, `manifest` or `notice`.
    pub role: String,
    /// Component within the package that owns this asset.
    pub component: String,
    /// Where the bytes came from: `<bundle_id>/<path>` in the model store, or `generated:<what>`.
    pub origin: String,
    /// sha256 of the decoded bytes: the file exactly as its source manifest names it.
    pub sha256: String,
    /// Decoded asset length in bytes.
    pub bytes: u64,
    /// Storage codec used for the asset's frames.
    pub codec: Codec,
    /// sha256 of the stored frames, concatenated in order.
    pub stored_sha256: String,
    /// Total stored frame bytes, excluding alignment gaps.
    pub stored_bytes: u64,
    /// Ordered stored frames that reconstruct the decoded asset.
    pub frames: Vec<Frame>,
    /// Redistribution terms that apply to this asset.
    pub license: License,
    /// Backends whose qualification covers this asset (`cpu`, `cuda`).
    pub backends: Vec<String>,
}

/// One independently decodable frame. `offset` is relative to the payload start.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct Frame {
    /// Byte offset relative to the payload start.
    pub offset: u64,
    /// Stored frame length in bytes.
    pub len: u64,
    /// Decoded frame length in bytes.
    pub raw_len: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// License and redistribution classification copied from the source inventory.
pub struct License {
    /// SPDX license identifier or declared license text.
    pub spdx: String,
    /// `permitted`, `internal`, `prohibited` or `conditional`, from the source manifest.
    pub redistribution: String,
}

fn bad(reason: impl Into<String>) -> AssetError {
    AssetError::Manifest(reason.into())
}

fn limit(reason: impl Into<String>) -> AssetError {
    AssetError::Limit(reason.into())
}

/// Ids and names: printable ASCII from a small alphabet, no `..` component, bounded length.
pub fn is_name(text: &str) -> bool {
    !text.is_empty()
        && text.len() <= limits::MAX_NAME_BYTES
        && text.bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-' | b'/' | b'+' | b':')
        })
        && !text
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
}

/// Return whether `text` is exactly 64 lowercase hexadecimal SHA-256 characters.
pub fn is_sha256(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// zstd's `ZSTD_compressBound`: the largest frame a well-formed encoder emits for `raw` bytes.
pub fn zstd_bound(raw: u64) -> u64 {
    let small = if raw < (128 << 10) {
        ((128 << 10) - raw) >> 11
    } else {
        0
    };
    raw + (raw >> 8) + small + 64
}

impl Manifest {
    /// Serialize this manifest into its canonical signed byte representation.
    pub fn to_bytes(&self) -> Vec<u8> {
        serde_json::to_vec(self).expect("manifest serializes")
    }

    /// Parse authenticated manifest bytes and check every structural rule a reader relies on.
    /// `payload_len` is the byte length from the payload start to the manifest.
    pub fn parse(bytes: &[u8], payload_len: u64) -> Result<Self, AssetError> {
        let manifest: Manifest = serde_json::from_slice(bytes).map_err(|e| bad(e.to_string()))?;
        if manifest.format_version != u32::from(FORMAT_VERSION) {
            return Err(AssetError::UnsupportedVersion(manifest.format_version));
        }
        if manifest.schema != MANIFEST_SCHEMA {
            return Err(bad(format!(
                "schema {} is not {MANIFEST_SCHEMA}",
                manifest.schema
            )));
        }
        if manifest.to_bytes() != bytes {
            return Err(bad("manifest bytes are not in canonical form"));
        }
        manifest.check(payload_len)?;
        Ok(manifest)
    }

    /// The structural rules [`Self::parse`] enforces; the writer runs them on what it stages.
    pub fn check(&self, payload_len: u64) -> Result<(), AssetError> {
        for name in [
            &self.product,
            &self.product_version,
            &self.target,
            &self.profile,
        ] {
            if !is_name(name) {
                return Err(bad(format!("{name:?} is not a valid name")));
            }
        }
        if !is_sha256(&self.code.sha256) || self.code.len == 0 {
            return Err(bad("bad executable identity"));
        }
        if self.methods.len() > limits::MAX_METHODS {
            return Err(limit("too many methods"));
        }
        if self.methods.windows(2).any(|w| w[0] >= w[1]) || !self.methods.iter().all(|m| is_name(m))
        {
            return Err(bad("methods are not sorted unique names"));
        }
        if self.entries.len() > limits::MAX_ENTRIES {
            return Err(limit("too many entries"));
        }
        // Identical frame lists (deduplicated content) must name identical content; any other
        // two frames may not overlap.
        let mut shared: BTreeMap<&[Frame], (&str, Codec, &str)> = BTreeMap::new();
        let mut previous: Option<&str> = None;
        for entry in &self.entries {
            let id = entry.id.as_str();
            if previous.is_some_and(|p| p >= id) {
                return Err(bad("entries are not sorted by unique id"));
            }
            previous = Some(id);
            if !is_name(id) || !is_name(&entry.role) || !is_name(&entry.component) {
                return Err(bad(format!("{id:?}: invalid id, role or component")));
            }
            if !is_sha256(&entry.sha256) || !is_sha256(&entry.stored_sha256) {
                return Err(bad(format!("{id}: bad sha256")));
            }
            if entry.bytes > limits::MAX_ENTRY_BYTES {
                return Err(limit(format!("{id}: entry larger than the entry bound")));
            }
            let (mut raw, mut stored) = (0u64, 0u64);
            let mut end_of_previous = 0u64;
            for (index, frame) in entry.frames.iter().enumerate() {
                let end = frame
                    .offset
                    .checked_add(frame.len)
                    .ok_or_else(|| bad(format!("{id}: frame offset overflows")))?;
                if !frame.offset.is_multiple_of(ALIGN) || end > payload_len {
                    return Err(bad(format!("{id}: frame outside the payload")));
                }
                if index > 0 && frame.offset < end_of_previous {
                    return Err(bad(format!("{id}: frames are not in order")));
                }
                end_of_previous = end;
                if frame.raw_len == 0 || frame.raw_len > limits::MAX_FRAME_RAW_BYTES {
                    return Err(limit(format!("{id}: frame raw length out of range")));
                }
                let stored_ok = match entry.codec {
                    Codec::Raw => frame.len == frame.raw_len,
                    Codec::Zstd | Codec::ShuffleZstd => {
                        frame.len > 0 && frame.len <= zstd_bound(frame.raw_len)
                    }
                };
                if !stored_ok {
                    return Err(limit(format!("{id}: stored frame length out of range")));
                }
                raw = raw
                    .checked_add(frame.raw_len)
                    .ok_or_else(|| limit(format!("{id}: decoded size overflows")))?;
                stored += frame.len;
            }
            if raw != entry.bytes || stored != entry.stored_bytes {
                return Err(bad(format!("{id}: frames do not add up to the entry")));
            }
            if entry.backends.windows(2).any(|w| w[0] >= w[1])
                || !entry.backends.iter().all(|b| is_name(b))
            {
                return Err(bad(format!("{id}: backends are not sorted unique names")));
            }
            if !entry.frames.is_empty() {
                let content = (
                    entry.sha256.as_str(),
                    entry.codec,
                    entry.stored_sha256.as_str(),
                );
                if let Some(existing) = shared.insert(&entry.frames, content) {
                    if existing != content {
                        return Err(bad(format!("{id}: shares frames with different content")));
                    }
                }
            }
        }
        // Only whole frame lists are shared: a frame appearing in two different lists, or two
        // frames overlapping, is refused.
        let mut frames: Vec<Frame> = shared.keys().flat_map(|f| f.iter().copied()).collect();
        frames.sort();
        if frames
            .windows(2)
            .any(|w| w[0].offset + w[0].len > w[1].offset)
        {
            return Err(bad(
                "frames of different entries overlap or are partly shared",
            ));
        }
        Ok(())
    }

    /// Find one entry by its logical id.
    pub fn entry(&self, id: &str) -> Option<&Entry> {
        self.entries
            .binary_search_by(|e| e.id.as_str().cmp(id))
            .ok()
            .map(|i| &self.entries[i])
    }

    /// Entries of one component, in id order.
    pub fn component<'a>(&'a self, component: &'a str) -> impl Iterator<Item = &'a Entry> + 'a {
        self.entries
            .iter()
            .filter(move |e| e.component == component)
    }

    /// Every distinct frame, in file order.
    pub fn unique_frames(&self) -> Vec<Frame> {
        let mut frames: Vec<Frame> = self
            .entries
            .iter()
            .flat_map(|e| e.frames.iter().copied())
            .collect();
        frames.sort();
        frames.dedup();
        frames
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_refuse_traversal_and_odd_bytes() {
        for good in [
            "model/policy/policy.safetensors",
            "notice/THIRD_PARTY.json",
            "rust.gpr.rbf_linear",
            "generated:notices",
        ] {
            assert!(is_name(good), "{good}");
        }
        for bad in ["", "../x", "a//b", "a/./b", "a b", "a\\b", "é", "/abs"] {
            assert!(!is_name(bad), "{bad}");
        }
    }

    #[test]
    fn zstd_bound_matches_the_c_library_formula_plus_headroom() {
        assert_eq!(zstd_bound(0), (128 << 10 >> 11) + 64);
        assert_eq!(zstd_bound(1 << 20), (1 << 20) + (1 << 12) + 64);
    }
}
