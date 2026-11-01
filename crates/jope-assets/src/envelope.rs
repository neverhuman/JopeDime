//! The 128-byte footer and the package reader.
//!
//! Footer, little-endian (SPECIFICATION.md §10.2, with bytes 96..128 carrying the manifest
//! sha256 per AMENDMENTS.md A5):
//!
//! | Offset | Bytes | Field |
//! | ---: | ---: | --- |
//! | 0 | 8 | magic `JOPEEXE1` |
//! | 8 | 2 | envelope version = 1 |
//! | 10 | 2 | flags = 0; unknown bits are refused |
//! | 12 | 4 | footer length = 128 |
//! | 16 | 8 | `signed_prefix_len` = `manifest_offset + manifest_len` |
//! | 24 | 8 | `base_executable_len` |
//! | 32 | 8 | `manifest_offset` |
//! | 40 | 8 | `manifest_len` |
//! | 48 | 8 | `signature_offset` = `signed_prefix_len` |
//! | 56 | 4 | `signature_len` = 64 |
//! | 60 | 4 | signature algorithm = 1 (P-256 ECDSA / SHA-256) |
//! | 64 | 32 | key id: sha256 of the signer's SPKI DER |
//! | 96 | 32 | sha256 of the manifest bytes |
//!
//! The payload starts at `base_executable_len` rounded up to 4096; the file is exactly
//! `signed_prefix_len + 64 + 128` bytes.

use crate::{
    ALGORITHM_P256_SHA256, AssetError, Entry, FOOTER_LEN, FORMAT_VERSION, KeyId, MAGIC,
    SIGNATURE_LEN, TrustedKey, align_up, decode, limits, manifest::Manifest, source::VerifiedBytes,
    verify,
};
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

/// sha256 of `len` bytes of `file` from `offset`, read in bounded chunks.
fn sha256_region(file: &File, offset: u64, len: u64) -> Result<String, AssetError> {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; (8 << 20).min(len as usize).max(1)];
    let mut at = 0u64;
    while at < len {
        let take = buffer.len().min((len - at) as usize);
        file.read_exact_at(&mut buffer[..take], offset + at)?;
        hasher.update(&buffer[..take]);
        at += take as u64;
    }
    Ok(hex::encode(hasher.finalize()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Parsed 128-byte package footer. Prefer [`Footer::decode`] for untrusted bytes.
pub struct Footer {
    /// Length of the executable bytes before alignment padding.
    pub base_executable_len: u64,
    /// Byte offset of the canonical manifest in the package file.
    pub manifest_offset: u64,
    /// Length of the canonical manifest bytes.
    pub manifest_len: u64,
    /// Identity of the key that signed this footer.
    pub key_id: KeyId,
    /// SHA-256 digest of the canonical manifest bytes.
    pub manifest_sha256: [u8; 32],
}

fn footer_error(reason: &str) -> AssetError {
    AssetError::Footer(reason.to_owned())
}

impl Footer {
    /// End offset of the signed prefix (`manifest_offset + manifest_len`).
    ///
    /// Call this on a footer returned by [`Self::decode`] or otherwise known to have a checked
    /// layout; the public fields allow callers to construct invalid layouts.
    pub fn signed_prefix_len(&self) -> u64 {
        self.manifest_offset + self.manifest_len
    }

    /// Where the asset frames start. Requires a footer with a valid checked layout.
    pub fn payload_offset(&self) -> u64 {
        align_up(self.base_executable_len).expect("checked by decode")
    }

    /// Total package length implied by this footer. Requires a valid checked layout.
    pub fn file_len(&self) -> u64 {
        self.signed_prefix_len() + SIGNATURE_LEN + FOOTER_LEN
    }

    /// Check the layout equations without a file: what [`Self::decode`] enforces.
    fn check(&self) -> Result<(), AssetError> {
        if self.base_executable_len == 0 {
            return Err(AssetError::Layout("no executable region".into()));
        }
        let payload = align_up(self.base_executable_len)
            .ok_or_else(|| AssetError::Layout("executable length overflows".into()))?;
        if self.manifest_offset < payload {
            return Err(AssetError::Layout("manifest precedes the payload".into()));
        }
        if self.manifest_len == 0 || self.manifest_len > limits::MAX_MANIFEST_BYTES {
            return Err(AssetError::Limit("manifest length out of range".into()));
        }
        self.manifest_offset
            .checked_add(self.manifest_len)
            .and_then(|prefix| prefix.checked_add(SIGNATURE_LEN + FOOTER_LEN))
            .ok_or_else(|| AssetError::Layout("lengths overflow".into()))?;
        Ok(())
    }

    /// Encode these field values as a footer. This writes the values as supplied; it does not
    /// validate a caller-constructed layout. Use [`Self::decode`] to validate serialized bytes.
    pub fn encode(&self) -> [u8; FOOTER_LEN as usize] {
        let mut out = [0u8; FOOTER_LEN as usize];
        out[0..8].copy_from_slice(&MAGIC);
        out[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        out[10..12].copy_from_slice(&0u16.to_le_bytes());
        out[12..16].copy_from_slice(&(FOOTER_LEN as u32).to_le_bytes());
        out[16..24].copy_from_slice(&self.signed_prefix_len().to_le_bytes());
        out[24..32].copy_from_slice(&self.base_executable_len.to_le_bytes());
        out[32..40].copy_from_slice(&self.manifest_offset.to_le_bytes());
        out[40..48].copy_from_slice(&self.manifest_len.to_le_bytes());
        out[48..56].copy_from_slice(&self.signed_prefix_len().to_le_bytes());
        out[56..60].copy_from_slice(&(SIGNATURE_LEN as u32).to_le_bytes());
        out[60..64].copy_from_slice(&ALGORITHM_P256_SHA256.to_le_bytes());
        out[64..96].copy_from_slice(&self.key_id.0);
        out[96..128].copy_from_slice(&self.manifest_sha256);
        out
    }

    /// `Ok(None)` when the bytes do not start with the magic: the file is an unpackaged build.
    /// Every other field must hold exactly; `file_len` is the length of the file read.
    pub fn decode(
        bytes: &[u8; FOOTER_LEN as usize],
        file_len: u64,
    ) -> Result<Option<Self>, AssetError> {
        if bytes[0..8] != MAGIC {
            return Ok(None);
        }
        let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("4"));
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().expect("8"));
        let version = u16_at(8);
        if version != FORMAT_VERSION {
            return Err(AssetError::UnsupportedVersion(u32::from(version)));
        }
        if u16_at(10) != 0 {
            return Err(footer_error("unknown flags"));
        }
        if u64::from(u32_at(12)) != FOOTER_LEN {
            return Err(footer_error("footer length is not 128"));
        }
        if u64::from(u32_at(56)) != SIGNATURE_LEN || u32_at(60) != ALGORITHM_P256_SHA256 {
            return Err(footer_error("unsupported signature length or algorithm"));
        }
        let footer = Self {
            base_executable_len: u64_at(24),
            manifest_offset: u64_at(32),
            manifest_len: u64_at(40),
            key_id: KeyId(bytes[64..96].try_into().expect("32")),
            manifest_sha256: bytes[96..128].try_into().expect("32"),
        };
        footer.check()?;
        if u64_at(16) != footer.signed_prefix_len() || u64_at(48) != footer.signed_prefix_len() {
            return Err(AssetError::Layout(
                "signed prefix and signature offset disagree with the manifest range".into(),
            ));
        }
        if footer.file_len() != file_len {
            return Err(AssetError::Layout(format!(
                "footer implies {} bytes, the file has {file_len}",
                footer.file_len()
            )));
        }
        Ok(Some(footer))
    }
}

/// A package whose footer, manifest hash and signature have all been checked. Assets are
/// authenticated and decoded on demand.
#[derive(Debug)]
pub struct Package {
    file: File,
    path: PathBuf,
    file_len: u64,
    footer: Footer,
    manifest: Manifest,
    signer: TrustedKey,
    signed_manifest: Vec<u8>,
    signed_footer: [u8; FOOTER_LEN as usize],
    signed_signature: [u8; SIGNATURE_LEN as usize],
}

/// What [`Package::verify_full`] checked.
#[derive(Clone, Debug, serde::Serialize)]
pub struct FullReport {
    /// Lowercase SHA-256 digest of the complete package file.
    pub file_sha256: String,
    /// Complete package file length in bytes.
    pub file_bytes: u64,
    /// Executable region length in bytes.
    pub base_executable_bytes: u64,
    /// Lowercase SHA-256 digest of the executable region.
    pub base_executable_sha256: String,
    /// Payload length, including alignment padding and gaps.
    pub payload_bytes: u64,
    /// Canonical manifest length in bytes.
    pub manifest_bytes: u64,
    /// Number of manifest entries checked.
    pub entries: usize,
    /// Total decoded asset bytes checked.
    pub decoded_bytes: u64,
    /// Total stored frame bytes checked.
    pub stored_bytes: u64,
}

impl Package {
    /// The running executable, checked against this build's trusted keys. `Ok(None)` for an
    /// unpackaged build. `current_exe` resolves the executable path on supported platforms.
    pub fn open_self() -> Result<Option<Self>, AssetError> {
        let executable = std::env::current_exe()?;
        Self::open_with_keys(&executable, &verify::trusted_keys())
    }

    /// Any package, checked against this build's trusted keys.
    pub fn open(path: &Path) -> Result<Option<Self>, AssetError> {
        Self::open_with_keys(path, &verify::trusted_keys())
    }

    /// Open a package using exactly `trusted`; `Ok(None)` means the file has no package footer.
    pub fn open_with_keys(path: &Path, trusted: &[TrustedKey]) -> Result<Option<Self>, AssetError> {
        let file = File::open(path)?;
        let file_len = file.metadata()?.len();
        if file_len < FOOTER_LEN {
            return Ok(None);
        }
        let mut raw = [0u8; FOOTER_LEN as usize];
        file.read_exact_at(&mut raw, file_len - FOOTER_LEN)?;
        let Some(footer) = Footer::decode(&raw, file_len)? else {
            return Ok(None);
        };
        // Authenticate before parsing: hash the manifest, verify the signature over the footer.
        let mut manifest_bytes = vec![0u8; footer.manifest_len as usize];
        file.read_exact_at(&mut manifest_bytes, footer.manifest_offset)?;
        let digest: [u8; 32] = {
            use sha2::{Digest, Sha256};
            Sha256::digest(&manifest_bytes).into()
        };
        if digest != footer.manifest_sha256 {
            return Err(AssetError::ManifestHash);
        }
        let mut signature = [0u8; SIGNATURE_LEN as usize];
        file.read_exact_at(&mut signature, footer.signed_prefix_len())?;
        let signer = verify::verify(
            &verify::signing_message(&raw),
            &signature,
            footer.key_id,
            trusted,
        )?
        .clone();
        let manifest = Manifest::parse(
            &manifest_bytes,
            footer.manifest_offset - footer.payload_offset(),
        )?;
        if manifest.code.len != footer.base_executable_len {
            return Err(AssetError::Layout(
                "manifest and footer disagree on the executable length".into(),
            ));
        }
        // The signer's role must match the audience: the development key signs only
        // development-scope packages, and a release key never signs one.
        let development = manifest.scope == crate::Scope::Development;
        if development != (signer.role == "development") {
            return Err(AssetError::Manifest(format!(
                "a {} key may not sign a {} package",
                signer.role,
                manifest.scope.as_str()
            )));
        }
        // The executable region is a few MiB: authenticate it on every open, so a package's
        // signed manifest cannot be grafted onto another executable. The payload stays lazy.
        if sha256_region(&file, 0, footer.base_executable_len)? != manifest.code.sha256 {
            return Err(AssetError::CodeHash);
        }
        Ok(Some(Self {
            file,
            path: path.to_owned(),
            file_len,
            footer,
            manifest,
            signer,
            signed_manifest: manifest_bytes,
            signed_footer: raw,
            signed_signature: signature,
        }))
    }

    /// Exact authenticated envelope, for a model loader that independently checks
    /// the release signature and its own semantic/tensor catalog. No reserialization
    /// or reopening a mutable pathname is involved.
    pub fn signed_envelope(&self) -> (&[u8], &[u8; 128], &[u8; 64]) {
        (
            &self.signed_manifest,
            &self.signed_footer,
            &self.signed_signature,
        )
    }

    /// Return the authenticated manifest.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Return the path used to open this package.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Return the trusted key that authenticated the package.
    pub fn signer(&self) -> &TrustedKey {
        &self.signer
    }

    /// Return the validated package footer.
    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    /// Return the opened package length in bytes.
    pub fn file_len(&self) -> u64 {
        self.file_len
    }

    /// Find one manifest entry by its logical id.
    pub fn entry(&self, id: &str) -> Option<&Entry> {
        self.manifest.entry(id)
    }

    /// Authenticate, decode and check one asset.
    pub fn read_entry(&self, id: &str) -> Result<VerifiedBytes, AssetError> {
        let entry = self
            .entry(id)
            .ok_or_else(|| AssetError::EntryMissing(id.to_owned()))?;
        Ok(VerifiedBytes::new(entry, self.decode(entry)?))
    }

    fn decode(&self, entry: &Entry) -> Result<Vec<u8>, AssetError> {
        let base = self.footer.payload_offset();
        let read = |frame: &crate::Frame| -> std::io::Result<Vec<u8>> {
            let mut buffer = vec![0u8; frame.len as usize];
            self.file.read_exact_at(&mut buffer, base + frame.offset)?;
            Ok(buffer)
        };
        decode::decode_entry(entry, &read)
    }

    fn zero(&self, from: u64, to: u64) -> Result<(), AssetError> {
        let mut offset = from;
        let mut buffer = vec![0u8; (1 << 20).min((to - from) as usize)];
        while offset < to {
            let take = buffer.len().min((to - offset) as usize);
            self.file.read_exact_at(&mut buffer[..take], offset)?;
            if buffer[..take].iter().any(|b| *b != 0) {
                return Err(AssetError::Padding);
            }
            offset += take as u64;
        }
        Ok(())
    }

    /// Cover every byte: the executable hash, the zero padding and every zero gap between frames,
    /// every asset authenticated and decoded, and the whole-file sha256.
    pub fn verify_full(&self) -> Result<FullReport, AssetError> {
        use sha2::{Digest, Sha256};
        let code_len = self.footer.base_executable_len;
        let mut buffer = vec![0u8; 8 << 20];
        let code_sha256 = sha256_region(&self.file, 0, code_len)?;
        if code_sha256 != self.manifest.code.sha256 {
            return Err(AssetError::CodeHash);
        }
        let payload = self.footer.payload_offset();
        self.zero(code_len, payload)?;
        let mut cursor = payload;
        for frame in self.manifest.unique_frames() {
            self.zero(cursor, payload + frame.offset)?;
            cursor = payload + frame.offset + frame.len;
        }
        self.zero(cursor, self.footer.manifest_offset)?;
        let (mut decoded, mut stored) = (0u64, 0u64);
        for entry in &self.manifest.entries {
            decoded += self.decode(entry)?.len() as u64;
            stored += entry.stored_bytes;
        }
        let mut hasher = Sha256::new();
        let mut offset = 0u64;
        while offset < self.file_len {
            let take = buffer.len().min((self.file_len - offset) as usize);
            self.file.read_exact_at(&mut buffer[..take], offset)?;
            hasher.update(&buffer[..take]);
            offset += take as u64;
        }
        Ok(FullReport {
            file_sha256: hex::encode(hasher.finalize()),
            file_bytes: self.file_len,
            base_executable_bytes: code_len,
            base_executable_sha256: code_sha256,
            payload_bytes: self.footer.manifest_offset - payload,
            manifest_bytes: self.footer.manifest_len,
            entries: self.manifest.entries.len(),
            decoded_bytes: decoded,
            stored_bytes: stored,
        })
    }

    /// sha256 of the executable region alone (authenticated when the package was opened).
    /// Recompute the executable-region digest from the already-opened package.
    pub fn base_executable_sha256(&self) -> Result<String, AssetError> {
        sha256_region(&self.file, 0, self.footer.base_executable_len)
    }
}

impl crate::VerifiedAssetSource for Package {
    fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    fn read_verified(&self, id: &str) -> Result<VerifiedBytes, AssetError> {
        self.read_entry(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Footer {
        Footer {
            base_executable_len: 10_000,
            manifest_offset: 20_480,
            manifest_len: 300,
            key_id: KeyId([9; 32]),
            manifest_sha256: [7; 32],
        }
    }

    #[test]
    fn footer_round_trips_with_the_specified_offsets() {
        let footer = sample();
        let bytes = footer.encode();
        assert_eq!(&bytes[0..8], b"JOPEEXE1");
        assert_eq!(
            u64::from_le_bytes(bytes[16..24].try_into().unwrap()),
            20_780
        );
        assert_eq!(
            u64::from_le_bytes(bytes[48..56].try_into().unwrap()),
            20_780
        );
        assert_eq!(footer.payload_offset(), 12_288);
        assert_eq!(footer.file_len(), 20_780 + 64 + 128);
        assert_eq!(
            Footer::decode(&bytes, footer.file_len()).unwrap(),
            Some(footer)
        );
    }

    #[test]
    fn missing_magic_means_unpackaged() {
        assert_eq!(Footer::decode(&[0u8; 128], 128).unwrap(), None);
    }

    #[test]
    fn every_footer_rule_is_enforced() {
        let good = sample().encode();
        let len = sample().file_len();
        let mutate = |at: usize, value: u8| {
            let mut bytes = good;
            bytes[at] = value;
            Footer::decode(&bytes, len)
        };
        assert!(matches!(
            mutate(8, 2),
            Err(AssetError::UnsupportedVersion(2))
        ));
        assert!(matches!(mutate(10, 1), Err(AssetError::Footer(_))));
        assert!(matches!(mutate(12, 64), Err(AssetError::Footer(_))));
        assert!(matches!(mutate(16, 0), Err(AssetError::Layout(_))));
        assert!(matches!(mutate(48, 0), Err(AssetError::Layout(_))));
        assert!(matches!(mutate(56, 72), Err(AssetError::Footer(_))));
        assert!(matches!(mutate(60, 2), Err(AssetError::Footer(_))));
        assert!(matches!(
            Footer::decode(&good, len + 1),
            Err(AssetError::Layout(_))
        ));
        let mut early = sample();
        early.manifest_offset = 4096;
        assert!(matches!(
            Footer::decode(&early.encode(), early.file_len()),
            Err(AssetError::Layout(_))
        ));
        let mut huge = sample();
        huge.manifest_len = limits::MAX_MANIFEST_BYTES + 1;
        assert!(matches!(
            Footer::decode(&huge.encode(), huge.file_len()),
            Err(AssetError::Limit(_))
        ));
        let mut overflow = good;
        overflow[32..40].copy_from_slice(&(u64::MAX - 10).to_le_bytes());
        assert!(matches!(
            Footer::decode(&overflow, len),
            Err(AssetError::Layout(_))
        ));
    }
}
