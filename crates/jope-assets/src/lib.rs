//! The signed single-file appliance package (`deploy/appliance/SPECIFICATION.md` §9–10, as
//! amended by `deploy/appliance/AMENDMENTS.md` A5).
//!
//! A packaged `jopedime` is one ELF file:
//!
//! ```text
//! [ executable ][ zero padding to 4096 ][ asset frames, each 4096-aligned ]
//! [ manifest (canonical JSON) ][ signature (64 B) ][ footer (128 B) ]
//! ```
//!
//! The [`Footer`] locates every region and carries the signing key id and the manifest's sha256.
//! The signature is ECDSA P-256/SHA-256 over [`signing_message`] — a domain string followed by the
//! exact footer bytes — so it authenticates the footer, therefore the manifest, and through the
//! manifest the executable's sha256 and every asset's stored and decoded sha256.
//!
//! [`Package`] checks the footer equations, the manifest hash and the signature before it parses
//! the manifest, and authenticates each asset's stored bytes against the manifest *before* they
//! reach the decompressor; the decoded bytes are checked again. [`Package::verify_full`]
//! additionally hashes the executable region, the padding and every gap, so every byte of the file
//! is covered. Startup therefore costs one footer, one manifest, one signature and a hash of the
//! executable region, rather than a hash of the whole file, and an asset is read and verified only when a method needs it.
//!
//! The reader is pure Rust: positioned reads, `ruzstd`, `p256`. The writer (feature `pack`) uses
//! the zstd C library and is linked only into the build-time packer, `jope-dist`.
#![forbid(unsafe_code)]

pub mod decode;
pub mod envelope;
pub mod manifest;
pub mod source;
pub mod verify;
#[cfg(feature = "pack")]
pub mod writer;

pub use decode::Codec;
pub use envelope::{Footer, FullReport, Package};
pub use manifest::{CodeIdentity, Entry, Frame, License, Manifest, Scope, SourceIdentity};
pub use source::{VerifiedAssetSource, VerifiedBytes};
pub use verify::{signing_message, KeyId, TrustedKey};

/// `schema` of every manifest this crate writes and accepts.
pub const MANIFEST_SCHEMA: &str = "jope.appliance-manifest/v1";
/// Envelope (footer) version and manifest `format_version`.
pub const FORMAT_VERSION: u16 = 1;
/// The footer's first eight bytes.
pub const MAGIC: [u8; 8] = *b"JOPEEXE1";
/// Length of the footer at the end of the file.
pub const FOOTER_LEN: u64 = 128;
/// Fixed-width `r || s` P-256 signature.
pub const SIGNATURE_LEN: u64 = 64;
/// Footer `algorithm`: ECDSA P-256 over SHA-256.
pub const ALGORITHM_P256_SHA256: u32 = 1;
/// Prefix of every signed message; a signature over anything else never verifies as a package.
pub const SIGNING_DOMAIN: &[u8] = b"jope.singlefile.manifest-signature.v1\0";
/// Alignment of the first frame and of every frame.
pub const ALIGN: u64 = 4096;
/// Default decoded size of one frame. Frames decode independently and in parallel.
pub const DEFAULT_FRAME_BYTES: usize = 16 << 20;

/// Bounds enforced before any allocation they size (deploy/appliance/limits.toml).
pub mod limits {
    /// Manifest bytes.
    pub const MAX_MANIFEST_BYTES: u64 = 16 << 20;
    /// Entries in one manifest.
    pub const MAX_ENTRIES: usize = 4096;
    /// Promised methods in one manifest.
    pub const MAX_METHODS: usize = 1024;
    /// Decoded bytes of one entry.
    pub const MAX_ENTRY_BYTES: u64 = 8 << 30;
    /// Decoded bytes of one frame.
    pub const MAX_FRAME_RAW_BYTES: u64 = 64 << 20;
    /// zstd window of one frame.
    pub const MAX_ZSTD_WINDOW: u64 = 64 << 20;
    /// Length of an entry id, method id, or other manifest string.
    pub const MAX_NAME_BYTES: usize = 256;
}

/// Every way a package is refused. Each check has its own variant so tests prove which one fired.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// The package file could not be read or written.
    #[error("package I/O: {0}")]
    Io(#[from] std::io::Error),
    /// The footer bytes are malformed or contain unsupported fields.
    #[error("package footer is malformed: {0}")]
    Footer(String),
    /// Package offsets or lengths do not describe a valid layout.
    #[error("package layout is inconsistent: {0}")]
    Layout(String),
    /// The manifest bytes do not match the digest stored in the footer.
    #[error("package manifest does not match the sha256 in the footer")]
    ManifestHash,
    /// The manifest is malformed or violates a structural rule.
    #[error("package manifest is not valid: {0}")]
    Manifest(String),
    /// The package format version is not supported by this build.
    #[error("package format version {0} is not supported (this build reads version 1)")]
    UnsupportedVersion(u32),
    /// No supplied trusted key matches the footer's key identity.
    #[error("package signing key {0} is not trusted by this build")]
    UntrustedKey(String),
    /// The package signature is malformed or does not verify.
    #[error("package signature does not verify")]
    Signature,
    /// A declared size or count exceeds a configured package bound.
    #[error("package exceeds a bound: {0}")]
    Limit(String),
    /// The requested logical asset id is absent from the manifest.
    #[error("package has no asset {0}")]
    EntryMissing(String),
    /// Stored bytes for the named logical asset do not match their digest.
    #[error("package asset {id} stored bytes do not match their sha256")]
    StoredHash {
        /// Logical id of the asset whose stored bytes failed verification.
        id: String,
    },
    /// The named logical asset could not be decoded.
    #[error("package asset {id} could not be decoded: {reason}")]
    Decode {
        /// Logical id of the asset that failed decoding.
        id: String,
        /// Decoder failure detail.
        reason: String,
    },
    /// Decoded bytes for the named logical asset do not match their digest.
    #[error("package asset {id} does not match its sha256")]
    EntryHash {
        /// Logical id of the asset whose decoded bytes failed verification.
        id: String,
    },
    /// Executable bytes do not match the manifest's executable digest.
    #[error("package executable bytes do not match the manifest")]
    CodeHash,
    /// Executable padding or a payload gap contains nonzero bytes.
    #[error("package padding or gap bytes are not zero")]
    Padding,
}

/// Lower-case hex sha256.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(Sha256::digest(bytes))
}

/// Round `value` up to the next multiple of [`ALIGN`], or `None` on overflow.
pub fn align_up(value: u64) -> Option<u64> {
    value.checked_next_multiple_of(ALIGN)
}
