//! Build a package (feature `pack`, used by `jope-dist`). [`stage`] encodes the assets and fixes
//! the manifest and footer; the [`signing_message`](crate::signing_message) over that footer is
//! then signed (by `cosign sign-blob` on the release host, or the development key for fixtures);
//! [`assemble`] or [`seal`] writes the final file. The same inputs give the same bytes: no
//! timestamps, paths or random values enter the package.

use crate::{
    align_up, decode, manifest::Manifest, sha256_hex, verify, AssetError, CodeIdentity, Codec,
    Entry, Footer, Frame, KeyId, License, Scope, SourceIdentity, ALIGN, FORMAT_VERSION,
    MANIFEST_SCHEMA,
};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

/// One asset to embed.
#[derive(Clone, Debug)]
pub struct EntryInput {
    /// Manifest role, such as `model`, `config`, `manifest`, or `notice`.
    pub role: String,
    /// Owning component name.
    pub component: String,
    /// The path inside the component, which ends the logical id.
    pub path: String,
    /// Source identity or generated-content description.
    pub origin: String,
    /// Redistribution terms carried into the manifest.
    pub license: License,
    /// Backends qualified for this asset, such as `cpu` or `cuda`.
    pub backends: Vec<String>,
    /// Requested storage codec; encoding may fall back to raw when compression is larger.
    pub codec: Codec,
    /// Decoded asset bytes to package.
    pub bytes: Vec<u8>,
}

impl EntryInput {
    /// Return the logical manifest id formed from role, component, and path.
    pub fn id(&self) -> String {
        format!("{}/{}/{}", self.role, self.component, self.path)
    }
}

/// Everything in the manifest that is not derived from the assets or the executable.
#[derive(Clone, Debug)]
pub struct StageOptions {
    /// Product name written to the manifest.
    pub product: String,
    /// Product version written to the manifest.
    pub product_version: String,
    /// Rust target triple for the executable region.
    pub target: String,
    /// Appliance profile path recorded in the manifest.
    pub profile: String,
    /// Maximum redistribution scope of this package.
    pub scope: Scope,
    /// Source commit and lock/profile hashes for this package.
    pub source: SourceIdentity,
    /// Sorted method ids promised by this profile.
    pub methods: Vec<String>,
    /// Key identity that the release signer must use.
    pub key_id: KeyId,
    /// Target maximum size for each encoded frame.
    pub frame_bytes: usize,
    /// Compression level passed to the selected codec.
    pub level: i32,
}

/// Everything but the signature.
#[derive(Clone, Debug)]
pub struct Staged {
    /// Parsed manifest represented by `manifest_bytes`.
    pub manifest: Manifest,
    /// Canonical manifest bytes that are signed into the footer.
    pub manifest_bytes: Vec<u8>,
    /// The bytes from the payload start to the manifest: aligned frames and zero gaps.
    pub frames: Vec<u8>,
    /// Footer whose signature is still to be supplied.
    pub footer: Footer,
}

impl Staged {
    /// The exact bytes to sign.
    pub fn signing_message(&self) -> Vec<u8> {
        verify::signing_message(&self.footer.encode())
    }
}

fn invalid(error: AssetError) -> std::io::Error {
    std::io::Error::other(error.to_string())
}

/// One physical copy of some content: the codec used, its frames, stored sha256 and length.
type Written = (Codec, Vec<Frame>, String, u64);

/// Encode the assets, deduplicating identical content, and fix the manifest and footer.
pub fn stage(
    code: &[u8],
    mut inputs: Vec<EntryInput>,
    options: &StageOptions,
) -> std::io::Result<Staged> {
    if code.is_empty() {
        return Err(std::io::Error::other("the executable region is empty"));
    }
    inputs.sort_by_key(EntryInput::id);
    if inputs.windows(2).any(|w| w[0].id() == w[1].id()) {
        return Err(std::io::Error::other("two assets share one logical id"));
    }
    let mut region = Vec::new();
    let mut entries = Vec::with_capacity(inputs.len());
    // One physical copy per identical (content, requested codec).
    let mut written: BTreeMap<(String, Codec), Written> = BTreeMap::new();
    for input in inputs {
        let sha256 = sha256_hex(&input.bytes);
        let key = (sha256.clone(), input.codec);
        let (codec, frames, stored_sha256, stored_bytes) = match written.get(&key) {
            Some(existing) => existing.clone(),
            None => {
                use sha2::{Digest, Sha256};
                let (codec, stored) = decode::encode(
                    input.codec,
                    &input.bytes,
                    options.frame_bytes,
                    options.level,
                )?;
                let mut frames = Vec::with_capacity(stored.len());
                let mut hasher = Sha256::new();
                let mut total = 0u64;
                for (bytes, raw_len) in stored {
                    let offset = align_up(region.len() as u64).expect("in memory");
                    region.resize(offset as usize, 0);
                    region.extend_from_slice(&bytes);
                    hasher.update(&bytes);
                    total += bytes.len() as u64;
                    frames.push(Frame {
                        offset,
                        len: bytes.len() as u64,
                        raw_len,
                    });
                }
                let value = (codec, frames, hex::encode(hasher.finalize()), total);
                written.insert(key, value.clone());
                value
            }
        };
        let mut backends = input.backends.clone();
        backends.sort();
        backends.dedup();
        entries.push(Entry {
            id: input.id(),
            role: input.role,
            component: input.component,
            origin: input.origin,
            sha256,
            bytes: input.bytes.len() as u64,
            codec,
            stored_sha256,
            stored_bytes,
            frames,
            license: input.license,
            backends,
        });
    }
    let mut methods = options.methods.clone();
    methods.sort();
    methods.dedup();
    let manifest = Manifest {
        schema: MANIFEST_SCHEMA.to_owned(),
        format_version: u32::from(FORMAT_VERSION),
        product: options.product.clone(),
        product_version: options.product_version.clone(),
        target: options.target.clone(),
        profile: options.profile.clone(),
        scope: options.scope,
        code: CodeIdentity {
            len: code.len() as u64,
            sha256: sha256_hex(code),
        },
        source: options.source.clone(),
        methods,
        entries,
    };
    manifest.check(region.len() as u64).map_err(invalid)?;
    let manifest_bytes = manifest.to_bytes();
    let payload_offset = align_up(code.len() as u64).expect("in memory");
    debug_assert!(payload_offset.is_multiple_of(ALIGN));
    let footer = Footer {
        base_executable_len: code.len() as u64,
        manifest_offset: payload_offset + region.len() as u64,
        manifest_len: manifest_bytes.len() as u64,
        key_id: options.key_id,
        manifest_sha256: {
            use sha2::{Digest, Sha256};
            Sha256::digest(&manifest_bytes).into()
        },
    };
    Ok(Staged {
        manifest,
        manifest_bytes,
        frames: region,
        footer,
    })
}

/// The complete file. `signature` is the fixed-width low-S form
/// ([`normalize_der`](crate::verify::normalize_der) of cosign's output); it is verified against
/// `signer` before anything is assembled.
pub fn assemble(
    code: &[u8],
    staged: &Staged,
    signature: &[u8; 64],
    signer: &crate::TrustedKey,
) -> std::io::Result<Vec<u8>> {
    if staged.manifest.code.sha256 != sha256_hex(code) {
        return Err(std::io::Error::other(
            "the staged manifest names different executable bytes",
        ));
    }
    if signer.id != staged.footer.key_id {
        return Err(std::io::Error::other(
            "the signer is not the key the footer names",
        ));
    }
    verify::verify(
        &staged.signing_message(),
        signature,
        staged.footer.key_id,
        std::slice::from_ref(signer),
    )
    .map_err(invalid)?;
    let payload_offset = staged.footer.payload_offset();
    let mut out = Vec::with_capacity(staged.footer.file_len() as usize);
    out.extend_from_slice(code);
    out.resize(payload_offset as usize, 0);
    out.extend_from_slice(&staged.frames);
    out.extend_from_slice(&staged.manifest_bytes);
    out.extend_from_slice(signature);
    out.extend_from_slice(&staged.footer.encode());
    debug_assert_eq!(out.len() as u64, staged.footer.file_len());
    Ok(out)
}

/// Sign with the development key and assemble: the fixture path.
pub fn assemble_development(code: &[u8], staged: &Staged) -> std::io::Result<Vec<u8>> {
    let der = verify::sign_with_development_key(&staged.signing_message());
    let signature = verify::normalize_der(&der).map_err(invalid)?;
    assemble(code, staged, &signature, &verify::development_key())
}

/// Write `bytes` to a temporary file beside `out` (mode 0755, synced) and return its path. The
/// caller checks it and then [`publish`]es it, so a file that fails a check never appears at
/// `out`.
pub fn write_temporary(bytes: &[u8], out: &Path) -> std::io::Result<std::path::PathBuf> {
    use std::os::unix::fs::OpenOptionsExt;
    let name = out
        .file_name()
        .ok_or_else(|| std::io::Error::other("output path has no file name"))?;
    let temp = out.with_file_name(format!(
        ".{}.{}.tmp",
        name.to_string_lossy(),
        std::process::id()
    ));
    // A create_new refusal does not confer ownership of the existing path. Only a
    // temporary file opened by this call may be removed on a later write failure.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o755)
        .open(&temp)?;
    let result = (|| -> std::io::Result<()> {
        file.write_all(bytes)?;
        file.sync_all()
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    Ok(temp)
}

/// Move a checked temporary file to `out` without overwriting anything (a hard link, then the
/// temporary name is removed). The temporary file is removed on failure too.
pub fn publish(temp: &Path, out: &Path) -> std::io::Result<()> {
    let result = std::fs::hard_link(temp, out);
    let _ = std::fs::remove_file(temp);
    result
}

/// Write `bytes` to `out`, which must not exist, atomically: no partial file is ever visible.
pub fn write_new(bytes: &[u8], out: &Path) -> std::io::Result<()> {
    if out.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            format!("{} exists", out.display()),
        ));
    }
    let temp = write_temporary(bytes, out)?;
    publish(&temp, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regression_temporary_collision_preserves_foreign_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("package");
        let temp = dir
            .path()
            .join(format!(".package.{}.tmp", std::process::id()));
        std::fs::write(&temp, b"pre-existing source").unwrap();
        let error = write_temporary(b"new package", &out).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(
            std::fs::read(&temp).expect("unowned file must remain"),
            b"pre-existing source"
        );
        assert!(!out.exists());
    }
}
