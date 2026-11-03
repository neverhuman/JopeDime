//! `jope-dist pack` and `jope-dist seal` (SPECIFICATION.md §10, §16.2).
//!
//! A development-scope profile (the CI fixture) is packed and signed in one step with the
//! development key. Every other scope stops before signing: `pack --stage DIR` writes the staged
//! frames, manifest and footer and `signing-message.bin`, the exact bytes the release host's
//! root-held cosign step signs; `seal` then checks that signature against the release key, joins
//! executable, payload, signature and footer, and re-verifies every byte of the result.

use crate::profile::{lock_profile, Lock, Profile};
use anyhow::{bail, ensure, Context, Result};
use jope_assets::verify::{self, TrustedKey};
use jope_assets::writer::{self, EntryInput, StageOptions, Staged};
use jope_assets::{Codec, Footer, License, Manifest, Package, Scope, SourceIdentity};
use serde::Serialize;
use std::path::{Path, PathBuf};
use std::time::Instant;

pub const RECEIPT_SCHEMA: &str = "jope.appliance-pack-receipt/v1";
pub const STAGE_SCHEMA: &str = "jope.appliance-stage/v1";

pub struct PackRequest<'a> {
    pub profile: &'a Path,
    pub lock: &'a Path,
    pub repo: &'a Path,
    /// The executable Cargo built (`--profile dist`, symbols stripped by Cargo).
    pub code: &'a Path,
    /// The verified model cache: files at `<models>/bundles/<bundle_id>/<path>`.
    pub models: &'a Path,
    /// `jope-dist notices` output: the third-party inventory with licence texts.
    pub notices: &'a Path,
    pub source_sha: &'a str,
    pub target: &'a str,
    /// The release key's public PEM, for non-development scopes; default: the compiled-in key.
    pub release_key: Option<&'a Path>,
}

pub enum Output<'a> {
    /// Development scope: sign with the development key and write the final file.
    Development { out: &'a Path },
    /// Release scopes: write the stage directory for external signing.
    Stage { dir: &'a Path },
    /// Internal full-profile inspection only: no signing message, footer or sealable stage.
    Rehearsal { dir: &'a Path },
}

#[derive(Serialize)]
pub struct Receipt {
    pub schema: &'static str,
    pub tool: String,
    pub profile: String,
    pub scope: &'static str,
    pub source_sha40: String,
    pub base_executable_sha256: String,
    pub base_executable_bytes: u64,
    pub manifest_sha256: String,
    pub signing_key: String,
    pub file: Option<String>,
    pub file_sha256: Option<String>,
    pub file_bytes: Option<u64>,
    pub payload_bytes: u64,
    pub entries: Vec<ReceiptEntry>,
    pub seconds: f64,
}

#[derive(Serialize)]
pub struct ReceiptEntry {
    pub id: String,
    pub sha256: String,
    pub bytes: u64,
    pub stored_bytes: u64,
    pub codec: Codec,
}

fn codec_of(name: &str) -> Codec {
    match name {
        "zstd" => Codec::Zstd,
        "raw" => Codec::Raw,
        _ => Codec::ShuffleZstd,
    }
}

/// Read one locked file from the model cache and check it against the lock.
fn read_locked(
    models: &Path,
    bundle: &str,
    path: &str,
    sha256: &str,
    bytes: u64,
) -> Result<Vec<u8>> {
    ensure!(
        jope_assets::manifest::is_name(bundle) && jope_assets::manifest::is_name(path),
        "{bundle}/{path} is not a plain relative path"
    );
    let file = models.join("bundles").join(bundle).join(path);
    let data = std::fs::read(&file).with_context(|| {
        format!(
            "read {} (fetch it first: jope models fetch {bundle})",
            file.display()
        )
    })?;
    ensure!(
        data.len() as u64 == bytes && jope_assets::sha256_hex(&data) == sha256,
        "{} does not match the lock (sha256 {sha256}, {bytes} bytes)",
        file.display()
    );
    Ok(data)
}

/// The key a scope is signed with.
fn signing_key(scope: Scope, release_key: Option<&Path>) -> Result<TrustedKey> {
    if scope == Scope::Development {
        return Ok(verify::development_key());
    }
    match release_key {
        Some(path) => {
            let pem = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            TrustedKey::from_pem(&pem, "release")
                .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))
        }
        None => verify::release_keys()
            .into_iter()
            .next()
            .context("this build compiles in no release key"),
    }
}

/// Refuse a binary whose executable format does not match the supported appliance target.
/// Linux remains ELF-only; the owner-supported macOS target is arm64 Mach-O. Other targets,
/// including Windows PE, are deliberately unsupported.
fn validate_executable_target(code: &[u8], target: &str, path: &Path) -> Result<()> {
    match target {
        "x86_64-unknown-linux-gnu" => {
            let x86_64_elf = code.len() >= 64
                && code.get(0..4) == Some(b"\x7fELF")
                && code.get(4) == Some(&2) // ELFCLASS64
                && code.get(5) == Some(&1) // ELFDATA2LSB
                && code.get(6) == Some(&1) // EI_VERSION: EV_CURRENT
                && code.get(18..20) == Some(&62u16.to_le_bytes()) // EM_X86_64
                && code.get(20..24) == Some(&1u32.to_le_bytes()) // e_version: EV_CURRENT
                && code.get(52..54) == Some(&64u16.to_le_bytes()); // e_ehsize: ELF64 header size
            ensure!(
                x86_64_elf,
                "{} is not a 64-bit little-endian x86_64 ELF file for Linux target {target}",
                path.display()
            );
        }
        "aarch64-apple-darwin" => {
            let arm64_mach_o = code.len() >= 32
                && code.get(0..4) == Some(&[0xcf, 0xfa, 0xed, 0xfe])
                && code.get(4..8) == Some(&0x0100000cu32.to_le_bytes())
                && code.get(12..16) == Some(&2u32.to_le_bytes());
            ensure!(
                arm64_mach_o,
                "{} is not a 64-bit arm64 Mach-O executable for macOS target {target}",
                path.display()
            );
        }
        _ => bail!(
            "unsupported appliance executable target {target}; supported targets are x86_64-unknown-linux-gnu and aarch64-apple-darwin"
        ),
    }
    Ok(())
}

fn notice(component: &str, path: &str, origin: &str, spdx: &str, bytes: Vec<u8>) -> EntryInput {
    EntryInput {
        role: "notice".into(),
        component: component.into(),
        path: path.into(),
        origin: origin.into(),
        license: License {
            spdx: spdx.into(),
            redistribution: "permitted".into(),
        },
        backends: Vec::new(),
        codec: Codec::Zstd,
        bytes,
    }
}

/// Resolve, read and encode everything; the manifest and footer are fixed, unsigned.
mod stage;
pub use stage::stage;

fn validate_proposal_build(
    version: &serde_json::Value,
    methods: &serde_json::Value,
    source: &str,
) -> Result<()> {
    ensure!(
        methods["methods"].as_array().is_some_and(|records| records
            .iter()
            .any(|record| record["id"] == jope_service::proposal::METHOD
                && record["compiled"] == true)),
        "candidate does not contain the compiled r882 proposal adapter"
    );
    ensure!(
        version["features"]["engine"] == true && version["features"]["embedded"] == true,
        "r882 requires an embedded Engine build"
    );
    ensure!(
        version["trusts_development_key"] == false,
        "full candidate must not trust the development key"
    );
    ensure!(
        version["source_sha40"] == source,
        "full candidate source differs from --source-sha"
    );
    Ok(())
}

fn receipt(
    started: Instant,
    staged: &Staged,
    profile: &str,
    out: Option<&Path>,
) -> Result<Receipt> {
    let (file, file_sha256, file_bytes) = match out {
        Some(path) => {
            let bytes = std::fs::read(path)?;
            (
                Some(path.display().to_string()),
                Some(jope_assets::sha256_hex(&bytes)),
                Some(bytes.len() as u64),
            )
        }
        None => (None, None, None),
    };
    Ok(Receipt {
        schema: RECEIPT_SCHEMA,
        tool: format!("jope-dist {}", env!("CARGO_PKG_VERSION")),
        profile: profile.to_owned(),
        scope: staged.manifest.scope.as_str(),
        source_sha40: staged.manifest.source.sha40.clone(),
        base_executable_sha256: staged.manifest.code.sha256.clone(),
        base_executable_bytes: staged.manifest.code.len,
        manifest_sha256: jope_assets::sha256_hex(&staged.manifest_bytes),
        signing_key: staged.footer.key_id.to_string(),
        file,
        file_sha256,
        file_bytes,
        payload_bytes: staged.frames.len() as u64,
        entries: staged
            .manifest
            .entries
            .iter()
            .map(|e| ReceiptEntry {
                id: e.id.clone(),
                sha256: e.sha256.clone(),
                bytes: e.bytes,
                stored_bytes: e.stored_bytes,
                codec: e.codec,
            })
            .collect(),
        seconds: started.elapsed().as_secs_f64(),
    })
}

pub fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut text = serde_json::to_string_pretty(value)?;
    text.push('\n');
    std::fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

/// Write the package beside `out`, re-open it with `key` alone and cover every byte, and only
/// then move it to `out` (which must not exist). A file that fails never appears at `out`.
fn write_verified(bytes: &[u8], out: &Path, key: &TrustedKey) -> Result<()> {
    ensure!(!out.exists(), "{} exists; give a new path", out.display());
    let temp = writer::write_temporary(bytes, out)
        .with_context(|| format!("write beside {}", out.display()))?;
    let checked = (|| -> Result<()> {
        let package = Package::open_with_keys(&temp, std::slice::from_ref(key))
            .context("re-open the written package")?
            .context("the written file carries no package")?;
        package
            .verify_full()
            .context("verify the written package")?;
        Ok(())
    })();
    if let Err(error) = checked {
        let _ = std::fs::remove_file(&temp);
        return Err(error);
    }
    writer::publish(&temp, out).with_context(|| format!("publish {}", out.display()))
}

pub fn pack(
    request: &PackRequest<'_>,
    output: Output<'_>,
    receipt_path: Option<&Path>,
) -> Result<()> {
    let started = Instant::now();
    let rehearsal = matches!(output, Output::Rehearsal { .. });
    let (staged, code, key, profile) = stage::stage_for(request, rehearsal)?;
    match output {
        Output::Rehearsal { dir } => {
            // Deliberately not a release stage. In particular there is nothing for a signer
            // to sign and no footer for `seal` to assemble into a package.
            std::fs::create_dir(dir)
                .with_context(|| format!("create new rehearsal directory {}", dir.display()))?;
            std::fs::write(dir.join("frames.bin"), &staged.frames)?;
            std::fs::write(dir.join("manifest.json"), &staged.manifest_bytes)?;
            let unavailable: Vec<serde_json::Value> =
                jope_service::proposal::uncompiled(&staged.manifest.methods)?
                    .into_iter()
                    .map(|(id, reason)| serde_json::json!({"id": id, "reason": reason}))
                    .collect();
            let record = serde_json::json!({
                "schema": "jope.appliance-rehearsal/v1",
                "purpose": "internal full-asset inspection; not a release candidate",
                "signable": false,
                "release_qualified": false,
                "profile": profile.name,
                "scope": profile.scope,
                "owner_ack": profile.owner_ack,
                "source_sha40": request.source_sha,
                "intended_methods": staged.manifest.methods,
                "unexecutable_methods": unavailable,
                "base_executable_sha256": staged.manifest.code.sha256,
                "base_executable_bytes": staged.manifest.code.len,
                "manifest_sha256": jope_assets::sha256_hex(&staged.manifest_bytes),
                "frames_sha256": jope_assets::sha256_hex(&staged.frames),
                "payload_bytes": staged.frames.len(),
                "assets": receipt(started, &staged, &profile.name, None)?.entries,
            });
            write_json(&dir.join("rehearsal.json"), &record)?;
            if let Some(path) = receipt_path {
                write_json(path, &record)?;
            }
        }
        Output::Development { out } => {
            ensure!(
                staged.manifest.scope == Scope::Development,
                "only development-scope profiles are signed here; release scopes use --stage"
            );
            let bytes = writer::assemble_development(&code, &staged)?;
            write_verified(&bytes, out, &key)?;
            if let Some(path) = receipt_path {
                write_json(path, &receipt(started, &staged, &profile.name, Some(out))?)?;
            }
        }
        Output::Stage { dir } => {
            ensure!(
                staged.manifest.scope != Scope::Development,
                "a development-scope profile is signed with the development key; use --out"
            );
            std::fs::create_dir(dir).with_context(|| {
                format!(
                    "create stage directory {} (it must not exist)",
                    dir.display()
                )
            })?;
            std::fs::write(dir.join("frames.bin"), &staged.frames)?;
            std::fs::write(dir.join("manifest.json"), &staged.manifest_bytes)?;
            std::fs::write(dir.join("footer.bin"), staged.footer.encode())?;
            std::fs::write(dir.join("signing-message.bin"), staged.signing_message())?;
            write_json(
                &dir.join("stage.json"),
                &serde_json::json!({
                    "schema": STAGE_SCHEMA,
                    "profile": profile.name,
                    "base_executable_sha256": staged.manifest.code.sha256,
                    "base_executable_bytes": staged.manifest.code.len,
                    "manifest_sha256": jope_assets::sha256_hex(&staged.manifest_bytes),
                    "frames_sha256": jope_assets::sha256_hex(&staged.frames),
                    "signing_message_sha256": jope_assets::sha256_hex(&staged.signing_message()),
                    "signing_key": staged.footer.key_id.to_string(),
                    "sign_with": "cosign sign-blob --key <release key> --output-signature manifest.sig signing-message.bin",
                }),
            )?;
            if let Some(path) = receipt_path {
                write_json(path, &receipt(started, &staged, &profile.name, None)?)?;
            }
        }
    }
    Ok(())
}

/// Join a stage directory, the executable it names, and the signature of its signing message.
pub fn seal(
    code: &Path,
    stage_dir: &Path,
    signature: &Path,
    release_key: Option<&Path>,
    out: &Path,
    receipt_path: Option<&Path>,
) -> Result<()> {
    let started = Instant::now();
    ensure!(
        !stage_dir.join("rehearsal.json").exists(),
        "rehearsal output cannot be sealed; resolve the profile blockers and pack a release stage"
    );
    let code_bytes = std::fs::read(code).with_context(|| format!("read {}", code.display()))?;
    let read = |name: &str| -> Result<Vec<u8>> {
        let path = stage_dir.join(name);
        std::fs::read(&path).with_context(|| format!("read {}", path.display()))
    };
    let frames = read("frames.bin")?;
    let manifest_bytes = read("manifest.json")?;
    let footer_bytes = read("footer.bin")?;
    let manifest =
        Manifest::parse(&manifest_bytes, frames.len() as u64).context("staged manifest")?;
    ensure!(
        manifest.code.sha256 == jope_assets::sha256_hex(&code_bytes),
        "the staged manifest names other executable bytes than {}",
        code.display()
    );
    let key = signing_key(manifest.scope, release_key)?;
    let footer = Footer {
        base_executable_len: code_bytes.len() as u64,
        manifest_offset: jope_assets::align_up(code_bytes.len() as u64).context("size")?
            + frames.len() as u64,
        manifest_len: manifest_bytes.len() as u64,
        key_id: key.id,
        manifest_sha256: {
            let digest = jope_assets::sha256_hex(&manifest_bytes);
            hex::decode(digest)?.try_into().expect("32 bytes")
        },
    };
    ensure!(
        footer.encode().as_slice() == footer_bytes.as_slice(),
        "the staged footer does not match the stage's executable, manifest and the release key"
    );
    let signature = crate::signature::read(signature)?;
    let staged = Staged {
        manifest,
        manifest_bytes,
        frames,
        footer,
    };
    let bytes = writer::assemble(&code_bytes, &staged, &signature, &key)
        .context("the signature does not verify against the release key over this stage")?;
    write_verified(&bytes, out, &key)?;
    if let Some(path) = receipt_path {
        let profile = staged.manifest.profile.clone();
        write_json(path, &receipt(started, &staged, &profile, Some(out))?)?;
    }
    Ok(())
}

/// `<repo>/deploy/appliance/<relative>`.
pub fn appliance_path(repo: &Path, relative: &str) -> PathBuf {
    repo.join("deploy/appliance").join(relative)
}

#[cfg(test)]
mod tests;
