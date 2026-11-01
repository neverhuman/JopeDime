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
        "x86_64-unknown-linux-gnu" => ensure!(
            code.get(0..4) == Some(b"\x7fELF"),
            "{} is not an ELF file for Linux target {target}",
            path.display()
        ),
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
pub fn stage(request: &PackRequest<'_>) -> Result<(Staged, Vec<u8>, TrustedKey, Profile)> {
    let (profile, profile_sha) = Profile::load(request.profile)?;
    let (lock, lock_sha) = Lock::load(request.lock)?;
    let locked = lock
        .profiles
        .get(&profile.name)
        .with_context(|| format!("the lock has no profile {}", profile.name))?;
    let derived = lock_profile(&profile, &profile_sha, request.repo)?;
    ensure!(
        &derived == locked,
        "{} is stale for profile {}: run `jope-dist lock` and review the diff",
        request.lock.display(),
        profile.name
    );
    if locked.status != "ready" {
        bail!(
            "profile {} is blocked:\n  {}",
            profile.name,
            locked.blockers.join("\n  ")
        );
    }
    ensure!(
        request.source_sha.len() == 40 && request.source_sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "--source-sha must be a full 40-hex commit"
    );
    let code = std::fs::read(request.code)
        .with_context(|| format!("read executable {}", request.code.display()))?;
    validate_executable_target(&code, request.target, request.code)?;
    match Package::open_with_keys(request.code, &[]) {
        Ok(None) => {}
        _ => bail!(
            "{} already ends in a package footer; pack the executable Cargo built",
            request.code.display()
        ),
    }
    let cargo_lock = std::fs::read(request.repo.join("Cargo.lock")).context("read Cargo.lock")?;
    let mut inputs = Vec::new();
    for component in &locked.components {
        for file in &component.files {
            let bytes = read_locked(
                request.models,
                &component.bundle_id,
                &file.path,
                &file.sha256,
                file.bytes,
            )?;
            let codec = if file.path.ends_with(".safetensors") {
                codec_of(&profile.codec.name)
            } else if profile.codec.name == "raw" {
                Codec::Raw
            } else {
                Codec::Zstd
            };
            inputs.push(EntryInput {
                role: component.role.clone(),
                component: component.id.clone(),
                path: file.path.clone(),
                origin: format!("{}/{}", component.bundle_id, file.path),
                license: License {
                    spdx: file.spdx.clone(),
                    redistribution: file.redistribution.clone(),
                },
                backends: component.backends.clone(),
                codec,
                bytes,
            });
        }
    }
    let license = std::fs::read(request.repo.join("LICENSE")).context("read LICENSE")?;
    inputs.push(notice(
        "jopedime",
        "LICENSE",
        "repository:LICENSE",
        "Apache-2.0",
        license,
    ));
    let notices = std::fs::read(request.notices)
        .with_context(|| format!("read notices {}", request.notices.display()))?;
    let parsed: serde_json::Value = serde_json::from_slice(&notices).context("parse notices")?;
    ensure!(
        parsed["schema"] == crate::notices::SCHEMA,
        "{} is not {}",
        request.notices.display(),
        crate::notices::SCHEMA
    );
    inputs.push(notice(
        "third-party",
        "THIRD_PARTY.json",
        "generated:cargo-metadata",
        "NOASSERTION",
        notices,
    ));
    let key = signing_key(profile.scope(), request.release_key)?;
    let options = StageOptions {
        product: crate::PRODUCT.to_owned(),
        product_version: env!("CARGO_PKG_VERSION").to_owned(),
        target: request.target.to_owned(),
        profile: profile.name.clone(),
        scope: profile.scope(),
        source: SourceIdentity {
            sha40: request.source_sha.to_ascii_lowercase(),
            cargo_lock_sha256: jope_assets::sha256_hex(&cargo_lock),
            profile_sha256: profile_sha,
            models_lock_sha256: lock_sha,
        },
        methods: locked.methods.clone(),
        key_id: key.id,
        frame_bytes: profile.codec.frame_mib << 20,
        level: profile.codec.level,
    };
    let staged = writer::stage(&code, inputs, &options).context("encode assets")?;
    Ok((staged, code, key, profile))
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
    let (staged, code, key, profile) = stage(request)?;
    match output {
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
mod executable_target_tests {
    use super::validate_executable_target;
    use std::path::Path;

    const LINUX_X86_64: &str = "x86_64-unknown-linux-gnu";
    const MACOS_ARM64: &str = "aarch64-apple-darwin";

    fn arm64_mach_o() -> Vec<u8> {
        let mut code = vec![0; 32];
        code[0..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]); // MH_MAGIC_64, little-endian
        code[4..8].copy_from_slice(&0x0100000cu32.to_le_bytes()); // CPU_TYPE_ARM64
        code[12..16].copy_from_slice(&2u32.to_le_bytes()); // MH_EXECUTE
        code
    }

    #[test]
    fn linux_target_accepts_elf_and_rejects_macho_and_pe() {
        assert!(
            validate_executable_target(b"\x7fELF fixture", LINUX_X86_64, Path::new("app")).is_ok()
        );
        for bytes in [arm64_mach_o(), b"MZ windows PE".to_vec()] {
            let error = validate_executable_target(&bytes, LINUX_X86_64, Path::new("app"))
                .unwrap_err()
                .to_string();
            assert!(
                error.contains("not an ELF file for Linux target"),
                "{error}"
            );
        }
    }

    #[test]
    fn macos_arm64_target_accepts_only_arm64_64_bit_mach_o_executables() {
        let mach_o = arm64_mach_o();
        assert!(validate_executable_target(&mach_o, MACOS_ARM64, Path::new("app")).is_ok());
        assert!(
            validate_executable_target(b"\x7fELF fixture", MACOS_ARM64, Path::new("app"))
                .unwrap_err()
                .to_string()
                .contains("not a 64-bit arm64 Mach-O executable")
        );

        let mut wrong_cpu = arm64_mach_o();
        wrong_cpu[4..8].copy_from_slice(&0x01000007u32.to_le_bytes()); // CPU_TYPE_X86_64
        assert!(validate_executable_target(&wrong_cpu, MACOS_ARM64, Path::new("app")).is_err());
        let mut dylib = arm64_mach_o();
        dylib[12..16].copy_from_slice(&6u32.to_le_bytes()); // MH_DYLIB, not MH_EXECUTE
        assert!(validate_executable_target(&dylib, MACOS_ARM64, Path::new("app")).is_err());
    }

    #[test]
    fn unsupported_targets_are_rejected_even_when_the_binary_is_elf() {
        let error = validate_executable_target(
            b"\x7fELF fixture",
            "x86_64-pc-windows-gnu",
            Path::new("app"),
        )
        .unwrap_err()
        .to_string();
        assert!(
            error.contains("unsupported appliance executable target"),
            "{error}"
        );
    }
}
