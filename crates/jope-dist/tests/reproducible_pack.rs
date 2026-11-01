//! Packing is deterministic and path-independent, a release stage seals only with a signature
//! over its own signing message, and a packed file is never repacked (SPECIFICATION.md §10.5,
//! §16.2). The release key is played by the development key's PEM passed as `--release-key`.
use jope_dist::pack::{pack, seal, Output, PackRequest};
use jope_dist::{fixture, report};
use std::path::{Path, PathBuf};

const SHA: &str = "0123456789abcdef0123456789abcdef01234567";

fn repo() -> PathBuf {
    jope_dist::repo_root()
}

fn code(dir: &Path) -> PathBuf {
    let mut bytes = vec![0u8; 64];
    bytes[0..4].copy_from_slice(b"\x7fELF");
    bytes[4] = 2; // ELFCLASS64
    bytes[5] = 1; // ELFDATA2LSB
    bytes[6] = 1; // EV_CURRENT
    bytes[18..20].copy_from_slice(&62u16.to_le_bytes()); // EM_X86_64
    bytes.extend((0..20_000u32).map(|i| (i * 7 % 251) as u8));
    let path = dir.join("jopedime.code");
    std::fs::write(&path, bytes).unwrap();
    path
}

fn notices(dir: &Path) -> PathBuf {
    let path = dir.join("notices.json");
    std::fs::write(
        &path,
        format!(
            r#"{{"schema":"{}","crates":[],"texts":{{}}}}"#,
            jope_dist::notices::SCHEMA
        ),
    )
    .unwrap();
    path
}

fn models(dir: &Path) -> PathBuf {
    let models = dir.join("models");
    fixture::write(&models, &repo().join("deploy/appliance/fixtures/bundles")).unwrap();
    models
}

fn request<'a>(
    repo: &'a Path,
    profile: &'a Path,
    lock: &'a Path,
    code: &'a Path,
    models: &'a Path,
    notices: &'a Path,
    key: Option<&'a Path>,
) -> PackRequest<'a> {
    PackRequest {
        profile,
        lock,
        repo,
        code,
        models,
        notices,
        source_sha: SHA,
        target: "x86_64-unknown-linux-gnu",
        release_key: key,
    }
}

#[test]
fn the_fixture_packs_identically_from_two_directories() {
    let repo = repo();
    let profile = repo.join("deploy/appliance/profiles/fixture.toml");
    let lock = repo.join("deploy/appliance/models.lock.json");
    let mut files = Vec::new();
    for _ in 0..2 {
        let dir = tempfile::tempdir().unwrap();
        let (code, models, notices) = (code(dir.path()), models(dir.path()), notices(dir.path()));
        let out = dir.path().join("jopedime");
        let receipt = dir.path().join("receipt.json");
        pack(
            &request(&repo, &profile, &lock, &code, &models, &notices, None),
            Output::Development { out: &out },
            Some(&receipt),
        )
        .unwrap();
        let verified =
            report::verify_file(&out, &report::trusted(&[], true, false).unwrap()).unwrap();
        assert_eq!(verified["profile"], "fixture");
        assert_eq!(verified["report"]["entries"], 5);
        let receipt: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&receipt).unwrap()).unwrap();
        assert_eq!(receipt["source_sha40"], SHA);
        // A packed file is not accepted as an executable to pack.
        assert!(pack(
            &request(&repo, &profile, &lock, &out, &models, &notices, None),
            Output::Development {
                out: &dir.path().join("twice")
            },
            None
        )
        .is_err());
        // A changed model file is refused against the lock.
        std::fs::write(
            models.join("bundles/fixture-policy/policy.safetensors"),
            b"other bytes",
        )
        .unwrap();
        let error = pack(
            &request(&repo, &profile, &lock, &code, &models, &notices, None),
            Output::Development {
                out: &dir.path().join("x"),
            },
            None,
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("does not match the lock"),
            "{error:#}"
        );
        files.push(std::fs::read(&out).unwrap());
        let (size, ok) =
            report::size_report(&out, &report::trusted(&[], true, false).unwrap(), None).unwrap();
        assert!(ok);
        assert_eq!(
            size["final_executable_bytes"],
            files.last().unwrap().len() as u64
        );
        let sbom = report::sbom_models(&out, &report::trusted(&[], true, false).unwrap()).unwrap();
        assert_eq!(sbom["components"].as_array().unwrap().len(), 5);
    }
    assert_eq!(files[0], files[1]);
}

/// A release scope: stage, sign the signing message elsewhere, seal, verify with that key only.
#[test]
fn a_release_stage_seals_only_with_its_own_signature() {
    use jope_assets::verify::{development_key, sign_with_development_key};
    let repo = repo();
    let dir = tempfile::tempdir().unwrap();
    let pem = dir.path().join("release.pub");
    std::fs::write(&pem, development_key().to_pem()).unwrap();
    let profile = repo.join("deploy/appliance/profiles/native-only.toml");
    let lock = repo.join("deploy/appliance/models.lock.json");
    let (code, models, notices) = (code(dir.path()), models(dir.path()), notices(dir.path()));
    let stage = dir.path().join("stage");
    let req = request(&repo, &profile, &lock, &code, &models, &notices, Some(&pem));
    // A release scope is never signed in-process.
    assert!(pack(
        &req,
        Output::Development {
            out: &dir.path().join("x")
        },
        None
    )
    .is_err());
    pack(&req, Output::Stage { dir: &stage }, None).unwrap();
    let message = std::fs::read(stage.join("signing-message.bin")).unwrap();
    assert!(message.starts_with(jope_assets::SIGNING_DOMAIN));
    // cosign writes base64 of the DER signature.
    let signature = dir.path().join("manifest.sig");
    let der = sign_with_development_key(&message);
    std::fs::write(&signature, jope_dist::signature::base64_encode(&der)).unwrap();
    let out = dir.path().join("jopedime");
    seal(&code, &stage, &signature, Some(&pem), &out, None).unwrap();
    let keys = report::trusted(std::slice::from_ref(&pem), false, false).unwrap();
    let verified = report::verify_file(&out, &keys).unwrap();
    assert_eq!(verified["scope"], "public");
    // Not trusted by the compiled-in release key alone.
    assert!(report::verify_file(&out, &report::trusted(&[], false, true).unwrap()).is_err());
    // A signature over other bytes is refused before anything is written.
    let wrong = dir.path().join("wrong.sig");
    std::fs::write(
        &wrong,
        jope_dist::signature::base64_encode(&sign_with_development_key(b"other")),
    )
    .unwrap();
    assert!(seal(
        &code,
        &stage,
        &wrong,
        Some(&pem),
        &dir.path().join("y"),
        None
    )
    .is_err());
    assert!(!dir.path().join("y").exists());
    // Another executable than the stage names is refused.
    let other = dir.path().join("other.code");
    std::fs::write(&other, b"\x7fELF other").unwrap();
    assert!(seal(
        &other,
        &stage,
        &signature,
        Some(&pem),
        &dir.path().join("z"),
        None
    )
    .is_err());
}

#[test]
fn a_blocked_profile_is_never_packed() {
    let repo = repo();
    let dir = tempfile::tempdir().unwrap();
    let profile = repo.join("deploy/appliance/profiles/full.toml");
    let lock = repo.join("deploy/appliance/models.lock.json");
    let (code, models, notices) = (code(dir.path()), models(dir.path()), notices(dir.path()));
    let error = pack(
        &request(&repo, &profile, &lock, &code, &models, &notices, None),
        Output::Stage {
            dir: &dir.path().join("s"),
        },
        None,
    )
    .unwrap_err();
    assert!(format!("{error:#}").contains("blocked"), "{error:#}");
}
