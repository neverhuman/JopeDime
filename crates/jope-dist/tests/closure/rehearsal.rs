use super::*;

struct RehearsalFixture {
    root: PathBuf,
    profile: PathBuf,
    lock: PathBuf,
    code: PathBuf,
    models: PathBuf,
    notices: PathBuf,
}

impl RehearsalFixture {
    fn new(root: &Path, scope: &str, rights: &str) -> Self {
        let profiles = root.join("deploy/appliance/profiles");
        let bundles = root.join("models/bundles");
        let models = root.join("cache");
        for dir in [&profiles, &bundles, &models.join("bundles/synthetic")] {
            std::fs::create_dir_all(dir).unwrap();
        }
        let bytes = b"synthetic policy bytes, not trained model weights";
        std::fs::write(models.join("bundles/synthetic/weights.bin"), bytes).unwrap();
        let bundle = serde_json::json!({
            "schema": "jope.model-bundle/v2", "bundle_id": "synthetic",
            "files": [{"path": "weights.bin", "sha256": jope_assets::sha256_hex(bytes),
                "bytes": bytes.len(), "kind": "policy",
                "license": {"spdx_expression": "LicenseRef-Internal", "redistribution": rights}}]
        });
        std::fs::write(bundles.join("synthetic.json"), bundle.to_string()).unwrap();
        std::fs::write(root.join("ACK.md"), "Internal rehearsal only").unwrap();
        let classes = if scope == "public" {
            "[\"permitted\"]"
        } else {
            "[\"permitted\",\"internal\",\"prohibited\"]"
        };
        let profile = write_profile(
            &profiles,
            "full",
            &format!(
                r#"
schema = "jope.appliance-profile/v1"
name = "full"
description = "synthetic non-signing rehearsal control"
scope = "{scope}"
redistribution = {classes}
owner_ack = "ACK.md"
methods = ["proposal.step60000.regression"]
bundles = "models/bundles"
max_payload_mib = 1
[codec]
name = "zstd"
level = 1
frame_mib = 1
[[component]]
id = "synthetic-adapter"
bundle = "synthetic"
files = ["weights.bin"]
role = "model"
"#
            ),
        );
        let lock = root.join("models.lock.json");
        std::fs::write(&lock, derive_lock(&profiles, root).unwrap().to_text()).unwrap();
        std::fs::write(root.join("Cargo.lock"), "version = 4\n").unwrap();
        std::fs::write(root.join("LICENSE"), "synthetic control notice").unwrap();
        let mut bytes = vec![0u8; 64];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4..7].copy_from_slice(&[2, 1, 1]);
        bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
        bytes[18..20].copy_from_slice(&62u16.to_le_bytes());
        bytes[20..24].copy_from_slice(&1u32.to_le_bytes());
        bytes[52..54].copy_from_slice(&64u16.to_le_bytes());
        let code = root.join("code");
        std::fs::write(&code, bytes).unwrap();
        let notices = root.join("notices.json");
        std::fs::write(
            &notices,
            serde_json::json!({
                "schema": jope_dist::notices::SCHEMA, "crates": [], "texts": {}
            })
            .to_string(),
        )
        .unwrap();
        Self {
            root: root.to_owned(),
            profile,
            lock,
            code,
            models,
            notices,
        }
    }

    fn request(&self) -> jope_dist::pack::PackRequest<'_> {
        jope_dist::pack::PackRequest {
            repo: &self.root,
            profile: &self.profile,
            lock: &self.lock,
            code: &self.code,
            models: &self.models,
            notices: &self.notices,
            source_sha: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            target: "x86_64-unknown-linux-gnu",
            release_key: None,
        }
    }
}

#[test]
fn full_rehearsal_records_missing_methods_and_cannot_be_sealed() {
    use jope_dist::pack::{pack, seal, Output};
    let dir = tempfile::tempdir().unwrap();
    let fixture = RehearsalFixture::new(dir.path(), "internal-restricted", "internal");
    let release = dir.path().join("release-stage");
    let error = pack(&fixture.request(), Output::Stage { dir: &release }, None).unwrap_err();
    assert!(error
        .to_string()
        .contains("neural adapters are not compiled"));
    assert!(!release.exists());
    let rehearsal = dir.path().join("rehearsal");
    pack(
        &fixture.request(),
        Output::Rehearsal { dir: &rehearsal },
        None,
    )
    .unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(rehearsal.join("manifest.json")).unwrap()).unwrap();
    assert!(manifest["entries"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["origin"] == "synthetic/weights.bin"));
    let record: serde_json::Value =
        serde_json::from_slice(&std::fs::read(rehearsal.join("rehearsal.json")).unwrap()).unwrap();
    assert_eq!(record["schema"], "jope.appliance-rehearsal/v1");
    assert_eq!(record["signable"], false);
    assert_eq!(record["release_qualified"], false);
    assert_eq!(
        record["unexecutable_methods"][0]["id"],
        "proposal.step60000.regression"
    );
    for name in ["footer.bin", "signing-message.bin", "stage.json"] {
        assert!(!rehearsal.join(name).exists());
    }
    let output = dir.path().join("sealed");
    let error = seal(
        &fixture.code,
        &rehearsal,
        &dir.path().join("no-signature"),
        None,
        &output,
        None,
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("rehearsal output cannot be sealed"));
    assert!(!output.exists());
}

#[test]
fn full_rehearsal_never_waives_scope_rights_or_asset_identity() {
    use jope_dist::pack::{pack, Output};
    for (scope, rights, expected) in [
        (
            "public",
            "permitted",
            "limited to the internal-restricted full profile",
        ),
        (
            "internal-restricted",
            "conditional",
            "cannot waive asset, rights or payload blockers",
        ),
    ] {
        let dir = tempfile::tempdir().unwrap();
        let fixture = RehearsalFixture::new(dir.path(), scope, rights);
        let out = dir.path().join("rehearsal");
        let error = pack(&fixture.request(), Output::Rehearsal { dir: &out }, None).unwrap_err();
        assert!(error.to_string().contains(expected), "{error:#}");
        assert!(!out.exists());
    }
    let dir = tempfile::tempdir().unwrap();
    let fixture = RehearsalFixture::new(dir.path(), "internal-restricted", "internal");
    std::fs::write(
        fixture.models.join("bundles/synthetic/weights.bin"),
        "changed",
    )
    .unwrap();
    let out = dir.path().join("rehearsal");
    let error = pack(&fixture.request(), Output::Rehearsal { dir: &out }, None).unwrap_err();
    assert!(
        error.to_string().contains("does not match the lock"),
        "{error:#}"
    );
    assert!(!out.exists());
}

#[test]
fn full_rehearsal_requires_the_owner_acknowledgement() {
    use jope_dist::pack::{pack, Output};
    let dir = tempfile::tempdir().unwrap();
    let fixture = RehearsalFixture::new(dir.path(), "internal-restricted", "internal");
    std::fs::remove_file(dir.path().join("ACK.md")).unwrap();
    let out = dir.path().join("rehearsal");
    let error = pack(&fixture.request(), Output::Rehearsal { dir: &out }, None).unwrap_err();
    assert!(error.to_string().contains("explicit owner acknowledgement"));
    assert!(!out.exists());
}
