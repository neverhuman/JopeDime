//! The committed lock is the derivation of the committed profiles (SPECIFICATION.md §5.3–5.4):
//! rights the scope does not admit, missing acknowledgements, unlicensed weights and promised
//! methods this build cannot run block a profile; structural mistakes are errors.
use jope_dist::profile::{derive_lock, lock_profile, Profile};
use std::path::{Path, PathBuf};

fn repo() -> PathBuf {
    jope_dist::repo_root()
}

#[test]
fn the_committed_lock_is_current() {
    let repo = repo();
    let lock = derive_lock(&repo.join("deploy/appliance/profiles"), &repo).unwrap();
    let committed =
        std::fs::read_to_string(repo.join("deploy/appliance/models.lock.json")).unwrap();
    assert_eq!(
        committed,
        lock.to_text(),
        "run `jope-dist lock` and review the diff"
    );
    assert_eq!(lock.profiles["fixture"].status, "ready");
    assert_eq!(lock.profiles["native-only"].status, "ready");
    assert_eq!(lock.profiles["native-only"].total_bytes, 0);
    let public = &lock.profiles["native-only"];
    assert!(public.components.iter().all(|component| {
        component.files.iter().all(|file| {
            !file.path.contains("starlight_core_regressor")
                && !file.path.contains("starforge/starlight")
        })
    }));
    let public_profile = Profile::load(&repo.join("deploy/appliance/profiles/native-only.toml"))
        .unwrap()
        .0;
    assert_eq!(public_profile.scope, "public");
    assert!(public_profile.owner_ack.is_none());
    assert!(public_profile.components.iter().all(|component| {
        component.files.iter().all(|file| {
            !file.contains("starlight_core_regressor") && !file.contains("starforge/starlight")
        })
    }));
    let full = &lock.profiles["full"];
    assert_eq!(full.status, "blocked");
    assert!(!full.blockers.iter().any(|b| {
        b.contains("starlight_core_regressor") && b.contains("owner acknowledgement")
    }));
    assert!(full
        .blockers
        .iter()
        .any(|b| b.contains("proposal.step60000.regression")));
    let acknowledgement = std::fs::read_to_string(
        repo.join("deploy/appliance/acknowledgements/full-internal-restricted.md"),
    )
    .unwrap();
    assert!(acknowledgement.contains("REL-07 T0115 D3"));
    assert!(acknowledgement.contains("Internal qualification and rehearsal candidates only"));
    assert!(acknowledgement.contains("Public distribution | Excluded"));
    let r882 = full
        .components
        .iter()
        .find(|component| component.id == "r882-regression-adapter")
        .expect("the full profile resolves the owner-selected r882 adapter");
    assert_eq!(r882.bundle_id, "r882-regression");
    assert_eq!(r882.files.len(), 1);
    assert_eq!(r882.files[0].path, "jope.safetensors");
    assert_eq!(
        r882.files[0].sha256,
        "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c"
    );
    assert_eq!(r882.files[0].bytes, 7_499_888);
    let r882_manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo.join("models/bundles/r882-regression.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        r882_manifest["export"]["bundle_digest"],
        "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
    );
    assert!(!full
        .components
        .iter()
        .any(|component| component.bundle_id == "policy-step-60000"));

    // Three encoder tensors plus the pinned r882 safetensors adapter remain in the closure.
    let tensors: u64 = full
        .components
        .iter()
        .flat_map(|c| &c.files)
        .filter(|f| f.path.ends_with(".safetensors"))
        .map(|f| f.bytes)
        .sum();
    assert_eq!(tensors, 325_949_288);
}

fn write_profile(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, body).unwrap();
    path
}

const HEADER: &str = r#"schema = "jope.appliance-profile/v1"
description = "test"
methods = ["native"]
bundles = "models/bundles"
max_payload_mib = 512
[codec]
name = "zstd"
level = 3
frame_mib = 16
"#;

#[test]
fn rights_outside_the_scope_block_and_never_pass() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_profile(
        dir.path(),
        "internal",
        &format!(
            "name = \"internal\"\nscope = \"internal\"\nredistribution = [\"permitted\", \"internal\"]\n{HEADER}\
             [[component]]\nid = \"policy\"\nbundle = \"policy-step-60000\"\nfiles = [\"policy.safetensors\"]\nrole = \"model\"\n\
             [[component]]\nid = \"tabicl\"\nbundle = \"jope-prime-gen31-v2\"\nfiles = [\"foundation/tabicl-regressor-v2-chimera-shuffled.safetensors\"]\nrole = \"model\"\n"
        ),
    );
    let (profile, sha) = Profile::load(&path).unwrap();
    let locked = lock_profile(&profile, &sha, &repo()).unwrap();
    assert_eq!(locked.status, "blocked");
    // The internal policy is admitted; the shuffled TabICL weights carry no licence record.
    assert_eq!(locked.blockers.len(), 1, "{:?}", locked.blockers);
    assert!(locked.blockers[0].contains("no licence record"));

    let public = write_profile(
        dir.path(),
        "public",
        &format!(
            "name = \"public\"\nscope = \"public\"\nredistribution = [\"permitted\"]\n{HEADER}\
             [[component]]\nid = \"policy\"\nbundle = \"policy-step-60000\"\nfiles = [\"policy.safetensors\"]\nrole = \"model\"\n"
        ),
    );
    let (profile, sha) = Profile::load(&public).unwrap();
    let locked = lock_profile(&profile, &sha, &repo()).unwrap();
    assert!(locked
        .blockers
        .iter()
        .any(|b| b.contains("`internal` is not admitted by scope public")));

    // A public profile may not even claim to admit internal files.
    let wide = write_profile(
        dir.path(),
        "wide",
        &format!("name = \"wide\"\nscope = \"public\"\nredistribution = [\"permitted\", \"internal\"]\n{HEADER}"),
    );
    assert!(Profile::load(&wide).is_err());
}

#[test]
fn structural_mistakes_are_errors() {
    let dir = tempfile::tempdir().unwrap();
    let cases = [
        (
            "name = \"other\"\nscope = \"public\"\nredistribution = [\"permitted\"]\n",
            "mismatched",
        ),
        (
            "name = \"mismatched\"\nscope = \"everyone\"\nredistribution = [\"permitted\"]\n",
            "mismatched",
        ),
        (
            "name = \"mismatched\"\nscope = \"public\"\nredistribution = [\"free\"]\n",
            "mismatched",
        ),
    ];
    for (head, file) in cases {
        let path = write_profile(dir.path(), file, &format!("{head}{HEADER}"));
        assert!(Profile::load(&path).is_err(), "{head}");
    }
    let missing = write_profile(
        dir.path(),
        "missing",
        &format!(
            "name = \"missing\"\nscope = \"internal\"\nredistribution = [\"internal\"]\n{HEADER}\
             [[component]]\nid = \"policy\"\nbundle = \"policy-step-60000\"\nfiles = [\"not-a-file.safetensors\"]\nrole = \"model\"\n"
        ),
    );
    let (profile, sha) = Profile::load(&missing).unwrap();
    assert!(lock_profile(&profile, &sha, &repo()).is_err());
    let dangling = write_profile(
        dir.path(),
        "dangling",
        &format!(
            "name = \"dangling\"\nscope = \"internal\"\nredistribution = [\"internal\"]\n{HEADER}\
             [[component]]\nid = \"policy\"\nbundle = \"policy-step-60000\"\nfiles = [\"policy.safetensors\"]\nrole = \"model\"\nrequires = [\"encoder\"]\n"
        ),
    );
    assert!(Profile::load(&dangling).is_err());
}

#[test]
fn limits_toml_matches_the_compiled_bounds() {
    let text = std::fs::read_to_string(repo().join("deploy/appliance/limits.toml")).unwrap();
    let doc: toml::Value = toml::from_str(&text).unwrap();
    let p = &doc["package"];
    let i = &doc["input"];
    use jope_assets::limits as l;
    let int = |v: &toml::Value| v.as_integer().unwrap() as u64;
    assert_eq!(int(&p["max_manifest_bytes"]), l::MAX_MANIFEST_BYTES);
    assert_eq!(int(&p["max_entries"]), l::MAX_ENTRIES as u64);
    assert_eq!(int(&p["max_methods"]), l::MAX_METHODS as u64);
    assert_eq!(int(&p["max_entry_bytes"]), l::MAX_ENTRY_BYTES);
    assert_eq!(int(&p["max_frame_raw_bytes"]), l::MAX_FRAME_RAW_BYTES);
    assert_eq!(int(&p["max_zstd_window_bytes"]), l::MAX_ZSTD_WINDOW);
    assert_eq!(int(&p["max_name_bytes"]), l::MAX_NAME_BYTES as u64);
    let r = jope_runtime::limits::Limits::default();
    assert_eq!(int(&i["max_input_bytes"]), r.max_input_bytes);
    assert_eq!(int(&i["max_rows"]), r.max_rows as u64);
    assert_eq!(int(&i["max_columns"]), r.max_columns as u64);
    assert_eq!(int(&i["max_classes"]), r.max_classes as u64);
    assert_eq!(int(&i["max_model_bytes"]), r.max_model_bytes);
}

#[test]
fn notices_include_dependencies_enabled_only_by_the_selected_feature() {
    let dir = tempfile::tempdir().unwrap();
    let app = dir.path().join("app");
    let optional = dir.path().join("optional");
    for path in [&app, &optional] {
        std::fs::create_dir_all(path.join("src")).unwrap();
    }
    std::fs::write(app.join("src/lib.rs"), "").unwrap();
    std::fs::write(optional.join("src/lib.rs"), "").unwrap();
    std::fs::write(
        app.join("Cargo.toml"),
        r#"[package]
name = "notice-app"
version = "0.1.0"
edition = "2021"
license = "MIT"
[features]
extra = ["dep:notice-optional"]
[dependencies]
notice-optional = { path = "../optional", optional = true }
"#,
    )
    .unwrap();
    std::fs::write(
        optional.join("Cargo.toml"),
        r#"[package]
name = "notice-optional"
version = "0.1.0"
edition = "2021"
license = "MIT"
"#,
    )
    .unwrap();
    std::fs::write(optional.join("LICENSE"), "MIT License (test fixture)\n").unwrap();
    let manifest = app.join("Cargo.toml");
    assert!(
        std::process::Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
            .args(["generate-lockfile", "--offline", "--manifest-path"])
            .arg(&manifest)
            .status()
            .unwrap()
            .success()
    );
    let collect = |features| {
        jope_dist::notices::collect(
            &manifest,
            "notice-app",
            "x86_64-unknown-linux-gnu",
            features,
        )
        .unwrap()
    };
    let without = collect(&[]);
    assert!(!without.crates.iter().any(|c| c.name == "notice-optional"));
    let with = collect(&["extra".into()]);
    let dependency = with
        .crates
        .iter()
        .find(|c| c.name == "notice-optional")
        .unwrap();
    assert_eq!(dependency.license, "MIT");
    assert!(with.texts.contains_key(&dependency.files["LICENSE"]));
}

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
        error.to_string().contains("hash") || error.to_string().contains("size"),
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
