//! The committed lock is the derivation of the committed profiles (SPECIFICATION.md §5.3–5.4):
//! rights the scope does not admit, missing acknowledgements, unlicensed weights and promised
//! methods this build cannot run block a profile; structural mistakes are errors.
use jope_dist::profile::{Profile, derive_lock, lock_profile};
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
    assert_eq!(full.status, "ready");
    assert!(full.blockers.is_empty());
    assert!(
        full.methods
            .iter()
            .any(|id| id == jope_service::proposal::METHOD)
    );
    assert!(!full.blockers.iter().any(|b| {
        b.contains("starlight_core_regressor") && b.contains("owner acknowledgement")
    }));

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
        .find(|component| component.id == "r882")
        .expect("the full profile resolves the owner-selected r882 adapter");
    assert_eq!(r882.bundle_id, "r882");
    assert_eq!(r882.files.len(), 56);
    let jope = r882
        .files
        .iter()
        .find(|file| file.path == "jope.safetensors")
        .unwrap();
    assert_eq!(
        jope.sha256,
        "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c"
    );
    assert_eq!(jope.bytes, 7_499_888);
    let r882_manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(repo.join("models/bundles/r882-regression.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(
        r882_manifest["export"]["bundle_digest"],
        "sha256:057e50a9a741d30d541ed8af8339cec84a5157af621100cb70857052ef14dacc"
    );
    assert!(
        !full
            .components
            .iter()
            .any(|component| component.bundle_id == "policy-step-60000")
    );

    // Three encoder tensors plus the pinned r882 safetensors adapter remain in the closure.
    let tensors: u64 = full
        .components
        .iter()
        .flat_map(|c| &c.files)
        .filter(|f| f.path.ends_with(".safetensors"))
        .map(|f| f.bytes)
        .sum();
    assert_eq!(tensors, 328235504);
}

fn write_profile(dir: &Path, name: &str, body: &str) -> PathBuf {
    let path = dir.join(format!("{name}.toml"));
    std::fs::write(&path, body).unwrap();
    path
}

#[test]
fn r882_promises_require_the_complete_exact_internal_closure() {
    let dir = tempfile::tempdir().unwrap();
    let assets = jope_service::proposal::required_assets().unwrap();
    let files = assets
        .iter()
        .map(|(id, _, _)| id.strip_prefix("model/r882/").unwrap())
        .collect::<Vec<_>>();
    let text = format!(
        r#"schema = "jope.appliance-profile/v1"
name = "r882"
description = "test"
scope = "internal"
redistribution = ["internal"]
methods = ["proposal.r882.regression"]
bundles = "models/bundles"
max_payload_mib = 512
[codec]
name = "zstd"
level = 3
frame_mib = 16
[[component]]
id = "r882"
bundle = "r882"
role = "model"
files = {}
"#,
        serde_json::to_string(&files).unwrap()
    );
    let path = write_profile(dir.path(), "r882", &text);
    let (profile, sha) = Profile::load(&path).unwrap();
    assert_eq!(
        lock_profile(&profile, &sha, &repo()).unwrap().status,
        "ready"
    );
    let mut missing = profile.clone();
    missing.components[0]
        .files
        .retain(|file| file != "dope.safetensors");
    let blocked = lock_profile(&missing, &sha, &repo()).unwrap();
    assert!(
        blocked
            .blockers
            .iter()
            .any(|why| why.contains("model/r882/dope.safetensors"))
    );
    let mut alias = profile.clone();
    alias.components[0].id = "wrong-logical-id".into();
    assert_eq!(
        lock_profile(&alias, &sha, &repo()).unwrap().status,
        "blocked"
    );
    let mut public = profile.clone();
    public.scope = "public".into();
    assert!(
        lock_profile(&public, &sha, &repo())
            .unwrap()
            .blockers
            .iter()
            .any(|why| why.contains("internal-only"))
    );
    // A descriptor with a different hash must not become a runnable method merely
    // because its file names and loader architecture are recognized.
    let mut descriptor: serde_json::Value =
        serde_json::from_slice(&std::fs::read(repo().join("models/bundles/r882.json")).unwrap())
            .unwrap();
    descriptor["files"][0]["sha256"] = serde_json::json!("0".repeat(64));
    std::fs::write(
        dir.path().join("r882.json"),
        serde_json::to_vec(&descriptor).unwrap(),
    )
    .unwrap();
    let mut drift = profile;
    drift.bundles = ".".into();
    assert!(
        lock_profile(&drift, &sha, dir.path())
            .unwrap()
            .blockers
            .iter()
            .any(|why| why.contains("missing or different asset"))
    );
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
    assert!(
        locked
            .blockers
            .iter()
            .any(|b| b.contains("`internal` is not admitted by scope public"))
    );

    // A public profile may not even claim to admit internal files.
    let wide = write_profile(
        dir.path(),
        "wide",
        &format!(
            "name = \"wide\"\nscope = \"public\"\nredistribution = [\"permitted\", \"internal\"]\n{HEADER}"
        ),
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
