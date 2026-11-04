use super::*;

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
    assert!(full
        .methods
        .iter()
        .any(|id| id == jope_service::proposal::METHOD));
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

    let fused = &lock.profiles["fused-cpu"];
    assert_eq!(fused.status, "ready");
    assert_eq!(fused.scope, "internal");
    assert_eq!(fused.methods, full.methods);
    assert_eq!(fused.components.len(), 1);
    let mut actual = fused
        .components
        .iter()
        .flat_map(|component| {
            component.files.iter().map(move |file| {
                (
                    format!("{}/{}/{}", component.role, component.id, file.path),
                    file.sha256.clone(),
                    file.bytes,
                )
            })
        })
        .collect::<Vec<_>>();
    let mut required = jope_service::proposal::required_assets().unwrap();
    actual.sort();
    required.sort();
    assert_eq!(
        actual, required,
        "fused must carry the exact complete compiled runtime closure"
    );
    assert!(fused.total_bytes <= 16 << 20);
    let fixture_fused = &lock.profiles["fixture-fused-cpu"];
    assert_eq!(fixture_fused.scope, "development");
    assert!(!fixture_fused
        .methods
        .iter()
        .any(|id| id == jope_service::proposal::METHOD));
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
    assert_eq!(tensors, 328235504);
}
