use super::loading::{open_bundle_root, read_manifest, verify_notices};
use super::*;

fn shipped_manifest() -> ModelBundleV1 {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/model-bundle.v1.json");
    serde_json::from_str(&std::fs::read_to_string(path).expect("read shipped manifest"))
        .expect("decode shipped manifest")
}

#[test]
fn rejects_absolute_and_unsupported_paths() {
    let mut manifest = shipped_manifest();
    manifest.files[0].path = "/tmp/model.pt".into();
    assert!(manifest.validate_metadata().is_err());
    manifest.files[0].path = "../model.safetensors".into();
    assert!(manifest.validate_metadata().is_err());
    manifest.files[0].path = "model.pt".into();
    assert!(manifest.validate_metadata().is_err());
    manifest.files[0].path = "model.json".into();
    manifest.files[0].shape.clear();
    manifest.files[0].tensor_count = 0;
    manifest.files[0].tensor_inventory_sha256 = None;
    assert!(manifest.validate_metadata().is_ok());
}

#[test]
fn rejects_duplicate_paths_and_inconsistent_license_policy() {
    let mut duplicate = shipped_manifest();
    duplicate.files[1].path = duplicate.files[0].path.clone();
    assert!(duplicate.validate_metadata().is_err());

    let mut open_only = shipped_manifest();
    open_only.files[0].license.policy_eligibility.open_only = true;
    assert!(open_only.validate_metadata().is_err());

    let mut wrong_identity = shipped_manifest();
    wrong_identity.files[0].license.sha256 = "0".repeat(64);
    assert!(wrong_identity.validate_metadata().is_err());

    let mut missing_notice = shipped_manifest();
    missing_notice.files[0].license.notice_paths.clear();
    assert!(missing_notice.validate_metadata().is_err());

    let mut unbound_notice = shipped_manifest();
    let notice = unbound_notice.files[0].license.notice_paths[0].clone();
    unbound_notice.notices.remove(&notice);
    assert!(unbound_notice.validate_metadata().is_err());

    let mut wrong_checkpoint = shipped_manifest();
    wrong_checkpoint.source_checkpoint_sha256 = "0".repeat(64);
    assert!(wrong_checkpoint.validate_metadata().is_err());

    let mut registry_masquerade = shipped_manifest();
    let registry = registry_masquerade
        .files
        .iter_mut()
        .find(|file| file.kind == "jope_operator_registry")
        .unwrap();
    registry.license.component_kind = "model".into();
    assert!(registry_masquerade.validate_metadata().is_err());

    let mut foundation_license = shipped_manifest();
    foundation_license
        .files
        .iter_mut()
        .find(|file| file.path == FOUNDATION_ARTIFACT)
        .unwrap()
        .license
        .spdx_expression = "Apache-2.0".into();
    assert!(foundation_license.validate_metadata().is_err());
}

#[test]
fn canonical_policy_identities_are_reproducible() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
    let metadata: JopePolicyMetadata =
        serde_json::from_slice(&std::fs::read(root.join(JOPE_POLICY_METADATA)).unwrap()).unwrap();
    let vocab = sha256_bytes(&canonical_json_bytes(&metadata.vocab).unwrap());
    let grammar = sha256_bytes(
        &canonical_json_bytes(&serde_json::Value::Array(metadata.grammar_specs)).unwrap(),
    );
    assert_eq!(
        vocab, JOPE_VOCABULARY_CANONICAL_SHA256,
        "actual vocab={vocab}"
    );
    assert_eq!(grammar, JOPE_GRAMMAR_SHA256, "actual grammar={grammar}");
}

#[test]
fn verified_reads_reject_bytes_changed_after_initial_verification() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let declared = shipped_manifest()
        .files
        .into_iter()
        .find(|file| file.path == JOPE_POLICY_METADATA)
        .unwrap();
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../")
        .join(JOPE_POLICY_METADATA);
    let bytes = std::fs::read(source).unwrap();
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("starforge-bundle-reverify-{nonce}"));
    let path = root.join(JOPE_POLICY_METADATA);
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let (canonical_root, root_dir) = open_bundle_root(&root).unwrap();
    let first =
        loading::read_regular_bytes(&root_dir, JOPE_POLICY_METADATA, declared.bytes).unwrap();
    assert!(verify_file_bytes(&declared, &first).is_ok());
    std::fs::write(&path, b"{}").unwrap();
    let changed =
        loading::read_regular_bytes(&root_dir, JOPE_POLICY_METADATA, declared.bytes).unwrap();
    assert!(verify_file_bytes(&declared, &changed).is_err());
    std::fs::remove_dir_all(canonical_root).unwrap();
}

#[test]
fn notice_content_is_checked_against_the_manifest() {
    let (_, root) =
        open_bundle_root(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../")).unwrap();
    let mut manifest = shipped_manifest();
    *manifest
        .notices
        .get_mut("docs/notices/jain-governed-models.txt")
        .unwrap() = "0".repeat(64);
    assert!(verify_notices(&root, &manifest).is_err());
}

#[test]
fn shipped_bundle_is_complete_and_digest_checked() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../");
    let manifest_bytes = std::fs::read(root.join("artifacts/model-bundle.v1.json")).unwrap();
    assert_eq!(sha256_bytes(&manifest_bytes), MODEL_BUNDLE_MANIFEST_SHA256);
    let bundle = LoadedModelBundle::load(&root).expect("shipped model bundle");
    assert_eq!(
        bundle.path(JOPE_POLICY_METADATA).unwrap(),
        root.canonicalize().unwrap().join(JOPE_POLICY_METADATA)
    );
    assert!(bundle.path("../outside.safetensors").is_err());
    assert!(bundle.path("artifacts/model-bundle/unlisted.json").is_err());
    assert_eq!(bundle.manifest.files.len(), 14);
    let actual_paths = bundle
        .manifest
        .files
        .iter()
        .map(|file| file.path.as_str())
        .collect::<BTreeSet<_>>();
    assert_eq!(
            actual_paths,
            BTreeSet::from([
                "artifacts/foundation/tabicl-classifier-v2-chimera-shuffled.safetensors",
                "artifacts/foundation/tabicl-regressor-v2-20260212.safetensors",
                "artifacts/model-bundle/hyperion_v5_arm_b.manifest.json",
                "artifacts/model-bundle/hyperion_v5_arm_b.safetensors",
                "artifacts/model-bundle/jope_policy.meta.json",
                "artifacts/model-bundle/jope_policy.safetensors",
                "artifacts/model-bundle/operator_registry.json",
                "artifacts/model-bundle/preprocessing.safetensors",
                "artifacts/model-bundle/target_encoder_fused_v1.be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f.safetensors",
                "artifacts/model-bundle/target_encoder_fused_v1.meta.json",
                "artifacts/starforge/chimera_classification_attnfusion.safetensors",
                "artifacts/starforge/chimera_classification_expert_a.safetensors",
                "artifacts/starforge/chimera_stackfusionreg.safetensors",
                "artifacts/starforge/starlight_core_regressor.safetensors",
            ])
        );
    assert!(!bundle
        .read_verified("artifacts/model-bundle/jope_policy.meta.json")
        .unwrap()
        .is_empty());
    assert!(bundle.manifest.digest().unwrap().len() == 64);
    assert_eq!(
        bundle.manifest.encoders["target"].sha256,
        "be96adc6549a478c6a1759e40ce8c403fa529694b54a74a7d93a6dd56d35a40f"
    );
    assert_eq!(
        bundle.manifest.source_checkpoint_sha256,
        JOPE_SOURCE_CHECKPOINT_SHA256
    );
    assert_eq!(bundle.manifest.vocabulary_sha256, JOPE_VOCABULARY_SHA256);
    assert_eq!(
        bundle.manifest.vocabulary_canonical_sha256,
        JOPE_VOCABULARY_CANONICAL_SHA256
    );
    assert_eq!(bundle.manifest.grammar_sha256, JOPE_GRAMMAR_SHA256);
    assert_eq!(bundle.manifest.registry_sha256, OPERATOR_REGISTRY_SHA256);
    assert_eq!(
        bundle.manifest.registry_semantic_sha256,
        OPERATOR_REGISTRY_SEMANTIC_SHA256
    );
}

#[cfg(unix)]
#[test]
fn rejects_manifest_final_and_intermediate_symlinks() {
    use std::os::unix::fs::symlink;
    use std::time::{SystemTime, UNIX_EPOCH};

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let base = std::env::temp_dir().join(format!("starforge-bundle-symlink-{nonce}"));
    let root = base.join("root");
    let outside = base.join("outside");
    std::fs::create_dir_all(root.join("safe")).unwrap();
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("file.json"), b"{}").unwrap();

    symlink(outside.join("file.json"), root.join("model-bundle.v1.json")).unwrap();
    let (_, root_dir) = open_bundle_root(&root).unwrap();
    assert!(read_manifest(&root_dir).is_err());
    std::fs::remove_file(root.join("model-bundle.v1.json")).unwrap();

    symlink(outside.join("file.json"), root.join("safe/final.json")).unwrap();
    assert!(loading::read_regular_bytes(&root_dir, "safe/final.json", 16).is_err());
    std::fs::remove_file(root.join("safe/final.json")).unwrap();

    symlink(&outside, root.join("escape")).unwrap();
    assert!(loading::read_regular_bytes(&root_dir, "escape/file.json", 16).is_err());

    let root_link = base.join("root-link");
    symlink(&root, &root_link).unwrap();
    assert!(open_bundle_root(&root_link).is_err());
    std::fs::remove_dir_all(&base).unwrap();
}

#[test]
fn descriptor_relative_reads_enforce_the_byte_bound() {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!("starforge-bundle-bound-{nonce}"));
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("bounded.json"), [0u8; 17]).unwrap();
    let (_, root_dir) = open_bundle_root(&root).unwrap();
    assert!(loading::read_regular_bytes(&root_dir, "bounded.json", 16).is_err());
    assert_eq!(
        loading::read_regular_bytes(&root_dir, "bounded.json", 17).unwrap(),
        [0u8; 17]
    );
    std::fs::remove_dir_all(root).unwrap();
}
