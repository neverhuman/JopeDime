use super::*;

#[test]
fn duplicate_json_metadata_is_refused_before_cross_language_disagreement() {
    let object = future();
    let text = String::from_utf8(object.manifest_bytes).unwrap();
    let duplicated = text.replace(
        "\"extension\": {",
        "\"extension\": {\"unicode\":\"other\"}, \"extension\": {",
    );
    assert!(ModelObject::decode(duplicated.into_bytes(), object.parts, 1024).is_err());
}

#[test]
fn new_gp_fields_cannot_be_silently_dropped_by_an_older_decoder() {
    let genome = feat_math::GenomeIr::parse("(pipe X (ridge))").unwrap();
    let x = (0..16)
        .map(|i| vec![i as f64, (i % 3) as f64])
        .collect::<Vec<_>>();
    let y = (0..16).map(|i| i as f64 * 0.5).collect::<Vec<_>>();
    let fitted = feat_math::advanced_v3::fit_legacy_numeric_bound(&genome, &x, &y).unwrap();
    let mut value = serde_json::to_value(&fitted).unwrap();
    let head = value
        .pointer_mut("/predictor/Legacy/Pipeline/head/Ridge")
        .unwrap();
    head.as_object_mut()
        .unwrap()
        .insert("future_required_state".into(), json!({"offset": 123}));
    let bytes = serde_json::to_vec(&value).unwrap();
    let mut manifest = ModelObject::from_bound(&fitted).unwrap().manifest;
    manifest.files = vec![Part::new("state.json", "application/json", &bytes)];
    let object = ModelObject::new(
        manifest,
        BTreeMap::from([("state.json".into(), bytes)]),
        1 << 20,
    )
    .unwrap();
    assert!(
        object.bound().is_err(),
        "older decoder discarded required fitted state"
    );
}

#[test]
fn future_variants_retain_all_bytes_but_require_their_decoder() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    let loaded = ModelObject::read(&path, 1024).unwrap();
    assert_eq!(object.manifest_bytes, loaded.manifest_bytes);
    assert_eq!(object.parts, loaded.parts);
    let inspection = loaded.inspect().unwrap();
    assert!(inspection.integrity_verified);
    assert!(!inspection.executable);
    assert_eq!(inspection.unsupported_capabilities.len(), 2);
    assert_eq!(
        crate::error::kind_of(&loaded.bound().unwrap_err()),
        crate::FailureKind::Backend
    );
    assert_eq!(object.save(root.path(), 1024).unwrap(), path);
}

#[test]
fn manifest_identity_uses_retained_bytes_including_foreign_whitespace() {
    let object = future();
    let mut bytes = b" \n".to_vec();
    bytes.extend_from_slice(&serde_json::to_vec(&object.manifest).unwrap());
    bytes.push(b'\n');
    let foreign = ModelObject::decode(bytes.clone(), object.parts.clone(), 1024).unwrap();
    assert_ne!(foreign.id, object.id);
    assert_eq!(foreign.id, format!("sha256-{}", sha256_hex(&bytes)));
    let root = tempfile::tempdir().unwrap();
    let path = foreign.save(root.path(), 1024).unwrap();
    assert_eq!(
        ModelObject::read(&path, 1024).unwrap().manifest_bytes,
        bytes
    );
}

#[test]
fn truncation_substitution_wrong_directory_and_uncommitted_store_are_refused() {
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    fs::write(path.join("weights.bin"), [0, 255]).unwrap();
    assert!(ModelObject::read(&path, 1024).is_err());
    assert!(object.save(root.path(), 1024).is_err());
    assert_eq!(fs::read(path.join("weights.bin")).unwrap(), [0, 255]);
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    let renamed = root.path().join(format!("sha256-{}", "0".repeat(64)));
    fs::rename(&path, &renamed).unwrap();
    assert!(ModelObject::read(&renamed, 1024).is_err());
    fs::remove_file(renamed.join(MANIFEST)).unwrap();
    assert!(ModelObject::read(&renamed, 1024).is_err());
}

#[test]
fn envelope_bounds_and_paths_are_checked_before_part_access() {
    let object = future();
    for bad_name in [
        "../escape",
        "/abs",
        "x/y",
        "x\\y",
        ".hidden",
        "C:state",
        MANIFEST,
    ] {
        let mut manifest = object.manifest.clone();
        manifest.files[0].path = bad_name.into();
        assert!(manifest.validate(1024).is_err(), "{bad_name}");
    }
    let mut manifest = object.manifest.clone();
    manifest.minimum_reader_version = 2;
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.format = "/invalid/".into();
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.files.push(manifest.files[0].clone());
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest
        .required_capabilities
        .push(manifest.required_capabilities[0].clone());
    assert!(manifest.validate(1024).is_err());
    manifest = object.manifest.clone();
    manifest.files[0].bytes = u64::MAX;
    assert!(manifest.validate(u64::MAX).is_err());
    assert!(object.manifest.validate(2).is_err());
    assert!(ModelObject::decode(
        vec![b' '; MAX_MANIFEST_BYTES as usize + 1],
        BTreeMap::new(),
        1024
    )
    .is_err());
    let mut parts = object.parts.clone();
    parts.insert("unlisted.bin".into(), vec![]);
    assert!(ModelObject::decode(object.manifest_bytes.clone(), parts, 1024).is_err());
}

#[test]
fn unknown_required_capabilities_are_never_ignored_for_a_known_format() {
    let object = future();
    let mut manifest = object.manifest.clone();
    manifest.format = APPLIANCE.into();
    let object = ModelObject::new(manifest, object.parts, 1024).unwrap();
    assert!(!object.inspect().unwrap().executable);
    assert!(object.fitted().is_err());
}

#[test]
fn atomic_create_only_exports_preserve_existing_files() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("export.bin");
    write_new(&path, b"original").unwrap();
    assert!(write_new(&path, b"replacement").is_err());
    assert_eq!(fs::read(path).unwrap(), b"original");
    assert_eq!(fs::read_dir(root.path()).unwrap().count(), 1);
}

#[cfg(unix)]
#[test]
fn symlinked_parts_and_object_directories_are_refused() {
    use std::os::unix::fs::{symlink, PermissionsExt};
    let object = future();
    let root = tempfile::tempdir().unwrap();
    let path = object.save(root.path(), 1024).unwrap();
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o700
    );
    let alias = root.path().join("alias");
    symlink(&path, &alias).unwrap();
    assert!(ModelObject::read(&alias, 1024).is_err());
    let payload = path.join("weights.bin");
    fs::rename(&payload, root.path().join("external.bin")).unwrap();
    symlink(root.path().join("external.bin"), &payload).unwrap();
    assert!(ModelObject::read(&path, 1024).is_err());
}
