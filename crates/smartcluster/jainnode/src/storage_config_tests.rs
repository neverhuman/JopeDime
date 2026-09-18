use std::{
    fs,
    os::unix::fs::{symlink, PermissionsExt},
    path::{Path, PathBuf},
};

use serde_json::{json, Value};

use crate::{StorageConfig, StorageConfigError};

fn write_private(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
    fs::set_permissions(path, fs::Permissions::from_mode(0o600)).unwrap();
}

fn fixture(root: &Path) -> (PathBuf, Value) {
    let staging = root.join("staging");
    let volume_a = root.join("volume-a");
    let volume_b = root.join("volume-b");
    for directory in [&staging, &volume_a, &volume_b] {
        fs::create_dir(directory).unwrap();
        fs::set_permissions(directory, fs::Permissions::from_mode(0o700)).unwrap();
    }
    let dedup = root.join("dedup.key");
    let kek_one = root.join("kek-1.key");
    let kek_two = root.join("kek-2.key");
    write_private(&dedup, &[1; 32]);
    write_private(&kek_one, &[2; 32]);
    write_private(&kek_two, &[3; 32]);
    let config = root.join("storage.json");
    let document = json!({
        "schema": "jain.node.storage-config/v1",
        "deployment_id": "deployment-1",
        "node_id": "node-1",
        "staging_parent": staging,
        "volumes": [
            {
                "root": volume_a,
                "filesystem_uuid": "11111111-1111-1111-1111-111111111111",
                "failure_domain": "domain-a"
            },
            {
                "root": volume_b,
                "filesystem_uuid": "22222222-2222-2222-2222-222222222222",
                "failure_domain": "domain-b"
            }
        ],
        "projects": [{
            "project_id": "project-1",
            "dedup_key_path": dedup,
            "active_kek_version": 2,
            "kek_files": [
                {"version": 1, "path": kek_one},
                {"version": 2, "path": kek_two}
            ]
        }]
    });
    (config, document)
}

fn write_config(path: &Path, document: &Value) {
    write_private(path, &serde_json::to_vec(document).unwrap());
}

#[test]
fn closed_config_loads_key_paths_without_serializing_secret_bytes() {
    let schema: Value = serde_json::from_str(include_str!(
        "../../schemas/jain.node.storage-config-v1.schema.json"
    ))
    .unwrap();
    assert_eq!(schema["$id"], "jain.node.storage-config/v1");
    assert_eq!(schema["additionalProperties"], false);

    let root = tempfile::tempdir().unwrap();
    let (path, document) = fixture(root.path());
    write_config(&path, &document);
    let config = StorageConfig::open(&path).unwrap();
    assert_eq!(config.deployment_id(), "deployment-1");
    assert_eq!(config.node_id(), "node-1");
    assert_eq!(config.projects().len(), 1);
    assert_eq!(config.volumes().len(), 2);
    assert_eq!(
        config.source_sha256(),
        jain_smartcluster_fabric::sha256(&fs::read(&path).unwrap())
    );

    let keys = config.load_project_keys("project-1").unwrap();
    assert_eq!(keys.project_id(), "project-1");
    assert_eq!(keys.active_kek_version(), 2);
    assert_eq!(keys.dedup_key(), &[1; 32]);
    assert_eq!(
        keys.keks()
            .map(|(version, key)| (version, key[0]))
            .collect::<Vec<_>>(),
        vec![(1, 2), (2, 3)]
    );
    let encoded = serde_json::to_vec(&document).unwrap();
    assert!(!encoded.windows(32).any(|window| window == [1; 32]));
}

#[test]
fn config_and_keys_require_direct_owner_only_single_link_files() {
    let root = tempfile::tempdir().unwrap();
    let (path, document) = fixture(root.path());
    write_config(&path, &document);

    fs::set_permissions(&path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(StorageConfig::open(&path).is_err());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();

    let hardlink = root.path().join("storage-hardlink.json");
    fs::hard_link(&path, &hardlink).unwrap();
    assert!(StorageConfig::open(&path).is_err());
    fs::remove_file(&hardlink).unwrap();

    let alias = root.path().join("storage-alias.json");
    symlink(&path, &alias).unwrap();
    assert!(StorageConfig::open(&alias).is_err());

    let key_path = PathBuf::from(document["projects"][0]["dedup_key_path"].as_str().unwrap());
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o640)).unwrap();
    assert!(StorageConfig::open(&path).is_err());
    fs::set_permissions(&key_path, fs::Permissions::from_mode(0o600)).unwrap();
    fs::write(&key_path, [9; 31]).unwrap();
    assert!(StorageConfig::open(&path).is_err());
}

#[test]
fn topology_registry_and_unknown_fields_fail_closed() {
    let root = tempfile::tempdir().unwrap();
    let (path, document) = fixture(root.path());
    let mut hostiles = Vec::new();

    let mut unknown = document.clone();
    unknown["host_command"] = json!("mount /dev/hostile");
    hostiles.push(unknown);

    let mut duplicate_domain = document.clone();
    duplicate_domain["volumes"][1]["failure_domain"] = json!("domain-a");
    hostiles.push(duplicate_domain);

    let mut active_missing = document.clone();
    active_missing["projects"][0]["active_kek_version"] = json!(3);
    hostiles.push(active_missing);

    let mut duplicate_project = document.clone();
    duplicate_project["projects"] = json!([
        document["projects"][0].clone(),
        document["projects"][0].clone()
    ]);
    hostiles.push(duplicate_project);

    let mut uppercase_uuid = document.clone();
    uppercase_uuid["volumes"][0]["filesystem_uuid"] = json!("AAAAAAAA-AAAA-AAAA-AAAA-AAAAAAAAAAAA");
    hostiles.push(uppercase_uuid);

    for hostile in hostiles {
        write_config(&path, &hostile);
        assert!(StorageConfig::open(&path).is_err(), "accepted {hostile}");
    }
}

#[test]
fn host_validation_rejects_claimed_uuids_that_do_not_match_devices() {
    let root = tempfile::tempdir().unwrap();
    let (path, document) = fixture(root.path());
    write_config(&path, &document);
    let config = StorageConfig::open(&path).unwrap();
    assert!(matches!(
        config.validate_host_topology(),
        Err(StorageConfigError::FilesystemUuidMismatch) | Err(StorageConfigError::SharedDevice)
    ));
}
