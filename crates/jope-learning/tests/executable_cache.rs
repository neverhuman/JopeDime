use jope_learning::exe_cache::{
    ExecutableCache, ExecutableCacheKey, KeeperPolicy, PackCatalog, PackManifest, PackMemberRange,
};

fn key() -> ExecutableCacheKey {
    ExecutableCacheKey {
        program_sha256: "a".repeat(64),
        emitter: "jope-emitter".into(),
        emitter_version: "1.2.3".into(),
        toolchain: "rustc-1.90.0".into(),
        target: "x86_64-unknown-linux-gnu".into(),
        numeric_profile: "f64-strict-v1".into(),
        fitted_identity: "fit-abc".into(),
    }
}

fn pack(members: Vec<PackMemberRange>) -> PackManifest {
    PackManifest {
        schema: "jope.executable-pack/v1".into(),
        pack_key: "packs/run/00001".into(),
        pack_version_id: "version-1".into(),
        pack_sha256: "b".repeat(64),
        pack_bytes: 20,
        members,
    }
}

fn member(key: &str, version: &str, offset: u64, length: u64) -> PackMemberRange {
    PackMemberRange {
        key: key.into(),
        version_id: version.into(),
        sha256: "c".repeat(64),
        offset,
        length,
    }
}

#[test]
fn malformed_overlapping_and_out_of_bounds_ranges_are_rejected() {
    assert!(
        pack(vec![member("a", "v1", 0, 12), member("b", "v1", 11, 4)])
            .validate()
            .is_err()
    );
    assert!(pack(vec![member("a", "v1", 19, 2)]).validate().is_err());
    assert!(pack(vec![member("a", "v1", u64::MAX, 2)])
        .validate()
        .is_err());
    assert!(
        pack(vec![member("a", "v1", 0, 10), member("a", "v2", 10, 10)])
            .validate()
            .is_err()
    );
}

#[test]
fn exact_ranges_and_member_versions_are_accepted() {
    let bytes = b"abcdefghij0123456789";
    let manifest = PackManifest {
        schema: "jope.executable-pack/v1".into(),
        pack_key: "packs/run/00001".into(),
        pack_version_id: "version-1".into(),
        pack_sha256: jope_learning::canon::sha256_hex(bytes),
        pack_bytes: bytes.len() as u64,
        members: vec![
            PackMemberRange {
                key: "a".into(),
                version_id: "version-a".into(),
                sha256: jope_learning::canon::sha256_hex(&bytes[..10]),
                offset: 0,
                length: 10,
            },
            PackMemberRange {
                key: "b".into(),
                version_id: "version-b".into(),
                sha256: jope_learning::canon::sha256_hex(&bytes[10..]),
                offset: 10,
                length: 10,
            },
        ],
    };
    manifest.verify_bytes(bytes).unwrap();
    let mut changed = bytes.to_vec();
    changed[0] = b'X';
    assert!(manifest.verify_bytes(&changed).is_err());
}

#[test]
fn immutable_pack_retry_is_idempotent_and_same_key_new_version_is_refused() {
    let bytes = b"abcdefghij";
    let manifest = PackManifest {
        schema: "jope.executable-pack/v1".into(),
        pack_key: "packs/run/00001".into(),
        pack_version_id: "version-1".into(),
        pack_sha256: jope_learning::canon::sha256_hex(bytes),
        pack_bytes: bytes.len() as u64,
        members: vec![PackMemberRange {
            key: "member-a".into(),
            version_id: "member-v1".into(),
            sha256: jope_learning::canon::sha256_hex(bytes),
            offset: 0,
            length: bytes.len() as u64,
        }],
    };
    let mut catalog = PackCatalog::default();
    catalog.admit(&manifest, bytes).unwrap(); // first publish
    catalog.admit(&manifest, bytes).unwrap(); // lost acknowledgement, exact retry

    let mut changed_pack = manifest.clone();
    changed_pack.pack_version_id = "version-2".into();
    assert!(catalog.admit(&changed_pack, bytes).is_err());

    let mut changed_member = manifest.clone();
    changed_member.pack_key = "packs/run/00002".into();
    changed_member.members[0].version_id = "member-v2".into();
    assert!(catalog.admit(&changed_member, bytes).is_err());
}

#[test]
fn cache_requires_reproducible_bytes_and_refuses_same_key_new_version() {
    let mut cache = ExecutableCache::default();
    assert!(cache.insert_verified(&key(), b"first", b"second").is_err());
    let stored = cache
        .insert_verified(&key(), b"executable", b"executable")
        .unwrap();
    assert_eq!(stored.bytes, b"executable");
    assert!(cache
        .insert_verified(&key(), b"different", b"different")
        .is_err());
    assert_eq!(cache.get(&key()).unwrap().unwrap().bytes, b"executable");
}

#[test]
fn all_executable_identity_fields_change_the_cache_key() {
    let original = key().digest().unwrap();
    let mutations: [fn(&mut ExecutableCacheKey); 7] = [
        |value| value.program_sha256 = "d".repeat(64),
        |value| value.emitter.push_str("-changed"),
        |value| value.emitter_version.push_str("+patch1"),
        |value| value.toolchain.push_str("+patch1"),
        |value| value.target.push_str("-changed"),
        |value| value.numeric_profile.push_str("-changed"),
        |value| value.fitted_identity.push_str("-changed"),
    ];
    for mutate in mutations {
        let mut changed = key();
        mutate(&mut changed);
        assert_ne!(original, changed.digest().unwrap());
    }
}

#[test]
fn keeper_evidence_requires_declared_permissions_before_omission() {
    let keeper = KeeperPolicy {
        is_keeper: true,
        proof_obligation_allows_omission: true,
        retention_obligation_allows_omission: true,
    };
    assert!(!keeper.may_omit());
    keeper.require_keep_or_permitted_omission().unwrap();
    let missing_retention = KeeperPolicy {
        is_keeper: false,
        proof_obligation_allows_omission: true,
        retention_obligation_allows_omission: false,
    };
    assert!(!missing_retention.may_omit());
    assert!(missing_retention
        .require_keep_or_permitted_omission()
        .is_err());
    let permitted = KeeperPolicy {
        is_keeper: false,
        proof_obligation_allows_omission: true,
        retention_obligation_allows_omission: true,
    };
    assert!(permitted.may_omit());
}
