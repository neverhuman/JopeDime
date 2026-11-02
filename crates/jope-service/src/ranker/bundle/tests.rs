use super::*;
use std::{cell::Cell, io::Cursor};

use super::*;
use std::{cell::Cell, io::Cursor};

fn identity(bytes: &[u8]) -> AssetIdentity {
    AssetIdentity {
        bytes: bytes.len() as u64,
        sha256: Sha256::digest(bytes).into(),
    }
}

fn limits() -> BundleLimits {
    BundleLimits {
        manifest_bytes: 1024,
        tensor_header_bytes: 64,
        file_bytes: 128,
        total_bytes: 256,
    }
}

fn refusal<T: fmt::Debug>(result: Result<T>) {
    let error = result.unwrap_err();
    assert_eq!(error.code(), ErrorCode::ModelIntegrityFailed);
    assert_eq!(error.exit_code(), 3);
    assert!(error.to_string().starts_with("MODEL_INTEGRITY_FAILED: "));
}

#[test]
fn exact_hashed_read_accepts_and_refuses_corruption_truncation_and_growth() {
    let original = b"signed catalog file";
    let expected = identity(original);
    assert_eq!(
        read_hashed_asset(Cursor::new(original), expected, limits()).unwrap(),
        original
    );
    refusal(read_hashed_asset(
        Cursor::new(b"corrupt catalog fil"),
        expected,
        limits(),
    ));
    refusal(read_hashed_asset(
        Cursor::new(&original[..original.len() - 1]),
        expected,
        limits(),
    ));
    refusal(read_hashed_asset(
        Cursor::new([original.as_slice(), b"!"].concat()),
        expected,
        limits(),
    ));
}

#[test]
fn caps_refuse_before_read_and_checked_total_never_wraps() {
    struct CountReads<'a>(&'a Cell<usize>);
    impl Read for CountReads<'_> {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            self.0.set(self.0.get() + 1);
            panic!("oversized asset must be refused before reading")
        }
    }
    let count = Cell::new(0);
    let too_big = AssetIdentity {
        bytes: 129,
        sha256: [0; 32],
    };
    refusal(read_hashed_asset(CountReads(&count), too_big, limits()));
    assert_eq!(count.get(), 0);
    let asset = AssetIdentity {
        bytes: 100,
        sha256: [0; 32],
    };
    assert!(check_byte_budget(&[asset, asset], limits()).is_ok());
    refusal(check_byte_budget(&[asset, asset, asset], limits()));
    let huge = AssetIdentity {
        bytes: u64::MAX,
        sha256: [0; 32],
    };
    refusal(check_byte_budget(
        &[huge, huge],
        BundleLimits {
            file_bytes: u64::MAX,
            total_bytes: u64::MAX,
            ..limits()
        },
    ));
}

#[test]
fn header_length_is_bounded_without_parsing_or_overflow() {
    let mut valid = 2u64.to_le_bytes().to_vec();
    valid.extend_from_slice(b"{}");
    assert!(check_tensor_header(&valid, limits()).is_ok());
    refusal(check_tensor_header(&valid[..7], limits()));
    refusal(check_tensor_header(&[0; 8], limits()));
    refusal(check_tensor_header(&65u64.to_le_bytes(), limits()));
    refusal(check_tensor_header(&u64::MAX.to_le_bytes(), limits()));
    refusal(check_tensor_header(&64u64.to_le_bytes(), limits()));
}

#[test]
fn bundle_digest_requires_exact_lowercase_prefixed_identity() {
    let text = format!("sha256:{}", "ab".repeat(32));
    assert_eq!(BundleDigest::parse(&text).unwrap().to_string(), text);
    for bad in [
        "ab".repeat(32),
        format!("sha256:{}", "AB".repeat(32)),
        format!("sha256:{}", "ab".repeat(31)),
    ] {
        refusal(BundleDigest::parse(&bad));
    }
}

pub(crate) fn manifest() -> serde_json::Value {
    serde_json::json!({
        "schema": "jope.fused-bundle/v1", "architecture": "dope-fused-ranker/h40b",
        "bundle_digest": format!("sha256:{}", "ab".repeat(32)),
        "dope_core_rev": "1".repeat(40),
        "dope_weights_sha256": "2".repeat(64), "jope_weights_sha256": "3".repeat(64),
        "reference_source_sha256": {"fixture.rs": "4".repeat(64)},
        "constants": {"scale": 0.4208, "ctx_feat_dim": 584, "revealed": 4,
            "proposals": 8, "proposals_explore": 16, "admit": 4, "elite_pool": 48},
        "surrogate_recipe": "fixture", "trained_objectives": ["regression/r2"],
        "landmark_recipe": "fixture", "landmark_sample_rows": 256,
        "qualified": [{"task": "regression", "landmarks": "fixture", "backend": "cpu"}],
        "asset_policy": "fixture, not a release"
    })
}

#[test]
fn frozen_manifest_is_strict_and_unknown_architectures_are_refused() {
    let bytes = serde_json::to_vec(&manifest()).unwrap();
    let parsed = inspect_manifest(&bytes, limits()).unwrap();
    assert_eq!(parsed.dope_weights_sha256, [0x22; 32]);
    assert_eq!(parsed.qualified[0].task, QualifiedTask::Regression);
    refusal(inspect_manifest(
        &bytes,
        BundleLimits {
            manifest_bytes: 10,
            ..limits()
        },
    ));
    for (field, value) in [
        ("architecture", serde_json::json!("future-architecture")),
        ("schema", serde_json::json!("jope.fused-bundle/v2")),
        ("bundle_digest", serde_json::json!("sha256:invalid")),
        ("dope_weights_sha256", serde_json::json!("invalid")),
        ("dope_core_rev", serde_json::json!("invalid")),
        ("extra", serde_json::json!(true)),
    ] {
        let mut bad = manifest();
        bad[field] = value;
        refusal(inspect_manifest(
            &serde_json::to_vec(&bad).unwrap(),
            limits(),
        ));
    }
    for pointer in ["/constants/extra", "/qualified/0/extra"] {
        let mut bad = manifest();
        let (parent, key) = pointer.rsplit_once('/').unwrap();
        bad.pointer_mut(parent).unwrap()[key] = true.into();
        refusal(inspect_manifest(
            &serde_json::to_vec(&bad).unwrap(),
            limits(),
        ));
    }
    let mut bad = manifest();
    bad["constants"]["ctx_feat_dim"] = 585.into();
    refusal(inspect_manifest(
        &serde_json::to_vec(&bad).unwrap(),
        limits(),
    ));
}
