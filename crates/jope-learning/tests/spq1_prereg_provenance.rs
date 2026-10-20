//! MR03 acceptance tests: 3-arm prospective full-qhat CPU cohort compact roots (`1136/1136` contexts per arm)
//! and training artifact location registry (`catalogs/training-artifacts.json` + `docs/runbooks/training-artifact-locations.md`).

use jope_learning::research::root_verifier::{
    verify_3arm_cohort_and_artifact_registry_roots as verify_roots,
    verify_prospective_3arm_cohort_roots, verify_training_artifact_location_registry,
};
use std::fs;
use std::path::{Path, PathBuf};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn copy_wave_tree(src_root: &Path, dst_root: &Path) {
    let waves_src = src_root.join("docs/research/waves");
    let waves_dst = dst_root.join("docs/research/waves");
    fs::create_dir_all(&waves_dst).expect("create waves_dst");
    fs::copy(waves_src.join("INDEX.jsonl"), waves_dst.join("INDEX.jsonl"))
        .expect("copy INDEX.jsonl");
    fn copy_directory(src: &Path, dst: &Path) {
        fs::create_dir_all(dst).expect("create fixture directory");
        for entry in fs::read_dir(src).expect("read fixture directory") {
            let entry = entry.expect("fixture entry");
            let kind = entry.file_type().expect("fixture type");
            assert!(!kind.is_symlink(), "fixture copies do not follow links");
            if kind.is_dir() {
                copy_directory(&entry.path(), &dst.join(entry.file_name()));
            } else {
                assert!(kind.is_file(), "unexpected fixture asset");
                fs::copy(entry.path(), dst.join(entry.file_name())).expect("copy wave file");
            }
        }
    }
    copy_directory(&waves_src, &waves_dst);
    let cat_dst = dst_root.join("catalogs");
    let runbook_dst = dst_root.join("docs/runbooks");
    fs::create_dir_all(&cat_dst).expect("create catalogs");
    fs::create_dir_all(&runbook_dst).expect("create docs/runbooks");
    fs::copy(
        src_root.join("catalogs/training-artifacts.json"),
        cat_dst.join("training-artifacts.json"),
    )
    .expect("copy training-artifacts.json");
    fs::copy(
        src_root.join("docs/runbooks/training-artifact-locations.md"),
        runbook_dst.join("training-artifact-locations.md"),
    )
    .expect("copy training-artifact-locations.md");
}

#[test]
fn cohort_3arm_roots_verified() {
    let root = repo_root();
    let summary = verify_prospective_3arm_cohort_roots(&root)
        .expect("prospective 3-arm cohort roots must verify");
    assert_eq!(summary.wave_id, "prospective-full-qhat-cpu-20261018T1516Z");
    assert_eq!(
        summary.archive_manifest_sha256,
        "76be65856630031714f10de9abc88183fab055baeff84a37228ceb910bd85cc7"
    );
    assert_eq!(summary.evidence_root_count, 11);
    assert_eq!(summary.assertion_count, 15);
    assert_eq!(summary.scored_contexts_total, 3408);
    // Versioned publications are verified too; corruption cannot disappear behind flat layout.
    let nested = tempfile::tempdir().expect("nested fixture");
    copy_wave_tree(&root, nested.path());
    let base = nested
        .path()
        .join("docs/research/waves/fresh-main-9f8d1a74-gpr-cuda-envelope-20261019T1851Z");
    let context = base.join("v1/context.md");
    let mut bytes = fs::read(&context).expect("nested context");
    bytes.push(b'\n');
    fs::write(&context, bytes).unwrap();
    let error = verify_prospective_3arm_cohort_roots(nested.path()).unwrap_err();
    assert!(error.contains("context.md sha256 mismatch"), "{error}");
    // A duplicate flat layout is ambiguous even if the versioned projection is otherwise valid.
    fs::copy(
        base.join("v1/wave.manifest.json"),
        base.join("wave.manifest.json"),
    )
    .unwrap();
    let error = verify_prospective_3arm_cohort_roots(nested.path()).unwrap_err();
    assert!(
        error.contains("ambiguous wave projection layout"),
        "{error}"
    );

    // Negative control: tampering with card.json must fail closed with a sha256 mismatch.
    let tmp = tempfile::tempdir().expect("tempdir");
    copy_wave_tree(&root, tmp.path());
    let tampered_card = tmp
        .path()
        .join("docs/research/waves/prospective-full-qhat-cpu-20261018T1516Z/card.json");
    let mut bytes = fs::read(&tampered_card).expect("read card.json");
    bytes.push(b'\n');
    fs::write(&tampered_card, bytes).expect("write tampered card.json");
    let err = verify_prospective_3arm_cohort_roots(tmp.path())
        .expect_err("tampered card.json must be rejected");
    assert!(
        err.contains("card.json sha256 mismatch"),
        "unexpected error: {err}"
    );
}

#[test]
fn artifact_location_registry_verified() {
    let root = repo_root();
    let summary = verify_training_artifact_location_registry(&root)
        .expect("training artifact location registry must verify");
    assert!(summary.artifact_count >= 9);
    assert_eq!(summary.declared_shards, 7877);
    assert_eq!(summary.shard_member_keys_located, 23631);

    // Negative control: corrupting declared_shards in catalogs/training-artifacts.json must fail closed.
    let tmp = tempfile::tempdir().expect("tempdir");
    copy_wave_tree(&root, tmp.path());
    let reg_path = tmp.path().join("catalogs/training-artifacts.json");
    let reg_text = fs::read_to_string(&reg_path).expect("read registry");
    let corrupted = reg_text.replacen("\"declared_shards\": 7877", "\"declared_shards\": 7876", 1);
    fs::write(&reg_path, corrupted).expect("write corrupted registry");
    let err = verify_training_artifact_location_registry(tmp.path())
        .expect_err("corrupted declared_shards must be rejected");
    assert!(
        err.contains("unexpected historical corpus shard counts"),
        "unexpected error: {err}"
    );
}

#[test]
fn verify_3arm_cohort_and_artifact_registry_roots() {
    let root = repo_root();
    let (cohort, registry) =
        verify_roots(&root).expect("combined MR03 cohort and artifact registry roots must verify");
    assert_eq!(cohort.scored_contexts_total, 3408);
    assert_eq!(registry.declared_shards, 7877);
    assert_eq!(registry.shard_member_keys_located, 23631);
}
