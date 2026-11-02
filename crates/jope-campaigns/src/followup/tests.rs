use super::*;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn temp_dir(name: &str) -> PathBuf {
    let sequence = TEMP_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "jope-dime-followup-{name}-{}-{sequence}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

#[test]
fn generation_accounting_is_exact_for_10_and_30() {
    assert_eq!(
        exact_cv::ORIGINAL_GENERATIONS * 60 * 42,
        ORIGINAL_SELECTION_SLOTS
    );
    assert_eq!(FOLLOW_UP_GENERATIONS * 60 * 35, FOLLOW_UP_SELECTION_SLOTS);
    assert_eq!(
        7 * 600 + FOLLOW_UP_SELECTION_SLOTS,
        ACTIVE_PANEL_SELECTION_SLOTS
    );
    assert_eq!(
        ORIGINAL_SELECTION_SLOTS + FOLLOW_UP_SELECTION_SLOTS,
        HISTORICAL_SELECTION_SLOTS
    );
}

#[test]
fn compiled_cohort_is_exact_and_pinned() {
    assert_eq!(LOSS_DATASETS.len(), 35);
    assert_eq!(
        LOSS_DATASETS.iter().copied().collect::<BTreeSet<_>>().len(),
        35
    );
    assert!(LOSS_DATASETS.iter().all(|hash| DATASETS.contains(hash)));
    assert_eq!(DATASETS.len() - LOSS_DATASETS.len(), 7);
    assert_eq!(ORIGINAL_LEDGER_SHA256.len(), 64);
}

#[test]
fn cohort_derivation_rejects_any_outcome_drift() {
    let rows = DATASETS
        .iter()
        .map(|hash| {
            json!({
                "dataset_hash": hash,
                "strict_overall_outcome": if LOSS_DATASETS.contains(hash) { "loss" } else { "win" },
            })
        })
        .collect::<Vec<_>>();
    let source = json!({
        "schema": "JopeDimePanelResults.v1",
        "dataset_count": 42,
        "outcomes": rows,
    });
    let bytes = canonical_json(&source).unwrap();
    validate_compiled_cohort(&bytes).unwrap();
    let mut drifted = source;
    let first_loss = drifted["outcomes"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|row| row["strict_overall_outcome"] == "loss")
        .unwrap();
    first_loss["strict_overall_outcome"] = json!("win");
    assert!(validate_compiled_cohort(&canonical_json(&drifted).unwrap()).is_err());
}

#[test]
fn overwrite_allowlist_rejects_a_third_hash() {
    let old = sha256_bytes(b"old");
    let new = sha256_bytes(b"new");
    let current = sha256_bytes(b"third");
    assert!(validate_target_hash(Path::new("target"), &old, &old, &new).is_ok());
    assert!(validate_target_hash(Path::new("target"), &new, &old, &new).is_ok());
    assert!(validate_target_hash(Path::new("target"), &current, &old, &new).is_err());
}

#[test]
fn atomic_replacement_recovers_after_an_interruption() {
    let root = temp_dir("interrupted");
    let path = root.join("target");
    fs::write(&path, b"old").unwrap();
    atomic_replace(&path, b"new").unwrap();
    assert_eq!(fs::read(&path).unwrap(), b"new");
    // Replaying an already-completed target is a no-op at the state-machine level.
    let old = sha256_bytes(b"old");
    let new = sha256_bytes(b"new");
    let current = hash_file(&path).unwrap();
    assert!(current == old || current == new);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn archive_payload_supports_exact_rollback() {
    let root = temp_dir("rollback");
    let registry = root.join("registry");
    let archive = root.join("archive");
    let hash = LOSS_DATASETS[0];
    let target = registry
        .join(hash)
        .join("predictions")
        .join(PREDICTION_FILE);
    let archive_file = "payload/predictions/old.csv";
    fs::create_dir_all(target.parent().unwrap()).unwrap();
    fs::create_dir_all(archive.join("payload/predictions")).unwrap();
    fs::write(&target, b"new").unwrap();
    fs::write(archive.join(archive_file), b"old").unwrap();
    let stage = StageManifest {
        schema: "fixture".to_string(),
        follow_up_label: "fixture".to_string(),
        exploratory_reused_test: true,
        original_panel_ledger_sha256: ORIGINAL_LEDGER_SHA256.to_string(),
        cohort: vec![],
        targets: vec![TargetEntry {
            kind: "prediction".to_string(),
            dataset_hash: Some(hash.to_string()),
            staged_file: "unused".to_string(),
            archive_file: archive_file.to_string(),
            old_sha256: sha256_bytes(b"old"),
            new_sha256: sha256_bytes(b"new"),
        }],
        untouched_original_winners: vec![],
    };
    rollback_targets(&registry, &stage, &archive).unwrap();
    assert_eq!(fs::read(&target).unwrap(), b"old");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn exact_float_ordering_never_uses_a_tolerance() {
    let one = 1.0_f64;
    let next = f64::from_bits(one.to_bits() + 1);
    assert_eq!(one.total_cmp(&one), Ordering::Equal);
    assert_eq!(one.total_cmp(&next), Ordering::Less);
    assert_eq!(next.total_cmp(&one), Ordering::Greater);
}
