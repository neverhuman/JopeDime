/// THE two-day outage, in numbers taken off the live root at 23:33Z: the
/// corpus pointer pinned at 1,805,437 by a tournament parked at
/// `phase_index 9`, against 4,691,549 ingested labels. Nothing in the alarm
/// set said a word about it before this check existed.
#[test]
fn a_severed_freeze_loop_is_critical() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 1_805_437, 4_691_549);
    let alarms = freeze_scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "corpus_freeze_starved")
        .expect("expected corpus_freeze_starved");
    assert_eq!(row["severity"], "critical");
    assert!(
        row["observed"].as_str().unwrap().contains("2886112 labels"),
        "{row}"
    );
}

/// One freeze behind is not an outage. The step is 50,000 labels, so a gap
/// of that order must stay silent or the alarm is noise.
#[test]
fn one_step_of_unfrozen_labels_is_silent() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_600_000, 4_650_000);
    assert!(freeze_scan(&root).rows.is_empty());
}

#[test]
fn a_few_missed_freezes_is_a_warning_not_a_critical() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_400_000, 4_650_000);
    let alarms = freeze_scan(&root);
    assert_eq!(
        alarm_named(&alarms, "corpus_freeze_starved")["severity"],
        "warning"
    );
}

/// Guardrail 1 of ops/generalization-plan.md. Evidence only accumulates, so
/// a fall in dataset count is always a defect.
#[test]
fn a_fallen_dataset_count_is_critical() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_650_000, 4_650_000);
    std::fs::write(
        root.join("ops/corpus-freeze-guard.v1.json"),
        serde_json::to_vec(&json!({
            "schema": "JopeCorpusFreezeGuard.v1",
            "regressed": true,
            "observed": "1204 datasets at watermark 4690000 (full_consolidation/eq32v32c128)",
            "threshold": ">=2857 datasets (high-water mark set at watermark 1805437)",
        }))
        .unwrap(),
    )
    .unwrap();
    let alarms = freeze_scan(&root);
    let row = alarm_named(&alarms, "corpus_dataset_count_fell");
    assert_eq!(row["severity"], "critical");
    assert!(row["threshold"].as_str().unwrap().contains("2857"));
}

#[test]
fn a_healthy_freeze_guard_is_silent() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_650_000, 4_650_000);
    std::fs::write(
        root.join("ops/corpus-freeze-guard.v1.json"),
        serde_json::to_vec(&json!({
            "schema": "JopeCorpusFreezeGuard.v1",
            "regressed": false,
            "observed": "18000 datasets at watermark 4690000",
            "threshold": ">=2857 datasets",
            "split_median_divergence": {
                "binary": {"train": 32.0, "validation": 32.0, "ratio": 1.0},
                "regression": {"train": 25.0, "validation": 25.0, "ratio": 1.0},
            },
        }))
        .unwrap(),
    )
    .unwrap();
    assert!(freeze_scan(&root).rows.is_empty());
}

/// Guardrail 2. The historical failure was train median 6 against validation
/// median 485 — an 80x inversion that was read as an overfit gap for a day.
#[test]
fn diverged_split_medians_are_critical_per_task() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_650_000, 4_650_000);
    std::fs::write(
        root.join("ops/corpus-freeze-guard.v1.json"),
        serde_json::to_vec(&json!({
            "schema": "JopeCorpusFreezeGuard.v1",
            "regressed": false,
            "split_median_divergence": {
                "binary": {"train": 32.0, "validation": 32.0, "ratio": 1.0},
                "regression": {"train": 6.0, "validation": 485.0, "ratio": 80.83},
            },
        }))
        .unwrap(),
    )
    .unwrap();
    let names = alarm_names(&freeze_scan(&root));
    assert!(
        names.contains(&"corpus_split_median_divergence:regression".to_owned()),
        "{names:?}"
    );
    assert!(
        !names.contains(&"corpus_split_median_divergence:binary".to_owned()),
        "{names:?}"
    );
}

/// The freezer publishing `label_pointer_behind_corpus` is the one way it
/// can stop the trainer watchdog, so it must arrive as a critical alarm.
#[test]
fn the_freezers_own_alarm_is_lifted() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_650_000, 4_650_000);
    std::fs::write(
        root.join("ops/corpus-freezer-v1.json"),
        serde_json::to_vec(&json!({
            "schema": "JopeSelfplayCorpusFreezer.v1",
            "lifecycle": "ok",
            "action": "froze",
            "alarm": "label_pointer_behind_corpus",
            "detail": "exact_rows_total 4690000 < frozen watermark 4691000",
        }))
        .unwrap(),
    )
    .unwrap();
    let alarms = freeze_scan(&root);
    assert_eq!(
        alarm_named(&alarms, "label_pointer_behind_corpus")["severity"],
        "critical"
    );
}

/// A root where the freezer was never deployed must not carry a permanent
/// alarm for a unit it does not have.
#[test]
fn an_undeployed_freezer_is_silent() {
    let temp = TempDir::new().unwrap();
    let root = freeze_root(&temp);
    write_pointers(&root, 4_650_000, 4_650_000);
    let names = alarm_names(&freeze_scan(&root));
    assert!(
        !names.iter().any(|name| name.contains("corpus-freezer")),
        "{names:?}"
    );
}

// -----------------------------------------------------------------------
// Unsubmittable trainer intent
// -----------------------------------------------------------------------

/// The live state document at 23:24Z on 2026-07-28, byte for byte, while
/// the trainer had been unable to submit for 31 minutes and every alarm in
/// this file was green.
const LIVE_BRICKED_INTENT: &str = r#"{"intent":{"checkpoint_mismatch_observations":0,"checkpoint_sha256":"43c64bde25df51e1f3771533b151c82d971cfaf026dada339e400d69950cba40","created_at_ms":1785279187223,"job_id":"jope-selfplay-train-001805437-c-r000025","last_error":"SCQ check rejected trainer: error[input-hash-mismatch]: input 'script' hash mismatch: expected e9f783d94c08274f8dbd1252efe964a5f8e369576ea55c7be43e7d04fd7a521d, got 2733e2da17dcc60a7667713ce70e552604c236bb7243ea4a530e20d09b152492","mode":"continuous_recent_plus_replay","seed":20260751,"sequence":25,"spec":"/srv/flywheel/test_exact_gen0/specs/jope-selfplay-train-001805437-c-r000025.toml","spec_sha256":"637280023e1ce5c90895896836cc6adbd2eb9ff52819cc03227e46da88234b1e","watermark":1805437},"last_evidence_watermark":1805437,"max_observed_labels":4683572,"max_observed_snapshot_watermark":1805437,"next_sequence":26,"schema":"JopeContinuousTrainerWatchdogState.v1","updated_at_ms":1785281041552}"#;

fn intent_scan(root: &Path) -> Alarms {
    let mut alarms = Alarms { rows: Vec::new() };
    check_trainer_intent(root, &mut alarms);
    alarms
}

#[test]
fn the_live_bricked_intent_is_critical_and_prints_the_recovery() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("ops")).unwrap();
    std::fs::write(
        temp.path().join("ops/trainer-watchdog-state.v1.json"),
        LIVE_BRICKED_INTENT,
    )
    .unwrap();
    let alarms = intent_scan(temp.path());
    let row = alarm_named(&alarms, "trainer_intent_unsubmittable");
    assert_eq!(row["severity"], "critical");
    let observed = row["observed"].as_str().unwrap();
    assert!(
        observed.contains("jope-selfplay-train-001805437-c-r000025"),
        "{observed}"
    );
    assert!(observed.contains("\"intent\": null"), "{observed}");
}

/// A capacity rejection is transient and self-heals. Alarming on it would
/// make this check noise on a busy cluster, which is where it matters most.
#[test]
fn a_transient_capacity_rejection_is_not_an_alarm() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("ops")).unwrap();
    std::fs::write(
            temp.path().join("ops/trainer-watchdog-state.v1.json"),
            serde_json::to_vec(&json!({
                "intent": {
                    "job_id": "jope-selfplay-train-001805437-c-r000026",
                    "last_error": "SCQ check rejected trainer: error[dependency-unsatisfied]: dependencies cannot be verified on any eligible node",
                },
            }))
            .unwrap(),
        )
        .unwrap();
    assert!(intent_scan(temp.path()).rows.is_empty());
}

#[test]
fn a_healthy_or_absent_intent_is_silent() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir_all(temp.path().join("ops")).unwrap();
    assert!(intent_scan(temp.path()).rows.is_empty());
    std::fs::write(
        temp.path().join("ops/trainer-watchdog-state.v1.json"),
        serde_json::to_vec(&json!({"intent": null, "next_sequence": 27})).unwrap(),
    )
    .unwrap();
    assert!(intent_scan(temp.path()).rows.is_empty());
}

fn alarm_named<'a>(alarms: &'a Alarms, name: &str) -> &'a Value {
    alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == name)
        .unwrap_or_else(|| panic!("missing {name}: {:?}", alarm_names(alarms)))
}
