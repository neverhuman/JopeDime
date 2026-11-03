// -----------------------------------------------------------------------
// Content contracts
// -----------------------------------------------------------------------

/// The real keeper error document, byte for byte off the live root at
/// 15:13 on 2026-07-28, while all 1,052 scheduler slots were idle.
const LIVE_KEEPER_ERROR_DOC: &str = r#"{"admission_allowed":false,"error":"native attempt submission transition count mismatch (attempt_no column reservation_owner)","generated_at_ms":1785251597154,"mode":"submit","schema":"JopeSelfplayQueueKeeperError.v1"}"#;

fn contract_root(temp: &TempDir) -> PathBuf {
    let root = temp.path().to_path_buf();
    std::fs::create_dir_all(root.join("ops")).unwrap();
    root
}

fn scan(root: &Path) -> Alarms {
    let mut alarms = Alarms { rows: Vec::new() };
    check_status_contracts(root, &mut alarms, "critical");
    alarms
}

#[test]
fn an_error_document_on_disk_is_an_alarm_however_fresh_the_file_is() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
    let alarms = scan(&root);
    let names = alarm_names(&alarms);
    assert!(
        names.contains(&"status_error_doc:ops/queue-keeper-v3.json".to_owned()),
        "{names:?}"
    );
    // check_status_ages sees a file written microseconds ago and says
    // nothing at all — that is the blind spot this replaces.
    let mut age_only = Alarms { rows: Vec::new() };
    check_status_ages(&root, &mut age_only);
    assert!(!alarm_names(&age_only)
        .iter()
        .any(|name| name.contains("queue-keeper")));
}

#[test]
fn the_error_text_and_the_expected_schema_are_both_in_the_alarm() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
    let alarms = scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "status_error_doc:ops/queue-keeper-v3.json")
        .unwrap();
    assert!(row["observed"]
        .as_str()
        .unwrap()
        .contains("native attempt submission transition count mismatch"));
    assert!(row["threshold"]
        .as_str()
        .unwrap()
        .contains("JopeSelfplayQueueKeeper.v3"));
    assert_eq!(row["severity"], "critical");
}

#[test]
fn a_persistent_failure_escalates_to_status_failing() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(root.join("ops/queue-keeper-v3.json"), LIVE_KEEPER_ERROR_DOC).unwrap();
    for cycle in 1..=FAILING_CYCLES {
        let names = alarm_names(&scan(&root));
        let failing = names.contains(&"status_failing:ops/queue-keeper-v3.json".to_owned());
        assert_eq!(failing, cycle >= FAILING_CYCLES, "cycle {cycle}: {names:?}");
    }
    // Recovery clears the streak.
    std::fs::write(
        root.join("ops/queue-keeper-v3.json"),
        r#"{"schema":"JopeSelfplayQueueKeeper.v3"}"#,
    )
    .unwrap();
    let names = alarm_names(&scan(&root));
    assert!(
        !names.iter().any(|name| name.starts_with("status_failing")),
        "{names:?}"
    );
}

#[test]
fn a_missing_required_field_names_the_file_and_the_field() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    // Correct schema, correct-looking document — but the field the supply
    // supervisor's own struct demands is gone. This is the class of defect,
    // detected without knowing the instance.
    std::fs::write(
        root.join("ops/queue-keeper-v3.json"),
        r#"{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":1785251597154,
                "admission_allowed":true,"queued_hard_cap":4096,"queued_starting":3,
                "dispatching":1,"admission_waves":[],
                "unclaimed_current_checkpoint_manifests":7}"#,
    )
    .unwrap();
    let alarms = scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "contract_violation:ops/queue-keeper-v3.json#queued_target")
        .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
    assert_eq!(row["observed"], "missing");
    assert!(row["threshold"]
        .as_str()
        .unwrap()
        .contains("src/bin/jope_selfplay_supply.rs"));
}

#[test]
fn a_wrong_typed_required_field_is_a_violation_too() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(
        root.join("ops/queue-keeper-v3.json"),
        r#"{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":1785251597154,
                "admission_allowed":true,"queued_target":"2048","queued_hard_cap":4096,
                "queued_starting":3,"dispatching":1,"admission_waves":[],
                "unclaimed_current_checkpoint_manifests":7}"#,
    )
    .unwrap();
    let alarms = scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "contract_violation:ops/queue-keeper-v3.json#queued_target")
        .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
    assert_eq!(row["observed"], "expected number, found string");
}

/// A keeper document carrying every field the supply consumer's
/// `QueueStatus` requires. Written out by hand, NOT generated from the
/// derived field list, so an over-strict extractor still fails
/// `a_complete_in_schema_document_is_silent` instead of agreeing with
/// itself. Its values also pass every aggregate check in the consumer's
/// `load_queue_status` (jope_selfplay_supply.rs): queued + starting =
/// queued_starting, reserve + admission manifests = unclaimed manifests,
/// complete_reserve_waves = protected_reserve_waves.len(), useful_work =
/// queued + starting + dispatching + validated_ready, and
/// candidate_evaluation_work >= useful_work, non-zero with it.
fn complete_keeper_document(generated_at_ms: u64) -> String {
    format!(
        r#"{{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":{generated_at_ms},
               "admission_allowed":true,"active_checkpoint_sha256":null,
               "queued_target":2048,"queued_hard_cap":4096,"queued_starting":3,
               "queued":2,"starting":1,"dispatching":1,"admission_waves":[],
               "protected_reserve_waves":[],"complete_current_checkpoint_waves":0,
               "partial_current_checkpoint_waves":0,"complete_reserve_waves":0,
               "reserve_manifests":0,"admission_manifests":0,"validated_ready":7,
               "useful_work":11,"candidate_evaluation_work":11,
               "unclaimed_current_checkpoint_manifests":0}}"#
    )
}

#[test]
fn a_complete_in_schema_document_is_silent() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(
        root.join("ops/queue-keeper-v3.json"),
        complete_keeper_document(now_ms() as u64),
    )
    .unwrap();
    let names = alarm_names(&scan(&root));
    assert!(names.is_empty(), "{names:?}");
}

/// "Complete" above is relative to `QueueStatus` as it stands today. When
/// the consumer gains a required field, this names the field the fixture
/// lacks instead of leaving an unexplained alarm in the test above.
#[test]
fn the_complete_keeper_fixture_covers_every_required_queue_status_field() {
    let fixture: Value = serde_json::from_str(&complete_keeper_document(0)).unwrap();
    let keys = fixture.as_object().unwrap();
    let source = live_source("src/bin/jope_selfplay_supply.rs");
    let (_, required) = derive_required_fields(&source, "QueueStatus").unwrap();
    assert!(!required.is_empty());
    let missing: Vec<&str> = required
        .iter()
        .map(|field| field.name.as_str())
        .filter(|name| !keys.contains_key(*name))
        .collect();
    assert!(
        missing.is_empty(),
        "complete_keeper_document lacks required QueueStatus fields {missing:?}"
    );
}

#[test]
fn a_fresh_file_whose_own_timestamp_is_frozen_is_not_healthy() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    // Written right now (mtime is seconds old, so check_status_ages is
    // happy) but the daemon stopped advancing generated_at_ms 4 h ago.
    let stale = now_ms() as u64 - 4 * 3_600 * 1_000;
    std::fs::write(
        root.join("ops/queue-keeper-v3.json"),
        format!(
            r#"{{"schema":"JopeSelfplayQueueKeeper.v3","generated_at_ms":{stale},
                   "admission_allowed":true,"queued_target":2048,"queued_hard_cap":4096,
                   "queued_starting":3,"dispatching":1,"admission_waves":[],
                   "unclaimed_current_checkpoint_manifests":7}}"#
        ),
    )
    .unwrap();
    let names = alarm_names(&scan(&root));
    assert!(
        names.contains(&"status_content_frozen:ops/queue-keeper-v3.json".to_owned()),
        "{names:?}"
    );
}

#[test]
fn an_in_schema_degraded_document_still_alarms() {
    let temp = TempDir::new().unwrap();
    let root = contract_root(&temp);
    std::fs::write(
        root.join("ops/task-learning-controller.v2.json"),
        format!(
            r#"{{"schema":"TaskLearningControllerStatus.v2","generated_at_ms":{},
                   "degraded":true,"error":"scq unreachable","action":"cycle_failed"}}"#,
            now_ms() as u64
        ),
    )
    .unwrap();
    let alarms = scan(&root);
    let row = alarms
        .rows
        .iter()
        .find(|row| row["alarm"] == "status_degraded:ops/task-learning-controller.v2.json")
        .unwrap_or_else(|| panic!("{:?}", alarm_names(&alarms)));
    assert_eq!(row["observed"], "scq unreachable");
}

// -----------------------------------------------------------------------
// Anti-drift: the required-field set is read out of the consumer's own
// source, so these tests fail LOCALLY (at `cargo test`) the moment the
// extractor stops understanding a struct it is enforcing.
// -----------------------------------------------------------------------

fn live_source(relative: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(relative)).unwrap()
}

#[test]
fn the_supply_queue_status_contract_is_derived_from_supplys_own_struct() {
    let source = live_source("src/bin/jope_selfplay_supply.rs");
    let (line, required) = derive_required_fields(&source, "QueueStatus").unwrap();
    assert!(line > 0);
    let by_name: Vec<&str> = required.iter().map(|f| f.name.as_str()).collect();
    // The exact field whose absence killed supply 5,893 times.
    assert!(by_name.contains(&"queued_target"), "{by_name:?}");
    assert_eq!(
        required
            .iter()
            .find(|f| f.name == "queued_target")
            .unwrap()
            .kind,
        Kind::Num
    );
    assert!(by_name.contains(&"admission_waves"));
    assert_eq!(
        required
            .iter()
            .find(|f| f.name == "admission_waves")
            .unwrap()
            .kind,
        Kind::Arr
    );
    assert!(by_name.contains(&"admission_allowed"));
    assert_eq!(
        required
            .iter()
            .find(|f| f.name == "admission_allowed")
            .unwrap()
            .kind,
        Kind::Bool
    );
    // Option fields are NOT required: serde accepts them absent.
    assert!(
        !by_name.contains(&"active_checkpoint_sha256"),
        "{by_name:?}"
    );
    assert!(!by_name.contains(&"checkpoint_epoch"), "{by_name:?}");
}

#[test]
fn a_container_level_serde_default_makes_every_field_optional() {
    // Both DurableState structs carry #[serde(default)], which is the only
    // reason jope_selfplay_freshness.rs:328 and
    // jope_selfplay_trainer_watchdog.rs:691 survive a wrong-shaped
    // document. The contract must agree with serde, not with intent.
    for relative in [
        "src/bin/jope_selfplay_freshness.rs",
        "src/bin/jope_selfplay_trainer_watchdog.rs",
    ] {
        let source = live_source(relative);
        let (_, required) = derive_required_fields(&source, "DurableState").unwrap();
        assert!(required.is_empty(), "{relative}: {required:?}");
    }
}

#[test]
fn every_contract_names_a_consumer_source_that_exists() {
    for contract in OPS_CONTRACTS {
        let Some((relative, struct_name)) = contract.consumer else {
            continue;
        };
        let source = live_source(relative);
        assert!(
            derive_required_fields(&source, struct_name).is_some(),
            "{relative} no longer defines struct {struct_name}"
        );
    }
}

#[test]
fn the_extractor_handles_renames_skips_and_defaults() {
    let source = r#"
#[derive(Deserialize)]
struct Sample {
    schema: String,
    #[serde(rename = "queued_target")]
    target: i64,
    #[serde(default)]
    optional_count: usize,
    maybe: Option<String>,
    #[serde(skip)]
    internal: usize,
    waves: Vec<String>,
    index: BTreeMap<String, i64>,
    nested: SomeOtherStruct,
}
"#;
    let (_, required) = derive_required_fields(source, "Sample").unwrap();
    let names: Vec<&str> = required.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(
        names,
        vec!["schema", "queued_target", "waves", "index", "nested"]
    );
    assert_eq!(required[4].kind, Kind::Any);
    assert_eq!(required[3].kind, Kind::Obj);
}

/// Builds <temp>/attempts/0/outputs/training/wm/{train_progress.json,
/// train_metrics.jsonl} the way a real trainer run lays them out.
fn run_dir_with_metrics(temp: &TempDir, name: &str, rows: &[&str]) -> PathBuf {
    let run_dir = temp.path().join(name);
    let outputs = run_dir.join("attempts/0/outputs/training/watermark-000000001");
    std::fs::create_dir_all(&outputs).unwrap();
    std::fs::write(outputs.join("train_progress.json"), "{\"phase\":\"B\"}").unwrap();
    std::fs::write(outputs.join("train_metrics.jsonl"), rows.join("\n") + "\n").unwrap();
    run_dir
}

fn alarm_names(alarms: &Alarms) -> Vec<String> {
    alarms
        .rows
        .iter()
        .map(|row| row["alarm"].as_str().unwrap().to_owned())
        .collect()
}

/// `check_training_history` also consults the live scq snapshot, so a real
/// completed trainer on this cluster can add `training_history_run_absent:*`
/// rows that have nothing to do with the temp root under test. Tests that
/// assert on an exact alarm set look only at the run they created.
fn alarms_for(alarms: &Alarms, run_id: &str) -> Vec<Value> {
    alarms
        .rows
        .iter()
        .filter(|row| {
            row["alarm"]
                .as_str()
                .map(|name| name.ends_with(&format!(":{run_id}")))
                .unwrap_or(false)
        })
        .cloned()
        .collect()
}
