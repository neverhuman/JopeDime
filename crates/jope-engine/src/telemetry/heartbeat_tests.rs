use super::*;
#[test]
fn expired_or_changed_process_activity_is_unknown_but_history_survives() {
    let mut value = serde_json::json!({"phase":"running","active":104,
            "timestamp":100,"expires_at":110,"last_progress_at":100,"completed_views":50801,
            "completed_evaluations":857000,"evaluations_per_second":158.1});
    value["status_binding"] = status_binding(&value).unwrap();
    assert_eq!(observed_status(&value, 109)["current_activity"], "current");
    for observed in [99, 110, 1000] {
        let stale = observed_status(&value, observed);
        assert_eq!(stale["current_activity"], "stale");
        assert!(stale["active"].is_null());
        assert_eq!(stale["completed_views"], 50801);
        assert_eq!(stale["last_recorded_activity"]["active"], 104);
        assert_eq!(observed_status(&stale, observed), stale);
    }
    for field in ["birth_ticks", "pid"] {
        let mut reused = value.clone();
        reused["status_binding"]["process"][field] = 1.into();
        assert_eq!(observed_status(&reused, 105)["current_activity"], "stale");
    }
    let mut replaced = value.clone();
    replaced["status_binding"]["executable"]["inode"] = 0.into();
    assert_eq!(observed_status(&replaced, 105)["current_activity"], "stale");
    let mut stalled = value.clone();
    stalled["timestamp"] = 200.into();
    stalled["expires_at"] = 210.into();
    assert_eq!(observed_status(&stalled, 205)["current_activity"], "stale");
    value.as_object_mut().unwrap().remove("status_binding");
    assert_eq!(observed_status(&value, 105)["current_activity"], "stale");
}

#[test]
fn publisher_preserves_generation_and_records_terminal_state() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("status.json");
    let publisher = StageHeartbeat::with_interval(
        &path,
        serde_json::json!({"phase":"running"}),
        Duration::from_millis(10),
    )
    .unwrap();
    let original: serde_json::Value = read_json(&path).unwrap();
    publisher
        .publish(serde_json::json!({"phase":"completed","completed_views":42,
            "status_binding":{"controller_generation":"forged"}}))
        .unwrap();
    publisher.finish().unwrap();
    let saved: serde_json::Value = read_json(&path).unwrap();
    assert_eq!(saved["status_binding"], original["status_binding"]);
    assert_eq!(saved["completed_views"], 42);
    assert_eq!(saved["terminal"], true);
    assert_eq!(saved["termination_reason"], "completed");
    assert_eq!(
        observed_status(&saved, now() + 100)["current_activity"],
        "terminal"
    );
    let publisher = StageHeartbeat::start(&path, serde_json::json!({"phase":"running"})).unwrap();
    drop(publisher);
    let saved: serde_json::Value = read_json(&path).unwrap();
    assert_eq!(saved["terminal"], true);
    assert!(saved["termination_reason"]
        .as_str()
        .unwrap()
        .contains("reconcile"));
}

#[test]
fn slow_stages_keep_liveness_separate_and_join_before_next_phase() {
    for phase in ["Fit and save predictors", "Pack newly scored programs"] {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("progress.json");
        let payload = dir.path().join("payload");
        fs::write(&payload, b"frozen fitted predictor or corpus").unwrap();
        let original = file_hash(&payload).unwrap();
        let stage = StageHeartbeat::with_interval(
            &path,
            serde_json::json!({"phase":phase}),
            Duration::from_millis(10),
        )
        .unwrap();
        let initial: serde_json::Value = read_json(&path).unwrap();
        let started = std::time::Instant::now();
        loop {
            let value: serde_json::Value = read_json(&path).unwrap();
            if value["heartbeat_sequence"].as_u64().unwrap() >= 2 {
                assert_eq!(value["phase"], phase);
                assert_eq!(value["last_progress_at"], initial["last_progress_at"]);
                assert!(value["work"].is_null());
                break;
            }
            assert!(started.elapsed() < Duration::from_secs(2));
            std::thread::sleep(Duration::from_millis(5));
        }
        stage.advance("published artifacts", 1, Some(2)).unwrap();
        stage.finish().unwrap();
        let value: serde_json::Value = read_json(&path).unwrap();
        assert_eq!(value["work"]["completed"], 1);
        assert_eq!(value["heartbeat_role"], "controller_liveness");
        write_json(&path, &serde_json::json!({"phase":"next"})).unwrap();
        std::thread::sleep(Duration::from_millis(40));
        assert_eq!(
            read_json::<serde_json::Value>(&path).unwrap()["phase"],
            "next"
        );
        assert_eq!(file_hash(&payload).unwrap(), original);
    }
}
#[test]
fn cancellation_joins_publisher_and_write_errors_are_reported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("progress.json");
    let stage = StageHeartbeat::start(&path, serde_json::json!({"phase":"packing"})).unwrap();
    drop(stage); // Same cleanup as early return or unwinding in a stage.
    let saved = fs::read(&path).unwrap();
    std::thread::sleep(Duration::from_millis(30));
    assert_eq!(fs::read(&path).unwrap(), saved);
    let stage = StageHeartbeat::start(&path, serde_json::json!({})).unwrap();
    fs::remove_file(&path).unwrap();
    fs::create_dir(&path).unwrap();
    assert!(stage.finish().is_err());
}
