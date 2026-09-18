use jain_smartcluster_daemon::event_log::{EventLog, EventLogOptions};

#[test]
#[ignore = "scheduled crash/reopen soak lane"]
fn repeated_event_log_recovery_keeps_monotonic_offsets() {
    let iterations = std::env::var("SCQ_SOAK_ITERATIONS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(1_000);
    let directory = tempfile::tempdir().expect("tempdir");
    for offset in 0..iterations {
        let mut log = EventLog::<u64>::open(directory.path(), EventLogOptions::default())
            .expect("recover log");
        assert_eq!(log.next_offset(), offset);
        assert_eq!(log.append(&offset).expect("append"), offset);
        drop(log);
    }
    let log = EventLog::<u64>::open(directory.path(), EventLogOptions::default())
        .expect("final recovery");
    assert_eq!(log.next_offset(), iterations);
    assert_eq!(log.recovery_report().truncated_segments, 0);
    println!(
        "JAIN_RELEASE_METRIC {}",
        serde_json::json!({
            "schema_version": "jain.smartcluster.release-metric/v1",
            "metric": "event_log_recovery_iterations",
            "observed": iterations,
            "limit": 1_000,
            "comparison": "gte",
            "unit": "iterations",
            "sample_count": iterations,
        })
    );
}
