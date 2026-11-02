use super::*;
use crate::manifest::RunMetadata;
use std::fs;
use std::thread;

#[test]
fn producer_is_nonblocking_bounded_and_counted() {
    let dir = tempfile::tempdir().unwrap();
    let mut b = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default())
        .with_queue_capacity(1)
        .unwrap();
    b.add_producer("test").unwrap();
    let (recorder, mut producers) = b.start().unwrap();
    let producer = &mut producers[0];
    assert!(producer.emit_scalar(crate::train::TRAIN_LOSS, 0, 0, 1.0, false));
    assert!(!producer.emit_scalar(crate::train::TRAIN_LOSS, 0, 1, 0.9, false));
    // Allow the one drain thread to consume the event before shutdown.
    thread::sleep(Duration::from_millis(10));
    let manifest = recorder.close().unwrap();
    assert_eq!(manifest.event_drops, 1);
    assert_eq!(manifest.events_written, 1);
    assert!(dir.path().join("run/events.jsonl").exists());
}

#[test]
fn record_lane_is_bounded_and_reports_oversize() {
    let dir = tempfile::tempdir().unwrap();
    let mut b = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default());
    b.add_producer("test").unwrap();
    let (recorder, _) = b.start().unwrap();
    let too_large = serde_json::json!({"payload": "x".repeat(MAX_RECORD_BYTES)});
    assert!(!recorder.record_value("oversize/v1", too_large));
    let manifest = recorder.close().unwrap();
    assert_eq!(manifest.oversized_record_drops, 1);
    assert_eq!(manifest.record_drops, 1);
    let text = fs::read_to_string(dir.path().join("run/events.jsonl")).unwrap();
    assert!(text.is_empty());
}
#[test]
fn interrupt_finalizes_manifest_as_interrupted() {
    let dir = tempfile::tempdir().unwrap();
    let mut builder = RecorderBuilder::new(dir.path().join("run"), RunMetadata::default());
    builder.add_producer("test").unwrap();
    let (recorder, mut producers) = builder.start().unwrap();
    assert!(producers[0].emit_scalar(crate::train::TRAIN_LOSS, 0, 0, 1.0, false));

    let manifest = recorder.interrupt().unwrap();

    assert!(matches!(manifest.status, RunStatus::Interrupted));
    #[cfg(feature = "parquet")]
    assert!(dir.path().join("run/metrics.parquet").exists());
}
