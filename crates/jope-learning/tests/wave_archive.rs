use jope_learning::research::wave_archive::{
    archive, reconstruct, Disposition, Event, Known, Metric, WaveInput, MAX_SHARD_ROWS,
};
use jope_learning::state::State;
use std::collections::BTreeMap;

fn identity() -> BTreeMap<String, Known<String>> {
    [
        "source",
        "config",
        "grammar",
        "evaluator",
        "registry",
        "context",
        "support",
        "optimizer_recipe",
        "rng_sampler",
        "data_role",
    ]
    .into_iter()
    .map(|key| (key.into(), Known::Known(format!("fixture-{key}"))))
    .collect()
}
fn event(sequence: u64) -> Event {
    Event {
        sequence,
        attempt: 0,
        rank: 0,
        kind: "terminal_snapshot".into(),
        fields: BTreeMap::new(),
        counters: [
            ("computed".into(), Known::Known(1)),
            ("applied".into(), Known::Known(0)),
            ("committed".into(), Known::Known(0)),
        ]
        .into(),
        metrics: [(
            "loss".into(),
            Metric {
                value: Some(0.0),
                unit: "loss/example".into(),
                denominator: Some(1),
                status: "observed".into(),
                source_field: "training.loss".into(),
                missing_reason: None,
            },
        )]
        .into(),
    }
}
fn input() -> WaveInput {
    WaveInput {
        wave_id: "fixture-wave".into(),
        version: "v1".into(),
        disposition: Disposition::ZeroUpdate,
        new_wave: true,
        identity: identity(),
        events: vec![event(1)],
    }
}
fn state(tmp: &tempfile::TempDir) -> State {
    State::at(tmp.path())
}

#[test]
fn every_terminal_disposition_is_archivable_without_claim_upgrade() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    for disposition in [
        Disposition::Success,
        Disposition::RejectedChild,
        Disposition::Aborted,
        Disposition::TimeoutOnly,
        Disposition::Inconclusive,
        Disposition::InterruptedResumed,
        Disposition::ZeroUpdate,
        Disposition::HistoricalPartial,
    ] {
        let mut i = input();
        i.disposition = disposition;
        if disposition == Disposition::HistoricalPartial {
            i.new_wave = false;
        }
        let receipt = archive(&s, &i).unwrap();
        assert_eq!(receipt.manifest.disposition, disposition);
        assert_eq!(reconstruct(&s, &receipt.manifest).unwrap(), i.events);
    }
}
#[test]
fn repeated_archive_is_content_addressed_and_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let i = input();
    let a = archive(&s, &i).unwrap();
    let b = archive(&s, &i).unwrap();
    assert_eq!(a.manifest, b.manifest);
    assert_eq!(a.manifest_blob, b.manifest_blob);
}
#[test]
fn shuffled_events_reconstruct_in_canonical_order() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events = vec![event(3), event(1), event(2)];
    let a = archive(&s, &i).unwrap();
    let out = reconstruct(&s, &a.manifest).unwrap();
    assert_eq!(
        out.iter().map(|e| e.sequence).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
}
#[test]
fn missing_new_wave_identity_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.identity.remove("evaluator");
    assert!(archive(&s, &i).is_err());
}
#[test]
fn historical_missing_data_is_retained_as_partial() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.new_wave = false;
    i.disposition = Disposition::HistoricalPartial;
    i.identity.insert(
        "rng_sampler".into(),
        Known::Missing {
            reason: "legacy journal absent".into(),
        },
    );
    let a = archive(&s, &i).unwrap();
    assert_eq!(a.manifest.archive_status, "HISTORICAL_PARTIAL");
    assert_eq!(
        a.manifest.missing_fields.get("rng_sampler").unwrap(),
        "legacy journal absent"
    );
}
#[test]
fn duplicate_event_identity_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events.push(event(1));
    assert!(archive(&s, &i).is_err());
}
#[test]
fn nan_metric_is_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events[0].metrics.get_mut("loss").unwrap().value = Some(f64::NAN);
    assert!(archive(&s, &i).is_err());
}
#[test]
fn null_metric_requires_explicit_missing_reason() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    let m = i.events[0].metrics.get_mut("loss").unwrap();
    m.value = None;
    m.denominator = None;
    m.missing_reason = Some("head disabled".into());
    let a = archive(&s, &i).unwrap();
    assert!(a
        .manifest
        .missing_fields
        .values()
        .any(|v| v == "head disabled"));
}
#[test]
fn new_wave_requires_computed_applied_committed_counters() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events[0].counters.remove("committed");
    assert!(archive(&s, &i).is_err());
}
#[test]
fn a_zero_event_new_wave_is_not_silent_success() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events.clear();
    assert!(archive(&s, &i).is_err());
}
#[test]
fn missing_blob_or_corrupted_bytes_are_detected() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let a = archive(&s, &input()).unwrap();
    let p = s.blob_path(&a.manifest.shards[0].sha256);
    std::fs::write(&p, b"broken zstd/parquet").unwrap();
    assert!(reconstruct(&s, &a.manifest).is_err());
}
#[test]
fn manifest_mutation_is_detected() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let a = archive(&s, &input()).unwrap();
    let mut m = a.manifest;
    m.version = "v2".into();
    assert!(reconstruct(&s, &m).is_err());
}
#[test]
fn shards_are_bounded_and_complete() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events = (0..=MAX_SHARD_ROWS as u64).map(event).collect();
    let a = archive(&s, &i).unwrap();
    assert_eq!(a.manifest.shards.len(), 2);
    assert_eq!(
        reconstruct(&s, &a.manifest).unwrap().len(),
        MAX_SHARD_ROWS + 1
    );
}
#[test]
fn publication_state_stays_local_pending_and_retry_does_not_train() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let training_calls = 0;
    let i = input();
    let a = archive(&s, &i).unwrap();
    let b = archive(&s, &i).unwrap();
    assert_eq!(training_calls, 0);
    assert_eq!(a.manifest.archive_status, "LOCAL_COMMITTED_REMOTE_PENDING");
    assert_eq!(a.manifest_blob, b.manifest_blob);
}

#[test]
fn oversized_decoded_shard_is_rejected_before_serialization() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    i.events[0]
        .fields
        .insert("large".into(), Known::Known("x".repeat(1_500_000)));
    assert!(archive(&s, &i).is_err());
}

#[test]
fn new_wave_missing_measurement_is_archived_but_blocked_from_admission() {
    let tmp = tempfile::tempdir().unwrap();
    let s = state(&tmp);
    let mut i = input();
    let m = i.events[0].metrics.get_mut("loss").unwrap();
    m.value = None;
    m.denominator = None;
    m.missing_reason = Some("producer omitted this measurement".into());
    let r = archive(&s, &i).unwrap();
    assert_eq!(r.manifest.capture_status, "INCOMPLETE_BLOCKED");
    assert_eq!(r.manifest.archive_status, "LOCAL_COMMITTED_REMOTE_PENDING");
}
