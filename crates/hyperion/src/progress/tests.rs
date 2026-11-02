use super::*;

#[test]
fn jsonl_sink_round_trips_an_event() {
    // The SDK parses these lines; prove a representative Event serializes to one JSON line and
    // deserializes back field-for-field (sparse: None fields are absent).
    let mut e = Event::new(5, "model", 41.7);
    e.done_frac = Some(0.5);
    e.eta = Some(12.5);
    e.best_val_auc = Some(0.71234);
    e.best_holdout_auc = Some(0.70011);
    e.best_size = Some(19);
    e.model = Some("catboost".to_string());
    e.metric = Some("auc".to_string());
    e.rows = Some(100);
    e.cols = Some(12);
    e.output_dim = Some(128);
    e.hyperion_status = Some("used".to_string());
    e.hyperion_engine = Some("v7".to_string());
    e.hyperion_device = Some("cpu".to_string());
    e.hyperion_preview = Some(HyperionPreview {
        rows: 100,
        threshold_z: 3.5,
        points: vec![HyperionPreviewPoint {
            row: 42,
            score: 12.5,
        }],
        top_rows: vec![HyperionConcernRow {
            row: 42,
            score: 12.5,
            z: 4.1,
        }],
        concern_ranges: vec![HyperionConcernRange {
            start: 41,
            end: 43,
            max_score: 12.5,
            max_z: 4.1,
        }],
    });

    let buf = std::sync::Arc::new(Mutex::new(Vec::<u8>::new()));
    struct SharedWriter(std::sync::Arc<Mutex<Vec<u8>>>);
    impl Write for SharedWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let sink = JsonlSink::new(Box::new(SharedWriter(buf.clone())));
    sink.emit(&e);

    let out = String::from_utf8(buf.lock().unwrap().clone()).unwrap();
    assert!(out.ends_with('\n'), "each event is a line");
    assert_eq!(out.lines().count(), 1);
    let back = Event::from_json_line(out.trim()).unwrap();
    assert_eq!(back.phase, 5);
    assert_eq!(back.event, "model");
    assert_eq!(back.t, 41.7);
    assert_eq!(back.done_frac, Some(0.5));
    assert_eq!(back.eta, Some(12.5));
    assert_eq!(back.best_val_auc, Some(0.71234));
    assert_eq!(back.best_holdout_auc, Some(0.70011));
    assert_eq!(back.best_size, Some(19));
    assert_eq!(back.model.as_deref(), Some("catboost"));
    assert_eq!(back.v, EVENT_SCHEMA_VERSION);
    assert_eq!(back.metric.as_deref(), Some("auc"));
    assert_eq!(back.rows, Some(100));
    assert_eq!(back.cols, Some(12));
    assert_eq!(back.output_dim, Some(128));
    assert_eq!(back.hyperion_status.as_deref(), Some("used"));
    assert_eq!(back.hyperion_engine.as_deref(), Some("v7"));
    assert_eq!(back.hyperion_device.as_deref(), Some("cpu"));
    assert_eq!(
        back.hyperion_preview.as_ref().map(|p| p.points.len()),
        Some(1)
    );
    // Track the constant instead of a hard-coded number so schema bumps can't strand this test.
    assert!(
        out.contains(&format!("\"v\":{EVENT_SCHEMA_VERSION}")),
        "schema version always serialized"
    );
}

fn context() -> RunContext {
    RunContext::new("bench")
        .dataset("demo.csv")
        .task("Binary")
        .rows(100, 25)
        .features(12)
        .selection_mode("mvp-fast")
}

#[test]
fn human_progress_formats_phase2_mvp_and_final_events() {
    let ctx = context();
    let mut state = HumanProgressState::default();

    let mut e = Event::new(2, "block", 3.0);
    e.done = Some(4);
    e.blocks_total = Some(10);
    e.done_frac = Some(0.4);
    e.eta = Some(6.0);
    let line = state
        .render(&e, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
        .unwrap();
    assert!(line.contains("blocks=4/10"));
    assert!(line.contains("eta=6.0s"));

    let mut e = Event::new(3, "done", 5.0);
    e.model = Some("mvp-fast".into());
    e.accepted_features = Some(20);
    e.selected_features = Some(7);
    e.final_features = Some(19);
    let line = state
        .render(&e, &ctx, "SELECT", DEFAULT_THROTTLE_SECS)
        .unwrap();
    assert!(line.contains("accepted=20"));
    assert!(line.contains("selected=7"));
    assert!(line.contains("final_features=19"));

    let mut e = Event::new(5, "model", 6.0);
    e.model = Some("catboost".into());
    e.best_val_auc = Some(0.71);
    let line = state
        .render(&e, &ctx, "FINAL", DEFAULT_THROTTLE_SECS)
        .unwrap();
    assert!(line.contains("[FINAL] candidate"));
    assert!(line.contains("model=catboost"));
}

#[test]
fn progress_v12_rejects_retired_phase_and_prior_schema() {
    let phase_one = r#"{"phase":1,"event":"start","t":0.0,"v":12}"#;
    let error = Event::from_json_line(phase_one).unwrap_err();
    assert!(error.contains("phase 1 was retired"));
    assert!(error.contains("retrain with Jain 10.0.0"));

    let v11 = r#"{"phase":2,"event":"start","t":0.0,"v":11}"#;
    let error = Event::from_json_line(v11).unwrap_err();
    assert!(error.contains("expected v12"));
    assert!(error.contains("retrain with Jain 10.0.0"));
}

#[test]
fn human_progress_throttles_repetitive_progress_but_not_done() {
    let ctx = context();
    let mut state = HumanProgressState::default();
    let mut e = Event::new(2, "block", 1.0);
    e.done = Some(1);
    e.blocks_total = Some(10);
    assert!(state
        .render(&e, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
        .is_some());

    let mut e2 = Event::new(2, "block", 1.5);
    e2.done = Some(2);
    e2.blocks_total = Some(10);
    assert!(state
        .render(&e2, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
        .is_none());

    let mut done = Event::new(2, "done", 1.6);
    done.done = Some(10);
    done.blocks_total = Some(10);
    assert!(state
        .render(&done, &ctx, "IMPORT", DEFAULT_THROTTLE_SECS)
        .is_some());
}

#[test]
fn human_seed_line_carries_the_count_but_never_the_genome() {
    let ctx = context();
    let mut state = HumanProgressState::default();
    let genome = "(pipe X (robust_scale) (histgbm depth=3 lr=0.1 rounds=40))";
    let mut e = Event::new(6, "seed", 2.0);
    e.genome = Some(genome.to_string());
    e.gen = Some(2);
    e.model = Some("jope".to_string());
    let line = state
        .render(&e, &ctx, "INVENT", DEFAULT_THROTTLE_SECS)
        .expect("seed events render immediately");
    assert!(line.contains("jope inference · seed #3"), "{line}");
    assert!(!line.contains(genome), "{line}");
    for token in ["pipe", "histgbm", "robust_scale", "("] {
        assert!(!line.contains(token), "genome text leaked: {line}");
    }
}
