use super::*;

use super::*;
use crate::gp::test_gen_summary;

fn gen(context: &str, g: u32, unique: f64) -> GenSummary {
    test_gen_summary(context, g, unique)
}

#[test]
fn collapse_needs_consecutive_generations_per_context() {
    let mut e = Engine::new("r", Thresholds::default());
    assert!(e.observe_generation(&gen("a", 0, 0.2), 0.0).is_none());
    assert!(e.observe_generation(&gen("b", 0, 0.2), 0.0).is_none());
    assert!(e.observe_generation(&gen("a", 1, 0.2), 0.0).is_none());
    assert!(
        e.observe_generation(&gen("a", 2, 0.9), 0.0).is_none(),
        "streak resets"
    );
    assert!(e.observe_generation(&gen("a", 3, 0.2), 0.0).is_none());
    assert!(e.observe_generation(&gen("a", 4, 0.2), 0.0).is_none());
    let a = e.observe_generation(&gen("a", 5, 0.2), 0.0).unwrap();
    assert_eq!(a.id, AlarmId::A2GpCollapse);
    assert_eq!(a.severity, Severity::Alarm);
}

#[test]
fn dope_guard_fails_closed() {
    let mut e = Engine::new("r", Thresholds::default());
    let ok = DopeGuardObservation {
        hashes_equal: Some(true),
        trainable_jope_params: Some(5),
        ..Default::default()
    };
    assert!(e.observe_dope_guard(&ok, 0.0).is_none());
    let bad = DopeGuardObservation {
        hashes_equal: Some(false),
        changed_tensors: vec!["dope_model.w".into()],
        trainable_jope_params: Some(5),
        ..Default::default()
    };
    let a = e.observe_dope_guard(&bad, 0.0).unwrap();
    assert_eq!(a.severity, Severity::FailClosed);
    assert!(a.reasons[0].contains("dope_model.w"));
}

#[test]
fn grad_spike_needs_consecutive_steps_and_nonfinite_fires_at_once() {
    let mut e = Engine::new("r", Thresholds::default());
    for s in 0..100 {
        assert!(e.observe_step(s, 1.0, Some(1.0), s as f64).is_none());
    }
    for s in 100..104 {
        assert!(e.observe_step(s, 1.0, Some(50.0), s as f64).is_none());
    }
    assert!(e.observe_step(104, 1.0, Some(50.0), 104.0).is_some());
    assert!(e.observe_step(105, f64::NAN, Some(1.0), 105.0).is_some());
}

#[test]
fn heartbeat_uses_median_step_time_with_floor() {
    let mut e = Engine::new("r", Thresholds::default());
    for s in 0..10 {
        e.observe_step(s, 1.0, None, s as f64);
    }
    assert!(e.heartbeat(60.0).is_none());
    assert!(e.heartbeat(80.0).is_some());
}

#[test]
fn utc_format() {
    let s = utc_now_rfc3339();
    assert_eq!(s.len(), 20);
    assert!(s.ends_with('Z') && s.as_bytes()[10] == b'T');
}
