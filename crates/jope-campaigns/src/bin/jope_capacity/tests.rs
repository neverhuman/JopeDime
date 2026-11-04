use super::*;
use jope_dime_regressor::selfplay_core::{unsafe_sample, RAMP_MID_DURATION_MS};
use tempfile::TempDir;

fn policy() -> &'static str {
    r#"
poll_ms = 2000
telemetry_ms = 1000
[[node]]
name = "xbabe1"
[[node]]
name = "xbabe2"
cpu_job_slot_multiplier = 1.0
[[node]]
name = "xbabe3"
[[node]]
name = "xbabe4"
cpu_job_slot_multiplier = 2.0
"#
}

fn pressure(cpu: f64) -> StoredPressure {
    StoredPressure {
        cpu_psi: cpu,
        memory_psi: 0.0,
        io_psi: 0.0,
        available_memory_fraction: Some(0.5),
        failure_rate: 0.0,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    }
}

fn sample(at: u128, cpu: f64, cluster_cpu: f64) -> StoredSample {
    StoredSample {
        sampled_at_ms: at,
        cluster_cpu_fraction: cluster_cpu,
        xbabe4: pressure(cpu),
        cluster: pressure(cpu),
    }
}

#[test]
fn policy_defaults_other_nodes_to_one_and_pins_xbabe4() {
    let parsed = parse_policy(policy()).unwrap();
    validate_policy(&parsed).unwrap();
    assert_eq!(parsed.multipliers["xbabe1"], 1.0);
    assert_eq!(parsed.multipliers["xbabe4"], 2.0);
}

#[test]
fn minute_windows_break_on_dense_or_missing_samples() {
    let valid = vec![sample(0, 0.0, 0.5), sample(60_000, 0.0, 0.5)];
    assert_eq!(trailing_minute_samples(&valid).len(), 2);
    let dense = vec![sample(0, 0.0, 0.5), sample(1_000, 0.0, 0.5)];
    assert_eq!(trailing_minute_samples(&dense).len(), 1);
    let missing = vec![sample(0, 0.0, 0.5), sample(120_000, 0.0, 0.5)];
    assert_eq!(trailing_minute_samples(&missing).len(), 1);
}

#[test]
fn exactly_three_minute_samples_trigger_rollback() {
    let baseline = PressureBaseline {
        failure_rate: 0.0,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    };
    let mut state = RampState::baseline(20, baseline).unwrap();
    state = ramp_decision(&state, &[], 0).unwrap();
    state = ramp_decision(&state, &[], RAMP_MID_DURATION_MS).unwrap();
    let samples = [
        sample(0, 0.25, 0.5),
        sample(60_000, 0.25, 0.5),
        sample(120_000, 0.25, 0.5),
    ];
    let two = samples[..2]
        .iter()
        .map(|row| row.xbabe4.core())
        .collect::<Vec<_>>();
    assert_eq!(
        ramp_decision(&state, &two, 180_000).unwrap().stage,
        RampStage::Target40
    );
    let three = samples
        .iter()
        .map(|row| row.xbabe4.core())
        .collect::<Vec<_>>();
    let rolled = ramp_decision(&state, &three, 180_000).unwrap();
    assert_eq!(rolled.stage, RampStage::RolledBack);
    assert_eq!(rolled.multiplier, 1.0);
}

#[test]
fn xbabe2_requires_fifteen_clear_minutes_and_fresh_work() {
    let baseline = PressureBaseline {
        failure_rate: 0.0,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    };
    let samples = (0..15)
        .map(|minute| {
            let row = sample(minute * 60_000, 0.0, 0.74);
            BurnSample {
                cluster_cpu_fraction: row.cluster_cpu_fraction,
                pressure: row.cluster.core(),
                baseline,
            }
        })
        .collect::<Vec<_>>();
    assert!(xbabe2_burn_eligible(&samples, 1).unwrap());
    assert!(!xbabe2_burn_eligible(&samples, 0).unwrap());
}

#[test]
fn receipts_are_content_addressed_and_idempotent() {
    let temporary = TempDir::new().unwrap();
    let decision = json!({"schema": "test", "slot_target": 30});
    let first = write_receipt(temporary.path(), "same-key", &decision).unwrap();
    let replay = replay_receipt(temporary.path(), "same-key")
        .unwrap()
        .unwrap();
    assert_eq!(first, replay);
    let (receipts, _) = receipt_dirs(temporary.path());
    let files = fs::read_dir(receipts).unwrap().count();
    assert_eq!(files, 1);
}

#[test]
fn all_gate_boundaries_remain_exact() {
    let baseline = PressureBaseline {
        failure_rate: 0.1,
        p95_runtime_seconds: 10.0,
        throughput_per_slot: 5.0,
    };
    let reasons = unsafe_sample(
        &PressureSample {
            cpu_psi: 0.25,
            memory_psi: 0.05,
            io_psi: 0.10,
            available_memory_fraction: Some(0.149),
            failure_rate: 0.1026,
            p95_runtime_seconds: 15.1,
            throughput_per_slot: 4.0,
        },
        &baseline,
    )
    .unwrap();
    assert_eq!(reasons.len(), 6);
}
