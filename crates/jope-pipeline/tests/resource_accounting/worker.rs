use super::*;

#[test]
fn search_adapters_preserve_legacy_budget_and_explicit_unknowns() {
    use jope_pipeline::search::{Budget, Observation, BUDGET_POLICY, FIT_FAILED};
    let budget = Budget::parse("seconds=1,evaluations=2").unwrap();
    assert_eq!(
        budget
            .resource_cap_v1(2, 1000)
            .unwrap()
            .get(Unit::CpuCoreMicros),
        2_000_000
    );
    assert_eq!(budget.seconds, 1);
    assert!(BUDGET_POLICY.contains("nonpreemptive_v1"));
    assert!(budget.resource_cap_v1(u64::MAX, 1000).is_err());
    let observation = Observation {
        genome: "fixture".into(),
        program_sha256: hash("fixture"),
        cut: 0,
        generation: 0,
        origin: "random",
        train_r2: None,
        hold_r2: None,
        rmse: None,
        objective_loss: None,
        seconds: 0.0001,
        status: FIT_FAILED,
        failure: Some("fit failed".into()),
        island: None,
    };
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let r = reserve(&mut ledger, request("observation", Stage::Fit), 100);
    ledger
        .start(&r.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let receipt = observation.resource_cost_v1(&r, 300).unwrap();
    assert!(matches!(
        receipt.measurements[&Unit::CpuCoreMicros].reading,
        Reading::Unknown { .. }
    ));
    ledger.finish(receipt, |_| Ok(())).unwrap();
    assert_eq!(
        ledger
            .reconcile()
            .unwrap()
            .computed_known
            .get(Unit::Evaluations),
        1
    );
}

#[cfg(target_os = "linux")]
fn worker(request_sha256: String) -> jope_pipeline::execution::WorkerOutcome {
    use jope_pipeline::{
        execution::{ObservedConsumption, WorkerOutcome, EXECUTION_SCHEMA},
        experiment::EvaluationStatus,
    };
    WorkerOutcome {
        schema: EXECUTION_SCHEMA.into(),
        job_sha256: hash("worker-job"),
        request_sha256,
        command_status: EvaluationStatus::Failed,
        reason: Some("failed child".into()),
        exit_code: Some(1),
        signal: None,
        outputs: vec![],
        stdout: None,
        stderr: None,
        consumption: ObservedConsumption {
            wall_seconds: Some(0.0001),
            gate_released_unix_ms: Some(1),
            sampled_peak_rss_bytes: Some(1024),
            accounting_reason: "final CPU and cgroup memory unavailable; RSS sampled".into(),
            ..Default::default()
        },
    }
}

#[test]
#[cfg(target_os = "linux")]
fn worker_missing_telemetry_and_unopened_gate_are_distinct() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let r = reserve(&mut ledger, request("worker-unknown", Stage::Fit), 100);
    let outcome = worker(r.request.ids.request_id.clone());
    let receipt = outcome.resource_cost_v1(&r, 300).unwrap();
    assert!(matches!(
        receipt.measurements[&Unit::CpuCoreMicros].reading,
        Reading::Unknown { .. }
    ));
    assert!(matches!(
        receipt.measurements[&Unit::PeakMemoryBytes].reading,
        Reading::Unknown { .. }
    ));
    assert_eq!(outcome.consumption.peak_rss_bytes, None);
    let mut unstarted = outcome;
    unstarted.exit_code = None;
    unstarted.consumption.gate_released_unix_ms = None;
    unstarted.consumption.wall_seconds = Some(0.);
    unstarted.consumption.cpu_seconds = Some(0.);
    unstarted.consumption.accounting_reason = "not_started: gate never released".into();
    let receipt = unstarted.resource_cost_v1(&r, 300).unwrap();
    ledger.finish(receipt, |_| Ok(())).unwrap();
    assert_eq!(ledger.reconcile().unwrap().started, 0);
    assert_eq!(
        ledger.reconcile().unwrap().charged.get(Unit::CpuCoreMicros),
        0
    );
}

#[test]
#[cfg(target_os = "linux")]
fn bounded_trusted_cpu_children_record_actual_compile_failure_sigkill_and_retry() {
    use std::{fs, os::unix::process::ExitStatusExt, process::Command, time::Instant};
    let scratch = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/task-evidence/I16/scratch");
    fs::create_dir_all(&scratch).unwrap();
    let directory = tempfile::Builder::new()
        .prefix("trusted-child-")
        .tempdir_in(&scratch)
        .unwrap()
        .keep();
    fs::write(
        directory.join("invalid.rs"),
        "fn main() { definitely_missing_symbol(); }",
    )
    .unwrap();
    let mut p = policy();
    p.opened_unix_us = 0;
    p.deadline_unix_us = 30_000_000;
    p.terminal_headroom_us = 1_000_000;
    p.cap.0.insert(Unit::CpuCoreMicros, 20_000_000);
    p.cap.0.insert(Unit::AttemptWallMicros, 20_000_000);
    let mut ledger = ResourceLedger::new(p).unwrap();
    let mut predecessor = None;
    for (index, label) in ["compile", "kill", "retry"].into_iter().enumerate() {
        let now = index as u64 * 6_000_000;
        let mut req = request(
            label,
            if index == 0 {
                Stage::Compile
            } else {
                Stage::Fit
            },
        );
        req.requested.0.insert(Unit::CpuCoreMicros, 5_000_000);
        req.requested.0.insert(Unit::AttemptWallMicros, 5_000_000);
        req.retry_of = predecessor.clone();
        if predecessor.is_some() {
            req.requested.0.insert(Unit::Retries, 1);
        }
        let r = reserve(&mut ledger, req, now);
        ledger
            .start(&r.request.ids.attempt_id, now, |_| Ok(()))
            .unwrap();
        let began = Instant::now();
        let output = if index == 0 {
            Command::new("/usr/bin/timeout")
                .args(["5s", "rustc", "--emit=metadata"])
                .arg(directory.join("invalid.rs"))
                .arg("-o")
                .arg(directory.join("invalid.rmeta"))
                .output()
                .unwrap()
        } else {
            Command::new("/usr/bin/timeout")
                .args([
                    "5s",
                    "/bin/sh",
                    "-c",
                    if index == 1 {
                        "kill -KILL $$"
                    } else {
                        "exit 0"
                    },
                ])
                .output()
                .unwrap()
        };
        let elapsed = seconds_to_micros(began.elapsed().as_secs_f64()).unwrap();
        fs::write(directory.join(format!("{label}.stdout")), &output.stdout).unwrap();
        fs::write(directory.join(format!("{label}.stderr")), &output.stderr).unwrap();
        let mut outcome = worker(r.request.ids.request_id.clone());
        outcome.job_sha256 = hash(label);
        outcome.exit_code = output.status.code();
        outcome.signal = output.status.signal();
        outcome.consumption.wall_seconds = Some(elapsed as f64 / 1_000_000.);
        outcome.command_status = if output.status.success() {
            jope_pipeline::experiment::EvaluationStatus::Completed
        } else {
            jope_pipeline::experiment::EvaluationStatus::Failed
        };
        if index == 0 {
            assert_eq!(outcome.exit_code, Some(1));
            assert!(String::from_utf8_lossy(&output.stderr).contains("definitely_missing_symbol"));
        } else if index == 1 {
            assert_eq!(outcome.signal, Some(9));
        } else {
            assert_eq!(outcome.exit_code, Some(0));
        }
        let receipt = outcome.resource_cost_v1(&r, now + elapsed).unwrap();
        fs::write(
            directory.join(format!("{label}.json")),
            serde_json::to_vec_pretty(&receipt).unwrap(),
        )
        .unwrap();
        ledger.finish(receipt, |_| Ok(())).unwrap();
        predecessor = Some(r.request.ids.attempt_id);
    }
    let totals = ledger.reconcile().unwrap();
    assert_eq!(totals.terminal, 3);
    assert_eq!(totals.unknown_attempts[&Unit::CpuCoreMicros], 3);
    assert_eq!(totals.computed_known.get(Unit::Compilations), 1);
    assert_eq!(totals.computed_known.get(Unit::Retries), 2);
    assert_eq!(totals.charged.get(Unit::CpuCoreMicros), 15_000_000);
    // These trusted exit/signal fixtures do not qualify systemd/cgroup isolation.
}
