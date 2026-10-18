use jope_pipeline::resource_ledger::*;
use std::collections::BTreeSet;

fn hash(label: &str) -> String {
    jope_pipeline::sha256_hex(label.as_bytes())
}
fn policy() -> Policy {
    Policy {
        schema: SCHEMA.into(),
        experiment_sha256: hash("experiment"),
        opened_unix_us: 100,
        deadline_unix_us: 10_000,
        terminal_headroom_us: 1000,
        cap: Amounts::new([
            (Unit::CpuCoreMicros, 1000),
            (Unit::AttemptWallMicros, 3000),
            (Unit::PeakMemoryBytes, 1000),
            (Unit::Evaluations, 10),
            (Unit::Compilations, 10),
            (Unit::Retries, 10),
            (Unit::CustodyOperations, 10),
            (Unit::StorageBytes, 1000),
            (Unit::StorageOperations, 10),
            (Unit::MemoryByteSeconds, 1000),
        ]),
        terminal_reserve: Amounts::new([
            (Unit::CpuCoreMicros, 100),
            (Unit::AttemptWallMicros, 100),
            (Unit::PeakMemoryBytes, 100),
        ]),
        gpu_allocation: None,
        conversion: None,
    }
}
fn request(label: &str, stage: Stage) -> ReservationRequest {
    let mut requested = Amounts::new([
        (Unit::CpuCoreMicros, 200),
        (Unit::AttemptWallMicros, 200),
        (Unit::PeakMemoryBytes, 400),
    ]);
    match stage {
        Stage::Compile => {
            requested.0.insert(Unit::Compilations, 1);
        }
        Stage::Fit | Stage::Evaluate => {
            requested.0.insert(Unit::Evaluations, 1);
        }
        Stage::Publication | Stage::Checkpoint | Stage::Custody => {
            requested.0.insert(Unit::CustodyOperations, 1);
        }
        _ => {}
    }
    ReservationRequest {
        ids: JoinIds {
            experiment_sha256: hash("experiment"),
            draw_id: hash("draw"),
            request_id: hash("request"),
            attempt_id: hash(label),
            logical_id: hash("logical"),
        },
        stage,
        requested,
        retry_of: None,
        cache: CacheState::Cold {
            lifetime_id: "cache-lifetime-1".into(),
        },
    }
}
fn cost(reservation: &Reservation, disposition: Disposition, ended: u64) -> AttemptCost {
    let measurements = reservation
        .request
        .requested
        .0
        .iter()
        .map(|(&unit, &amount)| {
            let value = if disposition == Disposition::ProvenNotStarted {
                0
            } else {
                match unit {
                    Unit::CpuCoreMicros | Unit::AttemptWallMicros => amount / 2,
                    _ => amount,
                }
            };
            (
                unit,
                Measurement {
                    reading: Reading::exact(value),
                    collector: "deterministic-fixture".into(),
                    clock: "fixture-unix-us".into(),
                    allocation: hash("experiment"),
                    scope: vec![
                        "experiment".into(),
                        reservation.request.ids.attempt_id.clone(),
                    ],
                    from_unix_us: reservation.reserved_unix_us,
                    to_unix_us: ended,
                },
            )
        })
        .collect();
    AttemptCost {
        schema: SCHEMA.into(),
        reservation_sha256: reservation.identity().unwrap(),
        ids: reservation.request.ids.clone(),
        disposition,
        ended_unix_us: ended,
        exit_code: None,
        signal: None,
        measurements,
        evidence_sha256: [hash("raw-receipt")].into(),
        retained_sha256: BTreeSet::new(),
        retained_bytes: Reading::unknown("deterministic fixture has no physical storage collector"),
        reason: if disposition == Disposition::ProvenNotStarted {
            "not_started: gate never released; fixture gate proof".into()
        } else {
            "deterministic collector fixture".into()
        },
    }
}
fn reserve(ledger: &mut ResourceLedger, request: ReservationRequest, now: u64) -> Reservation {
    ledger.reserve(request, now, |_| Ok(())).unwrap()
}
fn finish(
    ledger: &mut ResourceLedger,
    reservation: &Reservation,
    disposition: Disposition,
    ended: u64,
) {
    ledger
        .start(
            &reservation.request.ids.attempt_id,
            reservation.reserved_unix_us,
            |_| Ok(()),
        )
        .unwrap();
    ledger
        .finish(cost(reservation, disposition, ended), |_| Ok(()))
        .unwrap();
}

#[test]
fn failed_compile_killed_fit_and_real_retry_keep_every_physical_charge() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let compile = reserve(&mut ledger, request("compile", Stage::Compile), 100);
    ledger
        .start(&compile.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let mut failure = cost(&compile, Disposition::Failed, 300);
    failure.exit_code = Some(1);
    ledger.finish(failure.clone(), |_| Ok(())).unwrap();
    let mut retry = request("killed-fit", Stage::Fit);
    retry.retry_of = Some(compile.request.ids.attempt_id.clone());
    retry.requested.0.insert(Unit::Retries, 1);
    let killed = reserve(&mut ledger, retry, 301);
    ledger
        .start(&killed.request.ids.attempt_id, 301, |_| Ok(()))
        .unwrap();
    let mut killed_cost = cost(&killed, Disposition::Killed, 501);
    killed_cost.signal = Some(9);
    ledger.finish(killed_cost.clone(), |_| Ok(())).unwrap();
    let mut retry = request("retained-fit", Stage::Fit);
    retry.retry_of = Some(killed.request.ids.attempt_id.clone());
    retry.requested.0.insert(Unit::Retries, 1);
    let retried = reserve(&mut ledger, retry, 502);
    ledger
        .start(&retried.request.ids.attempt_id, 502, |_| Ok(()))
        .unwrap();
    let mut success = cost(&retried, Disposition::Completed, 702);
    success.retained_sha256.insert(hash("retained-predictor"));
    ledger.finish(success, |_| Ok(())).unwrap();
    let totals = ledger.reconcile().unwrap();
    assert_eq!(
        (totals.started, totals.terminal, totals.retained),
        (3, 3, 1)
    );
    assert_eq!(totals.computed_known.get(Unit::CpuCoreMicros), 300);
    assert_eq!(totals.requested.get(Unit::CpuCoreMicros), 600);
    assert_eq!(totals.released.get(Unit::CpuCoreMicros), 300);
    assert_eq!(totals.computed_known.get(Unit::Compilations), 1);
    assert_eq!(totals.computed_known.get(Unit::Evaluations), 2);
    assert_eq!(totals.computed_known.get(Unit::Retries), 2);
    assert_eq!(totals.retained_bytes_unknown_attempts, 3);
    assert_eq!(ledger.cost(&compile.request.ids.attempt_id), Some(&failure));
    assert_eq!(
        ledger.cost(&killed.request.ids.attempt_id),
        Some(&killed_cost)
    );
}

#[test]
fn original_deadline_cap_and_terminal_headroom_cannot_restart_on_retry() {
    for unit in [Unit::CpuCoreMicros, Unit::AttemptWallMicros] {
        let mut invalid = policy();
        invalid.terminal_reserve.0.remove(&unit);
        assert!(ResourceLedger::new(invalid).is_err());
    }
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    assert!(ledger
        .reserve(request("late", Stage::Fit), 8900, |_| Ok(()))
        .is_err());
    assert!(ledger
        .reserve(request("expired", Stage::Fit), 10_000, |_| Ok(()))
        .is_err());
    let mut excessive = request("large", Stage::Fit);
    excessive.requested.0.insert(Unit::CpuCoreMicros, 901);
    assert!(ledger.reserve(excessive, 100, |_| Ok(())).is_err());
    let initial = reserve(&mut ledger, request("initial", Stage::Compile), 100);
    finish(&mut ledger, &initial, Disposition::Failed, 300);
    let mut retry = request("retry", Stage::Compile);
    retry.retry_of = Some(initial.request.ids.attempt_id);
    retry.requested.0.insert(Unit::Retries, 1);
    assert!(ledger.reserve(retry, 8999, |_| Ok(())).is_err());
    let terminal = reserve(&mut ledger, request("publish", Stage::Publication), 9500);
    assert_eq!(terminal.deadline_unix_us, 10_000);
    assert_eq!(ledger.policy().deadline_unix_us, 10_000);
}

#[test]
fn concurrent_reservations_cannot_overbook_memory_or_cpu() {
    use std::sync::{Arc, Barrier, Mutex};
    let ledger = Arc::new(Mutex::new(ResourceLedger::new(policy()).unwrap()));
    let barrier = Arc::new(Barrier::new(3));
    let threads: Vec<_> = (0..2)
        .map(|n| {
            let ledger = ledger.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                let mut req = request(&format!("concurrent-{n}"), Stage::Fit);
                req.requested.0.insert(Unit::PeakMemoryBytes, 600);
                barrier.wait();
                ledger.lock().unwrap().reserve(req, 100, |_| Ok(())).is_ok()
            })
        })
        .collect();
    barrier.wait();
    let admitted = threads
        .into_iter()
        .map(|thread| thread.join().unwrap())
        .filter(|ok| *ok)
        .count();
    assert_eq!(admitted, 1);
    assert_eq!(
        ledger
            .lock()
            .unwrap()
            .reconcile()
            .unwrap()
            .active_reserved
            .get(Unit::PeakMemoryBytes),
        600
    );
}

#[test]
fn missing_final_telemetry_keeps_full_allowance_and_unstarted_zero_needs_proof() {
    let mut p = policy();
    p.cap.0.insert(Unit::CpuCoreMicros, 300);
    let mut ledger = ResourceLedger::new(p).unwrap();
    let reservation = reserve(&mut ledger, request("unknown", Stage::Fit), 100);
    ledger
        .start(&reservation.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let mut receipt = cost(&reservation, Disposition::Interrupted, 300);
    receipt
        .measurements
        .get_mut(&Unit::CpuCoreMicros)
        .unwrap()
        .reading = Reading::unknown("controller died before final cpu.stat");
    ledger.finish(receipt.clone(), |_| Ok(())).unwrap();
    let totals = ledger.reconcile().unwrap();
    assert_eq!(totals.charged.get(Unit::CpuCoreMicros), 200);
    assert_eq!(totals.computed_known.get(Unit::CpuCoreMicros), 0);
    assert_eq!(totals.unknown_attempts[&Unit::CpuCoreMicros], 1);
    assert!(ledger
        .reserve(request("no-free-retry", Stage::Fit), 301, |_| Ok(()))
        .is_err());
    assert!(serde_json::to_string(&receipt)
        .unwrap()
        .contains("controller died before final cpu.stat"));
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let r = reserve(&mut ledger, request("unopened", Stage::Fit), 100);
    let mut forged = cost(&r, Disposition::ProvenNotStarted, 100);
    forged.reason = "assumed free".into();
    assert!(ledger.finish(forged, |_| Ok(())).is_err());
    ledger
        .finish(cost(&r, Disposition::ProvenNotStarted, 100), |_| Ok(()))
        .unwrap();
    assert_eq!(
        ledger.reconcile().unwrap().charged.get(Unit::CpuCoreMicros),
        0
    );
    assert_eq!(
        ledger
            .reconcile()
            .unwrap()
            .released
            .get(Unit::CpuCoreMicros),
        200
    );
}

#[test]
fn duplicates_do_not_reserve_or_charge_twice_and_reordered_or_conflicting_receipts_refuse() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let mut events = Vec::new();
    let req = request("one", Stage::Fit);
    let r = ledger
        .reserve(req.clone(), 100, |e| {
            events.push(e.clone());
            Ok(())
        })
        .unwrap();
    assert_eq!(
        ledger
            .reserve(req.clone(), 9999, |_| panic!("duplicate persisted"))
            .unwrap(),
        r
    );
    let receipt = cost(&r, Disposition::Completed, 300);
    assert!(ledger.finish(receipt.clone(), |_| Ok(())).is_err());
    ledger
        .start(&req.ids.attempt_id, 100, |e| {
            events.push(e.clone());
            Ok(())
        })
        .unwrap();
    ledger
        .finish(receipt.clone(), |e| {
            events.push(e.clone());
            Ok(())
        })
        .unwrap();
    ledger
        .finish(receipt.clone(), |_| panic!("duplicate terminal persisted"))
        .unwrap();
    let mut conflict = receipt;
    conflict.reason = "different bytes".into();
    assert!(ledger.finish(conflict, |_| Ok(())).is_err());
    assert_eq!(ledger.reconcile().unwrap().terminal, 1);
    let restored = ResourceLedger::restore(policy(), &events).unwrap();
    assert_eq!(restored.reconcile().unwrap(), ledger.reconcile().unwrap());
    let mut enlarged = policy();
    enlarged.cap.0.insert(Unit::CpuCoreMicros, 2000);
    assert!(ResourceLedger::restore(enlarged, &events).is_err());
    events.swap(0, 2);
    assert!(ResourceLedger::restore(policy(), &events).is_err());
}

#[test]
fn independent_cost_arrival_order_and_cpu_overbooking_do_not_change_totals() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let mut first = request("cpu-first", Stage::Fit);
    first.requested.0.insert(Unit::CpuCoreMicros, 600);
    first.requested.0.insert(Unit::AttemptWallMicros, 600);
    let a = reserve(&mut ledger, first.clone(), 100);
    let mut second = first;
    second.ids.attempt_id = hash("cpu-second");
    assert!(ledger.reserve(second, 100, |_| Ok(())).is_err());
    finish(&mut ledger, &a, Disposition::Failed, 700);
    let b = reserve(&mut ledger, request("cost-first", Stage::Fit), 701);
    let c = reserve(&mut ledger, request("cost-second", Stage::Fit), 701);
    ledger
        .start(&b.request.ids.attempt_id, 701, |_| Ok(()))
        .unwrap();
    ledger
        .start(&c.request.ids.attempt_id, 701, |_| Ok(()))
        .unwrap();
    let before = ledger.clone();
    ledger
        .finish(cost(&c, Disposition::Completed, 1001), |_| Ok(()))
        .unwrap();
    ledger
        .finish(cost(&b, Disposition::Completed, 901), |_| Ok(()))
        .unwrap();
    let mut chronological = before;
    chronological
        .finish(cost(&b, Disposition::Completed, 901), |_| Ok(()))
        .unwrap();
    chronological
        .finish(cost(&c, Disposition::Completed, 1001), |_| Ok(()))
        .unwrap();
    assert_eq!(
        ledger.reconcile().unwrap(),
        chronological.reconcile().unwrap()
    );
    let mut orphan = request("orphan-retry", Stage::Fit);
    orphan.retry_of = Some(hash("missing-attempt"));
    orphan.requested.0.insert(Unit::Retries, 1);
    assert!(ledger.reserve(orphan, 1002, |_| Ok(())).is_err());
}

#[test]
fn journal_failure_cannot_authorize_work_or_release_resources() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let fail = |_: &Event| -> anyhow::Result<()> { anyhow::bail!("durable journal unavailable") };
    let req = request("journal", Stage::Fit);
    assert!(ledger.reserve(req.clone(), 100, fail).is_err());
    assert!(ledger.reservation(&req.ids.attempt_id).is_none());
    let r = reserve(&mut ledger, req, 100);
    assert!(ledger.start(&r.request.ids.attempt_id, 100, fail).is_err());
    assert_eq!(ledger.reconcile().unwrap().started, 0);
    ledger
        .start(&r.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    assert!(ledger
        .finish(cost(&r, Disposition::Completed, 300), fail)
        .is_err());
    assert_eq!(
        ledger
            .reconcile()
            .unwrap()
            .active_reserved
            .get(Unit::CpuCoreMicros),
        200
    );
}

#[test]
fn reservation_identity_uses_i02_canonical_payload_and_domain_separation() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let reservation = reserve(&mut ledger, request("canonical", Stage::Compile), 100);
    // Independent projection: this fixture contains ASCII keys and exact
    // integers, so serde_json::Value's ordered map emits its canonical payload.
    let canonical = serde_json::to_vec(&serde_json::to_value(&reservation).unwrap()).unwrap();
    let mut preimage = b"jope.resource-reservation/v1\0".to_vec();
    preimage.extend((canonical.len() as u64).to_be_bytes());
    preimage.extend(canonical);
    assert_eq!(
        reservation.identity().unwrap(),
        jope_pipeline::sha256_hex(&preimage)
    );
    assert_ne!(
        reservation.identity().unwrap(),
        jope_pipeline::sha256_hex(&serde_json::to_vec(&reservation).unwrap())
    );
    let mut changed = reservation;
    changed.request.ids.attempt_id = hash("other-attempt");
    assert_ne!(
        changed.identity().unwrap(),
        ledger
            .reservation(&hash("canonical"))
            .unwrap()
            .identity()
            .unwrap()
    );
    let mut p = policy();
    p.cap.0.insert(Unit::StorageBytes, 1_u64 << 53);
    assert!(ResourceLedger::new(p).is_err());
}

#[test]
fn mixed_cpu_gpu_units_require_prospective_conversion_and_exact_device() {
    let mut p = policy();
    p.gpu_allocation = Some("GPU-uuid/MIG-uuid:allocation-7".into());
    p.cap.0.insert(Unit::GpuDeviceMicros, 1000);
    let mut req = request("gpu", Stage::Fit);
    req.requested.0.insert(Unit::GpuDeviceMicros, 100);
    let mut ledger = ResourceLedger::new(p.clone()).unwrap();
    let r = reserve(&mut ledger, req.clone(), 100);
    ledger
        .start(&r.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let mut receipt = cost(&r, Disposition::Completed, 300);
    assert!(ledger.finish(receipt.clone(), |_| Ok(())).is_err());
    receipt
        .measurements
        .get_mut(&Unit::GpuDeviceMicros)
        .unwrap()
        .allocation = p.gpu_allocation.clone().unwrap();
    ledger.finish(receipt.clone(), |_| Ok(())).unwrap();
    assert!(ledger.credits().is_err());
    let totals = ledger.reconcile().unwrap();
    assert_eq!(totals.computed_known.get(Unit::CpuCoreMicros), 100);
    assert_eq!(totals.computed_known.get(Unit::GpuDeviceMicros), 100);
    p.conversion = Some(CreditConversion {
        version: "fixture-credits/v1".into(),
        preregistration_sha256: hash("prospective-registration"),
        registered_unix_us: 99,
        cpu_credits_per_micro: 2,
        gpu_credits_per_micro: 3,
    });
    let mut ledger = ResourceLedger::new(p.clone()).unwrap();
    let r = reserve(&mut ledger, req, 100);
    ledger
        .start(&r.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    assert!(ledger.finish(receipt, |_| Ok(())).is_err());
    let mut receipt = cost(&r, Disposition::Completed, 300);
    receipt
        .measurements
        .get_mut(&Unit::GpuDeviceMicros)
        .unwrap()
        .allocation = p.gpu_allocation.clone().unwrap();
    ledger.finish(receipt, |_| Ok(())).unwrap();
    assert_eq!(ledger.credits().unwrap(), 500);
    p.conversion.as_mut().unwrap().registered_unix_us = 101;
    assert!(ResourceLedger::new(p).is_err());
}

#[test]
fn integer_overflow_nonfinite_negative_and_unlisted_dimensions_refuse() {
    for seconds in [-1., f64::INFINITY, f64::NAN, u64::MAX as f64] {
        assert!(seconds_to_micros(seconds).is_err());
    }
    assert_eq!(seconds_to_micros(0.0000001).unwrap(), 1);
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let mut req = request("overflow", Stage::Fit);
    req.requested.0.insert(Unit::CpuCoreMicros, u64::MAX);
    assert!(ledger.reserve(req, 100, |_| Ok(())).is_err());
    let mut req = request("unlisted-gpu", Stage::Fit);
    req.requested.0.insert(Unit::GpuDeviceMicros, 1);
    assert!(ledger.reserve(req, 100, |_| Ok(())).is_err());
    let mut p = policy();
    p.terminal_headroom_us = u64::MAX;
    assert!(ResourceLedger::new(p).is_err());
}

#[test]
fn overrun_is_retained_then_blocks_more_work_and_uncertainty_is_not_released() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let r = reserve(&mut ledger, request("overrun", Stage::Fit), 100);
    ledger
        .start(&r.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let mut receipt = cost(&r, Disposition::Killed, 300);
    receipt
        .measurements
        .get_mut(&Unit::CpuCoreMicros)
        .unwrap()
        .reading = Reading::Known {
        value: 200,
        uncertainty: 1,
    };
    ledger.finish(receipt, |_| Ok(())).unwrap();
    assert!(ledger.reconcile().unwrap().over_cap);
    assert_eq!(
        ledger.reconcile().unwrap().charged.get(Unit::CpuCoreMicros),
        201
    );
    assert!(ledger
        .reserve(request("after-overrun", Stage::Fit), 301, |_| Ok(()))
        .is_err());
}

#[test]
fn inclusive_parent_and_child_counter_overlap_is_rejected() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let a = reserve(&mut ledger, request("parent", Stage::Fit), 100);
    let b = reserve(&mut ledger, request("child", Stage::Fit), 100);
    ledger
        .start(&a.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    ledger
        .start(&b.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let mut parent = cost(&a, Disposition::Completed, 300);
    parent
        .measurements
        .get_mut(&Unit::CpuCoreMicros)
        .unwrap()
        .scope = vec!["experiment".into()];
    ledger.finish(parent, |_| Ok(())).unwrap();
    assert!(ledger
        .finish(cost(&b, Disposition::Completed, 300), |_| Ok(()))
        .is_err());
}

#[test]
fn cache_warm_reuse_preserves_cold_lifetime_cost_and_stage_dimensions() {
    let mut ledger = ResourceLedger::new(policy()).unwrap();
    let mut req = request("cold-preparation", Stage::Prepare);
    req.requested.0.extend([
        (Unit::StorageBytes, 10),
        (Unit::StorageOperations, 1),
        (Unit::MemoryByteSeconds, 500),
    ]);
    let cold = reserve(&mut ledger, req, 100);
    ledger
        .start(&cold.request.ids.attempt_id, 100, |_| Ok(()))
        .unwrap();
    let cold_cost = cost(&cold, Disposition::Completed, 300);
    ledger.finish(cold_cost.clone(), |_| Ok(())).unwrap();
    let mut req = request("warm-fit", Stage::Fit);
    req.cache = CacheState::Warm {
        lifetime_id: "cache-lifetime-1".into(),
        preparation_cost_sha256: cold_cost.identity().unwrap(),
    };
    let warm = reserve(&mut ledger, req, 301);
    finish(&mut ledger, &warm, Disposition::Completed, 501);
    let totals = ledger.reconcile().unwrap();
    assert_eq!(totals.computed_known.get(Unit::CpuCoreMicros), 200);
    assert_eq!(totals.computed_known.get(Unit::PeakMemoryBytes), 400);
    assert_eq!(totals.computed_known.get(Unit::StorageBytes), 10);
    assert_eq!(totals.computed_known.get(Unit::MemoryByteSeconds), 500);
    assert_eq!(ledger.cost(&cold.request.ids.attempt_id), Some(&cold_cost));
    assert!(matches!(warm.request.cache, CacheState::Warm { .. }));
}

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
