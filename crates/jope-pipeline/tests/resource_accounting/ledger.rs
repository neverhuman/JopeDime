use super::*;

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
