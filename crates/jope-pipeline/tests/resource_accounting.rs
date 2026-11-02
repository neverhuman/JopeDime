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

mod ledger;
mod worker;
