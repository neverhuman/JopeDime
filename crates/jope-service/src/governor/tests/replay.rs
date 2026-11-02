use super::*;

/// Seeded controller fixture: actual live thread counts change, not native/model/OS state.
fn replay(case: &str) -> (serde_json::Value, Vec<u32>) {
    use crate::run::{
        EvaluationOutcome, Evaluator, FakeEvaluator, ResultDisposition, RunController,
    };
    use jope_api::{Effort, RunPhase};
    use sha2::{Digest, Sha256};
    use std::time::Duration;
    let mut gov = governor(8);
    let mut effects = Effects::default();
    let mut run = RunController::new(
        "run_01JBD8Q6N4V2T5Y8K3M7P9R1S0".parse().unwrap(),
        Effort::Medium,
        gov.planned_r(),
    );
    for phase in [RunPhase::Encoding, RunPhase::Seeding, RunPhase::Searching] {
        run.advance_phase(phase).unwrap();
    }
    let mut sensors = Readings::new(500, 100);
    let mut boundary = 0;
    if case == "one" {
        sensors.memory = Ok(99);
        gov.poll(&mut sensors, &mut effects).unwrap();
        for _ in 0..7 {
            gov.round_boundary(boundary, &mut effects).unwrap();
            boundary += 1;
        }
    }
    let mut worker_counts = Vec::new();
    for first in (0..6_u64).step_by(2) {
        sensors.memory = Ok(match case {
            "one" => 150,
            "eight" => 500,
            _ if first < 4 => 99,
            _ => 201,
        });
        let observed = gov.poll(&mut sensors, &mut effects).unwrap();
        assert!(observed.warnings.iter().all(|warning| warning.host));
        assert!(observed.admission_allowed);
        gov.round_boundary(boundary, &mut effects).unwrap();
        boundary += 1;
        let workers = gov.live_workers().get();
        worker_counts.push(workers);
        assert_eq!(gov.planned_r().get(), 2);
        assert_eq!(run.snapshot().planned_rounds_in_flight, gov.planned_r());
        let mut requests = Vec::new();
        for round in first..first + 2 {
            for slot in 0..8_u32 {
                let mut hash = Sha256::new();
                hash.update(73_u64.to_le_bytes());
                hash.update(round.to_le_bytes());
                hash.update(slot.to_le_bytes());
                let seed = u64::from_le_bytes(hash.finalize()[..8].try_into().unwrap());
                requests.push(run.dispatch(round, slot, seed).unwrap());
            }
        }
        assert!(
            run.dispatch(first + 2, 0, 0).is_err(),
            "live worker count cannot change planned R"
        );
        if first == 2 && matches!(case, "pause" | "recovery") {
            let late = FakeEvaluator::new(Duration::ZERO).evaluate(&requests[0]);
            run.receive(late.clone()).unwrap();
            if case == "pause" {
                run.control(crate::run::Control::Pause, None).unwrap();
                run.expire_pause().unwrap();
            } else {
                assert!(run.recover_interrupted().unwrap());
            }
            run.control(crate::run::Control::Continue, None).unwrap();
            requests = requests
                .iter()
                .map(|request| {
                    run.dispatch(request.key.round, request.key.slot, request.seed)
                        .unwrap()
                })
                .collect();
            assert_eq!(run.receive(late).unwrap(), ResultDisposition::DiscardedLate);
        }
        // Govern actual concurrent evaluation threads, and reverse result delivery.
        for batch in requests.chunks(workers as usize) {
            let results = std::thread::scope(|scope| {
                let jobs: Vec<_> = batch
                    .iter()
                    .map(|request| {
                        scope.spawn(move || {
                            FakeEvaluator::new(Duration::from_millis(u64::from(
                                request.key.slot % 3,
                            )))
                            .evaluate(request)
                        })
                    })
                    .collect();
                jobs.into_iter()
                    .map(|job| job.join().unwrap())
                    .collect::<Vec<_>>()
            });
            for result in results.into_iter().rev() {
                run.receive(result).unwrap();
            }
        }
        run.round_committed().unwrap();
        run.round_committed().unwrap();
    }
    let projection: Vec<_> = run.committed().iter().flat_map(|round| {
        round.evaluations.iter().map(|evaluation| {
            let EvaluationOutcome::Scored(value) = evaluation.outcome else { panic!("unexpected fixture failure") };
            serde_json::json!({"round":round.round,"slot":evaluation.key.slot,"effort":round.effort,"fixture_value_bits":format!("{:016x}",value.to_bits())})
        })
    }).collect();
    assert_eq!(projection.len(), 48);
    (serde_json::json!(projection), worker_counts)
}

#[test]
fn s2_controller_replay_preserves_r_under_retirement_restore_pause_and_recovery() {
    let (baseline, workers) = replay("eight");
    assert_eq!(workers, [8, 8, 8]);
    let mut receipts = Vec::new();
    for case in ["one", "retirement", "pause", "recovery"] {
        let (projection, workers) = replay(case);
        assert_eq!(projection, baseline, "{case}");
        assert_eq!(
            workers,
            if case == "one" {
                vec![1, 1, 1]
            } else {
                vec![7, 6, 7]
            }
        );
        receipts.push(serde_json::json!({"case":case,"live_threads_by_window":workers,"planned_r":2,"committed_fixture_evaluations":48,"projection":projection}));
    }
    if let Some(path) = std::env::var_os("JOPE_GOVERNOR_PROOF_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        let proof = serde_json::json!({"kind":"S2 seeded controller fixture replay","native_model_or_os_sigkill_qualification":false,"baseline_threads":[8,8,8],"baseline_projection":baseline,"cases":receipts});
        std::fs::write(
            path.join("controller-s2.json"),
            serde_json::to_vec_pretty(&proof).unwrap(),
        )
        .unwrap();
    }
}
