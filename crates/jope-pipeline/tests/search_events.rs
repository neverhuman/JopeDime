use jope_pipeline::search_events::*;
use jope_pipeline::sha256_hex;

fn binding() -> SearchBinding {
    SearchBinding {
        run_id: "fixture-run".into(),
        task: "regression".into(),
        experiment: EvidenceRef::unknown("fixture only"),
        role_exposure: EvidenceRef::unknown("no eligibility grant"),
        context_sha256: sha256_hex(b"fixture-context"),
        source_sha256: sha256_hex(b"fixture-source"),
        evaluator: EvidenceRef::unknown("fixture evaluator"),
        profile: EvidenceRef::unknown("fixture has no registered profile"),
        legal_set: EvidenceRef::unknown("fixture legal-set not resolved"),
        fidelity: "fixture".into(),
    }
}

fn draw(log: &mut SearchEventLog, coordinate: usize, subject: &str) -> String {
    let program = sha256_hex(subject.as_bytes());
    let canonical = Some(program.clone());
    let request = log.request_id(&program, &canonical).unwrap();
    log.append(SearchEvent::Draw {
        decision: SearchDecision {
            schema: DECISION_SCHEMA.into(),
            coordinate: coordinate.to_string(),
            source: "neural".into(),
            seed_hex: Some("ffffffffffffffff".into()),
            program_sha256: Some(program),
            canonical_sha256: canonical,
            request_id: Some(request),
            outcome: if coordinate == 2 {
                "duplicate"
            } else {
                "candidate"
            }
            .into(),
            detail: None,
            behavior_law: BehaviorLaw::Unknown {
                reason: "marginal law unknown".into(),
            },
            proposal_cost: Effort::unknown("proposal cost not instrumented"),
        },
    })
    .unwrap()
}

fn finish(log: &mut SearchEventLog, attempt: String, status: &str, wall: f64) {
    log.append(SearchEvent::Finished {
        attempt_id: attempt,
        status: status.into(),
        reason: (status != "measured").then(|| "original fit failure".into()),
        observation_sha256: Some(sha256_hex(status.as_bytes())),
        cost: Effort::wall(wall).unwrap(),
    })
    .unwrap();
}

fn fixture() -> SearchEventLog {
    let mut log = SearchEventLog::new(binding()).unwrap();
    let a = draw(&mut log, 0, "A");
    let b = draw(&mut log, 1, "B");
    draw(&mut log, 2, "A");
    let c = draw(&mut log, 3, "C");
    let first = log.start(&a, 1, None).unwrap();
    finish(&mut log, first.clone(), "fit_failed", 1.);
    let retry = log.start(&a, 2, Some(first)).unwrap();
    let b_attempt = log.start(&b, 1, None).unwrap();
    let c_attempt = log.start(&c, 1, None).unwrap();
    // Completion order is deliberately different from physical start/draw order.
    finish(&mut log, c_attempt, "measured", 4.);
    finish(&mut log, retry, "measured", 2.);
    finish(&mut log, b_attempt, "measured", 3.);
    log
}

#[test]
fn four_draws_duplicate_and_retry_reconcile_every_physical_cost() {
    let log = fixture();
    let counts = log.require_complete().unwrap();
    assert_eq!(
        (counts.draws, counts.duplicate_draws, counts.unique_requests),
        (4, 1, 3)
    );
    assert_eq!(
        (
            counts.evaluation_attempts,
            counts.failed_attempts,
            counts.logical_completed
        ),
        (4, 1, 3)
    );
    assert_eq!((counts.proposal_costs, counts.terminal_costs), (4, 4));
    assert_eq!(counts.evaluation_wall_seconds, 10.);
    assert_eq!(counts.unknown_evaluation_wall_costs, 0);
    let bytes = canonical_bytes(&log).unwrap();
    if let Some(path) = std::env::var_os("I05_EVIDENCE_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("four-draw-retry.json"), &bytes).unwrap();
        std::fs::write(
            path.join("four-draw-retry-reconciliation.json"),
            serde_json::to_vec_pretty(&counts).unwrap(),
        )
        .unwrap();
    }
    assert_eq!(SearchEventLog::from_bytes(&bytes).unwrap(), log);
    assert!(String::from_utf8(bytes).unwrap().contains("f64_bits"));
}

#[test]
fn reordered_and_duplicate_deliveries_keep_original_counts_and_ids() {
    let log = fixture();
    let expected = log.require_complete().unwrap();
    let mut delivered = log.clone();
    delivered.events.reverse();
    delivered.events.extend(log.events.clone());
    let actual = delivered.require_complete().unwrap();
    assert_eq!(actual.duplicate_deliveries, log.events.len());
    assert_eq!(actual.draws, expected.draws);
    assert_eq!(actual.evaluation_attempts, expected.evaluation_attempts);
    assert_eq!(actual.failed_attempts, 1);
    assert_eq!(actual.evaluation_wall_seconds, 10.);
}

#[test]
fn missing_altered_or_orphan_events_never_close_as_complete() {
    let original = fixture();
    let mut log = original.clone();
    let removed = log
        .events
        .iter()
        .position(
            |e| matches!(e.event, SearchEvent::Finished { ref status, .. } if status == "measured"),
        )
        .unwrap();
    log.events.remove(removed);
    assert!(log.reconcile().is_err());
    assert!(log.require_complete().is_err());
    let mut pending = SearchEventLog::new(binding()).unwrap();
    let pending_draw = draw(&mut pending, 0, "pending");
    pending.start(&pending_draw, 1, None).unwrap();
    assert!(pending.start(&pending_draw, 1, None).is_err());
    assert_eq!(pending.reconcile().unwrap().pending_attempts, 1);
    assert!(pending.require_complete().is_err());
    let mut lost_shadow = original.clone();
    lost_shadow.events.retain(
        |e| !matches!(&e.event, SearchEvent::Draw { decision } if decision.coordinate == "2"),
    );
    assert!(
        lost_shadow.reconcile().is_err(),
        "capture inventory detects even a lost unattempted duplicate"
    );
    let mut log = original.clone();
    log.events.retain(
        |e| !matches!(&e.event, SearchEvent::Draw { decision } if decision.coordinate == "0"),
    );
    assert!(log.reconcile().is_err());
    let mut log = original.clone();
    log.events.retain(
        |e| !matches!(&e.event, SearchEvent::Finished { status, .. } if status == "fit_failed"),
    );
    assert!(
        log.reconcile().is_err(),
        "dropping the original failure orphans its retry"
    );
    let mut log = original.clone();
    if let SearchEvent::Draw { decision } = &mut log.events[0].event {
        decision.source = "altered".into();
    }
    assert!(log.reconcile().is_err());
    let mut log = original;
    let mut terminal = log
        .events
        .iter()
        .find_map(|e| {
            matches!(&e.event, SearchEvent::Finished { status,.. } if status == "measured")
                .then(|| e.event.clone())
        })
        .unwrap();
    if let SearchEvent::Finished { cost, .. } = &mut terminal {
        *cost = Effort::wall(123.).unwrap();
    }
    log.append(terminal).unwrap();
    assert!(
        log.reconcile().is_err(),
        "conflicting terminal cannot pick a winner"
    );
}

#[test]
fn unknown_cost_is_retained_and_context_changes_request_identity() {
    let mut log = fixture();
    let terminal = log
        .events
        .iter()
        .position(
            |r| matches!(&r.event, SearchEvent::Finished { status,.. } if status == "measured"),
        )
        .unwrap();
    let removed = log.events.remove(terminal);
    log.captured_event_ids.retain(|id| id != &removed.event_id);
    let mut event = removed.event;
    if let SearchEvent::Finished { cost, .. } = &mut event {
        *cost = Effort::unknown("worker interrupted before telemetry collection");
    }
    log.append(event).unwrap();
    assert_eq!(
        log.require_complete()
            .unwrap()
            .unknown_evaluation_wall_costs,
        1
    );
    let program = sha256_hex(b"A");
    let request = log.request_id(&program, &Some(program.clone())).unwrap();
    log.binding.fidelity = "different-fidelity".into();
    assert_ne!(
        log.request_id(&program, &Some(program.clone())).unwrap(),
        request
    );
    assert!(log.reconcile().is_err());
}

#[test]
fn gp_child_cannot_inherit_seed_density_or_posterior_sample() {
    let mut log = SearchEventLog::new(binding()).unwrap();
    let id = draw(&mut log, 0, "child");
    let removed = log.events.remove(0);
    log.captured_event_ids.retain(|id| id != &removed.event_id);
    let mut event = removed.event;
    if let SearchEvent::Draw { decision } = &mut event {
        decision.source = "mutate".into();
        decision.behavior_law = BehaviorLaw::ContinuousDensity {
            subject_sha256: decision.program_sha256.clone().unwrap(),
            density: FloatBits::new(0.25).unwrap(),
            measure: "seed-ancestor".into(),
        };
    }
    log.append(event).unwrap();
    assert!(log
        .reconcile()
        .unwrap_err()
        .to_string()
        .contains("ancestry is not density"));
    assert!(!id.is_empty());
    assert!(
        serde_json::from_str::<BehaviorLaw>(r#"{"kind":"posterior_sample","sample":0.5}"#).is_err()
    );
}

#[test]
fn strict_new_version_dispatch_rejects_duplicate_keys_and_unknown_fields() {
    let log = fixture();
    let mut bytes = String::from_utf8(canonical_bytes(&log).unwrap()).unwrap();
    bytes = bytes.replacen(
        "\"schema\":\"jope.search-events/v1\"",
        "\"schema\":\"jope.search-events/v2\"",
        1,
    );
    assert!(SearchEventLog::from_bytes(bytes.as_bytes()).is_err());
    let mut value = serde_json::to_value(&log).unwrap();
    value["unknown"] = true.into();
    assert!(serde_json::from_value::<SearchEventLog>(value).is_err());
    assert!(SearchEventLog::from_bytes(
        br#"{"schema":"jope.search-events/v1","schema":"jope.search-events/v1"}"#
    )
    .is_err());
}

#[test]
fn explicit_failed_attempt_states_retain_unknown_costs_and_empty_closure_refuses() {
    let mut log = SearchEventLog::new(binding()).unwrap();
    assert!(log.require_complete().is_err());
    for (i, status) in [
        "parse_failed",
        "fit_failed",
        "score_failed",
        "panicked",
        "timed_out",
        "interrupted",
    ]
    .into_iter()
    .enumerate()
    {
        let draw = draw(&mut log, i + 10, status);
        let attempt = log.start(&draw, 1, None).unwrap();
        log.append(SearchEvent::Finished {
            attempt_id: attempt,
            status: status.into(),
            reason: Some(format!("original {status}")),
            observation_sha256: None,
            cost: Effort::unknown("terminal resource telemetry unavailable"),
        })
        .unwrap();
    }
    let summary = log.require_complete().unwrap();
    assert_eq!(summary.evaluation_attempts, 6);
    assert_eq!(summary.failed_attempts, 6);
    assert_eq!(summary.logical_completed, 0);
    assert_eq!(summary.terminal_costs, 6);
    assert_eq!(summary.unknown_evaluation_wall_costs, 6);
}
