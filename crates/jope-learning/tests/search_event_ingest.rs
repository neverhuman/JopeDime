use jope_learning::progress::{
    census,
    engine::{self, SearchWorkBinding},
    inventory::{Decoded, Event, RunManifest, WorkSpec},
};
use jope_pipeline::{
    search::{ObservationRecord, BOOTSTRAP_GENOME},
    search_events::*,
    sha256_hex,
};
use std::collections::BTreeMap;

fn fixture() -> (
    SearchEventLog,
    Decoded<RunManifest>,
    Vec<SearchWorkBinding>,
    Vec<Vec<u8>>,
) {
    let binding = SearchBinding {
        run_id: "search-ingest".into(),
        task: "regression".into(),
        experiment: EvidenceRef::unknown("CPU fixture"),
        role_exposure: EvidenceRef::unknown("no replay eligibility"),
        context_sha256: sha256_hex(b"context"),
        source_sha256: sha256_hex(b"source"),
        evaluator: EvidenceRef::unknown("synthetic fixture"),
        profile: EvidenceRef::unknown("not a profile execution"),
        legal_set: EvidenceRef::unknown("fixture"),
        fidelity: "fixture".into(),
    };
    let mut log = SearchEventLog::new(binding).unwrap();
    let mut draws = Vec::new();
    let mut specs = BTreeMap::new();
    for (i, genome) in [
        BOOTSTRAP_GENOME,
        "(pipe X (ridge lambda=0.1))",
        BOOTSTRAP_GENOME,
        "(pipe X (ridge lambda=2.0))",
    ]
    .into_iter()
    .enumerate()
    {
        let program = sha256_hex(genome.as_bytes());
        let canonical = canonical_recipe(genome);
        let request = log.request_id(&program, &canonical).unwrap();
        specs.entry(request.clone()).or_insert(WorkSpec {
            stage: "gp-fit".into(),
            subject: format!("recipe-{}", &program[..8]),
            inputs: vec![request.clone()],
        });
        let draw = log
            .append(SearchEvent::Draw {
                decision: SearchDecision {
                    schema: DECISION_SCHEMA.into(),
                    coordinate: i.to_string(),
                    source: "neural".into(),
                    seed_hex: None,
                    program_sha256: Some(program),
                    canonical_sha256: canonical,
                    request_id: Some(request),
                    outcome: if i == 2 { "duplicate" } else { "candidate" }.into(),
                    detail: None,
                    behavior_law: BehaviorLaw::Unknown {
                        reason: "actual marginal law unknown".into(),
                    },
                    proposal_cost: Effort::unknown("no proposal timer"),
                },
            })
            .unwrap();
        draws.push((draw, genome));
    }
    let manifest = Decoded::new(
        RunManifest::new(
            "search-ingest",
            "fixture-publisher",
            &"a".repeat(40),
            2,
            specs.values().cloned().collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let bindings = specs
        .into_iter()
        .map(|(request_id, spec)| SearchWorkBinding {
            request_id,
            work_id: spec.work_id("search-ingest").unwrap(),
        })
        .collect();
    let mut observations = Vec::new();
    let mut record =
        |log: &mut SearchEventLog, attempt: String, genome: &str, success: bool, wall: f64| {
            let status = if success { "measured" } else { "fit_failed" };
            let observation = ObservationRecord {
                genome: genome.into(),
                program_sha256: sha256_hex(genome.as_bytes()),
                cut: 0,
                generation: 0,
                origin: "random".into(),
                train_r2: success.then_some(0.8),
                hold_r2: success.then_some(0.7),
                rmse: success.then_some(0.3),
                seconds: wall,
                status: status.into(),
                failure: (!success).then(|| "original failed fit".into()),
                island: None,
            };
            let bytes = serde_json::to_vec(&observation).unwrap();
            let digest = sha256_hex(&bytes);
            observations.push(bytes);
            log.append(SearchEvent::Finished {
                attempt_id: attempt,
                status: status.into(),
                reason: observation.failure,
                observation_sha256: Some(digest),
                cost: Effort::wall(wall).unwrap(),
            })
            .unwrap();
        };
    let first = log.start(&draws[0].0, 1, None).unwrap();
    record(&mut log, first.clone(), draws[0].1, false, 1.);
    let retry = log.start(&draws[0].0, 2, Some(first)).unwrap();
    let b = log.start(&draws[1].0, 1, None).unwrap();
    let c = log.start(&draws[3].0, 1, None).unwrap();
    record(&mut log, c, draws[3].1, true, 4.);
    record(&mut log, b, draws[1].1, true, 3.);
    record(&mut log, retry, draws[0].1, true, 2.);
    (log, manifest, bindings, observations)
}

#[test]
fn actual_progress_bridge_keeps_failure_retry_costs_and_one_logical_result() {
    let (log, manifest, bindings, observations) = fixture();
    let bytes = canonical_bytes(&log).unwrap();
    let mapped = engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations,
    )
    .unwrap();
    assert_eq!(mapped.reconciliation.draws, 4);
    assert_eq!(mapped.reconciliation.evaluation_attempts, 4);
    assert_eq!(mapped.reconciliation.logical_completed, 3);
    assert_eq!(mapped.reconciliation.evaluation_wall_seconds, 10.);
    assert_eq!(mapped.events.len(), 12);
    assert_eq!(mapped.mapped.blobs[0], bytes);
    assert_eq!(mapped.mapped.blobs.len(), 5);
    // Exercise the existing production census rather than a second counter implementation.
    let census = census::reconcile(&manifest, &mapped.events, &[]);
    assert!(census.reconciled(), "{census:?}");
    assert_eq!(
        mapped
            .events
            .iter()
            .filter(|e| matches!(
                e.value().body,
                jope_learning::progress::inventory::EventBody::Failed { .. }
            ))
            .count(),
        1
    );
    let mut delivered = mapped.events.clone();
    delivered.reverse();
    delivered.extend(mapped.events.clone());
    let census = census::reconcile(&manifest, &delivered, &[]);
    assert!(census.reconciled(), "{census:?}");
}

#[test]
fn duplicate_delivery_is_idempotent_and_conflicting_terminal_refuses() {
    let (mut log, manifest, bindings, observations) = fixture();
    let original = canonical_bytes(&log).unwrap();
    let first = engine::search_events(
        &manifest,
        &original,
        &sha256_hex(&original),
        &bindings,
        &observations,
    )
    .unwrap();
    log.events.extend(log.events.clone());
    log.events.reverse();
    let bytes = canonical_bytes(&log).unwrap();
    let replay = engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations,
    )
    .unwrap();
    // Failure details bind the exact retained artifact; identical artifact replay is byte-identical.
    let replay2 = engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations,
    )
    .unwrap();
    assert_eq!(replay.mapped.objects, replay2.mapped.objects);
    assert_eq!(replay.mapped.objects, first.mapped.objects);
    assert_eq!(replay.events.len(), first.events.len());
    assert_eq!(
        replay.reconciliation.duplicate_deliveries,
        log.events.len() / 2
    );
    let mut conflict = log
        .events
        .iter()
        .find_map(|r| matches!(r.event, SearchEvent::Finished { .. }).then(|| r.event.clone()))
        .unwrap();
    if let SearchEvent::Finished { cost, .. } = &mut conflict {
        *cost = Effort::wall(99.).unwrap();
    }
    log.append(conflict).unwrap();
    let bytes = canonical_bytes(&log).unwrap();
    assert!(engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations
    )
    .is_err());
}

#[test]
fn missing_or_altered_event_cost_result_and_original_run_refuse() {
    let (log, manifest, bindings, observations) = fixture();
    let bytes = canonical_bytes(&log).unwrap();
    assert!(
        engine::search_events(&manifest, &bytes, &"0".repeat(64), &bindings, &observations)
            .is_err()
    );
    assert!(engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations[..3]
    )
    .is_err());
    assert!(engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings[..2],
        &observations
    )
    .is_err());
    let mut missing = log.clone();
    let index = missing
        .events
        .iter()
        .position(
            |r| matches!(&r.event, SearchEvent::Finished { status, .. } if status == "measured"),
        )
        .unwrap();
    missing.events.remove(index);
    let bytes = canonical_bytes(&missing).unwrap();
    assert!(engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations
    )
    .is_err());
    let mut altered = log.clone();
    if let SearchEvent::Finished { cost, .. } = &mut altered.events[index].event {
        cost.reason.clear();
    }
    let bytes = canonical_bytes(&altered).unwrap();
    assert!(engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations
    )
    .is_err());
    let mut wrong_run = manifest.value().clone();
    wrong_run.run_id = "another-run".into();
    // A valid new run with its own original plan is still not this event stream's run.
    let wrong_run = Decoded::new(
        RunManifest::new(
            "another-run",
            "fixture-publisher",
            &"a".repeat(40),
            2,
            wrong_run.units.iter().map(|u| u.spec.clone()).collect(),
        )
        .unwrap(),
    )
    .unwrap();
    let bytes = canonical_bytes(&log).unwrap();
    assert!(engine::search_events(
        &wrong_run,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations
    )
    .is_err());
}

#[test]
fn unknown_exposure_is_retained_without_replay_authority() {
    let (log, manifest, bindings, observations) = fixture();
    assert!(matches!(
        log.binding.role_exposure,
        EvidenceRef::Unknown { .. }
    ));
    let bytes = canonical_bytes(&log).unwrap();
    let mapped = engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &bindings,
        &observations,
    )
    .unwrap();
    assert!(mapped
        .mapped
        .objects
        .iter()
        .all(|o| o.kind == jope_learning::progress::inventory::Kind::Event));
    assert!(mapped
        .events
        .iter()
        .all(|e: &Decoded<Event>| e.value().run_id == "search-ingest"));
}

#[test]
fn produced_search_sidecar_and_original_observation_reach_progress_census() {
    let temp = tempfile::tempdir().unwrap();
    let rows: Vec<_> = (0..40)
        .map(|i| vec![i as f64, 2. * i as f64 + 1.])
        .collect();
    let cut = jope_pipeline::prepare::build(
        &rows,
        "search-event-ingest",
        7,
        1,
        &temp.path().join("prepared"),
    )
    .unwrap()
    .cuts
    .remove(0);
    let mut binding = fixture().0.binding;
    binding.context_sha256 = cut_context_sha256(&cut).unwrap();
    binding.source_sha256 = source_sha256();
    let empty = SearchEventLog::new(binding.clone()).unwrap();
    let request = empty
        .request_id(
            &sha256_hex(BOOTSTRAP_GENOME.as_bytes()),
            &canonical_recipe(BOOTSTRAP_GENOME),
        )
        .unwrap();
    let spec = WorkSpec {
        stage: "gp-fit".into(),
        subject: "bootstrap".into(),
        inputs: vec![request.clone()],
    };
    let work_id = spec.work_id(&binding.run_id).unwrap();
    let manifest = Decoded::new(
        RunManifest::new(
            &binding.run_id,
            "fixture-publisher",
            &"a".repeat(40),
            1,
            vec![spec],
        )
        .unwrap(),
    )
    .unwrap();
    let options = jope_pipeline::search::SearchOptions {
        event_binding: Some(binding),
        ..jope_pipeline::search::SearchOptions::new(
            jope_pipeline::search::Budget {
                evaluations: 1,
                seconds: 60,
            },
            7,
        )
    };
    let searched =
        jope_pipeline::search::run_cuts_with(&[cut], &options, &temp.path().join("searched"))
            .unwrap();
    let bytes = std::fs::read(searched[0].output.join("search-events.json")).unwrap();
    let observations = searched[0]
        .observations
        .iter()
        .map(|o| serde_json::to_vec(o).unwrap())
        .collect::<Vec<_>>();
    let mapped = engine::search_events(
        &manifest,
        &bytes,
        &sha256_hex(&bytes),
        &[SearchWorkBinding {
            request_id: request,
            work_id,
        }],
        &observations,
    )
    .unwrap();
    if let Some(path) = std::env::var_os("I05_EVIDENCE_DIR") {
        let path = std::path::PathBuf::from(path);
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("produced-search-events.json"), &bytes).unwrap();
        std::fs::write(
            path.join("produced-original-observation.json"),
            &observations[0],
        )
        .unwrap();
        std::fs::write(
            path.join("produced-progress-events.json"),
            serde_json::to_vec_pretty(&mapped.events.iter().map(|e| e.value()).collect::<Vec<_>>())
                .unwrap(),
        )
        .unwrap();
    }
    assert_eq!(mapped.reconciliation.evaluation_attempts, 1);
    assert_eq!(mapped.reconciliation.logical_completed, 1);
    assert!(census::reconcile(&manifest, &mapped.events, &[]).reconciled());
    let canonical =
        jope_records::wire_contract::CanonicalPayload::from_canonical_bytes(&bytes).unwrap();
    assert_eq!(
        canonical.domain_separated_id().unwrap().sha256_hex(),
        identity(&SearchEventLog::from_bytes(&bytes).unwrap()).unwrap()
    );
}
