use super::*;
use tempfile::tempdir;

fn pinned_test_spec(root: &Path, job_identity: &str) -> PathBuf {
    let bundle_sha256 = "e".repeat(64);
    let bundle = root.join("runtime-bundles/bundles").join(&bundle_sha256);
    let pythonpath = bundle.join("pytorch");
    fs::create_dir_all(pythonpath.join("proposals")).unwrap();
    fs::create_dir_all(pythonpath.join("shared")).unwrap();
    fs::write(
        bundle.join("bundle.json"),
        serde_json::to_vec(&json!({"bundle_sha256": bundle_sha256})).unwrap(),
    )
    .unwrap();
    let generator = pythonpath.join("proposals/test_exact_dl_generate.py");
    let common = pythonpath.join("shared/test_exact_common.py");
    fs::write(&generator, b"generator").unwrap();
    fs::write(&common, b"common").unwrap();
    let input_root = root.join("inputs");
    fs::create_dir_all(&input_root).unwrap();
    let pointer = input_root.join("pointer.json");
    let catalog = input_root.join("catalog.json");
    let data = input_root.join("contexts.f32.bin");
    let checkpoint = input_root.join("checkpoint.pt");
    let cohort = input_root.join("cohort.json");
    for (path, body) in [
        (&pointer, b"pointer".as_slice()),
        (&catalog, b"catalog".as_slice()),
        (&data, b"data".as_slice()),
        (&checkpoint, b"checkpoint".as_slice()),
        (&cohort, b"cohort".as_slice()),
    ] {
        fs::write(path, body).unwrap();
    }
    let mut lines = vec![
        "[job]".to_owned(),
        format!("id = {}", toml_quote(job_identity).unwrap()),
        String::new(),
        "[env]".to_owned(),
        format!(
            "PYTHONPATH = {}",
            toml_quote(&pythonpath.to_string_lossy()).unwrap()
        ),
        String::new(),
    ];
    for (name, path) in [
        ("generator", generator.as_path()),
        ("common", common.as_path()),
        ("pointer", pointer.as_path()),
        ("catalog", catalog.as_path()),
        ("data", data.as_path()),
        ("checkpoint", checkpoint.as_path()),
        ("cohort", cohort.as_path()),
    ] {
        append_spec_input(&mut lines, name, path, name).unwrap();
    }
    let directory = root.join("specs/cohort-propose");
    fs::create_dir_all(&directory).unwrap();
    let spec = directory.join(format!("{job_identity}.toml"));
    fs::write(&spec, lines.join("\n")).unwrap();
    spec
}

fn audit_args(root: &Path) -> Args {
    Args {
        root: root.to_owned(),
        build: false,
        once: true,
        interval_seconds: 5,
        queue_floor: 2_048,
        queued_target: 2_048,
        queued_hard_cap: 4_096,
        ready_reserve_min: 2_048,
        useful_work_min: MIN_USEFUL_WORK,
        candidate_work_min: MIN_CANDIDATE_EVALUATION_WORK,
        candidate_reserve_min: MIN_CANDIDATE_RESERVE,
        min_complete_reserve_waves: MIN_COMPLETE_RESERVE_WAVES,
        max_overlap_waves: 8,
        max_builds_per_cycle: 1,
        max_candidates_per_manifest: 50,
        queue_status_max_age_seconds: 30,
        performance_max_age_seconds: 300,
        scq: PathBuf::from("/unused-scq"),
        wave_builder: PathBuf::from("/unused"),
        proposal_generator: PathBuf::from("/unused-generator"),
        proposal_legacy_generator: PathBuf::from("/unused-legacy-generator"),
        proposal_common: PathBuf::from("/unused-common"),
        proposal_pythonpath: PathBuf::from("/unused-pythonpath"),
        catalog_pointer: None,
        checkpoint_pointer: None,
        proposals_pointer: None,
        corpus: None,
        sealed_cohort: None,
        robust_seed_bank: None,
        pair_registry: None,
        queue_status: None,
    }
}

#[test]
fn decision_builds_to_target_with_bounded_overlap() {
    let decision = supply_decision(
        2_048, 240_000, 65_536, 4, 2, 2_048, 2_048, 262_144, 4_096, 65_536, 2, 8, 1, 50,
    );
    assert!(!decision.below_floor);
    assert!(!decision.overloaded);
    assert_eq!(decision.candidate_evaluation_work, 240_000);
    assert_eq!(decision.estimated_manifests_per_wave, 768);
    assert_eq!(decision.builds_planned, 1);

    // Scheduler occupancy at the hard cap never prevents building ready
    // reserve; the cap applies only to queued/starting/dispatching jobs.
    let scheduler_capped = supply_decision(
        4_096, 204_800, 0, 0, 0, 2_048, 2_048, 262_144, 4_096, 65_536, 2, 8, 1, 50,
    );
    assert!(scheduler_capped.overloaded);
    assert_eq!(scheduler_capped.builds_planned, 1);

    let complete_reserve_deficit = supply_decision(
        2_048, 262_144, 32_768, 3, 1, 2_048, 2_048, 262_144, 4_096, 65_536, 2, 8, 1, 50,
    );
    assert_eq!(complete_reserve_deficit.candidate_evaluation_work, 262_144);
    assert_eq!(complete_reserve_deficit.builds_planned, 1);

    let infeasible = supply_decision(
        0, 0, 0, 0, 0, 2_048, 2_048, 500_000, 4_096, 65_536, 2, 8, 1, 50,
    );
    assert!(!infeasible.target_feasible);
    assert_eq!(infeasible.builds_planned, 0);
}

#[test]
fn candidate_work_target_has_an_eight_wave_floor_and_increases_dynamically() {
    let directory = tempdir().unwrap();
    let args = audit_args(directory.path());
    let path = directory.path().join("ops/supply-performance.v1.json");
    atomic_json(
        &path,
        &json!({
            "schema": "SupplyPerformance.v1",
            "generated_at_ms": now_ms(),
            "p95_jobs_per_minute": 10.0,
            "p95_candidate_evaluations_per_minute": 100.0,
            "proposer_to_ready_minutes_p95": 60.0,
            "proposer_samples": 4,
        }),
    )
    .unwrap();
    assert_eq!(
        performance_measurements(&args)
            .unwrap()
            .candidate_evaluation_work_target,
        262_144
    );

    atomic_json(
        &path,
        &json!({
            "schema": "SupplyPerformance.v1",
            "generated_at_ms": now_ms(),
            "p95_jobs_per_minute": 60.0,
            "p95_candidate_evaluations_per_minute": 3_000.0,
            "proposer_to_ready_minutes_p95": 60.0,
            "proposer_samples": 4,
        }),
    )
    .unwrap();
    assert_eq!(
        performance_measurements(&args)
            .unwrap()
            .candidate_evaluation_work_target,
        360_000
    );
}

#[test]
fn scheduler_hard_cap_is_independent_from_ready_reserve() {
    let directory = tempdir().unwrap();
    let mut args = audit_args(directory.path());
    assert!(validate_args(&args).is_ok());
    args.max_candidates_per_manifest = 24;
    assert!(validate_args(&args).is_err());
    args.max_candidates_per_manifest = 25;
    args.queued_hard_cap = 3_000;
    assert!(validate_args(&args).is_ok());
}

/// The proposer must NOT be placement-pinned to the state-root node.
///
/// It used to emit `[dependencies] binaries = ["<root>/catalog/current.json"]`,
/// a path that exists only on xbabe2. On 2026-08-04 that deadlocked the
/// flywheel: the mobile trainer held xbabe2's one GPU under a 168-hour cap
/// while the immobile proposer queued behind it forever and three other GPUs
/// sat idle. Everything the proposer reads is staged as a content-pinned
/// `[[input]]`, and node capability is covered by the GPU and
/// `import jope` smoke probes, so the pin bought nothing.
#[test]
fn the_cohort_proposer_is_not_pinned_to_one_node() {
    let directory = tempdir().unwrap();
    let args = audit_args(directory.path());
    assert!(
        proposer_placement_pin_lines(&args).unwrap().is_empty(),
        "a placement pin would confine the proposer to the state-root node"
    );
}

/// A busy cluster must queue, never fail the supply cycle.
///
/// scq probes dependencies only on nodes that already have the resources
/// free, so a healthy cluster rejects a submit whenever every capable node is
/// busy. Naming one probe literally is what produced 44 consecutive
/// `cycle_failed` records once already (see
/// `spec_policy::PROBE_FLYWHEEL_ROOT_XBABE2_ONLY`), so this matches the SHAPE
/// of a probe failure instead.
#[test]
fn any_busy_node_smoke_rejection_is_transient_not_fatal() {
    // The historical literal, still transient.
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: xbabe3: smoke proposal GPU capacity: \
             smoke test exited with exit 1"
    ));
    // The bundle probe the proposer actually carries.
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: xbabe0: smoke RuntimeBundle JOPE Python \
             package: smoke test exited with exit 1"
    ));
    // The probe named by spec_policy for this family.
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: xbabe1: smoke jope-prime policy source \
             present: smoke test exited with exit 1"
    ));
    assert!(temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: no eligible launch-ready node currently"
    ));
    // A real configuration error must still be fatal.
    assert!(!temporarily_lacks_launch_capacity(
        "error[spec-invalid]: unknown field `node_affinity`"
    ));
    assert!(!temporarily_lacks_launch_capacity(
        "error[dependency-unsatisfied]: binary /usr/bin/nope: not found in PATH"
    ));
}

#[test]
fn immutable_proposer_spec_survives_equivalent_bundle_activation() {
    let directory = tempdir().unwrap();
    let old_runtime = directory.path().join("old/pytorch");
    let new_runtime = directory.path().join("new/pytorch");
    for runtime in [&old_runtime, &new_runtime] {
        fs::create_dir_all(runtime.join("proposals")).unwrap();
        fs::create_dir_all(runtime.join("shared")).unwrap();
        fs::create_dir_all(runtime.join("jope/policy")).unwrap();
        fs::create_dir_all(runtime.join("registry")).unwrap();
        fs::write(
            runtime.join("proposals/test_exact_context_proposals.py"),
            b"context-generator",
        )
        .unwrap();
        fs::write(
            runtime.join("proposals/test_exact_dl_generate.py"),
            b"legacy-generator",
        )
        .unwrap();
        fs::write(runtime.join("shared/test_exact_common.py"), b"common").unwrap();
        fs::write(runtime.join("jope/__init__.py"), b"").unwrap();
        fs::write(runtime.join("jope/policy/model_v3.py"), b"model").unwrap();
        fs::write(runtime.join("jope/policy/decode.py"), b"decode").unwrap();
        fs::write(
            runtime.join("registry/operator_registry.json"),
            b"operator-registry",
        )
        .unwrap();
    }
    let mut args = audit_args(directory.path());
    args.proposal_pythonpath = new_runtime.clone();
    args.proposal_generator = new_runtime.join("proposals/test_exact_context_proposals.py");
    args.proposal_legacy_generator = new_runtime.join("proposals/test_exact_dl_generate.py");
    args.proposal_common = new_runtime.join("shared/test_exact_common.py");
    let existing = format!(
            "[env]\nPYTHONPATH = \"{}\"\nJOPE_OPERATOR_REGISTRY = \"{}\"\nOMP_NUM_THREADS = \"4\"\n\n[[input]]\nsource = \"{}\"\n",
            old_runtime.display(),
            old_runtime.join("registry/operator_registry.json").display(),
            old_runtime.join("registry/operator_registry.json").display(),
        );
    let expected = format!(
            "[env]\nPYTHONPATH = \"{}\"\nJOPE_OPERATOR_REGISTRY = \"{}\"\nOMP_NUM_THREADS = \"4\"\n\n[[input]]\nsource = \"{}\"\n",
            new_runtime.display(),
            new_runtime.join("registry/operator_registry.json").display(),
            new_runtime.join("registry/operator_registry.json").display(),
        );
    assert!(validate_proposer_runtime(&existing, &expected, &args).is_ok());

    fs::write(
        old_runtime.join("proposals/test_exact_context_proposals.py"),
        b"drift",
    )
    .unwrap();
    assert!(validate_proposer_runtime(&existing, &expected, &args).is_err());

    fs::write(
        old_runtime.join("proposals/test_exact_context_proposals.py"),
        b"context-generator",
    )
    .unwrap();
    fs::write(
        old_runtime.join("registry/operator_registry.json"),
        b"registry-drift",
    )
    .unwrap();
    assert!(validate_proposer_runtime(&existing, &expected, &args).is_err());

    let mutable_checkout = existing.replace(
        &old_runtime.to_string_lossy().to_string(),
        &jope_paths::to_string(&paths::python_root()),
    );
    assert!(validate_proposer_runtime(&mutable_checkout, &expected, &args).is_err());
}

#[test]
fn terminal_proposer_retry_preserves_cohort_and_advances_attempt() {
    let directory = tempdir().unwrap();
    let original_spec = pinned_test_spec(directory.path(), "cohort-proposer");
    let pinned = runtime_pinned_intent_from_spec(&original_spec, 0).unwrap();
    let mut state: PersistedSupplyState = serde_json::from_value(json!({
        "schema": "CohortSupplyState.v2",
        "wave": "strict-wave-000190",
        "checkpoint_sha256": "c".repeat(64),
        "cohort": "/tmp/cohort.json",
        "cohort_sha256": "a".repeat(64),
        "catalog_sha256": "b".repeat(64),
        "decoder_configuration_sha256": "d".repeat(64),
        "seed": 20260726,
        "stage": "submit_cohort_proposer",
        "revision": 1,
        "recovery_accounting_only": false,
        "proposal_job_id": "cohort-proposer",
        "runtime_pinned_attempts": [pinned]
    }))
    .unwrap();
    assert_eq!(
        state
            .retry_proposer(directory.path(), "cancelled", now_ms())
            .unwrap(),
        ProposerRetryDisposition::Retried
    );
    assert_eq!(state.proposal_attempt, 1);
    assert_eq!(
        state.proposal_job_id.as_deref(),
        Some("cohort-proposer-infra000002")
    );
    assert_eq!(state.revision, 2);
    assert_eq!(state.cohort_sha256, "a".repeat(64));
    assert_eq!(state.runtime_pinned_attempts.len(), 2);
    assert_eq!(
        state.runtime_pinned_attempts[0].runtime_bundle_sha256,
        state.runtime_pinned_attempts[1].runtime_bundle_sha256
    );
    assert_eq!(
        state.runtime_pinned_attempts[0].proposer_inputs,
        state.runtime_pinned_attempts[1].proposer_inputs
    );
    let payload = proposer_scientific_payload(&state, &state.runtime_pinned_attempts[0]).unwrap();
    let projection = proposer_recovery_projection(directory.path(), &state, &payload).unwrap();
    assert_eq!(projection.scientific_failures, 0);
    assert_eq!(projection.foreign_cancellation_timestamps_ms.len(), 1);
    assert!(projection.retry_pending_submission);
    let primary = serde_json::to_string(&state).unwrap();
    for forbidden in [
        "scientific_failures",
        "foreign_cancellations",
        "retry_pending_submission",
        "proposal_submitted_at_ms",
        "next_retry_not_before_ms",
    ] {
        assert!(!primary.contains(forbidden));
    }
}

#[test]
fn retry_five_foreign_cancellation_recovers_as_attempt_seven() {
    let directory = tempdir().unwrap();
    let original_spec = pinned_test_spec(directory.path(), "cohort-proposer");
    let pinned = runtime_pinned_intent_from_spec(&original_spec, 0).unwrap();
    let mut state: PersistedSupplyState = serde_json::from_value(json!({
        "schema": "CohortSupplyState.v2",
        "wave": "strict-wave-000190",
        "checkpoint_sha256": "c".repeat(64),
        "cohort": "/tmp/cohort.json",
        "cohort_sha256": "a".repeat(64),
        "catalog_sha256": "b".repeat(64),
        "decoder_configuration_sha256": "d".repeat(64),
        "seed": 20260726,
        "stage": "submit_cohort_proposer",
        "revision": 1,
        "recovery_accounting_only": false,
        "proposal_job_id": "cohort-proposer",
        "runtime_pinned_attempts": [pinned]
    }))
    .unwrap();
    let mut observed = now_ms();
    for _ in 0..=5 {
        state
            .retry_proposer(directory.path(), "cancelled", observed)
            .unwrap();
        let payload =
            proposer_scientific_payload(&state, state.runtime_pinned_attempts.last().unwrap())
                .unwrap();
        let mut projection =
            proposer_recovery_projection(directory.path(), &state, &payload).unwrap();
        projection
            .mark_submitted(
                &projection.current_job_identity.clone(),
                projection.retry_not_before_ms,
            )
            .unwrap();
        observed = projection.retry_not_before_ms;
        save_recovery_projection(directory.path(), &projection).unwrap();
    }
    assert_eq!(state.proposal_attempt, 6);
    assert_eq!(
        state.proposal_job_id.as_deref(),
        Some("cohort-proposer-infra000007")
    );
    assert_eq!(state.runtime_pinned_attempts.len(), 7);
    assert!(state.runtime_pinned_attempts.windows(2).all(|pair| {
        pair[0].runtime_bundle_sha256 == pair[1].runtime_bundle_sha256
            && pair[0].proposer_inputs == pair[1].proposer_inputs
    }));
}

#[test]
fn legacy_state_migrates_by_adopting_its_exact_existing_spec() {
    let directory = tempdir().unwrap();
    let spec = pinned_test_spec(directory.path(), "cohort-proposer");
    let immutable_spec_sha256 = sha256_file(&spec).unwrap();
    let wave = "strict-wave-000200";
    let checkpoint = "c".repeat(64);
    let legacy: PersistedSupplyState = serde_json::from_value(json!({
        "schema": "CohortSupplyState.v1",
        "wave": wave,
        "checkpoint_sha256": checkpoint,
        "cohort": "/tmp/cohort.json",
        "cohort_sha256": "a".repeat(64),
        "catalog_sha256": "b".repeat(64),
        "decoder_configuration_sha256": "d".repeat(64),
        "seed": 20260726,
        "stage": "submit_cohort_proposer",
        "revision": 1,
        "recovery_accounting_only": false,
        "proposal_job_id": "cohort-proposer"
    }))
    .unwrap();
    let legacy_path = supply_state_v1_path(directory.path(), wave);
    fs::create_dir_all(legacy_path.parent().unwrap()).unwrap();
    fs::write(&legacy_path, serde_json::to_vec(&legacy).unwrap()).unwrap();

    let migrated = load_supply_state(directory.path(), wave, &checkpoint)
        .unwrap()
        .unwrap();
    assert_eq!(migrated.schema, SUPPLY_STATE_V2_SCHEMA);
    assert_eq!(migrated.runtime_pinned_attempts.len(), 1);
    assert_eq!(
        migrated.runtime_pinned_attempts[0].immutable_spec_sha256,
        immutable_spec_sha256
    );
    assert!(legacy_path.is_file());
    assert!(supply_state_path(directory.path(), wave).is_file());
}

#[path = "tests/v2_cohort_contexts.rs"]
mod v2_cohort_contexts;

#[path = "tests_tail.rs"]
mod tests_tail;
