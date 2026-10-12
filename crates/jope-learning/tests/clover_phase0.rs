use anyhow::bail;
use jope_learning::{
    canon::{canonical_json, canonical_sha256, sha256_hex},
    clover::*,
};
use serde_json::json;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

fn digest(s: &str) -> String {
    sha256_hex(s.as_bytes())
}
fn scratch() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}
fn identity() -> ExperimentIdentity {
    ExperimentIdentity {
        schema: SCHEMA.into(),
        source: SourceIdentity {
            commit: "a".repeat(40),
            closure_sha256: digest("full source closure"),
        },
        initial_state: InitialState::ModelFree {
            search_state_sha256: digest("initial GP population and controller state"),
            rng_state_sha256: digest("initial rng"),
        },
        evaluator_sha256: digest("evaluator"),
        parity_contract_sha256: digest("parity"),
        support_sha256: digest("program set"),
        objective_sha256: digest("objective"),
        data_roster_sha256: digest("roster"),
        seed: 0,
        budget: Budget {
            attempts: 2,
            wall_seconds: 60,
            cpu_seconds: 60,
            gpu_seconds: 0,
            memory_bytes: 1024,
            daily_bytes: DAILY_BYTES_CAP,
            daily_puts: DAILY_PUTS_CAP,
        },
        guards: ["quantile", "label_visibility", "hold_recovery"]
            .into_iter()
            .map(|s| (s.into(), digest(s)))
            .collect(),
    }
}

#[test]
fn identity_is_canonical_complete_and_model_free() {
    let id = identity();
    let wire = serde_json::to_value(&id).unwrap();
    let parsed: ExperimentIdentity = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(id.sha256().unwrap(), parsed.sha256().unwrap());
    assert_eq!(
        id.sha256().unwrap(),
        sha256_hex(&canonical_json(&id).unwrap())
    );
    for field in wire.as_object().unwrap().keys() {
        let mut missing = wire.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<ExperimentIdentity>(missing).is_err(),
            "missing {field}"
        );
    }
    let mut extra = wire;
    extra["lane"] = json!("CONTROL");
    assert!(serde_json::from_value::<ExperimentIdentity>(extra).is_err());
    for field in ["search_state_sha256", "rng_state_sha256"] {
        let mut missing = serde_json::to_value(&id).unwrap();
        missing["initial_state"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        assert!(serde_json::from_value::<ExperimentIdentity>(missing).is_err());
    }
}

#[test]
fn every_comparison_dimension_changes_identity_and_invalid_pins_fail() {
    let changes: Vec<fn(&mut ExperimentIdentity)> = vec![
        |i| i.source.commit = "b".repeat(40),
        |i| i.source.closure_sha256 = digest("other source"),
        |i| {
            i.initial_state = InitialState::Checkpoint {
                parameters_sha256: digest("params"),
                optimizer_sha256: digest("optimizer"),
                rng_state_sha256: digest("rng"),
            }
        },
        |i| i.evaluator_sha256 = digest("other eval"),
        |i| i.parity_contract_sha256 = digest("other parity"),
        |i| i.support_sha256 = digest("expanded support"),
        |i| i.objective_sha256 = digest("other objective"),
        |i| i.data_roster_sha256 = digest("other data"),
        |i| i.seed = 1,
        |i| i.budget.attempts = 3,
        |i| {
            i.guards
                .insert("hold_recovery".into(), digest("other guard"))
                .map(|_| ())
                .unwrap()
        },
    ];
    let original = identity().sha256().unwrap();
    for change in changes {
        let mut id = identity();
        change(&mut id);
        assert_ne!(id.sha256().unwrap(), original);
    }
    let invalid: Vec<fn(&mut ExperimentIdentity)> = vec![
        |i| i.source.commit.truncate(8),
        |i| i.source.closure_sha256.clear(),
        |i| i.evaluator_sha256 = "unbound".into(),
        |i| i.support_sha256.clear(),
        |i| i.guards.clear(),
        |i| i.budget.attempts = 0,
        |i| i.budget.daily_bytes += 1,
        |i| i.budget.daily_puts += 1,
        |i| {
            i.initial_state = InitialState::Checkpoint {
                parameters_sha256: digest("params"),
                optimizer_sha256: String::new(),
                rng_state_sha256: digest("rng"),
            }
        },
    ];
    for change in invalid {
        let mut id = identity();
        change(&mut id);
        assert!(id.sha256().is_err());
    }
}

#[test]
fn conflicting_lane_labels_are_rejected_after_restart() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    drop(RunStore::create(&path, "CONTROL", identity()).unwrap());
    RunStore::open(&path, "CONTROL", identity()).unwrap();
    let mut other = identity();
    other.seed = 1;
    assert!(RunStore::open(&path, "CONTROL", other).is_err());
    RunStore::open(&path, "ordinary-gp", identity()).unwrap();
    let mut invalid = identity();
    invalid.evaluator_sha256.clear();
    let absent = tmp.path().join("invalid.sqlite");
    assert!(RunStore::open(&absent, "bad", invalid).is_err());
    assert!(!absent.exists());
}

fn visibility(labels: bool) -> Visibility {
    Visibility {
        task: "regression".into(),
        family: "family-a".into(),
        split_sha256: digest("split"),
        fold: 0,
        available_through: 10,
        fitting_rows: vec!["fit-a".into(), "fit-b".into()],
        forbidden_rows: BTreeSet::from(["held".into(), "future".into()]),
        training_labels: labels,
    }
}
fn observations() -> BTreeMap<String, Observation> {
    ["fit-a", "fit-b", "held", "future"]
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            (
                id.into(),
                Observation {
                    task: "regression".into(),
                    family: "family-a".into(),
                    role: if i < 2 { "training" } else { "confirmation" }.into(),
                    split_sha256: digest("split"),
                    fold: 0,
                    available_at: if i == 3 { 11 } else { 10 },
                    features: vec![i as f64, (i * i) as f64],
                    label: i as f64 + 100.,
                },
            )
        })
        .collect()
}

#[test]
fn feature_only_projection_never_reads_labels_heldout_or_future_payloads() {
    let mut rows = observations();
    let scope = visibility(false);
    let before = canonical_json(&project_encoder_input(&scope, &rows).unwrap()).unwrap();
    for row in rows.values_mut() {
        row.label = f64::NAN;
    }
    rows.get_mut("held").unwrap().features = vec![f64::NAN];
    rows.remove("future");
    assert_eq!(
        before,
        canonical_json(&project_encoder_input(&scope, &rows).unwrap()).unwrap()
    );
    assert!(project_encoder_input(&scope, &rows)
        .unwrap()
        .labels
        .is_none());
}

#[test]
fn supervised_projection_preserves_only_authorized_training_labels() {
    let mut rows = observations();
    let scope = visibility(true);
    let before = project_encoder_input(&scope, &rows).unwrap();
    assert_eq!(before.labels, Some(vec![100., 101.]));
    rows.get_mut("held").unwrap().label = 9e100;
    rows.get_mut("future").unwrap().features = vec![f64::NAN];
    assert_eq!(
        canonical_sha256(&before).unwrap(),
        canonical_sha256(&project_encoder_input(&scope, &rows).unwrap()).unwrap()
    );
    rows.get_mut("fit-a").unwrap().label = 123.;
    let after = project_encoder_input(&scope, &rows).unwrap();
    assert_eq!(before.features, after.features);
    assert_ne!(before.labels, after.labels);
}

#[test]
fn encoder_rejects_task_role_split_fold_family_time_and_membership_leakage() {
    let changes: Vec<fn(&mut Observation)> = vec![
        |r| r.task = "binary".into(),
        |r| r.family = "sealed-family".into(),
        |r| r.split_sha256 = digest("other split"),
        |r| r.fold = 1,
        |r| r.role = "validation".into(),
        |r| r.role = "confirmation".into(),
        |r| r.role = "delivery".into(),
        |r| r.available_at = 11,
    ];
    for change in changes {
        let mut rows = observations();
        change(rows.get_mut("fit-a").unwrap());
        assert!(project_encoder_input(&visibility(true), &rows).is_err());
    }
    for ids in [
        vec!["fit-a", "fit-a"],
        vec!["fit-a", "held"],
        vec!["fit-a", "missing"],
    ] {
        let mut scope = visibility(true);
        scope.fitting_rows = ids.into_iter().map(str::to_owned).collect();
        assert!(project_encoder_input(&scope, &observations()).is_err());
    }
}

#[test]
fn actual_preprocessing_and_target_copy_guard_respect_projected_inputs() {
    use jope_parallel::prepare::{reject_target_copies, Preprocessing};
    let mut rows = observations();
    // The production preprocessing implementation receives the projected training cut only.
    let prepare = |rows: &BTreeMap<String, Observation>| {
        let input = project_encoder_input(&visibility(true), rows).unwrap();
        let labels = input.labels.unwrap();
        let raw = input
            .features
            .into_iter()
            .zip(labels)
            .map(|(x, y)| {
                x.into_iter()
                    .chain([y])
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pp = Preprocessing::fit(&raw, "regression").unwrap();
        (canonical_json(&pp).unwrap(), pp.transform(&raw).unwrap())
    };
    let before = prepare(&rows);
    rows.get_mut("held").unwrap().label = 99999.;
    rows.get_mut("future").unwrap().features = vec![1e90];
    assert_eq!(before, prepare(&rows));
    let copied = (0..4)
        .map(|i| vec![i.to_string(), i.to_string()])
        .collect::<Vec<_>>();
    assert!(reject_target_copies(&["label_copy".into()], &copied).is_err());
}

fn recovery(id: &ExperimentIdentity, generation: u64) -> Vec<u8> {
    canonical_json(
        &json!({"schema":"clover.recovery/v1","experiment_sha256":id.sha256().unwrap(),
        "hold_generation":generation,"guard_sha256":id.guards["hold_recovery"]}),
    )
    .unwrap()
}

#[test]
fn hold_persists_and_explicit_recovery_is_generation_and_identity_bound() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let id = identity();
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let run = RunStore::create(&path, "gp", id.clone()).unwrap();
    run.hold("quantile guard").unwrap();
    drop(run);
    let mut run = RunStore::open(&path, "gp", id.clone()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("a").is_err());
    assert_eq!(run.recover().unwrap(), 0);
    assert!(run.require_ready().is_err());
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.hold("another violation").unwrap();
    assert!(run.release_hold(generation, &manifest, &evidence).is_err());
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.release_hold(generation, &manifest, &evidence).unwrap();
    run.require_ready().unwrap();
    drop(run);
    RunStore::open(&path, "gp", id)
        .unwrap()
        .require_ready()
        .unwrap();
}

#[test]
fn interrupted_attempts_terminalize_unknown_without_reward_or_budget_refund() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let id = identity();
    let mut run = RunStore::create(&path, "gp", id.clone()).unwrap();
    run.start_attempt("lost-worker").unwrap();
    drop(run);
    let mut run = RunStore::open(&path, "gp", id.clone()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("before-reconcile").is_err());
    assert_eq!(run.recover().unwrap(), 1);
    assert_eq!(run.recover().unwrap(), 0);
    assert!(run.require_ready().is_err());
    assert!(run
        .finish_attempt("lost-worker", AttemptOutcome::Qualified, Some(1.))
        .is_err());
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let generation = run.hold_generation().unwrap();
    let manifest = evidence
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    run.release_hold(generation, &manifest, &evidence).unwrap();
    run.start_attempt("second").unwrap();
    assert!(run
        .finish_attempt("second", AttemptOutcome::Unknown, Some(0.))
        .is_err());
    run.finish_attempt("second", AttemptOutcome::Unknown, None)
        .unwrap();
    assert!(run.start_attempt("third").is_err());
    let db = rusqlite::Connection::open(path).unwrap();
    let unknown: u64 = db
        .query_row(
            "SELECT count(*) FROM attempts WHERE outcome='\"unknown\"' AND reward IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(unknown, 2);
}

#[test]
fn every_publication_boundary_is_manifest_last_and_retry_is_immutable() {
    let id = identity();
    let payloads = vec![b"attempt failed".to_vec(), b"attempt unknown".to_vec()];
    for boundary in 0..4 {
        let tmp = scratch();
        let store = EvidenceStore::open(tmp.path()).unwrap();
        let mut writes = Vec::new();
        assert!(store
            .publish(&id, &payloads, |i, h| {
                writes.push(h.to_owned());
                if i == boundary {
                    bail!("injected crash/lost response");
                }
                Ok(())
            })
            .is_err());
        let before = writes
            .iter()
            .map(|h| (h.clone(), store.read(h).unwrap()))
            .collect::<Vec<_>>();
        let mut retry = Vec::new();
        let manifest = store
            .publish(&id, &payloads, |_, h| {
                retry.push(h.to_owned());
                Ok(())
            })
            .unwrap();
        assert_eq!(retry.last(), Some(&manifest));
        for (h, bytes) in before {
            assert_eq!(store.read(&h).unwrap(), bytes);
        }
        store.verify_manifest(&manifest, &id).unwrap();
        if boundary < 3 {
            assert!(!writes.contains(&manifest));
        }
    }
}

#[test]
fn publication_fault_persists_hold_across_controller_restart() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let evidence = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let run = RunStore::create(&path, "gp", identity()).unwrap();
    assert!(run
        .publish_evidence(&evidence, &[b"result".to_vec()], |_, _| bail!(
            "lost object response"
        ))
        .is_err());
    drop(run);
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("after-publish-fault").is_err());
    assert!(run
        .publish_evidence(&evidence, &[b"result".to_vec()], |_, _| Ok(()))
        .is_err());
}

#[test]
fn corrupt_missing_or_foreign_evidence_never_releases_hold() {
    let tmp = scratch();
    let id = identity();
    let store = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let mut run = RunStore::create(&tmp.path().join("state.sqlite"), "gp", id.clone()).unwrap();
    run.hold("fault").unwrap();
    let generation = run.hold_generation().unwrap();
    let manifest = store
        .publish(&id, &[recovery(&id, generation)], |_, _| Ok(()))
        .unwrap();
    let mut foreign = id.clone();
    foreign.seed = 1;
    assert!(store.verify_manifest(&manifest, &foreign).is_err());
    assert!(run
        .release_hold(generation, &digest("missing"), &store)
        .is_err());
    let object = sha256_hex(&recovery(&id, generation));
    fs::write(
        tmp.path().join("evidence/objects").join(&object),
        b"tampered",
    )
    .unwrap();
    assert!(store.put(&recovery(&id, generation)).is_err());
    assert!(run.release_hold(generation, &manifest, &store).is_err());
    assert!(run.require_ready().is_err());
}

#[test]
fn failed_recovery_transaction_keeps_unknown_work_fenced() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("interrupted").unwrap();
    drop(run);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute_batch("CREATE TRIGGER injected_failure BEFORE UPDATE ON attempts BEGIN SELECT RAISE(ABORT,'injected storage fault'); END;").unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.recover().is_err());
    assert!(run.require_ready().is_err());
    assert!(run.start_attempt("new").is_err());
    drop(run);
    db.execute_batch("DROP TRIGGER injected_failure;").unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert_eq!(run.recover().unwrap(), 1);
    assert!(run.require_ready().is_err());
}

#[test]
fn missing_payload_and_corrupted_earlier_readback_block_manifest_publication() {
    let tmp = scratch();
    let store = EvidenceStore::open(tmp.path()).unwrap();
    let mut written = Vec::new();
    assert!(store
        .publish(&identity(), &[b"attempt".to_vec()], |index, hash| {
            written.push(hash.to_owned());
            if index == 1 {
                fs::write(
                    tmp.path().join("objects").join(&written[0]),
                    b"corrupt earlier readback",
                )?;
            }
            Ok(())
        })
        .is_err());
    assert_eq!(
        written.len(),
        2,
        "manifest was published after failed closure read-back"
    );
    let tmp = scratch();
    let store = EvidenceStore::open(tmp.path()).unwrap();
    let missing = canonical_json(
        &json!({"schema":"clover.evidence/v1", "experiment_sha256": identity().sha256().unwrap(),
        "objects":[identity().sha256().unwrap(),digest("absent payload")]}),
    )
    .unwrap();
    let hash = store.put(&missing).unwrap();
    assert!(store.verify_manifest(&hash, &identity()).is_err());
}

#[test]
fn unrecognized_persisted_state_is_terminal_unknown_and_charged() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("alien-state").unwrap();
    drop(run);
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute(
        "UPDATE attempts SET outcome='unrecognized-after-partial-recovery'",
        [],
    )
    .unwrap();
    let mut run = RunStore::open(&path, "gp", identity()).unwrap();
    assert!(run.require_ready().is_err());
    assert_eq!(run.recover().unwrap(), 1);
    assert!(run.require_ready().is_err());
    let (outcome, reward): (String, Option<f64>) = db
        .query_row("SELECT outcome,reward FROM attempts", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(outcome, "\"unknown\"");
    assert_eq!(reward, None);
}

#[test]
fn over_budget_publication_writes_nothing_and_latches_hold() {
    let tmp = scratch();
    let store = EvidenceStore::open(&tmp.path().join("evidence")).unwrap();
    let mut id = identity();
    id.budget.daily_bytes = 1;
    let run = RunStore::create(&tmp.path().join("state.sqlite"), "gp", id).unwrap();
    assert!(run
        .publish_evidence(&store, &[b"result".to_vec()], |_, _| Ok(()))
        .is_err());
    assert!(run.require_ready().is_err());
    assert_eq!(
        fs::read_dir(tmp.path().join("evidence/objects"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn missing_or_partial_registry_never_resets_hold_or_attempt_accounting() {
    let tmp = scratch();
    let path = tmp.path().join("state.sqlite");
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    assert!(!path.exists());
    let mut run = RunStore::create(&path, "gp", identity()).unwrap();
    run.start_attempt("charged").unwrap();
    run.hold("guard").unwrap();
    drop(run);
    assert!(RunStore::create(&path, "gp", identity()).is_err());
    let retained = tmp.path().join("retained.sqlite");
    fs::rename(&path, &retained).unwrap();
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    assert!(!path.exists());
    fs::rename(retained, &path).unwrap();
    RunStore::open(&path, "gp", identity())
        .unwrap()
        .require_ready()
        .unwrap_err();
    let db = rusqlite::Connection::open(&path).unwrap();
    db.execute("DELETE FROM holds", []).unwrap();
    assert!(RunStore::open(&path, "gp", identity()).is_err());
    let count: u64 = db
        .query_row("SELECT count(*) FROM holds", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}
