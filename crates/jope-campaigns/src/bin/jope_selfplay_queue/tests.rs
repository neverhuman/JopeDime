#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn keeper_workers_follow_usable_cpus_and_stay_bounded() {
        let usable = thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
        assert_eq!(keeper_workers(), usable.min(MAX_KEEPER_WORKERS));
        assert!((1..=MAX_KEEPER_WORKERS).contains(&keeper_workers()));
    }

    #[test]
    fn rendered_spec_pins_every_input_and_tool_drift_fails_the_cycle() {
        let root = tempdir().unwrap();
        let write = |name: &str, bytes: &[u8]| {
            let path = root.path().join(name);
            fs::write(&path, bytes).unwrap();
            path
        };
        let runner = write("runner.sh", b"#!/bin/sh\n");
        let evaluator = write("corpus_eval", b"evaluator-v1");
        let train = write("train.csv", b"x,y\n1,2\n");
        let test = write("test.csv", b"x,y\n3,4\n");
        let manifest_path = write("manifest.json", b"{}");
        let manifest = ValidManifest {
            path: manifest_path.clone(),
            sha256: sha256_file(&manifest_path).unwrap(),
            generated_at_ms: 0,
            pairs: Vec::new(),
            value: json!({
                "train_csv": train.to_string_lossy(),
                "test_csv": test.to_string_lossy(),
                "train_sha256": sha256_file(&train).unwrap(),
                "test_sha256": sha256_file(&test).unwrap(),
            }),
        };
        let runner_sha256 = sha256_file(&runner).unwrap();
        let evaluator_sha256 = sha256_file(&evaluator).unwrap();
        let tools = PinnedTools {
            runner: (runner.as_path(), &runner_sha256),
            evaluator: (evaluator.as_path(), &evaluator_sha256),
        };
        let spec = root.path().join("job.toml");
        render_spec(&spec, "jte-test", &manifest, &tools).unwrap();
        let text = fs::read_to_string(&spec).unwrap();
        for digest in [
            &runner_sha256,
            &evaluator_sha256,
            &manifest.sha256,
            &sha256_file(&train).unwrap(),
            &sha256_file(&test).unwrap(),
        ] {
            assert!(text.contains(&format!("sha256 = \"{digest}\"")), "{text}");
        }
        tools.verify_unchanged().unwrap();

        // A manifest-declared input that drifted is still refused per manifest.
        fs::write(&train, b"x,y\n9,9\n").unwrap();
        let error = render_spec(&spec, "jte-test", &manifest, &tools).unwrap_err();
        assert!(
            format!("{error:#}").contains("content pin drift"),
            "{error:#}"
        );

        // A tool replaced after the cycle hashed it fails the cycle.
        fs::write(&evaluator, b"evaluator-v2").unwrap();
        let error = tools.verify_unchanged().unwrap_err();
        assert!(format!("{error:#}").contains("corpus_eval"), "{error:#}");
    }

    #[test]
    fn manifest_metric_map_agrees_with_task_authority() {
        use jope_dime_regressor::task::TaskKind;
        for kind in [
            TaskKind::Regression,
            TaskKind::Binary,
            TaskKind::Timeseries,
            TaskKind::Multiclass,
        ] {
            assert_eq!(
                expected_manifest_metric(kind.as_str()).unwrap(),
                kind.primary_metric()
            );
        }
        assert_eq!(expected_manifest_metric("multi_label"), None);
    }

    fn checkpoint() -> String {
        "a".repeat(64)
    }

    fn manifest_value(wave: &str, epoch: &str) -> Value {
        let checkpoint = checkpoint();
        let genome = "(pipe X (ridge))";
        let train_sha256 = "d".repeat(64);
        let test_sha256 = "e".repeat(64);
        let protocol_sha256 = jope_dime_regressor::selfplay_runtime::evaluation_protocol_sha256(
            "regression",
            "r2",
            "test_exact",
            &train_sha256,
            &test_sha256,
        )
        .unwrap();
        json!({
            "schema": MANIFEST_SCHEMA,
            "strict_wave": wave,
            "task": "regression",
            "dataset_hash": "dataset",
            "checkpoint_epoch": epoch,
            "checkpoint_sha256": checkpoint,
            "catalog_sha256": "b".repeat(64),
            "protocol_sha256": protocol_sha256,
            "generated_at_ms": 10,
            "micro_lane": "least_observed",
            "train_csv": "/train.csv",
            "test_csv": "/test.csv",
            "train_sha256": train_sha256,
            "test_sha256": test_sha256,
            "label_authority": {
                "score_tier": "test_exact",
                "metric": "r2",
                "test_adaptive": true,
                "benchmark_valid": false,
                "generation": wave,
                "model_checkpoint_sha256": checkpoint,
            },
            "candidates": [{
                "canonical_genome": genome,
                "semantic_sha256": sha256_bytes(genome.as_bytes()),
                "model_checkpoint_sha256": checkpoint,
            }]
        })
    }

    fn generated_registry() -> GeneratedOperatorRegistry {
        GeneratedOperatorRegistry::from_bytes(
            br#"{
              "registry_version":"generated-v1",
              "registry_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
              "maximum_output_columns":512,
              "operators":[
                {"id":0,"name":"ridge","role":"head","regression":true,"estimated_cost":2,
                 "lifecycle":{"parseable":true,"quarantined":false,"deprecated":false}}
              ]
            }"#,
        )
        .unwrap()
    }

    #[test]
    fn generated_registry_preflight_runs_before_reservation() {
        let root = tempdir().unwrap();
        let path = root.path().join("manifest.json");
        fs::write(&path, b"manifest").unwrap();
        let mut manifest = ValidManifest {
            path,
            sha256: "b".repeat(64),
            generated_at_ms: 1,
            pairs: Vec::new(),
            value: manifest_value("strict-wave-000001", "epoch"),
        };
        let accepted = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
        assert_eq!(accepted.disposition, TaskPreflightDisposition::Accepted);

        manifest.value["task"] = json!("binary");
        let rejected = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
        assert_eq!(
            rejected.disposition,
            TaskPreflightDisposition::PermanentRejection
        );
        assert_eq!(rejected.estimated_compute_avoided, rejected.requested_work);
    }

    #[test]
    fn typed_preflight_receipts_are_append_only() {
        let root = tempdir().unwrap();
        let connection = Connection::open_in_memory().unwrap();
        initialize_backlog(&connection).unwrap();
        let path = root.path().join("manifest.json");
        fs::write(&path, b"manifest").unwrap();
        let manifest = ValidManifest {
            path: path.clone(),
            sha256: sha256_file(&path).unwrap(),
            generated_at_ms: 1,
            pairs: Vec::new(),
            value: manifest_value("strict-wave-000001", "epoch"),
        };
        let accepted = preflight_manifest(&generated_registry(), &manifest, None).unwrap();
        record_preflight_receipt(&connection, &path, &accepted).unwrap();
        record_preflight_receipt(&connection, &path, &accepted).unwrap();
        let count: i64 = connection
            .query_row("SELECT COUNT(*) FROM task_preflight_receipts", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 1);

        let rejected = TaskPreflightOutcomeV1::failed_regression_screening(
            &accepted,
            REGRESSION_SCREENING_BUDGET_SECONDS,
        )
        .unwrap();
        assert!(record_preflight_receipt(&connection, &path, &rejected).is_err());
    }

    #[test]
    fn bounded_screening_requires_valid_candidate_receipts() {
        let root = tempdir().unwrap();
        let receipts = root.path().join("dataset.receipts.jsonl");
        fs::write(
            &receipts,
            b"{\"schema\":\"receipt\",\"valid\":true,\"failure\":null}\n",
        )
        .unwrap();
        validate_bounded_screening_receipts(root.path()).unwrap();

        fs::write(&receipts, b"{\"valid\":false,\"failure\":\"timeout\"}\n").unwrap();
        assert!(validate_bounded_screening_receipts(root.path())
            .unwrap_err()
            .to_string()
            .contains("timeout"));
    }

    #[test]
    fn queue_policy_has_a_dynamic_slot_horizon_and_held_is_not_a_queue_state() {
        let targets = dynamic_queue_targets(&[20.0, 30.0, 40.0], 200).unwrap();
        assert_eq!(targets.target, 2_048);
        assert_eq!(targets.hard_cap, 4_096);
        assert_eq!(dynamic_queue_targets(&[20.0], 2_000).unwrap().target, 2_500);
        assert!(QUEUE_STATES.contains(&"queued"));
        assert!(QUEUE_STATES.contains(&"starting"));
        assert!(!QUEUE_STATES.contains(&"held"));
        assert!(!QUEUE_STATES.contains(&"running"));
    }

    #[test]
    fn held_release_only_fills_the_gap_under_target() {
        // The 2026-08-01 pileup: 996 held, queue empty, target 2048 — release
        // a full batch, never more than the batch cap.
        assert_eq!(held_release_budget(0, 2_048, 996), HELD_RELEASE_BATCH_MAX);
        // Queue already at/over target: a busy cluster releases nothing, so
        // the overload guard's decision is respected while pressure is real.
        assert_eq!(held_release_budget(2_048, 2_048, 996), 0);
        assert_eq!(held_release_budget(3_000, 2_048, 996), 0);
        // Fewer held than gap: release exactly what exists.
        assert_eq!(held_release_budget(2_040, 2_048, 3), 3);
        // Gap smaller than batch cap: release exactly the gap.
        assert_eq!(held_release_budget(2_000, 2_048, 996), 48);
        // A zero/negative target (fail-closed cycles) releases nothing.
        assert_eq!(held_release_budget(0, 0, 996), 0);
        assert_eq!(held_release_budget(0, -5, 996), 0);
    }

    #[test]
    fn held_multiclass_jobs_are_outside_the_release_contract() {
        let job = SchedulerJob {
            id: "jope-mc272-protected".to_owned(),
            project: "jope-multiclass-test-a".to_owned(),
            state: "held".to_owned(),
            failure_class: None,
            queue_ts: None,
        };

        assert!(!scheduler_job_is_adoptable(&job));
    }

    #[test]
    fn failure_report_preserves_the_queue_status_envelope() {
        let value = serde_json::to_value(failure_report(true, "boom")).unwrap();
        assert_eq!(value["schema"], "JopeSelfplayQueueKeeper.v3");
        assert_eq!(value["admission_allowed"], false);
        assert_eq!(value["degraded"], true);
        assert_eq!(value["error"], "boom");
        for field in [
            "queued_target",
            "queued_hard_cap",
            "queued_starting",
            "dispatching",
            "admission_waves",
            "unclaimed_current_checkpoint_manifests",
        ] {
            assert!(value.get(field).is_some(), "missing {field}");
        }
    }

    #[test]
    fn backlog_initialization_does_not_rebuild_the_active_attempt_index() {
        let connection = Connection::open_in_memory().unwrap();
        initialize_backlog(&connection).unwrap();
        let before: i64 = connection
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        initialize_backlog(&connection).unwrap();
        let after: i64 = connection
            .query_row("PRAGMA schema_version", [], |row| row.get(0))
            .unwrap();
        assert_eq!(after, before);
        let sql: String = connection
            .query_row(
                "SELECT sql FROM sqlite_master WHERE name='keeper_one_active_attempt'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(sql.contains("deferred_horizon"));
    }

    #[test]
    fn manifest_validation_is_bounded_to_the_next_admission_batch() {
        assert_eq!(bounded_validation_budget(1_408, 512), 576);
        assert_eq!(bounded_validation_budget(40, 512), 40);
        assert_eq!(bounded_validation_budget(100, 1), 2);
        assert_eq!(bounded_validation_budget(100, 0), 0);
        assert!(evaluator_failure_is_permanent(&anyhow::anyhow!(
            "Rust evaluator rejected manifest"
        )));
        assert!(!evaluator_failure_is_permanent(&anyhow::anyhow!(
            "run Rust evaluator: resource temporarily unavailable"
        )));
    }

    #[test]
    fn admission_waves_are_oldest_first_and_exclude_claimed_manifests() {
        let root = tempdir().unwrap();
        let first = root
            .path()
            .join("waves/strict-wave-000001/manifests/first.json");
        let second = root
            .path()
            .join("waves/strict-wave-000002/manifests/second.json");
        fs::create_dir_all(first.parent().unwrap()).unwrap();
        fs::create_dir_all(second.parent().unwrap()).unwrap();
        fs::write(&first, br#"{"candidates":[{},{}]}"#).unwrap();
        fs::write(&second, br#"{"candidates":[{},{},{}]}"#).unwrap();
        let waves = vec![
            MicroWave {
                wave: "strict-wave-000001".into(),
                summary_sha256: "a".repeat(64),
            },
            MicroWave {
                wave: "strict-wave-000002".into(),
                summary_sha256: "b".repeat(64),
            },
        ];
        let mut claims = HashMap::new();
        claims.insert(
            "claimed".into(),
            Claim {
                manifest_sha256: "c".repeat(64),
                manifest: first,
                job_id: "jte-claimed".into(),
                spec: root.path().join("claimed.toml"),
                state: "submitted".into(),
                attempts: 1,
                owner: Some("owner".into()),
            },
        );
        let admission =
            current_admission_waves(root.path(), &waves, &claims, &BTreeSet::new()).unwrap();
        assert_eq!(admission.len(), 1);
        assert_eq!(admission[0].wave, "strict-wave-000002");
        assert_eq!(admission[0].unclaimed_manifests, 1);
        assert_eq!(admission[0].candidate_evaluation_count, 3);
        let rejected = BTreeSet::from([second]);
        assert!(
            current_admission_waves(root.path(), &waves, &claims, &rejected)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn newest_two_complete_waves_are_never_admitted() {
        let wave = |index: usize, unclaimed: usize, complete: bool| AdmissionWave {
            wave: format!("strict-wave-{index:06}"),
            summary_sha256: format!("{index:064x}"),
            manifest_count: 10,
            unclaimed_manifests: unclaimed,
            candidate_evaluation_count: (unclaimed * 50) as u64,
            complete,
        };
        let (admission, reserve) = partition_protected_reserve(vec![
            wave(1, 4, false),
            wave(2, 10, true),
            wave(3, 10, true),
            wave(4, 10, true),
        ]);
        assert_eq!(
            admission
                .iter()
                .map(|wave| wave.wave.as_str())
                .collect::<Vec<_>>(),
            ["strict-wave-000001", "strict-wave-000002"]
        );
        assert_eq!(
            reserve
                .iter()
                .map(|wave| wave.wave.as_str())
                .collect::<Vec<_>>(),
            ["strict-wave-000003", "strict-wave-000004"]
        );
        assert!(reserve.iter().all(|wave| wave.complete));
    }

    #[test]
    fn filters_stale_epoch_old_wave_and_protected_lane() {
        let root = tempdir().unwrap();
        let path = root.path().join("manifest.json");
        let current_epoch = format!("checkpoint-{}", checkpoint());
        atomic_json(&path, &manifest_value("strict-wave-000009", &current_epoch)).unwrap();
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_ok()
        );
        assert!(validate_manifest(
            &path,
            &checkpoint(),
            "checkpoint-stale",
            "strict-wave-000009"
        )
        .is_err());
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000008").is_err()
        );
        let mut protected = manifest_value("strict-wave-000009", &current_epoch);
        protected["evaluation_lane"] = json!("sealed_audit");
        atomic_json(&path, &protected).unwrap();
        assert!(
            validate_manifest(&path, &checkpoint(), &current_epoch, "strict-wave-000009").is_err()
        );
    }

    #[test]
    fn restart_submitting_claim_is_idempotently_adoptable() {
        let root = tempdir().unwrap();
        let connection = Connection::open(root.path().join("backlog.sqlite")).unwrap();
        initialize_backlog(&connection).unwrap();
        let genome = "(pipe X (ridge))";
        let manifest = ValidManifest {
            path: root.path().join("manifest.json"),
            sha256: "a".repeat(64),
            generated_at_ms: 1,
            pairs: vec![PairKey {
                task: "regression".into(),
                dataset_hash: "dataset".into(),
                semantic_sha256: sha256_bytes(genome.as_bytes()),
                protocol_sha256: "b".repeat(64),
            }],
            value: json!({}),
        };
        for _ in 0..2 {
            let tx = connection.unchecked_transaction().unwrap();
            append_keeper_attempts(&tx, &manifest, "wave", "jte-a", "submitting", 1).unwrap();
            tx.commit().unwrap();
        }
        let attempts: i64 = connection
            .query_row("SELECT COUNT(*) FROM keeper_attempts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(attempts, 1);
        let live = SchedulerJob {
            id: "jte-a".into(),
            project: PROJECT.into(),
            state: "queued".into(),
            failure_class: None,
            queue_ts: None,
        };
        assert_eq!(live.project, PROJECT);
        assert!(ACTIVE_JOB_STATES.contains(&live.state.as_str()));
        assert!(scheduler_job_is_adoptable(&live));
        let completed = SchedulerJob {
            state: "completed".into(),
            ..live
        };
        assert!(scheduler_job_is_adoptable(&completed));
    }

    #[test]
    fn append_keeper_attempts_treats_sql_shaped_pair_keys_as_values() {
        let connection = Connection::open_in_memory().unwrap();
        initialize_backlog(&connection).unwrap();
        let hostile = "' OR 1=1; DROP TABLE keeper_attempts; --";
        let manifest = ValidManifest {
            path: PathBuf::from("/manifest.json"),
            sha256: "a".repeat(64),
            generated_at_ms: 1,
            pairs: vec![PairKey {
                task: hostile.to_owned(),
                dataset_hash: hostile.to_owned(),
                semantic_sha256: hostile.to_owned(),
                protocol_sha256: hostile.to_owned(),
            }],
            value: json!({}),
        };

        let first = connection.unchecked_transaction().unwrap();
        append_keeper_attempts(&first, &manifest, "owner", "old-job", "completed", 1).unwrap();
        first.commit().unwrap();

        let retry = connection.unchecked_transaction().unwrap();
        append_keeper_attempts(&retry, &manifest, "owner", "new-job", "submitting", 0).unwrap();
        retry.commit().unwrap();

        let inserted: (String, String, String, String, i64) = connection
            .query_row(
                "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number \
                 FROM keeper_attempts WHERE job_id=?",
                ["new-job"],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            inserted,
            (
                hostile.to_owned(),
                hostile.to_owned(),
                hostile.to_owned(),
                hostile.to_owned(),
                2
            )
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = ?1",
                    ["keeper_attempts"],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    fn submission_transition_fixture(
        number_column: &'static str,
        owner_column: &'static str,
    ) -> (Connection, PairSchema, ValidManifest, String) {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(&format!(
                "CREATE TABLE pairs(
                   task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   state TEXT NOT NULL,{owner_column} TEXT NOT NULL,
                   manifest_sha256 TEXT,job_id TEXT,updated_at_ms INTEGER NOT NULL,
                   PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256));
                 CREATE TABLE attempts(
                   id INTEGER PRIMARY KEY,task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
                   semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
                   {number_column} INTEGER NOT NULL,state TEXT NOT NULL,
                   {owner_column} TEXT NOT NULL,manifest_path TEXT,
                   manifest_sha256 TEXT,job_id TEXT,
                   created_at_ms INTEGER NOT NULL DEFAULT 0,
                   UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,{number_column}));"
            ))
            .unwrap();
        let owner = "queue-keeper:fixture".to_owned();
        let manifest_sha256 = "a".repeat(64);
        let protocol_sha256 = "b".repeat(64);
        let selected_semantic = "c".repeat(64);
        let unrelated_semantic = "d".repeat(64);
        for semantic in [&selected_semantic, &unrelated_semantic] {
            connection
                .execute(
                    &format!(
                        "INSERT INTO pairs(task,dataset_hash,semantic_sha256,protocol_sha256,\
                         state,{owner_column},manifest_sha256,updated_at_ms) \
                         VALUES('regression','dataset',?,?,'reserved',?,?,1)"
                    ),
                    params![semantic, protocol_sha256, owner, manifest_sha256],
                )
                .unwrap();
            connection
                .execute(
                    &format!(
                        "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,\
                         {number_column},state,{owner_column},manifest_sha256) \
                         VALUES('regression','dataset',?,?,1,'reserved',?,?)"
                    ),
                    params![semantic, protocol_sha256, owner, manifest_sha256],
                )
                .unwrap();
        }
        let manifest = ValidManifest {
            path: PathBuf::from("/manifest.json"),
            sha256: manifest_sha256,
            generated_at_ms: 1,
            pairs: vec![PairKey {
                task: "regression".into(),
                dataset_hash: "dataset".into(),
                semantic_sha256: selected_semantic,
                protocol_sha256,
            }],
            value: json!({}),
        };
        (
            connection,
            PairSchema {
                owner_column,
                has_infrastructure_retries: false,
                attempts: Some(AttemptSchema {
                    number_column,
                    owner_column,
                }),
            },
            manifest,
            owner,
        )
    }

    fn assert_exact_submission_transition(number_column: &'static str, owner_column: &'static str) {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture(number_column, owner_column);
        for _ in 0..2 {
            let tx = connection.unchecked_transaction().unwrap();
            ensure_native_reserved_attempts(&tx, &schema, &manifest, &owner, "jte-fixture")
                .unwrap();
            update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-fixture").unwrap();
            update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-fixture")
                .unwrap();
            tx.commit().unwrap();
        }
        let selected_pair: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM pairs WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            selected_pair,
            ("submitted".into(), Some("jte-fixture".into()))
        );
        let unrelated_pair: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM pairs WHERE semantic_sha256=?",
                ["d".repeat(64)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(unrelated_pair, ("reserved".into(), None));
        let selected_attempt: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM attempts WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            selected_attempt,
            ("submitted".into(), Some("jte-fixture".into()))
        );
        let unrelated_attempt: (String, Option<String>) = connection
            .query_row(
                "SELECT state,job_id FROM attempts WHERE semantic_sha256=?",
                ["d".repeat(64)],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(unrelated_attempt, ("reserved".into(), None));
    }

    #[test]
    fn exact_submission_transition_is_idempotent_on_live_attempt_no_schema() {
        assert_exact_submission_transition("attempt_no", "reservation_owner");
    }

    #[test]
    fn exact_submission_repairs_a_pre_attempt_ledger_reservation() {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture("attempt_no", "reservation_owner");
        connection
            .execute(
                "DELETE FROM attempts WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
            )
            .unwrap();
        let tx = connection.unchecked_transaction().unwrap();
        ensure_native_reserved_attempts(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-repaired").unwrap();
        tx.commit().unwrap();
        let row: (String, Option<String>, String) = connection
            .query_row(
                "SELECT state,job_id,reservation_owner FROM attempts \
                 WHERE semantic_sha256=?",
                [manifest.pairs[0].semantic_sha256.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            row,
            ("submitted".into(), Some("jte-repaired".into()), owner)
        );
    }

    #[test]
    fn exact_submission_transition_is_idempotent_on_legacy_attempt_number_schema() {
        assert_exact_submission_transition("attempt_number", "owner");
    }

    #[test]
    fn exact_submission_transition_rejects_a_different_job() {
        let (connection, schema, manifest, owner) =
            submission_transition_fixture("attempt_no", "reservation_owner");
        let tx = connection.unchecked_transaction().unwrap();
        update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-first").unwrap();
        update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-first").unwrap();
        tx.commit().unwrap();
        let tx = connection.unchecked_transaction().unwrap();
        assert!(
            update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-other")
                .unwrap_err()
                .to_string()
                .contains("job conflict")
        );
    }

    #[test]
    fn rfc3339_parser_handles_fractional_utc_and_offset() {
        assert_eq!(parse_rfc3339_seconds("1970-01-01T00:00:00.000Z"), Some(0));
        assert_eq!(parse_rfc3339_seconds("1970-01-01T01:00:00+01:00"), Some(0));
    }

    #[test]
    fn only_explicit_infrastructure_classes_retry() {
        assert!(is_infrastructure_failure_class("storage-error"));
        assert!(is_infrastructure_failure_class("RPC-ERROR"));
        assert!(!is_infrastructure_failure_class("task-invalid"));
        assert!(!is_infrastructure_failure_class("timeout"));
    }

    #[test]
    fn node_slots_use_live_schema_without_fabricating_xbabe4_multiplier() {
        let value = json!({
            "Nodes": {
                "nodes": [
                    {
                        "node": "xbabe4",
                        "cores": 24,
                        "effective_cores": 20,
                        "running_jobs": 3
                    },
                    {
                        "node": "xbabe5",
                        "cores": 64,
                        "effective_cores": 56,
                        "cpu_job_slot_multiplier": 1.5,
                        "job_slot_target": 84,
                        "running_job_slots": 80
                    }
                ]
            }
        });
        let nodes = parse_nodes(&value).unwrap();
        assert_eq!(nodes[0].cpu_job_slot_multiplier, 1.0);
        assert_eq!(nodes[0].slot_target, 20);
        assert_eq!(nodes[0].running_slots, 3);
        assert_eq!(nodes[1].cpu_job_slot_multiplier, 1.5);
        assert_eq!(nodes[1].slot_target, 84);
        assert_eq!(nodes[1].running_slots, 80);
    }
}
