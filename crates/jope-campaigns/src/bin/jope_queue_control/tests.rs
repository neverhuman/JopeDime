use super::*;
use std::os::unix::net::UnixListener;

fn record(project: &str, state: &str, epoch: &str, wave: Option<&str>) -> JobRecord {
    JobRecord {
        job_id: format!("{project}-{state}-{epoch}-{}", wave.unwrap_or("none")),
        project: project.to_owned(),
        state: state.to_owned(),
        epoch: epoch.to_owned(),
        strict_wave: wave.map(str::to_owned),
        generation: wave.map(str::to_owned),
        checkpoint_sha256: None,
        claim_state: Some("submitted".to_owned()),
        manifest: Some("/manifest".to_owned()),
        manifest_sha256: Some("a".repeat(64)),
        manifest_checked: true,
        manifest_valid: true,
        invalid_reason: None,
    }
}

#[test]
fn sqlite_column_exists_binds_table_and_column_names() {
    let connection = Connection::open_in_memory().unwrap();
    connection
        .execute_batch("CREATE TABLE pairs (reservation_owner TEXT);")
        .unwrap();

    assert!(sqlite_column_exists(&connection, "pairs", "reservation_owner").unwrap());
    assert!(!sqlite_column_exists(&connection, "pairs", "missing").unwrap());

    let hostile_table = "pairs') WHERE name=?1 OR 1=1 --";
    assert!(!sqlite_column_exists(&connection, hostile_table, "missing").unwrap());
    assert!(sqlite_table_exists(&connection, "pairs").unwrap());
}

#[test]
fn legacy_hold_excludes_running_and_unclaimed() {
    let mut rows = vec![
        record(LEGACY_PROJECT, "queued", LEGACY_EPOCH, None),
        record(LEGACY_PROJECT, "running", LEGACY_EPOCH, None),
    ];
    let mut unclaimed = record(LEGACY_PROJECT, "queued", LEGACY_EPOCH, None);
    unclaimed.claim_state = None;
    rows.push(unclaimed);
    assert_eq!(hold_candidates(&rows, true, false, None).len(), 1);
}

#[test]
fn superseded_wave_hold_keeps_newest_and_all_running() {
    let rows = vec![
        record(
            SELFPLAY_PROJECT,
            "queued",
            LEGACY_EPOCH,
            Some("strict-wave-000009"),
        ),
        record(
            SELFPLAY_PROJECT,
            "running",
            LEGACY_EPOCH,
            Some("strict-wave-000009"),
        ),
        record(
            SELFPLAY_PROJECT,
            "queued",
            LEGACY_EPOCH,
            Some("strict-wave-000010"),
        ),
    ];
    assert_eq!(newest_wave(&rows).as_deref(), Some("strict-wave-000010"));
    let held = hold_candidates(&rows, false, true, Some("strict-wave-000010"));
    assert_eq!(held.len(), 1);
    assert_eq!(held[0].strict_wave.as_deref(), Some("strict-wave-000009"));
}

#[test]
fn protected_lanes_fail_scope() {
    for lane in ["checkpoint_gate", "sealed_audit", "evaluator_canary"] {
        assert!(protected_lane(&json!({"evaluation_lane": lane})));
    }
}

#[test]
fn checkpoint_bridge_is_exact_bounded_and_newest_wave_first() {
    let checkpoint = "f".repeat(64);
    let mut older = record(
        SELFPLAY_PROJECT,
        "held",
        "checkpoint-1",
        Some("strict-wave-000009"),
    );
    older.job_id = "job-a".to_owned();
    older.checkpoint_sha256 = Some(checkpoint.clone());
    let mut newer_b = record(
        SELFPLAY_PROJECT,
        "held",
        "checkpoint-1",
        Some("strict-wave-000010"),
    );
    newer_b.job_id = "job-b".to_owned();
    newer_b.checkpoint_sha256 = Some(checkpoint.clone());
    let mut newer_a = newer_b.clone();
    newer_a.job_id = "job-a-new".to_owned();
    let mut wrong_checkpoint = older.clone();
    wrong_checkpoint.job_id = "job-wrong".to_owned();
    wrong_checkpoint.checkpoint_sha256 = Some("e".repeat(64));
    let mut running = newer_b.clone();
    running.job_id = "job-running".to_owned();
    running.state = "running".to_owned();

    let selected = checkpoint_bridge_candidates(
        &[older, newer_b, newer_a, wrong_checkpoint, running],
        &checkpoint,
        2,
    );
    assert_eq!(
        selected
            .iter()
            .map(|candidate| candidate.job_id.as_str())
            .collect::<Vec<_>>(),
        ["job-a-new", "job-b"]
    );

    let manifest = json!({
        "checkpoint_sha256": checkpoint.clone(),
        "label_authority": {"model_checkpoint_sha256": checkpoint.clone()},
        "candidates": [
            {"model_checkpoint_sha256": checkpoint.clone()},
            {"model_checkpoint_sha256": checkpoint.clone()}
        ]
    });
    assert_eq!(
        manifest_checkpoint_binding(&manifest).as_deref(),
        Some(checkpoint.as_str())
    );
    let mut drifted = manifest;
    drifted["candidates"][1]["model_checkpoint_sha256"] = json!("e".repeat(64));
    assert!(manifest_checkpoint_binding(&drifted).is_none());
}

#[test]
fn blocker_lookup_scans_desired_and_probes_pair_primary_key() {
    let connection = Connection::open_in_memory().unwrap();
    connection
        .execute_batch(
            "CREATE TABLE pairs(
               task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
               state TEXT NOT NULL,manifest_path TEXT,manifest_sha256 TEXT,
               job_id TEXT,receipt_sha256 TEXT,
               PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
             ) WITHOUT ROWID;
             CREATE TEMP TABLE desired_pairs(
               task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,
               PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)
             ) WITHOUT ROWID;",
        )
        .unwrap();
    let mut statement = connection
        .prepare(&format!("EXPLAIN QUERY PLAN {DESIRED_BLOCKER_QUERY}"))
        .unwrap();
    let plan = statement
        .query_map([], |row| row.get::<_, String>(3))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap()
        .join("\n");
    assert!(plan.contains("SCAN d"));
    assert!(plan.contains("SEARCH p USING PRIMARY KEY"));
}

#[test]
fn missing_receipt_recovery_requires_every_manifest_pair_to_be_active_and_unscored() {
    let binding = |suffix: &str,
                   state: &str,
                   path: &str,
                   job: &str,
                   receipt: Option<&str>,
                   failure: Option<&str>| ManifestPairBinding {
        identity: RegistryPairIdentity {
            task: "regression".to_owned(),
            dataset_hash: format!("dataset-{suffix}"),
            semantic_sha256: format!("semantic-{suffix}"),
            protocol_sha256: "protocol".to_owned(),
        },
        state: state.to_owned(),
        manifest_path: Some(path.to_owned()),
        job_id: Some(job.to_owned()),
        receipt_sha256: receipt.map(str::to_owned),
        failure_class: failure.map(str::to_owned),
    };
    let pairs = validate_missing_receipt_manifest_bindings(
        vec![
            binding("a", "reserved", "/manifest", "job", None, None),
            binding("b", "submitted", "/manifest", "job", None, None),
            binding("c", "held", "/manifest", "job", None, None),
        ],
        "/manifest",
        "job",
    )
    .unwrap();
    assert_eq!(pairs.len(), 3);

    for invalid in [
        binding("a", "available", "/manifest", "job", None, None),
        binding("a", "submitted", "/manifest", "job", Some("receipt"), None),
        binding(
            "a",
            "submitted",
            "/manifest",
            "job",
            None,
            Some("semantic_failure"),
        ),
        binding("a", "submitted", "/other", "job", None, None),
        binding("a", "submitted", "/manifest", "other", None, None),
    ] {
        assert!(
            validate_missing_receipt_manifest_bindings(vec![invalid], "/manifest", "job").is_none()
        );
    }
    assert!(validate_missing_receipt_manifest_bindings(Vec::new(), "/manifest", "job").is_none());
}

#[test]
fn seed_bank_requires_repeats_and_ranks_within_task_dataset() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("corpus.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE observations(
               task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
               candidate_source TEXT NOT NULL,score REAL NOT NULL
             );",
        )
        .unwrap();
    let strong = "(pipe X (ridge))";
    let weak = "(pipe X (standardize) (ridge))";
    let single = "(pipe X (rank) (ridge))";
    for (dataset, strong_score, weak_score) in [("dataset-a", 0.9, 0.2), ("dataset-b", 0.8, 0.1)] {
        for (genome, score) in [(strong, strong_score), (weak, weak_score)] {
            connection
                .execute(
                    "INSERT INTO observations VALUES(?1,?2,?3,?4,'gp',?5)",
                    rusqlite::params![
                        "regression",
                        dataset,
                        sha256_bytes(genome.as_bytes()),
                        genome,
                        score
                    ],
                )
                .unwrap();
        }
    }
    connection
        .execute(
            "INSERT INTO observations VALUES('regression','dataset-a',?1,?2,'gp',1.0)",
            rusqlite::params![sha256_bytes(single.as_bytes()), single],
        )
        .unwrap();
    drop(connection);

    let bank = export_seed_bank(&database, 10, 2).unwrap();
    let rows = bank["tasks"]["regression"].as_array().unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["canonical_genome"], strong);
    assert_eq!(rows[0]["dataset_count"], 2);
    assert!(rows
        .iter()
        .all(|row| row["canonical_genome"].as_str() != Some(single)));
}

fn seed(semantic: &str, mean: f64, dl_origin: bool) -> SeedRow {
    SeedRow {
        task: "regression".to_owned(),
        semantic_sha256: semantic.to_owned(),
        canonical_genome: semantic.to_owned(),
        dataset_count: 2,
        mean_within_dataset_percentile: mean,
        worst_within_dataset_percentile: mean,
        strict_win_count: 0,
        dl_origin,
    }
}

#[test]
fn merge_seed_pools_caps_dl_and_keeps_rank_order() {
    let non_dl = vec![
        seed("a", 0.9, false),
        seed("b", 0.5, false),
        seed("c", 0.3, false),
    ];
    // "b" also appears in the DL pool: it must count as non-DL and never
    // consume DL cap. Cap floor(4/3)=1 admits only the best novel DL seed.
    let dl = vec![
        seed("b", 0.95, true),
        seed("d", 0.8, true),
        seed("e", 0.7, true),
    ];
    let merged = merge_seed_pools(non_dl, dl, 4, 4 / 3);
    let order: Vec<(&str, bool)> = merged
        .iter()
        .map(|row| (row.semantic_sha256.as_str(), row.dl_origin))
        .collect();
    assert_eq!(
        order,
        vec![("a", false), ("d", true), ("b", false), ("c", false)]
    );
    // Overall limit still binds after the cap.
    let merged = merge_seed_pools(
        vec![seed("a", 0.9, false), seed("b", 0.5, false)],
        vec![seed("d", 0.8, true)],
        2,
        1,
    );
    let order: Vec<&str> = merged
        .iter()
        .map(|row| row.semantic_sha256.as_str())
        .collect();
    assert_eq!(order, vec!["a", "d"]);
    // Zero cap excludes DL entirely.
    assert!(merge_seed_pools(Vec::new(), vec![seed("d", 0.8, true)], 4, 0).is_empty());
}

#[test]
fn seed_bank_admits_dl_on_merit_capped_per_task() {
    let directory = tempfile::tempdir().unwrap();
    let database = directory.path().join("corpus.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE observations(
               task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
               semantic_sha256 TEXT NOT NULL,canonical_genome TEXT NOT NULL,
               candidate_source TEXT NOT NULL,score REAL NOT NULL
             );",
        )
        .unwrap();
    let strong = "(pipe X (ridge))";
    let weak = "(pipe X (standardize) (ridge))";
    let dl_top = "(pipe X (rank) (ridge))";
    let dl_second = "(pipe X (winsor) (ridge))";
    for dataset in ["dataset-a", "dataset-b"] {
        for (genome, source, score) in [
            (strong, "gp", 0.9),
            (weak, "gp", 0.2),
            (dl_top, "accepted_dl_checkpoint", 0.95),
            (dl_second, "accepted_dl_checkpoint", 0.5),
        ] {
            connection
                .execute(
                    "INSERT INTO observations VALUES(?1,?2,?3,?4,?5,?6)",
                    rusqlite::params![
                        "regression",
                        dataset,
                        sha256_bytes(genome.as_bytes()),
                        genome,
                        source,
                        score
                    ],
                )
                .unwrap();
        }
    }
    drop(connection);

    // limit=3 -> dl_cap=1: dl_top graduates on merit, dl_second is capped
    // out, and non-DL fills the remaining slots.
    let bank = export_seed_bank(&database, 3, 2).unwrap();
    let rows = bank["tasks"]["regression"].as_array().unwrap();
    assert_eq!(rows.len(), 3);
    let dl_rows: Vec<&Value> = rows.iter().filter(|row| row["dl_origin"] == true).collect();
    assert_eq!(dl_rows.len(), 1);
    assert_eq!(dl_rows[0]["canonical_genome"], dl_top);
    assert!(rows
        .iter()
        .all(|row| row["canonical_genome"].as_str() != Some(dl_second)));
    assert!(rows
        .iter()
        .any(|row| row["canonical_genome"].as_str() == Some(strong) && row["dl_origin"] == false));
    assert_eq!(bank["selection"]["dl_origin_cap_per_task"], 1);
    assert_eq!(
        bank["dl_origin_counts"]["regression"],
        json!({"total": 3, "dl_origin": 1, "non_dl": 2})
    );
}

#[test]
fn mutation_client_reuses_one_newline_json_socket() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("scqd.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut reader = BufReader::new(stream.try_clone().unwrap());
        for (operation, expected) in [("Hold", "job-a"), ("Cancel", "job-b"), ("Release", "job-c")]
        {
            let mut request = String::new();
            reader.read_line(&mut request).unwrap();
            let request: Value = serde_json::from_str(request.trim()).unwrap();
            assert_eq!(request[operation]["id"], expected);
            stream.write_all(b"{\"Ok\":null}\n").unwrap();
            stream.flush().unwrap();
        }
    });
    let mut client = ScqHoldClient::connect(&socket).unwrap();
    client.hold("job-a").unwrap();
    client.cancel("job-b").unwrap();
    client.release("job-c").unwrap();
    server.join().unwrap();
}

#[test]
fn mutation_reconnects_when_server_closes_between_requests() {
    let directory = tempfile::tempdir().unwrap();
    let socket = directory.path().join("scqd.sock");
    let listener = UnixListener::bind(&socket).unwrap();
    let server = std::thread::spawn(move || {
        for expected in ["job-a", "job-b"] {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = String::new();
            BufReader::new(stream.try_clone().unwrap())
                .read_line(&mut request)
                .unwrap();
            let request: Value = serde_json::from_str(request.trim()).unwrap();
            assert_eq!(request["Cancel"]["id"], expected);
            stream.write_all(b"{\"Ok\":null}\n").unwrap();
        }
    });
    let mut client = ScqHoldClient::connect(&socket);
    assert!(socket_mutation_with_reconnect(&mut client, &socket, "Cancel", "job-a").is_none());
    assert!(socket_mutation_with_reconnect(&mut client, &socket, "Cancel", "job-b").is_none());
    server.join().unwrap();
}

#[test]
fn retirement_transaction_preserves_history_and_is_idempotent() {
    let directory = tempfile::tempdir().unwrap();
    let pair_path = directory.path().join("pairs.sqlite");
    let backlog_path = directory.path().join("backlog.sqlite");
    let manifest_hash = "a".repeat(64);
    let preview_hash = "b".repeat(64);
    let job_id = "job-held";
    let pair = Connection::open(&pair_path).unwrap();
    pair.execute_batch(
        "CREATE TABLE pairs(
           task TEXT,dataset_hash TEXT,semantic_sha256 TEXT,protocol_sha256 TEXT,
           state TEXT,reservation_owner TEXT,manifest_path TEXT,
           manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,
           failure_class TEXT,updated_at_ms INTEGER
         );
         CREATE TABLE events(
           id INTEGER PRIMARY KEY,kind TEXT,task TEXT,dataset_hash TEXT,
           semantic_sha256 TEXT,protocol_sha256 TEXT,detail_json TEXT,
           created_at_ms INTEGER
         );
         CREATE INDEX pair_manifest ON pairs(manifest_sha256);",
    )
    .unwrap();
    pair.execute(
        "INSERT INTO pairs VALUES(
           'regression','dataset',?1,?2,'reserved','owner','/manifest',?3,
           NULL,NULL,NULL,1
         )",
        params!["c".repeat(64), "d".repeat(64), manifest_hash],
    )
    .unwrap();
    drop(pair);
    let backlog = Connection::open(&backlog_path).unwrap();
    backlog
        .execute_batch(
            "CREATE TABLE claims(
               manifest_sha256 TEXT,manifest TEXT,job_id TEXT,state TEXT,
               updated_at_ms INTEGER
             );",
        )
        .unwrap();
    backlog
        .execute(
            "INSERT INTO claims VALUES(?1,'/manifest',?2,'submitted',1)",
            params![manifest_hash, job_id],
        )
        .unwrap();
    drop(backlog);
    let candidate = HoldCandidate {
        job_id: job_id.to_owned(),
        project: LEGACY_PROJECT.to_owned(),
        observed_state: "cancelled".to_owned(),
        epoch: LEGACY_EPOCH.to_owned(),
        strict_wave: None,
        generation: None,
        manifest: "/manifest".to_owned(),
        manifest_sha256: manifest_hash.clone(),
    };
    let mut results = vec![RetirementResult {
        job_id: job_id.to_owned(),
        state_before_cancel: Some("cancelled".to_owned()),
        cancel_requested: false,
        command_error: None,
        state_after_cancel: Some("cancelled".to_owned()),
        pair_updates: 0,
        claim_updated: false,
        recovery_from_prior_preview: true,
        registry_error: None,
    }];
    apply_retirement_registry(
        &pair_path,
        &backlog_path,
        std::slice::from_ref(&candidate),
        &mut results,
        &preview_hash,
    )
    .unwrap();
    assert!(results[0].claim_updated);
    let pair = Connection::open(&pair_path).unwrap();
    let (state, retained_manifest): (String, Option<String>) = pair
        .query_row("SELECT state,manifest_sha256 FROM pairs", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap();
    assert_eq!(state, "available");
    assert!(retained_manifest.is_none());
    let events: i64 = pair
        .query_row("SELECT COUNT(*) FROM events", [], |row| row.get(0))
        .unwrap();
    assert_eq!(events, 1);
    let backlog = Connection::open(&backlog_path).unwrap();
    let state: String = backlog
        .query_row("SELECT state FROM claims", [], |row| row.get(0))
        .unwrap();
    assert_eq!(state, "superseded_unstarted");

    let mut rerun = vec![RetirementResult {
        recovery_from_prior_preview: true,
        state_after_cancel: Some("cancelled".to_owned()),
        command_error: None,
        cancel_requested: false,
        job_id: job_id.to_owned(),
        state_before_cancel: Some("cancelled".to_owned()),
        pair_updates: 0,
        claim_updated: false,
        registry_error: None,
    }];
    apply_retirement_registry(
        &pair_path,
        &backlog_path,
        &[candidate],
        &mut rerun,
        &preview_hash,
    )
    .unwrap();
    assert!(!rerun[0].claim_updated);

    let pair = Connection::open(&pair_path).unwrap();
    pair.execute(
        "UPDATE pairs SET state='submitted',reservation_owner='owner',
           manifest_sha256=?1,receipt_sha256=?2",
        params![manifest_hash, "e".repeat(64)],
    )
    .unwrap();
    let backlog = Connection::open(&backlog_path).unwrap();
    backlog
        .execute(
            "UPDATE claims SET state='submitted' WHERE job_id=?1",
            [job_id],
        )
        .unwrap();
    drop(pair);
    drop(backlog);
    let mut rejected = vec![RetirementResult {
        recovery_from_prior_preview: true,
        state_after_cancel: Some("cancelled".to_owned()),
        command_error: None,
        cancel_requested: false,
        job_id: job_id.to_owned(),
        state_before_cancel: Some("cancelled".to_owned()),
        pair_updates: 0,
        claim_updated: false,
        registry_error: None,
    }];
    let receipt_candidate = HoldCandidate {
        job_id: job_id.to_owned(),
        project: LEGACY_PROJECT.to_owned(),
        observed_state: "cancelled".to_owned(),
        epoch: LEGACY_EPOCH.to_owned(),
        strict_wave: None,
        generation: None,
        manifest: "/manifest".to_owned(),
        manifest_sha256: manifest_hash,
    };
    assert!(apply_retirement_registry(
        &pair_path,
        &backlog_path,
        &[receipt_candidate],
        &mut rejected,
        &preview_hash,
    )
    .is_err());
    let backlog = Connection::open(&backlog_path).unwrap();
    let state: String = backlog
        .query_row("SELECT state FROM claims", [], |row| row.get(0))
        .unwrap();
    assert_eq!(state, "submitted");
}
