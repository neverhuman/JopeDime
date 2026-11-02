use super::*;

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
