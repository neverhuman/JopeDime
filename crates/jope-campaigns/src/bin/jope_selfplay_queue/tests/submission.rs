use super::*;
use serde_json::json;

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
        ensure_native_reserved_attempts(&tx, &schema, &manifest, &owner, "jte-fixture").unwrap();
        update_pairs_submitted(&tx, &schema, &manifest, &owner, "jte-fixture").unwrap();
        update_native_attempts_submitted(&tx, &schema, &manifest, &owner, "jte-fixture").unwrap();
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
