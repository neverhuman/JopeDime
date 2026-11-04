use super::*;

#[test]
fn table_count_rejects_sql_shaped_table_names() {
    let conn = Connection::open_in_memory().unwrap();
    conn.execute_batch("CREATE TABLE run (id INTEGER); INSERT INTO run VALUES (1); CREATE TABLE sentinel (id INTEGER); INSERT INTO sentinel VALUES (7);")
        .unwrap();

    assert_eq!(table_count(&conn, "run"), 1);
    assert_eq!(table_count(&conn, "run; DROP TABLE sentinel; --"), -1);

    let sentinel_rows: i64 = conn
        .query_row("SELECT count(*) FROM sentinel", [], |row| row.get(0))
        .unwrap();
    assert_eq!(sentinel_rows, 1);
}

#[test]
fn targets_parse_with_and_without_ports() {
    let parsed = parse_targets("xbabe3,xbabe4:2224").unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(parsed[0].node, "xbabe3");
    assert_eq!(parsed[0].port, 22);
    assert_eq!(parsed[1].port, 2224);
    assert_eq!(parsed[1].label(), "xbabe4:2224");
}

/// Replicating to the source node would produce a "replica" on the very
/// disk this exists to escape, and the `xbabe2` ssh alias does not even
/// reach this machine.
#[test]
fn source_node_is_rejected_as_a_target() {
    assert!(parse_targets("xbabe2").is_err());
    assert!(parse_targets("xbabe3,xbabe2").is_err());
}

#[test]
fn live_database_and_sidecars_are_never_part_of_the_payload() {
    assert!(skipped("history.sqlite3"));
    assert!(skipped("history.sqlite3-wal"));
    assert!(skipped("history.sqlite3-shm"));
    assert!(skipped(".capture.lock"));
    assert!(skipped(".index.v1.json.1234.tmp"));
    assert!(!skipped("index.v1.json"));
    assert!(!skipped("train_metrics.jsonl"));
}

#[test]
fn manifest_objects_covers_payload_and_database() {
    let db_sha = "a".repeat(64);
    let file_sha = "b".repeat(64);
    let manifest = json!({
        "db": {"sha256": db_sha},
        "files": [{"path": "x", "sha256": file_sha}],
    });
    let objects = manifest_objects(&manifest);
    assert_eq!(objects.len(), 2);
    assert!(objects.contains(&db_sha));
    assert!(objects.contains(&file_sha));
}

/// Manifest names must sort chronologically, because remote GC decides what
/// to retain by taking the lexical tail of `ls -1`.
#[test]
fn manifest_names_sort_in_time_order() {
    let sha = "c".repeat(64);
    let early = manifest_name(&json!({"generated_at_ms": 1785213126163u64}), &sha);
    let later = manifest_name(&json!({"generated_at_ms": 1785299526163u64}), &sha);
    assert!(early < later, "{early} !< {later}");
}

#[test]
fn object_paths_fan_out_by_hash_prefix() {
    let sha = format!("ab{}", "0".repeat(62));
    assert_eq!(object_rel(&sha), format!("ab/{sha}"));
}

#[test]
fn only_lowercase_hex_digests_are_accepted() {
    assert!(is_sha256(&"a".repeat(64)));
    assert!(!is_sha256(&"A".repeat(64)));
    assert!(!is_sha256(&"a".repeat(63)));
    assert!(!is_sha256("not-a-hash"));
}

#[test]
fn shell_quoting_survives_a_single_quote() {
    assert_eq!(quote("plain"), "'plain'");
    assert_eq!(quote("it's"), "'it'\\''s'");
}
