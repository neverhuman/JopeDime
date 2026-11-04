use super::fill::{active_claim_manifests, flush_fill};
use super::*;

fn fixture(root: &Path) -> Result<()> {
    fs::create_dir_all(root.join("ops"))?;
    let backlog = Connection::open(root.join("ops/backlog.sqlite"))?;
    backlog.execute_batch(
        "CREATE TABLE claims(manifest_sha256 TEXT PRIMARY KEY,manifest TEXT NOT NULL UNIQUE,
           job_id TEXT NOT NULL UNIQUE,spec TEXT NOT NULL,state TEXT NOT NULL,
           attempts INTEGER NOT NULL DEFAULT 0,updated_at_ms INTEGER NOT NULL,
           reservation_owner TEXT);
         CREATE TABLE keeper_attempts(id INTEGER PRIMARY KEY,task TEXT NOT NULL,
           dataset_hash TEXT NOT NULL,semantic_sha256 TEXT NOT NULL,
           protocol_sha256 TEXT NOT NULL,attempt_number INTEGER NOT NULL,
           state TEXT NOT NULL,owner TEXT NOT NULL,manifest_path TEXT NOT NULL,
           manifest_sha256 TEXT NOT NULL,job_id TEXT NOT NULL,failure_class TEXT,
           created_at_ms INTEGER NOT NULL,finalized_at_ms INTEGER,
           UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_number));
         CREATE INDEX keeper_attempt_manifest_state ON keeper_attempts(manifest_sha256,state);",
    )?;
    let registry = Connection::open(root.join("ops/pair-registry.sqlite"))?;
    registry.execute_batch(
        "CREATE TABLE pairs(task TEXT NOT NULL,dataset_hash TEXT NOT NULL,
           semantic_sha256 TEXT NOT NULL,protocol_sha256 TEXT NOT NULL,canonical_genome TEXT,
           state TEXT NOT NULL,reservation_owner TEXT NOT NULL,manifest_path TEXT,
           manifest_sha256 TEXT,job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
           infrastructure_retries INTEGER NOT NULL DEFAULT 0,created_at_ms INTEGER NOT NULL,
           updated_at_ms INTEGER NOT NULL,
           PRIMARY KEY(task,dataset_hash,semantic_sha256,protocol_sha256)) WITHOUT ROWID;
         CREATE TABLE attempts(id INTEGER PRIMARY KEY,task TEXT NOT NULL,
           dataset_hash TEXT NOT NULL,semantic_sha256 TEXT NOT NULL,
           protocol_sha256 TEXT NOT NULL,attempt_no INTEGER NOT NULL,state TEXT NOT NULL,
           reservation_owner TEXT NOT NULL,manifest_path TEXT,manifest_sha256 TEXT,
           job_id TEXT,receipt_sha256 TEXT,failure_class TEXT,
           detail_json TEXT NOT NULL DEFAULT '{}',created_at_ms INTEGER NOT NULL,
           finalized_at_ms INTEGER,
           UNIQUE(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no));
         CREATE UNIQUE INDEX one_active_attempt_per_pair
           ON attempts(task,dataset_hash,semantic_sha256,protocol_sha256)
           WHERE state IN ('reserved','submitted');
         CREATE INDEX pair_manifest ON pairs(manifest_sha256);
         CREATE INDEX attempt_manifest ON attempts(manifest_sha256);",
    )?;
    Ok(())
}

fn args_for(root: &Path, lane: Lane, apply: bool) -> Args {
    Args {
        root: root.to_path_buf(),
        lane,
        apply,
        max_fill: 1_000,
        max_reap: 1_000,
        batch_rows: 16,
        busy_timeout_ms: 5_000,
        claim_states: vec!["checked".to_owned(), "submitting".to_owned()],
        revert: None,
        receipt_dir: Some(root.join("ops/receipts")),
    }
}

fn temp_root(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(format!("jope-ledger-reconciler-{name}-{}", now_ms()));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

#[test]
fn claim_state_filter_binds_sql_shaped_states_as_values() {
    let backlog = Connection::open_in_memory().unwrap();
    backlog
        .execute_batch(
            "CREATE TABLE claims(manifest_sha256 TEXT PRIMARY KEY,manifest TEXT NOT NULL UNIQUE,
               job_id TEXT NOT NULL UNIQUE,spec TEXT NOT NULL,state TEXT NOT NULL,
               attempts INTEGER NOT NULL DEFAULT 0,updated_at_ms INTEGER NOT NULL,
               reservation_owner TEXT);
             INSERT INTO claims VALUES('m1','/1.json','job-1','/s.toml','checked',0,1,NULL);
             INSERT INTO claims VALUES('m2','/2.json','job-2','/s.toml','submitting',0,1,NULL);
             INSERT INTO claims VALUES('m3','/3.json','job-3','/s.toml','submitted',0,1,NULL);",
        )
        .unwrap();
    let states = |values: &[&str]| values.iter().map(|v| (*v).to_owned()).collect::<Vec<_>>();
    assert_eq!(
        active_claim_manifests(&backlog, &states(&["checked", "submitting"])).unwrap(),
        ["m1", "m2"]
    );
    assert!(active_claim_manifests(&backlog, &states(&["') OR 1=1 --"])).is_err());
    assert!(
        active_claim_manifests(&backlog, &states(&["checked'); DROP TABLE claims; --"])).is_err()
    );
    let rows: i64 = backlog
        .query_row("SELECT COUNT(*) FROM claims", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 3);
    assert!(active_claim_manifests(&backlog, &[]).is_err());
}

fn add_pair(registry: &Connection, semantic: &str, state: &str, manifest: Option<&str>) {
    registry
        .execute(
            "INSERT INTO pairs VALUES('regression','ds',?,'proto','genome',?,'owner-1',
               '/m.json',?,NULL,NULL,NULL,0,1,1)",
            params![semantic, state, manifest],
        )
        .unwrap();
}

#[test]
fn fill_inserts_exactly_one_reserved_attempt_per_bare_pair() {
    let root = temp_root("fill");
    fixture(&root).unwrap();
    let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
    backlog
        .execute(
            "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','submitting',0,1,'owner-1')",
            [],
        )
        .unwrap();
    let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
    add_pair(&registry, "sem-a", "reserved", Some("m1"));
    add_pair(&registry, "sem-b", "reserved", Some("m1"));
    drop(registry);
    drop(backlog);

    let args = args_for(&root, Lane::Fill, true);
    let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
    let backlog = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
    let mut registry = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
    let stats = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
    assert_eq!(stats.inserted, 2);
    assert_eq!(stats.pairs_examined, 2);

    // Exactly one active attempt per pair, matching what
    // update_native_attempts_submitted counts.
    let count: i64 = registry
        .query_row(
            "SELECT COUNT(*) FROM attempts WHERE manifest_sha256='m1' \
             AND reservation_owner='owner-1' AND state IN ('reserved','infrastructure_retry')",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(count, 2);

    // Idempotent: a second run must insert nothing.
    let stats2 = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
    assert_eq!(stats2.inserted, 0);
    assert_eq!(stats2.skipped_already_active, 2);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fill_treats_sql_shaped_pair_identifiers_as_values() {
    let root = temp_root("sql-shaped-pair");
    fixture(&root).unwrap();
    let mut registry = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
    let hostile = "' OR 1=1; DROP TABLE attempts; --";
    let mut pending = vec![MissingAttempt {
        task: hostile.to_owned(),
        dataset_hash: hostile.to_owned(),
        semantic_sha256: hostile.to_owned(),
        protocol_sha256: hostile.to_owned(),
        state: "reserved".to_owned(),
        reservation_owner: hostile.to_owned(),
        manifest_path: Some("/manifest.json".to_owned()),
        manifest_sha256: "manifest-sha".to_owned(),
    }];
    let args = args_for(&root, Lane::Fill, true);
    let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();

    assert_eq!(
        flush_fill(&mut registry, &args, "t", &mut journal, &mut pending).unwrap(),
        1
    );
    assert!(pending.is_empty());

    let inserted: (String, String, String, String, String) = registry
        .query_row(
            "SELECT task,dataset_hash,semantic_sha256,protocol_sha256,reservation_owner \
             FROM attempts WHERE manifest_sha256=?",
            ["manifest-sha"],
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
            hostile.to_owned()
        )
    );

    // The SQL-shaped data must remain a single row value; it must not
    // broaden the predicate or alter the table used by later ledger work.
    let rows: i64 = registry
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(rows, 1);
    assert!(registry.prepare("SELECT COUNT(*) FROM pairs").is_ok());
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn fill_leaves_pairs_with_evidence_alone() {
    let root = temp_root("evidence");
    fixture(&root).unwrap();
    let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
    backlog
        .execute(
            "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','checked',0,1,'owner-1')",
            [],
        )
        .unwrap();
    let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
    // Receipted pair: DONE, must not be resurrected.
    registry
        .execute(
            "INSERT INTO pairs VALUES('regression','ds','sem-a','proto','genome','reserved',
               'owner-1','/m.json','m1',NULL,'receipt-sha',NULL,0,1,1)",
            [],
        )
        .unwrap();
    // Pair whose evaluation for THIS manifest already completed.
    add_pair(&registry, "sem-b", "reserved", Some("m1"));
    registry
        .execute(
            "INSERT INTO attempts(task,dataset_hash,semantic_sha256,protocol_sha256,attempt_no,
               state,reservation_owner,manifest_sha256,created_at_ms)
             VALUES('regression','ds','sem-b','proto',1,'completed','owner-1','m1',1)",
            [],
        )
        .unwrap();
    drop(registry);
    drop(backlog);

    let args = args_for(&root, Lane::Fill, true);
    let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
    let backlog = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
    let mut registry = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
    let stats = run_fill(&backlog, &mut registry, &args, "t", &mut journal).unwrap();
    assert_eq!(stats.inserted, 0);
    assert_eq!(stats.skipped_evidence, 2);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn reap_retires_only_registry_released_manifests() {
    let root = temp_root("reap");
    fixture(&root).unwrap();
    let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
    // m1: registry still binds its pair -> must be left alone.
    // m2: registry released it (no bound pair rows) -> fossil, must retire.
    backlog
        .execute_batch(
            "INSERT INTO keeper_attempts VALUES(1,'regression','ds','sem-a','proto',1,
               'submitted','owner-1','/m.json','m1','job-1',NULL,1,NULL);
             INSERT INTO keeper_attempts VALUES(2,'regression','ds','sem-b','proto',1,
               'submitted','owner-2','/m2.json','m2','job-2',NULL,1,NULL);",
        )
        .unwrap();
    let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
    add_pair(&registry, "sem-a", "reserved", Some("m1"));
    // Released pair: state 'available' and manifest binding nulled out, the
    // shape freshness/jope_queue_control leave behind.
    add_pair(&registry, "sem-b", "available", None);
    drop(registry);
    drop(backlog);

    let args = args_for(&root, Lane::Reap, true);
    let mut journal = Journal::create(root.join("ops/receipts/t.journal.jsonl"), true).unwrap();
    let registry = open_ro(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
    let mut backlog = open_rw(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
    let stats = run_reap(&mut backlog, &registry, &args, "t", &mut journal).unwrap();
    assert_eq!(stats.manifests_examined, 2);
    assert_eq!(stats.manifests_released, 1);
    assert_eq!(stats.retired, 1);

    let kept: String = backlog
        .query_row("SELECT state FROM keeper_attempts WHERE id=1", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(kept, "submitted");
    let reaped: String = backlog
        .query_row("SELECT state FROM keeper_attempts WHERE id=2", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(reaped, KEEPER_RETIRED_STATE);
    let _ = fs::remove_dir_all(&root);
}

#[test]
fn revert_undoes_both_lanes_exactly() {
    let root = temp_root("revert");
    fixture(&root).unwrap();
    let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
    backlog
        .execute(
            "INSERT INTO claims VALUES('m1','/m.json','job-1','/s.toml','submitting',0,1,'owner-1')",
            [],
        )
        .unwrap();
    backlog
        .execute(
            "INSERT INTO keeper_attempts VALUES(2,'regression','ds','sem-b','proto',1,
               'submitted','owner-2','/m2.json','m2','job-2',NULL,1,NULL)",
            [],
        )
        .unwrap();
    let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
    add_pair(&registry, "sem-a", "reserved", Some("m1"));
    add_pair(&registry, "sem-b", "available", None);
    drop(registry);
    drop(backlog);

    let receipts = root.join("ops/receipts");
    fs::create_dir_all(&receipts).unwrap();
    let mut args = args_for(&root, Lane::Both, true);
    let mut journal = Journal::create(receipts.join("t.journal.jsonl"), true).unwrap();
    {
        let backlog_ro = open_ro(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
        let mut registry_rw = open_rw(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
        let fill = run_fill(&backlog_ro, &mut registry_rw, &args, "t", &mut journal).unwrap();
        assert_eq!(fill.inserted, 1);
    }
    {
        let registry_ro = open_ro(&root.join("ops/pair-registry.sqlite"), 5_000).unwrap();
        let mut backlog_rw = open_rw(&root.join("ops/backlog.sqlite"), 5_000).unwrap();
        let reap = run_reap(&mut backlog_rw, &registry_ro, &args, "t", &mut journal).unwrap();
        assert_eq!(reap.retired, 1);
    }
    journal.sync().unwrap();

    args.revert = Some("t".to_owned());
    let report = run_revert(&args, "t", &receipts).unwrap();
    assert_eq!(report["attempt_rows_deleted"], 1);
    assert_eq!(report["keeper_rows_restored"], 1);

    let registry = Connection::open(root.join("ops/pair-registry.sqlite")).unwrap();
    let remaining: i64 = registry
        .query_row("SELECT COUNT(*) FROM attempts", [], |row| row.get(0))
        .unwrap();
    assert_eq!(remaining, 0);
    let backlog = Connection::open(root.join("ops/backlog.sqlite")).unwrap();
    let restored: String = backlog
        .query_row("SELECT state FROM keeper_attempts WHERE id=2", [], |row| {
            row.get(0)
        })
        .unwrap();
    assert_eq!(restored, "submitted");
    let _ = fs::remove_dir_all(&root);
}
