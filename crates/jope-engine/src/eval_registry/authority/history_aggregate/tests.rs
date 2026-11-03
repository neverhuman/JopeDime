use super::*;
use super::validation::{batch_ranges, checked_chunk, original_source};

fn artifact(root: &Path, name: &str, bytes: &[u8]) -> HistoryArtifact {
    let path = root.join(name);
    fs::write(&path, bytes).unwrap();
    HistoryArtifact {
        path: fs::canonicalize(path).unwrap(),
        sha256: hash(bytes),
        bytes: bytes.len() as u64,
    }
}
fn index<T: Serialize>(root: &Path, name: &str, rows: &[T]) -> HistoryIndex {
    let mut bytes = Vec::new();
    for row in rows {
        bytes.extend(encoded(row).unwrap());
        bytes.push(b'\n');
    }
    HistoryIndex {
        artifact: artifact(root, name, &bytes),
        rows: rows.len() as u64,
    }
}
#[test]
fn original_payload_requires_exact_piece_order_and_complete_hash() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let payload = artifact(
        root,
        "piece",
        b"original SQLite bytes, not guard row counts",
    );
    let mut source = OriginalHistorySource {
        original_identity: "remote/original.sqlite".into(),
        original_sha256: payload.sha256.clone(),
        original_bytes: payload.bytes,
        role: OriginalHistoryRole::ConvertedDatabase,
        pieces: index(
            root,
            "pieces",
            &[OriginalHistoryPiece {
                ordinal: 0,
                offset: 0,
                payload: payload.clone(),
            }],
        ),
    };
    original_source(&source).unwrap();
    source.original_sha256 = "0".repeat(64);
    assert!(original_source(&source).is_err());
    source.original_sha256 = payload.sha256.clone();
    source.pieces = index(
        root,
        "gap",
        &[OriginalHistoryPiece {
            ordinal: 0,
            offset: 1,
            payload: payload.clone(),
        }],
    );
    assert!(original_source(&source).is_err());
    source.pieces = index(
        root,
        "extra",
        &[
            OriginalHistoryPiece {
                ordinal: 0,
                offset: 0,
                payload: payload.clone(),
            },
            OriginalHistoryPiece {
                ordinal: 1,
                offset: payload.bytes,
                payload,
            },
        ],
    );
    assert!(original_source(&source).is_err());
}
#[test]
fn index_rejects_false_count_torn_rows_and_modified_payload() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let mut i = index(root, "index", &[json!({"a":1})]);
    i.rows = 2;
    let mut lines = Lines::new(&i).unwrap();
    assert!(lines.next::<Value>().unwrap().is_some());
    assert!(lines.next::<Value>().is_err());
    let torn = HistoryIndex {
        artifact: artifact(root, "torn", b"{}"),
        rows: 1,
    };
    assert!(Lines::new(&torn).unwrap().next::<Value>().is_err());
    let i = index(root, "corrupt", &[json!({"a":1})]);
    fs::write(&i.artifact.path, b"{\"a\":2}\n").unwrap();
    let mut lines = Lines::new(&i).unwrap();
    lines.next::<Value>().unwrap();
    assert!(lines.next::<Value>().is_err());
}
#[test]
fn original_payload_refuses_symlinks_and_fifo_without_blocking() {
    let tmp = tempfile::tempdir().unwrap();
    let a = artifact(tmp.path(), "regular", b"x");
    let mut link = a.clone();
    link.path = tmp.path().join("link");
    std::os::unix::fs::symlink(&a.path, &link.path).unwrap();
    assert!(open(&link).is_err());
    let fifo = tmp.path().join("fifo");
    nix::unistd::mkfifo(&fifo, nix::sys::stat::Mode::S_IRUSR).unwrap();
    let a = HistoryArtifact {
        path: fifo,
        sha256: hash(b""),
        bytes: 0,
    };
    assert!(open(&a).is_err());
}
#[test]
fn imported_batches_must_cover_every_stage_row_once() {
    let db = Connection::open_in_memory().unwrap();
    db.execute_batch(
        "CREATE TABLE history_batches(stage_sha TEXT,start_offset TEXT,end_offset TEXT);",
    )
    .unwrap();
    let insert = |start: &str, end: &str| {
        db.execute(
            "INSERT INTO history_batches VALUES('stage',?,?)",
            params![start, end],
        )
        .unwrap();
    };
    insert("0", "2");
    assert!(batch_ranges(&db, "stage", 4).is_err());
    insert("2", "4");
    batch_ranges(&db, "stage", 4).unwrap();
    insert("1", "3");
    assert!(batch_ranges(&db, "stage", 4).is_err());
}
#[test]
fn guard_evidence_never_turns_unknown_outcomes_into_completion() {
    let tmp = tempfile::tempdir().unwrap();
    let mut row = json!({"schema":"jope.historical_program_exclusion_evidence.v1","original_database_sha256":"a".repeat(64),"canonical_program":"(pipe X (ridge alpha=1))","original_occurrences":17,"outcome":"historical_seen_result_unavailable","reusable_completion":false});
    let canonical = feat_math::GenomeIr::parse(row["canonical_program"].as_str().unwrap())
        .unwrap()
        .canonical();
    row["canonical_program"] = json!(canonical);
    let chunk = index(tmp.path(), "chunk", &[row.clone()]).artifact;
    assert_eq!(
        checked_chunk(&chunk, &"a".repeat(64), None, "scope").unwrap(),
        (1, 17)
    );
    row["reusable_completion"] = json!(true);
    let chunk = index(tmp.path(), "false-complete", &[row]).artifact;
    assert!(checked_chunk(&chunk, &"a".repeat(64), None, "scope").is_err());
}
