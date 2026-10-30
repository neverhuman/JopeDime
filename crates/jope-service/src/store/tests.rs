use super::*;
use jope_api::ids::RunId;
use serde_json::json;
#[cfg(feature = "failpoints")]
use std::path::Path;
use std::{
    fs,
    io::{self, Cursor, Read},
    path::PathBuf,
};

fn fixture() -> tempfile::TempDir {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/store-fixtures");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

fn run(number: u64) -> RunId {
    format!("run_{number:026}").parse().unwrap()
}
fn key(number: u64) -> RecordKey {
    RecordKey::Run(run(number))
}
fn artifact() -> RecordKey {
    RecordKey::Artifact("art_00112233445566778899aabb".parse().unwrap())
}
fn checkpoint() -> serde_json::Value {
    json!({"spec":{"target":"fixture","seed":"18446744073709551615"},
        "plan":{"workers":2},"last_committed_round":7,"attempt_high_water":91,
        "budget":{"evaluations":"64"},"rng":{"state":[0,1,2]},
        "elite":["alg_0001"],"champion":"alg_0001","extension":{"kept":true}})
}

#[test]
fn seven_tables_and_all_key_kinds_reopen_losslessly() {
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    let store_id = store.store_id();
    let blob = store
        .publish_blob(Cursor::new(b"fixture blob\0with exact bytes"))
        .unwrap();
    let keys = [
        key(0),
        RecordKey::Event {
            run: run(0),
            sequence: u64::MAX,
        },
        RecordKey::Evaluation {
            run: run(0),
            number: u64::MAX,
        },
        RecordKey::Command {
            principal: "owner".into(),
            idempotency_key: "fixture-1".into(),
        },
        RecordKey::Dataset("ds_00112233445566778899aabb".parse().unwrap()),
        artifact(),
    ];
    let records: Vec<_> = keys
        .iter()
        .map(|k| {
            if k.table() == Table::Runs {
                Record::run(checkpoint(), vec![blob.clone()], RunState::Completed, 17)
            } else {
                Record::metadata(checkpoint(), vec![blob.clone()])
            }
        })
        .collect();
    store
        .transaction(|write| {
            for (k, r) in keys.iter().zip(&records) {
                write.put(k, r)?;
            }
            Ok(())
        })
        .unwrap();
    let names: HashSet<_> = store
        .database
        .begin_read()
        .unwrap()
        .list_tables()
        .unwrap()
        .map(|t| t.name().to_owned())
        .collect();
    assert_eq!(
        names,
        Table::ALL
            .iter()
            .map(|t| t.name().to_owned())
            .chain(["meta".into()])
            .collect()
    );
    drop(store);
    let reopened = Store::open(dir.path()).unwrap();
    assert_eq!(reopened.store_id(), store_id);
    assert_eq!(
        fs::read(reopened.cas.path(&blob)).unwrap(),
        b"fixture blob\0with exact bytes"
    );
    for (k, record) in keys.iter().zip(records) {
        assert_eq!(reopened.get(k).unwrap(), Some(record));
    }
    reopened.verify_blob(&blob).unwrap();
    assert!(
        matches!(Store::open(dir.path()), Err(StoreError::Database(_))),
        "another writer cannot open the live database"
    );
}

#[test]
fn transaction_abort_and_missing_reference_never_publish_partial_metadata() {
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    let row = Record::metadata(json!({"receipt":"fixture"}), vec![]);
    let command = RecordKey::Command {
        principal: "owner".into(),
        idempotency_key: "atomic".into(),
    };
    let invalid = Record::metadata(
        json!(null),
        vec![BlobRef {
            sha256: [0; 32],
            bytes: 3,
        }],
    );
    let result = store.transaction(|write| {
        write.put(&command, &row)?;
        write.put(&artifact(), &invalid)
    });
    assert!(result.is_err());
    assert_eq!(store.get(&command).unwrap(), None);
    assert_eq!(store.get(&artifact()).unwrap(), None);
    let result = store.transaction(|write| {
        write.put(&command, &row)?;
        Err::<(), _>(StoreError::Invalid("fixture rollback"))
    });
    assert!(result.is_err());
    assert_eq!(store.get(&command).unwrap(), None);
    store
        .transaction(|write| {
            write.put(&command, &row)?;
            assert_eq!(write.get(&command)?, Some(row.clone()));
            Ok(())
        })
        .unwrap();
}

#[test]
fn recovery_pauses_only_interrupted_runs_and_preserves_checkpoint_and_fence() {
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    let states = [
        RunState::Running,
        RunState::Pausing,
        RunState::Paused,
        RunState::Completed,
        RunState::Cancelled,
        RunState::Failed,
    ];
    store
        .transaction(|write| {
            for (i, state) in states.iter().enumerate() {
                write.put(
                    &key(i as u64),
                    &Record::run(checkpoint(), vec![], *state, 7),
                )?;
            }
            Ok(())
        })
        .unwrap();
    drop(store);
    fs::create_dir(dir.path().join("tmp/nested")).unwrap();
    fs::write(
        dir.path().join("tmp/nested/partial"),
        b"uncommitted fixture",
    )
    .unwrap();
    let reopened = Store::open(dir.path()).unwrap();
    assert!(fs::read_dir(dir.path().join("tmp"))
        .unwrap()
        .next()
        .is_none());
    for (i, state) in states.iter().enumerate() {
        let record = reopened.get(&key(i as u64)).unwrap().unwrap();
        assert_eq!(record.data, checkpoint());
        let recovered = record.run.unwrap();
        let interrupted = matches!(state, RunState::Running | RunState::Pausing);
        assert_eq!(
            recovered.state,
            if interrupted {
                RunState::Paused
            } else {
                *state
            }
        );
        assert_eq!(recovered.interrupted, interrupted);
        assert_eq!(recovered.revision, if interrupted { 8 } else { 7 });
    }
    let paused = reopened.get(&key(0)).unwrap();
    drop(reopened);
    assert_eq!(
        Store::open(dir.path()).unwrap().get(&key(0)).unwrap(),
        paused,
        "recovery is idempotent"
    );
}

#[test]
fn future_schema_refuses_before_cleaning_staging_or_rewriting_metadata() {
    let dir = fixture();
    drop(Store::open(dir.path()).unwrap());
    let database = Database::open(dir.path().join("state.redb")).unwrap();
    let write = database.begin_write().unwrap();
    {
        write
            .open_table(META)
            .unwrap()
            .insert(VERSION_KEY, 2_u64.to_le_bytes().as_slice())
            .unwrap();
    }
    write.commit().unwrap();
    drop(database);
    let retained = dir.path().join("tmp/retain-on-unsupported-schema");
    fs::write(&retained, b"fixture evidence").unwrap();
    let error = match Store::open(dir.path()) {
        Err(error) => error,
        Ok(_) => panic!("accepted a future schema"),
    };
    assert!(matches!(
        error,
        StoreError::SchemaVersion {
            found: 2,
            supported: 1
        }
    ));
    assert!(error
        .to_string()
        .contains("unsupported store schema version 2"));
    assert_eq!(fs::read(retained).unwrap(), b"fixture evidence");
}

#[test]
fn corrupt_or_missing_referenced_blobs_refuse_and_preserve_staging() {
    for missing in [false, true] {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        let blob = store
            .publish_blob(Cursor::new(b"reference fixture"))
            .unwrap();
        let path = store.cas.path(&blob);
        store
            .transaction(|write| {
                write.put(
                    &artifact(),
                    &Record::metadata(json!({"fixture":1}), vec![blob]),
                )
            })
            .unwrap();
        drop(store);
        if missing {
            fs::remove_file(path).unwrap();
        } else {
            fs::write(path, b"corrupt").unwrap();
        }
        let retained = dir.path().join("tmp/retain-for-repair");
        fs::write(&retained, b"evidence").unwrap();
        assert!(Store::open(dir.path()).is_err());
        assert_eq!(fs::read(retained).unwrap(), b"evidence");
    }
}

#[test]
fn deduplication_verifies_existing_content_instead_of_replacing_corruption() {
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    let first = store.publish_blob(Cursor::new(b"dedup fixture")).unwrap();
    let second = store.publish_blob(Cursor::new(b"dedup fixture")).unwrap();
    assert_eq!(first, second);
    assert!(fs::read_dir(dir.path().join("tmp"))
        .unwrap()
        .next()
        .is_none());
    let path = store.cas.path(&first);
    fs::write(&path, b"corrupt existing content").unwrap();
    assert!(store.publish_blob(Cursor::new(b"dedup fixture")).is_err());
    assert_eq!(fs::read(path).unwrap(), b"corrupt existing content");
}

#[test]
fn interrupted_input_leaves_no_metadata_and_recovery_clears_the_partial_file() {
    struct Interrupted(bool);
    impl Read for Interrupted {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            if self.0 {
                Err(io::Error::other("fixture interruption"))
            } else {
                self.0 = true;
                buffer[0] = 42;
                Ok(1)
            }
        }
    }
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    assert!(store.publish_blob(Interrupted(false)).is_err());
    assert!(fs::read_dir(dir.path().join("tmp"))
        .unwrap()
        .next()
        .is_some());
    assert_eq!(store.get(&artifact()).unwrap(), None);
    drop(store);
    drop(Store::open(dir.path()).unwrap());
    assert!(fs::read_dir(dir.path().join("tmp"))
        .unwrap()
        .next()
        .is_none());
}

#[test]
fn keys_are_unambiguous_and_run_metadata_cannot_be_put_in_another_table() {
    let a = RecordKey::Command {
        principal: "a".into(),
        idempotency_key: "bc".into(),
    };
    let b = RecordKey::Command {
        principal: "ab".into(),
        idempotency_key: "c".into(),
    };
    assert_ne!(a.bytes().unwrap(), b.bytes().unwrap());
    let before = RecordKey::Event {
        run: run(0),
        sequence: 9,
    };
    let after = RecordKey::Event {
        run: run(0),
        sequence: 10,
    };
    assert!(before.bytes().unwrap() < after.bytes().unwrap());
    for value in ["".to_owned(), "x".repeat(129), "contains\nnewline".into()] {
        assert!(RecordKey::Command {
            principal: "a".into(),
            idempotency_key: value
        }
        .bytes()
        .is_err());
    }
    let dir = fixture();
    let store = Store::open(dir.path()).unwrap();
    assert!(store
        .transaction(|write| write.put(&key(0), &Record::metadata(json!(null), vec![])))
        .is_err());
    assert!(store
        .transaction(|write| write.put(
            &artifact(),
            &Record::run(json!(null), vec![], RunState::Paused, 1)
        ))
        .is_err());
}

#[cfg(unix)]
#[test]
fn symlinked_blobs_and_directories_refuse_without_touching_the_target() {
    use std::os::unix::fs::symlink;
    let dir = fixture();
    let home = dir.path().join("store");
    let store = Store::open(&home).unwrap();
    let blob = store.publish_blob(Cursor::new(b"symlink fixture")).unwrap();
    let path = store.cas.path(&blob);
    fs::remove_file(&path).unwrap();
    let victim = dir.path().join("victim");
    fs::write(&victim, b"protected fixture").unwrap();
    symlink(&victim, &path).unwrap();
    assert!(store.verify_blob(&blob).is_err());
    assert!(store
        .transaction(|w| w.put(&artifact(), &Record::metadata(json!(null), vec![blob])))
        .is_err());
    symlink(&victim, home.join("tmp/link")).unwrap();
    drop(store);
    drop(Store::open(&home).unwrap());
    assert_eq!(fs::read(victim).unwrap(), b"protected fixture");
}

#[cfg(feature = "failpoints")]
const CHILD_DATA: &[u8] = b"crash-boundary fixture blob";

#[cfg(feature = "failpoints")]
fn crash_keys() -> Vec<RecordKey> {
    vec![
        artifact(),
        RecordKey::Event {
            run: run(1),
            sequence: 2,
        },
        RecordKey::Command {
            principal: "owner".into(),
            idempotency_key: "crash-transaction".into(),
        },
        key(1),
    ]
}

#[cfg(feature = "failpoints")]
#[test]
#[ignore = "subprocess-only abort target; crash supervisor explicitly invokes it"]
fn crash_child() {
    let home = std::env::var_os("JOPE_STORE_CRASH_ROOT").expect("subprocess root required");
    let store = Store::open(Path::new(&home)).unwrap();
    let blob = store.publish_blob(Cursor::new(CHILD_DATA)).unwrap();
    store
        .transaction(|write| {
            for key in crash_keys() {
                let row = if key.table() == Table::Runs {
                    Record::run(checkpoint(), vec![blob.clone()], RunState::Running, 2)
                } else {
                    Record::metadata(checkpoint(), vec![blob.clone()])
                };
                write.put(&key, &row)?;
            }
            Ok(())
        })
        .unwrap();
    panic!("requested crash boundary was not reached");
}

#[cfg(feature = "failpoints")]
#[test]
fn actual_process_crash_at_every_publication_boundary_recovers_without_dangling_records() {
    use sha2::Digest;
    use std::process::Command;
    for point in [
        "after_write",
        "after_file_sync",
        "after_hash_verify",
        "after_rename",
        "after_directory_sync",
        "after_commit",
    ] {
        let dir = fixture();
        let store = Store::open(dir.path()).unwrap();
        let baseline = Record::run(checkpoint(), vec![], RunState::Paused, 1);
        store.transaction(|w| w.put(&key(0), &baseline)).unwrap();
        drop(store);
        #[cfg(unix)]
        let mut child = {
            let mut c = Command::new("sh");
            c.args(["-c", "ulimit -c 0; exec \"$@\"", "store-crash-child"]);
            c.arg(std::env::current_exe().unwrap());
            c
        };
        #[cfg(not(unix))]
        let mut child = Command::new(std::env::current_exe().unwrap());
        let output = child
            .args([
                "--exact",
                "store::tests::crash_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("JOPE_STORE_CRASH_ROOT", dir.path())
            .env("JOPE_STORE_FAILPOINT", point)
            .output()
            .unwrap();
        assert!(!output.status.success(), "{point}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(&format!("JOPE_STORE_CRASH:{point}")),
            "{point}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reopened = Store::open(dir.path()).unwrap();
        assert_eq!(reopened.get(&key(0)).unwrap(), Some(baseline));
        let expected_blob = BlobRef {
            sha256: sha2::Sha256::digest(CHILD_DATA).into(),
            bytes: CHILD_DATA.len() as u64,
        };
        let renamed = matches!(
            point,
            "after_rename" | "after_directory_sync" | "after_commit"
        );
        assert_eq!(
            reopened.cas.path(&expected_blob).exists(),
            renamed,
            "{point}"
        );
        if renamed {
            reopened.verify_blob(&expected_blob).unwrap();
            assert_eq!(
                fs::read(reopened.cas.path(&expected_blob)).unwrap(),
                CHILD_DATA
            );
        }
        for key in crash_keys() {
            let row = reopened.get(&key).unwrap();
            assert_eq!(row.is_some(), point == "after_commit", "{point}: {key:?}");
            if let Some(row) = row {
                assert_eq!(row.data, checkpoint());
                for reference in &row.blobs {
                    reopened.verify_blob(reference).unwrap();
                }
                if let Some(run) = row.run {
                    assert_eq!(run.state, RunState::Paused);
                    assert!(run.interrupted);
                    assert_eq!(run.revision, 3);
                }
            }
        }
        assert!(
            fs::read_dir(dir.path().join("tmp"))
                .unwrap()
                .next()
                .is_none(),
            "{point}"
        );
    }
}
