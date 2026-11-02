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

mod regular_tests;

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
