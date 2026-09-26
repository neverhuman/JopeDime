//! W6 follow-up: `manifest::atomic_write` / `write_json` temporary naming.
//!
//! Every `write_json` caller published through `<name>.partial.<pid>` opened
//! with `create_new`. A writer killed between creating that temporary and
//! renaming it left the file behind, and every later writer with the same
//! process id failed. Two threads of one process writing the same file
//! also collided on it.

use jope_parallel::manifest::{atomic_write, read_json, write_json};
use serde_json::{json, Value};
use std::{fs, sync::Arc, thread};

#[test]
fn a_stale_same_pid_temporary_cannot_block_write_json() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("neural-service.json");
    let pid = std::process::id();
    let stale = dir.path().join(format!("neural-service.partial.{pid}"));
    fs::write(&stale, b"{\"torn\":").unwrap();
    write_json(&target, &json!({"step":1})).unwrap();
    assert_eq!(read_json::<Value>(&target).unwrap(), json!({"step":1}));
    // The stale temporary is inert: never read, never deleted.
    assert_eq!(fs::read(&stale).unwrap(), b"{\"torn\":");
    // Replacement still works, and leaves no temporary of its own.
    write_json(&target, &json!({"step":2})).unwrap();
    assert_eq!(read_json::<Value>(&target).unwrap(), json!({"step":2}));
    let names = fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(
        names,
        [
            "neural-service.json".to_string(),
            format!("neural-service.partial.{pid}")
        ]
        .into_iter()
        .collect()
    );
}

#[test]
fn raw_atomic_write_is_unblocked_by_a_stale_temporary() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("values.bin");
    let pid = std::process::id();
    fs::write(dir.path().join(format!("values.partial.{pid}")), b"x").unwrap();
    atomic_write(&target, b"payload").unwrap();
    assert_eq!(fs::read(&target).unwrap(), b"payload");
}

#[test]
fn concurrent_writers_of_one_file_in_one_process_do_not_collide() {
    let dir = Arc::new(tempfile::tempdir().unwrap());
    let target = Arc::new(dir.path().join("rank-0.json"));
    let workers = (0..8)
        .map(|i| {
            let target = Arc::clone(&target);
            thread::spawn(move || {
                (0..50)
                    .map(|n| write_json(&target, &json!({"writer":i,"n":n})).is_ok())
                    .filter(|ok| !ok)
                    .count()
            })
        })
        .collect::<Vec<_>>();
    let failures: usize = workers.into_iter().map(|w| w.join().unwrap()).sum();
    assert_eq!(failures, 0, "concurrent writers collided on one temporary");
    // The published file is one complete write, never a mixture.
    let value: Value = read_json(&target).unwrap();
    assert!(value["writer"].is_u64() && value["n"].is_u64());
}

#[test]
fn a_failed_write_leaves_the_previous_file_intact() {
    // Positive control for the unchanged semantics: a destination that cannot
    // be replaced (a directory) fails without touching other files.
    let dir = tempfile::tempdir().unwrap();
    let blocked = dir.path().join("blocked.json");
    fs::create_dir(&blocked).unwrap();
    fs::write(blocked.join("keep"), b"kept").unwrap();
    assert!(write_json(&blocked, &json!({"x":1})).is_err());
    assert_eq!(fs::read(blocked.join("keep")).unwrap(), b"kept");
    let leftovers = fs::read_dir(dir.path())
        .unwrap()
        .filter(|e| e.as_ref().unwrap().file_name() != "blocked.json")
        .count();
    assert_eq!(leftovers, 0, "a failed write left its temporary behind");
}
