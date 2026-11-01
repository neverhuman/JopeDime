use jope_client::{InProcess, RunResult, Transport};
use jope_service::{dataset::register, engine::InProcessEngine, store::RecordKey};
use std::{fs, path::PathBuf, sync::Arc};

#[test]
fn in_process_transport_reads_a_dataset_from_its_owned_local_store() {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    fs::create_dir_all(&target).unwrap();
    let home = tempfile::tempdir_in(target).unwrap();
    let engine = Arc::new(
        InProcessEngine::open_with_worker(home.path(), env!("CARGO_BIN_EXE_jopedime")).unwrap(),
    );

    // Use committed TRAIN labels, with a small bounded view for this local-store
    // transport proof. No sealed or holdout fixture enters the worker.
    let raw = include_bytes!("../../../test-support/datasets/v1/d2-regression/train.csv");
    let mut columns: Vec<_> = (0..21).map(|index| format!("x{index}")).collect();
    columns.push("target".to_owned());
    let mut csv = columns.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(96) {
        csv.push_str(line);
        csv.push('\n');
    }
    let registered = register::register(engine.store(), csv.as_bytes(), &Default::default())
        .expect("register committed TRAIN rows in the temporary local store");

    let client = InProcess::with_engine(engine.clone(), 31234, &"ab".repeat(32)).unwrap();
    let page = client.datasets().unwrap();
    assert!(page.next_after.is_none());
    assert_eq!(page.datasets.len(), 1);
    assert_eq!(page.datasets[0].dataset_id, registered.id);
    let profile = client.dataset_inspect(&registered.id).unwrap();
    assert_eq!(profile.dataset_id, registered.id);
    assert_eq!(profile.rows.0, 96);
    assert_eq!(profile.columns.len(), 22);

    let durable = engine
        .store()
        .get(&RecordKey::Dataset(registered.id.clone()))
        .unwrap()
        .expect("dataset metadata must be durable in the same store");
    assert_eq!(durable.data["id"], registered.id.to_string());
    assert!(!durable.blobs.is_empty());

    let spec: jope_api::RunSpec = serde_json::from_value(serde_json::json!({
        "dataset": registered.id,
        "target": "target",
        "task": "regression",
        "ignore_columns": [],
        "test_dataset": null,
        "split": {"kind":"random","holdout_fraction":0.2,"group_column":null,"time_column":null},
        "objective": "rmse",
        "effort": "fast",
        "budget": {"evaluations":"4","wall_seconds":"10"},
        "landmarks": "none",
        "guidance": "heuristic",
        "languages": ["rust"],
        "require_all_languages": true,
        "device": "cpu",
        "seed": "73",
        "dry_run": false,
        "plan_digest": null
    }))
    .unwrap();
    let started = match client
        .run_start(&spec, Some("svc08-store-transport-run"))
        .unwrap()
    {
        RunResult::Started(snapshot) => snapshot,
        RunResult::Plan(_) => panic!("bounded run unexpectedly returned a plan"),
    };
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let completed = loop {
        let snapshot = client.run_status(&started.run_id).unwrap();
        if snapshot.state == jope_api::RunState::Completed {
            break snapshot;
        }
        assert_ne!(snapshot.state, jope_api::RunState::Failed, "run failed");
        assert!(
            std::time::Instant::now() < deadline,
            "run did not complete in 30s"
        );
        std::thread::sleep(std::time::Duration::from_millis(50));
    };
    assert_eq!(completed.run_id, started.run_id);
    let events = client
        .run_events(&started.run_id, jope_api::decimal::DecimalU64(0), 256)
        .unwrap();
    assert!(!events.events.is_empty());
    let stored_run = engine
        .store()
        .get(&RecordKey::Run(started.run_id.clone()))
        .unwrap()
        .expect("run snapshot must be committed to the bound local store");
    assert_eq!(stored_run.run.unwrap().state, jope_api::RunState::Completed);
    let first_sequence = events.events[0].sequence.0;
    assert!(engine
        .store()
        .get(&RecordKey::Event {
            run: started.run_id,
            sequence: first_sequence,
        })
        .unwrap()
        .is_some());
}
