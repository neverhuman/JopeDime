use jope_client::{InProcess, Transport};
use jope_service::{dataset::register, engine::InProcessEngine, store::RecordKey};
use std::{fs, path::PathBuf, sync::Arc};

#[test]
fn in_process_transport_reads_a_dataset_from_its_owned_local_store() {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/int01");
    fs::create_dir_all(&target).unwrap();
    let home = tempfile::tempdir_in(target).unwrap();
    let engine = Arc::new(InProcessEngine::open(home.path()).unwrap());

    // Use committed TRAIN labels, with a small bounded view for this local-store
    // transport proof. No sealed or holdout fixture enters the worker.
    let raw = include_bytes!("../../../test-support/datasets/v1/d2-regression/train.csv");
    let mut columns: Vec<_> = (0..21).map(|index| format!("x{index}")).collect();
    columns.push("target".to_owned());
    let mut csv = columns.join(",");
    csv.push('\n');
    for line in std::str::from_utf8(raw).unwrap().lines().take(24) {
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
    assert_eq!(profile.rows.0, 24);
    assert_eq!(profile.columns.len(), 22);

    let durable = engine
        .store()
        .get(&RecordKey::Dataset(registered.id.clone()))
        .unwrap()
        .expect("dataset metadata must be durable in the same store");
    assert_eq!(durable.data["id"], registered.id.to_string());
    assert!(!durable.blobs.is_empty());
}
