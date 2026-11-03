use super::*;
use sha2::{Digest, Sha256};
const REAL: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test-support/datasets/v1/messy-csv-like/raw-source-subset.csv"
));

#[test]
fn same_real_csv_stream_and_local_path_have_identical_id_durable_metadata_and_blobs() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::open(home.path()).unwrap();
    let first = register(&store, REAL, &Default::default()).unwrap();
    assert_eq!(first.id.to_string(), "ds_2bb5dee875d2a5c41d004c50");
    assert_eq!(first.csv.sha256.as_slice(), Sha256::digest(REAL).as_slice());
    let second = register(&store, REAL, &Default::default()).unwrap();
    assert_eq!(first, second);
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-support/datasets/v1/messy-csv-like/raw-source-subset.csv");
    assert_eq!(
        register_file(&store, &source, &Default::default()).unwrap(),
        first
    );
    store.verify_blob(&first.csv).unwrap();
    store.verify_blob(&first.matrix).unwrap();
    let record = store
        .get(&RecordKey::Dataset(first.id.clone()))
        .unwrap()
        .unwrap();
    assert_eq!(record.blobs, vec![first.csv.clone(), first.matrix.clone()]);
    assert_eq!(
        serde_json::from_value::<RegisteredDataset>(record.data).unwrap(),
        first
    );
    drop(store);
    let reopened = Store::open(home.path()).unwrap();
    assert!(reopened
        .get(&RecordKey::Dataset(first.id))
        .unwrap()
        .is_some());
}

#[cfg(unix)]
#[test]
fn fifo_without_writer_returns_dataset_invalid_promptly() {
    use std::{
        process::{Command, Stdio},
        time::{Duration, Instant},
    };

    const CHILD: &str = "JOPE_SVC11_F02_FIFO_CHILD";
    const TEST_DIR: &str = "JOPE_SVC11_F02_TEST_DIR";
    let test_name = std::thread::current().name().unwrap().to_owned();

    if std::env::var_os(CHILD).is_some() {
        let dir = std::path::PathBuf::from(std::env::var_os(TEST_DIR).unwrap());
        let store = Store::open(dir.join("store")).unwrap();
        let error = register_file(
            &store,
            &dir.join("input.fifo"),
            &RegistrationLimits::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            RegistrationError::Invalid(problem) if problem.code == ErrorCode::DatasetInvalid
        ));
        return;
    }

    let temp = tempfile::tempdir().unwrap();
    let fifo = temp.path().join("input.fifo");
    nix::unistd::mkfifo(
        &fifo,
        nix::sys::stat::Mode::S_IRUSR | nix::sys::stat::Mode::S_IWUSR,
    )
    .unwrap();

    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &test_name])
        .env(CHILD, "1")
        .env(TEST_DIR, temp.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(
                status.success(),
                "bounded FIFO registration child failed: {status}"
            );
            break;
        }
        if Instant::now() >= deadline {
            child.kill().expect("timed-out FIFO child is still running");
            let _ = child.wait();
            panic!("register_file did not refuse an unwritten FIFO within 5 seconds");
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn every_limit_returns_dataset_invalid_with_its_name_and_no_dataset_record() {
    let defaults = RegistrationLimits::default();
    let cases = [
        (
            "max_dataset_bytes",
            RegistrationLimits {
                max_dataset_bytes: REAL.len() as u64 - 1,
                ..defaults.clone()
            },
        ),
        (
            "max_rows",
            RegistrationLimits {
                max_rows: 0,
                ..defaults.clone()
            },
        ),
        (
            "max_columns",
            RegistrationLimits {
                max_columns: 0,
                ..defaults.clone()
            },
        ),
        (
            "max_cell_bytes",
            RegistrationLimits {
                max_cell_bytes: 0,
                ..defaults.clone()
            },
        ),
        (
            "max_categories_per_column",
            RegistrationLimits {
                max_categories_per_column: 0,
                ..defaults.clone()
            },
        ),
    ];
    for (name, limits) in cases {
        let home = tempfile::tempdir().unwrap();
        let store = Store::open(home.path()).unwrap();
        // Category text is a limit-protocol vector, not a learning dataset.
        let input = if name == "max_categories_per_column" {
            b"kind\ncategory\n".as_slice()
        } else {
            REAL
        };
        let error = register(&store, input, &limits).unwrap_err();
        let RegistrationError::Invalid(problem) = error else {
            panic!("expected typed refusal for {name}")
        };
        assert_eq!(problem.code, ErrorCode::DatasetInvalid);
        assert_eq!(problem.status, 422);
        assert_eq!(problem.field.as_deref(), Some(name));
        assert!(problem.detail.contains(name));
        let id = "ds_2bb5dee875d2a5c41d004c50".parse().unwrap();
        assert!(store.get(&RecordKey::Dataset(id)).unwrap().is_none());
    }
}

#[test]
fn malformed_utf8_quotes_width_headers_and_empty_input_are_redacted() {
    // Malformed CSV protocol cases, not generated learning datasets.
    for bytes in [
        b"".as_slice(),
        b"secret,secret\n1,2\n",
        b"a,b\n1\n",
        b"a\n\xff\n",
        b"a\n\"secret",
        b"a\nse\"cret\n",
        b"a\n\"secret\"tail\n",
        b"a,b\n",
    ] {
        let home = tempfile::tempdir().unwrap();
        let store = Store::open(home.path()).unwrap();
        let error = register(&store, bytes, &Default::default()).unwrap_err();
        assert!(!error.to_string().contains("secret"));
        let RegistrationError::Invalid(problem) = error else {
            panic!("typed invalid CSV expected")
        };
        assert_eq!(problem.code, ErrorCode::DatasetInvalid);
    }
    // Quoted delimiters, escaped quotes, multiline records and CRLF framing.
    assert!(framing(
        b"a,b\r\n\"one, two\",\"escaped\"\"quote\nline\"\r\n",
        &Default::default()
    )
    .is_ok());
}

#[test]
fn a_dataset_prefix_collision_cannot_replace_existing_metadata() {
    let home = tempfile::tempdir().unwrap();
    let store = Store::open(home.path()).unwrap();
    let key = RecordKey::Dataset("ds_2bb5dee875d2a5c41d004c50".parse().unwrap());
    let existing = Record::metadata(serde_json::json!({"different_full_digest":true}), vec![]);
    store.transaction(|w| w.put(&key, &existing)).unwrap();
    assert!(matches!(
        register(&store, REAL, &Default::default()),
        Err(RegistrationError::Store(StoreError::Integrity(_)))
    ));
    assert_eq!(store.get(&key).unwrap(), Some(existing));
}
