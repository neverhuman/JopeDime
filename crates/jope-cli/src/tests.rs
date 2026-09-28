use crate::{
    admission::validate_dataset,
    generate_fixtures,
    generator::{dataset, generated},
    schema::{digest, serialize, Catalog, Dataset, Kind, Split, Task, MAX_FILE_BYTES},
    storage::{components, Directory},
    verify_inputs,
};
use serde_json::{json, Value};
use std::{fs, path::PathBuf};
use tempfile::{tempdir, TempDir};

fn corpus() -> (TempDir, PathBuf) {
    let temporary = tempdir().unwrap();
    let root = temporary.path().join("fixtures");
    generate_fixtures(&root).unwrap();
    (temporary, root)
}

fn named(task: Task, name: &str) -> Dataset {
    let scenario_index = task
        .scenarios()
        .iter()
        .position(|scenario| *scenario == name)
        .unwrap_or_else(|| panic!("unknown scenario {name}"));
    dataset(task, scenario_index)
}

fn with_groups(mut data: Dataset) -> Dataset {
    for (index, row) in data.rows.iter_mut().enumerate() {
        row.group = Some(format!("group-{:02}", index / 4));
    }
    data
}

fn rewrite_catalog(root: &std::path::Path, mutate: impl FnOnce(&mut Catalog)) {
    let path = root.join("catalog.json");
    let mut catalog: Catalog = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    mutate(&mut catalog);
    fs::write(path, serialize(&catalog).unwrap()).unwrap();
}

#[test]
fn ten_scenarios_are_valid_and_regeneration_is_byte_identical() {
    let (first_catalog, first_files) = generated().unwrap();
    let (second_catalog, second_files) = generated().unwrap();
    assert_eq!(first_catalog, second_catalog);
    assert_eq!(first_files, second_files);
    assert_eq!(first_files.len(), 10);
    for bytes in first_files {
        let data: Dataset = serde_json::from_slice(&bytes).unwrap();
        validate_dataset(&data).unwrap_or_else(|error| panic!("{}: {error}", data.id));
    }
    let (_temporary, root) = corpus();
    let report = verify_inputs(&root).unwrap();
    assert_eq!(report.dataset_count, 10);
    assert_eq!(report.task_counts["binary"], 3);
    assert_eq!(report.task_counts["regression"], 3);
    assert_eq!(report.task_counts["time_series"], 2);
    assert_eq!(report.task_counts["multiclass"], 2);
    assert!(report.bytes < 10 * 1024 * 1024);
}

#[test]
fn altered_file_and_replaced_checksum_cannot_redefine_trust() {
    let (_temporary, root) = corpus();
    let path = root.join("binary/linear.json");
    let mut data: Dataset = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    data.rows[0].target = 1 - data.rows[0].target;
    let bytes = serialize(&data).unwrap();
    fs::write(&path, &bytes).unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("digest mismatch"));
    rewrite_catalog(&root, |catalog| {
        catalog.datasets[0].sha256 = digest(&bytes);
        catalog.datasets[0].bytes = bytes.len() as u64;
    });
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("pinned deterministic generator"));
}

#[test]
fn missing_extra_duplicate_and_untrusted_catalog_entries_fail() {
    for alteration in 0..6 {
        let (_temporary, root) = corpus();
        rewrite_catalog(&root, |catalog| match alteration {
            0 => {
                catalog.datasets.pop();
            }
            1 => catalog.datasets.push(catalog.datasets[0].clone()),
            2 => catalog.datasets[1] = catalog.datasets[0].clone(),
            3 => catalog.provenance.kind = "downloaded".into(),
            4 => catalog
                .provenance
                .external_inputs
                .push("external.csv".into()),
            _ => catalog.datasets[0].task = Task::Regression,
        });
        assert!(verify_inputs(&root).is_err(), "alteration {alteration}");
    }
}

#[test]
fn traversal_and_nonportable_paths_are_rejected() {
    for path in [
        "",
        "../outside.json",
        "/tmp/outside.json",
        "binary/../catalog.json",
        "binary//linear.json",
        "./catalog.json",
        "binary\\linear.json",
        "C:/outside.json",
    ] {
        assert!(components(path).is_err(), "path {path}");
    }
    let (_temporary, root) = corpus();
    rewrite_catalog(&root, |catalog| {
        catalog.datasets[0].path = "../outside.json".into()
    });
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("relative fixture path"));
}

#[test]
#[cfg(unix)]
fn individual_dataset_reads_reject_unsafe_objects_and_oversized_payloads() {
    use crate::kernel::read_dataset;
    use std::os::unix::fs::symlink;
    let temporary = tempdir().unwrap();
    let path = temporary.path().join("input.json");
    fs::write(&path, serialize(&dataset(Task::Regression, 0)).unwrap()).unwrap();
    // Positive control: the same bytes through a regular file are admitted.
    assert_eq!(read_dataset(&path).unwrap(), dataset(Task::Regression, 0));
    let linked = temporary.path().join("linked.json");
    symlink(&path, &linked).unwrap();
    let error = format!("{:#}", read_dataset(&linked).unwrap_err());
    assert!(error.contains("without symlinks"), "{error}");
    let error = format!("{:#}", read_dataset(temporary.path()).unwrap_err());
    assert!(error.contains("read dataset"), "{error}");
    let directory = temporary.path().join("directory.json");
    fs::create_dir(&directory).unwrap();
    let error = format!("{:#}", read_dataset(&directory).unwrap_err());
    assert!(error.contains("regular file"), "{error}");
    fs::write(&path, vec![b' '; MAX_FILE_BYTES as usize + 1]).unwrap();
    let error = format!("{:#}", read_dataset(&path).unwrap_err());
    assert!(error.contains("fixture exceeds byte limit"), "{error}");
}

#[test]
fn unknown_fields_and_unsupported_target_shapes_are_rejected() {
    let bytes = serialize(&dataset(Task::Regression, 0)).unwrap();
    let original: Value = serde_json::from_slice(&bytes).unwrap();
    for value in [
        json!([1, 2]),
        json!(null),
        json!(1.5),
        json!("1"),
        json!({"value": 1}),
    ] {
        let mut changed = original.clone();
        changed["rows"][0]["target"] = value;
        assert!(serde_json::from_value::<Dataset>(changed).is_err());
    }
    let mut changed = original;
    changed["unexpected_schema_field"] = json!(true);
    assert!(serde_json::from_value::<Dataset>(changed).is_err());
}

#[test]
fn malformed_features_targets_ids_and_splits_fail_schema_validation() {
    for alteration in 0..9 {
        let mut data = dataset(Task::Binary, 0);
        match alteration {
            0 => data.rows[0].target = 2,
            1 => data.rows[0].features.clear(),
            2 => data.rows[0].features[0] = Value::Null,
            3 => data.rows[0].features[0] = json!(1.25),
            4 => data.rows[0].id = data.rows[1].id,
            5 => data
                .rows
                .iter_mut()
                .for_each(|row| row.split = Split::Train),
            6 => data.columns[0].name = data.columns[1].name.clone(),
            7 => data.rows[0].series = Some("unexpected".into()),
            _ => data.classes = vec![0, 2],
        }
        assert!(validate_dataset(&data).is_err(), "alteration {alteration}");
    }
    let mut categorical = named(Task::Binary, "linear");
    categorical.columns[1].kind = Kind::Categorical;
    categorical.columns[1].categories = ["red", "green", "blue"]
        .into_iter()
        .map(str::to_string)
        .collect();
    for row in &mut categorical.rows {
        row.features[1] = json!("red");
    }
    categorical.rows[0].features[1] = "undeclared".into();
    assert!(validate_dataset(&categorical).is_err());
}

#[test]
fn time_reversal_duplicate_times_and_future_partitions_fail() {
    for alteration in 0..4 {
        let mut data = dataset(Task::TimeSeries, 0);
        match alteration {
            0 => data.rows[1].timestamp = data.rows[0].timestamp,
            1 => data.rows.swap(0, 1),
            2 => data.rows[0].split = Split::Test,
            _ => data.rows[0].timestamp = None,
        }
        assert!(validate_dataset(&data).is_err(), "alteration {alteration}");
    }
    let mut multiple = named(Task::TimeSeries, "trend");
    for (index, row) in multiple.rows.iter_mut().enumerate() {
        row.series = Some(format!("series-{}", index % 2));
    }
    for row in multiple
        .rows
        .iter_mut()
        .filter(|row| row.series.as_deref() == Some("series-0"))
    {
        row.timestamp = row.timestamp.map(|time| time + 100_000);
    }
    assert!(validate_dataset(&multiple)
        .unwrap_err()
        .to_string()
        .contains("overlap"));
}

#[test]
fn groups_cannot_cross_partitions_or_lose_identity() {
    for task in [Task::Binary, Task::Multiclass] {
        let mut data = with_groups(named(task, task.scenarios()[0]));
        data.rows[64].group = data.rows[0].group.clone();
        assert!(validate_dataset(&data)
            .unwrap_err()
            .to_string()
            .contains("group leaks"));
        data = with_groups(named(task, task.scenarios()[0]));
        data.rows[0].group = None;
        assert!(validate_dataset(&data).is_err());
    }
}

#[test]
fn missing_unmanifested_and_oversized_files_fail() {
    let (_temporary, root) = corpus();
    fs::remove_file(root.join("binary/linear.json")).unwrap();
    assert!(verify_inputs(&root).is_err());
    let (_temporary, root) = corpus();
    fs::write(root.join("regression/not_admitted.json"), b"{}").unwrap();
    assert!(verify_inputs(&root).is_err());
    let (_temporary, root) = corpus();
    fs::write(
        root.join("catalog.json"),
        vec![b' '; MAX_FILE_BYTES as usize + 1],
    )
    .unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("byte limit"));
}

#[test]
fn generation_never_overwrites_existing_or_incomplete_output() {
    let (_temporary, root) = corpus();
    assert!(generate_fixtures(&root).is_err());
    fs::remove_file(root.join("catalog.json")).unwrap();
    assert!(verify_inputs(&root).is_err());
    assert!(generate_fixtures(&root).is_err());
}

#[cfg(unix)]
#[test]
fn symlink_files_directories_and_roots_are_rejected() {
    use std::os::unix::fs::symlink;
    for relative in ["binary/linear.json", "binary", ""] {
        let (temporary, root) = corpus();
        let source = if relative.is_empty() {
            root.clone()
        } else {
            root.join(relative)
        };
        let detached = temporary.path().join("detached");
        fs::rename(&source, &detached).unwrap();
        symlink(&detached, &source).unwrap();
        assert!(verify_inputs(&root).is_err(), "symlink {relative}");
        if relative.is_empty() {
            assert!(verify_inputs(&root.join(".")).is_err());
            assert!(verify_inputs(&root.join("")).is_err());
        }
    }
}

#[cfg(unix)]
#[test]
fn swapped_root_and_file_paths_cannot_change_bytes_already_opened() {
    use std::os::unix::fs::symlink;
    let (temporary, root) = corpus();
    let directory = Directory::open(&root).unwrap();
    let original = directory
        .read("binary/linear.json", MAX_FILE_BYTES)
        .unwrap();
    let detached = temporary.path().join("detached");
    fs::rename(&root, &detached).unwrap();
    let outside = temporary.path().join("outside");
    fs::create_dir(&outside).unwrap();
    symlink(&outside, &root).unwrap();
    assert_eq!(
        directory
            .read("binary/linear.json", MAX_FILE_BYTES)
            .unwrap(),
        original
    );
    fs::rename(
        detached.join("binary/linear.json"),
        detached.join("old.json"),
    )
    .unwrap();
    fs::write(detached.join("binary/linear.json"), b"tampered").unwrap();
    let parsed: Dataset = serde_json::from_slice(&original).unwrap();
    validate_dataset(&parsed).unwrap();
    assert_ne!(
        digest(
            &directory
                .read("binary/linear.json", MAX_FILE_BYTES)
                .unwrap()
        ),
        digest(&original)
    );
}

#[cfg(unix)]
#[test]
fn a_fifo_is_rejected_without_waiting_for_a_writer() {
    let (_temporary, root) = corpus();
    let path = root.join("binary/linear.json");
    fs::remove_file(&path).unwrap();
    rustix::fs::mkfifoat(
        rustix::fs::CWD,
        &path,
        rustix::fs::Mode::from_raw_mode(0o600),
    )
    .unwrap();
    assert!(verify_inputs(&root)
        .unwrap_err()
        .to_string()
        .contains("regular file"));
}

#[test]
fn named_stress_scenarios_have_their_promised_structure() {
    let imbalance = named(Task::Binary, "imbalanced");
    assert_eq!(
        imbalance.rows.iter().filter(|row| row.target == 1).count(),
        8
    );
    let outliers = named(Task::Binary, "outliers");
    assert!(outliers
        .rows
        .iter()
        .any(|row| row.features[0].as_i64().unwrap().abs() > 10_000));
    let extrapolation = named(Task::Regression, "extrapolation");
    assert!(extrapolation
        .rows
        .iter()
        .filter(|row| row.split == Split::Train)
        .all(|row| row.features[0].as_i64().unwrap() <= 0));
    assert!(extrapolation
        .rows
        .iter()
        .filter(|row| row.split == Split::Test)
        .all(|row| row.features[0].as_i64().unwrap() >= 1000));
    assert_eq!(named(Task::Regression, "sparse_signal").columns.len(), 12);
    assert_eq!(named(Task::Multiclass, "many_classes").classes.len(), 7);
    let lagged = named(Task::TimeSeries, "lag_dependence");
    for (index, row) in lagged.rows.iter().enumerate().skip(2) {
        assert_eq!(row.features[0], lagged.rows[index - 1].target);
        assert_eq!(row.features[1], lagged.rows[index - 2].target);
    }
}
