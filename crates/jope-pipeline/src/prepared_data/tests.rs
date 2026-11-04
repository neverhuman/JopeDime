use super::*;
use serde_json::{json, Value};

fn fixture(task: Task) -> (PanelDataset, Value) {
    let classes: Vec<&str> = match task {
        Task::Regression => vec![],
        Task::Binary => vec!["2", "10"],
        Task::Multiclass => vec!["β", "00", "1"],
        _ => unreachable!(),
    };
    let width = classes.len().max(1);
    let rows: Vec<_> = (0..5).flat_map(|fold| (0..width).map(move |class| (fold, class))).enumerate().map(|(index, (fold, class))| {
            json!({ "id": format!("{index:064x}"), "source_index": index,
                "group": format!("{:064x}", index + 100), "fold": fold,
                "values": [if index == 0 { Value::Null } else { json!(index.to_string()) }, if fold == 0 { json!("NA") } else { json!("ordinary") }],
                "target": if classes.is_empty() { index.to_string() } else { classes[class].to_owned() } })
        }).collect();
    let features =
        json!([{"name":"number","kind":"numeric"},{"name":"category","kind":"categorical"}]);
    let source = "a".repeat(64);
    let table = json!({"schema":"jope.prepared-table/v1", "dataset":"fixture", "task":task,
            "source_sha256":source, "partition":"development", "features":features,
            "classes":classes, "rows":rows});
    let dataset: PanelDataset = serde_json::from_value(json!({
            "id":"fixture", "task":task, "metric":Metric::for_task(task).unwrap(),
            "source":{"url":"fixture", "sha256":source, "bytes":1, "metadata_url":"fixture", "metadata_sha256":"b".repeat(64)},
            "rows":rows.len()+width, "features":features, "classes":classes, "group_columns":[],
            "split":{"protocol":"synthetic-fixture", "sha256":"c".repeat(64),
                "development_sha256":sha256_hex(&serde_json::to_vec(&table).unwrap()), "sealed_sha256":"d".repeat(64),
                "development_rows":rows.len(), "sealed_rows":width, "fold_rows":[width,width,width,width,width]},
            "license":{"original":"synthetic", "evidence_url":"fixture", "repository":"synthetic", "note":"synthetic fixture only"},
            "checkpoint_exposure":"synthetic", "description":"synthetic fixture", "group_limitations":"synthetic groups"
        })).unwrap();
    (dataset, table)
}

fn parse_changed(dataset: &PanelDataset, table: &Value) -> Result<DevelopmentData> {
    let bytes = serde_json::to_vec(table)?;
    let mut dataset = dataset.clone();
    dataset.split.development_sha256 = sha256_hex(&bytes);
    parse_development(&bytes, &dataset)
}

/// W3-T04: the fitting-input digest is exactly a fold's training table and targets, in row
/// order: it differs between folds, from an all-row input, and with any cell, order or
/// target, and agrees with the free function a fit worker applies to its own input.
#[test]
fn fitting_input_digest_is_exactly_the_fold_training_input() {
    for task in [Task::Regression, Task::Binary, Task::Multiclass] {
        let (dataset, table) = fixture(task);
        let data = parse_changed(&dataset, &table).unwrap();
        let digests = (0..5)
            .map(|fold| {
                data.cut(fold)
                    .unwrap()
                    .train
                    .fitting_input_sha256()
                    .unwrap()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(digests.len(), 5, "every fold has its own input");
        let train = data.cut(1).unwrap().train;
        let digest = train.fitting_input_sha256().unwrap();
        assert_eq!(
            fitting_input_sha256(&train.table, &train.targets).unwrap(),
            digest
        );
        let all = raw_table(
            &dataset.features,
            &data
                .rows
                .iter()
                .map(|row| row.values.clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let all_targets = data
            .rows
            .iter()
            .map(|row| encode_target(task, &dataset.classes, &row.target).unwrap())
            .collect::<Vec<_>>();
        assert!(!digests.contains(&fitting_input_sha256(&all, &all_targets).unwrap()));
        let mut reordered = train.targets.clone();
        reordered.reverse();
        assert_ne!(
            fitting_input_sha256(&train.table, &reordered).unwrap(),
            digest
        );
        // Row 0 is in fold 0, so it is part of fold 1's training input.
        let mut changed = table.clone();
        changed["rows"][0]["values"][0] = json!("999");
        let changed = parse_changed(&dataset, &changed).unwrap();
        assert_ne!(
            changed
                .cut(1)
                .unwrap()
                .train
                .fitting_input_sha256()
                .unwrap(),
            digest
        );
        assert!(fitting_input_sha256(&train.table, &train.targets[1..]).is_err());
    }
}

#[test]
fn tampered_bytes_are_rejected_before_parsing() {
    let (dataset, _) = fixture(Task::Binary);
    assert!(parse_development(b"not json", &dataset)
        .unwrap_err()
        .to_string()
        .contains("sha256 mismatch"));
}

#[test]
fn authenticated_metadata_still_must_match_the_frozen_descriptor() {
    let (dataset, table) = fixture(Task::Binary);
    for (field, value) in [
        ("partition", json!("sealed_final")),
        ("source_sha256", json!("e".repeat(64))),
        ("classes", json!(["10", "2"])),
        ("task", json!("regression")),
    ] {
        let mut changed = table.clone();
        changed[field] = value;
        assert!(parse_changed(&dataset, &changed).is_err(), "{field}");
    }
    let mut changed = table;
    changed["features"][1]["kind"] = json!("numeric");
    assert!(parse_changed(&dataset, &changed).is_err());
}

#[test]
fn rows_groups_and_actual_fold_membership_are_checked() {
    let (dataset, table) = fixture(Task::Binary);
    for field in ["id", "source_index"] {
        let mut changed = table.clone();
        changed["rows"][1][field] = changed["rows"][0][field].clone();
        assert!(parse_changed(&dataset, &changed).is_err(), "{field}");
    }
    let mut changed = table.clone();
    changed["rows"][2]["group"] = changed["rows"][0]["group"].clone();
    assert!(parse_changed(&dataset, &changed)
        .unwrap_err()
        .to_string()
        .contains("group crossed"));
    let mut changed = table.clone();
    changed["rows"][0]["fold"] = json!(1);
    assert!(parse_changed(&dataset, &changed).is_err());
    let mut changed = table.clone();
    changed["rows"][0]["fold"] = Value::Null;
    assert!(parse_changed(&dataset, &changed).is_err());
    let mut changed = table;
    changed["rows"][0]["target"] = json!("10");
    assert!(parse_changed(&dataset, &changed)
        .unwrap_err()
        .to_string()
        .contains("lacks a declared class"));
}

#[test]
fn cuts_preserve_original_order_labels_nulls_and_declared_class_order() {
    let (dataset, mut table) = fixture(Task::Binary);
    table["rows"].as_array_mut().unwrap().reverse();
    let data = parse_changed(&dataset, &table).unwrap();
    let cut = data.cut(0).unwrap();
    assert_eq!(
        cut.hold
            .rows
            .iter()
            .map(|row| row.source_index)
            .collect::<Vec<_>>(),
        vec![1, 0]
    );
    assert_eq!(cut.hold.labels, ["10", "2"]);
    assert_eq!(cut.hold.targets, [1., 0.]);
    assert_eq!(cut.hold.rows[1].values[0], None);
    assert_eq!(cut.hold.table.rows[1][0], "");
    assert_eq!(cut.hold.rows[1].values[1].as_deref(), Some("NA"));
    assert_eq!(cut.hold.table.rows[1][1], "jope.category.v1:\"NA\"");
    assert!(cut.train.rows.iter().all(|row| row.fold != 0));
    let ids = cut.hold.rows.iter().map(|row| &row.id).collect::<Vec<_>>();
    assert_eq!(
        cut.hold.rows_sha256,
        sha256_hex(&serde_json::to_vec(&ids).unwrap())
    );
    assert!(data.cut(5).is_err());
}

#[test]
fn raw_encoding_distinguishes_missing_from_every_literal_category() {
    let features = vec![Column {
        name: "category".into(),
        kind: ColumnKind::Categorical,
    }];
    let originals = [
        None,
        Some(""),
        Some(" "),
        Some("null"),
        Some("NA"),
        Some("NaN"),
        Some("jope.category.v1:\"NA\""),
    ];
    let values = originals
        .iter()
        .map(|value| vec![value.map(str::to_owned)])
        .collect::<Vec<_>>();
    let table = raw_table(&features, &values).unwrap();
    assert_eq!(table.rows[0][0], "");
    let distinct = table
        .rows
        .iter()
        .map(|row| &row[0])
        .collect::<BTreeSet<_>>();
    assert_eq!(distinct.len(), originals.len());
    for (row, original) in table.rows.iter().skip(1).zip(originals.iter().skip(1)) {
        let decoded: String =
            serde_json::from_str(row[0].strip_prefix(CATEGORY_PREFIX).unwrap()).unwrap();
        assert_eq!(Some(decoded.as_str()), *original);
    }
}

/// W1-T01 at the ASM development loader: a dataset the committed
/// ten-regression release names is admitted only as that member. An
/// ad-hoc partition with a self-consistent panel entry under a member's
/// name is refused before any row is interpreted.
#[test]
fn a_release_members_name_never_admits_other_development_bytes() {
    let (mut dataset, mut table) = fixture(Task::Regression);
    // Positive control: a dataset outside the release, with no release
    // admission.
    assert!(parse_changed(&dataset, &table).unwrap().release.is_none());
    dataset.id = "197_cpu_act".into();
    table["dataset"] = json!("197_cpu_act");
    let refused = parse_changed(&dataset, &table);
    assert!(
        matches!(&refused, Err(e) if e.to_string().contains("regression release")),
        "an ad-hoc development input was admitted under a release member's name"
    );
}

/// Local evidence (the prepared ASM panel is not in the repository; run
/// with `JOPE_W1_DEVELOPMENT_ROOT=<prepared root>` and `--run-ignored
/// only`): every committed member's real development partition is admitted
/// as that member, no member's bytes pass under another member's entry,
/// and a dataset outside the release carries no admission.
#[test]
#[ignore = "needs the prepared ASM panel (JOPE_W1_DEVELOPMENT_ROOT)"]
fn the_real_development_partitions_are_admitted_as_their_release_members() {
    use jope_datasets::regression_release::{committed, COMMITTED_RELEASE_ID};
    let root = std::path::PathBuf::from(
        std::env::var_os("JOPE_W1_DEVELOPMENT_ROOT").expect("JOPE_W1_DEVELOPMENT_ROOT"),
    );
    let panel: crate::experiment::Panel = serde_json::from_slice(
        &std::fs::read(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../benchmarks/panel.json"
        ))
        .unwrap(),
    )
    .unwrap();
    let release = committed().unwrap();
    let members: Vec<_> = panel
        .datasets
        .iter()
        .filter(|d| d.task == Task::Regression)
        .collect();
    assert_eq!(members.len(), release.members.len());
    for d in &members {
        let admission = load_development(&root, d).unwrap().release.unwrap();
        let member = release.member(&d.id).unwrap();
        assert_eq!(admission.release_id, COMMITTED_RELEASE_ID);
        assert_eq!(admission.dataset_sha256, member.raw.sha256);
        assert_eq!(
            admission.development_alignment_sha256,
            member.rows.development_alignment_sha256
        );
    }
    let bytes = std::fs::read(
        root.join("development")
            .join(format!("{}.json", members[0].id)),
    )
    .unwrap();
    let mut other = members[1].clone();
    other.split.development_sha256 = sha256_hex(&bytes);
    let refused = parse_development(&bytes, &other).unwrap_err().to_string();
    assert!(refused.contains("regression release"), "{refused}");
    let outside = panel
        .datasets
        .iter()
        .find(|d| d.task != Task::Regression)
        .unwrap();
    assert!(load_development(&root, outside).unwrap().release.is_none());
}

#[test]
fn invalid_numeric_cells_and_targets_are_not_imputed_or_dropped() {
    let (dataset, table) = fixture(Task::Regression);
    for bad in ["nan", "inf", "not-a-number", ""] {
        let mut changed = table.clone();
        changed["rows"][0]["values"][0] = json!(bad);
        assert!(parse_changed(&dataset, &changed).is_err());
        let mut changed = table.clone();
        changed["rows"][0]["target"] = json!(bad);
        assert!(parse_changed(&dataset, &changed).is_err());
    }
}
