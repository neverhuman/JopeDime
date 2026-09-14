use jope_parallel::{manifest::*, prepare::*, transport};
use std::{collections::BTreeSet, fs};

fn rows(n: usize) -> Vec<Vec<String>> {
    (0..n)
        .map(|i| {
            vec![
                i.to_string(),
                (i % 7).to_string(),
                (i as f64 / 10.0).to_string(),
            ]
        })
        .collect()
}
#[test]
fn cap_is_total_and_evaluation_selection_ignores_targets() {
    let mut r = rows(60_000);
    let (a, b) = cap_indices(60_000, 10_000, "regression", &r).unwrap();
    assert_eq!(a.len() + b.len(), ROW_LIMIT);
    for row in &mut r {
        row[2] = "999999".into();
    }
    assert_eq!(
        (a, b),
        cap_indices(60_000, 10_000, "regression", &r).unwrap()
    );
}
#[test]
fn training_cap_preserves_rare_class() {
    let mut r = rows(69_624);
    for row in &mut r {
        row[2] = "0".into();
    }
    r[69_623][2] = "1".into();
    let (a, b) = cap_indices(r.len(), 1000, "binary", &r).unwrap();
    assert_eq!(a.len() + b.len(), ROW_LIMIT);
    assert!(a.contains(&69_623));
}
#[test]
fn five_folds_keep_feature_duplicates_and_entities_together() {
    let mut r = rows(100);
    r.push(r[2].clone());
    r[100][2] = "1234".into();
    let groups = (0..101).map(|i| (i / 3).to_string()).collect::<Vec<_>>();
    let f = folds(&r, "regression", 0, Some(&groups)).unwrap();
    assert_eq!(f.len(), 5);
    let mut held = BTreeSet::new();
    for (t, v) in f {
        assert_eq!(v.contains(&2), v.contains(&100));
        for &i in &v {
            assert!(!t.contains(&i));
            assert!(held.insert(i));
            for &j in &t {
                assert_ne!(groups[i], groups[j]);
            }
        }
    }
    assert_eq!(held.len(), 101);
}
#[test]
fn cut_preprocessing_does_not_use_validation_values() {
    let r = rows(100);
    let f = folds(&r, "regression", 0, None).unwrap();
    let train = f[0].0.iter().map(|&i| r[i].clone()).collect::<Vec<_>>();
    let p = Preprocessing::fit(&train, "regression").unwrap();
    let before = csv_bytes(&p.transform(&train).unwrap()).unwrap();
    let mut altered = r.clone();
    for &i in &f[0].1 {
        altered[i] = vec!["999999".into(), "999999".into(), "-999999".into()];
    }
    let train2 = f[0]
        .0
        .iter()
        .map(|&i| altered[i].clone())
        .collect::<Vec<_>>();
    let p2 = Preprocessing::fit(&train2, "regression").unwrap();
    assert_eq!(before, csv_bytes(&p2.transform(&train2).unwrap()).unwrap());
}
#[test]
fn chronological_origins_never_train_on_future() {
    let r = rows(200);
    for (t, v) in folds(&r, "timeseries", 24, None).unwrap() {
        assert_eq!(v.len(), 24);
        assert!(t.last().unwrap() < v.first().unwrap());
    }
    assert!(folds(&rows(120), "timeseries", 24, None).is_err());
    let (t, e) = cap_indices(60_000, 24, "timeseries", &rows(60_000)).unwrap();
    assert_eq!(t.len() + e.len(), ROW_LIMIT);
    assert_eq!(t.last(), Some(&59_999));
    assert_eq!(e, (0..24).collect::<Vec<_>>());
}
#[test]
fn unseen_categories_and_missing_are_fit_on_train() {
    let r = vec![
        vec!["2".into(), "red".into(), "10".into()],
        vec!["4".into(), "blue".into(), "20".into()],
    ];
    let p = Preprocessing::fit(&r, "regression").unwrap();
    let x = p.features(&["NaN".into(), "green".into()]).unwrap();
    assert_eq!(x, vec![0.0, -1.0]);
    assert_eq!(p.target_mean, 15.0);
}
#[test]
fn incomplete_class_groups_refuse_five_cuts() {
    let mut r = rows(25);
    for row in &mut r {
        row[2] = "0".into();
    }
    r[0][2] = "1".into();
    assert!(folds(&r, "binary", 0, None).is_err());
}
#[test]
fn header_is_schema_driven_and_nomad_style_target_is_last() {
    let tmp = tempfile::tempdir().unwrap();
    let p = tmp.path().join("train.csv");
    fs::write(&p, "x,y\n1,2\n3,4\n").unwrap();
    let meta = serde_json::json!({"output_headers":["x","y"],"target_is_final_column":true});
    let t = Table::read(&p, &meta).unwrap();
    assert!(t.header);
    assert_eq!(t.width, 2);
    assert_eq!(t.rows.len(), 2);
}
#[test]
fn stage_verifies_and_refuses_tampering_and_traversal() {
    let tmp = tempfile::tempdir().unwrap();
    let source = tmp.path().join("source.csv");
    fs::write(&source, "1,2\n").unwrap();
    let m = TransferManifest {
        schema: "parallel.transfer.v1".into(),
        files: vec![FileEntry {
            source: source.clone(),
            destination: "data/originals/a.csv".into(),
            bytes: 4,
            sha256: file_hash(&source).unwrap(),
            dataset: Some("a".into()),
            role: "original".into(),
        }],
    };
    let manifest = tmp.path().join("manifest.json");
    write_json(&manifest, &m).unwrap();
    let stage = tmp.path().join("stage");
    transport::stage(&manifest, &stage).unwrap();
    assert_eq!(transport::verify(&manifest, &stage).unwrap(), 1);
    fs::write(source, "3,4\n").unwrap();
    assert!(transport::verify(&manifest, &stage).is_err());
    let mut bad = m;
    bad.files[0].destination = "../escape".into();
    assert!(bad.validate().is_err());
}
#[test]
fn symlink_escape_is_rejected() {
    use std::os::unix::fs::symlink;
    let tmp = tempfile::tempdir().unwrap();
    symlink("/tmp", tmp.path().join("escape")).unwrap();
    assert!(resolve(tmp.path(), std::path::Path::new("escape/secret")).is_err());
}
