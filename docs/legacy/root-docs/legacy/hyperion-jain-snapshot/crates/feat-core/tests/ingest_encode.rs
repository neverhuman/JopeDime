//! Step-1 guarantees: typed ingestion scales numeric data, encodes categoricals instead of dropping
//! them, and auto-detects/scales targets correctly.
use feat_core::data::{load_csv, load_train_hold};
use feat_core::encode::CatEncoding;
use feat_core::ingest::read_header;
use feat_core::targets::resolve_target_columns;
use feat_core::task::TaskKind;
use std::io::Write;
use std::path::PathBuf;

fn write_tmp(name: &str, content: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    let mut f = std::fs::File::create(&p).unwrap();
    f.write_all(content.as_bytes()).unwrap();
    p
}

#[test]
fn all_numeric_is_scaled_finite() {
    let csv = "a,b,TARGET_y\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n7.0,8.0,1\n9,10,1\n";
    let tr = write_tmp("ie_num_tr.csv", csv);
    let ho = write_tmp("ie_num_ho.csv", csv);
    let old = load_csv(&tr, &["TARGET_y"], "TARGET_y").unwrap();
    let pair = load_train_hold(&tr, &ho, &["TARGET_y"], "TARGET_y", CatEncoding::Hybrid).unwrap();
    let neu = pair.train;
    assert_eq!(old.names, neu.names);
    assert_eq!(old.y, neu.y, "binary {{0,1}} target maps identically");
    assert!(
        pair.cat_cols.is_empty(),
        "no categorical columns on numeric data"
    );
    assert_eq!(neu.x.col(0), &[0.0, 0.25, 0.5, 0.75, 1.0]);
    assert!(neu
        .x
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
}

#[test]
fn categorical_is_encoded_not_dropped() {
    let csv = "num,region,TARGET_y\n1.0,EAST,0\n2.0,WEST,1\n3.0,EAST,0\n4.0,NORTH,1\n";
    let tr = write_tmp("ie_cat_tr.csv", csv);
    let ho = write_tmp("ie_cat_ho.csv", csv);
    // OLD behavior: the string column becomes all-NaN (silently dropped).
    let old = load_csv(&tr, &["TARGET_y"], "TARGET_y").unwrap();
    let ri = old.names.iter().position(|n| n == "region").unwrap();
    assert!(
        old.x.col(ri).iter().all(|v| v.is_nan()),
        "load_csv drops categorical to NaN"
    );
    // NEW behavior: expanded to finite hybrid one-hot columns.
    let pair = load_train_hold(&tr, &ho, &["TARGET_y"], "TARGET_y", CatEncoding::Hybrid).unwrap();
    let neu = pair.train;
    let east = neu.names.iter().position(|n| n == "region==EAST").unwrap();
    let west = neu.names.iter().position(|n| n == "region==WEST").unwrap();
    let other = neu
        .names
        .iter()
        .position(|n| n == "region::__OTHER__")
        .unwrap();
    assert!(pair.cat_cols.contains(&east), "region flagged categorical");
    assert_eq!(neu.x.col(east).to_vec(), vec![1.0, 0.0, 1.0, 0.0]);
    assert_eq!(neu.x.col(west).to_vec(), vec![0.0, 1.0, 0.0, 0.0]);
    assert_eq!(neu.x.col(other).to_vec(), vec![0.0, 0.0, 0.0, 0.0]);
    assert!(neu
        .x
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
}

#[test]
fn task_detection_regression_and_binary() {
    // Regression: float target with many distinct values -> scaled labels, not binarized.
    let reg = "a,y\n1,0.13\n2,0.27\n3,0.55\n4,0.91\n5,1.7\n";
    let p = write_tmp("ie_reg.csv", reg);
    let pair = load_train_hold(&p, &p, &["y"], "y", CatEncoding::Hybrid).unwrap();
    assert_eq!(pair.task.kind, TaskKind::Regression);
    assert_eq!(pair.train.y[0], 0.0);
    assert_eq!(pair.train.y[4], 1.0);
    assert!((pair.encoders.target_scaler.inverse(pair.train.y[2]) - 0.55).abs() < 1e-6);
    // Multi-class: integer target with 3 distinct values.
    let mc = "a,y\n1,0\n2,1\n3,2\n4,1\n5,0\n";
    let pm = write_tmp("ie_mc.csv", mc);
    let pair2 = load_train_hold(&pm, &pm, &["y"], "y", CatEncoding::Hybrid).unwrap();
    assert_eq!(pair2.task.kind, TaskKind::MultiClass);
    assert_eq!(pair2.task.n_classes, 3);
}

#[test]
fn auto_last_column_binary_routes_to_binary_task() {
    let csv = "f0,f1,label\n1,10,0\n2,11,1\n3,12,0\n4,13,1\n";
    let p = write_tmp("ie_auto_binary.csv", csv);
    let header = read_header(&p).unwrap();
    let resolved = resolve_target_columns(&header, "auto");
    let tref: Vec<&str> = resolved.targets.iter().map(|s| s.as_str()).collect();

    let pair = load_train_hold(&p, &p, &tref, &resolved.primary, CatEncoding::Hybrid).unwrap();

    assert_eq!(resolved.primary, "label");
    assert_eq!(pair.task.kind, TaskKind::Binary);
    assert_eq!(pair.train.names, vec!["f0", "f1"]);
    assert_eq!(pair.train.y, vec![0.0, 1.0, 0.0, 1.0]);
}

#[test]
fn auto_last_column_continuous_routes_to_regression_task() {
    let csv = "f0,f1,score\n1,10,0.12\n2,11,0.19\n3,12,0.37\n4,13,0.82\n5,14,1.33\n";
    let p = write_tmp("ie_auto_regression.csv", csv);
    let header = read_header(&p).unwrap();
    let resolved = resolve_target_columns(&header, "auto");
    let tref: Vec<&str> = resolved.targets.iter().map(|s| s.as_str()).collect();

    let pair = load_train_hold(&p, &p, &tref, &resolved.primary, CatEncoding::Hybrid).unwrap();

    assert_eq!(resolved.primary, "score");
    assert_eq!(pair.task.kind, TaskKind::Regression);
    assert_eq!(pair.train.names, vec!["f0", "f1"]);
    assert!((pair.encoders.target_scaler.inverse(pair.train.y[2]) - 0.37).abs() < 1e-6);
}

#[test]
fn auto_last_column_bounded_integer_routes_to_multiclass_task() {
    let csv = "f0,f1,class\n1,10,0\n2,11,1\n3,12,2\n4,13,1\n5,14,0\n";
    let p = write_tmp("ie_auto_multiclass.csv", csv);
    let header = read_header(&p).unwrap();
    let resolved = resolve_target_columns(&header, "auto");
    let tref: Vec<&str> = resolved.targets.iter().map(|s| s.as_str()).collect();

    let pair = load_train_hold(&p, &p, &tref, &resolved.primary, CatEncoding::Hybrid).unwrap();

    assert_eq!(resolved.primary, "class");
    assert_eq!(pair.task.kind, TaskKind::MultiClass);
    assert_eq!(pair.task.n_classes, 3);
    assert_eq!(pair.train.names, vec!["f0", "f1"]);
}
