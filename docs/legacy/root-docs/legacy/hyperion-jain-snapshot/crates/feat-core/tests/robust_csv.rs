use feat_core::data::load_train_hold;
use feat_core::encode::{fit_preprocess_plan_with_cap, CatEncoding, ColumnEncoding};
use feat_core::ingest::read_frame;
use feat_core::task::{infer_task, TaskInferConfig};
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(name);
    p
}

#[test]
fn robust_csv_ingestion_is_finite_and_lossy_safe() {
    let p = tmp("jain_robust_csv.csv");
    let bytes = b"\xEF\xBB\xBFnum,cat,cat,bool,huge,target\r\n\"1,2\",\"a\"\"b\",x,true,1e999,1\r\n,NA,y,false,-inf,0\r\n7,bad_\xFF,z,yes,NaN,1,EXTRA\r\n";
    std::fs::write(&p, bytes).unwrap();

    let pair = load_train_hold(&p, &p, &["target"], "target", CatEncoding::Hybrid).unwrap();
    assert_eq!(pair.train.n(), 3);
    assert!(pair.train.names.iter().any(|n| n.starts_with("cat__dup1")));
    assert!(pair
        .preprocess_issues
        .iter()
        .any(|i| i.contains("duplicate header")));
    assert!(
        pair.train
            .x
            .data
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "all emitted features must be finite [0,1]"
    );
}

#[test]
fn high_cardinality_uses_hash_and_unseen_categories_stay_finite() {
    let tr = tmp("jain_high_card_train.csv");
    let ho = tmp("jain_high_card_hold.csv");
    let mut train = String::from("id,cat,y\n");
    for i in 0..40 {
        train.push_str(&format!("{i},level_{i},{}\n", i % 2));
    }
    std::fs::write(&tr, train).unwrap();
    std::fs::write(&ho, "id,cat,y\n100,new_level,1\n101,,0\n").unwrap();

    let pair = load_train_hold(&tr, &ho, &["y"], "y", CatEncoding::Hybrid).unwrap();
    assert!(pair.train.names.iter().any(|n| n == "cat::hash00"));
    assert!(pair
        .hold
        .x
        .data
        .iter()
        .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
}

#[test]
fn width_cap_shrinks_then_frequency_backup() {
    let p = tmp("jain_width_cap.csv");
    let mut csv = String::from("a,b,y\n");
    for i in 0..80 {
        csv.push_str(&format!("a_{i},b_{i},{}\n", i % 2));
    }
    std::fs::write(&p, csv).unwrap();
    let frame = read_frame(&p, &["y"], "y").unwrap();
    let targets: Vec<Vec<f32>> = frame
        .targets
        .iter()
        .map(|c| c.tokens.iter().map(|t| t.parse::<f32>().unwrap()).collect())
        .collect();
    let task = infer_task(&targets, frame.primary, &TaskInferConfig::default());
    let plan = fit_preprocess_plan_with_cap(&frame, &task, CatEncoding::Hybrid, 12);
    assert!(plan.output_width() <= 12);
    assert!(plan.issues.iter().any(|i| i.contains("width cap")));
    assert!(plan
        .columns
        .iter()
        .any(|c| matches!(c.encoding, ColumnEncoding::Frequency { .. })));
}
