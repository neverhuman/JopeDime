use super::*;
fn packet(dir: &Path) -> PathBuf {
    let path = dir.join("selected.json");
    fs::write(&path, json!({"train":[{"context_sha256":"a","dataset_id":"a","family_id":"f1","source_split":"train","dl_split":"DL-Train","task":"regression"},{"context_sha256":"b","dataset_id":"b","family_id":"f1","source_split":"inner","dl_split":"DL-Train","task":"regression"},{"context_sha256":"c","dataset_id":"c","family_id":"f2","source_split":"train","dl_split":"DL-Train","task":"regression"}],"dev":[{"context_sha256":"d","dataset_id":"d","family_id":"f3","source_split":"train","dl_split":"DL-Val","task":"regression"}]}).to_string()).unwrap();
    path
}
#[test]
fn shuffle_covers_every_context_and_preserves_family_blocks() {
    let d = tempfile::tempdir().unwrap();
    let p = packet(d.path());
    let r = FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl")).unwrap();
    let a = r.training_order(7, 3, false).unwrap();
    assert_eq!(a.iter().collect::<BTreeSet<_>>().len(), 3);
    assert_eq!(
        (a.iter().position(|x| x == "a").unwrap() as i32
            - a.iter().position(|x| x == "b").unwrap() as i32)
            .abs(),
        1
    );
    assert!(r.training_order(7, 2, false).is_err());
    let g = r.training_order(1, 16, true).unwrap();
    assert_eq!(g.len(), 2);
}
#[test]
fn rejects_holdout_multiclass_and_family_overlap() {
    for (from, to) in [
        ("DL-Val", "DL-Holdout"),
        ("regression", "multiclass"),
        ("f3", "f1"),
    ] {
        let d = tempfile::tempdir().unwrap();
        let p = packet(d.path());
        let text = fs::read_to_string(&p).unwrap().replace(from, to);
        fs::write(&p, text).unwrap();
        assert!(
            FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl")).is_err()
        );
    }
}
#[test]
fn utility_has_task_counts_and_history_uses_exact_context_split() {
    let d = tempfile::tempdir().unwrap();
    let p = packet(d.path());
    let history = d.path().join("history.jsonl");
    let r = FullFlywheel::new(&p, d.path().join("out"), history.clone()).unwrap();
    let u = r
        .utility(&["d".into()], &[-1.5], &[true], &["p".into()], "dev")
        .unwrap();
    assert_eq!(u["val_reg"], 1.5);
    assert!(u["val_bin"].is_null());
    assert_eq!(u["n_bin"], 0);
    r.scores(
        &["d".into()],
        &[-1.5],
        "cv_fitness",
        false,
        "candidate",
        "sha",
        1,
    )
    .unwrap();
    let row: Value = serde_json::from_str(fs::read_to_string(history).unwrap().trim()).unwrap();
    assert_eq!(row["split"], "DL-Val/d");
    assert_eq!(row["task"], "regression");
    assert_eq!(row["model"], "candidate:sha");
}
#[test]
fn round_report_requires_coverage_and_emits_native_rates_and_na() {
    let d = tempfile::tempdir().unwrap();
    let p = packet(d.path());
    let mut r =
        FullFlywheel::new(&p, d.path().join("out"), d.path().join("history.jsonl")).unwrap();
    r.status = d.path().join("STATUS.md");
    r.begin();
    assert!(r.finish(json!({"round":1})).is_err());
    r.begin_pack();
    r.packed(&["a".into(), "b".into(), "c".into()], &[1000, 800])
        .unwrap();
    let row = r.finish(json!({"round":1,"promotions":2,"tokens_count":100,"train_wall_seconds":1.,"candidate_dev_utility":{"val_reg":1.5,"val_bin":null,"n_reg":1,"n_bin":0,"per_family":{"f3":1.5}},"candidate_train_utility":{"val_reg":2.,"val_bin":null}})).unwrap();
    assert_eq!(row["distinct_training_contexts"], 3);
    assert_eq!(row["batch_bytes"], 1800);
    assert_eq!(row["train_vs_val_gap_reg"], 0.5);
    assert!(row["train_vs_val_gap_bin"].is_null());
    assert_eq!(row["rust_tokens_per_second"], "N/A (fused path unused)");
    assert!(row["promotions_per_hour"].as_f64().unwrap() > 0.);
    assert_eq!(
        fs::read_to_string(d.path().join("out/summary.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
}
#[test]
fn scientific_receipt_floats_round_trip_bit_exactly() {
    for original in [
        1.1073772627425547_f64,
        1.2591052389885997_f64,
        0.9088130596333871_f64,
    ] {
        let text = serde_json::to_string(&original).unwrap();
        let parsed: Value = serde_json::from_str(&text).unwrap();
        assert_eq!(
            parsed.as_f64().unwrap().to_bits(),
            original.to_bits(),
            "{text}"
        );
    }
}
