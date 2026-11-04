use super::*;

#[test]
fn a_three_rank_checkpoint_commits_and_reloads_through_the_real_path() {
    let committed = three_rank_checkpoint(256, &|_| {}).unwrap();
    let loaded = committed.load().unwrap();
    assert_eq!((loaded.step, loaded.format), (1, Format::Complete));
    assert_eq!(loaded.state["execution_partition"], committed.plan.record());
    // Exposure, journals and RNG/sampler state resume per rank.
    let rows =
        task_contract::partition_rows_per_rank(&loaded.state["execution_partition"]).unwrap();
    assert_eq!(rows, [688, 680, 680]);
    let state: partition::State =
        serde_json::from_value(loaded.state["partition_state"].clone()).unwrap();
    let next = state.resume(&committed.plan, SEED).unwrap();
    for rank in 0..3 {
        assert_eq!(
            training_exposure::checkpoint_baseline_rows(&loaded.state, rank, &rows).unwrap(),
            Some(counts(rows[rank] as u64))
        );
        let reconciled =
            consumption::reconcile(&fs::read(&committed.journals[rank]).unwrap(), Some(1)).unwrap();
        assert_eq!(reconciled.committed_through, 1);
        assert_eq!(reconciled.committed.presentations, rows[rank] as u64);
        assert_eq!(reconciled.next.serial, next[rank][0].serial);
        assert_eq!(reconciled.next.dropout_seed, next[rank][0].dropout_seed);
    }
    // The in-commit verification never publishes an incomplete v2 manifest.
    for (case, edit) in [
        (
            "partition state absent",
            Box::new(|s: &mut Value| drop(s.as_object_mut().unwrap().remove("partition_state")))
                as Box<dyn Fn(&mut Value)>,
        ),
        (
            "v1 objective",
            Box::new(|s: &mut Value| {
                s["objective"] = objective::identity();
                s["objective_sha256"] = json!(objective::identity_sha256());
            }),
        ),
        (
            "legacy format",
            Box::new(|s: &mut Value| drop(s.as_object_mut().unwrap().remove("checkpoint_format"))),
        ),
    ] {
        assert!(
            three_rank_checkpoint(256, &*edit).is_err(),
            "{case}: published"
        );
    }
    // Journals of another partition never bind a v2 checkpoint.
    let error = three_rank_checkpoint(128, &|_| {}).err().unwrap();
    assert!(error.to_string().contains("another partition"), "{error:#}");
    // Every published-state alteration is refused on load.
    type Edit = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(&str, Edit)> = vec![
        (
            "partition state removed",
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("partition_state"))),
        ),
        (
            "rank cursor",
            Box::new(|s| s["partition_state"]["ranks"][1]["sampler_cursor"] = json!(1)),
        ),
        (
            "committed parameters",
            Box::new(|s| {
                s["partition_state"]["committed_update"]["parameter_sha256"] = json!("d".repeat(64))
            }),
        ),
        ("partition seed", Box::new(|s| s["seed"] = json!(SEED + 1))),
        (
            "shard record",
            Box::new(|s| s["execution_partition"]["shards"][0]["groups"] = json!(87)),
        ),
        (
            "equal exposure split",
            Box::new(|s| {
                s["committed_exposure_by_rank"][0]["counts"] = json!(counts(680));
                s["committed_exposure_by_rank"][1]["counts"] = json!(counts(688));
            }),
        ),
        (
            "format removed",
            Box::new(|s| drop(s.as_object_mut().unwrap().remove("checkpoint_format"))),
        ),
        (
            "v1 record",
            Box::new(|s| s["execution_partition"] = training_partition(1, 256, 2048).unwrap()),
        ),
    ];
    for (case, edit) in cases {
        let committed = three_rank_checkpoint(256, &|_| {}).unwrap();
        committed.rewrite_state(edit);
        assert!(committed.load().is_err(), "{case}: accepted");
        // Not even a legacy-tolerant load accepts it.
        assert!(
            load_latest(&committed.root, &committed.run, &Expectation::default()).is_err(),
            "{case}: accepted without require_complete"
        );
    }
}
