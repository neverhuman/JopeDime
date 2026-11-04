use super::*;

#[test]
fn three_rank_exposure_baselines_follow_each_shard() {
    let rows = [688usize, 680, 680];
    let state = json!({"step":2,"committed_exposure_by_rank":
        (0..3).map(|r| json!({"rank":r,"counts":counts(2 * rows[r] as u64)})).collect::<Vec<_>>()});
    for rank in 0..3 {
        assert_eq!(
            training_exposure::checkpoint_baseline_rows(&state, rank, &rows).unwrap(),
            Some(counts(2 * rows[rank] as u64))
        );
        // No single equal split describes these ranks (the reproduced defect).
        assert!(training_exposure::checkpoint_baseline(&state, rank, 3, rows[rank]).is_err());
    }
    assert!(training_exposure::checkpoint_baseline_rows(&state, 0, &[680, 688, 680]).is_err());
    assert!(training_exposure::checkpoint_baseline_rows(&state, 3, &rows).is_err());
    // An empty rank consumes nothing and must commit nothing.
    let mut state = json!({"step":2,"committed_exposure_by_rank":[
        {"rank":0,"counts":counts(16)},{"rank":1,"counts":counts(0)}]});
    assert!(
        training_exposure::checkpoint_baseline_rows(&state, 1, &[8, 0])
            .unwrap()
            .is_some()
    );
    state["committed_exposure_by_rank"][1]["counts"] = json!(counts(8));
    assert!(training_exposure::checkpoint_baseline_rows(&state, 1, &[8, 0]).is_err());
    assert!(training_exposure::checkpoint_baseline_rows(&state, 0, &[0, 0]).is_err());
}

#[test]
fn the_v2_validation_panel_is_the_v1_eight_rank_panel_split_by_the_plan() {
    let index = validation_index();
    let v1_eight = validation_panel(&index, 8, SEED).unwrap().concat();
    assert_eq!(v1_eight.len(), partition::VALIDATION_PANEL_ROWS);
    for (world, micro) in [(1, 256), (3, 256), (3, 64), (5, 128), (8, 256)] {
        let plan = Plan::new(world, partition::VALIDATION_PANEL_ROWS, micro).unwrap();
        let panel = validation_panel_partitioned(&index, &plan, SEED).unwrap();
        assert_eq!(panel.len(), world);
        for (rank, slots) in panel.iter().enumerate() {
            assert_eq!(
                slots.iter().map(Vec::len).collect::<Vec<_>>(),
                plan.slots(rank, SEED, 0)
                    .unwrap()
                    .iter()
                    .map(|s| s.rows)
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            panel.concat().concat(),
            v1_eight,
            "world {world}: the panel does not depend on the rank count"
        );
    }
    // v1 keeps its world set; a panel plan of another size is refused.
    assert!(validation_panel(&index, 3, SEED).is_err());
    assert!(validation_panel_partitioned(&index, &Plan::new(3, 4096, 256).unwrap(), SEED).is_err());
}
