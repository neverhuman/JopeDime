use super::*;

#[test]
fn a_three_rank_original_continues_with_its_own_partition() {
    let v2 = Plan::new(3, 2048, 256).unwrap().record();
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let policy = continuation_fixture(root, &v2);
    let recovered = continue_with(root, &policy, &v2).unwrap();
    assert_eq!(recovered["execution_partition"], v2);
    assert_eq!(recovered["target_optimizer_step"], 4000);
    assert_eq!(recovered["original_next_global_example"], 3000 * 2048);
    // The v1 entry point cannot express it; any other request is refused.
    assert!(checkpoint_recovery::verify_continuation(
        root,
        &policy,
        &"e".repeat(64),
        &"c".repeat(64),
        jope_parallel::checkpoint_recovery::TrainingTopology {
            world: 3,
            micro: 256,
            global: 2048
        },
        1000,
    )
    .is_err());
    for other in [
        Plan::new(3, 2048, 128).unwrap().record(),
        Plan::new(4, 2048, 256).unwrap().record(),
        training_partition(4, 256, 2048).unwrap(),
    ] {
        assert!(continue_with(root, &policy, &other).is_err());
    }
    // A parent whose recorded v2 partition disagrees with its launch topology.
    let mut state: Value = read_json(&root.join("state.json")).unwrap();
    state["execution_partition"]["shards"][0]["groups"] = json!(87);
    write_json(&root.join("state.json"), &state).unwrap();
    let forged = rebind_policy(root, &policy);
    let mut complete: Value = read_json(&root.join("complete.json")).unwrap();
    complete["latest"]["state_sha256"] = forged["parent_checkpoint_sha256"].clone();
    write_json(&root.join("complete.json"), &complete).unwrap();
    let forged = rebind_policy(root, &forged);
    let error = continue_with(root, &forged, &v2).unwrap_err();
    assert!(error.to_string().contains("launch topology"), "{error}");
    // v1 originals recover identically through both entry points.
    let dir = tempfile::tempdir().unwrap();
    let v1 = training_partition(8, 256, 2048).unwrap();
    let policy = continuation_fixture(dir.path(), &v1);
    assert_eq!(
        checkpoint_recovery::verify_continuation(
            dir.path(),
            &policy,
            &"e".repeat(64),
            &"c".repeat(64),
            jope_parallel::checkpoint_recovery::TrainingTopology {
                world: 8,
                micro: 256,
                global: 2048
            },
            1000,
        )
        .unwrap(),
        continue_with(dir.path(), &policy, &v1).unwrap()
    );
    assert!(continue_with(dir.path(), &policy, &v2).is_err());
}
