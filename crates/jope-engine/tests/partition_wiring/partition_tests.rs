use super::*;

#[test]
fn the_trainer_gate_keeps_v1_and_admits_only_an_explicit_v2() {
    // v1 is unchanged: world 3 is still refused; the v1 record is the old one.
    let error = execution(Version::V1, 3, 256, 2048, false).unwrap_err();
    assert!(
        error.to_string().contains("invalid training topology"),
        "{error}"
    );
    let v1 = execution(Version::V1, 8, 256, 2048, false).unwrap();
    assert_eq!(*v1.record(), training_partition(8, 256, 2048).unwrap());
    assert_eq!(v1.rows_per_rank(), vec![256; 8]);
    assert_eq!(v1.accumulation(3).unwrap(), 1);
    assert_eq!(v1.shard(3).unwrap(), None);
    assert_eq!(v1.objective_sha256(), objective::identity_sha256());
    assert!(execution(Version::V1, 8, 1024, 8192, true).is_ok());
    // v2: three ranks of 86, 85 and 85 comparisons.
    let v2 = execution(Version::V2, 3, 256, 2048, false).unwrap();
    assert_eq!(*v2.record(), Plan::new(3, 2048, 256).unwrap().record());
    assert_eq!(v2.rows_per_rank(), [688, 680, 680]);
    assert_eq!(
        (0..3)
            .map(|r| v2.accumulation(r).unwrap())
            .collect::<Vec<_>>(),
        [3, 3, 3]
    );
    assert_eq!(v2.shard(2).unwrap().unwrap().microbatches, [32, 32, 21]);
    assert_eq!(v2.objective_sha256(), partition::objective_sha256());
    assert_eq!(v2.record()["rng"], consumption::RNG_ID);
    for (world, micro, global, qualification) in [
        (0, 256, 2048, false),
        (9, 256, 2048, false),
        (3, 512, 2048, false),
        (3, 256, 1000, false),
        (3, 1024, 8192, true),
    ] {
        assert!(
            execution(Version::V2, world, micro, global, qualification).is_err(),
            "v2 {world} {micro} {global} {qualification}"
        );
    }
    // Selection is by the frozen record, checked against the topology.
    let declared = Execution::declared(v2.record(), 3, 256, 2048, false).unwrap();
    assert_eq!(declared.version(), Version::V2);
    assert_eq!(
        Execution::declared(v1.record(), 8, 256, 2048, false)
            .unwrap()
            .version(),
        Version::V1
    );
    assert!(Execution::declared(v2.record(), 4, 256, 2048, false).is_err());
    assert!(Execution::declared(v2.record(), 3, 128, 2048, false).is_err());
    let mut forged = v2.record().clone();
    forged["shards"][0]["groups"] = json!(87);
    assert!(Execution::declared(&forged, 3, 256, 2048, false).is_err());
    assert!(Version::of_record(&json!({"schema":"parallel.training_partition.v3"})).is_err());
    assert!(Version::of_record(&json!({})).is_err());
    // The records are self-describing and recomputable outside the neural build.
    for record in [v1.record(), v2.record()] {
        assert_eq!(task_contract::recompute_partition(record).unwrap(), *record);
    }
    assert_eq!(
        task_contract::partition_rows_per_rank(v2.record()).unwrap(),
        [688, 680, 680]
    );
    assert!(task_contract::partition_rows_per_rank(&forged).is_err());
}

#[test]
fn a_three_rank_journal_records_each_uneven_shard() {
    let dir = tempfile::tempdir().unwrap();
    let plan = Plan::new(3, 2048, 256).unwrap();
    // Negative controls: the v1 equation still refuses an uneven header, and
    // a v2 header must carry exactly its own rank's shard.
    let error = Journal::create(&dir.path().join("v1.jsonl"), header(0, 3, 256, 2048, 3))
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("partition is inconsistent"),
        "{error}"
    );
    assert!(consumption::sampler_serial(0, 2048, 3, 1, 0, 256).is_err());
    for (case, h, shard) in [
        (
            "another rank's shard",
            header(0, 3, 256, 2048, 3),
            plan.shards()[1].clone(),
        ),
        (
            "wrong accumulation",
            header(0, 3, 256, 2048, 2),
            plan.shards()[0].clone(),
        ),
        (
            "another microbatch",
            header(0, 3, 128, 2048, 3),
            plan.shards()[0].clone(),
        ),
    ] {
        let path = dir.path().join(format!("{case}.jsonl"));
        assert!(
            Journal::create_with_shard(&path, h, Some(shard)).is_err(),
            "{case}"
        );
    }
    for rank in 0..3 {
        let shard = plan.shards()[rank].clone();
        let path = dir.path().join(format!("rank-{rank}.jsonl"));
        let mut journal = Journal::create_with_shard(
            &path,
            header(rank, 3, 256, 2048, shard.microbatches.len()),
            Some(shard.clone()),
        )
        .unwrap();
        journal_step(&mut journal, &plan, rank, 0);
        // Step 1: the last microbatch is short (22 or 21 comparisons), so a
        // full 256-row microbatch or another slot's cursor is refused; a
        // refused event leaves the journal usable.
        let slots = plan.slots(rank, SEED, 1).unwrap();
        record(&mut journal, 1, &slots[0], slots[0].serial, slots[0].rows).unwrap();
        record(&mut journal, 1, &slots[1], slots[1].serial, slots[1].rows).unwrap();
        assert!(slots[2].rows < 256);
        assert!(record(&mut journal, 1, &slots[2], slots[2].serial, 256).is_err());
        assert!(record(&mut journal, 1, &slots[2], slots[1].serial, slots[2].rows).is_err());
        record(&mut journal, 1, &slots[2], slots[2].serial, slots[2].rows).unwrap();
        journal.applied(1, 0.5).unwrap();
        journal.commit(2, &"c".repeat(64)).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert!(String::from_utf8_lossy(&bytes)
            .lines()
            .next()
            .unwrap()
            .contains("\"shard\""));
        let reconciled = consumption::reconcile(&bytes, Some(2)).unwrap();
        assert_eq!(reconciled.shard.as_ref(), Some(&shard));
        assert_eq!(
            reconciled.committed.presentations,
            (2 * shard.groups * 8) as u64
        );
        assert_eq!(
            reconciled.next.serial,
            plan.slots(rank, SEED, 2).unwrap()[0].serial,
            "the next cursor is the rank's first slot of the next step"
        );
    }
    // An empty rank: more ranks than comparisons, zero microbatches.
    let wide = Plan::new(12, 88, 16).unwrap();
    let path = dir.path().join("empty.jsonl");
    let mut journal = Journal::create_with_shard(
        &path,
        header(11, 12, 16, 88, 0),
        Some(wide.shards()[11].clone()),
    )
    .unwrap();
    journal.applied(0, 0.0).unwrap();
    journal.commit(1, &"c".repeat(64)).unwrap();
    let reconciled = consumption::reconcile(&fs::read(&path).unwrap(), Some(1)).unwrap();
    assert_eq!(reconciled.committed.presentations, 0);
    assert_eq!(reconciled.applied_updates, 1);
}

#[test]
fn v1_records_are_unchanged() {
    // The v1 journal attempt line is byte-identical to the pre-change bytes.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rank-1.jsonl");
    drop(Journal::create(&path, header(1, 8, 256, 2048, 1)).unwrap());
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        format!(
            "{{\"event\":\"attempt\",\"header\":{{\"schema\":\"jope.training_consumption.v1\",\
             \"rank\":1,\"world\":8,\"microbatch\":256,\"global_batch\":2048,\"accumulation\":1,\
             \"seed\":7,\"sampler_id\":\"s\",\"rng\":\"mix_seed_local_step_rank_microbatch_v1\",\
             \"sampling_offset\":0,\"initial_optimizer_step\":0,\"target_optimizer_step\":10,\
             \"corpus_sha256\":\"{CORPUS}\",\"replay_policy_sha256\":null,\
             \"resume_state_sha256\":null}}}}\n"
        )
    );
    // The v1 execution record is the literal it always was.
    assert_eq!(
        training_partition(8, 256, 2048).unwrap(),
        json!({"schema":"parallel.training_partition.v1","world":8,"microbatch":256,
            "global_batch":2048,"accumulation":1,"rng":"mix_seed_local_step_rank_microbatch_v1",
            "normalization":"microbatch_weighted_means_rank_mean_v1",
            "autocast":"scoped_bf16_fp32_master_v2"})
    );
    assert_eq!(
        task_contract::partition_rows_per_rank(&training_partition(8, 256, 2048).unwrap()).unwrap(),
        vec![256; 8]
    );
    // The v1 exposure baseline keeps its equal-row rule.
    let state = json!({"step":2,"committed_exposure_by_rank":
        (0..2).map(|r| json!({"rank":r,"counts":counts(2 * 1024)})).collect::<Vec<_>>()});
    assert!(training_exposure::checkpoint_baseline(&state, 1, 2, 1024)
        .unwrap()
        .is_some());
    assert!(training_exposure::checkpoint_baseline(&state, 1, 2, 1000).is_err());
}
