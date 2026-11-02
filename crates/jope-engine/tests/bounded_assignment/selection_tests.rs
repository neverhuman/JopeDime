use super::*;

/// W7-T06: a plan's training selection is admitted exactly when the trainer's partition gate
/// accepts it for the plan's world size, and the plan derives exactly the record the trainer
/// declares. The contract mirrors the gate without the neural build; this pins the two.
#[test]
fn a_plan_selects_exactly_the_partitions_the_trainer_runs() {
    use jope_parallel::{
        native_loop::bounded::contract::TrainingSelection,
        task_contract::{PARTITION_V1_SCHEMA, PARTITION_V2_SCHEMA},
        training::partition::{execution, Execution, Version},
    };
    let (mut admitted, mut refused) = (0, 0);
    for world in 1..=8usize {
        let base = with_training_inputs(contract(now_ms(), world, &keys(world)));
        for (schema, version) in [
            (PARTITION_V1_SCHEMA, Version::V1),
            (PARTITION_V2_SCHEMA, Version::V2),
        ] {
            for micro in [8u32, 64, 128, 256, 512] {
                for global in [1024u32, 2048, 3000, 4096, 8192, 16384] {
                    let mut c = base.clone();
                    c.training = Some(TrainingSelection {
                        execution_partition: schema.into(),
                        microbatch: micro,
                        global_batch: global,
                        ..selection()
                    });
                    let gate = execution(version, world, micro as usize, global as usize, false);
                    match c.validate_structure() {
                        Ok(()) => {
                            admitted += 1;
                            let record = c.execution_partition().unwrap().unwrap();
                            let gate = gate.unwrap_or_else(|e| {
                                panic!("{schema} world {world} {micro}/{global}: {e}")
                            });
                            assert_eq!(*gate.record(), record);
                            let declared = Execution::declared(
                                &record,
                                world,
                                micro as usize,
                                global as usize,
                                false,
                            )
                            .unwrap();
                            assert_eq!(declared.version(), version);
                            assert_eq!(
                                declared.rows_per_rank().iter().sum::<usize>(),
                                global as usize
                            );
                        }
                        Err(e) => {
                            refused += 1;
                            assert!(
                                gate.is_err(),
                                "{schema} world {world} {micro}/{global} refused by the plan \
                                 but run by the trainer: {e}"
                            );
                        }
                    }
                }
            }
        }
    }
    assert!(
        admitted > 0 && refused > 0,
        "{admitted} admitted, {refused} refused"
    );
    // Three ranks train only under v2; its rows are lossless and uneven.
    let mut three = with_training_inputs(contract(now_ms(), 3, &keys(3)));
    three.training = Some(TrainingSelection {
        execution_partition: PARTITION_V2_SCHEMA.into(),
        ..selection()
    });
    three.validate_structure().unwrap();
    let record = three.execution_partition().unwrap().unwrap();
    assert_eq!(
        Execution::declared(&record, 3, 256, 2048, false)
            .unwrap()
            .rows_per_rank(),
        vec![688, 680, 680]
    );
}

const CORPUS_INDEX: &[u8] = br#"{"fixture":"corpus index"}"#;
/// The fixture counters start at 60 000 committed updates: the parent is at that step.
const PARENT_STATE: &[u8] = br#"{"fixture":"parent","step":60000}"#;
/// `c` with the corpus index and parent checkpoint state a training selection pins.
fn with_training_inputs(mut c: BoundedRunContract) -> BoundedRunContract {
    use jope_parallel::native_loop::bounded::contract::ClosureEntry;
    for (path, bytes) in [
        ("corpus/index.json", CORPUS_INDEX),
        ("parent/state.json", PARENT_STATE),
    ] {
        c.artifact_closure.push(ClosureEntry {
            path: path.into(),
            sha256: sha(bytes),
            bytes: bytes.len() as u64,
        });
    }
    c.artifact_closure.sort_by(|a, b| a.path.cmp(&b.path));
    c
}
fn selection() -> jope_parallel::native_loop::bounded::contract::TrainingSelection {
    jope_parallel::native_loop::bounded::contract::TrainingSelection {
        execution_partition: jope_parallel::task_contract::PARTITION_V2_SCHEMA.into(),
        microbatch: 256,
        global_batch: 2048,
        corpus: "corpus".into(),
        checkpoint_steps: 1,
        initialize: Some("parent".into()),
    }
}

/// W7-D2, up to the device boundary: a bounded rank's prepared training run enters the
/// trainer with its plan's partition, its plan's assignment and only the admitted bootstrap.
/// A shared `nccl.id` in its launch directory is refused before any native call.
#[test]
fn a_bounded_training_rank_takes_only_the_admitted_bootstrap_into_the_trainer() {
    use jope_parallel::{
        native_loop::bounded::training_run,
        training::partition::{Execution, Version},
    };
    let k = keys(3);
    let (mut c, _) = plan(&[0, 1, 2]);
    c = with_training_inputs(c);
    c.training = Some(selection());
    resign(&mut c, &k);
    c.validate_structure().unwrap();
    let plan_sha = sha(&bytes(&c));
    let (_, _, _, members) = admit_group(&c, &k, Some(hex::encode([7u8; 128])));
    let (_t, root) = tempdir();
    write_closure(&root);
    for (path, content) in [
        ("corpus/index.json", CORPUS_INDEX),
        ("parent/state.json", PARENT_STATE),
    ] {
        std::fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
        std::fs::write(root.join(path), content).unwrap();
    }
    let assignment = trainer_assignment(&c).unwrap();
    for rank in 0..3u32 {
        let record = launch::launch_record(&c, &plan_sha, rank).unwrap();
        let bootstrap =
            launch::nccl_bootstrap(&c, &plan_sha, &record, &members[rank as usize]).unwrap();
        let prepared = training_run::prepare(&c, &plan_sha, &record, &bootstrap, &root).unwrap();
        // The trainer's gate accepts exactly the partition the plan selected.
        let declared = Execution::declared(
            &prepared.spec["execution_partition"],
            prepared.world,
            prepared.microbatch,
            prepared.global_batch,
            false,
        )
        .unwrap();
        assert_eq!(declared.version(), Version::V2);
        assert_eq!(declared.rows_per_rank(), vec![688, 680, 680]);
        assert_eq!(assignment.plan_sha256, plan_sha);
        // Only the admitted bootstrap, never a shared file, feeds the communicator.
        assert_eq!(
            communicator_source(
                Some(&assignment),
                Some(&bootstrap),
                rank as usize,
                &prepared.launch_dir
            )
            .unwrap(),
            CommunicatorSource::Admitted
        );
        let e = communicator_source(Some(&assignment), None, rank as usize, &prepared.launch_dir)
            .unwrap_err();
        assert!(
            e.to_string()
                .contains("the shared nccl.id rendezvous is refused"),
            "{e}"
        );
        std::fs::write(prepared.launch_dir.join("nccl.id"), [0u8; 128]).unwrap();
        let e = communicator_source(
            Some(&assignment),
            Some(&bootstrap),
            rank as usize,
            &prepared.launch_dir,
        )
        .unwrap_err();
        assert!(e.to_string().contains("holds a shared nccl.id"), "{e}");
        // Another rank's bootstrap never enters this rank's trainer.
        if rank > 0 {
            let other = launch::launch_record(&c, &plan_sha, 0).unwrap();
            let other = launch::nccl_bootstrap(&c, &plan_sha, &other, &members[0]).unwrap();
            assert!(communicator_source(
                Some(&assignment),
                Some(&other),
                rank as usize,
                &root.join("fresh")
            )
            .is_err());
        }
    }
}
