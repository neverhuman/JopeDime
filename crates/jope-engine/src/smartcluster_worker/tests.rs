use super::execution::{
    create_immutable, read_frame, record_control_stop, scratch_usage, stop_control_with,
    within_allocation, ControlStopReason, ScratchUsage,
};
use super::*;

#[test]
fn blocked_control_diagnostic_cannot_delay_stop_or_original_watchdog() {
    use super::*;
    let stop = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&stop);
    let (began_send, began) = mpsc::channel();
    let (release, wait) = mpsc::channel();
    let (expired_send, expired) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        stop_control_with(
            &flag,
            Duration::from_millis(30),
            |deadline| {
                std::thread::spawn(move || {
                    std::thread::sleep(
                        deadline.saturating_duration_since(std::time::Instant::now()),
                    );
                    expired_send.send(()).unwrap();
                });
            },
            |_| {
                began_send.send(()).unwrap();
                wait.recv().unwrap();
                anyhow::bail!("injected diagnostic storage failure")
            },
        )
    });
    began.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(stop.load(Ordering::Acquire));
    expired.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        !worker.is_finished(),
        "diagnostic remains blocked while watchdog has already fired"
    );
    release.send(()).unwrap();
    assert!(worker.join().unwrap().is_err());
}

#[test]
fn control_stop_records_only_bounded_reason_and_preserves_original_attempt() {
    use super::*;
    for (reason, label) in [
        (ControlStopReason::EndOfFile, "end_of_file"),
        (ControlStopReason::FrameReadError, "frame_read_error"),
        (ControlStopReason::Checkpoint, "checkpoint"),
        (
            ControlStopReason::MalformedNamedControl,
            "malformed_named_control",
        ),
        (ControlStopReason::UnknownControl, "unknown_control"),
        (ControlStopReason::QueueFull, "queue_full"),
        (ControlStopReason::QueueDisconnected, "queue_disconnected"),
    ] {
        let root = tempfile::tempdir().unwrap();
        let attempt: FencedAttempt = serde_json::from_value(
            json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":9}),
        )
        .unwrap();
        let stop = AtomicBool::new(false);
        stop_control_with(
            &stop,
            Duration::from_secs(120),
            |_| {},
            |deadline| record_control_stop(root.path(), attempt, 17, reason, deadline),
        )
        .unwrap();
        assert!(stop.load(Ordering::Acquire));
        let path = root.path().join(format!(
            "control-stop-{}-{}.json",
            attempt.job_id, attempt.fence
        ));
        let bytes = fs::read(&path).unwrap();
        assert!(bytes.len() <= 4096);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["reason"], label);
        assert_eq!(value["attempt"], serde_json::to_value(attempt).unwrap());
        assert_eq!(value["sequence"], 17);
        assert_eq!(value.as_object().unwrap().len(), 7);
        assert!(stop_control_with(
            &stop,
            Duration::from_secs(120),
            |_| {},
            |deadline| record_control_stop(
                root.path(),
                attempt,
                18,
                ControlStopReason::UnknownControl,
                deadline
            )
        )
        .is_err());
        assert_eq!(
            fs::read(path).unwrap(),
            bytes,
            "original stop evidence must not be overwritten"
        );
    }
    let root = tempfile::tempdir().unwrap();
    let bad = root.path().join("regular-file");
    fs::write(&bad, b"preserved").unwrap();
    let attempt: FencedAttempt = serde_json::from_value(
        json!({"job_id":jain_smartcluster::JobId::new(),"number":1,"fence":1}),
    )
    .unwrap();
    let stop = AtomicBool::new(false);
    assert!(stop_control_with(
        &stop,
        Duration::from_secs(120),
        |_| {},
        |deadline| record_control_stop(&bad, attempt, 0, ControlStopReason::EndOfFile, deadline)
    )
    .is_err());
    assert!(
        stop.load(Ordering::Acquire),
        "failed diagnostic storage must not keep numerical admission open"
    );
}

#[test]
fn logical_dataset_retry_preserves_original_operation_and_legacy_binding() {
    let mut operation: super::Operation = serde_json::from_value(
        serde_json::json!({"kind":"PackAndPublish","manifest":"m.json","sha256":"a".repeat(64)}),
    )
    .unwrap();
    assert_eq!(operation.dataset_attempt(2).unwrap(), 2);
    let legacy = serde_json::to_vec(&operation).unwrap();
    if let super::Operation::PackAndPublish {
        logical_attempt, ..
    } = &mut operation
    {
        *logical_attempt = Some(1)
    }
    assert_eq!(operation.dataset_attempt(2).unwrap(), 1);
    assert_ne!(legacy, serde_json::to_vec(&operation).unwrap());
    if let super::Operation::PackAndPublish {
        logical_attempt, ..
    } = &mut operation
    {
        *logical_attempt = Some(0)
    }
    assert!(operation.dataset_attempt(2).is_err());
}
#[test]
fn dataset_manifest_cannot_select_an_unapproved_cpu_packer() {
    let approved = std::collections::BTreeMap::from([(
        std::path::PathBuf::from("/fixed/packer"),
        "a".repeat(64),
    )]);
    let mut request = serde_json::json!({"operation":{"kind":"pack_continuation_and_publish","packer":{"path":"/fixed/packer","sha256":"a".repeat(64)}}});
    assert!(super::execution::validate_dataset_packer(&approved, &request).is_ok());
    request["operation"]["packer"]["path"] = serde_json::json!("/tmp/unreviewed");
    assert!(super::execution::validate_dataset_packer(&approved, &request).is_err());
    request["operation"]["packer"]["path"] = serde_json::json!("/fixed/packer");
    request["operation"]["packer"]["sha256"] = serde_json::json!("b".repeat(64));
    assert!(super::execution::validate_dataset_packer(&approved, &request).is_err());
}

fn task() -> TaskManifest {
    TaskManifest {
        schema: jain_smartcluster::jope::MANIFEST_SCHEMA.into(),
        run: "xbabe2".into(),
        generation: 1,
        task_id: "train".into(),
        kind: TaskKind::ResidentGpu,
        input_manifest_sha256: "a".repeat(64),
        dependencies: BTreeMap::new(),
        parent_checkpoint_sha256: Some("b".repeat(64)),
        corpus_sha256: Some("c".repeat(64)),
        idempotency_key: jain_smartcluster::IdempotencyKey::new("train").unwrap(),
        deadline: Timestamp::from(100000),
        retry_allowance: 2,
        resources: jain_smartcluster::ResourceRequest::cpu_and_ram(1000, 1024),
        expected_receipt_id: "result".into(),
        checkpoint_allowance_ms: Some(30000),
    }
}
#[test]
fn duplicate_dispatch_preserves_bytes_and_conflict_fails() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("intent");
    let command = SubtaskCommand {
        manifest: "/task.json".into(),
        manifest_sha256: "a".repeat(64),
    };
    freeze_dispatch_intent(&path, &command).unwrap();
    let before = fs::read(&path).unwrap();
    freeze_dispatch_intent(&path, &command).unwrap();
    let changed = SubtaskCommand {
        manifest_sha256: "b".repeat(64),
        ..command
    };
    assert!(freeze_dispatch_intent(&path, &changed).is_err());
    assert_eq!(fs::read(path).unwrap(), before);
}
#[test]
fn subtask_cannot_escape_run_resources_or_deadline() {
    let parent = task();
    within_allocation(&parent, &parent).unwrap();
    let mut child = parent.clone();
    child.run = "different".into();
    assert!(within_allocation(&child, &parent).is_err());
    child = parent.clone();
    child.resources.scratch_bytes = 1;
    assert!(within_allocation(&child, &parent).is_err());
    child = parent.clone();
    child.checkpoint_allowance_ms = Some(30001);
    assert!(within_allocation(&child, &parent).is_err());
    child = parent.clone();
    child.deadline = Timestamp::from(100001);
    assert!(within_allocation(&child, &parent).is_err());
    child = parent.clone();
    child.resources.ram.hard_bytes += 1;
    assert!(within_allocation(&child, &parent).is_err());
}
#[test]
fn original_optimizer_window_is_checked_before_numerical_work() {
    let paths = ["corpus", "run", "launch", "initialize", "policy"];
    let mut input = InputManifest {
        schema: "jope.smartcluster_input.v1".into(),
        root: "/tmp".into(),
        operation: Operation::Train {
            corpus: paths[0].into(),
            run: paths[1].into(),
            launch: paths[2].into(),
            micro: 256,
            checkpoint_steps: 100,
            initialize: paths[3].into(),
            replay_policy: paths[4].into(),
            initial_optimizer_step: 60000,
            requested_step_increment: 3000,
            target_optimizer_step: 63000,
            continuation_mode: None,
        },
        closure: None,
        input_artifacts: paths
            .iter()
            .map(|p| (PathBuf::from(p), "a".repeat(64)))
            .collect(),
        output_receipt: "result".into(),
    };
    let mut allocation = task();
    allocation.resources.scratch_bytes = 1024;
    validate_operation(&allocation, &input).unwrap();
    if let Operation::Train {
        target_optimizer_step,
        ..
    } = &mut input.operation
    {
        *target_optimizer_step = 66000;
    }
    assert!(validate_operation(&task(), &input).is_err());
}
#[test]
fn training_closure_refuses_non_admitted_contexts_by_closure_key() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("corpus")).unwrap();
    let mut input = InputManifest {
        schema: "jope.smartcluster_input.v1".into(),
        root: dir.path().to_path_buf(),
        operation: Operation::Train {
            corpus: "corpus".into(),
            run: "run".into(),
            launch: "launch".into(),
            micro: 256,
            checkpoint_steps: 100,
            initialize: "initialize".into(),
            replay_policy: "policy".into(),
            initial_optimizer_step: 60000,
            requested_step_increment: 3000,
            target_optimizer_step: 63000,
            continuation_mode: None,
        },
        closure: None,
        input_artifacts: BTreeMap::new(),
        output_receipt: "result".into(),
    };
    assert!(verify_training_closure(dir.path(), &input)
        .unwrap_err()
        .to_string()
        .contains("not in dependency closure"));
    for (partition, admitted) in [("meta_validation", true), ("meta_test", false)] {
        let contexts = json!([{"partition": "meta_train"}, {"partition": partition}]);
        let index = json!({"schema": "parallel.teaching_shards.v1", "contexts": contexts});
        let index = serde_json::to_vec(&index).unwrap();
        let entry = json!({"path": "corpus/index.json", "sha256": hash(&index)});
        let mut closure = serde_json::to_vec(&entry).unwrap();
        closure.push(b'\n');
        fs::write(dir.path().join("corpus/index.json"), &index).unwrap();
        fs::write(dir.path().join("closure.jsonl"), &closure).unwrap();
        input.input_artifacts = read_closure(
            dir.path(),
            &ClosureManifest {
                path: "closure.jsonl".into(),
                sha256: hash(&closure),
            },
        )
        .unwrap();
        let result = verify_training_closure(dir.path(), &input);
        if admitted {
            result.unwrap();
        } else {
            assert!(result
                .unwrap_err()
                .to_string()
                .contains("non-admitted context partition at position 1"));
        }
    }
}
#[test]
fn unsupported_service_and_unbound_inputs_fail_closed() {
    let mut input = InputManifest {
        schema: "jope.smartcluster_input.v1".into(),
        root: "/tmp".into(),
        operation: Operation::Verify,
        closure: None,
        input_artifacts: BTreeMap::new(),
        output_receipt: "result".into(),
    };
    assert!(validate_operation(&task(), &input).is_err());
    input.operation = Operation::Proposal {
        preparation_overlay: None,
        corpus: "corpus".into(),
        checkpoint: "checkpoint".into(),
        output: "output".into(),
        cuts: 1,
        cut_start: 0,
        shuffled: false,
        champion_snapshot: "champion".into(),
    };
    assert!(validate_operation(&task(), &input).is_err());
}
#[test]
fn scratch_accounting_rejects_symlink_escape() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("bytes"), [0u8; 4096]).unwrap();
    fs::create_dir_all(dir.path().join("a/t")).unwrap();
    let usage = |root: &Path| scratch_usage(root, &root.join("a"), &root.join("a/t"));
    assert_eq!(
        usage(dir.path()).unwrap(),
        ScratchUsage {
            root: 4096,
            allocation: 0,
            task: 0
        }
    );
    std::os::unix::fs::symlink("/tmp", dir.path().join("escape")).unwrap();
    assert!(usage(dir.path()).is_err());
}
#[test]
fn scratch_accounting_sums_nested_buckets_in_one_walk_and_needs_its_root() {
    let dir = tempfile::tempdir().unwrap();
    let (root, allocation) = (dir.path(), dir.path().join("job"));
    let task = allocation.join("task");
    fs::create_dir_all(task.join("deep")).unwrap();
    fs::write(root.join("other"), [0u8; 10]).unwrap();
    fs::write(allocation.join("sibling"), [0u8; 100]).unwrap();
    fs::write(task.join("deep/file"), [0u8; 1000]).unwrap();
    // A same-prefix name next to the task directory is not inside it.
    fs::write(allocation.join("task-not"), [0u8; 7]).unwrap();
    assert_eq!(
        scratch_usage(root, &allocation, &task).unwrap(),
        ScratchUsage {
            root: 1117,
            allocation: 1107,
            task: 1000
        }
    );
    assert!(scratch_usage(&root.join("absent"), &allocation, &task).is_err());
    assert!(scratch_usage(root, &allocation, &allocation.join("absent")).is_err());
    assert!(scratch_usage(root, &root.join("absent"), &task).is_err());
}
#[test]
fn closure_fifo_is_rejected_without_waiting_for_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fifo");
    let name = std::ffi::CString::new(path.as_os_str().as_encoded_bytes()).unwrap();
    // SAFETY: `name` is a CString that outlives the call, so the path is NUL-terminated.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    let started = std::time::Instant::now();
    assert!(read_closure(
        dir.path(),
        &ClosureManifest {
            path: "fifo".into(),
            sha256: "a".repeat(64)
        }
    )
    .is_err());
    assert!(started.elapsed() < Duration::from_secs(1));
}
#[test]
fn large_streamed_closure_preserves_every_identity_and_refuses_duplicates() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("closure.jsonl");
    let mut bytes = Vec::new();
    for n in 0..12000 {
        serde_json::to_writer(&mut bytes,&json!({"path":format!("historical-observations/shards/shard-{n:08}.json"),"sha256":"a".repeat(64)})).unwrap();
        bytes.push(b'\n');
    }
    assert!(bytes.len() > FRAME_LIMIT);
    fs::write(&path, &bytes).unwrap();
    let closure = ClosureManifest {
        path: "closure.jsonl".into(),
        sha256: hash(&bytes),
    };
    assert_eq!(read_closure(dir.path(), &closure).unwrap().len(), 12000);
    bytes.extend_from_slice(b"{\"path\":\"historical-observations/shards/shard-00000000.json\",\"sha256\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}\n");
    fs::write(&path, &bytes).unwrap();
    assert!(read_closure(dir.path(), &closure).is_err());
    assert!(read_closure(
        dir.path(),
        &ClosureManifest {
            sha256: hash(&bytes),
            ..closure
        }
    )
    .is_err());
}
#[test]
fn control_descriptor_is_absent_after_child_exec() {
    let (stream, _) = UnixStream::pair().unwrap();
    let fd = stream.as_raw_fd();
    // SAFETY: `stream` stays alive for the whole test, so `fd` is open; F_SETFD takes an
    // integer flag and touches no memory.
    assert_eq!(unsafe { libc::fcntl(fd, libc::F_SETFD, 0) }, 0);
    let retained = retain_control_descriptor(fd).unwrap();
    assert!(
        // SAFETY: `retained` is a live UnixStream, so its descriptor is open; F_GETFD takes
        // no third argument and only reads the descriptor flags.
        unsafe { libc::fcntl(retained.as_raw_fd(), libc::F_GETFD) } & libc::FD_CLOEXEC != 0
    );
    let status = std::process::Command::new("sh")
        .args(["-c", "test ! -e /proc/self/fd/$1", "sh", &fd.to_string()])
        .status()
        .unwrap();
    assert!(status.success());
}
#[test]
fn atomic_record_publication_preserves_old_and_interrupted_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("record");
    let interrupted = dir.path().join(".immutable-pending-crashed");
    fs::write(&interrupted, b"partial").unwrap();
    assert!(!target.exists());
    create_immutable(&target, b"complete").unwrap();
    create_immutable(&target, b"complete").unwrap();
    assert!(create_immutable(&target, b"conflict").is_err());
    assert_eq!(fs::read(&target).unwrap(), b"complete");
    assert_eq!(fs::read(interrupted).unwrap(), b"partial");
    assert!(fs::read_dir(dir.path()).unwrap().count() >= 3);
}
#[test]
fn concurrent_immutable_publish_never_exposes_partial_result() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("record");
    let bytes = vec![42u8; FRAME_LIMIT];
    std::thread::scope(|scope| {
        for _ in 0..4 {
            let target = &target;
            let bytes = &bytes;
            scope.spawn(move || create_immutable(target, bytes).unwrap());
        }
        while !target.exists() {
            std::thread::yield_now();
        }
        assert_eq!(fs::read(&target).unwrap(), bytes);
    });
}
#[test]
fn truncated_control_frame_is_not_completion() {
    let (mut writer, mut reader) = UnixStream::pair().unwrap();
    writer.write_all(&8u32.to_be_bytes()).unwrap();
    writer.write_all(&[1, 2]).unwrap();
    drop(writer);
    assert!(read_frame(&mut reader).is_err());
}
