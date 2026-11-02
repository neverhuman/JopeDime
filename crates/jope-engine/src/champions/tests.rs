use super::*;
fn observation(rmse: f64) -> Value {
    let genome = "(pipe X (ridge))";
    json!({"dataset":"d","family":"f","partition":"meta_train","cut":0,
        "prepared_sha256":"1".repeat(64),"training_sha256":"2".repeat(64),
        "validation_sha256":"3".repeat(64),"evaluator_sha256":"4".repeat(64),
        "vocabulary_sha256":"5".repeat(64),"genome":genome,"program_sha256":hash(genome.as_bytes()),
        "metrics":{"rmse":rmse},"target_scale":2.0,"failure":null,
        "evaluation_bias":"adaptive_cut_heldout_teacher","independent_evaluation":false})
}
#[test]
fn recovered_link_before_unlink_restores_single_link_winner_and_keeps_partial() {
    use std::os::unix::fs::MetadataExt;
    let tmp = tempfile::tempdir().unwrap();
    record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
    let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
    let directory = tmp
        .path()
        .join("runs/champions")
        .join(identity.key().unwrap());
    let event = fs::read_dir(&directory)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("event-")
        })
        .unwrap();
    let committed = fs::read(&event).unwrap();
    let alias = directory.join(".pending-crashed-link");
    fs::hard_link(&event, &alias).unwrap();
    let partial = directory.join(".pending-unrelated-partial");
    fs::write(&partial, b"partial").unwrap();
    assert_eq!(fs::metadata(&event).unwrap().nlink(), 2);
    recover_outbox(tmp.path()).unwrap();
    assert_eq!(fs::metadata(&event).unwrap().nlink(), 1);
    assert!(!alias.exists());
    assert_eq!(fs::read(&partial).unwrap(), b"partial");
    assert_eq!(fs::read(&event).unwrap(), committed);
}
#[test]
fn only_improvements_replace_exact_identity_and_preserve_evidence() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let first = observation(3.0);
    assert!(record(root, &first, Path::new("run1/observations.json")).unwrap());
    let identity = Identity::observation(&first).unwrap().unwrap();
    let dir = root.join("runs/champions").join(identity.key().unwrap());
    let original = fs::read(dir.join("current.json")).unwrap();
    assert!(!record(
        root,
        &observation(4.0),
        Path::new("worse/observations.json")
    )
    .unwrap());
    assert_eq!(original, fs::read(dir.join("current.json")).unwrap());
    assert!(record(
        root,
        &observation(1.0),
        Path::new("better/observations.json")
    )
    .unwrap());
    let best: Champion = read_json(&dir.join("current.json")).unwrap();
    best.validate().unwrap();
    assert_eq!(best.rmse_original_units, 2.0);
    assert_eq!(best.previous_rmse_original_units, Some(6.0));
    let previous: Champion = read_json(&dir.join(format!(
        "event-{}.json",
        best.previous_event_sha256.unwrap()
    )))
    .unwrap();
    assert_eq!(previous.observation, first);
}
#[test]
fn interrupted_winner_is_published_before_a_later_improvement() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    record(root, &observation(3.0), Path::new("first")).unwrap();
    let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
    let dir = root.join("runs/champions").join(identity.key().unwrap());
    let first: Champion = read_json(&dir.join("current.json")).unwrap();
    let mut intermediate = first.clone();
    intermediate.observation = observation(2.0);
    intermediate.observation_sha256 = hash(&serde_json::to_vec(&intermediate.observation).unwrap());
    intermediate.rmse_original_units = 4.0;
    intermediate.previous_event_sha256 = Some(hash(&serde_json::to_vec(&first).unwrap()));
    intermediate.previous_rmse_original_units = Some(6.0);
    intermediate.improvement_number = 1;
    let pending = PendingChampion {
        champion: intermediate,
        publish: true,
    };
    immutable_bytes(
        &dir.join("pending.json"),
        &serde_json::to_vec(&pending).unwrap(),
    )
    .unwrap();
    // This is the persisted state after intent fsync and before event/current.
    record(root, &observation(1.0), Path::new("latest")).unwrap();
    let mut winners = fs::read_dir(root.join("runs/champions-outbox"))
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with(|c: char| c.is_ascii_digit())
        })
        .collect::<Vec<_>>();
    winners.sort();
    assert_eq!(winners.len(), 3);
    for (sequence, (path, rmse)) in winners.iter().zip([6.0, 4.0, 2.0]).enumerate() {
        let bytes = fs::read(path).unwrap();
        assert_eq!(
            path.file_name().unwrap().to_string_lossy(),
            format!("{sequence:020}-{}.json", hash(&bytes))
        );
        assert_eq!(
            serde_json::from_slice::<Champion>(&bytes)
                .unwrap()
                .rmse_original_units,
            rmse
        );
    }
}

#[test]
fn outbox_replays_published_bytes_without_reallocating_sequence() {
    let tmp = tempfile::tempdir().unwrap();
    record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
    let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
    let champion: Champion = read_json(
        &tmp.path()
            .join("runs/champions")
            .join(identity.key().unwrap())
            .join("current.json"),
    )
    .unwrap();
    let outbox = tmp.path().join("runs/champions-outbox");
    let digest = hash(&serde_json::to_vec(&champion).unwrap());
    fs::remove_file(outbox.join("by-event").join(format!("{digest}.json"))).unwrap();
    fs::remove_file(outbox.join("next.json")).unwrap();
    immutable_bytes(
        &outbox.join("pending.json"),
        &serde_json::to_vec(&PendingPublication {
            sequence: 0,
            champion: champion.clone(),
        })
        .unwrap(),
    )
    .unwrap();
    // Crash after publishing the event, before derived index/counter commit.
    publish_winner(tmp.path(), &champion).unwrap();
    assert_eq!(read_json::<u64>(&outbox.join("next.json")).unwrap(), 1);
    assert!(!outbox.join("pending.json").exists());
    assert_eq!(
        read_json::<u64>(&outbox.join("by-event").join(format!("{digest}.json"))).unwrap(),
        0
    );
    fs::write(outbox.join(format!("{:020}-{digest}.json", 0)), b"corrupt").unwrap();
    assert!(publish_winner(tmp.path(), &champion).is_err());
}
#[test]
fn corrupt_counter_cannot_assign_a_second_event_to_an_existing_sequence() {
    let tmp = tempfile::tempdir().unwrap();
    record(tmp.path(), &observation(3.0), Path::new("first")).unwrap();
    let outbox = tmp.path().join("runs/champions-outbox");
    let original = fs::read(outbox.join("by-sequence/00000000000000000000.json")).unwrap();
    write_json(&outbox.join("next.json"), &0u64).unwrap();
    assert!(record(tmp.path(), &observation(2.0), Path::new("second")).is_err());
    assert_eq!(
        fs::read(outbox.join("by-sequence/00000000000000000000.json")).unwrap(),
        original
    );
    assert_eq!(
        fs::read_dir(&outbox)
            .unwrap()
            .filter(|entry| entry
                .as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(|c: char| c.is_ascii_digit()))
            .count(),
        1
    );
}

#[test]
fn historical_origin_persists_and_sealed_rows_leave_no_live_pending_work() {
    let tmp = tempfile::tempdir().unwrap();
    let row = observation(3.0);
    record_inner(tmp.path(), &row, Path::new("historical"), false).unwrap();
    let identity = Identity::observation(&row).unwrap().unwrap();
    let directory = tmp
        .path()
        .join("runs/champions")
        .join(identity.key().unwrap());
    let champion: Champion = read_json(&directory.join("current.json")).unwrap();
    let event = hash(&serde_json::to_vec(&champion).unwrap());
    let origin: Value =
        read_json(&directory.join("origins").join(format!("{event}.json"))).unwrap();
    assert_eq!(origin["origin"], "historical_import");
    assert_eq!(champion.observation, row);
    assert!(!tmp.path().join("runs/champions-outbox").exists());
    let mut sealed = row;
    sealed["partition"] = json!("meta_test");
    assert!(record(tmp.path(), &sealed, Path::new("sealed")).is_err());
    assert!(record_inner(tmp.path(), &sealed, Path::new("sealed"), false).is_err());
    assert_eq!(recover_outbox(tmp.path()).unwrap(), 0);
    assert!(!tmp.path().join("runs/champions-outbox").exists());
}
#[test]
fn different_heldout_or_evaluator_cannot_replace_an_incumbent() {
    let tmp = tempfile::tempdir().unwrap();
    let first = observation(3.0);
    record(tmp.path(), &first, Path::new("original")).unwrap();
    for field in [
        "validation_sha256",
        "training_sha256",
        "prepared_sha256",
        "evaluator_sha256",
        "vocabulary_sha256",
    ] {
        let mut other = observation(0.0);
        other[field] = json!("a".repeat(64));
        assert!(record(tmp.path(), &other, Path::new("other")).unwrap());
    }
    let id = Identity::observation(&first).unwrap().unwrap();
    let saved: Champion = read_json(
        &tmp.path()
            .join("runs/champions")
            .join(id.key().unwrap())
            .join("current.json"),
    )
    .unwrap();
    assert_eq!(saved.rmse_original_units, 6.0);
    assert_eq!(saved.improvement_number, 0);
}
#[test]
fn changed_program_or_score_bias_is_never_admitted() {
    let tmp = tempfile::tempdir().unwrap();
    let mut o = observation(1.0);
    o["program_sha256"] = json!("0".repeat(64));
    assert!(record(tmp.path(), &o, Path::new("x")).is_err());
    o = observation(1.0);
    o["evaluation_bias"] = json!("independent_test");
    assert!(!record(tmp.path(), &o, Path::new("x")).unwrap());
}
#[test]
fn partial_ledgers_are_not_imported() {
    let tmp = tempfile::tempdir().unwrap();
    let ledger = tmp.path().join("runs/incomplete/observations.json");
    write_json(&ledger, &vec![observation(1.0)]).unwrap();
    let report = import(tmp.path(), &[PathBuf::from("runs/incomplete")]).unwrap();
    assert_eq!(report["identities"], 0);
    assert!(!tmp.path().join("runs/champions").exists());
}
#[test]
fn both_arms_freeze_same_programs_and_versions_never_share_scores() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    for path in [
        "runs",
        "corpus",
        "checkpoint",
        "cycle/neural",
        "cycle/shuffled",
        "prepared",
    ] {
        fs::create_dir_all(root.join(path)).unwrap();
    }
    fs::write(root.join("evaluator"), b"new evaluator").unwrap();
    fs::write(root.join("corpus/vocabulary.json"), b"{}").unwrap();
    fs::write(root.join("checkpoint/state.json"), b"checkpoint").unwrap();
    let vocab = file_hash(&root.join("corpus/vocabulary.json")).unwrap();
    let evaluator = file_hash(&root.join("evaluator")).unwrap();
    let prepared = json!({"schema":"parallel.prepared.v1","dataset":"d","family":"f","task":"regression",
        "meta_partition":"meta_train","seed":SEED,"family_partition_frozen":false,"clean_evaluation":true,
        "clean_status":"raw_reconstructed","original_train_sha256":"0","original_test_sha256":"1",
        "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
        "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"2".repeat(64),
            "validation_sha256":"3".repeat(64),"preprocessing_sha256":"6".repeat(64)}]});
    write_json(&root.join("prepared/prepared.json"), &prepared).unwrap();
    let prepared_sha = file_hash(&root.join("prepared/prepared.json")).unwrap();
    let mut old = observation(1.0);
    old["prepared_sha256"] = json!(prepared_sha);
    old["vocabulary_sha256"] = json!(vocab);
    record(root, &old, Path::new("old/observations.json")).unwrap();
    let mut current = old.clone();
    current["evaluator_sha256"] = json!(evaluator);
    current["metrics"]["rmse"] = json!(100.0);
    record(root, &current, Path::new("new/observations.json")).unwrap();
    write_json(
        &root.join("runs/gp-service.json"),
        &json!({"executable":"evaluator","executable_sha256":evaluator,
        "vocabulary_sha256":vocab,"timestamp":crate::telemetry::now()}),
    )
    .unwrap();
    write_json(&root.join("corpus/index.json"),&json!({"schema":"parallel.teaching_corpus.v1","qualification":true,
        "vocabulary_sha256":vocab,"context_index_sha256":"x","records_sha256":"x","record_bytes":0,"records":0,
        "legal_sets":[],"groups":[],"contexts":[{"dataset":"d","family":"f","cut":0,"partition":"meta_train",
            "path":"x","sha256":"x","prepared":"prepared","prepared_sha256":prepared_sha}]})).unwrap();
    let a = freeze_snapshot(
        root,
        &root.join("corpus"),
        &root.join("checkpoint"),
        &root.join("cycle/neural"),
        2,
        0,
    )
    .unwrap();
    let frozen = fs::read(&a).unwrap();
    let s: Snapshot = read_json(&a).unwrap();
    assert_eq!(s.entries.len(), 2);
    assert_eq!(s.entries[0].rmse_original_units, 200.0);
    assert_eq!(s.entries[1].rmse_original_units, 2.0);
    old["metrics"]["rmse"] = json!(0.1);
    record(root, &old, Path::new("improved/observations.json")).unwrap();
    let b = freeze_snapshot(
        root,
        &root.join("corpus"),
        &root.join("checkpoint"),
        &root.join("cycle/shuffled"),
        2,
        0,
    )
    .unwrap();
    assert_eq!(a, b);
    assert_eq!(frozen, fs::read(b).unwrap());
    assert!(freeze_snapshot(
        root,
        &root.join("corpus"),
        &root.join("checkpoint"),
        &root.join("cycle/shuffled"),
        3,
        0
    )
    .is_err());
}
#[test]
fn recovered_sealed_and_unknown_pending_never_publish_or_materialize() {
    let donor = tempfile::tempdir().unwrap();
    record(donor.path(), &observation(3.0), Path::new("donor")).unwrap();
    let identity = Identity::observation(&observation(3.0)).unwrap().unwrap();
    let original: Champion = read_json(
        &donor
            .path()
            .join("runs/champions")
            .join(identity.key().unwrap())
            .join("current.json"),
    )
    .unwrap();
    for partition in ["meta_test", "unknown"] {
        let mut champion = original.clone();
        champion.observation["partition"] = json!(partition);
        champion.identity.partition = partition.into();
        champion.observation_sha256 = hash(&serde_json::to_vec(&champion.observation).unwrap());
        champion.validate().unwrap();
        let root = tempfile::tempdir().unwrap();
        let directory = root
            .path()
            .join("runs/champions")
            .join(champion.identity.key().unwrap());
        fs::create_dir_all(&directory).unwrap();
        let pending = PendingChampion {
            champion: champion.clone(),
            publish: true,
        };
        let bytes = serde_json::to_vec(&pending).unwrap();
        fs::write(directory.join("pending.json"), &bytes).unwrap();
        assert!(recover_outbox(root.path()).is_err());
        assert_eq!(fs::read(directory.join("pending.json")).unwrap(), bytes);
        assert!(!directory.join("origins").exists());
        assert!(!directory.join("current.json").exists());
        assert!(!root.path().join("runs/champions-outbox").exists());
        assert!(!fs::read_dir(&directory).unwrap().any(|e| e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with("event-")));
        let outbox = root.path().join("standalone-outbox");
        fs::create_dir(&outbox).unwrap();
        let pending = PendingPublication {
            sequence: 0,
            champion,
        };
        let bytes = serde_json::to_vec(&pending).unwrap();
        fs::write(outbox.join("pending.json"), &bytes).unwrap();
        assert!(finish_publication(&outbox, &pending).is_err());
        assert_eq!(fs::read(outbox.join("pending.json")).unwrap(), bytes);
        assert_eq!(fs::read_dir(&outbox).unwrap().count(), 1);
    }
}
