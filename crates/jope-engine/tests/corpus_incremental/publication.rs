use super::*;

#[test]
fn sharded_export_matches_reference_and_only_changed_groups_are_rebuilt() {
    use jope_parallel::teaching_store as store;
    let f = Fixture::new();
    let cache = f.root().join("runs/shards/store");
    let publish = |sources: &[PathBuf], name: &str| {
        store::publish(
            f.root(),
            jope_parallel::teaching_store::PublishRequest {
                contexts_path: &f.contexts(),
                vocabulary: &f.vocab(),
                sources,
                output: &f.root().join(name),
                store: &cache,
                qualification: true,
            },
            "reference_v1",
            |_, _, _| Ok(()),
        )
    };
    publish(&f.observations(), "sharded").unwrap();
    f.full("reference", true);
    store::export(
        f.root(),
        &f.root().join("sharded"),
        &f.root().join("export"),
    )
    .unwrap();
    f.equal("reference", "export");
    let index = store::read_index(f.root(), &f.root().join("sharded")).unwrap();
    let reference = store::read_index(f.root(), &f.root().join("reference")).unwrap();
    assert_eq!(
        store::teacher_programs(f.root(), &f.root().join("sharded"), &index, 10).unwrap(),
        store::teacher_programs(f.root(), &f.root().join("reference"), &reference, 10).unwrap()
    );
    publish(&f.observations(), "unchanged").unwrap();
    let first: Value = read_json(&f.root().join("sharded/index.json")).unwrap();
    let next: Value = read_json(&f.root().join("unchanged/index.json")).unwrap();
    assert_eq!(first, next);
    let timing: Value = read_json(&f.root().join("unchanged/packing-timing.json")).unwrap();
    assert_eq!(timing["rebuilt_groups"], 0);
    assert_eq!(timing["new_ledgers"], 0);
    let added = f.root().join("new/observations.json");
    let mut row: Value = read_json::<Vec<Value>>(&f.ledger("family-a"))
        .unwrap()
        .remove(0);
    let genome = "(pipe X (standardize) (rank) (ridge))";
    row["genome"] = json!(genome);
    row["program_sha256"] = json!(hash(genome.as_bytes()));
    row["metrics"]["rmse"] = json!(0.2);
    write_json(&added, &vec![row]).unwrap();
    write_json(&added.with_file_name("search.json"), &json!({})).unwrap();
    let sources = vec![f.observations()[0].clone(), added];
    publish(&sources, "changed-sharded").unwrap();
    corpus::build(
        f.root(),
        &f.contexts(),
        &f.vocab(),
        &sources,
        &f.root().join("changed-reference"),
        true,
    )
    .unwrap();
    store::export(
        f.root(),
        &f.root().join("changed-sharded"),
        &f.root().join("changed-export"),
    )
    .unwrap();
    f.equal("changed-reference", "changed-export");
    let changed: Value = read_json(&f.root().join("changed-sharded/index.json")).unwrap();
    assert_ne!(first["shards"][0], changed["shards"][0]);
    assert_eq!(first["shards"][1], changed["shards"][1]);
    assert_eq!(first["shards"][2], changed["shards"][2]);
    let timing: Value = read_json(&f.root().join("changed-sharded/packing-timing.json")).unwrap();
    assert_eq!(timing["rebuilt_groups"], 1);
    assert!(timing["duplicate_program_cut"].as_u64().is_some());
    assert!(timing["new_teacher_beats"].as_u64().is_some());
    // Old readers retain exactly the old immutable groups.
    store::load(f.root(), &f.root().join("sharded")).unwrap();
}

#[test]
fn frozen_z_keeps_admitted_percentiles_when_a_new_teacher_arrives() {
    use jope_parallel::teaching_store as store;
    let f = Fixture::new();
    let cache = f.root().join("runs/frozen/store");
    store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &f.root().join("frozen-a"),
            store: &cache,
            qualification: true,
        },
        "frozen_z_v1",
        |_, _, _| Ok(()),
    )
    .unwrap();
    let (_, first) = store::load(f.root(), &f.root().join("frozen-a")).unwrap();
    let old_value = first.get(0).value;
    let old_percentile = first.get(0).percentile;
    let added = f.root().join("frozen-new/observations.json");
    let mut row: Value = read_json::<Vec<Value>>(&f.ledger("family-a"))
        .unwrap()
        .remove(0);
    let genome = "(pipe X (standardize) (rank) (ridge))";
    row["genome"] = json!(genome);
    row["program_sha256"] = json!(hash(genome.as_bytes()));
    row["metrics"]["rmse"] = json!(0.05);
    write_json(&added, &vec![row]).unwrap();
    write_json(&added.with_file_name("search.json"), &json!({})).unwrap();
    store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &[f.observations()[0].clone(), added],
            output: &f.root().join("frozen-b"),
            store: &cache,
            qualification: true,
        },
        "frozen_z_v1",
        |_, _, _| Ok(()),
    )
    .unwrap();
    let (_, next) = store::load(f.root(), &f.root().join("frozen-b")).unwrap();
    assert_eq!(next.get(1).value, old_value);
    assert_eq!(next.get(1).percentile, old_percentile);
    let timing: Value = read_json(&f.root().join("frozen-b/packing-timing.json")).unwrap();
    assert!(timing["new_teacher_beats"].as_u64().unwrap() >= 1);
    let provenance: Value = read_json(&f.root().join("frozen-b/provenance.json")).unwrap();
    assert_eq!(provenance["label_policy"], "frozen_z_v1");
    assert!(provenance["duplicate_program_cut"].is_number());
}

#[test]
fn sharded_interruption_tampering_and_source_order_are_enforced() {
    use jope_parallel::teaching_store as store;
    for boundary in [
        "search ledger committed",
        "teaching group committed",
        "manifest durable before publication",
    ] {
        let f = Fixture::new();
        let cache = f.root().join("runs/shards/store");
        let output = f.root().join("sharded");
        assert!(store::publish(
            f.root(),
            jope_parallel::teaching_store::PublishRequest {
                contexts_path: &f.contexts(),
                vocabulary: &f.vocab(),
                sources: &f.observations(),
                output: &output,
                store: &cache,
                qualification: true
            },
            "reference_v1",
            |op, _, _| {
                if op == boundary {
                    anyhow::bail!("injected interruption");
                }
                Ok(())
            },
        )
        .is_err());
        assert!(!output.exists());
        store::publish(
            f.root(),
            jope_parallel::teaching_store::PublishRequest {
                contexts_path: &f.contexts(),
                vocabulary: &f.vocab(),
                sources: &f.observations(),
                output: &output,
                store: &cache,
                qualification: true,
            },
            "reference_v1",
            |_, _, _| Ok(()),
        )
        .unwrap();
        f.full("reference", true);
        store::export(f.root(), &output, &f.root().join("export")).unwrap();
        f.equal("reference", "export");
        let manifest: Value = read_json(&output.join("index.json")).unwrap();
        let shard = f
            .root()
            .join(manifest["shards"][0]["path"].as_str().unwrap())
            .join("records.bin");
        use std::io::Write;
        fs::OpenOptions::new()
            .write(true)
            .open(&shard)
            .unwrap()
            .write_all(b"bad")
            .unwrap();
        assert!(store::load(f.root(), &output).is_err());
    }
    let f = Fixture::new();
    let cache = f.root().join("runs/shards/store");
    store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &f.root().join("one"),
            store: &cache,
            qualification: true,
        },
        "reference_v1",
        |_, _, _| Ok(()),
    )
    .unwrap();
    fs::write(f.ledger("family-a"), b"[]").unwrap();
    assert!(store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &f.root().join("two"),
            store: &cache,
            qualification: true
        },
        "reference_v1",
        |_, _, _| Ok(()),
    )
    .is_err());
}

#[test]
fn versioned_midranks_use_original_precision_and_preserve_observations() {
    use jope_parallel::teaching_store as store;
    assert_eq!(
        store::midranks(&[1., 1., 2., 3., 3.]),
        vec![0.875, 0.875, 0.5, 0.125, 0.125]
    );
    assert_eq!(store::midranks(&[1., 1., 1.]), vec![0.5; 3]);
    // Distinct f64 scores that round to the same packed f32 remain distinct.
    assert_eq!(store::midranks(&[1., 1. + f64::EPSILON]), vec![1., 0.]);
    let f = Fixture::new();
    for family in ["family-a", "family-b", "family-c"] {
        let path = f.ledger(family);
        let mut rows: Vec<Value> = read_json(&path).unwrap();
        rows[1]["metrics"]["rmse"] = rows[0]["metrics"]["rmse"].clone();
        write_json(&path, &rows).unwrap();
    }
    f.full("reference", true);
    let source_sha = file_hash(&f.ledger("family-a")).unwrap();
    store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &f.root().join("midrank"),
            store: &f.root().join("runs/midrank/store"),
            qualification: true,
        },
        "midrank_v2",
        |_, _, _| Ok(()),
    )
    .unwrap();
    let (_, records) = store::load(f.root(), &f.root().join("midrank")).unwrap();
    assert_eq!(records.get(0).percentile, records.get(1).percentile);
    store::export(
        f.root(),
        &f.root().join("midrank"),
        &f.root().join("midrank-export"),
    )
    .unwrap();
    assert_eq!(
        fs::read(f.root().join("reference/labels.jsonl")).unwrap(),
        fs::read(f.root().join("midrank-export/labels.jsonl")).unwrap()
    );
    assert_eq!(source_sha, file_hash(&f.ledger("family-a")).unwrap());
}

#[test]
fn verified_records_remain_immutable_when_source_is_tampered_or_truncated() {
    let f = Fixture::new();
    f.full("reference", true);
    let path = f.root().join("reference/records.bin");
    let digest = file_hash(&path).unwrap();
    let records = corpus::Records::open_verified(&path, 24, &digest).unwrap();
    let original = records.get(0).bytes();
    fs::write(&path, b"tampered").unwrap();
    assert_eq!(records.get(0).bytes(), original);
    assert!(corpus::Records::open_verified(&path, 24, &digest).is_err());
}

#[test]
fn strict_completion_rejects_files_without_original_exit_and_preserves_unknown_runtime() {
    use jope_parallel::label_admission as admission;
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();
    let base = json!({"genome":"(pipe X (ridge))","program_sha256":hash(b"(pipe X (ridge))"),"evaluator_sha256":hash(b"evaluator"),
        "evaluation_identity_sha256":hash(b"evaluation"),"training_sha256":hash(b"train"),
        "failure":null,"metrics":{"rmse":0.,"units":"cut_normalized_target"},"runtime_seconds":0.1});
    let mut row = base.clone();
    attach_completion(root, root, 0, &mut row);
    admission::validate_completion(root, &row).unwrap();
    for (field, value) in [
        ("exit_observed", json!(false)),
        ("exit_code", json!(1)),
        ("signal", json!(9)),
        ("deadline_exceeded", json!(true)),
        ("deadline_seconds", json!(0.01)),
        ("runtime_seconds", Value::Null),
        ("outcome", json!("timeout")),
        ("fit_completion", Value::Null),
    ] {
        let mut modified = row.clone();
        let mut completion = admission::artifact(root, &row["clean_completion"]).unwrap();
        completion[field] = value;
        let path = root.join(format!("bad-{field}.json"));
        write_json(&path, &completion).unwrap();
        modified["clean_completion"] =
            json!({"path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap()});
        assert!(
            admission::validate_completion(root, &modified).is_err(),
            "{field}"
        );
    }
    let mut missing = row.clone();
    missing.as_object_mut().unwrap().remove("clean_completion");
    assert!(admission::validate_completion(root, &missing).is_err());
    let mut stale = row.clone();
    stale["metrics"]["rmse"] = json!(99.);
    assert!(admission::validate_completion(root, &stale).is_err());
    fs::write(root.join("result-0.json"), b"{\"rmse\":1}").unwrap();
    assert!(admission::validate_completion(root, &row).is_err());

    let mut unknown = base;
    unknown["failure"] = json!("unknown original execution");
    unknown["metrics"] = Value::Null;
    unknown["runtime_seconds"] = Value::Null;
    attach_completion(root, root, 1, &mut unknown);
    let mut completion = admission::artifact(root, &unknown["clean_completion"]).unwrap();
    let mut native = admission::artifact(root, &completion["fit_completion"]).unwrap();
    native["outcome"] = json!("unknown");
    native["observed_exit_code"] = Value::Null;
    native
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    native["clean_completion_sha256"] = json!(hash(&serde_json::to_vec(&native).unwrap()));
    write_json(&root.join("native-1.json"), &native).unwrap();
    completion["fit_completion"]["sha256"] = json!(file_hash(&root.join("native-1.json")).unwrap());
    completion["outcome"] = json!("unknown");
    completion["exit_observed"] = json!(false);
    completion["exit_code"] = Value::Null;
    write_json(&root.join("completion-1.json"), &completion).unwrap();
    unknown["clean_completion"]["sha256"] =
        json!(file_hash(&root.join("completion-1.json")).unwrap());
    admission::validate_completion(root, &unknown).unwrap();
    assert!(unknown["runtime_seconds"].is_null());
}

#[test]
fn legacy_publication_cannot_bypass_production_admission() {
    let f = Fixture::new();
    assert!(jope_parallel::teaching_store::publish(
        f.root(),
        jope_parallel::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &f.root().join("must-not-publish"),
            store: &f.root().join("runs/new-store"),
            qualification: false
        },
        "reference_v1",
        |_, _, _| Ok(()),
    )
    .is_err());
    assert!(!f.root().join("runs/new-store").exists());
    assert!(f.cached("must-not-build", false).is_err());
    assert!(!f.cache().exists());
}
