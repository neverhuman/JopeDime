use super::*;

#[test]
fn native_strict_admission_quarantines_conflicts_before_dedup_and_keeps_outcomes() {
    use jope_parallel::{label_admission as admission, teaching_store as store};
    let f = Fixture::new();
    fs::copy(std::env::current_exe().unwrap(), f.root().join("evaluator")).unwrap();
    let evaluator = file_hash(&f.root().join("evaluator")).unwrap();
    let mut contexts: Value = read_json(&f.contexts()).unwrap();
    for (i, family) in ["family-a", "family-b", "family-c"].iter().enumerate() {
        let pp = f.root().join(family).join("cut-0/preprocessing.json");
        write_json(&pp,&json!({"schema":"parallel.preprocessing.v1","columns":[],"labels":[],"target_mean":0.,"target_scale":2.})).unwrap();
        let pp_sha = file_hash(&pp).unwrap();
        let prepared = f.root().join(family).join("prepared.json");
        let bytes = jope_parallel::prepare::csv_bytes(&[vec![1., 2.]]).unwrap();
        fs::write(pp.with_file_name("train.csv"), &bytes).unwrap();
        fs::write(pp.with_file_name("validation.csv"), &bytes).unwrap();
        let mut p: Value = read_json(&prepared).unwrap();
        p["cuts"][0]["preprocessing_sha256"] = json!(pp_sha);
        p["cuts"][0]["training_sha256"] = json!(hash(&bytes));
        p["cuts"][0]["validation_sha256"] = json!(hash(&bytes));
        write_json(&prepared, &p).unwrap();
        let psha = file_hash(&prepared).unwrap();
        contexts["encodings"][i]["prepared_sha256"] = json!(psha);
        let encoding = f.root().join(family).join("context.json");
        let mut e: Value = read_json(&encoding).unwrap();
        e["prepared_sha256"] = json!(psha);
        write_json(&encoding, &e).unwrap();
        let path = f.ledger(family);
        let mut rows: Vec<Value> = read_json(&path).unwrap();
        for transform in ["standardize", "robust_scale"] {
            let mut row = rows[0].clone();
            let program = format!("(pipe X ({transform}) (rank) (ridge))");
            row["genome"] = json!(program);
            row["program_sha256"] = json!(hash(program.as_bytes()));
            rows.push(row);
        }
        for row in &mut rows {
            row["evaluation_origin"] = json!("execution");
            row["metrics"]["evaluation_origin"] = json!("execution");
            row["task"] = json!("regression");
            row["prepared_sha256"] = json!(psha);
            row["evaluator_sha256"] = json!(evaluator);
            row["training_sha256"] = json!(hash(&bytes));
            row["validation_sha256"] = json!(hash(&bytes));
            row["evaluator_protocol"] = json!(admission::PROTOCOL);
            row["resource_protocol"] = json!(admission::RESOURCE);
            row["preprocessing_sha256"] = json!(pp_sha);
            row["metric_units"] = json!("cut_normalized_target");
            row["metrics"]["units"] = json!("cut_normalized_target");
            row["target_scale"] = json!(2.);
            row["vocabulary_sha256"] = json!(file_hash(&f.vocab()).unwrap());
            row["evaluation_seed"] = json!(0);
            row["feature_view_sha256"] = json!(admission::raw_view(&hash(&bytes), &pp_sha));
            row["evaluation_identity_sha256"] = json!(hash(
                &serde_json::to_vec(&admission::identity(row)).unwrap()
            ));
        }
        write_json(&path, &rows).unwrap();
        f.strict_receipt(&path, &rows[0]);
    }
    write_json(&f.contexts(), &contexts).unwrap();
    let expected = f.root().join("admission.json");
    let authority: Value = read_json(&f.root().join("test-authority/authority.json")).unwrap();
    let first_rows: Vec<Value> = read_json(&f.ledger("family-a")).unwrap();
    let completion = admission::artifact(f.root(), &first_rows[1]["clean_completion"]).unwrap();
    let native = admission::artifact(f.root(), &completion["fit_completion"]).unwrap();
    write_json(&expected,&json!({"schema":"parallel.label_admission.v2","evaluator":"evaluator",
        "evaluator_sha256":evaluator,"evaluator_protocol":admission::PROTOCOL,
        "authority_root":"test-authority","authority_witness":"test-witness","authority_instance":authority["instance"],
        "runtime_sha256":native["fit_identity"]["runtime_sha256"],
        "numerical_contract_sha256":native["fit_identity"]["numerical_contract_sha256"]})).unwrap();
    let cache = f.root().join("runs/strict/store");
    let output = f.root().join("strict");
    let source_rows: Vec<Value> = read_json(&f.ledger("family-a")).unwrap();
    let context: jope_parallel::corpus::ContextRow =
        serde_json::from_value(contexts["encodings"][0].clone()).unwrap();
    let prepared = read_json(&f.root().join("family-a/prepared.json")).unwrap();
    let policy: Value = read_json(&expected).unwrap();
    let mut counterfeit = source_rows[1].clone();
    let mut completion = admission::artifact(f.root(), &counterfeit["clean_completion"]).unwrap();
    let mut original = admission::artifact(f.root(), &completion["fit_completion"]).unwrap();
    original["task_id"] = json!("counterfeit-uncommitted-task");
    original
        .as_object_mut()
        .unwrap()
        .remove("clean_completion_sha256");
    original["clean_completion_sha256"] = json!(hash(&serde_json::to_vec(&original).unwrap()));
    let forged = f.root().join("forged-native.json");
    write_json(&forged, &original).unwrap();
    completion["fit_completion"] =
        json!({"path":"forged-native.json","sha256":file_hash(&forged).unwrap()});
    let forged_wrapper = f.root().join("forged-completion.json");
    write_json(&forged_wrapper, &completion).unwrap();
    counterfeit["clean_completion"] =
        json!({"path":"forged-completion.json","sha256":file_hash(&forged_wrapper).unwrap()});
    admission::validate_completion(f.root(), &counterfeit).unwrap();
    assert!(
        admission::validate(
            f.root(),
            &counterfeit,
            &context,
            &prepared,
            &file_hash(&f.vocab()).unwrap(),
            &policy
        )
        .is_err(),
        "self-consistent JSON must not replace an approved authority commit"
    );
    for field in [
        "authority_instance",
        "runtime_sha256",
        "numerical_contract_sha256",
    ] {
        let mut denied = policy.clone();
        denied[field] = json!(hash(b"not approved"));
        assert!(
            admission::validate(
                f.root(),
                &source_rows[1],
                &context,
                &prepared,
                &file_hash(&f.vocab()).unwrap(),
                &denied
            )
            .is_err(),
            "{field}"
        );
    }
    let source_receipt: Value =
        read_json(&f.ledger("family-a").with_file_name("search.json")).unwrap();
    for (field, value) in [
        ("completed_evaluations", json!(0)),
        ("completed_at", Value::Null),
        ("final_tests_opened", json!(true)),
        ("evaluator_sha256", json!("different evaluator")),
        ("evaluator_protocol", json!("different protocol")),
        ("vocabulary_sha256", json!("different vocabulary")),
    ] {
        let mut receipt = source_receipt.clone();
        receipt[field] = value;
        assert!(
            admission::validate_receipt(&receipt, &source_rows[1], source_rows.len(), &policy)
                .is_err(),
            "{field}"
        );
    }
    for invalid in [None, Some(json!("")), Some(json!("  ")), Some(json!(false))] {
        let mut row = source_rows[1].clone();
        match invalid {
            Some(value) => row["failure"] = value,
            None => {
                row.as_object_mut().unwrap().remove("failure");
            }
        }
        assert!(admission::validate(
            f.root(),
            &row,
            &context,
            &prepared,
            &file_hash(&f.vocab()).unwrap(),
            &policy
        )
        .is_err());
    }
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &f.observations(),
            output: &output,
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    let db = rusqlite::Connection::open(cache.join("store.sqlite")).unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM outcomes", [], |r| r
            .get::<_, usize>(0))
            .unwrap(),
        3
    );
    assert!(
        db.query_row("SELECT count(*) FROM quarantine", [], |r| r
            .get::<_, usize>(0))
            .unwrap()
            >= 6
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM programs WHERE program='(pipe X (standardize) (ridge))'",
            [],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        0
    );
    let manifest: store::Manifest = read_json(&output.join("index.json")).unwrap();
    assert_eq!(manifest.admission_policy, Some(policy));
    assert_eq!(manifest.counts["observations_seen"], 45);
    assert_eq!(manifest.counts["admission_failed_outcomes"], 3);
    for shard in &manifest.shards {
        for label in fs::read_to_string(f.root().join(&shard.path).join("labels.jsonl"))
            .unwrap()
            .lines()
        {
            let label: Value = serde_json::from_str(label).unwrap();
            let provenance = &label["admission"];
            let source = f.root().join(provenance["source"].as_str().unwrap());
            let original: Vec<Value> = read_json(&source).unwrap();
            let row = &original[provenance["row_index"].as_u64().unwrap() as usize];
            assert_eq!(provenance["ledger_sha256"], file_hash(&source).unwrap());
            assert_eq!(
                provenance["search_receipt_sha256"],
                file_hash(&source.with_file_name("search.json")).unwrap()
            );
            assert_eq!(
                provenance["observation_sha256"],
                hash(&serde_json::to_vec(row).unwrap())
            );
            assert_eq!(
                provenance["evaluation_identity_sha256"],
                row["evaluation_identity_sha256"]
            );
            assert_eq!(label["program"], row["genome"]);
        }
    }
    let blocked_before = db
        .query_row("SELECT count(*) FROM blocked WHERE context=0", [], |r| {
            r.get::<_, usize>(0)
        })
        .unwrap();
    // Conflicts in a later ledger remove an earlier accepted label, even when
    // the wrong row repeats an already-seen program.
    let rows: Vec<Value> = read_json(&f.ledger("family-a")).unwrap();
    let mut altered = Vec::new();
    for (n, field) in [
        (1, "evaluator_sha256"),
        (2, "metric_units"),
        (3, "validation_sha256"),
    ] {
        let mut row = rows[n].clone();
        row[field] = json!("conflict");
        altered.push(row);
    }
    let added = f.root().join("new/observations.json");
    write_json(&added, &altered).unwrap();
    f.strict_receipt(&added, &rows[0]);
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &[f.observations()[0].clone(), added.clone()],
            output: &f.root().join("strict-next"),
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM blocked WHERE context=0", [], |r| {
            r.get::<_, usize>(0)
        })
        .unwrap(),
        blocked_before + 3
    );
    for n in [1, 2, 3] {
        assert_eq!(
            db.query_row(
                "SELECT count(*) FROM programs WHERE context=0 AND program=?",
                [rows[n]["genome"].as_str().unwrap()],
                |r| r.get::<_, usize>(0)
            )
            .unwrap(),
            0
        );
    }
    // A failure in a later receipt removes the old quality label while its
    // observation remains available for feasibility/censored-runtime learning.
    let failure = f.root().join("later/observations.json");
    let mut failed = rows[4].clone();
    failed["failure"] = json!("timeout");
    failed["metrics"] = Value::Null;
    write_json(&failure, &vec![failed]).unwrap();
    f.strict_receipt(&failure, &rows[0]);
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &[f.observations()[0].clone(), added.clone(), failure.clone()],
            output: &f.root().join("strict-failure"),
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM outcomes", [], |r| r
            .get::<_, usize>(0))
            .unwrap(),
        4
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM programs WHERE context=0 AND program=?",
            [rows[4]["genome"].as_str().unwrap()],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        0
    );
    // An internally valid row from another search identity cannot enter labels.
    let mismatch = f.root().join("mismatch/observations.json");
    write_json(&mismatch, &vec![rows[5].clone()]).unwrap();
    f.strict_receipt(&mismatch, &rows[0]);
    let mut receipt: Value = read_json(&mismatch.with_file_name("search.json")).unwrap();
    receipt["cut"] = json!(999);
    write_json(&mismatch.with_file_name("search.json"), &receipt).unwrap();
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &[
                f.observations()[0].clone(),
                added.clone(),
                failure.clone(),
                mismatch.clone(),
            ],
            output: &f.root().join("strict-mismatched-receipt"),
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM programs WHERE context=0 AND program=?",
            [rows[5]["genome"].as_str().unwrap()],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        0
    );
    let new = f.root().join("failure-first/observations.json");
    let mut success = rows[6].clone();
    let program = "(pipe X (row_stats) (rank) (ridge))";
    success["genome"] = json!(program);
    success["program_sha256"] = json!(hash(program.as_bytes()));
    success["evaluation_identity_sha256"] = json!(hash(
        &serde_json::to_vec(&admission::identity(&success)).unwrap()
    ));
    let mut failed = success.clone();
    failed["failure"] = json!("timeout");
    failed["metrics"] = Value::Null;
    write_json(&new, &vec![failed, success]).unwrap();
    f.strict_receipt(&new, &rows[0]);
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &[f.observations()[0].clone(), added, failure, mismatch, new],
            output: &f.root().join("strict-failure-first"),
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM programs WHERE context=0 AND program=?",
            [program],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        0
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM blocked WHERE context=0 AND program=?",
            [program],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        1
    );
    let reuse = f.root().join("reuse/observations.json");
    let mut row = rows[7].clone();
    let program = "(pipe X (rank) (standardize) (ridge))";
    row["genome"] = json!(program);
    row["program_sha256"] = json!(hash(program.as_bytes()));
    row["evaluation_origin"] = json!("reuse");
    row["metrics"]["evaluation_origin"] = json!("reuse");
    row["evaluation_identity_sha256"] = json!(hash(
        &serde_json::to_vec(&admission::identity(&row)).unwrap()
    ));
    write_json(&reuse, &vec![row]).unwrap();
    f.strict_receipt(&reuse, &rows[0]);
    let sources: Vec<String> = db
        .prepare("SELECT path FROM sources ORDER BY ordinal")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<std::result::Result<_, _>>()
        .unwrap();
    let mut sources: Vec<PathBuf> = sources.into_iter().map(|p| f.root().join(p)).collect();
    sources.push(reuse);
    store::publish_admitted(
        f.root(),
        crate::teaching_store::PublishRequest {
            contexts_path: &f.contexts(),
            vocabulary: &f.vocab(),
            sources: &sources,
            output: &f.root().join("strict-reuse"),
            store: &cache,
            qualification: true,
        },
        &expected,
        |_, _, _| Ok(()),
    )
    .unwrap();
    assert_eq!(
        db.query_row("SELECT count(*) FROM reuses", [], |r| r.get::<_, usize>(0))
            .unwrap(),
        1
    );
    assert_eq!(
        db.query_row(
            "SELECT count(*) FROM programs WHERE program=?",
            [program],
            |r| r.get::<_, usize>(0)
        )
        .unwrap(),
        0
    );
    let manifest: store::Manifest = read_json(&f.root().join("strict-reuse/index.json")).unwrap();
    assert_eq!(manifest.counts["registry_reuse_not_observation"], 1);
}
