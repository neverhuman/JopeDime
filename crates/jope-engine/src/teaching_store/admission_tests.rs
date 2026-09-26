use crate as jope_parallel;
use crate::manifest::*;
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(super) struct Fixture {
    directory: tempfile::TempDir,
}
impl Fixture {
    pub(super) fn new() -> Self {
        let this = Self {
            directory: tempfile::tempdir().unwrap(),
        };
        let root = this.root();
        fs::write(
            this.vocab(),
            fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/grammar/featmath.json"))
                .unwrap(),
        )
        .unwrap();
        let mut contexts = Vec::new();
        for family in ["family-a", "family-b", "family-c"] {
            let prepared = root.join(family);
            write_json(&prepared.join("prepared.json"), &json!({
                "schema":"parallel.prepared.v1","dataset":family,"family":family,"task":"regression",
                "meta_partition":"meta_train","seed":17,"clean_evaluation":true,"clean_status":"fixture",
                "original_train_sha256":"original","original_test_sha256":"sealed",
                "selected_train_indices":[0,1],"selected_test_indices":[],"target_last":true,"rows_total":2,
                "cuts":[{"index":0,"train_indices":[0],"validation_indices":[1],"training_sha256":"train",
                    "validation_sha256":"validation","preprocessing_sha256":"preprocessing"}]})).unwrap();
            let prepared_sha = file_hash(&prepared.join("prepared.json")).unwrap();
            let path = prepared.join("context.bin");
            fs::write(&path, b"frozen context fixture").unwrap();
            write_json(&path.with_extension("json"), &json!({"schema":"parallel.encoding.v1","dataset":family,
                "family":family,"partition":"meta_train","cut":0,"prepared_sha256":prepared_sha,
                "training_sha256":"train","selected_training_rows_sha256":"rows","preprocessing_sha256":"preprocessing",
                "encoder_spec_sha256":"assembly","numerical_backend":"fixture","dataset_dimensions":1536,
                "target_dimensions":512,"output_sha256":file_hash(&path).unwrap()})).unwrap();
            contexts.push(
                json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                "path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap(),
                "prepared":family,"prepared_sha256":prepared_sha}),
            );
            let mut rows = Vec::new();
            for (n, transform) in [
                "standardize",
                "robust_scale",
                "norm_row",
                "row_stats",
                "rank",
                "blom",
                "sigmoid",
                "step",
            ]
            .iter()
            .enumerate()
            {
                let genome = format!("(pipe X ({transform}) (ridge))");
                rows.push(
                    json!({"dataset":family,"family":family,"cut":0,"partition":"meta_train",
                    "training_sha256":"train","validation_sha256":"validation","genome":genome,
                    "program_sha256":hash(genome.as_bytes()),"metrics":{"rmse":n as f64+1.},
                    "runtime_seconds":0.1,"evaluation_bias":"first","failure":null,
                    "family_partition_frozen":true,"qualification":false}),
                );
            }
            let mut duplicate = rows[0].clone();
            duplicate["metrics"]["rmse"] = json!(99.0);
            duplicate["evaluation_bias"] = json!("duplicate must not win");
            rows.push(duplicate);
            let mut failed = rows[0].clone();
            failed["failure"] = json!("fixture failure");
            rows.push(failed);
            let mut missing = rows[0].clone();
            missing["dataset"] = json!("out-of-index");
            rows.push(missing);
            let mut rejected = rows[0].clone();
            let genome = "(pipe X (rff_cos n=17 gamma=0.513 seed=7) (ridge))";
            rejected["genome"] = json!(genome);
            rejected["program_sha256"] = json!(hash(genome.as_bytes()));
            rows.push(rejected.clone());
            rows.push(rejected);
            write_json(&this.ledger(family), &rows).unwrap();
            write_json(
                &this.ledger(family).with_file_name("search.json"),
                &json!({"schema":"parallel.search.v1"}),
            )
            .unwrap();
        }
        write_json(
            &this.contexts(),
            &json!({"schema":"parallel.context_index.v1","qualification":false,
            "assembly_sha256":"assembly","encodings":contexts}),
        )
        .unwrap();
        this
    }
    pub(super) fn root(&self) -> &Path {
        self.directory.path()
    }
    pub(super) fn vocab(&self) -> PathBuf {
        self.root().join("vocabulary.json")
    }
    pub(super) fn contexts(&self) -> PathBuf {
        self.root().join("contexts.json")
    }
    pub(super) fn observations(&self) -> Vec<PathBuf> {
        vec![self.root().join("observations")]
    }
    pub(super) fn ledger(&self, family: &str) -> PathBuf {
        self.root()
            .join("observations")
            .join(family)
            .join("observations.json")
    }
    pub(super) fn strict_receipt(&self, path: &Path, identity: &Value) {
        let mut rows: Vec<Value> = read_json(path).unwrap();
        for (n, row) in rows.iter_mut().enumerate() {
            if !row["failure"].is_null() {
                row["metrics"] = Value::Null;
            }
            attach_completion(self.root(), path.parent().unwrap(), n, row);
        }
        write_json(path, &rows).unwrap();
        write_json(&path.with_file_name("search.json"),&json!({"schema":"parallel.search.v1",
            "observations_sha256":file_hash(path).unwrap(),"completed_evaluations":rows.len(),"completed_at":1,
            "final_tests_opened":false,"evaluator_sha256":identity["evaluator_sha256"],
            "evaluator_protocol":identity["evaluator_protocol"],"prepared_sha256":identity["prepared_sha256"],
            "cut":identity["cut"],"vocabulary_sha256":identity["vocabulary_sha256"]})).unwrap();
    }
}

pub(super) fn attach_completion(root: &Path, directory: &Path, index: usize, row: &mut Value) {
    let prepared = root
        .join(row["family"].as_str().unwrap())
        .join("cut-0/preprocessing.json");
    let preprocessing = fs::read(prepared).unwrap();
    let feature_view=serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":row["training_sha256"],"preprocessing_sha256":row["preprocessing_sha256"]})).unwrap();
    let authority_root = root.join("test-authority");
    let witness = root.join("test-witness");
    let (sha, _) = crate::eval_registry::synthetic_completion_for_test(
        &authority_root,
        &witness,
        row["genome"].as_str().unwrap(),
        vec![vec![1.]],
        vec![2.],
        preprocessing,
        feature_view,
        0,
    )
    .unwrap();
    let native: Value = read_json(&authority_root.join("objects").join(&sha)).unwrap();
    let native_ref = json!({"path":format!("test-authority/objects/{sha}"),"sha256":sha});
    let model_sha = native["model_sha256"].as_str().unwrap();
    let model_ref =
        json!({"path":format!("test-authority/objects/{model_sha}"),"sha256":model_sha});
    row["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
    let success = row["failure"].is_null();
    let timeout = row["failure"].as_str() == Some("timeout");
    let attempt_path = directory.join(format!("attempt-{index}.json"));
    write_json(
        &attempt_path,
        &json!({"evaluation_identity_sha256":row["evaluation_identity_sha256"],
        "program_sha256":row["program_sha256"],"evaluator_sha256":row["evaluator_sha256"]}),
    )
    .unwrap();
    let result_path = directory.join(format!("result-{index}.json"));
    if success {
        write_json(&result_path, &row["metrics"]).unwrap();
    }
    let completion = json!({"schema":"parallel.clean_completion.v3","fit_completion":native_ref,"model":model_ref,
        "attempt":{"path":attempt_path.strip_prefix(root).unwrap(),"sha256":file_hash(&attempt_path).unwrap()},
        "evaluation_identity_sha256":row["evaluation_identity_sha256"],"program_sha256":row["program_sha256"],
        "evaluator_sha256":row["evaluator_sha256"],"outcome":if success {"success"} else if timeout {"timeout"} else {"infrastructure_failure"},
        "exit_observed":true,"exit_code":if success {0} else {1},"signal":null,"deadline_seconds":30.,
        "deadline_exceeded":timeout,"runtime_seconds":row["runtime_seconds"],
        "output":if success {json!({"path":result_path.strip_prefix(root).unwrap(),"sha256":file_hash(&result_path).unwrap()})} else {Value::Null}});
    let path = directory.join(format!("completion-{index}.json"));
    write_json(&path, &completion).unwrap();
    row["clean_completion"] =
        json!({"path":path.strip_prefix(root).unwrap(),"sha256":file_hash(&path).unwrap()});
}

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
        &f.contexts(),
        &f.vocab(),
        &f.observations(),
        &output,
        &cache,
        true,
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
        &f.contexts(),
        &f.vocab(),
        &[f.observations()[0].clone(), added.clone()],
        &f.root().join("strict-next"),
        &cache,
        true,
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
        &f.contexts(),
        &f.vocab(),
        &[f.observations()[0].clone(), added.clone(), failure.clone()],
        &f.root().join("strict-failure"),
        &cache,
        true,
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
        &f.contexts(),
        &f.vocab(),
        &[
            f.observations()[0].clone(),
            added.clone(),
            failure.clone(),
            mismatch.clone(),
        ],
        &f.root().join("strict-mismatched-receipt"),
        &cache,
        true,
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
        &f.contexts(),
        &f.vocab(),
        &[f.observations()[0].clone(), added, failure, mismatch, new],
        &f.root().join("strict-failure-first"),
        &cache,
        true,
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
        &f.contexts(),
        &f.vocab(),
        &sources,
        &f.root().join("strict-reuse"),
        &cache,
        true,
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
