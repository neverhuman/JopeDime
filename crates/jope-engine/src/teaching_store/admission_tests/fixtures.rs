use super::*;

pub(super) struct Fixture {
    directory: tempfile::TempDir,
}
impl Fixture {
    pub(in crate::teaching_store) fn new() -> Self {
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
    pub(in crate::teaching_store) fn root(&self) -> &Path {
        self.directory.path()
    }
    pub(in crate::teaching_store) fn vocab(&self) -> PathBuf {
        self.root().join("vocabulary.json")
    }
    pub(in crate::teaching_store) fn contexts(&self) -> PathBuf {
        self.root().join("contexts.json")
    }
    pub(in crate::teaching_store) fn observations(&self) -> Vec<PathBuf> {
        vec![self.root().join("observations")]
    }
    pub(in crate::teaching_store) fn ledger(&self, family: &str) -> PathBuf {
        self.root()
            .join("observations")
            .join(family)
            .join("observations.json")
    }
    pub(in crate::teaching_store) fn strict_receipt(&self, path: &Path, identity: &Value) {
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

pub(in crate::teaching_store) fn attach_completion(
    root: &Path,
    directory: &Path,
    index: usize,
    row: &mut Value,
) {
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
        crate::eval_registry::FitInputs {
            x: vec![vec![1.]],
            y: vec![2.],
            preprocessing,
            feature_view,
        },
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
