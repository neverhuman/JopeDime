use jope_parallel::{
    corpus::{self, ScanCacheStats},
    manifest::*,
};
use serde_json::{json, Value};
use std::{
    fs,
    path::{Path, PathBuf},
};

struct Fixture {
    directory: tempfile::TempDir,
}
impl Fixture {
    fn new() -> Self {
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
    fn root(&self) -> &Path {
        self.directory.path()
    }
    fn vocab(&self) -> PathBuf {
        self.root().join("vocabulary.json")
    }
    fn contexts(&self) -> PathBuf {
        self.root().join("contexts.json")
    }
    fn observations(&self) -> Vec<PathBuf> {
        vec![self.root().join("observations")]
    }
    fn ledger(&self, family: &str) -> PathBuf {
        self.root()
            .join("observations")
            .join(family)
            .join("observations.json")
    }
    fn cache(&self) -> PathBuf {
        self.root().join("runs/experiment/scan-cache")
    }
    fn full(&self, name: &str, qualification: bool) {
        corpus::build(
            self.root(),
            &self.contexts(),
            &self.vocab(),
            &self.observations(),
            &self.root().join(name),
            qualification,
        )
        .unwrap();
    }
    fn cached(&self, name: &str, qualification: bool) -> anyhow::Result<ScanCacheStats> {
        corpus::build_observed_cached(
            self.root(),
            jope_parallel::corpus::CorpusSources {
                contexts_path: &self.contexts(),
                vocab_path: &self.vocab(),
                observations: &self.observations(),
            },
            &self.root().join(name),
            qualification,
            &self.cache(),
            |_, _, _| Ok(()),
        )
    }
    fn equal(&self, a: &str, b: &str) {
        for name in [
            "records.bin",
            "labels.jsonl",
            "index.json",
            "provenance.json",
            "vocabulary.json",
        ] {
            assert_eq!(
                fs::read(self.root().join(a).join(name)).unwrap(),
                fs::read(self.root().join(b).join(name)).unwrap(),
                "changed {name}"
            );
        }
    }
    fn entries(&self) -> Vec<PathBuf> {
        let mut paths = fs::read_dir(self.cache())
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| !p.file_name().unwrap().to_string_lossy().starts_with('.'))
            .collect::<Vec<_>>();
        paths.sort();
        paths
    }
}

// Synthetic original artifacts exercise admission without any numerical fitting.
fn attach_completion(root: &Path, directory: &Path, index: usize, row: &mut Value) {
    fn put(root: &Path, path: &Path, value: &Value) -> Value {
        write_json(path, value).unwrap();
        json!({"path":path.strip_prefix(root).unwrap(),"sha256":file_hash(path).unwrap()})
    }
    let success = row["failure"].is_null();
    let timeout = row["failure"].as_str() == Some("timeout");
    let outcome = if success {
        "success"
    } else if timeout {
        "timeout"
    } else {
        "fit_failure"
    };
    if timeout {
        row["runtime_seconds"] = json!(30.1);
    }
    let model = put(
        root,
        &directory.join(format!("model-{index}.json")),
        &json!({"synthetic_model":index}),
    );
    let program = row["genome"].as_str().unwrap_or("(pipe X (ridge))");
    let training =
        jope_parallel::eval_registry::ordered_training_sha256(&[vec![1.]], &[2.]).unwrap();
    let fit = jope_parallel::eval_registry::FitIdentity {
        schema: "parallel.fit_identity.v2".into(),
        canonical_program: program.into(),
        program_sha256: row["program_sha256"].as_str().unwrap().into(),
        training_sha256: training.clone(),
        preprocessing_sha256: row["preprocessing_sha256"]
            .as_str()
            .unwrap_or(&hash(b"preprocessing"))
            .into(),
        feature_view_sha256: row["feature_view_sha256"]
            .as_str()
            .unwrap_or(&hash(b"feature-view"))
            .into(),
        executable_sha256: row["evaluator_sha256"].as_str().unwrap().into(),
        runtime_sha256: hash(b"runtime"),
        fit_seed: 0,
        numerical_contract_sha256: hash(b"contract"),
    };
    let attempt = json!({"evaluation_identity_sha256":row["evaluation_identity_sha256"],
        "program_sha256":row["program_sha256"],"evaluator_sha256":row["evaluator_sha256"]});
    let attempt = put(
        root,
        &directory.join(format!("attempt-{index}.json")),
        &attempt,
    );
    let output = if success {
        put(
            root,
            &directory.join(format!("result-{index}.json")),
            &row["metrics"],
        )
    } else {
        Value::Null
    };
    let mut native = json!({"schema":"parallel.fit_completion.v2", "program_sha256":row["program_sha256"],
        "fit_identity":fit,"authority_instance":hash(b"synthetic-authority"),
        "fit_key":hash(&serde_json::to_vec(&fit).unwrap()),"attempt_token":hash(format!("token-{index}").as_bytes()),
        "process":{"boot_id":"synthetic-boot","pid":1,"birth_ticks":1},"task_id":"synthetic-admission-test",
        "executable_sha256":row["evaluator_sha256"],"training_sha256":training,
        "outcome":if outcome == "fit_failure" {"deterministic_failure"} else {outcome},
        "observed_exit_code":if success {0} else {1},"observed_signal":null,
        "elapsed_seconds":row["runtime_seconds"],"deadline_seconds":30.,"output_sha256":hash(b"original-output"),
        "model_sha256":if success {model["sha256"].clone()} else {Value::Null},
        "predictions_sha256":null,"metrics_sha256":null});
    row["numerical_runtime_seconds"] = native["elapsed_seconds"].clone();
    native["clean_completion_sha256"] = json!(hash(&serde_json::to_vec(&native).unwrap()));
    let native = put(
        root,
        &directory.join(format!("native-{index}.json")),
        &native,
    );
    let completion = json!({"schema":"parallel.clean_completion.v3","attempt":attempt,"fit_completion":native,
        "evaluation_identity_sha256":row["evaluation_identity_sha256"],"program_sha256":row["program_sha256"],
        "evaluator_sha256":row["evaluator_sha256"],"outcome":outcome,"exit_observed":true,
        "exit_code":if success {0} else {1},"signal":null,"deadline_seconds":30.,"deadline_exceeded":timeout,
        "runtime_seconds":row["runtime_seconds"],"output":output,"model":model});
    row["clean_completion"] = put(
        root,
        &directory.join(format!("completion-{index}.json")),
        &completion,
    );
}

#[path = "corpus_incremental/cache.rs"]
mod cache;
#[path = "corpus_incremental/publication.rs"]
mod publication;
#[path = "corpus_incremental/reconcile.rs"]
mod reconcile;
