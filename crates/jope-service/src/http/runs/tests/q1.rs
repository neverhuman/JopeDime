use super::*;

const Q1_FAIL_ALL_TEST_SEED: u64 = 2;

struct Q1AllFailed;
impl crate::discover::r#loop::Executor for Q1AllFailed {
    fn evaluate(&self, request: &crate::worker::Request) -> crate::worker::Response {
        crate::worker::Response::failed(
            request.attempt.clone(),
            crate::worker::WorkerFailure::NonFinite,
            0,
        )
    }
    fn fitted_model(&self, _: &crate::worker::Request) -> anyhow::Result<Vec<u8>> {
        panic!("failed candidates cannot be refitted")
    }
}

fn q1_service_fixture() -> (tempfile::TempDir, RunService) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target/q1-native");
    std::fs::create_dir_all(&root).unwrap();
    let directory = tempfile::tempdir_in(root).unwrap();
    let lease = Arc::new(EngineLock::acquire(directory.path()).unwrap());
    let store = Arc::new(Store::open(directory.path().join("store")).unwrap());
    let config = Config::load(
        &directory.path().join("config.toml"),
        &Default::default(),
        &Default::default(),
    )
    .unwrap();
    let executable = std::env::current_exe().unwrap();
    let binary = format!(
        "{}:{}",
        env!("CARGO_PKG_VERSION"),
        hash(&std::fs::read(&executable).unwrap())
    );
    let service = RunService {
        journal: Arc::new(EventLog::new(Arc::clone(&store))),
        store,
        config,
        home: directory.path().to_path_buf(),
        executable,
        binary,
        activity: Arc::new(EngineActivity::default()),
        admission: Mutex::new(()),
        _lease: lease,
    };
    (directory, service)
}

fn q1_request(directory: &std::path::Path, store: &Store) -> Value {
    let train = b"x,target\n0,0\n1,1\n2,2\n3,3\n4,4\n5,5\n";
    let blob = store.publish_blob(train.as_slice()).unwrap();
    let protocol_sha = "a".repeat(64);
    let plan_sha = "b".repeat(64);
    let source_commit = "c".repeat(40);
    let pins = [
        "bundle_sha256",
        "dope_sha256",
        "evaluator_sha256",
        "vocabulary_sha256",
        "protocol_sha256",
        "training_family_manifest_sha256",
    ]
    .into_iter()
    .map(|key| {
        (
            key.to_owned(),
            if key == "protocol_sha256" {
                protocol_sha.clone()
            } else {
                "d".repeat(64)
            },
        )
    })
    .collect::<std::collections::BTreeMap<_, _>>();
    let mut binding = Q1DatasetBinding {
        id: "0123456789abcdef".into(),
        task: "regression".into(),
        dataset_sha256: "0123456789abcdef".to_owned() + &"e".repeat(48),
        family: "f".repeat(40),
        family_authoritative: true,
        rows: 10,
        features: 1,
        cut_sha256: String::new(),
        holdout_group: "sealed_meta_test".into(),
        catalog_version: "v246".into(),
        catalog_record_sha256: "2".repeat(64),
        dataset_binding_sha256: String::new(),
        target: "target".into(),
        feature_columns: vec!["x".into()],
        train_rows: 6,
        train_blob_sha256: hash(train),
        train_blob_bytes: blob.bytes,
        folds: (0..3)
            .map(|fold| Q1Fold {
                repeat: 0,
                train: (0..6).filter(|row| row / 2 != fold).collect(),
                validation: (fold * 2..(fold + 1) * 2).collect(),
            })
            .collect(),
    };
    binding.cut_sha256 = q1_cut_sha256(&binding).unwrap();
    let mut binding_value = serde_json::to_value(&binding).unwrap();
    binding_value
        .as_object_mut()
        .unwrap()
        .remove("dataset_binding_sha256");
    binding.dataset_binding_sha256 = hash(&serde_json::to_vec(&binding_value).unwrap());
    let manifest = json!({
        "schema":"jope.q1-protocol-manifest/v1",
        "protocol_sha256":protocol_sha,
        "source_commit":source_commit,
        "plan_sha256":plan_sha,
        "pins":pins,
        "datasets":{"0123456789abcdef":binding}
    });
    let protocols = directory.join("q1/protocols");
    std::fs::create_dir_all(&protocols).unwrap();
    std::fs::write(
        protocols.join(format!("{}.json", "a".repeat(64))),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let mut request = json!({
        "schema":"jope.q1-request/v1",
        "mode":"qualification",
        "source_commit":source_commit,
        "plan_sha256":plan_sha,
        "pins":pins,
        "metric":"rmse",
        "dataset":{
            "id":"0123456789abcdef",
            "task":"regression",
            "dataset_sha256":"0123456789abcdefe".to_owned() + &"e".repeat(47),
            "family":"f".repeat(40),
            "family_authoritative":true,
            "rows":10,
            "features":1,
            "cut_sha256":binding.cut_sha256,
            "holdout_group":"sealed_meta_test"
        },
        "guidance":"heuristic",
        "seed":0,
        "budget":24
    });
    // Match the catalog's full dataset identity; the test partition is never stored here.
    request["dataset"]["dataset_sha256"] = json!("0123456789abcdef".to_owned() + &"e".repeat(48));
    let sha = canonical_sha256(&request).unwrap();
    request["request_sha256"] = json!(sha);
    request
}

fn q1_http_request(body: &Value) -> Request<Body> {
    Request::builder()
        .method("POST")
        .uri("/v1/q1/evaluations")
        .header(header::HOST, "127.0.0.1:31234")
        .header(header::AUTHORIZATION, format!("Bearer {}", "ab".repeat(32)))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(body).unwrap()))
        .unwrap()
}

fn request_with_guidance(mut request: Q1Request, guidance: &str) -> Q1Request {
    request.guidance = guidance.into();
    request
}

include!("q1/validation.rs");
include!("q1/attempts.rs");
include!("q1/execution.rs");
