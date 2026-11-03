use super::*;
use std::sync::Barrier;

mod executable_cache;
mod query_tests;
mod service_tests;

fn inputs(alpha: u32) -> ServiceInputs {
    ServiceInputs {
        program: format!("(pipe X (ridge alpha={alpha}))"),
        x: vec![vec![1.], vec![2.]],
        y: vec![2., 3.],
        preprocessing: b"preprocess".to_vec(),
        feature_view: b"raw".to_vec(),
    }
}
fn provenance() -> RequestProvenance {
    RequestProvenance {
        scope: "synthetic".into(),
        task_id: "synthetic-service-task".into(),
        details: json!({"synthetic_only":true,"deadline_monotonic_ns":request_monotonic_ns().unwrap()+120_000_000_000u64}),
    }
}
struct Fixture {
    temp: tempfile::TempDir,
    _socket_custody: Option<tempfile::TempDir>,
    root: PathBuf,
    witness: PathBuf,
    socket: PathBuf,
    config: ServiceConfig,
}
impl Fixture {
    fn new() -> Self {
        Self::new_at(None)
    }
    fn new_at(evidence: Option<&Path>) -> Self {
        let mut temp = if let Some(evidence) = evidence {
            fs::create_dir_all(evidence).unwrap();
            tempfile::tempdir_in(evidence).unwrap()
        } else {
            tempfile::tempdir().unwrap()
        };
        if evidence.is_some() {
            temp.disable_cleanup(true);
            println!(
                "retained original synthetic query fixture: {}",
                temp.path().display()
            );
        }
        let root = temp.path().join("authority");
        let witness = temp.path().join("witness");
        // Retained evidence paths can exceed sockaddr_un's pathname limit.
        // Socket custody remains machine-local and is cleaned independently.
        let socket_custody = evidence.map(|_| tempfile::tempdir().unwrap());
        let socket = socket_custody
            .as_ref()
            .map_or(temp.path(), |dir| dir.path())
            .join("service.sock");
        let i = inputs(0);
        // Explicit test-only coverage and an unrelated original synthetic model;
        // no numerical fit is used to prepare an eligible service fixture.
        super::super::synthetic_completion_for_test(
            &root,
            &witness,
            "(sym raw \"col(0)\")",
            crate::eval_registry::FitInputs {
                x: i.x,
                y: i.y,
                preprocessing: i.preprocessing,
                feature_view: i.feature_view,
            },
            0,
        )
        .unwrap();
        let mut config = ServiceConfig::regression(
            root.clone(),
            witness.clone(),
            socket.clone(),
            std::env::current_exe().unwrap(),
            104,
        );
        config.worker_arguments = vec![
            "--exact".into(),
            "eval_registry::service::tests::synthetic_worker".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ];
        config.query_worker_arguments = vec![
            "--exact".into(),
            "eval_registry::service::tests::synthetic_query_worker".into(),
            "--ignored".into(),
            "--nocapture".into(),
        ];
        config.handshake_timeout = Duration::from_secs(60);
        Self {
            temp,
            _socket_custody: socket_custody,
            root,
            witness,
            socket,
            config,
        }
    }
    fn launch(
        &self,
    ) -> (
        Arc<AtomicBool>,
        std::thread::JoinHandle<Result<ServiceStats>>,
    ) {
        let service = Service::bind(self.config.clone()).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let signal = stop.clone();
        (stop, std::thread::spawn(move || service.run(signal)))
    }
}

#[test]
fn invalid_cpu_evaluator_binding_rejects_before_opening_authority() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("unopened-authority");
    let witness = temp.path().join("unopened-witness");
    let socket = temp.path().join("unopened.sock");
    let other = temp.path().join("other-evaluator");
    fs::write(&other, b"#!/bin/sh\nexit 99\n").unwrap();
    fs::set_permissions(&other, fs::Permissions::from_mode(0o700)).unwrap();
    for (path, expected) in [
        (
            PathBuf::from("relative-evaluator"),
            "CPU evaluator binding must be an absolute path",
        ),
        (
            temp.path().to_path_buf(),
            "CPU evaluator binding must name an executable regular file",
        ),
        (
            other,
            "service and explicitly bound CPU evaluator executable differ",
        ),
    ] {
        let result = Service::bind(ServiceConfig::regression(
            root.clone(),
            witness.clone(),
            socket.clone(),
            path,
            1,
        ));
        let error = result.err().expect("invalid evaluator binding accepted");
        let expected = if cfg!(feature = "neural") {
            "CPU fit authority requires a non-neural evaluator build"
        } else {
            expected
        };
        assert!(error.to_string().contains(expected), "{error:#}");
        assert!(!root.exists());
        assert!(!witness.exists());
        assert!(!socket.exists());
    }
}

#[test]
#[ignore = "internal synthetic service-owned executor; never runs numerical fits"]
fn synthetic_worker() {
    let (fit, permit, output) = await_start_from_environment().unwrap();
    permit.verify().unwrap();
    let program = &fit.identity().canonical_program;
    if program.contains("alpha=2") {
        panic!("injected synthetic worker failure after start authorization");
    }
    if program.contains("alpha=3") {
        unsafe {
            libc::raise(libc::SIGTERM);
        }
    }
    if program.contains("alpha=4") {
        std::thread::sleep(Duration::from_secs(30));
    }
    if program.contains("alpha=5") {
        let old = permit.custody_path().with_extension("old");
        fs::rename(&permit.custody_path(), &old).unwrap();
        fs::write(&permit.custody_path(), b"replacement").unwrap();
        assert!(permit.verify().is_err());
        panic!("injected synthetic worker failure after custody tampering");
    }
    if program.contains("alpha=6") {
        // The evaluated program would run here, after lockdown. Its artifact
        // is <root>/worker-output/<ticket>/output.json.
        let outputs = output.parent().unwrap().parent().unwrap();
        let root = outputs.parent().unwrap();
        let parent = unsafe { libc::getppid() };
        let unreadable = [
            PathBuf::from("/etc/passwd"),
            PathBuf::from(format!("/proc/{parent}/environ")),
            PathBuf::from(format!("/proc/{parent}/cmdline")),
            root.join("authority.json"),
            root.join("registry.sqlite"),
        ];
        let found = super::sandbox::tests::violations(&unreadable, outputs, None);
        if !found.is_empty() {
            panic!("unconfined original fit child: {found:?}");
        }
    }
    let model = feat_math::Model::Symbolic {
        expr: feat_math::math_expr::MathExpr::parse("col(0)").unwrap(),
        a: 1.,
        b: 0.,
    };
    write_new(
        &output,
        &serde_json::to_vec(&json!({"status":"success","model":model,"synthetic_only":true}))
            .unwrap(),
    )
    .unwrap();
    permit.verify().unwrap();
}

/// The original service observes a worker that serializes a directly
/// constructed symbolic model. This fixture never calls numerical fitting.
fn retained_query_fixture() -> (Fixture, query::QueryInputs) {
    let evidence = std::env::var_os("JOPE_QUERY_TEST_EVIDENCE").map(PathBuf::from);
    let mut fixture = Fixture::new_at(evidence.as_deref());
    let mut fit_inputs = inputs(0);
    fit_inputs.preprocessing = serde_json::to_vec(&crate::prepare::Preprocessing {
        schema: "parallel.preprocessing.v1".into(),
        columns: vec![crate::prepare::Column {
            kind: "numeric".into(),
            impute: 0.,
            mean: 0.,
            scale: 1.,
            categories: vec![],
        }],
        labels: vec![],
        target_mean: 10.,
        target_scale: 2.,
    })
    .unwrap();
    fit_inputs.feature_view = serde_json::to_vec(&json!({"protocol":"raw_training_fold_v1",
        "training_sha256":sha(b"synthetic training csv"),
        "preprocessing_sha256":sha(&fit_inputs.preprocessing)}))
    .unwrap();
    let (stop, task) = fixture.launch();
    let original = submit_to(
        &fixture.socket,
        &fixture.root,
        &fixture.witness,
        &fit_inputs,
        provenance(),
    )
    .unwrap();
    assert_eq!(original.receipt.outcome, ExecutionOutcome::Success);
    stop.store(true, Ordering::SeqCst);
    assert_eq!(task.join().unwrap().unwrap().worker_spawns, 1);
    let query = query::QueryInputs {
        schema: "parallel.query_inputs.v1".into(),
        completion_sha256: original.completion_sha256,
        scope: "synthetic".into(),
        task_id: "synthetic-query-service-task".into(),
        rows: vec![vec!["3".into()], vec!["4".into()]],
        targets: vec![16., 18.],
        preprocessing: fit_inputs.preprocessing,
        feature_view: fit_inputs.feature_view,
        contract: query::QueryContract::regression_rmse(),
        access_receipt_sha256: sha(b"exact private query capability"),
        deadline_monotonic_ns: request_monotonic_ns().unwrap() + 120_000_000_000,
    };
    fixture
        .config
        .synthetic_query_access
        .insert(query.descriptor_sha256().unwrap());
    (fixture, query)
}

fn query_events(root: &Path) -> Vec<Value> {
    fs::read_to_string(root.join("dispatch.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .filter(|event| event["kind"].as_str().unwrap().starts_with("query_"))
        .collect()
}

#[test]
#[ignore = "internal retained-model prediction child; no fit permission"]
fn synthetic_query_worker() {
    execute_query_worker().unwrap();
}
